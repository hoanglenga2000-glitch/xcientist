"""A single pinned user transport, with no process-global provider changes/fallback."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time

from research_os.agent.messaging import AnthropicTransport, DeepSeekTransport, OpenAITransport
from research_os.llm_client import ProviderConfig
from .chat_stream_transport import consume_chat_stream
from .model_transport import _response_error
from .models import utc_now


def client_for_run(runtime, session, before_attempt=None, text_observer=None, during_request=None):
    from .model_profiles import ModelProfiles
    from .tenant_access import AccessStore
    profile_store = ModelProfiles(runtime.runtime_root)
    owner = AccessStore(runtime.runtime_root).owner_for('session', session['id'])
    frozen = profile_store.run_binding(owner, session['id'])
    declared = (session.get('metadata') or {}).get('model_profile') or {}
    if any(frozen[key] != declared.get(key) for key in ('id', 'version')):
        raise ValueError('model_run_binding_mismatch')
    return PersonalModelClient(profile_store, owner, frozen, before_attempt=before_attempt, record_use=True,
                               text_observer=text_observer, during_request=during_request)


def redact(value, key):
    if isinstance(value, str):
        return value.replace(key, '[REDACTED]')
    if isinstance(value, list):
        return [redact(item, key) for item in value]
    if isinstance(value, dict):
        return {redact(name, key): redact(item, key) for name, item in value.items()}
    return value


def post_public_json(url, headers, payload):
    """External-service boundary. Fixtures may replace this, never the task/store."""
    if os.environ.get('EVOMIND_TEST_FIXTURE_ROOT'):
        raise ValueError('model_isolated_network_disabled')
    request = json.dumps({'url': url, 'headers': headers, 'payload': payload}).encode()
    if len(request) > 3 * 1024 * 1024:
        raise ValueError('model_request_too_large')
    try:
        result = subprocess.run([sys.executable, '-I', str(Path(__file__).with_name('personal_model_http.py'))],
            input=request, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    except subprocess.TimeoutExpired:
        raise ValueError('model_total_timeout') from None
    if result.returncode or len(result.stdout) > 2 * 1024 * 1024:
        raise ValueError('model_transport_failed')
    result = json.loads(result.stdout)
    if 'error' in result:
        raise ValueError(result['error'])
    return result['body']


_POST_PUBLIC_JSON = post_public_json
_STREAM_PROVIDERS = frozenset({'anthropic', 'openai', 'deepseek'})
_MODEL_ERROR = re.compile(r'model_[a-z0-9_]{1,80}')


def _usage_int(value):
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise ValueError('stream_chunk_invalid')
    if isinstance(value, float) and not value.is_integer():
        raise ValueError('stream_chunk_invalid')
    number = int(value)
    if number < 0:
        raise ValueError('stream_chunk_invalid')
    return number


def consume_anthropic_stream(records, expected_model, on_text):
    """Rebuild a /v1/messages body. Thinking and tool-argument fragments stay off the UI."""
    model_ok, blocks, order = False, {}, []
    stop_reason, stopped = None, False
    usage = {'input_tokens': 0, 'output_tokens': 0}
    for record in records:
        if not isinstance(record, str) or not record:
            raise ValueError('stream_chunk_invalid')
        try:
            event = json.loads(record)
        except json.JSONDecodeError:
            raise ValueError('stream_chunk_invalid') from None
        if not isinstance(event, dict):
            raise ValueError('stream_chunk_invalid')
        kind = event.get('type')
        if kind == 'ping':
            continue
        if kind == 'error':
            raise ValueError('stream_provider_error')
        if stopped:
            raise ValueError('stream_data_after_finish')
        if kind == 'message_start':
            message = event.get('message')
            if not isinstance(message, dict) or message.get('model') != expected_model:
                raise ValueError('stream_model_identity_invalid')
            model_ok = True
            raw_usage = message.get('usage') if isinstance(message.get('usage'), dict) else {}
            if 'input_tokens' in raw_usage:
                usage['input_tokens'] = _usage_int(raw_usage.get('input_tokens') or 0)
            if 'output_tokens' in raw_usage:
                usage['output_tokens'] = _usage_int(raw_usage.get('output_tokens') or 0)
            continue
        if not model_ok:
            raise ValueError('stream_model_identity_invalid')
        if kind == 'content_block_start':
            index, block = event.get('index'), event.get('content_block')
            if type(index) is not int or not 0 <= index < 128 or index in blocks or not isinstance(block, dict):
                raise ValueError('stream_chunk_invalid')
            btype = block.get('type')
            if btype == 'text':
                text = block.get('text') or ''
                if not isinstance(text, str):
                    raise ValueError('stream_content_invalid')
                blocks[index] = {'type': 'text', 'text': text}
                if text:
                    on_text(text)
            elif btype == 'tool_use':
                ident, name = block.get('id') or '', block.get('name') or ''
                if not isinstance(ident, str) or not isinstance(name, str):
                    raise ValueError('stream_tool_fragment_invalid')
                blocks[index] = {'type': 'tool_use', 'id': ident, 'name': name, 'input': ''}
            elif btype in {'thinking', 'redacted_thinking'}:
                blocks[index] = {'type': btype, 'hidden': True}
            else:
                raise ValueError('stream_chunk_invalid')
            order.append(index)
        elif kind == 'content_block_delta':
            index, delta = event.get('index'), event.get('delta')
            if type(index) is not int or index not in blocks or not isinstance(delta, dict):
                raise ValueError('stream_chunk_invalid')
            block, dtype = blocks[index], delta.get('type')
            if block['type'] == 'text' and dtype == 'text_delta':
                text = delta.get('text')
                if not isinstance(text, str):
                    raise ValueError('stream_content_invalid')
                block['text'] += text
                if text:
                    on_text(text)
            elif block['type'] == 'tool_use' and dtype == 'input_json_delta':
                fragment = delta.get('partial_json')
                if not isinstance(fragment, str):
                    raise ValueError('stream_tool_fragment_invalid')
                block['input'] += fragment
            elif block.get('hidden') and dtype in {'thinking_delta', 'signature_delta'}:
                continue
            else:
                raise ValueError('stream_chunk_invalid')
        elif kind == 'content_block_stop':
            index = event.get('index')
            if type(index) is not int or index not in blocks:
                raise ValueError('stream_chunk_invalid')
        elif kind == 'message_delta':
            delta = event.get('delta') if isinstance(event.get('delta'), dict) else {}
            if 'stop_reason' in delta and delta.get('stop_reason') is not None:
                if not isinstance(delta.get('stop_reason'), str):
                    raise ValueError('stream_chunk_invalid')
                stop_reason = delta['stop_reason']
            raw_usage = event.get('usage') if isinstance(event.get('usage'), dict) else {}
            if 'output_tokens' in raw_usage:
                usage['output_tokens'] = _usage_int(raw_usage.get('output_tokens') or 0)
        elif kind == 'message_stop':
            if not isinstance(stop_reason, str):
                raise ValueError('stream_terminal_missing')
            stopped = True
        else:
            raise ValueError('stream_chunk_invalid')
    if not stopped:
        raise ValueError('stream_terminal_missing')
    content = []
    for index in order:
        block = blocks[index]
        if block['type'] == 'text':
            content.append({'type': 'text', 'text': block['text']})
        elif block['type'] == 'tool_use':
            raw = block['input']
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                raise ValueError('stream_tool_fragment_invalid') from None
            if not isinstance(parsed, dict):
                raise ValueError('stream_tool_fragment_invalid')
            content.append({'type': 'tool_use', 'id': block['id'], 'name': block['name'], 'input': parsed})
    return {'model': expected_model, 'content': content, 'stop_reason': stop_reason, 'usage': usage}


def decode_personal_stream(provider, records, expected_model, on_text):
    if provider == 'anthropic':
        return consume_anthropic_stream(records, expected_model, on_text)
    if provider in {'openai', 'deepseek'}:
        return consume_chat_stream(records, expected_model, on_text)
    raise ValueError('model_response_invalid')


def assemble_stream_body(provider, records, expected_model, on_text):
    try:
        return decode_personal_stream(provider, records, expected_model, on_text)
    except ValueError as error:
        # Decoder failures stay inside the personal model error namespace.
        # A pause or cancel raised by the caller must keep its own type.
        if str(error).startswith('stream_'):
            raise ValueError('model_response_invalid') from None
        raise


class _RedactingDeltas:
    """Hold back a key-sized tail so a credential split across chunks is not stored."""

    def __init__(self, secret, observer):
        self.secret = secret or ''
        self.observer = observer
        self.pending = ''

    def __call__(self, text):
        if not isinstance(text, str) or text == '':
            return
        self.pending += text
        if self.secret:
            self.pending = self.pending.replace(self.secret, '[REDACTED]')
        hold = max(len(self.secret) - 1, 0)
        if len(self.pending) <= hold:
            return
        emit, self.pending = (self.pending, '') if hold == 0 else (self.pending[:-hold], self.pending[-hold:])
        if emit:
            self.observer({'phase': 'delta', 'text': emit})

    def flush(self):
        if self.secret:
            self.pending = self.pending.replace(self.secret, '[REDACTED]')
        if self.pending:
            self.observer({'phase': 'delta', 'text': self.pending})
            self.pending = ''


def post_public_stream(url, headers, payload, on_text, *, provider, expected_model, during_request=None):
    """Stream a personal provider over the same pinned worker. Deltas are forwarded as they arrive."""
    if os.environ.get('EVOMIND_TEST_FIXTURE_ROOT'):
        raise ValueError('model_isolated_network_disabled')
    if provider not in _STREAM_PROVIDERS:
        raise ValueError('model_response_invalid')
    wire = dict(payload)
    wire['stream'] = True
    if provider != 'anthropic':
        wire['stream_options'] = {'include_usage': True}
    request = json.dumps({'url': url, 'headers': {**headers, 'Accept': 'text/event-stream'}, 'payload': wire, 'stream': True}).encode()
    if len(request) > 3 * 1024 * 1024:
        raise ValueError('model_request_too_large')
    process = subprocess.Popen([sys.executable, '-I', '-u', str(Path(__file__).with_name('personal_model_http.py'))],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    events = queue.Queue()

    def read_worker():
        total = 0
        try:
            while True:
                line = process.stdout.readline(2 * 1024 * 1024 + 1)
                if not line:
                    break
                total += len(line)
                if len(line) > 2 * 1024 * 1024 or total > 2 * 1024 * 1024:
                    events.put({'error': 'model_response_too_large'})
                    break
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    events.put({'error': 'model_response_invalid'})
                    break
                if not isinstance(item, dict):
                    events.put({'error': 'model_response_invalid'})
                    break
                events.put(item)
        except Exception:
            events.put({'error': 'model_transport_failed'})
        finally:
            events.put({'end': True})

    reader = threading.Thread(target=read_worker, daemon=True)
    reader.start()
    deadline = time.monotonic() + 30

    def records():
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError('model_total_timeout')
            try:
                item = events.get(timeout=min(0.25, remaining))
            except queue.Empty:
                if during_request:
                    during_request()
                continue
            if during_request:
                during_request()
            if item.get('end') is True:
                return
            if 'error' in item:
                code = item['error']
                if not isinstance(code, str) or not _MODEL_ERROR.fullmatch(code):
                    code = 'model_transport_failed'
                raise ValueError(code)
            data = item.get('data')
            if not isinstance(data, str):
                raise ValueError('model_response_invalid')
            yield data

    try:
        try:
            process.stdin.write(request)
            process.stdin.close()
        except (OSError, subprocess.SubprocessError):
            raise ValueError('model_transport_failed') from None
        body = assemble_stream_body(provider, records(), expected_model, on_text)
        try:
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise ValueError('model_total_timeout') from None
        if process.returncode:
            raise ValueError('model_transport_failed')
        return body
    finally:
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        reader.join(timeout=2)


class PersonalModelClient:
    def __init__(self, store, principal, binding, before_attempt=None, record_use=False, text_observer=None, during_request=None):
        self.store, self.principal, self.binding = store, principal, binding
        self.before_attempt, self.record_use = before_attempt, record_use
        self.text_observer, self.during_request = text_observer, during_request
        profile = store.get(principal, binding['id'], binding['version'])
        if not profile['enabled']:
            raise ValueError('model_profile_disabled')
        key = store.secrets.load(principal.tenant_id, principal.owner_id, binding['id'], binding['version'])
        config = ProviderConfig(profile['provider'], profile['base_url'], profile['model'], key)
        transport = {'anthropic': AnthropicTransport, 'deepseek': DeepSeekTransport, 'openai': OpenAITransport}[profile['provider']]
        self.transport = transport(config)
        self.timeout, self.max_retries = 30, 0

    @property
    def model(self):
        return self.transport.config.model

    def is_available(self):
        return True

    def _resolve_transports(self):
        return [self.transport]

    def send(self, messages, *, system, tools, max_tokens=4096, temperature=0.3):
        if self.before_attempt:
            self.before_attempt()
        # Disabling blocks the next request; edits keep this run's old revision.
        if not self.store.get(self.principal, self.binding['id'])['enabled']:
            raise ValueError('model_profile_disabled')
        url, headers, payload = self.transport.build(messages, system, tools, min(max_tokens, 4096), temperature)
        started = False
        live = False
        redactor = None
        try:
            if self.text_observer is not None:
                started = True
                self.text_observer({'phase': 'start'})
            # Fixtures replace post_public_json. Keep that one-shot seam so existing tests still bind the provider.
            if self.text_observer is not None and post_public_json is _POST_PUBLIC_JSON:
                live = True
                redactor = _RedactingDeltas(self.transport.config.api_key, self.text_observer)
                body = post_public_stream(url, headers, payload, redactor, provider=self.transport.name,
                                          expected_model=self.model, during_request=self.during_request)
            else:
                body = post_public_json(url, headers, payload)
            # A malicious provider must not reflect the credential into stored output.
            body = redact(body, self.transport.config.api_key)
            if self.transport.name != 'anthropic':
                _response_error(body, expected_model=self.model, tools=tools)
            elif (not isinstance(body, dict) or body.get('model') != self.model or 'error' in body
                  or not isinstance(body.get('content'), list) or body.get('stop_reason') not in {'end_turn', 'tool_use'}):
                raise ValueError('model_response_invalid')
            turn = self.transport.parse(body)
            allowed = {tool.name for tool in tools}
            seen = set()
            for call in turn.tool_calls:
                if not call.id or call.id in seen or call.name not in allowed or not isinstance(call.input, dict):
                    raise ValueError('model_tool_response_invalid')
                seen.add(call.id)
            if not turn.text and not turn.tool_calls:
                raise ValueError('model_response_empty')
            if self.record_use:
                with self.store.connect() as connection:
                    connection.execute('UPDATE revisions SET used_at=? WHERE id=? AND version=?', (utc_now(), self.binding['id'], self.binding['version']))
            if started:
                if live and redactor is not None:
                    redactor.flush()
                elif turn.text:
                    self.text_observer({'phase': 'delta', 'text': turn.text})
                self.text_observer({'phase': 'complete'})
            return turn
        except Exception:
            if started and self.text_observer is not None:
                try:
                    self.text_observer({'phase': 'reset'})
                except Exception:
                    pass
            raise
