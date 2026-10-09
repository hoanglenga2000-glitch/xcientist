"""Opt-in native Responses transport for EvoMind, not other applications.

Only a complete terminal envelope may reach the tool dispatcher. Opaque
reasoning items stay in the private, durable message history for the next turn.
"""
from __future__ import annotations

import copy
import json
import time
import urllib.request

from research_os.agent import messaging


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # The approved endpoint is the only recipient of its Authorization
        # header. Let urllib surface the original HTTP error, even for same-
        # origin redirects; following one would change the frozen route.
        return None


# A private opener leaves other HTTP clients and applications unchanged.
_responses_opener = urllib.request.build_opener(_NoRedirectHandler())


def response_url(base):
    base = base.rstrip('/')
    if base.endswith('/chat/completions'):
        base = base[:-len('/chat/completions')]
    if base.endswith('/responses'):
        return base
    return base + ('/responses' if base.endswith('/v1') else '/v1/responses')


def response_input(messages):
    wire = []
    for message in messages:
        role, content = message.get('role', 'user'), message.get('content', '')
        if isinstance(content, str):
            wire.append({'role': role, 'content': content})
            continue
        if not isinstance(content, list):
            raise ValueError('responses_history_invalid')
        for block in content:
            kind = block.get('type')
            if kind == 'responses_item' and role == 'assistant':
                wire.append(copy.deepcopy(block['item']))
            elif kind == 'text':
                wire.append({'role': role, 'content': str(block.get('text', ''))})
            elif kind == 'tool_use' and role == 'assistant':
                wire.append({'type': 'function_call', 'call_id': block['id'], 'name': block['name'],
                             'arguments': json.dumps(block.get('input', {}), ensure_ascii=False)})
            elif kind == 'tool_result' and role == 'user':
                output = block.get('content', '')
                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False)
                if block.get('is_error'):
                    output = '[tool_error] ' + output
                wire.append({'type': 'function_call_output', 'call_id': block['tool_use_id'], 'output': output})
            else:
                raise ValueError('responses_history_block_unsupported')
    return wire


def as_chat_envelope(body):
    """Reuse the common identity/tool contract without losing native history."""
    if not isinstance(body, dict):
        return body
    if body.get('error'):
        return {'error': body['error']}
    status = body.get('status')
    if status not in {'completed', 'incomplete', 'failed'}:
        raise ValueError('responses_terminal_missing')
    if status == 'failed':
        return {'error': {'type': 'provider_error'}}
    output = body.get('output')
    if not isinstance(output, list):
        raise ValueError('responses_output_invalid')
    text, calls, refusal = [], [], False
    for item in output:
        kind = item.get('type')
        if kind == 'function_call':
            if item.get('status') not in (None, 'completed'):
                raise ValueError('responses_tool_incomplete')
            calls.append({'id': item.get('call_id'), 'type': 'function', 'function': {
                'name': item.get('name'), 'arguments': item.get('arguments')}})
        elif kind == 'message':
            if item.get('role') != 'assistant' or item.get('status') not in (None, 'completed'):
                raise ValueError('responses_message_invalid')
            for part in item.get('content', []):
                if part.get('type') == 'output_text' and isinstance(part.get('text'), str):
                    text.append(part['text'])
                elif part.get('type') == 'refusal':
                    refusal = True
                else:
                    raise ValueError('responses_content_invalid')
        elif kind != 'reasoning':
            raise ValueError('responses_unoffered_output_type')
    finish = 'length' if status == 'incomplete' else ('tool_calls' if calls else 'stop')
    usage = body.get('usage') or {}
    return {'model': body.get('model'), 'choices': [{'finish_reason': finish, 'message': {
        'content': ''.join(text), 'tool_calls': calls, 'refusal': refusal}}],
        'usage': {'prompt_tokens': usage.get('input_tokens', 0), 'completion_tokens': usage.get('output_tokens', 0)}}


def terminal_from_events(records):
    for data in records:
        if data == '[DONE]':
            break
        event = json.loads(data)
        if event.get('error'):
            return {'error': event['error']}
        if event.get('type') in {'response.completed', 'response.incomplete', 'response.failed'}:
            response = event.get('response')
            if not isinstance(response, dict):
                raise ValueError('responses_terminal_invalid')
            expected = 'response.' + str(response.get('status'))
            if event['type'] != expected:
                raise ValueError('responses_terminal_status_mismatch')
            return response
    raise ValueError('responses_stream_missing_terminal')


def post_responses(url, headers, payload, timeout):
    """One HTTP attempt, bounded by time/bytes; no SDK/gateway retry loop here."""
    started = time.monotonic()
    request = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), method='POST',
        headers={'User-Agent': 'evomind-responses/1.0', 'Accept': 'text/event-stream', **headers})
    with _responses_opener.open(request, timeout=timeout) as response:
        def records():
            size, data_lines = 0, []
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError('responses_deadline_exceeded')
                # urllib's socket timeout is normally idle-only. Rebind it to
                # the remaining total budget before each potentially blocking read.
                sock = getattr(getattr(getattr(response, 'fp', None), 'raw', None), '_sock', None)
                if sock is not None:
                    sock.settimeout(remaining)
                raw = response.readline(1_048_577)
                if not raw:
                    break
                size += len(raw)
                if len(raw) > 1_048_576 or size > 16_777_216:
                    raise ValueError('responses_size_limit')
                if time.monotonic() - started >= timeout:
                    raise TimeoutError('responses_deadline_exceeded')
                line = raw.decode('utf-8').rstrip('\r\n')
                if not line:
                    if data_lines:
                        yield '\n'.join(data_lines)
                        data_lines = []
                elif line.startswith('data:'):
                    data_lines.append(line[5:].lstrip())
                elif line.startswith('{') and not data_lines:
                    # Some compatible servers label JSON errors as SSE.
                    yield line
            if data_lines:
                yield '\n'.join(data_lines)
        return terminal_from_events(records())


class OpenAIResponsesTransport(messaging.OpenAITransport):
    wire_protocol = 'responses_stream_v1'

    def build(self, messages, system, tools, max_tokens, temperature):
        payload = {'model': self.config.model, 'input': response_input(messages), 'instructions': system,
                   'stream': True, 'store': False, 'include': ['reasoning.encrypted_content'],
                   'max_output_tokens': max_tokens}
        if self.config.reasoning_effort:
            payload['reasoning'] = {'effort': self.config.reasoning_effort}
        if self.config.service_tier:
            payload['service_tier'] = self.config.service_tier
        if tools:
            payload['tools'] = [{'type': 'function', 'name': t.name, 'description': t.description,
                                 'parameters': copy.deepcopy(t.input_schema), 'strict': False} for t in tools]
        return response_url(self.config.base_url), {'Authorization': 'Bearer ' + self.config.api_key,
                                                   'content-type': 'application/json'}, payload

    def parse(self, body):
        turn = messaging._parse_openai_turn(as_chat_envelope(body), self.config.model, provider=self.name)
        turn.raw_content = [{'type': 'responses_item', 'item': copy.deepcopy(item)} for item in body['output']]
        turn.request_profile = {'wire_protocol': self.wire_protocol}
        return turn
