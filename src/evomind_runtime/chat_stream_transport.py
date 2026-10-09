"""Bounded Chat Completions SSE decoding; tools remain private until terminal validation."""
from __future__ import annotations

import json
import time
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def consume_chat_stream(records, expected_model, on_text):
    text, calls, usage = [], {}, {}
    finish, identity, done, refusal = None, False, False, None
    for record in records:
        if record == '[DONE]':
            done = True
            break
        chunk = json.loads(record)
        if not isinstance(chunk, dict):
            raise ValueError('stream_chunk_invalid')
        if chunk.get('error'):
            return {'error': chunk['error']}
        if chunk.get('model') is not None:
            if chunk['model'] != expected_model:
                raise ValueError('stream_model_identity_invalid')
            identity = True
        if isinstance(chunk.get('usage'), dict):
            usage = chunk['usage']
        choices = chunk.get('choices')
        if choices == []:
            continue
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict) or choices[0].get('index', 0) != 0:
            raise ValueError('stream_choices_invalid')
        choice = choices[0]
        delta = choice.get('delta') or {}
        if not isinstance(delta, dict) or not identity:
            raise ValueError('stream_model_identity_invalid')
        if finish is not None:
            raise ValueError('stream_data_after_finish')
        content = delta.get('content')
        if content is not None:
            if not isinstance(content, str):
                raise ValueError('stream_content_invalid')
            text.append(content)
            if content:
                on_text(content)
        # Never expose reasoning_content or partial tool arguments to the UI.
        if delta.get('refusal'):
            refusal = True
        parts = delta.get('tool_calls') or []
        if not isinstance(parts, list):
            raise ValueError('stream_tool_calls_invalid')
        for part in parts:
            if not isinstance(part, dict):
                raise ValueError('stream_tool_fragment_invalid')
            index = part.get('index')
            if type(index) is not int or not 0 <= index < 128:
                raise ValueError('stream_tool_index_invalid')
            call = calls.setdefault(index, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
            if part.get('type') not in (None, 'function'):
                raise ValueError('stream_tool_type_invalid')
            if part.get('id'):
                if call['id'] and call['id'] != part['id']:
                    raise ValueError('stream_tool_id_changed')
                call['id'] = part['id']
            function = part.get('function') or {}
            if not isinstance(function, dict):
                raise ValueError('stream_tool_fragment_invalid')
            for name in ('name', 'arguments'):
                fragment = function.get(name)
                if fragment is not None:
                    if not isinstance(fragment, str):
                        raise ValueError('stream_tool_fragment_invalid')
                    call['function'][name] += fragment
        if choice.get('finish_reason') is not None:
            finish = choice['finish_reason']
    if not identity or not done or finish is None:
        raise ValueError('stream_terminal_missing')
    return {'model': expected_model, 'choices': [{'finish_reason': finish, 'message': {
        'content': ''.join(text), 'tool_calls': [calls[i] for i in sorted(calls)], 'refusal': refusal}}], 'usage': usage}


def post_chat_stream(url, headers, payload, timeout, on_text):
    payload = {**payload, 'stream': True, 'stream_options': {'include_usage': True}}
    request = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), method='POST',
                                     headers={**headers, 'Accept': 'text/event-stream'})
    deadline = time.monotonic() + timeout
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
        def records():
            total, lines = 0, []
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('stream_deadline_exceeded')
                sock = getattr(getattr(getattr(response, 'fp', None), 'raw', None), '_sock', None)
                if sock is not None:
                    sock.settimeout(remaining)
                raw = response.readline(1_048_577)
                if not raw:
                    break
                total += len(raw)
                if len(raw) > 1_048_576 or total > 16_777_216:
                    raise ValueError('stream_size_exceeded')
                line = raw.decode('utf-8').rstrip('\r\n')
                if not line:
                    if lines:
                        yield '\n'.join(lines)
                        lines = []
                elif line.startswith('data:'):
                    lines.append(line[5:].lstrip())
            if lines:
                yield '\n'.join(lines)
        return consume_chat_stream(records(), str(payload['model']), on_text)


class PublicTextStream:
    """Cumulative, throttled snapshots make cursor replay idempotent."""
    def __init__(self, emit, guard=lambda: None):
        self.emit, self.guard = emit, guard
        self.text, self.last_emit = '', 0.0

    def __call__(self, event):
        self.guard()
        phase = event['phase']
        if phase in {'start', 'reset'}:
            self.text = ''
        elif phase == 'delta':
            self.text += event['text']
        now = time.monotonic()
        if phase != 'delta' or now - self.last_emit >= 0.1:
            self.emit({'text': self.text, 'phase': 'streaming' if phase in {'start', 'delta'} else phase})
            self.last_emit = now
