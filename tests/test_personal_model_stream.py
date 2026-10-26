"""Personal-model streaming stays on the pinned public worker and never surfaces hidden fragments."""
import io
import json

import pytest

from evomind_runtime import personal_model_client
from evomind_runtime.personal_model_client import (
    PersonalModelClient, _RedactingDeltas, assemble_stream_body, consume_anthropic_stream,
    decode_personal_stream, post_public_stream,
)
from evomind_runtime.personal_model_http import SseAssembler, public_error, request, stream_request
from research_os.agent.messaging import AnthropicTransport, DeepSeekTransport, OpenAITransport
from research_os.llm_client import ProviderConfig


MODEL = 'fixture-chat'
SECRET = 'secret-key'


def anthropic_event(payload):
    return json.dumps(payload)


def chat_chunk(delta, finish=None, model=MODEL):
    return json.dumps({'model': model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]})


class _Store:
    def get(self, *args, **kwargs):
        return {'enabled': True}


def client(observer=None, provider='openai'):
    obj = object.__new__(PersonalModelClient)
    obj.before_attempt = None
    obj.record_use = False
    obj.text_observer = observer
    obj.during_request = None
    obj.binding = {'id': 'profile', 'version': 1}
    obj.principal = None
    obj.store = _Store()
    config = ProviderConfig(provider, 'https://api.example.com/v1', MODEL, SECRET)
    transport = {'openai': OpenAITransport, 'anthropic': AnthropicTransport, 'deepseek': DeepSeekTransport}[provider]
    obj.transport = transport(config)
    return obj


def test_sse_assembler_joins_data_and_ignores_event_lines():
    assembler = SseAssembler()
    assert assembler.feed(b'event: content_block_delta\n') is None
    assert assembler.feed(b'data: {"text": "a"}\n') is None
    assert assembler.feed(b'\n') == '{"text": "a"}'
    assert assembler.feed(b': keep-alive\n') is None
    assert assembler.finish() is None


def test_sse_assembler_rejects_non_sse_and_oversized_bytes():
    assembler = SseAssembler(limit=8)
    with pytest.raises(ValueError, match='model_response_invalid'):
        assembler.feed(b'not-sse\n')
    with pytest.raises(ValueError, match='model_response_too_large'):
        SseAssembler(limit=8).feed(b'data: too-long\n')


def test_oneshot_request_still_returns_json(monkeypatch):
    class Body:
        def read(self, limit):
            assert limit == 1024 * 1024 + 1
            return b'{"ok": 1}'

    class Closable:
        def close(self):
            self.closed = True

    monkeypatch.setattr('evomind_runtime.personal_model_http._exchange', lambda incoming: (Closable(), Closable(), Body()))
    assert request({'url': 'https://api.example.com/v1', 'payload': {}}) == {'ok': 1}


def test_oneshot_request_keeps_the_one_megabyte_cap(monkeypatch):
    class Body:
        def read(self, limit):
            return b'x' * limit

    class Closable:
        def close(self):
            return None

    monkeypatch.setattr('evomind_runtime.personal_model_http._exchange', lambda incoming: (Closable(), Closable(), Body()))
    with pytest.raises(ValueError, match='model_response_too_large'):
        request({'url': 'https://api.example.com/v1', 'payload': {}})


def test_stream_request_prints_one_json_object_per_sse_event(capsys, monkeypatch):
    class Response:
        def __init__(self):
            self.lines = [b'event: ping\n', b'data: {"type":"ping"}\n', b'\n', b'data: [DONE]\n', b'\n']

        def readline(self, limit=-1):
            assert limit > 8
            return self.lines.pop(0) if self.lines else b''

    class Closable:
        def close(self):
            return None

    monkeypatch.setattr('evomind_runtime.personal_model_http._exchange', lambda incoming: (Closable(), Closable(), Response()))
    stream_request({'url': 'https://api.example.com/v1', 'headers': {}, 'payload': {'stream': True}})
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines == [{'data': '{"type":"ping"}'}, {'data': '[DONE]'}]


def test_public_error_does_not_echo_unclassified_exceptions():
    assert public_error(ValueError('model_http_401')) == {'error': 'model_http_401'}
    assert public_error(ValueError('timed out with secret-key')) == {'error': 'model_network_failed'}


def test_anthropic_stream_hides_thinking_and_tool_fragments():
    records = [
        anthropic_event({'type': 'message_start', 'message': {'model': MODEL, 'usage': {'input_tokens': 3, 'output_tokens': 1}}}),
        anthropic_event({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'thinking', 'thinking': ''}}),
        anthropic_event({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'thinking_delta', 'thinking': 'PRIVATE_THOUGHT'}}),
        anthropic_event({'type': 'content_block_stop', 'index': 0}),
        anthropic_event({'type': 'content_block_start', 'index': 1, 'content_block': {'type': 'text', 'text': ''}}),
        anthropic_event({'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'text_delta', 'text': 'Hello'}}),
        anthropic_event({'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'text_delta', 'text': ' there'}}),
        anthropic_event({'type': 'content_block_stop', 'index': 1}),
        anthropic_event({'type': 'content_block_start', 'index': 2, 'content_block': {'type': 'tool_use', 'id': 'toolu_1', 'name': 'probe', 'input': {}}}),
        anthropic_event({'type': 'content_block_delta', 'index': 2, 'delta': {'type': 'input_json_delta', 'partial_json': '{"v":'}}),
        anthropic_event({'type': 'content_block_delta', 'index': 2, 'delta': {'type': 'input_json_delta', 'partial_json': '1}'}}),
        anthropic_event({'type': 'content_block_stop', 'index': 2}),
        anthropic_event({'type': 'ping'}),
        anthropic_event({'type': 'message_delta', 'delta': {'stop_reason': 'tool_use'}, 'usage': {'output_tokens': 6}}),
        anthropic_event({'type': 'message_stop'}),
    ]
    seen = []
    body = consume_anthropic_stream(records, MODEL, seen.append)
    assert seen == ['Hello', ' there']
    assert 'PRIVATE_THOUGHT' not in json.dumps(body)
    assert body['content'] == [
        {'type': 'text', 'text': 'Hello there'},
        {'type': 'tool_use', 'id': 'toolu_1', 'name': 'probe', 'input': {'v': 1}},
    ]
    assert body['stop_reason'] == 'tool_use'
    assert body['usage'] == {'input_tokens': 3, 'output_tokens': 6}


def test_anthropic_stream_requires_the_expected_model_and_a_terminal_event():
    with pytest.raises(ValueError, match='stream_model_identity_invalid'):
        consume_anthropic_stream([
            anthropic_event({'type': 'message_start', 'message': {'model': 'other'}}),
        ], MODEL, lambda text: None)
    with pytest.raises(ValueError, match='model_response_invalid'):
        assemble_stream_body('anthropic', [
            anthropic_event({'type': 'message_start', 'message': {'model': MODEL, 'usage': {'input_tokens': 1}}}),
        ], MODEL, lambda text: None)


def test_deepseek_uses_chat_completions_sse_without_reasoning_or_tool_text():
    seen = []
    body = decode_personal_stream('deepseek', [
        chat_chunk({'content': 'Hi', 'reasoning_content': 'SECRET_REASON'}),
        chat_chunk({'tool_calls': [{'index': 0, 'id': 'call_1', 'function': {'name': 'probe', 'arguments': '{"a":'}}]}),
        chat_chunk({'tool_calls': [{'index': 0, 'function': {'arguments': '1}'}}]}),
        chat_chunk({}, finish='tool_calls'),
        '[DONE]',
    ], MODEL, seen.append)
    assert seen == ['Hi']
    assert body['choices'][0]['message']['tool_calls'][0]['function']['arguments'] == '{"a":1}'
    assert 'SECRET_REASON' not in json.dumps(seen)


def test_split_credential_is_redacted_before_any_observer_delta():
    events = []
    redactor = _RedactingDeltas(SECRET, events.append)
    redactor('sec')
    redactor('ret-key!')
    assert events == [] or all(SECRET not in item['text'] for item in events)
    redactor.flush()
    assert ''.join(item['text'] for item in events) == '[REDACTED]!'
    assert all(SECRET not in item['text'] for item in events)


def test_observer_streams_openai_text_and_redacts_the_stored_turn(monkeypatch):
    events = []
    seen = {}

    def fake_stream(url, headers, payload, on_text, *, provider, expected_model, during_request=None):
        seen.update(provider=provider, model=expected_model, during=during_request)
        assert 'stream' not in payload
        on_text('Hello ')
        on_text(SECRET)
        return {'model': MODEL, 'choices': [{'finish_reason': 'stop', 'message': {'content': 'Hello ' + SECRET}}]}

    monkeypatch.setattr(personal_model_client, 'post_public_stream', fake_stream)
    current = client(events.append)
    marker = object()
    current.during_request = marker
    turn = current.send([{'role': 'user', 'content': 'hi'}], system='s', tools=[])
    assert turn.text == 'Hello [REDACTED]'
    assert [item['phase'] for item in events] == ['start', 'delta', 'delta', 'complete']
    assert ''.join(item['text'] for item in events if item['phase'] == 'delta') == 'Hello [REDACTED]'
    assert SECRET not in json.dumps(events)
    assert seen == {'provider': 'openai', 'model': MODEL, 'during': marker}


def test_failed_stream_resets_the_observer(monkeypatch):
    events = []

    def fake_stream(url, headers, payload, on_text, *, provider, expected_model, during_request=None):
        on_text('partial-text-that-is-visible')
        raise ValueError('model_total_timeout')

    monkeypatch.setattr(personal_model_client, 'post_public_stream', fake_stream)
    with pytest.raises(ValueError, match='model_total_timeout'):
        client(events.append).send([{'role': 'user', 'content': 'hi'}], system='s', tools=[])
    assert [item['phase'] for item in events] == ['start', 'delta', 'reset']
    assert events[-1]['phase'] == 'reset'


def test_fixture_replacement_of_post_public_json_stays_one_shot(monkeypatch):
    events = []
    payloads = []

    def fake_json(url, headers, payload):
        payloads.append(payload)
        return {'model': MODEL, 'choices': [{'finish_reason': 'stop', 'message': {'content': 'Hello ' + SECRET}}]}

    def fake_stream(*args, **kwargs):
        raise AssertionError('fixture seam must keep using post_public_json')

    monkeypatch.setattr(personal_model_client, 'post_public_json', fake_json)
    monkeypatch.setattr(personal_model_client, 'post_public_stream', fake_stream)
    turn = client(events.append).send([{'role': 'user', 'content': 'hi'}], system='s', tools=[])
    assert 'stream' not in payloads[0]
    assert turn.text == 'Hello [REDACTED]'
    assert [item['phase'] for item in events] == ['start', 'delta', 'complete']
    assert events[1]['text'] == 'Hello [REDACTED]'


def test_without_an_observer_the_client_does_not_open_a_stream(monkeypatch):
    calls = []

    def fake_json(url, headers, payload):
        calls.append('json')
        return {'model': MODEL, 'choices': [{'finish_reason': 'stop', 'message': {'content': 'OK'}}]}

    def fake_stream(*args, **kwargs):
        calls.append('stream')

    monkeypatch.setattr(personal_model_client, 'post_public_json', fake_json)
    monkeypatch.setattr(personal_model_client, 'post_public_stream', fake_stream)
    turn = client(None).send([{'role': 'user', 'content': 'hi'}], system='s', tools=[])
    assert turn.text == 'OK'
    assert calls == ['json']


def test_anthropic_observer_parses_the_reconstructed_message(monkeypatch):
    events = []

    def fake_stream(url, headers, payload, on_text, *, provider, expected_model, during_request=None):
        assert provider == 'anthropic'
        on_text('OK')
        return {'model': MODEL, 'content': [{'type': 'text', 'text': 'OK'}], 'stop_reason': 'end_turn',
                'usage': {'input_tokens': 1, 'output_tokens': 1}}

    monkeypatch.setattr(personal_model_client, 'post_public_stream', fake_stream)
    turn = client(events.append, provider='anthropic').send([{'role': 'user', 'content': 'hi'}], system='s', tools=[])
    assert turn.provider == 'anthropic' and turn.text == 'OK' and turn.output_tokens == 1
    assert [item['phase'] for item in events] == ['start', 'delta', 'complete']


class _Stdin(bytearray):
    def __init__(self):
        super().__init__()
        self.closed = False

    def write(self, data):
        self.extend(data)
        return len(data)

    def close(self):
        self.closed = True


class _Proc:
    def __init__(self, stdout_bytes):
        self.stdin = _Stdin()
        self.stdout = io.BytesIO(stdout_bytes)
        self.returncode = None

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


def test_post_public_stream_decodes_worker_lines_as_they_arrive(monkeypatch):
    lines = [
        {'data': chat_chunk({'content': 'Hi'})},
        {'data': chat_chunk({}, finish='stop')},
        {'data': '[DONE]'},
    ]
    proc = _Proc(b''.join(json.dumps(item).encode() + b'\n' for item in lines))
    monkeypatch.setattr(personal_model_client.subprocess, 'Popen', lambda *args, **kwargs: proc)
    seen = []
    body = post_public_stream('https://api.example.com/v1/chat/completions', {'Authorization': 'Bearer ' + SECRET},
                              {'model': MODEL, 'messages': []}, seen.append, provider='openai', expected_model=MODEL)
    sent = json.loads(bytes(proc.stdin))
    assert sent['stream'] is True and sent['payload']['stream'] is True
    assert sent['payload']['stream_options'] == {'include_usage': True}
    assert sent['headers']['Accept'] == 'text/event-stream'
    assert SECRET not in json.dumps({key: value for key, value in sent['headers'].items() if key != 'Authorization'})
    assert seen == ['Hi']
    assert body['choices'][0]['message']['content'] == 'Hi'
    assert proc.stdin.closed


def test_post_public_stream_forwards_worker_model_errors_and_blocks_isolated_mode(monkeypatch):
    proc = _Proc(b'{"error": "model_http_401"}\n')
    monkeypatch.setattr(personal_model_client.subprocess, 'Popen', lambda *args, **kwargs: proc)
    with pytest.raises(ValueError, match='model_http_401'):
        post_public_stream('https://api.example.com/v1/messages', {}, {'model': MODEL}, lambda text: None,
                           provider='anthropic', expected_model=MODEL)
    sent = json.loads(bytes(proc.stdin))
    assert sent['payload']['stream'] is True and 'stream_options' not in sent['payload']

    def refuse_popen(*args, **kwargs):
        raise AssertionError('isolated mode must not start the worker')

    monkeypatch.setattr(personal_model_client.subprocess, 'Popen', refuse_popen)
    monkeypatch.setenv('EVOMIND_TEST_FIXTURE_ROOT', 'isolated-test')
    with pytest.raises(ValueError, match='model_isolated_network_disabled'):
        post_public_stream('https://api.example.com/v1', {}, {'model': MODEL}, lambda text: None,
                           provider='openai', expected_model=MODEL)
