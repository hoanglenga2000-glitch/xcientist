import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from evomind_runtime.chat_stream_transport import consume_chat_stream, PublicTextStream
from evomind_runtime import model_transport as transport
from research_os.agent import messaging
from research_os.llm_client import ProviderConfig


MODEL = 'deepseek-flash'


def chunk(delta=None, finish=None, model=MODEL):
    return json.dumps({'model': model, 'choices': [{'index': 0, 'delta': delta or {}, 'finish_reason': finish}]})


def test_deltas_arrive_before_terminal_and_exclude_reasoning_and_tool_arguments():
    seen = []
    def records():
        yield chunk({'content': 'First', 'reasoning_content': 'PRIVATE_REASONING'})
        assert seen == ['First']
        yield chunk({'tool_calls': [{'index': 0, 'id': 'call_fixture', 'function': {'name': 'probe', 'arguments': '{"v":'}}]})
        yield chunk({'tool_calls': [{'index': 0, 'function': {'arguments': '1}'}}]})
        assert seen == ['First']
        yield chunk(finish='tool_calls')
        yield '[DONE]'
    reply = consume_chat_stream(records(), MODEL, seen.append)
    assert reply['choices'][0]['message']['tool_calls'][0]['function']['arguments'] == '{"v":1}'
    assert 'PRIVATE_REASONING' not in json.dumps(reply)


@pytest.mark.parametrize('records,code', [
    ([chunk({'content': 'wrong'}, model='other')], 'stream_model_identity_invalid'),
    ([chunk({'content': 'incomplete'})], 'stream_terminal_missing'),
    ([chunk({'content': 'incomplete'}), '[DONE]'], 'stream_terminal_missing'),
    ([json.dumps({'model': MODEL, 'choices': [None]})], 'stream_choices_invalid'),
    ([chunk({'tool_calls': ['invalid']})], 'stream_tool_fragment_invalid'),
])
def test_incomplete_or_invalid_stream_never_returns_a_turn(records, code):
    with pytest.raises(ValueError, match=code):
        consume_chat_stream(records, MODEL, lambda text: None)


def test_cumulative_text_throttles_without_losing_last_delta(monkeypatch):
    times = iter([1, 1.02, 1.03, 1.2, 1.21, 1.22])
    monkeypatch.setattr('evomind_runtime.chat_stream_transport.time.monotonic', lambda: next(times))
    rows = []
    stream = PublicTextStream(rows.append)
    for event in [{'phase': 'start'}, {'phase': 'delta', 'text': 'a'}, {'phase': 'delta', 'text': 'b'},
                  {'phase': 'delta', 'text': 'c'}, {'phase': 'complete'}, {'phase': 'reset'}]:
        stream(event)
    assert rows == [{'text': '', 'phase': 'streaming'}, {'text': 'abc', 'phase': 'streaming'},
                    {'text': 'abc', 'phase': 'complete'}, {'text': '', 'phase': 'reset'}]


@contextmanager
def server_for(script):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert request['stream'] is True
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            def emit(data):
                self.wfile.write(('data: ' + data + '\n\n').encode())
                self.wfile.flush()
            try:
                script(emit)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield 'http://127.0.0.1:%d/v1/chat/completions' % server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_private_worker_delivers_real_http_delta_before_server_finishes():
    delivered = threading.Event()
    observed_before_terminal = []
    def script(emit):
        emit(chunk({'content': '第一段'}))
        observed_before_terminal.append(delivered.wait(3))
        emit(chunk({'content': '第二段'}))
        emit(chunk(finish='stop'))
        emit('[DONE]')
    seen = []
    def on_text(text):
        seen.append(text)
        delivered.set()
    with server_for(script) as url:
        body = transport.post_stream_with_deadline(url, {'Content-Type': 'application/json'}, {'model': MODEL}, 5, on_text)
    assert observed_before_terminal == [True]
    assert seen == ['第一段', '第二段']
    assert body['choices'][0]['message']['content'] == '第一段第二段'


def test_trickling_stream_cannot_extend_total_deadline():
    def script(emit):
        for _ in range(100):
            emit(chunk({'content': '.'}))
            time.sleep(0.06)
    with server_for(script) as url:
        start = time.monotonic()
        with pytest.raises(transport.ModelTransportError, match='model_total_deadline_exceeded'):
            transport.post_stream_with_deadline(url, {}, {'model': MODEL}, 0.8, lambda text: None)
        assert time.monotonic() - start < 3


def test_pause_can_interrupt_silent_stream():
    cancel = threading.Event()
    def script(emit):
        emit(chunk({'role': 'assistant'}))
        cancel.set()
        time.sleep(1)
    def guard():
        if cancel.is_set():
            raise RuntimeError('user_pause_requested')
    with server_for(script) as url:
        with pytest.raises(RuntimeError, match='user_pause_requested'):
            transport.post_stream_with_deadline(url, {}, {'model': MODEL}, 5, lambda text: None, guard)


def test_truncation_clears_partial_text_and_never_returns_tools(monkeypatch):
    events = []
    def post(url, headers, payload, timeout, on_text, check_cancel):
        on_text('unfinished')
        return consume_chat_stream([chunk({'content': 'unfinished'}), chunk(finish='length'), '[DONE]'], MODEL, lambda text: None)
    monkeypatch.setattr(transport, 'post_stream_with_deadline', post)
    base = messaging.AgentMessageClient(max_retries=2, transports=[
        messaging.OpenAITransport(ProviderConfig('openai', 'http://fixture.invalid/v1', MODEL, 'fixture'))])
    client = transport.governed_client(base, text_observer=events.append)
    with pytest.raises(transport.ModelTransportError, match='model_output_truncated'):
        client.send([], system='fixture', tools=[])
    assert [row['phase'] for row in events] == ['start', 'delta', 'reset']
    assert client.last_attempt_count == 1


def test_persisted_partial_snapshot_and_cursor_replay(tmp_path):
    from evomind_runtime.runtime import AgentRuntime
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt='Stream fixture', start=False)
        runtime.store.update_assistant_run(run['id'], status='running')
        runtime.store.append_event(run['id'], 'assistant.stream', {'text': 'First', 'phase': 'streaming'})
        first = runtime.assistant.events(run['id'])[-1]
        assert first['event_type'] == 'assistant_text'
        assert runtime.assistant.snapshot(run['id'])['streaming_text'] == 'First'
        runtime.store.append_event(run['id'], 'assistant.stream', {'text': 'First second', 'phase': 'streaming'})
        replay = runtime.assistant.events(run['id'], first['seq'])
        assert len(replay) == 1 and replay[0]['payload']['text'] == 'First second'
        runtime.store.append_event(run['id'], 'assistant.stream', {'text': '', 'phase': 'reset'})
        assert runtime.assistant.snapshot(run['id'])['streaming_text'] is None
        assert not runtime.store.list_turns(run['id'])
    finally:
        runtime.close()


def test_stream_polling_does_not_repeat_model_budget_admission(monkeypatch):
    admissions, cancellations = [], []
    def post(url, headers, payload, timeout, on_text, check_cancel):
        for _ in range(3):
            check_cancel()
        on_text('done')
        return consume_chat_stream([chunk({'content': 'done'}), chunk(finish='stop'), '[DONE]'], MODEL, lambda text: None)
    monkeypatch.setattr(transport, 'post_stream_with_deadline', post)
    base = messaging.AgentMessageClient(max_retries=0, transports=[
        messaging.OpenAITransport(ProviderConfig('openai', 'http://fixture.invalid/v1', MODEL, 'fixture'))])
    client = transport.governed_client(base, text_observer=lambda event: None,
        before_attempt=lambda: admissions.append(1), during_request=lambda: cancellations.append(1))
    assert client.send([], system='fixture', tools=[]).text == 'done'
    assert admissions == [1] and len(cancellations) == 3
