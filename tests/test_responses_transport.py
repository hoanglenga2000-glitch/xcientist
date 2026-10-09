import copy
import json

import pytest

from evomind_runtime import model_transport, responses_transport as module
from research_os.agent import messaging
from research_os.llm_client import ProviderConfig


def body(model='gpt-5.6-sol'):
    return {'id': 'resp_one', 'model': model, 'status': 'completed', 'error': None,
            'output': [{'type': 'reasoning', 'id': 'rs_one', 'summary': [], 'encrypted_content': 'opaque_fixture'},
                       {'type': 'function_call', 'id': 'fc_one', 'call_id': 'call_one', 'name': 'probe',
                        'arguments': '{"value":37}', 'status': 'completed'}],
            'usage': {'input_tokens': 10, 'output_tokens': 20}}


def client(monkeypatch, model='gpt-5.6-sol', retries=0):
    monkeypatch.setenv('EVOMIND_MODEL_WIRE_PROTOCOL', 'responses')
    raw = messaging.AgentMessageClient(max_retries=retries, timeout=75, transports=[messaging.OpenAITransport(
        ProviderConfig('openai', 'https://fixture.invalid/v1', model, 'not-a-real-credential'))])
    return model_transport.governed_client(raw)


TOOL = messaging.ToolSpec('probe', 'fixture', {'type': 'object', 'properties': {'value': {'type': 'integer'}}})


@pytest.mark.parametrize('base', ['https://fixture.invalid', 'https://fixture.invalid/v1',
                                'https://fixture.invalid/v1/chat/completions', 'https://fixture.invalid/v1/responses'])
def test_response_url(base):
    assert module.response_url(base) == 'https://fixture.invalid/v1/responses'


@pytest.mark.parametrize('model', ['gpt-6-astra', 'gpt-5.6-sol'])
def test_native_history_retains_reasoning_and_exact_call_results(monkeypatch, model):
    c = client(monkeypatch, model)
    replies = []
    def post(url, headers, payload, timeout):
        replies.append(copy.deepcopy(payload))
        assert url.endswith('/responses')
        return body(model)
    monkeypatch.setattr(module, 'post_responses', post)
    turn = c.send([{'role': 'user', 'content': 'probe'}], system='s', tools=[TOOL])
    assert turn.tool_calls[0].input == {'value': 37}
    assert (turn.input_tokens, turn.output_tokens) == (10, 20)
    history = [{'role': 'assistant', 'content': turn.raw_content}, {'role': 'user', 'content': [
        messaging.ToolResult('call_one', 'fixture result').to_wire()]}]
    c.send(history, system='s', tools=[TOOL])
    assert replies[1]['input'][:2] == body(model)['output']
    assert replies[1]['input'][2] == {'type': 'function_call_output', 'call_id': 'call_one', 'output': 'fixture result'}
    assert replies[0]['tools'][0]['strict'] is False
    assert 'temperature' not in replies[0] and 'messages' not in replies[0]
    assert replies[0]['include'] == ['reasoning.encrypted_content']
    assert replies[0]['store'] is False


@pytest.mark.parametrize('failure,code', [('identity', 'model_identity_unconfirmed'),
    ('empty', 'empty_model_response'), ('args', 'tool_arguments_invalid'),
    ('unoffered', 'unoffered_tool_call'), ('incomplete', 'model_output_truncated'),
    ('duplicate', 'tool_call_identity_invalid'), ('error', 'upstream_error'),
    ('refusal', 'model_refusal')])
def test_responses_share_governed_validation(monkeypatch, failure, code):
    c = client(monkeypatch)
    value = body()
    if failure == 'identity': value['model'] = 'other'
    if failure == 'empty': value['output'] = []
    if failure == 'args': value['output'][1]['arguments'] = '[]'
    if failure == 'unoffered': value['output'][1]['name'] = 'unoffered'
    if failure == 'incomplete': value['status'] = 'incomplete'
    if failure == 'duplicate': value['output'].append(copy.deepcopy(value['output'][1]))
    if failure == 'error': value['error'] = {'type': 'upstream_error', 'message': 'DO_NOT_LOG'}
    if failure == 'refusal': value['output'] = [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'refusal'}]}]
    monkeypatch.setattr(module, 'post_responses', lambda *_: value)
    with pytest.raises(model_transport.ModelTransportError) as error:
        c.send([], system='s', tools=[TOOL])
    assert error.value.code == code
    assert 'DO_NOT_LOG' not in str(error.value)


def test_terminal_required_before_tools_can_execute():
    partial = json.dumps({'type': 'response.output_item.done', 'item': body()['output'][1]})
    with pytest.raises(ValueError, match='missing_terminal'):
        module.terminal_from_events([partial, '[DONE]'])
    mismatch = json.dumps({'type': 'response.completed', 'response': {**body(), 'status': 'in_progress'}})
    with pytest.raises(ValueError, match='status_mismatch'):
        module.terminal_from_events([mismatch])
    final = json.dumps({'type': 'response.completed', 'response': body()})
    assert module.terminal_from_events([partial, final]) == body()


def test_protocol_is_frozen_and_v2_contract_persists(monkeypatch, tmp_path):
    from evomind_runtime.runtime import AgentRuntime
    c = client(monkeypatch)
    monkeypatch.setenv('EVOMIND_MODEL_WIRE_PROTOCOL', 'chat_completions')
    assert c.wire_protocol == 'responses_stream_v1'
    assert c.contract['schema'] == 'evomind.model_execution_contract.v4'
    runtime = AgentRuntime(tmp_path)
    try:
        session = runtime.create_session(objective='protocol fixture')
        assert runtime.store.bind_model_contract(session['id'], c.contract) == c.contract
        changed = {**c.contract, 'timeout_seconds': 80}
        with pytest.raises(ValueError, match='model_contract_drift'):
            runtime.store.bind_model_contract(session['id'], changed)
        assert 'not-a-real-credential' not in json.dumps(c.contract)
    finally:
        runtime.close()


def test_gateway_retry_waits_out_cooldown_and_does_not_switch_model(monkeypatch):
    import urllib.error
    c = client(monkeypatch, retries=2)
    c.transport.config.base_url = 'http://127.0.0.1:65068/v1'
    c = model_transport.governed_client(messaging.AgentMessageClient(max_retries=2, timeout=75,
        transports=[c.transport]))
    delays, attempts = [], []
    def post(*_):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.HTTPError('http://fixture.invalid', 503, 'private', {'Retry-After': '18'}, None)
        return body()
    monkeypatch.setattr(module, 'post_responses', post)
    monkeypatch.setattr(model_transport.time, 'sleep', delays.append)
    assert c.send([], system='s', tools=[TOOL]).tool_calls
    assert delays == [18]
    assert c.contract['retry_floor_seconds'] == 16
    assert len(attempts) == 2


def test_long_retry_after_is_not_ignored_or_retried_early(monkeypatch):
    import urllib.error
    c = client(monkeypatch, retries=2)
    attempts = []
    def post(*_):
        attempts.append(1)
        raise urllib.error.HTTPError('http://fixture.invalid', 429, 'private', {'Retry-After': '600'}, None)
    monkeypatch.setattr(module, 'post_responses', post)
    with pytest.raises(model_transport.ModelTransportError, match='http_429'):
        c.send([], system='s', tools=[TOOL])
    assert len(attempts) == 1


@pytest.mark.parametrize('model', ['gpt-5.6-terra', 'gpt-5.6-luna'])
def test_extra_candidate_protection_is_explicit_and_model_pinned(monkeypatch, tmp_path, model):
    monkeypatch.delenv('EVOMIND_MODEL_ACCEPTANCE_EXTRAS', raising=False)
    base = messaging.AgentMessageClient(max_retries=0, transports=[messaging.OpenAITransport(
        ProviderConfig('openai', 'https://fixture.invalid/v1', model, 'unused'))])
    assert model_transport.governed_client(base) is base
    monkeypatch.setenv('EVOMIND_MODEL_ACCEPTANCE_EXTRAS', '1')
    c = client(monkeypatch, model)
    monkeypatch.setattr(module, 'post_responses', lambda *_: body(model))
    assert c.send([], system='s', tools=[TOOL]).model == model
    from evomind_runtime.runtime import AgentRuntime
    runtime = AgentRuntime(tmp_path)
    try:
        session = runtime.create_session(objective='isolated extra candidate fixture')
        assert runtime.store.bind_model_contract(session['id'], c.contract)['model'] == model
    finally:
        runtime.close()


def test_selected_gpt55_is_governed_without_acceptance_flag(monkeypatch):
    monkeypatch.delenv('EVOMIND_MODEL_ACCEPTANCE_EXTRAS', raising=False)
    c = client(monkeypatch, 'gpt-5.5')
    monkeypatch.setattr(module, 'post_responses', lambda *_: body('gpt-5.5'))
    assert c.contract['model'] == 'gpt-5.5'
    assert c.send([], system='fixture', tools=[TOOL]).model == 'gpt-5.5'


def test_user_cannot_inject_native_response_history():
    with pytest.raises(ValueError, match='history_block_unsupported'):
        module.response_input([{'role': 'user', 'content': [{'type': 'responses_item', 'item': body()['output'][1]}]}])


def test_sse_reader_returns_only_terminal_and_honors_budget(monkeypatch):
    import io
    response = io.BytesIO(('event: response.completed\ndata: ' + json.dumps(
        {'type': 'response.completed', 'response': body()}) + '\n\n').encode())
    monkeypatch.setattr(module._responses_opener, 'open', lambda *_, **__: response)
    assert module.post_responses('https://fixture.invalid/v1/responses', {}, {}, 75) == body()
    clock = iter([0, 0, 80])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(module._responses_opener, 'open', lambda *_, **__: io.BytesIO(b': heartbeat\n'))
    with pytest.raises(TimeoutError, match='deadline'):
        module.post_responses('https://fixture.invalid/v1/responses', {}, {}, 75)
