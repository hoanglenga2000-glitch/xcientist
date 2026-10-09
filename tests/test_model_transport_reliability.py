import json
import urllib.error
from types import SimpleNamespace

import pytest

from evomind_runtime import model_transport as module
from research_os.agent import messaging
from research_os.llm_client import ProviderConfig


def client(retries=0, observer=None, model="gpt-6-astra", before_attempt=None):
    base = messaging.AgentMessageClient(max_retries=retries, transports=[
        messaging.OpenAITransport(ProviderConfig("openai", "http://fixture.invalid/v1", model, "not-used"))
    ])
    return module.governed_client(base, observer, before_attempt=before_attempt)


def reply(text="ready", calls=None, model="gpt-6-astra"):
    return {"model": model, "choices": [{"finish_reason": "tool_calls" if calls else "stop",
            "message": {"content": text, "tool_calls": calls or []}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def test_astra_explicit_optional_tool_schema_and_empty_tools(monkeypatch):
    payloads = []
    def post(_url, _headers, payload, _timeout):
        payloads.append(payload)
        return reply()
    monkeypatch.setattr(messaging, "_post_json", post)
    tool = messaging.ToolSpec("probe", "fixture", {"type": "object", "properties": {"optional": {"type": "string"}}})
    original = json.loads(json.dumps(tool.input_schema))
    assert client().send([], system="s", tools=[tool]).text == "ready"
    assert payloads[0]["tools"][0]["function"]["strict"] is False
    assert tool.input_schema == original
    client().send([], system="s", tools=[])
    assert "tools" not in payloads[1]


def test_http200_error_retries_without_leaking_body(monkeypatch):
    answers = iter([{"error": {"type": "upstream_error", "message": "DO_NOT_LOG"}}, reply()])
    events = []
    monkeypatch.setattr(messaging, "_post_json", lambda *_: next(answers))
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    c = client(2, events.append)
    assert c.send([], system="s", tools=[]).text == "ready"
    assert c.last_attempt_count == 2
    assert [row["status"] for row in events] == ["failed", "completed"]
    assert events[0]["token_usage_known"] is False
    assert "DO_NOT_LOG" not in json.dumps(events)


def test_reasoning_only_length_response_is_not_retried_or_exposed(monkeypatch):
    body=reply('',model='deepseek-v4-pro')
    body['choices'][0].update(finish_reason='length')
    body['choices'][0]['message']['reasoning_content']='PRIVATE_REASONING_NEVER_LOG'
    body['usage']={'prompt_tokens':500,'completion_tokens':8192,
                   'completion_tokens_details':{'reasoning_tokens':8192}}
    requests=[];events=[]
    def post(*args):
        requests.append(args[2])
        return body
    monkeypatch.setattr(messaging,'_post_json',post)
    with pytest.raises(module.ModelTransportError,match='model_output_truncated'):
        client(2,events.append,model='deepseek-v4-pro').send([],system='fixture',tools=[],max_tokens=8192)
    assert len(requests)==1 and requests[0]['max_tokens']==8192
    assert events[0]['response_metadata']['finish_reason']=='length'
    assert events[0]['response_metadata']['reasoning_tokens']==8192
    assert events[0]['output_tokens']==8192 and events[0]['token_usage_known']
    assert 'PRIVATE_REASONING_NEVER_LOG' not in json.dumps(events)


@pytest.mark.parametrize('calls', [
    [],
    [{'id': 'truncated_partial', 'function': {'name': 'probe', 'arguments': '{"value":'}}],
    [{'id': 'truncated_complete_looking', 'function': {'name': 'probe', 'arguments': '{}'}}],
])
def test_flash_length_response_never_retries_or_returns_tool_calls(monkeypatch, calls):
    body = reply('UNTRUSTED_PARTIAL_TEXT', calls, model='deepseek-flash')
    body['choices'][0]['finish_reason'] = 'length'
    requests, events = [], []
    monkeypatch.setattr(messaging, '_post_json', lambda *args: requests.append(args[2]) or body)
    monkeypatch.setattr(module.time, 'sleep', lambda *_: pytest.fail('Truncation must not sleep/retry'))
    tool = messaging.ToolSpec('probe', 'fixture', {'type': 'object'})
    guarded = client(2, events.append, model='deepseek-flash')
    assert guarded.contract['model'] == 'deepseek-flash'
    with pytest.raises(module.ModelTransportError) as caught:
        guarded.send([], system='fixture', tools=[tool], max_tokens=16384)
    assert caught.value.code == 'model_output_truncated' and not caught.value.retryable
    assert len(requests) == len(events) == guarded.last_attempt_count == 1
    assert requests[0]['max_tokens'] == events[0]['request_max_output_tokens'] == 16384
    assert 'UNTRUSTED_PARTIAL_TEXT' not in json.dumps(events)


@pytest.mark.parametrize('metadata,model,expected', [
    ({}, 'deepseek-flash', 32768),  # Hotfix D9 (production 2026-09-20)
    ({}, 'deepseek-v4-pro', 4096),
    ({}, 'gpt-6-astra', 4096),
    ({'siim_calibration': {}}, 'deepseek-flash', 8192),
    ({'official_calibration': {}}, 'deepseek-flash', 8192),
    ({'siim_calibration': {}}, 'deepseek-v4-pro', 8192),
])
def test_message_output_budget_preserves_calibration_priority(metadata, model, expected):
    from evomind_runtime.runtime import _message_max_output_tokens
    assert _message_max_output_tokens({'metadata': metadata}, SimpleNamespace(model=model)) == expected
    assert _message_max_output_tokens({}, object()) == 4096


@pytest.mark.parametrize('confirmed_cap', [1024, 32768])
def test_confirmed_limits_override_flash_default_without_contract_mutation(monkeypatch, confirmed_cap):
    events, requests = [], []
    guarded = client(2, events.append, model='deepseek-flash')
    before = dict(guarded.contract)
    limits = {'max_output_tokens': confirmed_cap, 'total_request_seconds': 1800,
              'deadline_scope': 'logical_send_including_retries', 'authorization': 'synthetic-fixture'}
    guarded.confirmed_request_limits = dict(limits)
    monkeypatch.setattr(module, 'post_json_with_deadline',
                        lambda *args: requests.append(args[2]) or reply(model='deepseek-flash'))
    assert guarded.send([], system='fixture', tools=[], max_tokens=16384).text == 'ready'
    assert requests[0]['max_tokens'] == events[0]['request_max_output_tokens'] == confirmed_cap
    assert events[0]['request_limits'] == limits
    assert guarded.confirmed_request_limits == limits and guarded.contract == before


def test_flash_tool_receipt_survives_truncation_and_same_run_resume(tmp_path, monkeypatch):
    from evomind_runtime.runtime import AgentRuntime
    from evomind_runtime.message_journal import MessageJournal
    monkeypatch.delenv('EVOMIND_MODEL_ROUTE_CONFIG_PATH', raising=False)
    monkeypatch.setenv('EVOMIND_MODEL_WIRE_PROTOCOL', 'chat_completions')
    monkeypatch.setenv('EVOMIND_AIBUILD_MODE', 'off')
    guarded = client(2, model='deepseek-flash')
    monkeypatch.setattr(messaging, 'AgentMessageClient', lambda: guarded)
    monkeypatch.setattr(module.time, 'sleep', lambda *_: pytest.fail('Truncation must not retry'))
    requests = []
    complete_call = {'id': 'read_once', 'function': {'name': 'file_read',
                     'arguments': json.dumps({'path': str(tmp_path / 'fixture.txt'), 'start_line': 1, 'end_line': 2})}}
    truncated_call = {'id': 'must_not_execute', 'function': {'name': 'file_read',
                      'arguments': complete_call['function']['arguments']}}
    cut = reply('UNTRUSTED_PARTIAL', [truncated_call], model='deepseek-flash')
    cut['choices'][0]['finish_reason'] = 'length'
    replies = iter([reply('', [complete_call], model='deepseek-flash'), cut,
                    reply('Recovered from the persisted receipt.', model='deepseek-flash')])
    def post(*args):
        requests.append(args[2])
        return next(replies)
    monkeypatch.setattr(messaging, '_post_json', post)
    (tmp_path / 'fixture.txt').write_text('fixture evidence\n', encoding='utf-8')
    runtime = AgentRuntime(tmp_path)
    try:
        session = runtime.create_session(objective='Read fixture evidence', permission_level='observe')
        first = runtime.message(session['id'], 'Read fixture.txt and report the evidence.')
        assert first['status'] == 'blocked' and first['error_code'] == 'model_output_truncated'
        receipts = runtime.store.list_tool_calls(session['id'])
        assert len(receipts) == 1 and receipts[0]['status'] == 'completed' and receipts[0]['result']['ok']
        assert receipts[0]['idempotency_key'] == 'read_once'
        contract = runtime.get_session(session['id'])['metadata']['model_execution_contract']
        history_before = MessageJournal(runtime, session['id']).history([])
        assert 'must_not_execute' not in json.dumps(history_before)
        assert 'UNTRUSTED_PARTIAL' not in json.dumps(history_before)
        assert runtime.close()
        runtime = AgentRuntime(tmp_path)
        monkeypatch.setattr(runtime.registry, 'invoke',
                            lambda *_: pytest.fail('Resume must consume the saved receipt, not execute a tool'))
        second = runtime.message(session['id'], 'Continue using the recorded tool result; do not repeat it.')
        assert second['status'] == 'completed'
        assert runtime.store.list_tool_calls(session['id']) == receipts
        assert runtime.get_session(session['id'])['metadata']['model_execution_contract'] == contract
        assert len(requests) == 3 and all(item['max_tokens'] == 32768 for item in requests)
        wire_results = [item for item in requests[-1]['messages'] if item['role'] == 'tool']
        assert len(wire_results) == 1 and wire_results[0]['tool_call_id'] == 'read_once'
        assert 'must_not_execute' not in json.dumps(requests[-1])
    finally:
        assert runtime.close()


def test_response_metadata_uses_only_allowlisted_fields():
    body=reply('PRIVATE_CONTENT')
    body['choices'][0]['finish_reason']='PRIVATE_FINISH'
    body['choices'][0]['message']['reasoning_content']='PRIVATE_REASONING'
    body['choices'][0]['message']['tool_calls']=[{'id':'PRIVATE_ID','function':{'arguments':'PRIVATE_ARGUMENT'}}]
    body['usage']={'prompt_tokens':'PRIVATE_USAGE','completion_tokens':True}
    metadata=module.response_metadata(body,expected_model='gpt-6-astra')
    assert metadata['finish_reason']=='other_or_missing' and metadata['tool_call_count']==1
    assert not metadata['reported_usage_valid'] and 'PRIVATE_' not in json.dumps(metadata)


def test_response_metadata_does_not_reuse_previous_reply_on_network_error(monkeypatch):
    events=[];calls=[]
    def post(*args):
        calls.append(1)
        if len(calls)==1:return reply('')
        raise TimeoutError('PRIVATE_NETWORK_ERROR')
    monkeypatch.setattr(messaging,'_post_json',post)
    monkeypatch.setattr(module.time,'sleep',lambda *_:None)
    with pytest.raises(module.ModelTransportError):
        client(1,events.append).send([],system='fixture',tools=[])
    assert events[0]['response_metadata']['response_object']
    assert not events[1]['response_metadata']['response_object']
    assert not events[1]['token_usage_known'] and events[1]['input_tokens'] is None


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_nontransient_http_errors_fail_once(monkeypatch, status):
    seen = []
    def post(*_):
        seen.append(1)
        raise urllib.error.HTTPError("http://fixture.invalid/?key=DO_NOT_LOG", status, "DO_NOT_LOG", {}, None)
    monkeypatch.setattr(messaging, "_post_json", post)
    with pytest.raises(module.ModelTransportError) as caught:
        client(2).send([], system="s", tools=[])
    assert len(seen) == 1
    assert caught.value.http_status == status
    assert not caught.value.retryable
    assert "DO_NOT_LOG" not in str(caught.value)


@pytest.mark.parametrize("body,code", [
    ({"error": {"type": "invalid_api_key", "message": "DO_NOT_LOG"}}, "provider_error"),
    ({"model": "gpt-6-astra", "choices": []}, "response_choices_invalid"),
    (reply(""), "empty_model_response"),
    ({**reply(), "model": "unrequested-model"}, "model_identity_unconfirmed"),
    ({"model": "gpt-6-astra", "choices": [{"finish_reason": "content_filter", "message": {}}]}, "model_refusal"),
])
def test_error_and_empty_success_envelopes_never_complete(monkeypatch, body, code):
    monkeypatch.setattr(messaging, "_post_json", lambda *_: body)
    with pytest.raises(module.ModelTransportError) as caught:
        client().send([], system="s", tools=[])
    assert caught.value.code == code
    assert "DO_NOT_LOG" not in str(caught.value)


@pytest.mark.parametrize("arguments", ["{not json", "[]", "null", "1"])
def test_invalid_tool_arguments_are_not_coerced_to_empty_object(monkeypatch, arguments):
    tool = messaging.ToolSpec("probe", "fixture", {"type": "object"})
    monkeypatch.setattr(messaging, "_post_json", lambda *_: reply("", [{"id": "c1", "function": {"name": "probe", "arguments": arguments}}]))
    with pytest.raises(module.ModelTransportError, match="tool_arguments_invalid"):
        client().send([], system="s", tools=[tool])


def test_valid_native_tool_call_and_bounded_exhaustion(monkeypatch):
    tool = messaging.ToolSpec("probe", "fixture", {"type": "object"})
    monkeypatch.setattr(messaging, "_post_json", lambda *_: reply("", [{"id": "c1", "function": {"name": "probe", "arguments": "{}"}}]))
    turn = client().send([], system="s", tools=[tool])
    assert turn.tool_calls[0].input == {}
    count = []
    def fail(*_):
        count.append(1)
        raise TimeoutError("DO_NOT_LOG")
    monkeypatch.setattr(messaging, "_post_json", fail)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    with pytest.raises(module.ModelTransportError, match="model_network_error"):
        client(2).send([], system="s", tools=[])
    assert len(count) == 3


def test_other_models_and_injected_clients_are_unchanged():
    injected = object()
    assert module.governed_client(injected) is injected
    base = messaging.AgentMessageClient(transports=[messaging.OpenAITransport(ProviderConfig("openai", "http://fixture.invalid", "unrelated-model", "unused"))])
    assert module.governed_client(base) is base


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-5.6-sol", "deepseek-flash"])
@pytest.mark.parametrize("failure", ["empty", "error", "truncated", "wrong_model", "invalid_arguments", "unoffered"])
def test_both_candidates_share_failure_contract(monkeypatch, model, failure):
    tool = messaging.ToolSpec("probe", "fixture", {"type": "object"})
    body = reply(model=model)
    if failure == "empty": body["choices"][0]["message"]["content"] = ""
    if failure == "error": body = {"error": {"type": "upstream_error"}}
    if failure == "truncated": body["choices"][0]["finish_reason"] = "length"
    if failure == "wrong_model": body["model"] = "unrequested-model"
    if failure in {"invalid_arguments", "unoffered"}:
        body["choices"][0]["message"]["tool_calls"] = [{"id": "one", "function": {
            "name": "not_offered" if failure == "unoffered" else "probe",
            "arguments": "{}" if failure == "unoffered" else "[]"}}]
    monkeypatch.setattr(messaging, "_post_json", lambda *_: body)
    with pytest.raises(module.ModelTransportError):
        client(model=model).send([], system="s", tools=[tool])


def test_selected_route_is_frozen_and_failure_never_falls_back(monkeypatch):
    primary = messaging.OpenAITransport(ProviderConfig("openai", "http://fixture.invalid", "gpt-5.6-sol", "not-logged"))
    secondary = messaging.OpenAITransport(ProviderConfig("openai", "http://other.invalid", "gpt-6-astra", "not-logged"))
    base = messaging.AgentMessageClient(max_retries=2, transports=[primary, secondary])
    guarded = module.governed_client(base)
    primary.config.model = "changed-after-pin"
    seen = []
    def fail(url, _headers, payload, _timeout):
        seen.append((url, payload["model"]))
        raise TimeoutError("private-body")
    monkeypatch.setattr(messaging, "_post_json", fail)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    with pytest.raises(module.ModelTransportError):
        guarded.send([], system="s", tools=[])
    assert len(seen) == 3
    assert {model for _url, model in seen} == {"gpt-5.6-sol"}
    assert all("other.invalid" not in url for url, _model in seen)
    assert "not-logged" not in json.dumps(guarded.contract)


def test_attempt_budget_guard_counts_each_real_retry(monkeypatch):
    count = []
    requests = []
    def reserve():
        count.append(1)
        if len(count) > 2:
            raise RuntimeError("research_budget_exhausted")
    def post(*_):
        requests.append(1)
        raise TimeoutError()
    monkeypatch.setattr(messaging, "_post_json", post)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError, match="research_budget_exhausted"):
        client(2, model="gpt-5.6-sol", before_attempt=reserve).send([], system="s", tools=[])
    assert len(requests) == 2


def test_model_contract_is_persistent_and_drift_is_not_a_migration(tmp_path):
    from evomind_runtime.runtime import AgentRuntime
    runtime = AgentRuntime(tmp_path)
    try:
        session = runtime.create_session(objective="model pin test", metadata={"unrelated": "preserved"})
        contract = client(model="gpt-5.6-sol").contract
        assert runtime.store.bind_model_contract(session["id"], contract) == contract
        assert runtime.store.bind_model_contract(session["id"], contract) == contract
        with pytest.raises(ValueError, match="model_contract_drift"):
            runtime.store.bind_model_contract(session["id"], client().contract)
        assert runtime.get_session(session["id"])["metadata"]["unrelated"] == "preserved"
        legacy = runtime.create_session(objective="legacy model test")
        runtime.store.append_event(legacy["id"], "model.response", {"model": "legacy"})
        with pytest.raises(ValueError, match="legacy_model_contract_requires_review"):
            runtime.store.bind_model_contract(legacy["id"], contract)
    finally:
        runtime.close()


def test_aibuild_has_one_retry_owner_and_counts_transport_attempts(tmp_path, monkeypatch):
    from evomind_runtime.aibuild_engine import AIBuildEngine
    from evomind_runtime.runtime import AgentRuntime
    runtime = AgentRuntime(tmp_path)
    calls = []
    def fail(*_):
        calls.append(1)
        raise TimeoutError()
    monkeypatch.setattr(messaging, "_post_json", fail)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    try:
        session = runtime.create_session(objective="bounded attempts fixture")
        engine = AIBuildEngine(runtime, session, client_factory=lambda: client(2, model="gpt-5.6-sol"))
        with pytest.raises(module.ModelTransportError):
            engine._send(engine._client(), [], "s", [])
        assert len(calls) == 3 and engine.calls == 3
    finally:
        runtime.close()
