import json
import threading

import pytest

from evomind_runtime import model_transport as module, responses_transport
from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.store import RuntimeStore
from research_os.agent import messaging
from research_os.llm_client import ProviderConfig


def candidate(key='fixture-key-a', model='gpt-5.6-sol'):
    return module.governed_client(messaging.AgentMessageClient(max_retries=0, transports=[
        messaging.OpenAITransport(ProviderConfig('openai', 'https://fixture.invalid/v1', model, key))]))


def test_same_model_and_endpoint_key_rotation_is_detected_without_storing_key(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        session = runtime.create_session(objective='pin fixture')
        first, second = candidate(), candidate('fixture-key-b')
        assert first.contract != second.contract
        module.bind_run_client(runtime.store, session['id'], first)
        with pytest.raises(ValueError, match='model_contract_drift'):
            module.bind_run_client(runtime.store, session['id'], second)
        captured = json.dumps(runtime.store.get_session(session['id'])) + json.dumps(runtime.store.list_events(session['id']))
        assert 'fixture-key-a' not in captured and 'fixture-key-b' not in captured
    finally:
        runtime.close()


def test_pinned_runtime_rejects_ungoverned_client_before_http_or_tools(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    try:
        session = runtime.create_session(objective='pin fixture')
        module.bind_run_client(runtime.store, session['id'], candidate())
        raw = messaging.AgentMessageClient(transports=[messaging.OpenAITransport(
            ProviderConfig('openai', 'https://fixture.invalid/v1', 'unrelated-model', 'fixture-key-a'))])
        monkeypatch.setattr(messaging, 'AgentMessageClient', lambda **_: raw)
        monkeypatch.setattr(messaging, '_post_json', lambda *_: pytest.fail('must not send a model request'))
        monkeypatch.setattr(runtime, 'invoke_tool', lambda *_a, **_k: pytest.fail('must not dispatch a tool'))
        result = runtime.message(session['id'], 'continue')
        assert result['status'] in {'failed', 'blocked'}
        assert not any(event['event_type'] == 'model.response' for event in runtime.store.list_events(session['id']))
    finally:
        runtime.close()


@pytest.mark.parametrize('as_json', [False, True])
def test_stale_metadata_preserves_pin_and_other_updates(tmp_path, as_json):
    runtime = AgentRuntime(tmp_path)
    try:
        session = runtime.create_session(objective='pin fixture')
        stale = {**session['metadata'], 'scope_update': 'preserved'}
        pinned = candidate().contract
        runtime.store.bind_model_contract(session['id'], pinned)
        updated = runtime.store.update_session(session['id'], metadata_json=json.dumps(stale) if as_json else stale)
        assert updated['metadata']['model_execution_contract'] == pinned
        assert updated['metadata']['scope_update'] == 'preserved'
    finally:
        runtime.close()


@pytest.mark.parametrize('replacement', [None, {}, {'model': 'unrelated-model'}])
def test_explicit_metadata_pin_removal_or_replacement_is_refused(tmp_path, replacement):
    runtime = AgentRuntime(tmp_path)
    try:
        session = runtime.create_session(objective='pin fixture')
        pinned = candidate().contract
        runtime.store.bind_model_contract(session['id'], pinned)
        with pytest.raises(ValueError, match='model_contract_drift'):
            runtime.store.update_session(session['id'], metadata_json={'model_execution_contract': replacement})
        assert runtime.store.get_session(session['id'])['metadata']['model_execution_contract'] == pinned
    finally:
        runtime.close()


def test_aibuild_pending_batch_rejects_missing_pin_before_mutation(tmp_path, monkeypatch):
    from evomind_runtime.aibuild_engine import AIBuildEngine
    runtime = AgentRuntime(tmp_path)
    try:
        session = runtime.create_session(objective='pin fixture')
        module.bind_run_client(runtime.store, session['id'], candidate())
        engine = AIBuildEngine(runtime, session, client_factory=lambda: object())
        pending = {'messages': [], 'remaining_calls': [{'id': 'call_one', 'key': 'one',
            'name': 'file_write', 'input': {'path': 'outputs/x.txt', 'content': 'fixture'}}]}
        before = json.dumps(pending, sort_keys=True)
        monkeypatch.setattr(runtime, 'invoke_tool', lambda *_a, **_k: pytest.fail('must not dispatch pending tool'))
        with pytest.raises(module.ModelTransportError, match='model_contract_missing_on_resume'):
            engine._settle_tool_batch('coder', 'CoderAgent', pending)
        assert json.dumps(pending, sort_keys=True) == before
        assert not (engine.path / 'coder.pending.json').exists()
        assert engine.calls == 0
    finally:
        runtime.close()


@pytest.mark.parametrize('during_request', [False, True])
def test_route_file_change_stops_before_tool_results_are_accepted(tmp_path, monkeypatch, during_request):
    route = tmp_path / 'route.fixture'
    route.write_text('version one', encoding='utf-8')
    monkeypatch.setenv('EVOMIND_MODEL_ROUTE_CONFIG_PATH', str(route))
    monkeypatch.setenv('EVOMIND_MODEL_WIRE_PROTOCOL', 'responses')
    c = candidate()
    attempts = []
    def post(*_):
        attempts.append(1)
        route.write_text('version two', encoding='utf-8')
        return {'model': 'gpt-5.6-sol', 'status': 'completed', 'output': [
            {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'ready'}]}]}
    monkeypatch.setattr(responses_transport, 'post_responses', post)
    if not during_request:
        route.write_text('version two', encoding='utf-8')
    with pytest.raises(module.ModelTransportError, match='model_route_configuration_drift'):
        c.send([], system='fixture', tools=[])
    assert len(attempts) == int(during_request)


def test_two_store_instances_cannot_bind_different_contracts(tmp_path):
    runtime = AgentRuntime(tmp_path)
    peer = None
    try:
        session = runtime.create_session(objective='concurrent pin fixture')
        peer = RuntimeStore(runtime.runtime_root / 'runtime.sqlite3')
        barrier, results = threading.Barrier(2), []
        def bind(store, key):
            c = candidate(key).contract
            barrier.wait(timeout=3)
            try:
                store.bind_model_contract(session['id'], c)
                results.append('bound')
            except ValueError as error:
                results.append(str(error))
        workers = [threading.Thread(target=bind, args=(store, key)) for store, key in [
            (runtime.store, 'fixture-key-a'), (peer, 'fixture-key-b')]]
        for worker in workers: worker.start()
        for worker in workers: worker.join(timeout=5)
        assert all(not worker.is_alive() for worker in workers)
        assert sorted(results) == ['bound', 'model_contract_drift']
    finally:
        if peer is not None:
            peer.close()
        runtime.close()


def test_uncontracted_admission_and_peer_binding_are_mutually_exclusive(tmp_path):
    runtime = AgentRuntime(tmp_path)
    peer = None
    try:
        session = runtime.create_session(objective='admission fixture')
        peer = RuntimeStore(runtime.runtime_root / 'runtime.sqlite3')
        # Reading a stale no-contract snapshot is no longer the enforcement boundary.
        stale = runtime.store.get_session(session['id'])
        peer.bind_model_contract(session['id'], candidate().contract)
        with pytest.raises(module.ModelTransportError, match='model_contract_missing_on_resume'):
            module.bind_run_client(runtime.store, stale['id'], object())
        second = runtime.create_session(objective='reverse admission fixture')
        module.bind_run_client(runtime.store, second['id'], object())
        with pytest.raises(ValueError, match='legacy_model_contract_requires_review'):
            peer.bind_model_contract(second['id'], candidate().contract)
        runtime.store.update_session(second['id'], metadata_json={})
        assert runtime.store.get_session(second['id'])['metadata']['model_uncontracted_execution_started'] is True
    finally:
        if peer is not None: peer.close()
        runtime.close()


def test_missing_route_at_client_construction_is_a_blocked_error(tmp_path, monkeypatch):
    monkeypatch.setenv('EVOMIND_MODEL_ROUTE_CONFIG_PATH', str(tmp_path / 'unavailable.route'))
    with pytest.raises(module.ModelTransportError, match='model_route_configuration_unavailable'):
        candidate()


def test_explicit_application_timeout_is_frozen_and_used_on_the_wire(monkeypatch):
    monkeypatch.setenv('EVOMIND_MODEL_TIMEOUT_SECONDS', '90')
    c = candidate(model='gpt-5.5')
    monkeypatch.setenv('EVOMIND_MODEL_TIMEOUT_SECONDS', '120')
    seen = []
    def post(_url, _headers, _payload, timeout):
        seen.append(timeout)
        return {'model': 'gpt-5.5', 'choices': [{'finish_reason': 'stop', 'message': {'content': 'ready'}}]}
    monkeypatch.setattr(messaging, '_post_json', post)
    assert c.send([], system='fixture', tools=[]).text == 'ready'
    assert c.contract['timeout_seconds'] == 90 and seen == [90]


def test_route_change_in_pre_request_callback_sends_nothing(tmp_path, monkeypatch):
    route = tmp_path / 'callback.route'
    route.write_text('one', encoding='utf-8')
    monkeypatch.setenv('EVOMIND_MODEL_ROUTE_CONFIG_PATH', str(route))
    c = candidate()
    c.before_attempt = lambda: route.write_text('two', encoding='utf-8')
    monkeypatch.setattr(messaging, '_post_json', lambda *_: pytest.fail('configuration changed before HTTP'))
    with pytest.raises(module.ModelTransportError, match='model_route_configuration_drift'):
        c.send([], system='fixture', tools=[])


def test_cancel_pause_update_serializes_with_other_store_pin(tmp_path):
    runtime = AgentRuntime(tmp_path)
    peer = None
    try:
        session = runtime.create_session(objective='cancel pin fixture')
        runtime.store.set_user_pause_requested(session['id'], True)
        peer = RuntimeStore(runtime.runtime_root / 'runtime.sqlite3')
        barrier, errors = threading.Barrier(2), []
        def action(operation):
            try:
                barrier.wait(timeout=3)
                operation()
            except Exception as error:
                errors.append(type(error).__name__)
        pinned = candidate().contract
        workers = [threading.Thread(target=action, args=(operation,)) for operation in [
            lambda: runtime.store.set_user_pause_requested(session['id'], False),
            lambda: peer.bind_model_contract(session['id'], pinned)]]
        for worker in workers: worker.start()
        for worker in workers: worker.join(timeout=5)
        assert not errors and all(not worker.is_alive() for worker in workers)
        metadata = runtime.store.get_session(session['id'])['metadata']
        assert metadata['model_execution_contract'] == pinned
        assert not metadata.get('user_pause_requested')
    finally:
        if peer is not None: peer.close()
        runtime.close()


def test_cancelled_aibuild_batch_never_creates_fake_settlement(tmp_path, monkeypatch):
    from evomind_runtime.aibuild_engine import AIBuildEngine
    runtime = AgentRuntime(tmp_path)
    try:
        session = runtime.create_session(objective='cancel pending fixture')
        engine = AIBuildEngine(runtime, session, client_factory=lambda: object())
        pending = {'messages': [], 'remaining_calls': [{'id': 'one', 'key': 'one', 'name': 'file_write',
            'input': {'path': 'outputs/never.txt', 'content': 'fixture'}}], 'completed_results': [], 'tool_outcomes': []}
        path = engine.path / 'coder.pending.json'
        engine._save_json(path, pending)
        original = path.read_bytes()
        def cancel_then_invoke(*args, **kwargs):
            runtime.store.update_session(session['id'], status='cancelled')
            return actual_invoke(*args, **kwargs)
        actual_invoke = runtime.invoke_tool
        monkeypatch.setattr(runtime, 'invoke_tool', cancel_then_invoke)
        assert engine._settle_tool_batch('coder', 'CoderAgent', pending) is None
        assert engine.pending['status'] == 'cancelled'
        assert path.read_bytes() == original
        assert not list(engine.path.glob('coder.*.settled.json'))
        assert runtime.store.list_tool_calls(session['id']) == []
        assert not (runtime.workspace_root / 'outputs/never.txt').exists()
    finally:
        runtime.close()
