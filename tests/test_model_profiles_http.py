"""BYOK public HTTP contracts, using temporary credentials only; no provider calls."""
import json
import pytest
from test_user_tasks_http import task_api, no_external_model_credentials
import threading
import time


@pytest.mark.parametrize('provider', ['anthropic', 'deepseek', 'openai'])
def test_three_supported_protocols_have_explicit_bounded_manual_test(tmp_path, monkeypatch, provider):
    from evomind_runtime import personal_model_client
    calls = []
    def external_service(url, headers, payload):
        calls.append((url, headers, payload))
        assert payload['max_tokens'] == 16
        if provider == 'anthropic':
            assert headers['x-api-key'] == 'test-key-which-must-not-be-reflected'
            return {'model': 'fixture-chat', 'content': [{'type': 'text', 'text': 'OK'}], 'stop_reason': 'end_turn'}
        return {'model': 'fixture-chat', 'choices': [{'finish_reason': 'stop', 'message': {'content': 'OK'}}]}
    monkeypatch.setattr(personal_model_client, 'post_public_json', external_service)
    with task_api(tmp_path) as request:
        profile = request('POST', '/v1/model-profiles', {'name': provider, 'provider': provider,
            'base_url': 'https://api.example.com', 'model': 'fixture-chat', 'api_key': 'test-key-which-must-not-be-reflected'})[1]['profile']
        result = request('POST', '/v1/model-profiles/' + profile['id'] + '/test',
            {'version': 1, 'confirm_cost': True, 'idempotency_key': 'check-protocol-once'})[1]
        assert result['status'] == 'passed' and len(calls) == 1
        assert result['profile']['verified_at'] and not result['profile']['used_at']
        assert request('GET', '/v1/runs')[1]['runs'] == []


def test_failed_test_has_no_fallback_or_retry_and_hourly_budget_is_shared(tmp_path, monkeypatch):
    from evomind_runtime import personal_model_client
    calls = []
    def unavailable(*args):
        calls.append(1)
        raise ValueError('model_total_timeout')
    monkeypatch.setattr(personal_model_client, 'post_public_json', unavailable)
    with task_api(tmp_path) as request:
        profile = request('POST', '/v1/model-profiles', {'name': '超时配置', 'provider': 'openai',
            'base_url': 'https://api.example.com', 'model': 'fixture-chat', 'api_key': 'fixture-timeout-key'})[1]['profile']
        path = '/v1/model-profiles/' + profile['id'] + '/test'
        for attempt in range(3):
            body = {'version': 1, 'confirm_cost': True, 'idempotency_key': 'attempt-key-' + str(attempt)}
            assert request('POST', path, body)[1]['status'] == 'model_total_timeout'
            assert request('POST', path, body)[1]['replayed']
        assert len(calls) == 3
        assert request('POST', path, {**body, 'idempotency_key': 'fourth-attempt'})[0] == 429
        assert request('GET', '/v1/runs')[1]['runs'] == []


def test_provider_cannot_reflect_credentials_into_run_or_artifact_records(tmp_path, monkeypatch):
    from evomind_runtime import personal_model_client
    secret = 'reflected-public-fixture-secret'

    def reflecting_service(url, headers, payload):
        return {'model': payload['model'], 'choices': [{'finish_reason': 'stop', 'message': {'content': 'Server says ' + secret}}]}

    monkeypatch.setattr(personal_model_client, 'post_public_json', reflecting_service)
    with task_api(tmp_path) as request:
        profile = request('POST', '/v1/model-profiles', {'name': '回显防护', 'provider': 'openai',
            'base_url': 'https://api.example.com', 'model': 'fixture-chat', 'api_key': secret})[1]['profile']
        status, run = request('POST', '/v1/runs', {'prompt': 'Say hello.', 'model_profile_id': profile['id'],
            'model_profile_version': 1, 'idempotency_key': 'reflected-key-run'})
        assert status == 201
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            snapshot = request('GET', '/v1/runs/' + run['id'])[1]
            if snapshot['status'] in {'completed', 'failed', 'blocked'}:
                break
            threading.Event().wait(.03)
        assert snapshot['status'] == 'completed' and '[REDACTED]' in snapshot['answer']
        assert secret not in json.dumps(snapshot)
    for file in tmp_path.rglob('*'):
        if file.is_file():
            assert secret.encode() not in file.read_bytes(), file.name


def test_manual_test_and_run_use_exact_profile_revision_without_fallback(tmp_path, monkeypatch):
    from evomind_runtime import personal_model_client
    calls, started, release = [], threading.Event(), threading.Event()

    def provider(url, headers, payload):
        calls.append({'url': url, 'key': headers.get('Authorization'), 'model': payload['model'], 'limit': payload['max_tokens']})
        if payload['max_tokens'] > 16:
            started.set()
            assert release.wait(8)
        return {'model': payload['model'], 'choices': [{'finish_reason': 'stop', 'message': {'content': 'OK'}}]}

    monkeypatch.setattr(personal_model_client, 'post_public_json', provider)  # Only external provider seam.
    body = {'name': '个人模型', 'provider': 'deepseek', 'base_url': 'https://api.example.com/v1',
            'model': 'fixture-model-one', 'api_key': 'fixture-key-version-one'}
    with task_api(tmp_path) as request:
        p = request('POST', '/v1/model-profiles', body)[1]['profile']
        assert calls == []
        path = '/v1/model-profiles/' + p['id']
        check = {'version': 1, 'confirm_cost': True, 'idempotency_key': 'one-manual-check'}
        assert request('POST', path + '/test', check)[1]['status'] == 'passed'
        assert request('POST', path + '/test', check)[1]['replayed'] is True
        assert len(calls) == 1 and calls[0]['limit'] == 16
        task = request('POST', '/v1/user-tasks', {'title': '隔离模型测试', 'idempotency_key': 'personal-model-task'})[1]['task']
        run_body = {'prompt': 'Say hello.', 'user_task_id': task['id'], 'model_profile_id': p['id'],
                    'model_profile_version': 1, 'idempotency_key': 'personal-model-run'}
        assert request('POST', '/v1/runs', run_body, owner='bob')[0] == 404
        status, run = request('POST', '/v1/runs', run_body)
        assert status == 201, run
        assert run['model_profile'] == {'id': p['id'], 'version': 1, 'provider': 'deepseek', 'model': 'fixture-model-one'}
        assert started.wait(5)
        assert request('POST', '/v1/runs', {**run_body, 'idempotency_key': 'second-concurrent-run'})[0] == 409
        update = request('POST', path, {**body, 'model': 'fixture-model-two', 'api_key': 'fixture-key-version-two', 'version': 1})
        assert update[0] == 200
        release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            snapshot = request('GET', '/v1/runs/' + run['id'])[1]
            if snapshot['status'] in {'completed', 'failed', 'blocked'}:
                break
            threading.Event().wait(.03)
        assert snapshot['status'] == 'completed', snapshot
        assert snapshot['model'] == 'fixture-model-one'
        assert len(calls) == 2 and calls[1]['key'] == 'Bearer fixture-key-version-one'
        assert request('POST', '/v1/runs', run_body)[1]['id'] == run['id']
        assert request('POST', '/v1/runs', {**run_body, 'idempotency_key': 'outdated-model-run'})[0] == 409


def test_model_config_is_owned_encrypted_versioned_and_survives_restart(tmp_path):
    secret = 'isolated-fixture-key-do-not-use'
    body = {'name': '我的模型', 'provider': 'openai', 'base_url': 'https://api.example.com/v1',
            'model': 'fixture-chat', 'api_key': secret}
    with task_api(tmp_path) as request:
        status, payload = request('POST', '/v1/model-profiles', body)
        assert status == 201, payload
        profile = payload['profile']
        assert profile['version'] == 1 and profile['verification'] == 'not_tested'
        assert secret not in json.dumps(payload)
        path = '/v1/model-profiles/' + profile['id']
        assert request('GET', path, owner='bob')[0] == 404
        assert request('GET', '/v1/model-profiles', tenant='b')[1]['profiles'] == []
        assert request('GET', '/v1/runs')[1]['runs'] == []
        assert request('POST', path + '/default', {'version': 1}, owner='bob')[0] == 404
        assert request('POST', path + '/default', {'version': 1})[0] == 200
        updated = request('POST', path, {**body, 'api_key': '', 'model': 'fixture-v2', 'version': 1})
        assert updated[0] == 200, updated
        assert updated[1]['profile']['version'] == 2
        assert request('POST', path, {**body, 'version': 1})[0] == 409
        assert request('POST', path + '/test', {'version': 2})[0] == 400
        # Invalid addresses are rejected before DNS or any network call.
        for address in ['http://api.example.com', 'https://127.0.0.1', 'https://[::1]',
                        'https://169.254.169.254', 'https://10.0.0.1', 'https://localhost',
                        'https://user:pass@example.com', 'https://api.example.com?q=secret',
                        'https://api.example.com#secret', 'https://api.example.com:8080']:
            assert request('POST', '/v1/model-profiles', {**body, 'base_url': address})[0] == 400, address
        assert request('POST', path + '/disable', {'version': 2})[0] == 200
        assert request('POST', path + '/default', {'version': 2})[0] == 409
    with task_api(tmp_path) as request:
        result = request('GET', '/v1/model-profiles')[1]
        assert len(result['profiles']) == 1 and result['profiles'][0]['enabled'] is False
        assert result['default_profile_id'] == ''
        assert secret not in json.dumps(result)
    for file in tmp_path.rglob('*'):
        if file.is_file():
            assert secret.encode() not in file.read_bytes(), file.name


def test_personal_selection_also_reaches_research_orchestrator_without_platform_fallback(tmp_path, monkeypatch):
    from evomind_runtime import personal_model_client
    seen = threading.Event()
    monkeypatch.setenv('EVOMIND_AIBUILD_MODE', 'enabled')
    monkeypatch.setenv('EVOMIND_AIBUILD_TENANTS', 'tenant_' + 'a' * 24)

    def unavailable_provider(url, headers, payload):
        assert headers['Authorization'] == 'Bearer research-fixture-only'
        seen.set()
        raise ValueError('model_fixture_unavailable')

    monkeypatch.setattr(personal_model_client, 'post_public_json', unavailable_provider)
    with task_api(tmp_path) as request:
        profile = request('POST', '/v1/model-profiles', {'name': '研究配置', 'provider': 'openai',
            'base_url': 'https://api.example.com', 'model': 'research-fixture', 'api_key': 'research-fixture-only'})[1]['profile']
        status, run = request('POST', '/v1/runs', {'prompt': 'Train a model. Do not execute training or tools in this fixture.',
            'model_profile_id': profile['id'], 'model_profile_version': 1, 'idempotency_key': 'research-fixture-run'})
        assert status == 201
        assert seen.wait(4), 'Research orchestration ignored the personal model selection'
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            snapshot = request('GET', '/v1/runs/' + run['id'])[1]
            if snapshot['status'] in {'blocked', 'failed'}:
                break
            threading.Event().wait(.03)
        assert snapshot['status'] in {'blocked', 'failed'}
        # Existing research orchestration publishes diagnostic receipts on
        # failure. Preserve them, but never call them a trained model/result.
        assert all(file['name'] in {'task_graph.json', 'claim-audit.json', 'search-graph.json',
            'search-selection.json', 'capability-contract.json'} for file in snapshot['artifacts'])
        assert all(file['source_tool_call'] == 'research_coordinator' for file in snapshot['artifacts'])
