"""Platform catalog is the key-gated transport list. Selection reuses that list."""
import os

from evomind_runtime.model_profiles import ModelProfiles
from evomind_runtime.model_router import available_platform_models
from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.tenant_access import Principal
from research_os.agent import messaging
from test_user_tasks_http import task_api


def _clear_model_env(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(('OPENAI_', 'ANTHROPIC_', 'DEEPSEEK_', 'CLAUDE_', 'EVOLUTION_')):
            monkeypatch.delenv(name, raising=False)


def _principal():
    return Principal('tenant_' + 'a' * 24, 'alice')


def _post(bucket):
    def post(url, headers, payload, timeout):
        bucket.append({'url': url, 'model': payload.get('model'), 'effort': payload.get('reasoning_effort')})
        if str(url).rstrip('/').endswith('/messages'):
            return {'model': payload['model'], 'stop_reason': 'end_turn',
                    'content': [{'type': 'text', 'text': '你好'}], 'usage': {'input_tokens': 1, 'output_tokens': 1}}
        return {'model': payload['model'], 'choices': [{'finish_reason': 'stop', 'message': {'content': '你好'}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}
    return post


def test_platform_models_lists_only_providers_with_a_key(tmp_path, monkeypatch):
    _clear_model_env(monkeypatch)
    assert available_platform_models() == []
    assert ModelProfiles(tmp_path).list(_principal())['platform_model'] is None
    assert ModelProfiles(tmp_path).list(_principal())['platform_models'] == []

    monkeypatch.setenv('ANTHROPIC_API_KEY', '')
    monkeypatch.setenv('DEEPSEEK_API_KEY', '')
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-fixture')
    monkeypatch.setenv('OPENAI_MODEL', 'gpt-4o')
    listed = available_platform_models()
    assert listed == [{'id': 'openai:gpt-4o', 'provider': 'openai', 'model': 'gpt-4o', 'label': 'gpt-4o'}]
    body = ModelProfiles(tmp_path).list(_principal())
    assert body['platform_model'] == {'provider': 'openai', 'model': 'gpt-4o'}
    assert body['platform_models'] == listed

    monkeypatch.setenv('ANTHROPIC_API_KEY', 'anthropic-fixture')
    monkeypatch.setenv('CLAUDE_CODE_MODEL', 'claude-opus-4-8')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'deepseek-fixture')
    monkeypatch.setenv('DEEPSEEK_MODEL', 'deepseek-chat')
    monkeypatch.setenv('EVOLUTION_PRIMARY_PROVIDER', 'anthropic')
    monkeypatch.delenv('EVOLUTION_PROVIDER_STRICT', raising=False)
    listed = available_platform_models()
    assert [item['id'] for item in listed] == ['anthropic:claude-opus-4-8', 'deepseek:deepseek-chat', 'openai:gpt-4o']
    assert ModelProfiles(tmp_path).list(_principal())['platform_model'] == {'provider': 'anthropic', 'model': 'claude-opus-4-8'}

    monkeypatch.setenv('EVOLUTION_PROVIDER_STRICT', '1')
    assert [item['id'] for item in available_platform_models()] == ['anthropic:claude-opus-4-8']


def test_model_list_endpoint_returns_the_key_gated_catalog(tmp_path, monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'anthropic-fixture')
    monkeypatch.setenv('CLAUDE_CODE_MODEL', 'claude-opus-4-8')
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-fixture')
    monkeypatch.setenv('OPENAI_MODEL', 'gpt-4o')
    monkeypatch.setenv('EVOLUTION_PRIMARY_PROVIDER', 'anthropic')
    monkeypatch.setenv('EVOLUTION_PROVIDER_STRICT', '1')
    with task_api(tmp_path) as request:
        status, body = request('GET', '/v1/model-profiles')
    assert status == 200, body
    assert body['platform_model'] == {'provider': 'anthropic', 'model': 'claude-opus-4-8'}
    assert body['platform_models'] == [{'id': 'anthropic:claude-opus-4-8', 'provider': 'anthropic', 'model': 'claude-opus-4-8', 'label': 'claude-opus-4-8'}]
    assert 'api_key' not in str(body['platform_models'])


def test_run_create_uses_a_non_default_platform_model_and_rejects_unknown_ids(tmp_path, monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'anthropic-fixture')
    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'http://127.0.0.1:9/anthropic')
    monkeypatch.setenv('CLAUDE_CODE_MODEL', 'claude-opus-4-8')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'deepseek-fixture')
    monkeypatch.setenv('DEEPSEEK_BASE_URL', 'http://127.0.0.1:9/deepseek')
    monkeypatch.setenv('DEEPSEEK_MODEL', 'deepseek-chat')
    monkeypatch.setenv('EVOLUTION_PRIMARY_PROVIDER', 'anthropic')
    monkeypatch.delenv('EVOLUTION_PROVIDER_STRICT', raising=False)
    calls = []
    monkeypatch.setattr(messaging, '_post_json', _post(calls))
    runtime = AgentRuntime(tmp_path)
    try:
        chosen = runtime.assistant.create_run(prompt='你好', platform_model_id='deepseek:deepseek-chat', start=False)
        runtime.assistant._execute(chosen['id'], resume=False)
        observed = [item for item in runtime.assistant.events(chosen['id']) if item['event_type'] == 'model_observed']
        assert calls[-1]['model'] == 'deepseek-chat'
        assert observed[-1]['payload']['provider'] == 'deepseek'
        assert observed[-1]['payload']['model'] == 'deepseek-chat'
        assert runtime.assistant.snapshot(chosen['id'])['model'] == 'deepseek-chat'

        calls.clear()
        default = runtime.assistant.create_run(prompt='你好', start=False)
        runtime.assistant._execute(default['id'], resume=False)
        observed = [item for item in runtime.assistant.events(default['id']) if item['event_type'] == 'model_observed']
        assert calls[-1]['model'] == 'claude-opus-4-8'
        assert observed[-1]['payload']['model'] == 'claude-opus-4-8'
        assert observed[-1]['payload']['provider'] == 'anthropic'
    finally:
        runtime.close()

    with task_api(tmp_path) as request:
        status, body = request('POST', '/v1/runs', {'prompt': '你好', 'platform_model_id': 'openai:gpt-4o'})
        assert status == 400 and body['error'] == 'invalid_platform_model', body
        status, body = request('POST', '/v1/runs', {'prompt': '你好', 'platform_model_id': ''})
        assert status == 400 and body['error'] == 'invalid_platform_model', body


def test_openai_run_effort_is_sent_and_absent_effort_keeps_the_old_payload(tmp_path, monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-fixture')
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://127.0.0.1:9/openai')
    monkeypatch.setenv('OPENAI_MODEL', 'gpt-4o')
    monkeypatch.setenv('EVOLUTION_PRIMARY_PROVIDER', 'openai')
    monkeypatch.setenv('EVOLUTION_PROVIDER_STRICT', '1')
    calls = []
    monkeypatch.setattr(messaging, '_post_json', _post(calls))
    runtime = AgentRuntime(tmp_path)
    try:
        forced = runtime.assistant.create_run(prompt='你好', reasoning_effort='high', start=False)
        runtime.assistant._execute(forced['id'], resume=False)
        assert calls[-1]['model'] == 'gpt-4o'
        assert calls[-1]['effort'] == 'high'
        plain = runtime.assistant.create_run(prompt='你好', start=False)
        runtime.assistant._execute(plain['id'], resume=False)
        assert calls[-1]['effort'] is None
        assert runtime.assistant.snapshot(plain['id'])['model'] == 'gpt-4o'
    finally:
        runtime.close()
