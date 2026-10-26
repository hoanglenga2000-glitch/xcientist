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


def _run_openai(tmp_path, monkeypatch, model, *, effort=None, env_effort=None):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-fixture')
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://127.0.0.1:9/openai')
    monkeypatch.setenv('OPENAI_MODEL', model)
    monkeypatch.setenv('EVOLUTION_PRIMARY_PROVIDER', 'openai')
    monkeypatch.setenv('EVOLUTION_PROVIDER_STRICT', '1')
    if env_effort:
        monkeypatch.setenv('OPENAI_REASONING_EFFORT', env_effort)
    calls = []
    monkeypatch.setattr(messaging, '_post_json', _post(calls))
    runtime = AgentRuntime(tmp_path)
    try:
        kwargs = {'reasoning_effort': effort} if effort else {}
        created = runtime.assistant.create_run(prompt='你好', start=False, **kwargs)
        runtime.assistant._execute(created['id'], resume=False)
        return calls[-1]
    finally:
        runtime.close()


def test_non_reasoning_openai_omits_user_effort_and_keeps_env(tmp_path, monkeypatch):
    sent = _run_openai(tmp_path, monkeypatch, 'gpt-4o', effort='high')
    assert sent['model'] == 'gpt-4o'
    assert sent['effort'] is None
    sent = _run_openai(tmp_path, monkeypatch, 'gpt-4.1', effort='high', env_effort='low')
    assert sent['model'] == 'gpt-4.1'
    assert sent['effort'] == 'low'
    sent = _run_openai(tmp_path, monkeypatch, 'gpt-4o', env_effort='low')
    assert sent['effort'] == 'low'


def test_reasoning_openai_sends_user_effort_over_env(tmp_path, monkeypatch):
    sent = _run_openai(tmp_path, monkeypatch, 'gpt-5', effort='high', env_effort='low')
    assert sent['model'] == 'gpt-5'
    assert sent['effort'] == 'high'
    sent = _run_openai(tmp_path, monkeypatch, 'o3-mini', effort='medium')
    assert sent['model'] == 'o3-mini'
    assert sent['effort'] == 'medium'
    sent = _run_openai(tmp_path, monkeypatch, 'o4-mini')
    assert sent['effort'] is None


def test_failover_chain_stamps_only_a_reasoning_openai_transport(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'anthropic-fixture')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'deepseek-fixture')
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-fixture')
    monkeypatch.setenv('OPENAI_MODEL', 'gpt-4o')
    monkeypatch.setenv('OPENAI_REASONING_EFFORT', 'low')
    monkeypatch.delenv('EVOLUTION_PROVIDER_STRICT', raising=False)
    from evomind_runtime.model_router import apply_reasoning_effort

    stamped = [apply_reasoning_effort(item, 'high') for item in messaging._resolve_transports()]
    configs = {item.name: item.config for item in stamped}
    assert configs['openai'].model == 'gpt-4o'
    assert configs['openai'].reasoning_effort == 'low'
    assert configs['anthropic'].reasoning_effort is None
    assert configs['deepseek'].reasoning_effort is None

    monkeypatch.setenv('OPENAI_MODEL', 'gpt-5.6-sol')
    stamped = [apply_reasoning_effort(item, 'high') for item in messaging._resolve_transports()]
    openai = next(item.config for item in stamped if item.name == 'openai')
    assert openai.reasoning_effort == 'high'
    assert all(item.config.reasoning_effort is None for item in stamped if item.name != 'openai')


def test_personal_openai_effort_follows_the_same_model_gate():
    from research_os.llm_client import ProviderConfig
    from research_os.agent.messaging import OpenAITransport
    from evomind_runtime.personal_model_client import PersonalModelClient

    class _Secrets:
        def load(self, *args):
            return 'secret'

    def client(model, effort):
        class _Store:
            secrets = _Secrets()

            def get(self, principal, profile_id, version=None):
                return {'enabled': True, 'provider': 'openai', 'base_url': 'http://127.0.0.1:9', 'model': model}

        return PersonalModelClient(_Store(), _principal(), {'id': 'p', 'version': 1}, reasoning_effort=effort)

    assert client('gpt-4o', 'high').transport.config.reasoning_effort is None
    kept = OpenAITransport(ProviderConfig('openai', 'http://127.0.0.1:9', 'gpt-4o', 'secret', reasoning_effort='low'))
    from evomind_runtime.model_router import apply_reasoning_effort
    apply_reasoning_effort(kept, 'high')
    assert kept.config.reasoning_effort == 'low'
    assert client('o1', 'low').transport.config.reasoning_effort == 'low'
    assert client('deepseek-chat', 'high').transport.config.reasoning_effort is None
