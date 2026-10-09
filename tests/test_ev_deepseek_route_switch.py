"""Offline checks for the user-selected DeepSeek route; no API or GPU calls."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
PAYLOAD=ROOT/'artifacts/ev-domestic-model-route-20260908/switch-payload'


def module():
    spec=importlib.util.spec_from_file_location('evomind_runtime._deepseek_candidate',PAYLOAD/'model_transport.py')
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_selected_model_keeps_strict_transport_validation():
    mod=module()
    assert 'deepseek-v4-pro' in mod.governed_model_names()
    body={'model':'deepseek-v4-pro','choices':[{'message':{'content':'READY'},'finish_reason':'stop'}]}
    mod._response_error(body,expected_model='deepseek-v4-pro',tools=[])
    body['model']='unexpected-alias'
    with pytest.raises(mod.ModelTransportError,match='identity'):
        mod._response_error(body,expected_model='deepseek-v4-pro',tools=[])


def test_route_is_exact_and_not_an_automatic_fallback():
    route=json.loads((PAYLOAD/'route.json').read_text())
    assert route['model']=='deepseek-v4-pro'
    assert route['base_url']=='https://api.pezayo.com/v1'
    assert route['wire_protocol']=='chat_completions' and route['timeout_seconds']==180
    assert route['fallback_to_other_models'] is False
    assert route['old_gpt_runs_not_resumed'] is True
    assert 'api_key' not in route


def test_launcher_changes_are_confined_to_model_profile():
    text=(PAYLOAD/'Start-Node.ps1').read_text()
    assert "$baseEnv['OPENAI_MODEL'] = 'deepseek-v4-pro'" in text
    assert "$baseEnv['EVOMIND_MODEL_WIRE_PROTOCOL'] = 'chat_completions'" in text
    assert "pezayo_ev_deepseek_v4_pro.xml" in text
    assert "$baseEnv['EVOLUTION_PROVIDER_STRICT'] = 'true'" in text
    assert 'Assert-ServiceAllocationBinding' in text
    assert 'Assert-BundleIntegrity' in text


def test_deepseek_contract_admission_and_drift_guards(tmp_path,monkeypatch):
    from evomind_runtime.model_transport import governed_client
    from evomind_runtime.store import RuntimeStore
    from evomind_runtime.models import Session
    from research_os.agent.messaging import AgentMessageClient,OpenAITransport
    from research_os.llm_client import ProviderConfig
    monkeypatch.setenv('EVOMIND_MODEL_WIRE_PROTOCOL','chat_completions')
    monkeypatch.delenv('EVOMIND_MODEL_ROUTE_CONFIG_PATH',raising=False)
    client=governed_client(AgentMessageClient(transports=[OpenAITransport(ProviderConfig('openai','https://fixture.invalid/v1','deepseek-v4-pro','fixture-only'))]))
    store=RuntimeStore(tmp_path/'runtime.sqlite3')
    try:
        store.create_session(Session(id='fixture',workspace_root=str(tmp_path)))
        assert store.bind_model_contract('fixture',client.contract)==client.contract
        assert store.bind_model_contract('fixture',client.contract)==client.contract
        with pytest.raises(ValueError,match='model_contract_invalid'):
            store.bind_model_contract('fixture',{**client.contract,'wire_protocol':'responses_stream_v1'})
        with pytest.raises(ValueError,match='model_contract_drift'):
            store.bind_model_contract('fixture',{**client.contract,'endpoint_sha256':'f'*64})
    finally:store.close()
