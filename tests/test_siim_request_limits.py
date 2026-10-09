import hashlib
import json
from types import SimpleNamespace

import pytest
from evomind_runtime import siim_request_limits as limits


def fixture(tmp_path, monkeypatch):
    path = tmp_path/'policy.json'
    policy = dict(campaign_id=limits.CAMPAIGN, enabled=True, cases={'case': {}},
                  tenant_id='tenant', owner_principal_id='owner', managed_hpc_identity={'job': 1})
    path.write_text(json.dumps(policy))
    monkeypatch.setattr(limits, 'POLICY', path)
    meta = dict(tenant_id='tenant', owner_principal_id='owner', managed_hpc_identity={'job': 1},
                siim_calibration={'policy_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    store = SimpleNamespace(get_session=lambda _: {'metadata':meta}, get_assistant_run=lambda _: {'id':limits.WEB_RUN})
    client = SimpleNamespace(contract=dict(model='deepseek-v4-pro', wire_protocol='chat_completions_v1',
        endpoint_sha256=hashlib.sha256(b'https://api.pezayo.com/v1').hexdigest()), timeout=180)
    return store, client, meta


@pytest.mark.parametrize('run', ['case', limits.WEB_RUN])
def test_scoped_limits_preserve_pinned_contract(tmp_path, monkeypatch, run):
    store, client, _ = fixture(tmp_path, monkeypatch)
    before = dict(client.contract)
    limits.configure_client(store, run, client)
    assert client.confirmed_request_limits['max_output_tokens']==16384
    assert client.confirmed_request_limits['total_request_seconds']==600
    assert client.contract==before and client.timeout==180


def test_unrelated_run_unchanged(tmp_path, monkeypatch):
    store, client, _ = fixture(tmp_path, monkeypatch)
    limits.configure_client(store, 'unrelated', client)
    assert not hasattr(client,'confirmed_request_limits')


def test_owner_drift_rejected(tmp_path, monkeypatch):
    store, client, meta = fixture(tmp_path, monkeypatch)
    meta['owner_principal_id']='other'
    with pytest.raises(ValueError,match='identity_mismatch'):
        limits.configure_client(store, 'case', client)
