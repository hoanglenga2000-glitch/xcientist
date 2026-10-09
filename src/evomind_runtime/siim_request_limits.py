"""Explicit SIIM request amendment, independent of frozen compute budgets."""
from __future__ import annotations

import hashlib
from pathlib import Path

POLICY = Path('C:/ProgramData/EvoMind/config/official-calibration/siim-mlebench-20260908.json')
WEB_RUN = 'run_1a6fd23d660245d4b23ee0c1c4d479e0'
CAMPAIGN = 'siim_mlebench_calibration_20260908'


def configure_client(store, session_id, client):
    if not POLICY.is_file():
        return
    from .training_control import read_json, identity
    policy = read_json(POLICY)
    if session_id not in policy.get('cases', {}) and session_id != WEB_RUN:
        return
    session = store.get_session(session_id)
    meta = (session or {}).get('metadata') or {}
    if (policy.get('campaign_id') != CAMPAIGN or policy.get('enabled') is not True
            or identity(meta)[:2] != (policy['tenant_id'], policy['owner_principal_id'])
            or meta.get('managed_hpc_identity') != policy['managed_hpc_identity']):
        raise ValueError('siim_request_limits_identity_mismatch')
    if session_id == WEB_RUN and not store.get_assistant_run(session_id):
        raise ValueError('siim_request_limits_web_run_missing')
    if session_id in policy['cases'] and meta.get('siim_calibration') != {
            'policy_sha256': hashlib.sha256(POLICY.read_bytes()).hexdigest()}:
        raise ValueError('siim_request_limits_policy_mismatch')
    contract = getattr(client, 'contract', {})
    if (contract.get('model') != 'deepseek-v4-pro'
            or contract.get('wire_protocol') != 'chat_completions_v1'
            or contract.get('endpoint_sha256') != hashlib.sha256(b'https://api.pezayo.com/v1').hexdigest()):
        raise ValueError('siim_request_limits_model_mismatch')
    # Preserve the pinned per-attempt timeout and retry contract. The additional
    # 600s cap bounds the complete logical send; it does not extend case clocks.
    client.confirmed_request_limits = {
        'max_output_tokens': 16384, 'total_request_seconds': 600,
        'deadline_scope': 'logical_send_including_retries',
        'authorization': 'user_confirmed_recovery_20260910',
        'policy_sha256': hashlib.sha256(POLICY.read_bytes()).hexdigest(),
    }
