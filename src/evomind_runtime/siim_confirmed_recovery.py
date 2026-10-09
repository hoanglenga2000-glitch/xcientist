"""Administrator-owned, user-confirmed request and remaining-work amendment."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

CONFIG = Path('C:/ProgramData/EvoMind/config/official-calibration/siim-confirmed-recovery-20260909.json')
POLICY = Path('C:/ProgramData/EvoMind/config/official-calibration/siim-mlebench-20260908.json')
PARENT = 'run_3eb4898a120847c19784eb9cdc9dd4a2'


def load_recovery(budget_path=None):
    if not CONFIG.exists():
        return None
    from .training_control import read_json
    value = read_json(CONFIG)
    if budget_path is not None and Path(value.get('budget_path', '')).resolve() != Path(budget_path).resolve():
        return None
    policy = read_json(POLICY)
    if (value.get('schema') != 'evomind.siim_confirmed_recovery.v1'
            or value.get('authorization') != 'explicit_user_confirmation_20260909'
            or value.get('amendment_id') != 'confirmed-16384-600-20260909'
            or value.get('parent_run_id') != PARENT
            or value.get('policy_sha256') != hashlib.sha256(POLICY.read_bytes()).hexdigest()
            or value.get('max_output_tokens') != 16384 or value.get('total_request_seconds') != 600
            or value.get('global_gpu_seconds') != 86400 or value.get('headroom_factor') != 1.15
            or value.get('step_ceiling_seconds') != 32400
            or not isinstance(value.get('work_units'), dict)
            or set(value['work_units']) | set(value.get('completed_runs', [])) != set(policy['cases'])
            or set(value['work_units']) & set(value.get('completed_runs', []))
            or any(units != 6 for units in value['work_units'].values())
            or value.get('managed_hpc_identity') != policy['managed_hpc_identity']
            or value.get('tenant_id') != policy['tenant_id'] or value.get('owner_principal_id') != policy['owner_principal_id']
            or Path(value['budget_path']).resolve() != Path(policy['budget_path']).resolve()):
        raise ValueError('confirmed_recovery_binding_invalid')
    return {**value, 'amendment_sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest()}


def request_limits(store, session_id):
    recovery = load_recovery()
    if not recovery or session_id not in set(recovery['work_units']) | {recovery['parent_run_id']}:
        return None
    session = store.get_session(session_id)
    from .training_control import identity
    if (not session or identity(session.get('metadata') or {})[:2] != (recovery['tenant_id'], recovery['owner_principal_id'])
            or (session.get('metadata') or {}).get('managed_hpc_identity') != recovery['managed_hpc_identity']):
        raise ValueError('confirmed_recovery_session_identity_mismatch')
    return recovery


def configure_client(store, session_id, client):
    recovery = request_limits(store, session_id)
    if recovery is None:
        return
    if (not hasattr(client, 'contract') or client.contract.get('model') != 'deepseek-v4-pro'
            or client.contract.get('wire_protocol') != 'chat_completions_v1'
            or client.contract.get('endpoint_sha256') != hashlib.sha256(b'https://api.pezayo.com/v1').hexdigest()):
        raise ValueError('confirmed_recovery_model_identity_mismatch')
    client.timeout = 600
    client.contract = {**client.contract, 'timeout_seconds': 600}
    client.confirmed_request_limits = {
        'max_output_tokens': 16384, 'total_request_seconds': 600,
        'deadline_scope': 'logical_send_including_retries', 'amendment_sha256': recovery['amendment_sha256'],
    }
    if session_id in recovery['work_units']:
        def remaining():
            from .siim_calibration_control import CaseBudget
            from .training_control import read_json
            policy = read_json(POLICY)
            return CaseBudget(policy['budget_path'], policy['ev_budget_path'], policy['resource_limits']).remaining_case_seconds(session_id)
        client.confirmed_time_budget = remaining


def limits_receipt(recovery):
    return {'schema': 'evomind.siim_request_amendment.v1', 'amendment_id': recovery['amendment_id'],
            'amendment_sha256': recovery['amendment_sha256'], 'max_output_tokens': 16384,
            'total_request_seconds': 600, 'deadline_scope': 'logical_send_including_retries',
            'authorization': recovery['authorization']}
