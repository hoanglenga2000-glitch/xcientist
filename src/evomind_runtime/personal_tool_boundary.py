"""Fail closed for authenticated personal sessions until an OS sandbox is proven.

This is a restricted capability set, NOT a claim to sandbox arbitrary code.
Legacy administrative/service sessions retain their existing policy.

t237 C4: read-only research tools are allowed (AUDIT §4 policy a). Writeback,
training_route, and HPC stay refused and must never be offered to personal sessions.
The execution-ledger read is local and is offered so a plan cannot invent a fee.
"""
import json
import os
import re
from pathlib import Path
from .models import PermissionDecision

PERSONAL_FILE_TOOLS = frozenset({
    'file_list', 'file_search', 'file_read', 'file_write', 'file_patch',
    'file_copy', 'file_move', 'file_delete', 'attachment_list', 'attachment_read',
    'artifact_publish',
    'memory_remember',  # t237 C3.4: one user_memory DB row; no filesystem access
})

# t237 C4: read-only research tools — AUDIT §4 policy (a)
PERSONAL_READ_RESEARCH_TOOLS = frozenset({
    'verified_context', 'research_capabilities', 'skill_list',
    # Local ledger and compute-verification label. No connection and no reservation.
    'hpc_execution_budget_status',
})

PERSONAL_ALLOWED_TOOLS = PERSONAL_FILE_TOOLS | PERSONAL_READ_RESEARCH_TOOLS

# Still refused for personal/tenant sessions (never offer; keep behind exact approval elsewhere)
PERSONAL_REFUSED_WRITE_TRAIN_HPC = frozenset({
    'memory_writeback', 'training_route',
    'hpc_execute_solution', 'hpc_calibration_run', 'hpc_cancel',
    'managed_tensor_train', 'managed_model_download',
})


# t238 C7c-4: the one HPC tool a tenant may use, and only while tenant GPU training is on
# (scheduler flag + gpu_pool with tenant_training). It goes through the time-share admission.
PERSONAL_GPU_TRAINING_TOOLS = frozenset({'managed_tensor_train'})


def tenant_training_on() -> bool:
    from .tenant_training import enabled
    return enabled() is not None


def session_tenant(session: dict) -> str:
    scope = (session.get('metadata') or {}).get('http_access_scope') or {}
    return str(scope.get('tenant_id') or '')


def is_personal(tenant_id: str) -> bool:
    return str(tenant_id or '').startswith('tenant_')


def offered(tenant_id: str, tool_name: str) -> bool:
    """constrain() refuses every other name, so never offer or list it to this tenant."""
    if not is_personal(tenant_id):
        return True
    if tool_name in PERSONAL_GPU_TRAINING_TOOLS and tenant_training_on():
        return True
    if tool_name in PERSONAL_REFUSED_WRITE_TRAIN_HPC:
        return False
    return tool_name in PERSONAL_ALLOWED_TOOLS


def constrain(session: dict, tool_name: str, decision: PermissionDecision) -> PermissionDecision:
    tenant_id = session_tenant(session)
    if not is_personal(tenant_id):
        return decision
    reason = ''
    if not offered(tenant_id, tool_name):
        reason = 'Personal execution sandbox is not validated for this capability; no tool was executed.'
    elif decision.scope.get('outside_workspace'):
        reason = 'Personal tools cannot access paths outside this Run workspace, even with approval.'
    elif tool_name in {'file_write', 'file_patch', 'file_copy', 'file_move', 'file_delete'}:
        root = Path(session['workspace_root']).resolve()
        for key, path in (decision.scope.get('paths') or {}).items():
            # Copying FROM inputs is allowed; changing the frozen input is not.
            if tool_name == 'file_copy' and key == 'source':
                continue
            if Path(path).is_relative_to(root / 'inputs'):
                reason = 'Run inputs are frozen; write derived files under work/ or outputs/.'
    if not reason:
        return decision
    return PermissionDecision(False, False, 'high', reason,
        {**decision.scope, 'personal_tool_boundary': True}, decision.normalized_arguments)


_TENANT_ID = re.compile(r'tenant_[a-f0-9]{24}')
_ZERO_COST = re.compile(
    r'(费用\s*(?:为|是|：|:)?\s*0|不(?:消耗|影响|扣减|扣除)配额|无配额影响|配额(?:影响|扣减)为\s*0|零成本|零费用|'
    r'\b(?:fee|cost)\s*(?:is\s*)?0\b|no quota)',
    re.IGNORECASE,
)
_QUOTE_NOTE = '费用和配额扣减没有经过核对，不能当成 0。'


def managed_training_clause(tenant_id: str) -> str:
    """How a personal session reaches the sanctioned trainer. Empty for operator sessions.

    ``managed_tensor_train`` is the only training tool a personal session may be offered,
    and only while tenant GPU training is on. shell_exec and training_route stay refused.
    """
    if not is_personal(tenant_id):
        return ''
    if offered(tenant_id, 'managed_tensor_train'):
        return (
            ' For a tabular baseline this person can approve, call managed_tensor_train. '
            'Write the configuration JSON under work/ with data_path under inputs/, task, and target_column; '
            'leave model_family unset unless the person named one. protocol_id is the sha256 of that inputs file. '
            'Pass configuration_path, configuration_sha256, protocol_id and a solution_id. '
            'The first call creates the approval card. Do not ask the person to reply go instead of that call. '
            'After approval the tool publishes metrics.json, the model and predictions. '
            'Do not use shell_exec, training_route, or a local Python fit.'
        )
    return (
        ' managed_tensor_train is not offered in this session, so a baseline cannot be requested or approved here. '
        'Do not tell the person to reply go to start training, and do not describe a local Python or CPU fit as available.'
    )


def planning_resource_clause(tenant_id: str) -> str:
    """Tell a turn that can read the ledger not to invent a price or a Python runtime."""
    if not offered(tenant_id, 'hpc_execution_budget_status'):
        return ''
    return (
        ' When the user asks about cost, quota, compute access, or running a baseline, call '
        'hpc_execution_budget_status before answering. Report its compute_status and the ledger '
        'seconds it returns. incremental_price_known and zero_cost_confirmed are false: the exact '
        'fee and quota deduction are unknown, so do not say the cost is 0 or that quota is unaffected. '
        'If python_execution_offered is false, this session has no Python execution tool; do not claim '
        'a script can run here. If compute_verified is false, the compute connection has not passed verification.'
    )


def public_compute_status(tenant_id: str, state_root: Path) -> dict:
    """Resources-page label from local binding files. Does not open a connection.

    Matches the page: an active binding counts as verified only when its activation
    request is completed and the receipt file exists. Anything short of that stays
    unverified. The result has no job id, profile, or path.
    """
    unknown = {'compute_status': 'unknown', 'compute_verified': False}
    if not _TENANT_ID.fullmatch(str(tenant_id or '')):
        return unknown
    root = Path(state_root)
    tenant_root = root / 'tenants' / tenant_id
    binding_path = tenant_root / 'hpc-binding.json'
    if binding_path.is_symlink() or not binding_path.is_file():
        return {'compute_status': '未设置计算资源', 'compute_verified': False}
    try:
        binding = json.loads(binding_path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {'compute_status': '未设置计算资源', 'compute_verified': False}
    if not isinstance(binding, dict) or binding.get('tenant_id') != tenant_id or binding.get('schema') != 'evomind.tenant_hpc_binding.v1':
        return {'compute_status': '未设置计算资源', 'compute_verified': False}
    state = str(binding.get('state') or '')
    if state != 'active':
        return {'compute_status': '计算资源暂不可用' if state else '未设置计算资源', 'compute_verified': False}
    generation = binding.get('allocation_generation')
    verified = False
    if isinstance(generation, int) and not isinstance(generation, bool) and generation > 0:
        request_root = tenant_root / 'activation-requests'
        prefix = f'g{generation}.'
        try:
            names = [name for name in os.listdir(request_root) if name.startswith(prefix) and name.endswith('.json')]
        except OSError:
            names = []
        if len(names) == 1:
            request_path = request_root / names[0]
            try:
                request = json.loads(request_path.read_text(encoding='utf-8-sig')) if not request_path.is_symlink() else {}
            except (OSError, ValueError):
                request = {}
            receipt = tenant_root / 'activation-receipts' / f'g{generation}.json'
            owner = str(binding.get('username') or '')
            verified = (
                isinstance(request, dict)
                and request.get('state') == 'completed'
                and request.get('tenant_id') == tenant_id
                and request.get('owner_principal_id') == owner
                and request.get('job_id') == binding.get('job_id')
                and request.get('credential_profile') == binding.get('credential_profile')
                and request.get('allocation_binding_id') == binding.get('allocation_binding_id')
                and request.get('allocation_generation') == generation
                and request.get('profile_instance_id') == binding.get('profile_instance_id')
                and receipt.is_file() and not receipt.is_symlink()
            )
    if verified:
        return {'compute_status': '计算资源已核对', 'compute_verified': True}
    return {'compute_status': '计算资源未通过核对', 'compute_verified': False}


def zero_cost_was_confirmed(calls: list) -> bool:
    """True only when a completed ledger read itself set zero_cost_confirmed."""
    for call in calls or []:
        if not isinstance(call, dict) or call.get('tool_name') != 'hpc_execution_budget_status' or call.get('status') != 'completed':
            continue
        result = call.get('result') if isinstance(call.get('result'), dict) else {}
        content = result.get('content') if isinstance(result.get('content'), dict) else {}
        if content.get('zero_cost_confirmed') is True:
            return True
    return False


def annotate_unverified_zero_cost(answer: str, *, confirmed: bool = False) -> str:
    """Keep an unverified zero-fee sentence from standing as the answer."""
    text = str(answer or '')
    if confirmed or not _ZERO_COST.search(text) or _QUOTE_NOTE in text:
        return text
    return f'{text.rstrip()}\n\n{_QUOTE_NOTE}'
