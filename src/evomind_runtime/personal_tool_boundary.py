"""Fail closed for authenticated personal sessions until an OS sandbox is proven.

This is a restricted capability set, NOT a claim to sandbox arbitrary code.
Legacy administrative/service sessions retain their existing policy.

t237 C4: read-only research tools are allowed (AUDIT §4 policy a). Writeback,
training_route, and HPC stay refused and must never be offered to personal sessions.
"""
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
