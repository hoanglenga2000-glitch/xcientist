"""Fail closed for authenticated personal sessions until an OS sandbox is proven.

This is a restricted capability set, NOT a claim to sandbox arbitrary code.
Legacy administrative/service sessions retain their existing policy.
"""
from pathlib import Path
from .models import PermissionDecision

PERSONAL_FILE_TOOLS = frozenset({
    'file_list', 'file_search', 'file_read', 'file_write', 'file_patch',
    'file_copy', 'file_move', 'file_delete', 'attachment_list', 'attachment_read',
    'artifact_publish',
})


def constrain(session: dict, tool_name: str, decision: PermissionDecision) -> PermissionDecision:
    scope = (session.get('metadata') or {}).get('http_access_scope') or {}
    if not str(scope.get('tenant_id') or '').startswith('tenant_'):
        return decision
    reason = ''
    if tool_name not in PERSONAL_FILE_TOOLS:
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
