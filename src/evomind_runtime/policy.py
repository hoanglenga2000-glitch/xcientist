from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .models import PermissionDecision, PermissionLevel

READ_ONLY_TOOLS = {
    "file_list", "file_search", "file_read", "process_list", "process_poll", "process_log",
    "browser_health", "browser_dom", "desktop_windows", "desktop_screenshot", "skill_list", "mcp_list_tools",
    "research_capabilities", "verified_context", "runtime_health", "attachment_list", "attachment_read",
    "artifact_list", "artifact_preview",
    "report_status",
    "memory_search", "kaggle_list", "hpc_verify", "competition_data_status",
    "super_agent_status", "capability_discover", "ecosystem_refresh", "connector_health",
    "directory_list", "directory_stat", "directory_read", "directory_hash",
    "job_status",
    "model_protocols",
}
WORKSPACE_WRITE_TOOLS = {
    "file_write", "file_patch", "file_copy", "file_move", "shell_exec", "process_start", "process_stdin",
    "artifact_publish", "artifact_bundle", "artifact_import", "managed_model_prepare",
    "literature_search", "citation_audit", "report_generate", "memory_writeback", "evolution_evaluate",
    "kaggle_download", "training_route", "hpc_execute_solution", "competition_data_prepare",
    "directory_mkdir", "directory_write_atomic", "directory_copy", "directory_sync",
    "tool_synthesize", "tool_test", "tool_canary", "tool_promote", "tool_reject",
    "transfer_fetch", "job_execute",
    "managed_tensor_train",
}
HIGH_RISK_TOOLS = {
    "file_delete", "process_cancel", "browser_open", "browser_click", "browser_type",
    "desktop_click", "desktop_type", "mcp_call", "research_invoke",
    "kaggle_submit", "hpc_cancel", "artifact_import",
    "directory_delete",
    "job_cancel",
}
ALWAYS_APPROVAL_TOOLS = {
    "file_delete", "directory_delete", "job_cancel", "tool_reject", "desktop_click", "desktop_type", "kaggle_submit", "hpc_cancel", "artifact_import", "managed_model_prepare",
    "managed_tensor_train",
}
SENSITIVE_PATH_NAMES = {".env", ".ssh", "credentials", "secrets", "windows", "system32"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def argument_fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
    return hashlib.sha256(f"{tool_name}\0{canonical_json(arguments)}".encode("utf-8")).hexdigest()


def _resolve(path: str | Path, workspace_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    return candidate.resolve(strict=False)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _crosses_assistant_run_boundary(path: Path, root: Path) -> bool:
    """Reject a model-copied sibling Run path instead of asking a human to approve it."""
    task_parent = root.parent
    if task_parent.name.casefold() != "assistant_tasks" or not root.name.casefold().startswith("run_"):
        return False
    return _within(path, task_parent) and not _within(path, root)


def _path_arguments(arguments: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for key in ("path", "source", "destination", "cwd", "download_path", "artifact_path", "submission_path", "script_path", "data_dir", "configuration_path"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            yield key, value


class PolicyEngine:
    def evaluate(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        permission_level: str,
        workspace_root: str | Path,
        approved_fingerprint: str = "",
    ) -> PermissionDecision:
        root = Path(workspace_root).resolve(strict=False)
        normalized = dict(arguments)
        paths: dict[str, str] = {}
        outside = False
        sensitive = False
        cross_run_workspace = False
        for key, value in _path_arguments(arguments):
            resolved = _resolve(value, root)
            normalized[key] = str(resolved)
            paths[key] = str(resolved)
            outside = outside or not _within(resolved, root)
            cross_run_workspace = cross_run_workspace or _crosses_assistant_run_boundary(resolved, root)
            sensitive = sensitive or bool({part.lower() for part in resolved.parts} & SENSITIVE_PATH_NAMES)

        level = PermissionLevel(permission_level)
        fingerprint = argument_fingerprint(tool_name, normalized)
        exact_approval = bool(approved_fingerprint) and approved_fingerprint == fingerprint
        scope = {
            "workspace_root": str(root),
            "paths": paths,
            "outside_workspace": outside,
            "cross_run_workspace": cross_run_workspace,
        }

        if cross_run_workspace:
            return PermissionDecision(
                False,
                False,
                "high",
                "path targets a different assistant Run; use a workspace-relative path such as outputs/<name>",
                scope,
                normalized,
            )

        if tool_name in READ_ONLY_TOOLS and not sensitive:
            return PermissionDecision(True, False, "low", "read-only capability", scope, normalized)

        if level is PermissionLevel.OBSERVE:
            return PermissionDecision(False, False, "medium", "observe sessions cannot mutate state", scope, normalized)

        argument_requires_approval = tool_name == "directory_sync" and bool(arguments.get("delete_extraneous"))
        requires = (
            tool_name in ALWAYS_APPROVAL_TOOLS
            or outside
            or sensitive
            or tool_name in HIGH_RISK_TOOLS
            or argument_requires_approval
        )
        if tool_name == "shell_exec" or tool_name == "process_start":
            raw_argv = arguments.get("argv")
            command = " ".join(map(str, raw_argv)).lower() if isinstance(raw_argv, list) else ""
            high_impact_terms = ("remove-item", " del ", "format ", "shutdown", "restart-computer", "winget install", "choco install", "git push", "kaggle competitions submit")
            requires = requires or any(term in f" {command} " for term in high_impact_terms)

        if level is PermissionLevel.FULL_AUTO and tool_name not in ALWAYS_APPROVAL_TOOLS and not sensitive:
            requires = False
        if argument_requires_approval:
            requires = True
        if exact_approval:
            requires = False
        if requires:
            return PermissionDecision(False, True, "high" if (outside or sensitive or tool_name in ALWAYS_APPROVAL_TOOLS) else "medium", "exact approval required", scope, normalized, reversible=tool_name != "file_delete")

        allowed = exact_approval or tool_name in WORKSPACE_WRITE_TOOLS or tool_name in READ_ONLY_TOOLS or level in {PermissionLevel.COMPUTER_APPROVED, PermissionLevel.FULL_AUTO}
        return PermissionDecision(allowed, False, "medium" if tool_name in WORKSPACE_WRITE_TOOLS else "low", "allowed by session policy" if allowed else "permission level does not grant this capability", scope, normalized)
