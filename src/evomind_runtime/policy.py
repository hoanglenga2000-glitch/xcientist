from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .models import PermissionDecision, PermissionLevel

READ_ONLY_TOOLS = {
    "file_list", "file_search", "file_read", "process_list", "process_poll", "process_log",
    "browser_health", "browser_dom", "desktop_windows", "desktop_screenshot", "skill_list", "mcp_list_tools",
    "research_capabilities", "runtime_health",
}
WORKSPACE_WRITE_TOOLS = {"file_write", "file_patch", "file_copy", "file_move", "shell_exec", "process_start", "process_stdin"}
HIGH_RISK_TOOLS = {
    "file_delete", "process_cancel", "browser_open", "browser_click", "browser_type",
    "desktop_click", "desktop_type", "mcp_call", "research_invoke",
}
ALWAYS_APPROVAL_TOOLS = {"file_delete", "desktop_click", "desktop_type"}
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


def _path_arguments(arguments: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for key in ("path", "source", "destination", "cwd", "download_path", "artifact_path"):
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
        for key, value in _path_arguments(arguments):
            resolved = _resolve(value, root)
            normalized[key] = str(resolved)
            paths[key] = str(resolved)
            outside = outside or not _within(resolved, root)
            sensitive = sensitive or bool({part.lower() for part in resolved.parts} & SENSITIVE_PATH_NAMES)

        level = PermissionLevel(permission_level)
        fingerprint = argument_fingerprint(tool_name, normalized)
        exact_approval = bool(approved_fingerprint) and approved_fingerprint == fingerprint
        scope = {"workspace_root": str(root), "paths": paths, "outside_workspace": outside}

        if tool_name in READ_ONLY_TOOLS and not sensitive:
            return PermissionDecision(True, False, "low", "read-only capability", scope, normalized)

        if level is PermissionLevel.OBSERVE:
            return PermissionDecision(False, False, "medium", "observe sessions cannot mutate state", scope, normalized)

        requires = tool_name in ALWAYS_APPROVAL_TOOLS or outside or sensitive or tool_name in HIGH_RISK_TOOLS
        if tool_name == "shell_exec" or tool_name == "process_start":
            raw_argv = arguments.get("argv")
            command = " ".join(map(str, raw_argv)).lower() if isinstance(raw_argv, list) else ""
            high_impact_terms = ("remove-item", " del ", "format ", "shutdown", "restart-computer", "winget install", "choco install", "git push", "kaggle competitions submit")
            requires = requires or any(term in f" {command} " for term in high_impact_terms)

        if level is PermissionLevel.FULL_AUTO and tool_name not in ALWAYS_APPROVAL_TOOLS and not sensitive:
            requires = False
        if exact_approval:
            requires = False
        if requires:
            return PermissionDecision(False, True, "high" if (outside or sensitive or tool_name in ALWAYS_APPROVAL_TOOLS) else "medium", "exact approval required", scope, normalized, reversible=tool_name != "file_delete")

        allowed = exact_approval or tool_name in WORKSPACE_WRITE_TOOLS or tool_name in READ_ONLY_TOOLS or level in {PermissionLevel.COMPUTER_APPROVED, PermissionLevel.FULL_AUTO}
        return PermissionDecision(allowed, False, "medium" if tool_name in WORKSPACE_WRITE_TOOLS else "low", "allowed by session policy" if allowed else "permission level does not grant this capability", scope, normalized)
