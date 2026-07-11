"""Bounded, auditable workspace loop for EvoMind code tasks.

The decision source may be a local callback or a provider-neutral message
client.  It chooses one structured action at a time; this module owns all
filesystem and process execution.  Source changes are applied only in a
detached Git worktree, acceptance commands are allowlisted, and the final
candidate remains behind a human merge gate.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from research_os.agent.messaging import AgentMessageClient, ToolResult, ToolSpec

_ACTIONS = {"search", "read", "patch", "test", "diff", "finish"}
_MODEL_TOOL_TO_ACTION = {f"workspace_{name}": name for name in _ACTIONS}
_FORBIDDEN_PATH_PARTS = {
    ".git",
    ".xsci",
    ".env",
    ".next",
    "__pycache__",
    "credentials",
    "node_modules",
    "reports",
    "secrets",
    "workspace",
}
_EXPLICIT_EDIT_SCOPE_PARTS = {"data"}
_SENSITIVE_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|cookie|password|passwd|private[_-]?key|secret|token)\s*[:=]\s*\S+"
)
_SAFE_COMMAND_PATTERNS = (
    re.compile(r"^python(?:\.exe)?\s+-m\s+(?:pytest|unittest|py_compile|compileall)\b", re.I),
    re.compile(r"^pytest(?:\.exe)?\b", re.I),
    re.compile(r"^(?:npm(?:\.cmd)?|pnpm(?:\.cmd)?|yarn(?:\.cmd)?)\s+(?:run\s+)?(?:test|build|typecheck|lint|check)\b", re.I),
    re.compile(r"^git\s+diff\s+--check\s*$", re.I),
    re.compile(r"^dotnet\s+test\b", re.I),
    re.compile(r"^cargo\s+(?:test|check)\b", re.I),
    re.compile(r"^go\s+test\b", re.I),
)
_CANONICAL_CLAIMS = {
    "workspace_searched",
    "workspace_file_read",
    "candidate_patch_applied",
    "acceptance_commands_passed",
    "final_diff_captured",
    "main_worktree_unchanged",
    "review_candidate_ready",
}
_MODEL_AUDIT_PATH_KEYS = {"artifact_path", "candidate_diff_path", "log_path", "patch_path"}


@dataclass(frozen=True)
class WorkspaceAgentLimits:
    """Hard budgets for one workspace-agent run."""

    max_steps: int = 18
    max_patch_attempts: int = 3
    max_test_runs: int = 8
    max_search_results: int = 80
    max_read_bytes: int = 24_000
    max_patch_bytes: int = 200_000
    max_diff_bytes: int = 400_000
    command_timeout_seconds: int = 120
    total_timeout_seconds: int = 600
    model_max_tokens: int = 1400

    def bounded(self) -> "WorkspaceAgentLimits":
        return WorkspaceAgentLimits(
            max_steps=max(1, min(int(self.max_steps), 40)),
            max_patch_attempts=max(1, min(int(self.max_patch_attempts), 8)),
            max_test_runs=max(1, min(int(self.max_test_runs), 20)),
            max_search_results=max(1, min(int(self.max_search_results), 300)),
            max_read_bytes=max(1000, min(int(self.max_read_bytes), 200_000)),
            max_patch_bytes=max(1000, min(int(self.max_patch_bytes), 2_000_000)),
            max_diff_bytes=max(1000, min(int(self.max_diff_bytes), 4_000_000)),
            command_timeout_seconds=max(1, min(int(self.command_timeout_seconds), 900)),
            total_timeout_seconds=max(2, min(int(self.total_timeout_seconds), 3600)),
            model_max_tokens=max(200, min(int(self.model_max_tokens), 4000)),
        )


class WorkspacePlanner(Protocol):
    """A decision runner that chooses the next structured workspace action."""

    def __call__(self, context: dict[str, Any]) -> dict[str, Any]: ...


def _safe_text(value: Any, *, limit: int = 12_000) -> str:
    text = str(value or "").replace("\x00", " ")
    return _SENSITIVE_RE.sub(r"\1=[redacted]", text)[:limit]


def _safe_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 10:
        return "[nested]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _safe_text(value, limit=16_000)
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item, depth=depth + 1) for item in value[:80]]
    if isinstance(value, dict):
        return {
            str(key)[:120]: _safe_json_value(item, depth=depth + 1)
            for key, item in list(value.items())[:120]
        }
    return _safe_text(value, limit=2000)


def _model_observation(value: Any, *, depth: int = 0) -> Any:
    """Hide audit-only host paths from the model while retaining them in the manifest."""

    if depth >= 10:
        return "[nested]"
    if isinstance(value, dict):
        return {
            str(key)[:120]: (
                "[audit artifact recorded]"
                if str(key) in _MODEL_AUDIT_PATH_KEYS
                else _model_observation(item, depth=depth + 1)
            )
            for key, item in list(value.items())[:120]
        }
    if isinstance(value, (list, tuple)):
        return [_model_observation(item, depth=depth + 1) for item in value[:80]]
    return _safe_json_value(value, depth=depth)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_safe_json_value(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout: int,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(args),
            cwd=str(cwd),
            env=env,
            text=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        raise subprocess.TimeoutExpired(exc.cmd, exc.timeout, output=output, stderr=None) from None
    output = completed.stdout
    decoded = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output or "")
    return subprocess.CompletedProcess(completed.args, completed.returncode, stdout=decoded, stderr=None)


def _git(root: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE", "GIT_PREFIX"):
        env.pop(key, None)
    return _run(["git", *args], cwd=root, timeout=timeout, env=env)


def _status_digest(root: Path) -> tuple[str, bool]:
    result = _git_status(root, timeout=60)
    raw = result.stdout if result.returncode == 0 else f"git-status-error:{result.returncode}:{result.stdout}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest(), bool(raw)


def _git_status(root: Path, *, timeout: int) -> subprocess.CompletedProcess[str]:
    return _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", timeout=timeout)


def _normalize_repo_path(value: str) -> str:
    path = str(value or "").strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path.rstrip("/")


def _path_reason(path: str) -> str:
    normalized = _normalize_repo_path(path)
    if not normalized or normalized == "/dev/null":
        return "empty_path"
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        return "absolute_path"
    parts = [part.lower() for part in normalized.split("/") if part]
    if any(part in {".", ".."} for part in parts):
        return "path_traversal"
    if any(
        part in _FORBIDDEN_PATH_PARTS
        or part.startswith(".env.")
        or "credential" in part
        or "secret" in part
        for part in parts
    ):
        return "forbidden_path"
    return ""


def _within_root(root: Path, relative: str, *, must_exist: bool = False) -> Path:
    reason = _path_reason(relative)
    if reason:
        raise ValueError(reason)
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root / _normalize_repo_path(relative)
    resolved = candidate.resolve(strict=must_exist)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError("path_escapes_root")
    return resolved


def _python_fixed_search(
    search_root: Path,
    *,
    workspace_root: Path,
    query: str,
    glob: str,
    timeout: int,
    max_results: int,
) -> subprocess.CompletedProcess[str]:
    """Provide a bounded fixed-string search when ripgrep is unavailable."""

    started = time.monotonic()
    resolved_workspace = workspace_root.resolve(strict=True)
    resolved_search_root = search_root.resolve(strict=True)
    candidates = [resolved_search_root] if resolved_search_root.is_file() else resolved_search_root.rglob("*")
    skip_parts = _FORBIDDEN_PATH_PARTS | {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "build",
        "dist",
        "venv",
    }
    matches: list[str] = []
    for candidate in candidates:
        if time.monotonic() - started >= timeout:
            raise subprocess.TimeoutExpired(["python-fixed-search", query], timeout, output="\n".join(matches))
        try:
            resolved = candidate.resolve(strict=True)
            relative_path = resolved.relative_to(resolved_workspace)
        except (OSError, ValueError):
            continue
        if not resolved.is_file() or any(part.lower() in skip_parts for part in relative_path.parts[:-1]):
            continue
        relative = relative_path.as_posix()
        if glob and not (fnmatch.fnmatch(relative, glob) or fnmatch.fnmatch(resolved.name, glob)):
            continue
        try:
            if resolved.stat().st_size > 2_000_000:
                continue
            raw = resolved.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw:
            continue
        text = raw.decode("utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            column = line.find(query)
            if column < 0:
                continue
            matches.append(f"{relative}:{line_number}:{column + 1}:{line}")
            if len(matches) > max_results:
                break
        if len(matches) > max_results:
            break
    stdout = "\n".join(matches) + ("\n" if matches else "")
    args = ["python-fixed-search", "--fixed-strings", query, str(resolved_search_root)]
    return subprocess.CompletedProcess(args, 0 if matches else 1, stdout=stdout, stderr=None)


def _goal_referenced_files(goal: str, root: Path) -> list[str]:
    """Return safe existing repository files named explicitly in backticks."""

    files: list[str] = []
    for raw in re.findall(r"`([^`\r\n]{1,300})`", str(goal or "")):
        normalized = _normalize_repo_path(raw)
        if not normalized or _path_reason(normalized):
            continue
        try:
            candidate = _within_root(root, normalized, must_exist=True)
        except (OSError, ValueError):
            continue
        if candidate.is_file() and normalized not in files:
            files.append(normalized)
    return files


def _matches_edit_scope(path: str, allowed: tuple[str, ...]) -> bool:
    if not allowed:
        return True
    normalized = _normalize_repo_path(path)
    return any(normalized == item or normalized.startswith(item + "/") for item in allowed)


def _edit_path_reason(path: str, allowed: tuple[str, ...]) -> str:
    reason = _path_reason(path)
    if reason:
        return reason
    normalized = _normalize_repo_path(path)
    if not _matches_edit_scope(normalized, allowed):
        return "outside_allowed_edit_paths"
    parts = {part.lower() for part in normalized.split("/") if part}
    if parts.intersection(_EXPLICIT_EDIT_SCOPE_PARTS) and not allowed:
        return "explicit_allowed_edit_path_required"
    return ""


def _is_scope_violation_reason(reason: str) -> bool:
    normalized = str(reason or "").strip().lower()
    return normalized in {
        "absolute_path",
        "forbidden_path",
        "path_escapes_root",
        "path_traversal",
        "unsafe_glob",
    }


def _candidate_changed_paths(root: Path, *, timeout: int) -> tuple[list[str], list[str], bool]:
    diff_names = _git(root, "diff", "--name-only", "--", timeout=timeout)
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "--", timeout=timeout)
    diff_paths = [_normalize_repo_path(line) for line in diff_names.stdout.splitlines() if line.strip()]
    untracked_paths = [_normalize_repo_path(line) for line in untracked.stdout.splitlines() if line.strip()]
    paths = list(dict.fromkeys([*diff_paths, *untracked_paths]))
    return paths, untracked_paths, diff_names.returncode == 0 and untracked.returncode == 0


def _patch_changed_files(patch_text: str) -> list[str]:
    files: list[str] = []
    for line in patch_text.splitlines():
        candidates: list[str] = []
        if line.startswith("diff --git "):
            try:
                parts = shlex.split(line, posix=True)
            except ValueError:
                parts = []
            if len(parts) >= 4:
                candidates.extend(parts[2:4])
        elif line.startswith(("+++ ", "--- ")):
            raw = line[4:].split("\t", 1)[0].strip()
            if raw.startswith('"'):
                try:
                    parsed = shlex.split(raw, posix=True)
                except ValueError:
                    parsed = []
                if parsed:
                    raw = parsed[0]
            candidates.append(raw)
        elif line.startswith(("rename from ", "rename to ")):
            candidates.append(line.split(" ", 2)[2].strip())
        for candidate in candidates:
            normalized = _normalize_repo_path(candidate)
            if normalized and normalized != "/dev/null" and normalized not in files:
                files.append(normalized)
    return files


def _command_is_safe(command: str) -> bool:
    normalized = " ".join(str(command or "").strip().split())
    return bool(normalized) and any(pattern.search(normalized) for pattern in _SAFE_COMMAND_PATTERNS)


def _command_args(command: str) -> list[str]:
    parts = shlex.split(command.strip(), posix=False)
    cleaned = [part.strip('"') for part in parts]
    if not cleaned:
        return []
    first = cleaned[0].lower()
    if first in {"python", "python.exe"}:
        cleaned[0] = os.environ.get("WORKSTATION_PYTHON") or sys.executable
    elif os.name == "nt" and first in {"npm", "pnpm", "yarn"}:
        cleaned[0] = first + ".cmd"
    return cleaned


def _acceptance_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if re.search(r"api[_-]?key|authorization|cookie|credential|password|passwd|secret|token", key, re.I):
            env.pop(key, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _tool_specs(acceptance_commands: tuple[str, ...]) -> list[ToolSpec]:
    claim_values = sorted(_CANONICAL_CLAIMS)
    command_values = list(acceptance_commands)
    return [
        ToolSpec(
            "workspace_search",
            "Search tracked workspace text by fixed string inside the repository root.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["query"],
            },
        ),
        ToolSpec(
            "workspace_read",
            "Read a bounded line range from one repository file.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                    "rationale": {"type": "string"},
                },
                "required": ["path"],
            },
        ),
        ToolSpec(
            "workspace_patch",
            "Apply one unified diff in the detached candidate worktree.",
            {
                "type": "object",
                "properties": {
                    "unified_diff": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["unified_diff"],
            },
        ),
        ToolSpec(
            "workspace_test",
            "Run one configured acceptance command, or all pending commands.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "enum": [*command_values, "all"]},
                    "rationale": {"type": "string"},
                },
                "required": ["command"],
            },
        ),
        ToolSpec(
            "workspace_diff",
            "Capture and validate the final candidate diff.",
            {"type": "object", "properties": {"rationale": {"type": "string"}}},
        ),
        ToolSpec(
            "workspace_finish",
            "Finish only after search, read, patch, acceptance tests, final diff, and semantic review are evidenced.",
            {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "review": {
                        "type": "string",
                        "description": (
                            "Compare the latest diff_preview against the goal, including target selection, "
                            "preserved data, and every requested output."
                        ),
                    },
                    "claims": {"type": "array", "items": {"type": "string", "enum": claim_values}},
                },
                "required": ["summary", "review"],
            },
        ),
    ]


class _CallableDecisionSource:
    def __init__(self, planner: WorkspacePlanner | Any) -> None:
        self.planner = planner
        self.provider = "callable"
        self.model = type(planner).__name__
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def next_action(self, context: dict[str, Any]) -> dict[str, Any]:
        method = getattr(self.planner, "next_action", None)
        result = method(context) if callable(method) else self.planner(context)
        return result if isinstance(result, dict) else {"action": "invalid", "value": result}

    def observe(self, _action: dict[str, Any], _observation: dict[str, Any]) -> None:
        return None


class _ModelDecisionSource:
    def __init__(
        self,
        client: Any,
        *,
        goal: str,
        acceptance_commands: tuple[str, ...],
        max_tokens: int,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.specs = _tool_specs(acceptance_commands)
        self.provider = ""
        self.model = ""
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": (
                f"[GOAL]\n{goal[:6000]}\n\n"
                "Use exactly one workspace tool per turn. Inspect before editing. After a failed test, read the "
                "failure and issue a repair patch. Read every file that the requested change depends on before "
                "patching it. Keep patches narrowly scoped, but include every requested output in the final "
                "candidate. After the final patch, re-read every changed file and compare the resulting content "
                "against the goal before testing and finishing. Read goal_referenced_files directly instead of "
                "repeatedly searching for paths already named by the goal. After workspace_diff, inspect its "
                "diff_preview and write an explicit semantic comparison in workspace_finish.review. Do not claim "
                "completion until workspace_finish is accepted."
            ),
        }]
        self.pending_calls: list[Any] = []

    def next_action(self, context: dict[str, Any]) -> dict[str, Any]:
        self.messages.append({
            "role": "user",
            "content": "[CURRENT STATE]\n" + json.dumps(_safe_json_value(context), ensure_ascii=False)[:24_000],
        })
        turn = self.client.send(
            self.messages,
            system=(
                "You are EvoMind's bounded workspace decision loop. Choose the next evidence-gathering or repair "
                "tool. All execution happens in an isolated detached worktree. Never request shell commands beyond "
                "the configured acceptance list, never access credentials, never commit, merge, deploy, or claim "
                "parity with another agent. Paths in observations such as patch_path, log_path, candidate_diff_path, "
                "and artifact_path are read-only execution evidence outside the repository; never search, read, or "
                "patch them. Follow workflow.phase and workflow.recommended_actions to conserve the bounded step "
                "budget. A plain-text answer is not completion; call workspace_finish only after reviewing the "
                "latest diff_preview against the goal."
            ),
            tools=self.specs,
            max_tokens=self.max_tokens,
            temperature=0.1,
        )
        self.provider = str(getattr(turn, "provider", "") or self.provider)
        self.model = str(getattr(turn, "model", "") or self.model)
        self.usage["input_tokens"] += int(getattr(turn, "input_tokens", 0) or 0)
        self.usage["output_tokens"] += int(getattr(turn, "output_tokens", 0) or 0)
        self.messages.append({"role": "assistant", "content": turn.raw_content})
        self.pending_calls = list(getattr(turn, "tool_calls", []) or [])
        if not self.pending_calls:
            return {
                "action": "finish",
                "summary": str(getattr(turn, "text", "") or "Model returned no workspace tool."),
                "claims": [],
                "_implicit_finish": True,
            }
        call = self.pending_calls[0]
        action = _MODEL_TOOL_TO_ACTION.get(str(call.name or ""), "invalid")
        payload = dict(call.input or {})
        payload["action"] = action
        payload["_tool_name"] = str(call.name or "")
        return payload

    def observe(self, _action: dict[str, Any], observation: dict[str, Any]) -> None:
        if not self.pending_calls:
            self.messages.append({
                "role": "user",
                "content": "A plain-text finish was rejected. Call workspace_finish after satisfying all gates.",
            })
            return
        results = []
        for index, call in enumerate(self.pending_calls):
            if index == 0:
                content = json.dumps(_safe_json_value(observation), ensure_ascii=False)[:18_000]
                is_error = not bool(observation.get("ok"))
            else:
                content = "Only one workspace action is executed per turn; replan after the first result."
                is_error = True
            results.append(ToolResult(tool_use_id=call.id, content=content, is_error=is_error).to_wire())
        self.messages.append({"role": "user", "content": results})
        self.pending_calls = []


def _claim_snapshot(evidence: dict[str, Any], main_unchanged: bool) -> dict[str, dict[str, Any]]:
    tests_passed = bool(evidence.get("all_acceptance_passed"))
    final_diff = bool(evidence.get("final_diff_current"))
    values = {
        "workspace_searched": bool(evidence.get("searched")),
        "workspace_file_read": bool(evidence.get("read_files")),
        "candidate_patch_applied": int(evidence.get("patch_generation") or 0) > 0,
        "acceptance_commands_passed": tests_passed,
        "final_diff_captured": final_diff,
        "main_worktree_unchanged": main_unchanged,
        "review_candidate_ready": bool(
            evidence.get("searched")
            and evidence.get("read_files")
            and int(evidence.get("patch_generation") or 0) > 0
            and tests_passed
            and final_diff
            and main_unchanged
        ),
    }
    return {
        name: {"claim": name, "supported": supported, "source": "workspace_runtime_evidence"}
        for name, supported in values.items()
    }


def _workflow_guidance(
    evidence: dict[str, Any],
    *,
    required_paths: tuple[str, ...],
    require_post_patch_read: bool,
) -> dict[str, Any]:
    """Describe the next unfinished workflow gate without taking control from the planner."""

    read_files = set(evidence.get("read_files") or [])
    patch_generation = int(evidence.get("patch_generation") or 0)
    patched_files = list(evidence.get("patched_files") or [])
    unread_required = [path for path in required_paths if path not in read_files]
    stale_changed = [
        path for path in patched_files
        if evidence.get("read_generation", {}).get(path) != patch_generation
    ]
    if not evidence.get("searched"):
        phase = "discovery"
        recommended = ["search"]
        next_gate = "search"
    elif patch_generation <= 0 and (not read_files or unread_required):
        phase = "inspection"
        recommended = ["read", "search"]
        next_gate = "read_dependencies"
    elif patch_generation <= 0:
        phase = "edit"
        recommended = ["patch", "read", "search"]
        next_gate = "patch"
    elif require_post_patch_read and stale_changed:
        phase = "post_patch_review"
        recommended = ["read", "patch"]
        next_gate = "read_changed_files"
    elif not evidence.get("all_acceptance_passed"):
        phase = "acceptance"
        recommended = ["test", "patch", "read"]
        next_gate = "acceptance_tests"
    elif not evidence.get("final_diff_current"):
        phase = "diff_review"
        recommended = ["diff"]
        next_gate = "capture_diff"
    else:
        phase = "finish"
        recommended = ["finish", "patch"]
        next_gate = "semantic_review_and_finish"
    return {
        "phase": phase,
        "next_gate": next_gate,
        "recommended_actions": recommended,
        "unread_required_paths": unread_required,
        "post_patch_unread_paths": stale_changed if require_post_patch_read else [],
    }


def run_workspace_agent(
    root: Path | str,
    *,
    goal: str,
    planner: WorkspacePlanner | Any | None = None,
    client: Any | None = None,
    acceptance_commands: Sequence[str] = ("git diff --check",),
    allowed_edit_paths: Sequence[str] = (),
    required_edit_paths: Sequence[str] = (),
    require_post_patch_read: bool = False,
    limits: WorkspaceAgentLimits | None = None,
    artifact_dir: Path | str | None = None,
    observer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a model- or callback-directed search/read/patch/test/diff loop.

    ``planner`` receives a fresh structured context before every action.  A
    provider-neutral ``client`` may be supplied instead; when neither is given,
    ``AgentMessageClient`` is used.  The function never commits or merges.
    """

    if planner is not None and client is not None:
        raise ValueError("provide planner or client, not both")
    root_path = Path(root).resolve()
    bounded = (limits or WorkspaceAgentLimits()).bounded()
    commands = tuple(dict.fromkeys(" ".join(str(item).strip().split()) for item in acceptance_commands if str(item).strip()))
    commands = commands or ("git diff --check",)
    unsafe_commands = [command for command in commands if not _command_is_safe(command)]
    if unsafe_commands:
        raise ValueError(f"unsafe acceptance command(s): {unsafe_commands}")
    allowed_paths = tuple(dict.fromkeys(_normalize_repo_path(item) for item in allowed_edit_paths if str(item).strip()))
    unsafe_allowed_paths = [path for path in allowed_paths if _path_reason(path)]
    if unsafe_allowed_paths:
        raise ValueError(f"unsafe allowed_edit_paths: {unsafe_allowed_paths}")
    required_paths = tuple(dict.fromkeys(_normalize_repo_path(item) for item in required_edit_paths if str(item).strip()))
    unsafe_required_paths = [path for path in required_paths if _path_reason(path)]
    if unsafe_required_paths:
        raise ValueError(f"unsafe required_edit_paths: {unsafe_required_paths}")
    if allowed_paths and any(not _matches_edit_scope(path, allowed_paths) for path in required_paths):
        raise ValueError("required_edit_paths must be inside allowed_edit_paths")

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = "workspace_" + generated_at.replace(":", "").replace("+00:00", "Z") + "_" + uuid.uuid4().hex[:8]
    run_artifacts = (
        Path(artifact_dir).resolve()
        if artifact_dir
        else Path(tempfile.gettempdir()).resolve() / "evomind" / "workspace_agent_runs" / run_id
    )
    steps_dir = run_artifacts / "steps"
    commands_dir = run_artifacts / "commands"
    patches_dir = run_artifacts / "patches"
    manifest_path = run_artifacts / "manifest.json"
    candidate_diff_path = run_artifacts / "candidate.diff"

    command_logs: list[str] = []
    scope_violations: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    test_results: list[dict[str, Any]] = []
    requested_claims: list[str] = []
    final_summary = ""
    final_review = ""
    final_diff = ""
    stop_reason = "not_started"
    status = "blocked"
    completed = False
    cleanup_ok = False
    temp_parent: Path | None = None
    worktree: Path | None = None
    start = time.monotonic()

    def emit(phase: str, state: str, message: str, **details: Any) -> None:
        if observer is None:
            return
        try:
            observer({
                "source": "workspace_agent",
                "phase": phase,
                "status": state,
                "message": message,
                "details": _safe_json_value(details),
            })
        except Exception:
            pass

    def command_log(step: int, action: str, command: Sequence[str] | str, output: str) -> str:
        path = commands_dir / f"{step:02d}_{action}_{len(command_logs) + 1:02d}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = command if isinstance(command, str) else subprocess.list2cmdline(list(command))
        path.write_text(f"$ {rendered}\n{_safe_text(output, limit=200_000)}", encoding="utf-8")
        command_logs.append(str(path))
        return str(path)

    def remaining_timeout() -> int:
        remaining = bounded.total_timeout_seconds - int(time.monotonic() - start)
        return max(1, min(bounded.command_timeout_seconds, remaining))

    setup_error = ""
    setup_stop_reason = "workspace_setup_failed"
    main_head_before = ""
    main_head_after = ""
    main_status_before = ""
    main_status_after = ""
    main_dirty_before = False
    main_dirty_after = False
    base_head = ""
    evidence: dict[str, Any] = {
        "searched": False,
        "read_files": [],
        "read_generation": {},
        "patch_generation": 0,
        "patch_attempts": 0,
        "patched_files": [],
        "intent_to_add_files": [],
        "test_runs": 0,
        "current_test_status": {},
        "all_acceptance_passed": False,
        "diff_generation": -1,
        "final_diff_current": False,
    }

    try:
        if not root_path.is_dir():
            raise RuntimeError("workspace root does not exist")
        top = _git(root_path, "rev-parse", "--show-toplevel", timeout=60)
        if top.returncode != 0:
            raise RuntimeError("workspace root is not a Git repository")
        if Path(top.stdout.strip()).resolve() != root_path:
            raise RuntimeError("workspace root must be the Git top-level directory")
        head = _git(root_path, "rev-parse", "HEAD", timeout=60)
        if head.returncode != 0:
            raise RuntimeError("workspace repository has no HEAD commit")
        base_head = head.stdout.strip()
        main_head_before = base_head
        main_status_before, main_dirty_before = _status_digest(root_path)
        if main_dirty_before:
            setup_stop_reason = "dirty_main_worktree"
            raise RuntimeError(
                "dirty_main_worktree: refusing to build a detached HEAD candidate while the real workspace "
                "contains uncommitted changes"
            )
        for directory in (steps_dir, commands_dir, patches_dir):
            directory.mkdir(parents=True, exist_ok=True)
        temp_parent = Path(tempfile.mkdtemp(prefix="evomind-workspace-agent-"))
        worktree = temp_parent / "worktree"
        added = _git(root_path, "worktree", "add", "--detach", str(worktree), base_head, timeout=120)
        command_log(0, "setup", ["git", "worktree", "add", "--detach", str(worktree), base_head], added.stdout)
        if added.returncode != 0:
            raise RuntimeError(f"git worktree add failed: {_safe_text(added.stdout, limit=1000)}")
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        setup_error = _safe_text(exc, limit=1600)

    if setup_error:
        if worktree is not None and worktree.exists():
            removed = _git(root_path, "worktree", "remove", "--force", str(worktree), timeout=120)
            command_log(99, "cleanup", ["git", "worktree", "remove", "--force", str(worktree)], removed.stdout)
        _git(root_path, "worktree", "prune", timeout=60)
        if temp_parent is not None:
            shutil.rmtree(temp_parent, ignore_errors=True)
        payload = {
            "ok": False,
            "completed": False,
            "needs_continuation": False,
            "status": "blocked_dirty_main_worktree" if setup_stop_reason == "dirty_main_worktree" else "blocked",
            "stop_reason": setup_stop_reason,
            "message": setup_error,
            "schema": "evomind.workspace_agent.v1",
            "tool": "workspace_agent",
            "goal": goal[:4000],
            "generated_at": generated_at,
            "run_id": run_id,
            "artifact_path": str(manifest_path),
            "command_logs": command_logs,
            "scope_violations": scope_violations,
            "claims": [],
            "unsupported_claims": [],
            "final_diff": "",
            "candidate_diff_path": str(candidate_diff_path),
            "source_revision": base_head,
            "main_dirty_before": main_dirty_before,
            "main_worktree_modified": False,
            "human_gate": "review_candidate_before_merge",
        }
        _write_json(manifest_path, payload)
        return payload

    assert worktree is not None
    goal_referenced_files = _goal_referenced_files(goal, worktree)
    if planner is not None:
        source: Any = _CallableDecisionSource(planner)
    else:
        model_client = client or AgentMessageClient(max_retries=1, timeout=bounded.command_timeout_seconds)
        if not bool(getattr(model_client, "is_available", lambda: True)()):
            source = None
            stop_reason = "provider_unavailable"
        else:
            source = _ModelDecisionSource(
                model_client,
                goal=goal,
                acceptance_commands=commands,
                max_tokens=bounded.model_max_tokens,
            )

    def add_scope_violation(step: int, action: str, path: str, reason: str) -> None:
        item = {"step": step, "action": action, "path": _safe_text(path, limit=800), "reason": reason}
        scope_violations.append(item)

    try:
        if source is not None:
            emit("start", "running", "Workspace decision loop started.", base_head=base_head)
            for step in range(1, bounded.max_steps + 1):
                elapsed = time.monotonic() - start
                if elapsed >= bounded.total_timeout_seconds:
                    stop_reason = "total_timeout_exhausted"
                    break
                current_tests = dict(evidence["current_test_status"])
                evidence["all_acceptance_passed"] = bool(commands) and all(current_tests.get(command) is True for command in commands)
                evidence["final_diff_current"] = bool(final_diff) and evidence["diff_generation"] == evidence["patch_generation"]
                context = {
                    "schema": "evomind.workspace_agent.context.v1",
                    "goal": goal[:6000],
                    "step": step,
                    "budget": {
                        "max_steps": bounded.max_steps,
                        "remaining_steps": bounded.max_steps - step + 1,
                        "max_patch_attempts": bounded.max_patch_attempts,
                        "remaining_patch_attempts": bounded.max_patch_attempts - evidence["patch_attempts"],
                        "max_test_runs": bounded.max_test_runs,
                        "remaining_test_runs": bounded.max_test_runs - evidence["test_runs"],
                        "remaining_seconds": max(0, bounded.total_timeout_seconds - int(elapsed)),
                    },
                    "acceptance_commands": list(commands),
                    "allowed_edit_paths": list(allowed_paths),
                    "required_edit_paths": list(required_paths),
                    "goal_referenced_files": goal_referenced_files,
                    "evidence": _safe_json_value(evidence),
                    "workflow": _workflow_guidance(
                        evidence,
                        required_paths=required_paths,
                        require_post_patch_read=require_post_patch_read,
                    ),
                    "last_observation": _model_observation(observations[-1]) if observations else {},
                    "recent_observations": _model_observation(observations[-6:]),
                    "completion_contract": [
                        "search",
                        "read",
                        "patch every required edit path",
                        "read every changed file after the final patch" if require_post_patch_read else "inspect changed files",
                        "all acceptance commands pass",
                        "diff",
                        "finish",
                    ],
                    "source_revision": base_head,
                    "execution_mode": "detached_worktree_review_candidate",
                }
                try:
                    action = source.next_action(context)
                except Exception as exc:
                    stop_reason = f"decision_error:{type(exc).__name__}"
                    observations.append({"step": step, "action": "decision", "ok": False, "error": _safe_text(exc, limit=1200)})
                    break
                name = str(action.get("action") or "").strip().lower()
                observation: dict[str, Any] = {"step": step, "action": name, "ok": False}
                rationale = _safe_text(action.get("rationale"), limit=1200)
                terminal = False

                if name not in _ACTIONS:
                    observation.update({"error": "unknown_action", "allowed_actions": sorted(_ACTIONS)})
                elif name == "search":
                    query = str(action.get("query") or "")
                    relative = _normalize_repo_path(str(action.get("path") or ".")) or "."
                    glob = str(action.get("glob") or "").strip()
                    try:
                        search_root = worktree if relative == "." else _within_root(worktree, relative, must_exist=True)
                        if not search_root.exists():
                            raise ValueError("search_path_missing")
                        if glob and (".." in glob.replace("\\", "/").split("/") or re.match(r"^[A-Za-z]:|^/", glob)):
                            raise ValueError("unsafe_glob")
                        if not query:
                            raise ValueError("empty_query")
                        ripgrep = shutil.which("rg")
                        if ripgrep:
                            args = [
                                ripgrep, "--line-number", "--column", "--no-heading", "--color", "never",
                                "--fixed-strings", "-e", query,
                            ]
                            if glob:
                                args.extend(["--glob", glob])
                            args.extend(["--", str(search_root)])
                            searched = _run(args, cwd=worktree, timeout=remaining_timeout())
                            backend = "ripgrep"
                        else:
                            searched = _python_fixed_search(
                                search_root,
                                workspace_root=worktree,
                                query=query,
                                glob=glob,
                                timeout=remaining_timeout(),
                                max_results=bounded.max_search_results,
                            )
                            args = list(searched.args)
                            backend = "python"
                        log_path = command_log(step, name, args, searched.stdout)
                        lines = searched.stdout.splitlines()
                        limited = lines[: bounded.max_search_results]
                        ok = searched.returncode in {0, 1}
                        if ok:
                            evidence["searched"] = True
                        observation.update({
                            "ok": ok,
                            "query": query[:500],
                            "path": relative,
                            "backend": backend,
                            "matches": limited,
                            "match_count": len(lines),
                            "truncated": len(lines) > len(limited),
                            "exit_code": searched.returncode,
                            "log_path": log_path,
                        })
                    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                        reason = "search_timeout" if isinstance(exc, subprocess.TimeoutExpired) else _safe_text(exc, limit=500)
                        if _is_scope_violation_reason(reason):
                            add_scope_violation(step, name, relative, reason)
                        observation["error"] = reason
                elif name == "read":
                    relative = _normalize_repo_path(str(action.get("path") or ""))
                    try:
                        path = _within_root(worktree, relative, must_exist=True)
                        if not path.is_file():
                            raise ValueError("read_path_not_file")
                        start_line = max(1, int(action.get("start_line") or 1))
                        end_line = max(start_line, int(action.get("end_line") or start_line + 239))
                        raw = path.read_text(encoding="utf-8", errors="replace")
                        selected = raw.splitlines()[start_line - 1:end_line]
                        rendered = "\n".join(f"{start_line + index}: {line}" for index, line in enumerate(selected))
                        rendered = rendered[: bounded.max_read_bytes]
                        evidence["read_files"] = list(dict.fromkeys([*evidence["read_files"], relative]))
                        evidence["read_generation"][relative] = evidence["patch_generation"]
                        observation.update({
                            "ok": True,
                            "path": relative,
                            "start_line": start_line,
                            "end_line": start_line + max(0, len(selected) - 1),
                            "content": rendered,
                            "content_sha256": hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest(),
                            "ends_with_newline": raw.endswith(("\n", "\r")),
                            "newline_style": (
                                "crlf" if "\r\n" in raw else "lf" if "\n" in raw else "cr" if "\r" in raw else "none"
                            ),
                            "truncated": len(rendered.encode("utf-8")) >= bounded.max_read_bytes,
                        })
                    except (OSError, ValueError, TypeError) as exc:
                        reason = _safe_text(exc, limit=500)
                        if _is_scope_violation_reason(reason):
                            add_scope_violation(step, name, relative, reason)
                        observation["error"] = reason
                elif name == "patch":
                    evidence["patch_attempts"] += 1
                    patch_text = str(action.get("unified_diff") or action.get("patch") or "")
                    normalized_patch = patch_text.replace("\r\n", "\n").replace("\r", "\n")
                    terminal_newline_added = bool(normalized_patch) and not normalized_patch.endswith("\n")
                    if terminal_newline_added:
                        normalized_patch += "\n"
                    if evidence["patch_attempts"] > bounded.max_patch_attempts:
                        observation["error"] = "patch_budget_exhausted"
                    elif len(normalized_patch.encode("utf-8")) > bounded.max_patch_bytes:
                        observation["error"] = "patch_too_large"
                    else:
                        changed = _patch_changed_files(normalized_patch)
                        violations = []
                        for path in changed:
                            reason = _edit_path_reason(path, allowed_paths)
                            if reason:
                                violations.append({"path": path, "reason": reason})
                                add_scope_violation(step, name, path, reason)
                        if not normalized_patch.strip() or not changed:
                            observation["error"] = "empty_or_unparseable_patch"
                        elif violations:
                            observation.update({"error": "patch_scope_violation", "violations": violations})
                        else:
                            new_files = [
                                path for path in changed
                                if not _within_root(worktree, path).exists()
                            ]
                            patch_path = patches_dir / f"{step:02d}.diff"
                            with patch_path.open("w", encoding="utf-8", newline="\n") as handle:
                                handle.write(normalized_patch)
                            apply_args = ["apply", "--recount", "--ignore-space-change"]
                            checked = _git(worktree, *apply_args, "--check", str(patch_path), timeout=remaining_timeout())
                            check_log = command_log(
                                step,
                                "patch_check",
                                ["git", *apply_args, "--check", str(patch_path)],
                                checked.stdout,
                            )
                            if checked.returncode != 0:
                                observation.update({
                                    "error": "git_apply_check_failed",
                                    "exit_code": checked.returncode,
                                    "output": _safe_text(checked.stdout, limit=8000),
                                    "log_path": check_log,
                                })
                            else:
                                applied = _git(worktree, *apply_args, str(patch_path), timeout=remaining_timeout())
                                apply_log = command_log(
                                    step,
                                    "patch_apply",
                                    ["git", *apply_args, str(patch_path)],
                                    applied.stdout,
                                )
                                intent_result: subprocess.CompletedProcess[str] | None = None
                                intent_log = ""
                                if applied.returncode == 0 and new_files:
                                    intent_result = _git(
                                        worktree,
                                        "add",
                                        "--intent-to-add",
                                        "--",
                                        *new_files,
                                        timeout=remaining_timeout(),
                                    )
                                    intent_log = command_log(
                                        step,
                                        "patch_intent_to_add",
                                        ["git", "add", "--intent-to-add", "--", *new_files],
                                        intent_result.stdout,
                                    )
                                actual, untracked, candidate_state_ok = _candidate_changed_paths(
                                    worktree,
                                    timeout=remaining_timeout(),
                                )
                                unexpected = [path for path in actual if path not in set([*evidence["patched_files"], *changed])]
                                intent_failed = intent_result is not None and intent_result.returncode != 0
                                if applied.returncode != 0 or intent_failed or not candidate_state_ok or unexpected:
                                    for path in unexpected:
                                        add_scope_violation(step, name, path, "unexpected_changed_file")
                                    observation.update({
                                        "error": (
                                            "git_apply_failed"
                                            if applied.returncode
                                            else "intent_to_add_failed"
                                            if intent_failed
                                            else "candidate_state_failed"
                                            if not candidate_state_ok
                                            else "unexpected_changed_files"
                                        ),
                                        "exit_code": intent_result.returncode if intent_failed and intent_result else applied.returncode,
                                        "unexpected_files": unexpected,
                                        "untracked_files": untracked,
                                        "output": _safe_text(applied.stdout, limit=8000),
                                        "log_path": intent_log or apply_log,
                                    })
                                else:
                                    evidence["patch_generation"] += 1
                                    evidence["patched_files"] = list(dict.fromkeys([*evidence["patched_files"], *changed]))
                                    evidence["intent_to_add_files"] = list(
                                        dict.fromkeys([*evidence["intent_to_add_files"], *new_files])
                                    )
                                    evidence["current_test_status"] = {}
                                    evidence["all_acceptance_passed"] = False
                                    evidence["diff_generation"] = -1
                                    evidence["final_diff_current"] = False
                                    final_diff = ""
                                    observation.update({
                                        "ok": True,
                                        "changed_files": changed,
                                        "intent_to_add_files": new_files,
                                        "patch_generation": evidence["patch_generation"],
                                        "terminal_newline_added": terminal_newline_added,
                                        "patch_path": str(patch_path),
                                        "log_path": apply_log,
                                    })
                elif name == "test":
                    requested = " ".join(str(action.get("command") or "all").strip().split())
                    selected = list(commands) if requested == "all" else [requested]
                    if any(command not in commands for command in selected):
                        observation.update({"error": "command_not_allowlisted", "requested": requested, "allowed": list(commands)})
                        add_scope_violation(step, name, requested, "command_not_allowlisted")
                    elif evidence["test_runs"] + len(selected) > bounded.max_test_runs:
                        observation["error"] = "test_budget_exhausted"
                    elif evidence["patch_generation"] <= 0:
                        observation["error"] = "test_requires_applied_patch"
                    else:
                        command_results = []
                        all_passed = True
                        for command in selected:
                            evidence["test_runs"] += 1
                            before_diff = _git(worktree, "diff", "--binary", "--", timeout=remaining_timeout())
                            before_status = _git_status(worktree, timeout=remaining_timeout())
                            args = _command_args(command)
                            timed_out = False
                            try:
                                ran = _run(args, cwd=worktree, timeout=remaining_timeout(), env=_acceptance_env())
                                output = ran.stdout
                                exit_code = ran.returncode
                            except subprocess.TimeoutExpired as exc:
                                output = _safe_text((exc.stdout or "") if isinstance(exc.stdout, str) else "", limit=20_000)
                                output += f"\nCommand timed out after {remaining_timeout()}s."
                                exit_code = 124
                                timed_out = True
                            after_diff = _git(worktree, "diff", "--binary", "--", timeout=remaining_timeout())
                            after_status = _git_status(worktree, timeout=remaining_timeout())
                            diff_mutated = before_diff.stdout != after_diff.stdout
                            status_mutated = before_status.stdout != after_status.stdout
                            snapshot_ok = all(
                                result.returncode == 0
                                for result in (before_diff, before_status, after_diff, after_status)
                            )
                            mutated = diff_mutated or status_mutated
                            passed = exit_code == 0 and snapshot_ok and not mutated
                            evidence["current_test_status"][command] = passed
                            all_passed = all_passed and passed
                            if mutated:
                                final_diff = ""
                                evidence["diff_generation"] = -1
                                evidence["final_diff_current"] = False
                            log_path = command_log(step, "test", args, output)
                            item = {
                                "step": step,
                                "command": command,
                                "allowed": True,
                                "exit_code": exit_code,
                                "passed": passed,
                                "timed_out": timed_out,
                                "candidate_mutated_by_test": mutated,
                                "candidate_diff_mutated_by_test": diff_mutated,
                                "candidate_status_mutated_by_test": status_mutated,
                                "candidate_snapshot_valid": snapshot_ok,
                                "status_sha256_before": hashlib.sha256(
                                    before_status.stdout.encode("utf-8", errors="replace")
                                ).hexdigest(),
                                "status_sha256_after": hashlib.sha256(
                                    after_status.stdout.encode("utf-8", errors="replace")
                                ).hexdigest(),
                                "output_tail": _safe_text(output[-6000:], limit=6000),
                                "log_path": log_path,
                                "patch_generation": evidence["patch_generation"],
                            }
                            command_results.append(item)
                            test_results.append(item)
                        evidence["all_acceptance_passed"] = all(
                            evidence["current_test_status"].get(command) is True for command in commands
                        )
                        observation.update({
                            "ok": all_passed,
                            "results": command_results,
                            "all_acceptance_passed": evidence["all_acceptance_passed"],
                        })
                elif name == "diff":
                    diff_result = _git(worktree, "diff", "--binary", "--", timeout=remaining_timeout())
                    log_path = command_log(step, name, ["git", "diff", "--binary", "--"], diff_result.stdout)
                    changed, untracked, candidate_state_ok = _candidate_changed_paths(
                        worktree,
                        timeout=remaining_timeout(),
                    )
                    unexpected = [path for path in changed if path not in evidence["patched_files"]]
                    unsafe = [path for path in changed if _edit_path_reason(path, allowed_paths)]
                    for path in list(dict.fromkeys([*unexpected, *unsafe])):
                        add_scope_violation(step, name, path, "unexpected_or_unsafe_final_diff")
                    encoded = diff_result.stdout.encode("utf-8")
                    if diff_result.returncode != 0:
                        observation.update({"error": "git_diff_failed", "exit_code": diff_result.returncode, "log_path": log_path})
                    elif not candidate_state_ok:
                        observation.update({"error": "candidate_state_failed", "log_path": log_path})
                    elif not diff_result.stdout.strip():
                        observation.update({"error": "empty_candidate_diff", "log_path": log_path})
                    elif len(encoded) > bounded.max_diff_bytes:
                        observation.update({"error": "candidate_diff_too_large", "bytes": len(encoded), "log_path": log_path})
                    elif unexpected or unsafe:
                        observation.update({
                            "error": "final_diff_scope_violation",
                            "unexpected_files": unexpected,
                            "unsafe_files": unsafe,
                            "untracked_files": untracked,
                            "log_path": log_path,
                        })
                    else:
                        preview_limit = min(24_000, bounded.max_read_bytes)
                        final_diff = diff_result.stdout
                        candidate_diff_path.write_bytes(final_diff.encode("utf-8"))
                        evidence["diff_generation"] = evidence["patch_generation"]
                        evidence["final_diff_current"] = True
                        observation.update({
                            "ok": True,
                            "changed_files": changed,
                            "bytes": len(encoded),
                            "candidate_diff_path": str(candidate_diff_path),
                            "diff_sha256": hashlib.sha256(encoded).hexdigest(),
                            "diff_preview": _safe_text(diff_result.stdout, limit=preview_limit),
                            "diff_preview_truncated": len(diff_result.stdout) > preview_limit,
                            "review_instruction": (
                                "Compare every changed line with the goal. Patch again if the target, preserved "
                                "data, or any requested output is wrong; otherwise explain the comparison in "
                                "workspace_finish.review."
                            ),
                            "log_path": log_path,
                        })
                elif name == "finish":
                    requested_claims = [str(item) for item in action.get("claims") or []]
                    final_summary = _safe_text(action.get("summary"), limit=8000)
                    final_review = _safe_text(action.get("review"), limit=8000)
                    model_finish = action.get("_tool_name") == "workspace_finish"
                    evidence["all_acceptance_passed"] = all(
                        evidence["current_test_status"].get(command) is True for command in commands
                    )
                    evidence["final_diff_current"] = bool(final_diff) and evidence["diff_generation"] == evidence["patch_generation"]
                    gates = {
                        "search": bool(evidence["searched"]),
                        "read": bool(evidence["read_files"]),
                        "patch": evidence["patch_generation"] > 0,
                        "required_paths": all(
                            any(path == required or path.startswith(required + "/") for path in evidence["patched_files"])
                            for required in required_paths
                        ),
                        "post_patch_read": (
                            not require_post_patch_read
                            or all(
                                evidence["read_generation"].get(path) == evidence["patch_generation"]
                                for path in evidence["patched_files"]
                            )
                        ),
                        "test": bool(evidence["all_acceptance_passed"]),
                        "diff": bool(evidence["final_diff_current"]),
                        "semantic_review": not model_finish or bool(final_review.strip()),
                        "explicit_finish": not bool(action.get("_implicit_finish")),
                    }
                    missing = [gate for gate, passed in gates.items() if not passed]
                    if missing:
                        observation.update({"error": "completion_contract_not_met", "missing_gates": missing, "gates": gates})
                    else:
                        observation.update({
                            "ok": True,
                            "summary": final_summary,
                            "review": final_review,
                            "gates": gates,
                        })
                        completed = True
                        status = "completed"
                        stop_reason = "finish_accepted"
                        terminal = True

                observation["rationale"] = rationale
                observation = _safe_json_value(observation)
                observations.append(observation)
                record = {
                    "step": step,
                    "action": name,
                    "rationale": rationale,
                    "observation": observation,
                    "patch_generation": evidence["patch_generation"],
                }
                step_path = steps_dir / f"{step:02d}_{name or 'invalid'}.json"
                record["artifact_path"] = str(step_path)
                _write_json(step_path, record)
                step_records.append(record)
                source.observe(action, _model_observation(observation))
                emit("step", "passed" if observation.get("ok") else "blocked", f"{name}: {observation.get('error') or 'completed'}", step=step)
                if terminal:
                    break
            else:
                stop_reason = "step_budget_exhausted"
        if not completed:
            status = "needs_continuation" if source is not None else "blocked"
    finally:
        if worktree.exists():
            removed = _git(root_path, "worktree", "remove", "--force", str(worktree), timeout=120)
            command_log(99, "cleanup", ["git", "worktree", "remove", "--force", str(worktree)], removed.stdout)
            cleanup_ok = removed.returncode == 0
        _git(root_path, "worktree", "prune", timeout=60)
        if temp_parent is not None:
            shutil.rmtree(temp_parent, ignore_errors=True)

    main_head_result = _git(root_path, "rev-parse", "HEAD", timeout=60)
    main_head_after = main_head_result.stdout.strip() if main_head_result.returncode == 0 else "git-head-error"
    main_status_after, main_dirty_after = _status_digest(root_path)
    main_unchanged = main_head_before == main_head_after and main_status_before == main_status_after
    if not main_unchanged:
        completed = False
        status = "blocked"
        stop_reason = "main_worktree_changed"

    evidence["all_acceptance_passed"] = all(
        evidence["current_test_status"].get(command) is True for command in commands
    )
    evidence["final_diff_current"] = bool(final_diff) and evidence["diff_generation"] == evidence["patch_generation"]
    claim_map = _claim_snapshot(evidence, main_unchanged)
    claims = [item for item in claim_map.values() if item["supported"]]
    unsupported_claims: list[dict[str, Any]] = []
    for claim in requested_claims:
        item = claim_map.get(claim)
        if item is None:
            unsupported_claims.append({"claim": claim, "reason": "claim_not_auditable"})
        elif not item["supported"]:
            unsupported_claims.append({"claim": claim, "reason": "runtime_evidence_missing"})

    payload = {
        "ok": completed and main_unchanged,
        "completed": completed and main_unchanged,
        "needs_continuation": status == "needs_continuation",
        "status": status,
        "stop_reason": stop_reason,
        "schema": "evomind.workspace_agent.v1",
        "tool": "workspace_agent",
        "generated_at": generated_at,
        "run_id": run_id,
        "goal": goal[:4000],
        "provider": getattr(source, "provider", "") if source is not None else "",
        "model": getattr(source, "model", "") if source is not None else "",
        "usage": getattr(source, "usage", {"input_tokens": 0, "output_tokens": 0}) if source is not None else {},
        "limits": asdict(bounded),
        "budget": {
            "steps_used": len(step_records),
            "patch_attempts": evidence["patch_attempts"],
            "test_runs": evidence["test_runs"],
            "elapsed_seconds": round(time.monotonic() - start, 3),
        },
        "acceptance_commands": list(commands),
        "allowed_edit_paths": list(allowed_paths),
        "required_edit_paths": list(required_paths),
        "require_post_patch_read": bool(require_post_patch_read),
        "evidence": evidence,
        "steps": step_records,
        "test_results": test_results,
        "failure_observed": any(item.get("passed") is False for item in test_results),
        "repair_attempted": evidence["patch_attempts"] > 1,
        "replanned_after_failure": any(
            patch_record["step"] > test_item["step"]
            and patch_record["action"] == "patch"
            and patch_record["observation"].get("ok") is True
            for test_item in test_results
            if test_item.get("passed") is False
            for patch_record in step_records
        ),
        "scope_violations": scope_violations,
        "claims": claims,
        "unsupported_claims": unsupported_claims,
        "summary": final_summary,
        "semantic_review": final_review,
        "final_diff": final_diff,
        "candidate_diff_path": str(candidate_diff_path),
        "candidate_diff_sha256": hashlib.sha256(final_diff.encode("utf-8")).hexdigest() if final_diff else "",
        "artifact_path": str(manifest_path),
        "command_logs": command_logs,
        "source_revision": base_head,
        "cleanup_ok": cleanup_ok,
        "main_head_before": main_head_before,
        "main_head_after": main_head_after,
        "main_status_digest_before": main_status_before,
        "main_status_digest_after": main_status_after,
        "main_dirty_before": main_dirty_before,
        "main_dirty_after": main_dirty_after,
        "main_worktree_modified": not main_unchanged,
        "commit_created": False,
        "merged": False,
        "human_gate": "review_candidate_before_merge",
        "execution_boundary": "workspace actions are root-limited; acceptance processes run without an OS sandbox",
        "epistemic_status": (
            "validated_review_candidate_not_merged"
            if completed and main_unchanged
            else "incomplete_candidate_not_merged"
        ),
        "next_safe_action": "human_review_candidate_diff" if completed else "resume_workspace_agent_from_last_observation",
    }
    _write_json(manifest_path, payload)
    emit("finish", "passed" if payload["ok"] else "blocked", f"Workspace loop ended with {status}.")
    return payload


__all__ = ["WorkspaceAgentLimits", "WorkspacePlanner", "run_workspace_agent"]
