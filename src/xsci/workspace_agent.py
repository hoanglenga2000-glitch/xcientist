"""Bounded, auditable workspace loop for EvoMind code tasks.

The decision source may be a local callback or a provider-neutral message
client.  It chooses one structured action at a time; this module owns all
filesystem and process execution.  Source changes are applied only in a
detached Git worktree, acceptance commands are allowlisted, and the final
candidate remains behind a human merge gate.
"""
from __future__ import annotations

import ast
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
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from research_os.agent.messaging import AgentMessageClient, ToolResult, ToolSpec
from research_os.llm_client import LLMError

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
_SHELL_OPERATOR_RE = re.compile(r"[\r\n;&|<>`]|\$\(")
_NON_EXECUTING_TEST_FLAGS = {
    "--cache-show",
    "--co",
    "--collect-only",
    "--collectonly",
    "--fixtures",
    "--fixtures-per-test",
    "--help",
    "--list-tests",
    "--list",
    "--markers",
    "--no-run",
    "--setup-only",
    "--setup-plan",
    "--version",
    "-h",
}
_TEST_PATH_CONTROL_FLAGS = {
    "--confcutdir",
    "--config-file",
    "--override-ini",
    "--prefix",
    "--pyargs",
    "--rootdir",
    "--userconfig",
    "-c",
}
_DYNAMIC_PYTEST_FLAGS = {"--disable-warnings", "-q", "-qq", "-x"}
_DYNAMIC_PYTEST_TB_RE = re.compile(r"^--tb=(?:auto|long|short|line|native|no)$", re.I)
_DYNAMIC_PYTEST_MAXFAIL_RE = re.compile(r"^--maxfail=[1-9][0-9]?$", re.I)
_PYTEST_BOOTSTRAP = (
    "import importlib.util,pathlib,sys\n"
    "root=pathlib.Path(sys.argv[1]).resolve()\n"
    "trusted=(pathlib.Path(sys.prefix).resolve(),pathlib.Path(sys.base_prefix).resolve())\n"
    "for name in ('pytest','_pytest','pluggy'):\n"
    " spec=importlib.util.find_spec(name)\n"
    " if spec is None or not spec.origin: raise SystemExit(126)\n"
    " origin=pathlib.Path(spec.origin).resolve()\n"
    " if not any(origin==base or base in origin.parents for base in trusted): raise SystemExit(126)\n"
    "import pytest\n"
    "src=root/'src'\n"
    "sys.path[:0]=[str(path) for path in (src,root) if path.is_dir()]\n"
    "raise SystemExit(pytest.main(sys.argv[2:]))\n"
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
_MODEL_AUDIT_PATH_KEYS = {"artifact_path", "candidate_diff_path", "junit_path", "log_path", "patch_path"}


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
    max_decision_retries: int = 2

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
            max_decision_retries=max(0, min(int(self.max_decision_retries), 5)),
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


def _candidate_content_digest(
    root: Path,
    *,
    timeout: int,
    required_paths: Sequence[str] = (),
) -> tuple[str, bool]:
    """Hash every changed path's content, including files Git cannot diff yet."""

    changed, _, paths_ok = _candidate_changed_paths(root, timeout=timeout)
    content_paths = list(dict.fromkeys([*changed, *map(_normalize_repo_path, required_paths)]))
    digest = hashlib.sha256()
    digest.update(b"evomind-candidate-content-v1\0")
    try:
        for relative in sorted(path for path in content_paths if path):
            path = _within_root(root, relative, must_exist=False)
            digest.update(relative.encode("utf-8", errors="replace"))
            digest.update(b"\0")
            if not os.path.lexists(path):
                digest.update(b"missing\0")
                continue
            if path.is_symlink():
                digest.update(b"symlink\0")
                digest.update(os.readlink(path).encode("utf-8", errors="replace"))
                digest.update(b"\0")
                continue
            if path.is_file():
                digest.update(b"file\0")
                with path.open("rb") as handle:
                    while True:
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                digest.update(b"\0")
            else:
                digest.update(b"other\0")
    except (OSError, ValueError):
        return digest.hexdigest(), False
    return digest.hexdigest(), paths_ok


def _candidate_state_digest(
    root: Path,
    *,
    timeout: int,
    content_paths: Sequence[str] = (),
) -> tuple[str, bool]:
    diff = _git(root, "diff", "--binary", "--", timeout=timeout)
    status = _git_status(root, timeout=timeout)
    content_digest, content_ok = _candidate_content_digest(
        root,
        timeout=timeout,
        required_paths=content_paths,
    )
    payload = diff.stdout.encode("utf-8", errors="replace") + b"\0" + status.stdout.encode(
        "utf-8",
        errors="replace",
    ) + b"\0" + content_digest.encode("ascii")
    return hashlib.sha256(payload).hexdigest(), diff.returncode == 0 and status.returncode == 0 and content_ok


def _candidate_snapshot(
    root: Path,
    *,
    timeout: int,
    content_paths: Sequence[str] = (),
) -> _CandidateSnapshot:
    diff = _git(root, "diff", "--binary", "--", timeout=timeout)
    changed, untracked, paths_ok = _candidate_changed_paths(root, timeout=timeout)
    state_sha256, state_ok = _candidate_state_digest(
        root,
        timeout=timeout,
        content_paths=content_paths,
    )
    content_sha256, content_ok = _candidate_content_digest(
        root,
        timeout=timeout,
        required_paths=content_paths,
    )
    encoded = diff.stdout.encode("utf-8", errors="replace")
    return _CandidateSnapshot(
        diff=diff.stdout,
        diff_sha256=hashlib.sha256(encoded).hexdigest(),
        state_sha256=state_sha256,
        content_sha256=content_sha256,
        changed_paths=tuple(changed),
        untracked_paths=tuple(untracked),
        ok=diff.returncode == 0 and paths_ok and state_ok and content_ok,
    )


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_id(root: Path, relative: str, *, revision: str, timeout: int) -> str:
    result = _git(root, "rev-parse", f"{revision}:{_normalize_repo_path(relative)}", timeout=timeout)
    return result.stdout.strip() if result.returncode == 0 else ""


def _working_blob_id(root: Path, relative: str, *, timeout: int) -> str:
    result = _git(root, "hash-object", "--", _normalize_repo_path(relative), timeout=timeout)
    return result.stdout.strip() if result.returncode == 0 else ""


def _dynamic_test_support_paths(root: Path, target: str, *, timeout: int) -> list[str]:
    tracked = _git(root, "ls-files", "-z", timeout=timeout)
    if tracked.returncode != 0:
        return []
    target_path = _normalize_repo_path(target)
    target_parts = target_path.split("/")
    directories = ["."]
    for index in range(1, len(target_parts)):
        directories.append("/".join(target_parts[:index]))
    candidates = {target_path}
    for directory in directories:
        prefix = "" if directory == "." else directory + "/"
        candidates.update({prefix + "conftest.py", prefix + "pytest.ini", prefix + "tox.ini"})
    candidates.update({"pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml", "conftest.py"})
    tracked_paths = {
        _normalize_repo_path(item)
        for item in tracked.stdout.split("\0")
        if item.strip()
    }
    return sorted(path for path in candidates if path in tracked_paths)


def _dynamic_test_baseline_digests(
    root: Path,
    target: str,
    *,
    revision: str,
    timeout: int,
) -> tuple[dict[str, str], bool]:
    support_paths = _dynamic_test_support_paths(root, target, timeout=timeout)
    digests: dict[str, str] = {}
    for path in support_paths:
        current = _working_blob_id(root, path, timeout=timeout)
        baseline = _git_blob_id(root, path, revision=revision, timeout=timeout)
        if not current or not baseline or current != baseline:
            return digests, False
        digests[path] = baseline
    return digests, bool(digests and target in digests)


def _pytest_targets(command: str) -> list[str]:
    plan, reason = _parse_command_plan(command)
    if reason or plan is None or plan.runner not in {"python_pytest", "pytest_executable"}:
        return []
    start = 3 if plan.runner == "python_pytest" else 1
    targets: list[str] = []
    for token in plan.parts[start:]:
        if token.startswith("-"):
            continue
        candidate = _normalize_repo_path(token.split("::", 1)[0])
        if candidate.lower().endswith(".py"):
            targets.append(candidate)
    return list(dict.fromkeys(targets))


def _pytest_plugin_modules(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    modules: list[str] = []
    for node in ast.walk(tree):
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytest_plugins"
            for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "pytest_plugins"
        ):
            value = node.value
        if value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
        items = [literal] if isinstance(literal, str) else list(literal) if isinstance(literal, (list, tuple)) else []
        modules.extend(item for item in items if isinstance(item, str) and item)
    return list(dict.fromkeys(modules))


def _pytest_plugin_paths(
    root: Path,
    *,
    revision: str,
    initial_paths: Sequence[str],
    timeout: int,
) -> set[str]:
    tracked = _git(root, "ls-files", "-z", timeout=timeout)
    if tracked.returncode != 0:
        return set()
    tracked_paths = {
        _normalize_repo_path(item)
        for item in tracked.stdout.split("\0")
        if item.strip()
    }
    selected = set(map(_normalize_repo_path, initial_paths))
    queue = list(selected)
    parsed: set[str] = set()
    while queue:
        path = queue.pop()
        if path in parsed or path not in tracked_paths:
            continue
        parsed.add(path)
        source = _git(root, "show", f"{revision}:{path}", timeout=timeout)
        if source.returncode != 0:
            continue
        for module in _pytest_plugin_modules(source.stdout):
            module_path = module.replace(".", "/")
            candidates = (
                f"{module_path}.py",
                f"{module_path}/__init__.py",
                f"src/{module_path}.py",
                f"src/{module_path}/__init__.py",
            )
            for candidate in candidates:
                if candidate in tracked_paths and candidate not in selected:
                    selected.add(candidate)
                    queue.append(candidate)
    return selected


def _pytest_surface_digests(root: Path, *, revision: str, targets: Sequence[str], timeout: int) -> tuple[dict[str, str], bool]:
    if targets:
        selected: set[str] = set()
        for target in targets:
            selected.update(_dynamic_test_support_paths(root, target, timeout=timeout))
    else:
        tracked = _git(root, "ls-files", "-z", timeout=timeout)
        if tracked.returncode != 0:
            return {}, False
        selected = {
            _normalize_repo_path(item)
            for item in tracked.stdout.split("\0")
            if item.strip() and _is_test_support_path(item)
        }
    selected = _pytest_plugin_paths(
        root,
        revision=revision,
        initial_paths=sorted(selected),
        timeout=timeout,
    )
    digests: dict[str, str] = {}
    for path in sorted(selected):
        current = _working_blob_id(root, path, timeout=timeout)
        baseline = _git_blob_id(root, path, revision=revision, timeout=timeout)
        if not current or not baseline or current != baseline:
            return digests, False
        digests[path] = baseline
    return digests, bool(digests)


def _is_test_support_path(relative: str) -> bool:
    normalized = _normalize_repo_path(relative).lower()
    basename = normalized.rsplit("/", 1)[-1]
    return (
        basename == "conftest.py"
        or basename in {"pytest.ini", "tox.ini", "setup.cfg"}
        or basename == "pyproject.toml"
        or normalized.startswith("tests/")
        or "/tests/" in normalized
    )


def _restore_candidate_snapshot(
    root: Path,
    *,
    diff_text: str,
    intent_to_add_files: Sequence[str],
    patch_path: Path,
    timeout: int,
) -> tuple[bool, str]:
    """Restore a candidate worktree to the exact pre-patch diff when possible."""

    reset = _git(root, "reset", "--hard", "HEAD", timeout=timeout)
    clean = _git(root, "clean", "-fdx", timeout=timeout)
    if reset.returncode != 0 or clean.returncode != 0:
        return False, f"reset={reset.returncode}; clean={clean.returncode}"
    if diff_text.strip():
        patch_path.write_text(diff_text, encoding="utf-8", newline="\n")
        restored = _git(
            root,
            "apply",
            "--recount",
            "--ignore-space-change",
            str(patch_path),
            timeout=timeout,
        )
        if restored.returncode != 0:
            return False, f"restore_apply={restored.returncode}: {_safe_text(restored.stdout, limit=2000)}"
    if intent_to_add_files:
        intent = _git(
            root,
            "add",
            "--intent-to-add",
            "--",
            *intent_to_add_files,
            timeout=timeout,
        )
        if intent.returncode != 0:
            return False, f"restore_intent_to_add={intent.returncode}: {_safe_text(intent.stdout, limit=2000)}"
    return True, ""


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


def _normalize_command(command: str) -> str:
    return " ".join(str(command or "").strip().split())


def _command_parts(command: str) -> list[str]:
    try:
        parts = shlex.split(str(command or "").strip(), posix=os.name != "nt")
    except ValueError:
        return []
    return [part.strip('"') for part in parts]


@dataclass(frozen=True)
class _CommandPlan:
    normalized: str
    parts: tuple[str, ...]
    runner: str
    kind: str
    targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class _CandidateSnapshot:
    diff: str
    diff_sha256: str
    state_sha256: str
    content_sha256: str
    changed_paths: tuple[str, ...]
    untracked_paths: tuple[str, ...]
    ok: bool


def _argument_path_reason(token: str) -> str:
    value = str(token or "").strip().strip('"').strip("'")
    if not value or value == ".":
        return ""
    candidate = value.split("=", 1)[1] if value.startswith("-") and "=" in value else value
    if candidate.startswith("@"):
        return "response_file_not_allowed"
    if "$" in candidate or "%" in candidate or candidate.startswith("~"):
        return "environment_or_home_expansion"
    if not candidate:
        return ""
    if candidate.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", candidate):
        return "absolute_path"
    if "://" in candidate:
        return "external_uri"
    if ".." in [part for part in re.split(r"[\\/]", candidate) if part]:
        return "path_traversal"
    return ""


def _parse_command_plan(command: str, *, dynamic: bool = False) -> tuple[_CommandPlan | None, str]:
    raw = str(command or "").strip()
    if not raw:
        return None, "empty_command"
    if len(raw) > 1000:
        return None, "command_too_long"
    if any(ord(character) < 32 or ord(character) == 127 for character in raw):
        return None, "control_character_not_allowed"
    if _SHELL_OPERATOR_RE.search(raw):
        return None, "shell_operator_not_allowed"
    parts = _command_parts(raw)
    if not parts:
        return None, "command_parse_failed"
    if len(parts) > 64:
        return None, "too_many_arguments"

    lowered = [part.lower() for part in parts]
    executable = lowered[0]
    kind = "unsupported"
    runner = ""
    argument_start = 1
    if executable in {"python", "python.exe"}:
        if len(parts) < 3 or lowered[1] != "-m":
            return None, "python_module_mode_required"
        module = lowered[2]
        argument_start = 3
        if module in {"pytest", "unittest"}:
            kind = "behavioral" if module == "pytest" else "host_smoke"
            runner = f"python_{module}"
            if module == "unittest" and len(parts) == 3:
                return None, "test_target_required"
        elif module in {"py_compile", "compileall"}:
            kind = "structural"
            runner = f"python_{module}"
        else:
            return None, "python_module_not_allowed"
    elif executable in {"pytest", "pytest.exe"}:
        kind = "behavioral"
        runner = "pytest_executable"
    elif executable in {"npm", "npm.cmd", "pnpm", "pnpm.cmd", "yarn", "yarn.cmd"}:
        if len(parts) < 2:
            return None, "package_script_required"
        script_index = 2 if lowered[1] == "run" else 1
        if len(parts) <= script_index:
            return None, "package_script_required"
        script = lowered[script_index]
        argument_start = script_index + 1
        if script == "test" or script.startswith("test:"):
            kind = "host_smoke"
        elif script in {"build", "check", "lint", "typecheck"}:
            kind = "structural"
        else:
            return None, "package_script_not_allowed"
        runner = "package_script"
    elif executable == "git":
        if lowered != ["git", "diff", "--check"]:
            return None, "git_command_not_allowed"
        kind = "structural"
        runner = "git_diff_check"
        argument_start = len(parts)
    elif executable == "dotnet":
        if len(parts) < 2 or lowered[1] != "test":
            return None, "dotnet_command_not_allowed"
        kind = "host_smoke"
        runner = "dotnet_test"
        argument_start = 2
    elif executable == "cargo":
        if len(parts) < 2 or lowered[1] not in {"check", "test"}:
            return None, "cargo_command_not_allowed"
        kind = "host_smoke" if lowered[1] == "test" else "structural"
        runner = f"cargo_{lowered[1]}"
        argument_start = 2
    elif executable == "go":
        if len(parts) < 2 or lowered[1] != "test":
            return None, "go_command_not_allowed"
        kind = "host_smoke"
        runner = "go_test"
        argument_start = 2
    else:
        return None, "executable_not_allowed"

    for index, token in enumerate(parts[argument_start:], start=argument_start):
        lowered_token = lowered[index]
        flag = lowered_token.split("=", 1)[0]
        if flag in _NON_EXECUTING_TEST_FLAGS:
            return None, "non_executing_test_mode"
        if flag in _TEST_PATH_CONTROL_FLAGS or (
            runner in {"python_pytest", "pytest_executable"} and lowered_token.startswith("-p")
        ):
            return None, "test_configuration_override_not_allowed"
        if lowered_token in {"--ignore-scripts", "--pdb", "--trace"}:
            return None, "test_execution_bypass_not_allowed"
        if runner == "package_script" and re.search(r"[()^!]", token):
            return None, "package_shell_metacharacter_not_allowed"
        if runner == "go_test" and flag in {"-exec", "-list", "-toolexec"}:
            return None, "test_runner_override_not_allowed"
        if runner.startswith("cargo_") and flag == "--config":
            return None, "test_runner_override_not_allowed"
        if runner == "dotnet_test" and flag in {"--no-build", "--settings", "--test-adapter-path", "-t"}:
            return None, "test_runner_override_not_allowed"
        path_reason = _argument_path_reason(token)
        if path_reason:
            return None, path_reason

    targets: list[str] = []
    if dynamic:
        if runner != "python_pytest":
            return None, "dynamic_runner_not_allowed"
        for token in parts[argument_start:]:
            lowered_token = token.lower()
            if (
                lowered_token in _DYNAMIC_PYTEST_FLAGS
                or _DYNAMIC_PYTEST_TB_RE.fullmatch(token)
                or _DYNAMIC_PYTEST_MAXFAIL_RE.fullmatch(token)
            ):
                continue
            if token.startswith("-"):
                return None, "dynamic_pytest_option_not_allowed"
            if any(character in token for character in "*?["):
                return None, "dynamic_test_glob_not_allowed"
            target = _normalize_repo_path(token.split("::", 1)[0])
            if not target.lower().endswith(".py"):
                return None, "dynamic_pytest_target_required"
            target_parts = Path(target).parts
            target_name = target_parts[-1].lower() if target_parts else ""
            if "tests" not in {part.lower() for part in target_parts} and not (
                target_name.startswith("test_") or target_name.endswith("_test.py")
            ):
                return None, "dynamic_pytest_target_required"
            path_reason = _path_reason(target)
            if path_reason:
                return None, path_reason
            targets.append(target)
        if not targets:
            return None, "dynamic_pytest_target_required"
    plan = _CommandPlan(
        normalized=_normalize_command(raw),
        parts=tuple(parts),
        runner=runner,
        kind=kind,
        targets=tuple(dict.fromkeys(targets)),
    )
    return plan, ""


def _test_command_rejection_reason(command: str, *, require_behavioral: bool = False) -> str:
    plan, reason = _parse_command_plan(command, dynamic=require_behavioral)
    if reason:
        return reason
    if require_behavioral and (plan is None or plan.kind != "behavioral"):
        return "behavioral_test_required"
    return ""


def _command_is_safe(command: str) -> bool:
    return not _test_command_rejection_reason(command)


def _acceptance_evidence_kind(command: str) -> str:
    """Classify allowlisted commands by the evidence they can provide."""

    plan, reason = _parse_command_plan(command)
    return "unsupported" if reason or plan is None else plan.kind


def _refresh_acceptance_evidence(evidence: dict[str, Any], commands: Sequence[str]) -> None:
    current_tests = dict(evidence.get("current_test_status") or {})
    current_generations = dict(evidence.get("current_test_generation") or {})
    current_state_digests = dict(evidence.get("current_test_state_digest") or {})
    current_diff_digests = dict(evidence.get("current_test_diff_digest") or {})
    current_behavioral = dict(evidence.get("current_test_behavioral_evidence") or {})
    patch_generation = int(evidence.get("patch_generation") or 0)
    candidate_state_digest = str(evidence.get("candidate_state_sha256") or "")
    candidate_diff_digest = str(evidence.get("candidate_diff_sha256") or "")
    dynamic_commands = list(evidence.get("dynamic_behavioral_commands") or [])
    all_commands = list(dict.fromkeys([*commands, *dynamic_commands]))
    kinds = {command: _acceptance_evidence_kind(command) for command in all_commands}
    behavioral_commands = [command for command, kind in kinds.items() if kind == "behavioral"]
    structural_commands = [command for command, kind in kinds.items() if kind == "structural"]
    host_smoke_commands = [command for command, kind in kinds.items() if kind == "host_smoke"]
    current = {
        command: current_tests.get(command)
        for command in all_commands
        if current_generations.get(command) == patch_generation
        and current_state_digests.get(command) == candidate_state_digest
        and current_diff_digests.get(command) == candidate_diff_digest
    }
    configured_passed = bool(commands) and all(current.get(command) is True for command in commands)
    latched_failures = list(evidence.get("failed_tests_this_generation") or [])
    current_failures = list(dict.fromkeys([
        *latched_failures,
        *(command for command, passed in current.items() if passed is False),
    ]))
    behavioral_passed = any(
        current.get(command) is True and current_behavioral.get(command) is True
        for command in behavioral_commands
    )
    evidence.update({
        "acceptance_command_kinds": kinds,
        "behavioral_commands": behavioral_commands,
        "structural_commands": structural_commands,
        "host_smoke_commands": host_smoke_commands,
        "all_configured_acceptance_passed": configured_passed,
        "all_acceptance_passed": configured_passed and not current_failures,
        "current_test_failures": current_failures,
        "behavioral_acceptance_passed": behavioral_passed,
        "behavioral_acceptance_patch_generation": (
            patch_generation if behavioral_passed and patch_generation > 0 else None
        ),
    })


def _command_args(command: str, *, execution_root: Path | None = None) -> list[str]:
    plan, reason = _parse_command_plan(command)
    if reason or plan is None:
        return []
    cleaned = list(plan.parts)
    if plan.runner in {"python_pytest", "pytest_executable"}:
        if execution_root is None:
            return []
        start = 3 if plan.runner == "python_pytest" else 1
        python = os.environ.get("WORKSTATION_PYTHON") or sys.executable
        return [
            python,
            "-I",
            "-c",
            _PYTEST_BOOTSTRAP,
            str(execution_root.resolve()),
            *cleaned[start:],
        ]
    first = cleaned[0].lower()
    if first in {"python", "python.exe"}:
        cleaned[0] = os.environ.get("WORKSTATION_PYTHON") or sys.executable
    elif os.name == "nt" and first in {"npm", "pnpm", "yarn"}:
        cleaned[0] = first + ".cmd"
    return cleaned


def _acceptance_env(home: Path, execution_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if re.search(r"api[_-]?key|authorization|cookie|credential|password|passwd|secret|token", key, re.I):
            env.pop(key, None)
    home.mkdir(parents=True, exist_ok=True)
    temp_dir = home / "tmp"
    appdata = home / "appdata"
    local_appdata = home / "local-appdata"
    for directory in (temp_dir, appdata, local_appdata):
        directory.mkdir(parents=True, exist_ok=True)
    env.update({
        "HOME": str(home),
        "USERPROFILE": str(home),
        "APPDATA": str(appdata),
        "LOCALAPPDATA": str(local_appdata),
        "TEMP": str(temp_dir),
        "TMP": str(temp_dir),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "PIP_CONFIG_FILE": os.devnull,
        "PYTHONPATH": str(execution_root / "src") if (execution_root / "src").is_dir() else str(execution_root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "ALL_PROXY": "http://127.0.0.1:9",
        "NO_PROXY": "127.0.0.1,localhost,::1",
        "EVOMIND_TEST_NETWORK_POLICY": "loopback_only_best_effort",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    })
    for key in (
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AZURE_CONFIG_DIR",
        "DOCKER_CONFIG",
        "KUBECONFIG",
        "SSH_AGENT_PID",
        "SSH_AUTH_SOCK",
    ):
        env.pop(key, None)
    return env


def _junit_counts(path: Path) -> dict[str, int]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "executed": 0, "passed": 0}
    elements = [root] if root.attrib.get("tests", "").isdigit() else list(root.findall(".//testsuite"))
    counts = {
        key: sum(
            int(element.attrib.get(key, "0"))
            for element in elements
            if element.attrib.get(key, "").isdigit()
        )
        for key in ("tests", "failures", "errors", "skipped")
    }
    counts["passed"] = max(
        0,
        counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"],
    )
    counts["executed"] = max(0, counts["tests"] - counts["skipped"])
    return counts


def _tool_specs(
    acceptance_commands: tuple[str, ...],
    *,
    allow_dynamic_behavioral_tests: bool,
) -> list[ToolSpec]:
    claim_values = sorted(_CANONICAL_CLAIMS)
    configured = ", ".join(acceptance_commands)
    if allow_dynamic_behavioral_tests:
        test_description = (
            "Run one configured acceptance command, all configured commands, or propose one bounded behavioral "
            "test after inspecting it. Dynamic tests must use python -m pytest with an explicit repository-relative "
            "test.py or test.py::node target and bounded flags. Configured commands: " + configured
        )
    else:
        test_description = (
            "Run only one configured acceptance command or all configured commands. Dynamic commands are disabled; "
            "do not propose, synthesize, or run any other command. Configured commands: " + configured
        )
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
            test_description,
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "minLength": 1, "maxLength": 1000},
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
        allow_dynamic_behavioral_tests: bool,
        max_tokens: int,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.specs = _tool_specs(
            acceptance_commands,
            allow_dynamic_behavioral_tests=allow_dynamic_behavioral_tests,
        )
        self.provider = ""
        self.model = ""
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        validation_guidance = (
            "Before proposing a behavioral test command, search for and read the relevant existing tests in the "
            "current patch generation, then run the narrowest repository-local `python -m pytest "
            "tests/test_file.py::test_node -q` target that can falsify the patch."
            if allow_dynamic_behavioral_tests
            else "Use workspace_test only with `all` or an exact configured acceptance command; dynamic commands are disabled."
        )
        self.command_boundary = (
            "Never request shell commands beyond the configured acceptance list or a bounded repository-local "
            "behavioral test."
            if allow_dynamic_behavioral_tests
            else "Never request shell commands beyond the configured acceptance list; dynamic tests are disabled."
        )
        self.messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": (
                f"[GOAL]\n{goal[:6000]}\n\n"
                "Use exactly one workspace tool per turn. Inspect before editing. After a failed test, read the "
                "failure and issue a repair patch. Read every file that the requested change depends on before "
                "patching it. Keep patches narrowly scoped, but include every requested output in the final "
                "candidate. After the final patch, re-read every changed file and compare the resulting content "
                f"against the goal before testing and finishing. {validation_guidance} "
                "Structural checks never substitute for behavioral validation. Read "
                "goal_referenced_files directly instead of "
                "repeatedly searching for paths already named by the goal. After workspace_diff, inspect its "
                "diff_preview and write an explicit semantic comparison in workspace_finish.review. Do not claim "
                "completion until workspace_finish is accepted."
            ),
        }]
        self.pending_calls: list[Any] = []

    def next_action(self, context: dict[str, Any]) -> dict[str, Any]:
        state_message = {
            "role": "user",
            "content": "[CURRENT STATE]\n" + json.dumps(_safe_json_value(context), ensure_ascii=False)[:24_000],
        }
        self.messages.append(state_message)
        try:
            turn = self.client.send(
                self.messages,
                system=(
                    "You are EvoMind's bounded workspace decision loop. Choose the next evidence-gathering or repair "
                    f"tool. All execution happens in an isolated detached worktree. {self.command_boundary} Never access "
                    "credentials, never commit, merge, deploy, or claim "
                    "parity with another agent. Paths in observations such as patch_path, log_path, candidate_diff_path, "
                    "and artifact_path are read-only execution evidence outside the repository; never search, read, or "
                    "patch them. Follow workflow.phase and workflow.recommended_actions to conserve the bounded step "
                    "budget. Search and read relevant existing tests in the current patch generation before proposing "
                    "an explicit repository-relative python -m pytest target. Shared-host tests are smoke evidence "
                    "only; an external oracle outside this loop owns strong behavioral validation. "
                    "A plain-text answer is not completion; call workspace_finish only after reviewing the latest "
                    "diff_preview against the goal."
                ),
                tools=self.specs,
                max_tokens=self.max_tokens,
                temperature=0.1,
            )
        except Exception:
            self.messages.pop()
            raise
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
    behavioral_tests_passed = bool(evidence.get("behavioral_acceptance_passed"))
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
            and behavioral_tests_passed
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
    allow_dynamic_behavioral_tests: bool,
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
    elif not evidence.get("all_configured_acceptance_passed"):
        phase = "acceptance"
        recommended = ["test", "patch", "read"]
        next_gate = "acceptance_tests"
    elif evidence.get("current_test_failures"):
        phase = "repair"
        recommended = ["read", "patch", "test"]
        next_gate = "repair_failed_test"
    elif allow_dynamic_behavioral_tests and not evidence.get("behavioral_acceptance_passed"):
        phase = "behavioral_validation"
        recommended = ["search", "read", "test", "patch"]
        next_gate = "targeted_behavioral_test"
    elif allow_dynamic_behavioral_tests and not evidence.get("all_acceptance_passed"):
        phase = "acceptance"
        recommended = ["test", "patch", "read"]
        next_gate = "resolve_test_failures"
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
    allow_dynamic_behavioral_tests: bool = False,
    limits: WorkspaceAgentLimits | None = None,
    artifact_dir: Path | str | None = None,
    observer: Callable[[dict[str, Any]], None] | None = None,
    behavioral_oracle: Callable[[dict[str, Any]], bool] | None = None,
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
    raw_commands = [str(item).strip() for item in acceptance_commands if str(item).strip()]
    unsafe_commands = {
        command: _test_command_rejection_reason(command)
        for command in raw_commands
        if _test_command_rejection_reason(command)
    }
    if unsafe_commands:
        raise ValueError(f"unsafe acceptance command(s): {unsafe_commands}")
    commands = tuple(dict.fromkeys(_normalize_command(item) for item in raw_commands))
    commands = commands or ("git diff --check",)
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
    cleanup_failures: list[str] = []
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

    def remove_worktree(path: Path, *, label: str) -> bool:
        if not path.exists():
            return True
        removed = _git(root_path, "worktree", "remove", "--force", str(path), timeout=120)
        command_log(98, f"cleanup_{label}", ["git", "worktree", "remove", "--force", str(path)], removed.stdout)
        ok = removed.returncode == 0 and not path.exists()
        if not ok:
            cleanup_failures.append(label)
        return ok

    def expected_patch_snapshot(
        *,
        step: int,
        pre_patch_diff: str,
        pre_patch_intent: Sequence[str],
        patch_path: Path,
        new_files: Sequence[str],
        content_paths: Sequence[str],
    ) -> tuple[_CandidateSnapshot | None, str]:
        if temp_parent is None:
            return None, "patch_verifier_parent_missing"
        verifier_root = temp_parent / f"patch-verify-{step:02d}"
        snapshot: _CandidateSnapshot | None = None
        error = ""
        added = _git(
            root_path,
            "worktree",
            "add",
            "--detach",
            str(verifier_root),
            base_head,
            timeout=remaining_timeout(),
        )
        command_log(
            step,
            "patch_verifier_add",
            ["git", "worktree", "add", "--detach", str(verifier_root), base_head],
            added.stdout,
        )
        try:
            if added.returncode != 0:
                error = "patch_verifier_add_failed"
            else:
                restored, restore_error = _restore_candidate_snapshot(
                    verifier_root,
                    diff_text=pre_patch_diff,
                    intent_to_add_files=pre_patch_intent,
                    patch_path=patches_dir / f"{step:02d}_verifier_pre.diff",
                    timeout=remaining_timeout(),
                )
                if not restored:
                    error = f"patch_verifier_restore_failed:{restore_error}"
                else:
                    applied = _git(
                        verifier_root,
                        "apply",
                        "--recount",
                        "--ignore-space-change",
                        str(patch_path),
                        timeout=remaining_timeout(),
                    )
                    command_log(
                        step,
                        "patch_verifier_apply",
                        ["git", "apply", "--recount", "--ignore-space-change", str(patch_path)],
                        applied.stdout,
                    )
                    if applied.returncode != 0:
                        error = "patch_verifier_apply_failed"
                    elif new_files:
                        intent = _git(
                            verifier_root,
                            "add",
                            "--intent-to-add",
                            "--",
                            *new_files,
                            timeout=remaining_timeout(),
                        )
                        if intent.returncode != 0:
                            error = "patch_verifier_intent_failed"
                    if not error:
                        snapshot = _candidate_snapshot(
                            verifier_root,
                            timeout=remaining_timeout(),
                            content_paths=content_paths,
                        )
                        if not snapshot.ok:
                            error = "patch_verifier_snapshot_failed"
        finally:
            if not remove_worktree(verifier_root, label=f"patch-verifier-{step:02d}"):
                error = error or "patch_verifier_cleanup_failed"
        return (snapshot if not error else None), error

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
        "decision_failures": 0,
        "current_test_status": {},
        "current_test_generation": {},
        "current_test_state_digest": {},
        "current_test_diff_digest": {},
        "current_test_behavioral_evidence": {},
        "failed_tests_this_generation": [],
        "dynamic_behavioral_commands": [],
        "all_acceptance_passed": False,
        "behavioral_acceptance_passed": False,
        "behavioral_acceptance_patch_generation": None,
        "candidate_state_sha256": "",
        "candidate_content_sha256": "",
        "candidate_diff_sha256": "",
        "dynamic_test_target_digests": {},
        "dynamic_test_support_paths": {},
        "diff_generation": -1,
        "final_diff_current": False,
        "final_diff_sha256": "",
        "final_diff_state_sha256": "",
        "final_diff_content_sha256": "",
    }
    _refresh_acceptance_evidence(evidence, commands)

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
        initial_snapshot = _candidate_snapshot(worktree, timeout=60)
        if not initial_snapshot.ok:
            raise RuntimeError("initial candidate snapshot failed")
        evidence["candidate_state_sha256"] = initial_snapshot.state_sha256
        evidence["candidate_content_sha256"] = initial_snapshot.content_sha256
        evidence["candidate_diff_sha256"] = initial_snapshot.diff_sha256
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
                allow_dynamic_behavioral_tests=allow_dynamic_behavioral_tests,
                max_tokens=bounded.model_max_tokens,
            )

    def add_scope_violation(step: int, action: str, path: str, reason: str) -> None:
        item = {"step": step, "action": action, "path": _safe_text(path, limit=800), "reason": reason}
        scope_violations.append(item)

    try:
        if source is not None:
            emit("start", "running", "Workspace decision loop started.", base_head=base_head)
            consecutive_decision_failures = 0
            for step in range(1, bounded.max_steps + 1):
                elapsed = time.monotonic() - start
                if elapsed >= bounded.total_timeout_seconds:
                    stop_reason = "total_timeout_exhausted"
                    break
                if evidence["patch_generation"] > 0:
                    observed_state, observed_ok = _candidate_state_digest(
                        worktree,
                        timeout=remaining_timeout(),
                        content_paths=evidence["patched_files"],
                    )
                    if not observed_ok or observed_state != evidence["candidate_state_sha256"]:
                        evidence["current_test_status"] = {}
                        evidence["current_test_generation"] = {}
                        evidence["current_test_state_digest"] = {}
                        evidence["current_test_diff_digest"] = {}
                        evidence["current_test_behavioral_evidence"] = {}
                        evidence["failed_tests_this_generation"] = []
                        status = "blocked"
                        stop_reason = "candidate_state_changed_outside_patch_action"
                        break
                _refresh_acceptance_evidence(evidence, commands)
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
                    "allow_dynamic_behavioral_tests": bool(allow_dynamic_behavioral_tests),
                    "allowed_edit_paths": list(allowed_paths),
                    "required_edit_paths": list(required_paths),
                    "goal_referenced_files": goal_referenced_files,
                    "evidence": _safe_json_value(evidence),
                    "workflow": _workflow_guidance(
                        evidence,
                        required_paths=required_paths,
                        require_post_patch_read=require_post_patch_read,
                        allow_dynamic_behavioral_tests=allow_dynamic_behavioral_tests,
                    ),
                    "last_observation": _model_observation(observations[-1]) if observations else {},
                    "recent_observations": _model_observation(observations[-6:]),
                    "completion_contract": [
                        "search",
                        "read",
                        "patch every required edit path",
                        "read every changed file after the final patch" if require_post_patch_read else "inspect changed files",
                        "all acceptance commands pass",
                        "an independent caller-supplied behavioral oracle validates the current patch generation",
                        "diff",
                        "finish",
                    ],
                    "source_revision": base_head,
                    "execution_mode": "detached_worktree_review_candidate",
                }
                try:
                    action = source.next_action(context)
                    consecutive_decision_failures = 0
                except LLMError as exc:
                    evidence["decision_failures"] += 1
                    consecutive_decision_failures += 1
                    retrying = consecutive_decision_failures <= bounded.max_decision_retries
                    observation = _safe_json_value({
                        "step": step,
                        "action": "decision_retry" if retrying else "decision_error",
                        "ok": False,
                        "error": type(exc).__name__,
                        "retrying": retrying,
                        "consecutive_failures": consecutive_decision_failures,
                    })
                    observations.append(observation)
                    record = {
                        "step": step,
                        "action": observation["action"],
                        "rationale": "",
                        "observation": observation,
                        "patch_generation": evidence["patch_generation"],
                    }
                    step_path = steps_dir / f"{step:02d}_{observation['action']}.json"
                    record["artifact_path"] = str(step_path)
                    _write_json(step_path, record)
                    step_records.append(record)
                    emit(
                        "step",
                        "blocked",
                        f"{observation['action']}: {type(exc).__name__}",
                        step=step,
                        retrying=retrying,
                    )
                    if retrying:
                        continue
                    stop_reason = f"decision_error:{type(exc).__name__}"
                    break
                except Exception as exc:
                    stop_reason = f"decision_error:{type(exc).__name__}"
                    observations.append({"step": step, "action": "decision", "ok": False, "error": _safe_text(exc, limit=1200)})
                    break
                name = str(action.get("action") or "").strip().lower()
                observation: dict[str, Any] = {"step": step, "action": name, "ok": False}
                rationale = _safe_text(action.get("rationale"), limit=1200)
                terminal = False
                state_guard_failed = False
                if name in _ACTIONS and evidence["patch_generation"] > 0:
                    observed_state, observed_ok = _candidate_state_digest(
                        worktree,
                        timeout=remaining_timeout(),
                        content_paths=evidence["patched_files"],
                    )
                    state_guard_failed = (
                        not observed_ok or observed_state != evidence["candidate_state_sha256"]
                    )
                    if state_guard_failed:
                        evidence["current_test_status"] = {}
                        evidence["current_test_generation"] = {}
                        evidence["current_test_state_digest"] = {}
                        evidence["current_test_diff_digest"] = {}
                        evidence["current_test_behavioral_evidence"] = {}
                        evidence["failed_tests_this_generation"] = []
                        observation.update({
                            "error": "candidate_state_changed_outside_patch_action",
                            "expected_state_sha256": evidence["candidate_state_sha256"],
                            "observed_state_sha256": observed_state,
                        })
                        status = "blocked"
                        stop_reason = "candidate_state_changed_outside_patch_action"
                        terminal = True

                if state_guard_failed:
                    pass
                elif name not in _ACTIONS:
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
                            pre_patch_diff_result = _git(
                                worktree,
                                "diff",
                                "--binary",
                                "--",
                                timeout=remaining_timeout(),
                            )
                            pre_patch_diff = pre_patch_diff_result.stdout
                            pre_patch_state, pre_patch_state_ok = _candidate_state_digest(
                                worktree,
                                timeout=remaining_timeout(),
                                content_paths=evidence["patched_files"],
                            )
                            pre_patch_content, pre_patch_content_ok = _candidate_content_digest(
                                worktree,
                                timeout=remaining_timeout(),
                                required_paths=evidence["patched_files"],
                            )
                            pre_patch_intent_to_add = list(evidence["intent_to_add_files"])
                            new_files = [
                                path for path in changed
                                if not _within_root(worktree, path).exists()
                            ]
                            patch_path = patches_dir / f"{step:02d}.diff"
                            with patch_path.open("w", encoding="utf-8", newline="\n") as handle:
                                handle.write(normalized_patch)
                            prospective_paths = list(dict.fromkeys([*evidence["patched_files"], *changed]))
                            pre_patch_diff_sha256 = hashlib.sha256(
                                pre_patch_diff.encode("utf-8", errors="replace")
                            ).hexdigest()
                            pre_patch_matches = bool(
                                pre_patch_diff_result.returncode == 0
                                and pre_patch_state_ok
                                and pre_patch_content_ok
                                and pre_patch_state == evidence["candidate_state_sha256"]
                                and pre_patch_content == evidence["candidate_content_sha256"]
                                and pre_patch_diff_sha256 == evidence["candidate_diff_sha256"]
                            )
                            apply_args = ["apply", "--recount", "--ignore-space-change"]
                            checked = (
                                _git(
                                    worktree,
                                    *apply_args,
                                    "--check",
                                    str(patch_path),
                                    timeout=remaining_timeout(),
                                )
                                if pre_patch_matches
                                else subprocess.CompletedProcess([], 125, stdout="", stderr=None)
                            )
                            check_log = command_log(
                                step,
                                "patch_check",
                                ["git", *apply_args, "--check", str(patch_path)],
                                checked.stdout,
                            )
                            expected_snapshot: _CandidateSnapshot | None = None
                            expected_error = ""
                            if pre_patch_matches and checked.returncode == 0:
                                expected_snapshot, expected_error = expected_patch_snapshot(
                                    step=step,
                                    pre_patch_diff=pre_patch_diff,
                                    pre_patch_intent=pre_patch_intent_to_add,
                                    patch_path=patch_path,
                                    new_files=new_files,
                                    content_paths=prospective_paths,
                                )
                            if not pre_patch_matches:
                                observation.update({
                                    "error": "candidate_changed_during_patch_action",
                                    "expected_state_sha256": evidence["candidate_state_sha256"],
                                    "observed_state_sha256": pre_patch_state,
                                    "expected_diff_sha256": evidence["candidate_diff_sha256"],
                                    "observed_diff_sha256": pre_patch_diff_sha256,
                                })
                                status = "blocked"
                                stop_reason = "candidate_changed_during_patch_action"
                                terminal = True
                            elif checked.returncode != 0:
                                observation.update({
                                    "error": "git_apply_check_failed",
                                    "exit_code": checked.returncode,
                                    "output": _safe_text(checked.stdout, limit=8000),
                                    "log_path": check_log,
                                })
                            elif expected_error or expected_snapshot is None:
                                observation.update({
                                    "error": "patch_verifier_failed",
                                    "verifier_error": expected_error or "missing_expected_snapshot",
                                })
                                status = "blocked"
                                stop_reason = "patch_transaction_failed"
                                terminal = True
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
                                actual_snapshot = _candidate_snapshot(
                                    worktree,
                                    timeout=remaining_timeout(),
                                    content_paths=prospective_paths,
                                )
                                unexpected = [path for path in actual if path not in set([*evidence["patched_files"], *changed])]
                                intent_failed = intent_result is not None and intent_result.returncode != 0
                                transaction_mismatch = bool(
                                    expected_snapshot is None
                                    or not actual_snapshot.ok
                                    or actual_snapshot.diff != expected_snapshot.diff
                                    or actual_snapshot.diff_sha256 != expected_snapshot.diff_sha256
                                    or actual_snapshot.state_sha256 != expected_snapshot.state_sha256
                                    or actual_snapshot.content_sha256 != expected_snapshot.content_sha256
                                    or actual_snapshot.changed_paths != expected_snapshot.changed_paths
                                )
                                if (
                                    applied.returncode != 0
                                    or intent_failed
                                    or not candidate_state_ok
                                    or unexpected
                                    or transaction_mismatch
                                ):
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
                                            else "patch_transaction_mismatch"
                                            if transaction_mismatch
                                            else "unexpected_changed_files"
                                        ),
                                        "exit_code": intent_result.returncode if intent_failed and intent_result else applied.returncode,
                                        "unexpected_files": unexpected,
                                        "untracked_files": untracked,
                                        "transaction_mismatch": transaction_mismatch,
                                        "output": _safe_text(applied.stdout, limit=8000),
                                        "log_path": intent_log or apply_log,
                                    })
                                    rollback_ok, rollback_error = _restore_candidate_snapshot(
                                        worktree,
                                        diff_text=pre_patch_diff,
                                        intent_to_add_files=pre_patch_intent_to_add,
                                        patch_path=patches_dir / f"{step:02d}_rollback.diff",
                                        timeout=remaining_timeout(),
                                    )
                                    evidence["current_test_status"] = {}
                                    evidence["current_test_generation"] = {}
                                    evidence["current_test_state_digest"] = {}
                                    evidence["current_test_diff_digest"] = {}
                                    evidence["current_test_behavioral_evidence"] = {}
                                    evidence["failed_tests_this_generation"] = []
                                    evidence["diff_generation"] = -1
                                    evidence["final_diff_current"] = False
                                    final_diff = ""
                                    restored_state, restored_state_ok = _candidate_state_digest(
                                        worktree,
                                        timeout=remaining_timeout(),
                                        content_paths=evidence["patched_files"],
                                    )
                                    restored_content, restored_content_ok = _candidate_content_digest(
                                        worktree,
                                        timeout=remaining_timeout(),
                                        required_paths=evidence["patched_files"],
                                    )
                                    rollback_digest_matches = bool(
                                        rollback_ok
                                        and pre_patch_diff_result.returncode == 0
                                        and pre_patch_state_ok
                                        and pre_patch_content_ok
                                        and restored_state_ok
                                        and restored_content_ok
                                        and restored_state == pre_patch_state
                                        and restored_content == pre_patch_content
                                    )
                                    evidence["candidate_state_sha256"] = restored_state if restored_state_ok else ""
                                    evidence["candidate_content_sha256"] = (
                                        restored_content if restored_content_ok else ""
                                    )
                                    evidence["candidate_diff_sha256"] = (
                                        pre_patch_diff_sha256 if rollback_digest_matches else ""
                                    )
                                    observation["rollback_ok"] = rollback_digest_matches
                                    if rollback_error:
                                        observation["rollback_error"] = rollback_error
                                    status = "blocked"
                                    stop_reason = "patch_transaction_failed"
                                    terminal = True
                                else:
                                    state_digest = actual_snapshot.state_sha256
                                    content_digest = actual_snapshot.content_sha256
                                    state_ok = actual_snapshot.ok
                                    content_ok = actual_snapshot.ok
                                    if not state_ok or not content_ok:
                                        rollback_ok, rollback_error = _restore_candidate_snapshot(
                                            worktree,
                                            diff_text=pre_patch_diff,
                                            intent_to_add_files=pre_patch_intent_to_add,
                                            patch_path=patches_dir / f"{step:02d}_rollback.diff",
                                            timeout=remaining_timeout(),
                                        )
                                        restored_state, restored_state_ok = _candidate_state_digest(
                                            worktree,
                                            timeout=remaining_timeout(),
                                            content_paths=evidence["patched_files"],
                                        )
                                        restored_content, restored_content_ok = _candidate_content_digest(
                                            worktree,
                                            timeout=remaining_timeout(),
                                            required_paths=evidence["patched_files"],
                                        )
                                        rollback_digest_matches = bool(
                                            rollback_ok
                                            and pre_patch_diff_result.returncode == 0
                                            and pre_patch_state_ok
                                            and pre_patch_content_ok
                                            and restored_state_ok
                                            and restored_content_ok
                                            and restored_state == pre_patch_state
                                            and restored_content == pre_patch_content
                                        )
                                        observation.update({"error": "candidate_state_digest_failed"})
                                        observation["rollback_ok"] = rollback_digest_matches
                                        if rollback_error:
                                            observation["rollback_error"] = rollback_error
                                        evidence["current_test_status"] = {}
                                        evidence["current_test_generation"] = {}
                                        evidence["current_test_state_digest"] = {}
                                        evidence["current_test_diff_digest"] = {}
                                        evidence["current_test_behavioral_evidence"] = {}
                                        evidence["failed_tests_this_generation"] = []
                                        evidence["candidate_state_sha256"] = (
                                            restored_state if restored_state_ok else ""
                                        )
                                        evidence["candidate_content_sha256"] = (
                                            restored_content if restored_content_ok else ""
                                        )
                                        evidence["candidate_diff_sha256"] = (
                                            pre_patch_diff_sha256 if rollback_digest_matches else ""
                                        )
                                        evidence["diff_generation"] = -1
                                        evidence["final_diff_current"] = False
                                        final_diff = ""
                                        status = "blocked"
                                        stop_reason = "patch_transaction_failed"
                                        terminal = True
                                    else:
                                        evidence["patch_generation"] += 1
                                        evidence["patched_files"] = list(
                                            dict.fromkeys([*evidence["patched_files"], *changed])
                                        )
                                        evidence["intent_to_add_files"] = list(
                                            dict.fromkeys([*evidence["intent_to_add_files"], *new_files])
                                        )
                                        evidence["current_test_status"] = {}
                                        evidence["current_test_generation"] = {}
                                        evidence["current_test_state_digest"] = {}
                                        evidence["current_test_diff_digest"] = {}
                                        evidence["current_test_behavioral_evidence"] = {}
                                        evidence["failed_tests_this_generation"] = []
                                        evidence["candidate_state_sha256"] = state_digest
                                        evidence["candidate_content_sha256"] = content_digest
                                        evidence["candidate_diff_sha256"] = actual_snapshot.diff_sha256
                                        _refresh_acceptance_evidence(evidence, commands)
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
                    requested = _normalize_command(action.get("command") or "all")
                    dynamic_commands = list(evidence.get("dynamic_behavioral_commands") or [])
                    dynamic_requested = requested != "all" and requested not in commands
                    dynamic_added = dynamic_requested and requested not in dynamic_commands
                    selected = list(commands) if requested == "all" else [requested]
                    rejection_reason = ""
                    dynamic_plan: _CommandPlan | None = None
                    if dynamic_requested:
                        if not allow_dynamic_behavioral_tests:
                            rejection_reason = "dynamic_behavioral_tests_disabled"
                        else:
                            dynamic_plan, rejection_reason = _parse_command_plan(requested, dynamic=True)
                            if not rejection_reason and dynamic_plan is not None:
                                for target in dynamic_plan.targets:
                                    try:
                                        _within_root(worktree, target, must_exist=True)
                                    except ValueError as exc:
                                        rejection_reason = str(exc)
                                        break
                                    if evidence["read_generation"].get(target) != evidence["patch_generation"]:
                                        rejection_reason = "dynamic_test_target_not_read_for_current_patch"
                                        break
                                    if target in evidence["patched_files"]:
                                        rejection_reason = "dynamic_test_target_modified_by_candidate"
                                        break
                                    support_digests, support_ok = _dynamic_test_baseline_digests(
                                        worktree,
                                        target,
                                        revision=base_head,
                                        timeout=remaining_timeout(),
                                    )
                                    changed_support = [
                                        path
                                        for path in evidence["patched_files"]
                                        if path in support_digests or _is_test_support_path(path)
                                    ]
                                    if changed_support:
                                        rejection_reason = "dynamic_test_support_modified_by_candidate"
                                        break
                                    if not support_ok:
                                        rejection_reason = "dynamic_test_surface_not_baseline_identical"
                                        break
                                    evidence["dynamic_test_target_digests"][target] = support_digests[target]
                                    evidence["dynamic_test_support_paths"][target] = sorted(support_digests)
                    if rejection_reason:
                        observation.update({
                            "error": "unsafe_dynamic_test_command",
                            "requested": requested,
                            "rejection_reason": rejection_reason,
                            "configured": list(commands),
                        })
                        if rejection_reason == "dynamic_behavioral_tests_disabled":
                            observation["policy_rejection"] = True
                        else:
                            add_scope_violation(step, name, requested, rejection_reason)
                    elif evidence["test_runs"] + len(selected) > bounded.max_test_runs:
                        observation["error"] = "test_budget_exhausted"
                    elif evidence["patch_generation"] <= 0:
                        observation["error"] = "test_requires_applied_patch"
                    else:
                        if dynamic_added:
                            evidence["dynamic_behavioral_commands"] = list(dict.fromkeys([
                                *dynamic_commands,
                                requested,
                            ]))
                        command_results = []
                        all_passed = True
                        for command_index, command in enumerate(selected, start=1):
                            evidence["test_runs"] += 1
                            plan, plan_reason = _parse_command_plan(command)
                            is_pytest = bool(
                                not plan_reason
                                and plan is not None
                                and plan.runner in {"python_pytest", "pytest_executable"}
                            )
                            pytest_targets = _pytest_targets(command) if is_pytest else []
                            pytest_surface_digests: dict[str, str] = {}
                            pytest_surface_ok = not is_pytest
                            if is_pytest:
                                pytest_surface_digests, pytest_surface_ok = _pytest_surface_digests(
                                    worktree,
                                    revision=base_head,
                                    targets=pytest_targets,
                                    timeout=remaining_timeout(),
                                )
                            before_diff = _git(worktree, "diff", "--binary", "--", timeout=remaining_timeout())
                            before_status = _git_status(worktree, timeout=remaining_timeout())
                            before_state_digest, before_state_ok = _candidate_state_digest(
                                worktree,
                                timeout=remaining_timeout(),
                                content_paths=evidence["patched_files"],
                            )
                            before_content_digest, before_content_ok = _candidate_content_digest(
                                worktree,
                                timeout=remaining_timeout(),
                                required_paths=evidence["patched_files"],
                            )
                            before_diff_sha256 = hashlib.sha256(
                                before_diff.stdout.encode("utf-8", errors="replace")
                            ).hexdigest()
                            before_snapshot_matches = bool(
                                before_diff.returncode == 0
                                and before_state_ok
                                and before_content_ok
                                and before_state_digest == evidence["candidate_state_sha256"]
                                and before_content_digest == evidence["candidate_content_sha256"]
                                and before_diff_sha256 == evidence["candidate_diff_sha256"]
                            )
                            candidate_patch = before_diff.stdout
                            test_case_root = temp_parent / f"test-execution-{step:02d}-{command_index:02d}"
                            test_root = test_case_root / "workspace"
                            snapshot_patch_path = patches_dir / f"{step:02d}_test_{command_index:02d}.diff"
                            snapshot_patch_path.write_bytes(candidate_patch.encode("utf-8"))
                            added = _git(
                                root_path,
                                "worktree",
                                "add",
                                "--detach",
                                str(test_root),
                                base_head,
                                timeout=remaining_timeout(),
                            )
                            applied = subprocess.CompletedProcess([], 125, stdout="test worktree setup failed", stderr=None)
                            if added.returncode == 0:
                                applied = _git(
                                    test_root,
                                    "apply",
                                    "--recount",
                                    "--ignore-space-change",
                                    str(snapshot_patch_path),
                                    timeout=remaining_timeout(),
                                )
                            setup_ok = added.returncode == 0 and applied.returncode == 0
                            test_before_diff = (
                                _git(test_root, "diff", "--binary", "--", timeout=remaining_timeout())
                                if setup_ok
                                else subprocess.CompletedProcess([], 125, stdout="", stderr=None)
                            )
                            test_before_status = (
                                _git_status(test_root, timeout=remaining_timeout())
                                if setup_ok
                                else subprocess.CompletedProcess([], 125, stdout="", stderr=None)
                            )
                            args = _command_args(command, execution_root=test_root)
                            runtime_junit_path = (
                                run_artifacts / "junit" / f"{step:02d}_{command_index:02d}_runtime.xml"
                            )
                            candidate_junit_path = (
                                run_artifacts / "junit" / f"{step:02d}_{command_index:02d}_candidate.xml"
                            )
                            if is_pytest:
                                runtime_junit_path.parent.mkdir(parents=True, exist_ok=True)
                                runtime_junit_path.unlink(missing_ok=True)
                                args.append(f"--junitxml={runtime_junit_path}")
                            timed_out = False
                            output = ""
                            exit_code = 125
                            test_home = (
                                run_artifacts / "test-homes" / f"{step:02d}_{command_index:02d}_runtime"
                            )
                            if setup_ok:
                                try:
                                    ran = _run(
                                        args,
                                        cwd=test_root,
                                        timeout=remaining_timeout(),
                                        env=_acceptance_env(test_home, test_root),
                                    )
                                    output = ran.stdout
                                    exit_code = ran.returncode
                                except subprocess.TimeoutExpired as exc:
                                    output = _safe_text(
                                        (exc.stdout or "") if isinstance(exc.stdout, str) else "",
                                        limit=20_000,
                                    )
                                    output += f"\nCommand timed out after {remaining_timeout()}s."
                                    exit_code = 124
                                    timed_out = True
                                except (OSError, ValueError) as exc:
                                    output = f"Command launch failed: {type(exc).__name__}: {_safe_text(exc, limit=1000)}"
                                    exit_code = 127
                            else:
                                output = (
                                    f"Disposable test worktree setup failed: add={added.returncode}, "
                                    f"apply={applied.returncode}\n{added.stdout}\n{applied.stdout}"
                                )
                            test_after_diff = (
                                _git(test_root, "diff", "--binary", "--", timeout=remaining_timeout())
                                if setup_ok
                                else subprocess.CompletedProcess([], 125, stdout="", stderr=None)
                            )
                            test_after_status = (
                                _git_status(test_root, timeout=remaining_timeout())
                                if setup_ok
                                else subprocess.CompletedProcess([], 125, stdout="", stderr=None)
                            )
                            after_diff = _git(worktree, "diff", "--binary", "--", timeout=remaining_timeout())
                            after_status = _git_status(worktree, timeout=remaining_timeout())
                            after_state_digest, after_state_ok = _candidate_state_digest(
                                worktree,
                                timeout=remaining_timeout(),
                                content_paths=evidence["patched_files"],
                            )
                            after_content_digest, after_content_ok = _candidate_content_digest(
                                worktree,
                                timeout=remaining_timeout(),
                                required_paths=evidence["patched_files"],
                            )
                            after_diff_sha256 = hashlib.sha256(
                                after_diff.stdout.encode("utf-8", errors="replace")
                            ).hexdigest()
                            diff_mutated = before_diff.stdout != after_diff.stdout
                            status_mutated = before_status.stdout != after_status.stdout
                            candidate_snapshot_ok = all(
                                result.returncode == 0
                                for result in (before_diff, before_status, after_diff, after_status)
                            ) and before_snapshot_matches and after_state_ok and after_content_ok
                            candidate_state_matches = (
                                after_state_digest == evidence["candidate_state_sha256"]
                                and after_content_digest == evidence["candidate_content_sha256"]
                                and after_diff_sha256 == evidence["candidate_diff_sha256"]
                            )
                            candidate_mutated = (
                                diff_mutated
                                or status_mutated
                                or before_state_digest != after_state_digest
                                or before_content_digest != after_content_digest
                            )
                            test_snapshot_ok = setup_ok and all(
                                result.returncode == 0
                                for result in (
                                    test_before_diff,
                                    test_before_status,
                                    test_after_diff,
                                    test_after_status,
                                )
                            )
                            test_diff_mutated = test_before_diff.stdout != test_after_diff.stdout
                            test_status_mutated = test_before_status.stdout != test_after_status.stdout
                            test_snapshot_mutated = test_diff_mutated or test_status_mutated
                            candidate_junit = _junit_counts(runtime_junit_path) if is_pytest else None
                            if is_pytest and runtime_junit_path.is_file():
                                shutil.copyfile(runtime_junit_path, candidate_junit_path)
                            tests_executed = candidate_junit["executed"] if candidate_junit is not None else None
                            test_cleanup_ok = remove_worktree(
                                test_root,
                                label=f"test-{step:02d}-{command_index:02d}",
                            )
                            shutil.rmtree(test_home, ignore_errors=True)
                            shutil.rmtree(test_case_root, ignore_errors=True)
                            test_cleanup_ok = (
                                test_cleanup_ok
                                and not test_home.exists()
                                and not test_case_root.exists()
                            )
                            candidate_passed = bool(
                                exit_code == 0
                                and candidate_snapshot_ok
                                and candidate_state_matches
                                and test_snapshot_ok
                                and not candidate_mutated
                                and not test_snapshot_mutated
                                and test_cleanup_ok
                                and pytest_surface_ok
                                and (
                                    not is_pytest
                                    or (
                                        candidate_junit is not None
                                        and candidate_junit["executed"] > 0
                                        and candidate_junit["passed"] > 0
                                        and candidate_junit["failures"] == 0
                                        and candidate_junit["errors"] == 0
                                    )
                                )
                            )
                            baseline_exit_code: int | None = None
                            baseline_junit: dict[str, int] | None = None
                            baseline_snapshot_mutated: bool | None = None
                            baseline_cleanup_ok: bool | None = None
                            baseline_junit_path: Path | None = None
                            baseline_args: list[str] | None = None
                            baseline_validated = not is_pytest
                            if is_pytest and candidate_passed:
                                baseline_root = test_root
                                runtime_junit_path.unlink(missing_ok=True)
                                baseline_added = _git(
                                    root_path,
                                    "worktree",
                                    "add",
                                    "--detach",
                                    str(baseline_root),
                                    base_head,
                                    timeout=remaining_timeout(),
                                )
                                baseline_setup_ok = baseline_added.returncode == 0
                                baseline_before_diff = (
                                    _git(baseline_root, "diff", "--binary", "--", timeout=remaining_timeout())
                                    if baseline_setup_ok
                                    else subprocess.CompletedProcess([], 125, stdout="", stderr=None)
                                )
                                baseline_before_status = (
                                    _git_status(baseline_root, timeout=remaining_timeout())
                                    if baseline_setup_ok
                                    else subprocess.CompletedProcess([], 125, stdout="", stderr=None)
                                )
                                baseline_junit_path = (
                                    run_artifacts / "junit" / f"{step:02d}_{command_index:02d}_baseline.xml"
                                )
                                baseline_args = _command_args(command, execution_root=baseline_root)
                                baseline_args.append(f"--junitxml={runtime_junit_path}")
                                baseline_output = ""
                                baseline_exit_code = 125
                                if baseline_setup_ok:
                                    try:
                                        baseline_run = _run(
                                            baseline_args,
                                            cwd=baseline_root,
                                            timeout=remaining_timeout(),
                                            env=_acceptance_env(test_home, baseline_root),
                                        )
                                        baseline_output = baseline_run.stdout
                                        baseline_exit_code = baseline_run.returncode
                                    except subprocess.TimeoutExpired as exc:
                                        baseline_output = _safe_text(
                                            (exc.stdout or "") if isinstance(exc.stdout, str) else "",
                                            limit=20_000,
                                        )
                                        baseline_exit_code = 124
                                    except (OSError, ValueError) as exc:
                                        baseline_output = (
                                            f"Baseline launch failed: {type(exc).__name__}: "
                                            f"{_safe_text(exc, limit=1000)}"
                                        )
                                        baseline_exit_code = 127
                                baseline_after_diff = (
                                    _git(baseline_root, "diff", "--binary", "--", timeout=remaining_timeout())
                                    if baseline_setup_ok
                                    else subprocess.CompletedProcess([], 125, stdout="", stderr=None)
                                )
                                baseline_after_status = (
                                    _git_status(baseline_root, timeout=remaining_timeout())
                                    if baseline_setup_ok
                                    else subprocess.CompletedProcess([], 125, stdout="", stderr=None)
                                )
                                baseline_snapshot_ok = baseline_setup_ok and all(
                                    result.returncode == 0
                                    for result in (
                                        baseline_before_diff,
                                        baseline_before_status,
                                        baseline_after_diff,
                                        baseline_after_status,
                                    )
                                )
                                baseline_snapshot_mutated = (
                                    baseline_before_diff.stdout != baseline_after_diff.stdout
                                    or baseline_before_status.stdout != baseline_after_status.stdout
                                )
                                baseline_junit = _junit_counts(runtime_junit_path)
                                if runtime_junit_path.is_file():
                                    shutil.copyfile(runtime_junit_path, baseline_junit_path)
                                baseline_cleanup_ok = remove_worktree(
                                    baseline_root,
                                    label=f"baseline-{step:02d}-{command_index:02d}",
                                )
                                shutil.rmtree(test_home, ignore_errors=True)
                                shutil.rmtree(test_case_root, ignore_errors=True)
                                baseline_cleanup_ok = (
                                    baseline_cleanup_ok
                                    and not test_home.exists()
                                    and not test_case_root.exists()
                                )
                                baseline_validated = bool(
                                    baseline_exit_code not in {0, 124, 125, 127}
                                    and baseline_snapshot_ok
                                    and not baseline_snapshot_mutated
                                    and baseline_cleanup_ok
                                    and baseline_junit["executed"] > 0
                                    and baseline_junit["failures"] + baseline_junit["errors"] > 0
                                )
                                if not baseline_validated:
                                    output += (
                                        "\n[baseline differential did not fail with executed assertions]\n"
                                        + _safe_text(baseline_output[-3000:], limit=3000)
                                    )
                            passed = candidate_passed and baseline_validated
                            host_differential_validated = bool(
                                is_pytest and passed and pytest_surface_ok and baseline_validated
                            )
                            behavioral_oracle_passed = False
                            if behavioral_oracle is not None and host_differential_validated:
                                try:
                                    behavioral_oracle_passed = bool(
                                        behavioral_oracle({
                                            "command": command,
                                            "patch_generation": evidence["patch_generation"],
                                            "candidate_state_sha256": after_state_digest,
                                            "candidate_content_sha256": after_content_digest,
                                            "candidate_diff_sha256": after_diff_sha256,
                                            "host_differential_validated": host_differential_validated,
                                            "candidate_junit": candidate_junit,
                                            "baseline_junit": baseline_junit,
                                        })
                                    )
                                except Exception:
                                    behavioral_oracle_passed = False
                            if candidate_mutated:
                                evidence["current_test_status"] = {}
                                evidence["current_test_generation"] = {}
                                evidence["current_test_state_digest"] = {}
                                evidence["current_test_diff_digest"] = {}
                                evidence["current_test_behavioral_evidence"] = {}
                            else:
                                evidence["current_test_status"][command] = passed
                                evidence["current_test_generation"][command] = evidence["patch_generation"]
                                evidence["current_test_state_digest"][command] = after_state_digest
                                evidence["current_test_diff_digest"][command] = after_diff_sha256
                                evidence["current_test_behavioral_evidence"][command] = behavioral_oracle_passed
                            if not passed:
                                evidence["failed_tests_this_generation"] = list(dict.fromkeys([
                                    *evidence["failed_tests_this_generation"],
                                    command,
                                ]))
                            all_passed = all_passed and passed
                            if candidate_mutated:
                                final_diff = ""
                                evidence["diff_generation"] = -1
                                evidence["final_diff_current"] = False
                            log_path = command_log(step, "test", args, output)
                            item = {
                                "step": step,
                                "command": command,
                                "evidence_kind": _acceptance_evidence_kind(command),
                                "allowed": True,
                                "exit_code": exit_code,
                                "passed": passed,
                                "timed_out": timed_out,
                                "candidate_mutated_by_test": candidate_mutated,
                                "candidate_diff_mutated_by_test": diff_mutated,
                                "candidate_status_mutated_by_test": status_mutated,
                                "candidate_snapshot_valid": candidate_snapshot_ok,
                                "candidate_state_matches_patch": candidate_state_matches,
                                "candidate_state_sha256": after_state_digest,
                                "candidate_content_sha256": after_content_digest,
                                "candidate_diff_sha256": after_diff_sha256,
                                "candidate_snapshot_matches_before_test": before_snapshot_matches,
                                "disposable_test_worktree": True,
                                "test_snapshot_valid": test_snapshot_ok,
                                "test_snapshot_mutated": test_snapshot_mutated,
                                "test_diff_mutated": test_diff_mutated,
                                "test_status_mutated": test_status_mutated,
                                "test_worktree_cleanup_ok": test_cleanup_ok,
                                "tests_executed": tests_executed,
                                "pytest_surface_baseline_identical": pytest_surface_ok,
                                "pytest_surface_digests": pytest_surface_digests,
                                "candidate_junit": candidate_junit,
                                "candidate_passed": candidate_passed,
                                "baseline_exit_code": baseline_exit_code,
                                "baseline_junit": baseline_junit,
                                "baseline_snapshot_mutated": baseline_snapshot_mutated,
                                "baseline_worktree_cleanup_ok": baseline_cleanup_ok,
                                "baseline_differential_validated": baseline_validated,
                                "host_differential_validated": host_differential_validated,
                                "behavioral_oracle_configured": behavioral_oracle is not None,
                                "behavioral_oracle_passed": behavioral_oracle_passed,
                                "junit_path": str(candidate_junit_path) if is_pytest else "",
                                "baseline_junit_path": (
                                    str(baseline_junit_path) if baseline_junit_path is not None else ""
                                ),
                                "runner_argv_equivalent": (
                                    baseline_args == args if baseline_args is not None else None
                                ),
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
                            if candidate_mutated:
                                break
                        _refresh_acceptance_evidence(evidence, commands)
                        candidate_test_mutation = any(
                            item.get("candidate_mutated_by_test") is True for item in command_results
                        )
                        test_cleanup_failure = any(
                            item.get("test_worktree_cleanup_ok") is False
                            or item.get("baseline_worktree_cleanup_ok") is False
                            for item in command_results
                        )
                        observation.update({
                            "ok": all_passed and not candidate_test_mutation,
                            "results": command_results,
                            "dynamic_command_added": dynamic_added,
                            "dynamic_behavioral_commands": list(evidence["dynamic_behavioral_commands"]),
                            "all_acceptance_passed": evidence["all_acceptance_passed"],
                            "behavioral_acceptance_passed": evidence["behavioral_acceptance_passed"],
                        })
                        if candidate_test_mutation:
                            observation["error"] = "acceptance_process_modified_candidate"
                            status = "blocked"
                            stop_reason = "acceptance_process_modified_candidate"
                            terminal = True
                        elif test_cleanup_failure:
                            observation["error"] = "acceptance_process_cleanup_failed"
                            status = "blocked"
                            stop_reason = "acceptance_process_cleanup_failed"
                            terminal = True
                elif name == "diff":
                    first_snapshot = _candidate_snapshot(
                        worktree,
                        timeout=remaining_timeout(),
                        content_paths=evidence["patched_files"],
                    )
                    second_snapshot = _candidate_snapshot(
                        worktree,
                        timeout=remaining_timeout(),
                        content_paths=evidence["patched_files"],
                    )
                    log_path = command_log(
                        step,
                        name,
                        ["git", "diff", "--binary", "--"],
                        first_snapshot.diff,
                    )
                    changed = list(second_snapshot.changed_paths)
                    untracked = list(second_snapshot.untracked_paths)
                    unexpected = [path for path in changed if path not in evidence["patched_files"]]
                    unsafe = [path for path in changed if _edit_path_reason(path, allowed_paths)]
                    for path in list(dict.fromkeys([*unexpected, *unsafe])):
                        add_scope_violation(step, name, path, "unexpected_or_unsafe_final_diff")
                    encoded = second_snapshot.diff.encode("utf-8")
                    snapshot_stable = first_snapshot == second_snapshot
                    snapshot_matches = bool(
                        snapshot_stable
                        and second_snapshot.ok
                        and second_snapshot.state_sha256 == evidence["candidate_state_sha256"]
                        and second_snapshot.content_sha256 == evidence["candidate_content_sha256"]
                        and second_snapshot.diff_sha256 == evidence["candidate_diff_sha256"]
                    )
                    if not snapshot_matches:
                        evidence["current_test_status"] = {}
                        evidence["current_test_generation"] = {}
                        evidence["current_test_state_digest"] = {}
                        evidence["current_test_diff_digest"] = {}
                        evidence["current_test_behavioral_evidence"] = {}
                        evidence["failed_tests_this_generation"] = []
                        observation.update({
                            "error": "candidate_snapshot_mismatch",
                            "snapshot_stable": snapshot_stable,
                            "expected_state_sha256": evidence["candidate_state_sha256"],
                            "observed_state_sha256": second_snapshot.state_sha256,
                            "expected_diff_sha256": evidence["candidate_diff_sha256"],
                            "observed_diff_sha256": second_snapshot.diff_sha256,
                            "log_path": log_path,
                        })
                        status = "blocked"
                        stop_reason = "candidate_snapshot_mismatch"
                        terminal = True
                    elif not second_snapshot.diff.strip():
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
                        final_diff = second_snapshot.diff
                        candidate_diff_path.write_bytes(final_diff.encode("utf-8"))
                        evidence["diff_generation"] = evidence["patch_generation"]
                        evidence["final_diff_sha256"] = second_snapshot.diff_sha256
                        evidence["final_diff_state_sha256"] = second_snapshot.state_sha256
                        evidence["final_diff_content_sha256"] = second_snapshot.content_sha256
                        evidence["final_diff_current"] = True
                        observation.update({
                            "ok": True,
                            "changed_files": changed,
                            "bytes": len(encoded),
                            "candidate_diff_path": str(candidate_diff_path),
                            "diff_sha256": second_snapshot.diff_sha256,
                            "diff_preview": _safe_text(second_snapshot.diff, limit=preview_limit),
                            "diff_preview_truncated": len(second_snapshot.diff) > preview_limit,
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
                    finish_snapshot_a = _candidate_snapshot(
                        worktree,
                        timeout=remaining_timeout(),
                        content_paths=evidence["patched_files"],
                    )
                    finish_snapshot_b = _candidate_snapshot(
                        worktree,
                        timeout=remaining_timeout(),
                        content_paths=evidence["patched_files"],
                    )
                    finish_snapshot_matches = bool(
                        finish_snapshot_a == finish_snapshot_b
                        and finish_snapshot_b.ok
                        and finish_snapshot_b.state_sha256 == evidence["candidate_state_sha256"]
                        and finish_snapshot_b.content_sha256 == evidence["candidate_content_sha256"]
                        and finish_snapshot_b.diff_sha256 == evidence["candidate_diff_sha256"]
                        and finish_snapshot_b.diff == final_diff
                        and finish_snapshot_b.diff_sha256 == evidence["final_diff_sha256"]
                        and finish_snapshot_b.state_sha256 == evidence["final_diff_state_sha256"]
                        and finish_snapshot_b.content_sha256 == evidence["final_diff_content_sha256"]
                    )
                    _refresh_acceptance_evidence(evidence, commands)
                    evidence["final_diff_current"] = bool(
                        final_diff
                        and evidence["diff_generation"] == evidence["patch_generation"]
                        and finish_snapshot_matches
                    )
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
                        "behavioral_test": bool(evidence["behavioral_acceptance_passed"]),
                        "candidate_snapshot": finish_snapshot_matches,
                        "diff": bool(evidence["final_diff_current"]),
                        "semantic_review": not model_finish or bool(final_review.strip()),
                        "explicit_finish": not bool(action.get("_implicit_finish")),
                    }
                    missing = [gate for gate, passed in gates.items() if not passed]
                    if not finish_snapshot_matches:
                        evidence["current_test_status"] = {}
                        evidence["current_test_generation"] = {}
                        evidence["current_test_state_digest"] = {}
                        evidence["current_test_diff_digest"] = {}
                        evidence["current_test_behavioral_evidence"] = {}
                        evidence["failed_tests_this_generation"] = []
                        observation.update({
                            "ok": False,
                            "error": "candidate_snapshot_mismatch",
                            "gates": gates,
                            "expected_diff_sha256": evidence["candidate_diff_sha256"],
                            "observed_diff_sha256": finish_snapshot_b.diff_sha256,
                        })
                        status = "blocked"
                        stop_reason = "candidate_snapshot_mismatch"
                        terminal = True
                    elif missing == ["behavioral_test"]:
                        observation.update({
                            "ok": False,
                            "error": "behavioral_acceptance_missing",
                            "summary": (
                                "Candidate passed configured host checks only; "
                                "no behavioral correctness evidence is available."
                            ),
                            "gates": gates,
                        })
                        status = "format_validated_only"
                        stop_reason = "behavioral_acceptance_missing"
                        terminal = True
                    elif missing:
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
        if not completed and status != "format_validated_only":
            status = "needs_continuation" if source is not None else "blocked"
    finally:
        if worktree.exists():
            if completed:
                cleanup_snapshot = _candidate_snapshot(
                    worktree,
                    timeout=120,
                    content_paths=evidence["patched_files"],
                )
                cleanup_snapshot_matches = bool(
                    cleanup_snapshot.ok
                    and cleanup_snapshot.diff == final_diff
                    and cleanup_snapshot.diff_sha256 == evidence["candidate_diff_sha256"]
                    and cleanup_snapshot.state_sha256 == evidence["candidate_state_sha256"]
                    and cleanup_snapshot.content_sha256 == evidence["candidate_content_sha256"]
                )
                if not cleanup_snapshot_matches:
                    completed = False
                    status = "blocked"
                    stop_reason = "candidate_snapshot_changed_before_cleanup"
                    evidence["final_diff_current"] = False
            cleanup_ok = remove_worktree(worktree, label="candidate")
        _git(root_path, "worktree", "prune", timeout=60)
        if temp_parent is not None:
            shutil.rmtree(temp_parent, ignore_errors=True)
            cleanup_ok = cleanup_ok and not temp_parent.exists()

    main_head_result = _git(root_path, "rev-parse", "HEAD", timeout=60)
    main_head_after = main_head_result.stdout.strip() if main_head_result.returncode == 0 else "git-head-error"
    main_status_after, main_dirty_after = _status_digest(root_path)
    main_unchanged = main_head_before == main_head_after and main_status_before == main_status_after
    if not main_unchanged:
        completed = False
        status = "blocked"
        stop_reason = "main_worktree_changed"
    if not cleanup_ok or cleanup_failures:
        completed = False
        status = "blocked"
        stop_reason = "workspace_cleanup_failed"

    _refresh_acceptance_evidence(evidence, commands)
    evidence["final_diff_current"] = bool(
        evidence["final_diff_current"]
        and final_diff
        and evidence["diff_generation"] == evidence["patch_generation"]
    )
    claim_map = _claim_snapshot(evidence, main_unchanged)
    claims = [item for item in claim_map.values() if item["supported"]]
    unsupported_claims: list[dict[str, Any]] = []
    for claim in requested_claims:
        item = claim_map.get(claim)
        if item is None:
            unsupported_claims.append({"claim": claim, "reason": "claim_not_auditable"})
        elif not item["supported"]:
            unsupported_claims.append({"claim": claim, "reason": "runtime_evidence_missing"})

    format_validated_only = status == "format_validated_only"
    completion_trusted = bool(completed and main_unchanged)
    if completion_trusted:
        trusted_summary = final_summary
        trusted_review = final_review
    elif format_validated_only:
        trusted_summary = (
            "Candidate passed configured host checks only; no strong behavioral correctness evidence is available."
        )
        trusted_review = ""
    else:
        trusted_summary = f"Workspace agent did not complete ({status}; {stop_reason or 'no_stop_reason'})."
        trusted_review = ""
    payload = {
        "ok": completion_trusted,
        "completed": completion_trusted,
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
            "decision_failures": evidence["decision_failures"],
            "patch_attempts": evidence["patch_attempts"],
            "test_runs": evidence["test_runs"],
            "elapsed_seconds": round(time.monotonic() - start, 3),
        },
        "acceptance_commands": list(commands),
        "allow_dynamic_behavioral_tests": bool(allow_dynamic_behavioral_tests),
        "behavioral_oracle_configured": behavioral_oracle is not None,
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
        "summary": trusted_summary,
        "semantic_review": trusted_review,
        "untrusted_model_summary": final_summary if not completion_trusted else "",
        "untrusted_model_semantic_review": final_review if not completion_trusted else "",
        "final_diff": final_diff,
        "candidate_diff_path": str(candidate_diff_path) if final_diff and candidate_diff_path.is_file() else "",
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
        "execution_boundary": (
            "acceptance commands run in disposable Git worktrees with sanitized temporary homes and best-effort "
            "loopback-only proxy settings; no kernel, VM, or container filesystem/network sandbox is present, so "
            "shared-host test results never become strong behavioral evidence without an external oracle"
        ),
        "epistemic_status": (
            "validated_review_candidate_not_merged"
            if completion_trusted
            else "format_validated_only_not_behaviorally_validated"
            if format_validated_only
            else "incomplete_candidate_not_merged"
        ),
        "next_safe_action": (
            "human_review_candidate_diff"
            if completion_trusted
            else "run_external_behavioral_oracle"
            if format_validated_only
            else "resume_workspace_agent_from_last_observation"
        ),
    }
    _write_json(manifest_path, payload)
    emit("finish", "passed" if payload["ok"] else "blocked", f"Workspace loop ended with {status}.")
    return payload


__all__ = ["WorkspaceAgentLimits", "WorkspacePlanner", "run_workspace_agent"]
