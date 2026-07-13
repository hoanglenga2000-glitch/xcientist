"""Isolated engineering execution loop for EvoMind self-upgrades.

The loop turns an auditable patch work order into a tested review candidate:

1. optionally ask the configured read-only Code Agent to generate a diff;
2. validate every patch path against the work order and public source roots;
3. apply the diff only inside a detached temporary Git worktree;
4. run allowlisted acceptance checks;
5. preserve diff, logs, hashes, rollback evidence, and a human merge gate.

The main worktree is never modified by this module.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .kaggle_session import SessionState

_ALLOWED_EDIT_PREFIXES = (
    "src/",
    "tests/",
    "scripts/",
    "web/research-agent-workstation/src/",
)
_ALLOWED_EDIT_FILES = {
    "README.md",
    "install.ps1",
    "pyproject.toml",
    "requirements.txt",
}
_FORBIDDEN_PATH_PARTS = {
    ".env",
    ".git",
    ".xsci",
    "secrets",
    "credentials",
    "workspace",
    "data",
    "reports",
    "node_modules",
    ".next",
    "__pycache__",
}
_ALLOWED_CHECK_PATTERNS = (
    re.compile(r"^python(?:\.exe)?\s+-m\s+py_compile\b", re.I),
    re.compile(r"^python(?:\.exe)?\s+-m\s+pytest\b", re.I),
    re.compile(r"^python(?:\.exe)?\s+scripts[\\/][A-Za-z0-9_.\\/-]+\.py\b", re.I),
    re.compile(r"^npm(?:\.cmd)?\s+run\s+(?:typecheck|build|test)\b", re.I),
    re.compile(r"^git\s+diff\s+--check\b", re.I),
)
_SENSITIVE_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|cookie|password|passwd|secret|private[_-]?key)\s*[:=]\s*\S+"
)


def _safe_text(value: Any, *, limit: int = 8000) -> str:
    text = str(value or "").replace("\x00", " ")
    text = _SENSITIVE_RE.sub(r"\1=[redacted]", text)
    return text[:limit]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(
    args: list[str],
    *,
    cwd: Path,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def _git(root: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE", "GIT_PREFIX"):
        env.pop(key, None)
    return _run(["git", *args], cwd=root, timeout=timeout, env=env)


def _normalize_repo_path(value: str) -> str:
    path = value.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path


def _path_is_safe(path: str) -> bool:
    normalized = _normalize_repo_path(path)
    if not normalized or normalized == "/dev/null":
        return True
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        return False
    parts = [part.lower() for part in normalized.split("/") if part]
    if ".." in parts or any(part in _FORBIDDEN_PATH_PARTS for part in parts):
        return False
    return normalized in _ALLOWED_EDIT_FILES or normalized.startswith(_ALLOWED_EDIT_PREFIXES)


def _patch_changed_files(patch_text: str) -> list[str]:
    files: list[str] = []
    for line in patch_text.splitlines():
        candidate = ""
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                candidate = parts[3]
        elif line.startswith("+++ ") or line.startswith("--- "):
            candidate = line[4:].split("\t", 1)[0]
        if not candidate:
            continue
        normalized = _normalize_repo_path(candidate)
        if normalized and normalized != "/dev/null" and normalized not in files:
            files.append(normalized)
    return files


def _repair_unified_diff_counts(patch_text: str) -> tuple[str, bool]:
    """Recount malformed LLM hunk lengths without changing patch semantics."""
    lines = patch_text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    repaired = list(lines)
    changed = not patch_text.endswith("\n")
    hunk_re = re.compile(
        r"^@@ -(?P<old_start>\d+)(?:,\d+)? \+(?P<new_start>\d+)(?:,\d+)? @@(?P<suffix>.*)$"
    )
    index = 0
    while index < len(lines):
        match = hunk_re.match(lines[index])
        if not match:
            index += 1
            continue
        end = index + 1
        old_count = 0
        new_count = 0
        while end < len(lines):
            line = lines[end]
            if line.startswith("@@ ") or line.startswith("diff --git ") or line.startswith("--- "):
                break
            if line.startswith("\\ No newline at end of file"):
                end += 1
                continue
            if line.startswith(" "):
                old_count += 1
                new_count += 1
            elif line.startswith("-"):
                old_count += 1
            elif line.startswith("+"):
                new_count += 1
            else:
                break
            end += 1
        replacement = (
            f"@@ -{match.group('old_start')},{old_count} "
            f"+{match.group('new_start')},{new_count} @@{match.group('suffix')}"
        )
        if replacement != lines[index]:
            repaired[index] = replacement
            changed = True
        index = max(end, index + 1)
    return "\n".join(repaired) + "\n", changed


def _work_order_body(payload: dict[str, Any]) -> dict[str, Any]:
    body = payload.get("work_order")
    return body if isinstance(body, dict) else payload


def _work_order_is_external_only(payload: dict[str, Any], body: dict[str, Any]) -> bool:
    context = (
        body.get("self_evolution_context")
        if isinstance(body.get("self_evolution_context"), dict)
        else payload.get("self_evolution_context")
        if isinstance(payload.get("self_evolution_context"), dict)
        else {}
    )
    partition = (
        context.get("execution_partition")
        if isinstance(context.get("execution_partition"), dict)
        else body.get("execution_partition")
        if isinstance(body.get("execution_partition"), dict)
        else {}
    )
    code_fixable = partition.get("code_agent_fixable_requirements") if isinstance(partition, dict) else []
    external = partition.get("external_resource_blockers") if isinstance(partition, dict) else []
    issue_id = str(body.get("issue_id") or body.get("selected_backlog_id") or payload.get("selected_issue_id") or "")
    return bool(external) and not bool(code_fixable) and issue_id in {
        "resource_gate_truthfulness",
        "setup_gate_clearance",
        "gpu_blocked",
    }


def _load_work_order(root: Path, work_order_path: Path | None) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = [work_order_path] if work_order_path else [
        root / ".xsci" / "scientist_patch_work_order.json",
        root / ".xsci" / "scientist_self_upgrade_work_order.json",
    ]
    for path in candidates:
        if path is None:
            continue
        resolved = path if path.is_absolute() else root / path
        payload = _read_json(resolved)
        if payload:
            return resolved, payload
    return None, None


def _resolve_patch_path(root: Path, patch_path: Path | None) -> Path | None:
    if patch_path:
        resolved = patch_path if patch_path.is_absolute() else root / patch_path
        return resolved if resolved.exists() else None
    sessions_root = root / "workspace" / "code_agent_sessions"
    manifests = sorted(
        sessions_root.glob("*/session_manifest.json"),
        key=lambda item: item.stat().st_mtime if item.exists() else 0,
        reverse=True,
    )
    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        if not manifest or manifest.get("status") != "completed":
            continue
        relative = manifest.get("patch_path")
        if not relative:
            continue
        candidate = root / str(relative)
        if candidate.exists():
            return candidate
    return None


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    hostname = parsed.hostname
    is_loopback = hostname == "localhost"
    if hostname and not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username
        or parsed.password
        or (parsed.scheme == "http" and not is_loopback)
    ):
        raise ValueError(
            "dashboard endpoint must be an http(s) URL with a hostname and no embedded credentials; "
            "remote endpoints require https"
        )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", "user-agent": "evomind-engineering-loop/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 - scheme validated above
        body = json.loads(response.read().decode("utf-8"))
    return body if isinstance(body, dict) else {}


def _generate_patch_via_code_agent(
    *,
    root: Path,
    session: SessionState,
    work_order: dict[str, Any],
    dashboard_url: str,
    timeout_seconds: int,
    repair_context: dict[str, Any] | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    endpoint = dashboard_url.rstrip("/") + "/api/code-agents/claude/sessions"
    prompt = str(work_order.get("code_agent_prompt") or work_order.get("objective") or "")
    if repair_context:
        prompt = (
            f"{prompt}\n\n"
            "The previous isolated candidate failed. Produce a replacement unified diff, not an incremental diff. "
            "Use the validation evidence below and stay inside files_to_edit.\n"
            f"{json.dumps(repair_context, ensure_ascii=False, indent=2)}"
        )
    body = {
        "task_id": session.selected_task or "system",
        "prompt": prompt,
        "model": os.environ.get("CLAUDE_CODE_MODEL", "claude-opus-4-8"),
        "max_turns": 5,
        "timeout_seconds": timeout_seconds,
    }
    try:
        result = _post_json(endpoint, body, timeout_seconds + 30)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        return None, {"status": "failed", "error": type(exc).__name__, "endpoint": endpoint}
    relative = result.get("patch_path")
    patch = root / str(relative) if relative else None
    return (
        patch if patch and patch.exists() else None,
        {
            "status": result.get("status") or "unknown",
            "session_id": result.get("session_id"),
            "provider": result.get("provider"),
            "model": result.get("model"),
            "manifest_path": result.get("manifest_path"),
            "patch_path": relative,
            "error": _safe_text(result.get("error"), limit=500),
            "repair_context_supplied": bool(repair_context),
        },
    )


def _repair_context(
    *,
    attempt: int,
    changed_files: list[str],
    candidate_diff: str,
    acceptance_results: list[dict[str, Any]],
    message: str,
) -> dict[str, Any]:
    failures = [
        {
            "command": str(item.get("command") or ""),
            "exit_code": item.get("exit_code"),
            "output_tail": _safe_text(item.get("output_tail"), limit=1600),
        }
        for item in acceptance_results
        if item.get("passed") is not True
    ]
    return {
        "previous_attempt": attempt,
        "message": _safe_text(message, limit=1600),
        "changed_files": changed_files,
        "failed_checks": failures,
        "previous_candidate_diff": _safe_text(candidate_diff, limit=12000),
        "required_response": "complete replacement unified diff",
    }


def _command_is_allowed(command: str) -> bool:
    normalized = " ".join(command.strip().split())
    return any(pattern.search(normalized) for pattern in _ALLOWED_CHECK_PATTERNS)


def _command_args(command: str) -> list[str]:
    import shlex

    normalized = command.strip()
    parts = shlex.split(normalized, posix=False)
    if not parts:
        return []
    first = parts[0].strip('"').lower()
    if first in {"python", "python.exe"}:
        parts[0] = os.environ.get("WORKSTATION_PYTHON") or sys.executable
    elif first in {"npm", "npm.cmd"} and os.name == "nt":
        parts[0] = "npm.cmd"
    return [part.strip('"') for part in parts]


def _run_acceptance_checks(
    *,
    worktree: Path,
    commands: list[str],
    logs_dir: Path,
) -> tuple[list[dict[str, Any]], bool]:
    results: list[dict[str, Any]] = []
    all_passed = True
    logs_dir.mkdir(parents=True, exist_ok=True)
    for index, command in enumerate(commands[:12], start=1):
        allowed = _command_is_allowed(command)
        log_path = logs_dir / f"{index:02d}.log"
        if not allowed:
            output = "Command rejected by engineering acceptance allowlist."
            log_path.write_text(output, encoding="utf-8")
            results.append({
                "command": command,
                "allowed": False,
                "exit_code": None,
                "passed": False,
                "log_path": str(log_path),
                "output_tail": output,
            })
            all_passed = False
            continue
        args = _command_args(command)
        timeout = 600 if "npm" in command.lower() or "pytest" in command.lower() else 180
        try:
            completed = _run(args, cwd=worktree, timeout=timeout)
            output = _safe_text(completed.stdout, limit=20000)
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            output = f"Command timed out after {timeout}s."
            exit_code = 124
        log_path.write_text(output, encoding="utf-8")
        passed = exit_code == 0
        all_passed = all_passed and passed
        results.append({
            "command": command,
            "allowed": True,
            "exit_code": exit_code,
            "passed": passed,
            "log_path": str(log_path),
            "output_tail": output[-1600:],
        })
    return results, all_passed


def run_scientist_engineering_loop(
    session: SessionState,
    root: Path,
    *,
    work_order_path: Path | None = None,
    patch_path: Path | None = None,
    generate_patch: bool = False,
    dashboard_url: str = "http://127.0.0.1:8088",
    timeout_seconds: int = 180,
    max_patch_attempts: int = 2,
) -> dict[str, Any]:
    root = Path(root).resolve()
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = f"engineering_{generated_at.replace(':', '').replace('+', 'Z')}"
    artifact_root = root / ".xsci" / "engineering_runs" / run_id
    artifact_path = root / ".xsci" / "scientist_engineering_loop.json"
    run_manifest_path = artifact_root / "manifest.json"
    candidate_diff_path = artifact_root / "candidate.diff"
    trials_path = root / ".xsci" / "scientist_engineering_trials.jsonl"
    work_order_resolved, work_order_payload = _load_work_order(root, work_order_path)

    base: dict[str, Any] = {
        "ok": False,
        "schema": "evomind.ai_scientist.engineering_loop.v2",
        "tool": "scientist_engineering_loop",
        "generated_at": generated_at,
        "run_id": run_id,
        "selected_task": session.selected_task or "",
        "status": "blocked",
        "work_order_path": str(work_order_resolved) if work_order_resolved else "",
        "patch_path": "",
        "changed_files": [],
        "acceptance_checks": [],
        "candidate_diff_path": str(candidate_diff_path),
        "run_manifest_path": str(run_manifest_path),
        "trials_path": str(trials_path),
        "main_worktree_modified": False,
        "merge_ready": False,
        "human_gate": "review_candidate_before_merge",
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "max_patch_attempts": max(1, min(int(max_patch_attempts), 4)),
        "attempts": [],
    }
    if not work_order_payload:
        base.update({
            "status": "blocked_missing_work_order",
            "message": "No patch or self-upgrade work order is available.",
            "next_safe_command": "evomind patch-order",
        })
        _write_json(artifact_path, base)
        return base

    work_order = _work_order_body(work_order_payload)
    if _work_order_is_external_only(work_order_payload, work_order):
        base.update({
            "status": "blocked_external_gate_not_code",
            "message": "The selected blocker is an external GPU/data/setup gate, not a source-code defect.",
            "next_safe_command": "evomind ready",
            "execution_partition": work_order.get("execution_partition")
            or work_order.get("self_evolution_context", {}).get("execution_partition")
            or {},
        })
        _write_json(artifact_path, base)
        return base

    files_to_edit = [
        _normalize_repo_path(str(item))
        for item in work_order.get("files_to_edit") or []
        if str(item).strip()
    ]
    unsafe_declared = [path for path in files_to_edit if not _path_is_safe(path)]
    if unsafe_declared:
        base.update({
            "status": "blocked_unsafe_work_order_paths",
            "message": "Work order contains paths outside the engineering allowlist.",
            "unsafe_paths": unsafe_declared,
            "next_safe_command": "evomind patch-order",
        })
        _write_json(artifact_path, base)
        return base

    max_attempts = max(1, min(int(max_patch_attempts), 4))
    code_agent: dict[str, Any] = {"status": "not_requested"}
    resolved_patch = _resolve_patch_path(root, patch_path)
    if generate_patch:
        resolved_patch, code_agent = _generate_patch_via_code_agent(
            root=root,
            session=session,
            work_order=work_order,
            dashboard_url=dashboard_url,
            timeout_seconds=timeout_seconds,
        )
    if not resolved_patch:
        base.update({
            "status": "blocked_missing_patch",
            "message": "No completed Code Agent patch is available for isolated validation.",
            "code_agent": code_agent,
            "next_safe_command": "evomind engineer --generate",
        })
        _write_json(artifact_path, base)
        return base

    artifact_root.mkdir(parents=True, exist_ok=True)
    main_head_before = _git(root, "rev-parse", "HEAD").stdout.strip()
    main_hashes_before = {path: _sha256(root / path) for path in files_to_edit}
    attempts: list[dict[str, Any]] = []
    acceptance_results: list[dict[str, Any]] = []
    changed_files: list[str] = []
    patch_applied = False
    cleanup_ok = False
    status = "failed_rolled_back"
    message = ""
    candidate_diff = ""
    patch_normalized = False
    normalized_patch_path = artifact_root / "normalized_input.diff"
    repair_evidence: dict[str, Any] | None = None

    for attempt_number in range(1, max_attempts + 1):
        if attempt_number > 1:
            if not generate_patch:
                break
            repair_patch, code_agent = _generate_patch_via_code_agent(
                root=root,
                session=session,
                work_order=work_order,
                dashboard_url=dashboard_url,
                timeout_seconds=timeout_seconds,
                repair_context=repair_evidence,
            )
            if not repair_patch:
                message = "Code Agent did not return a repair patch after isolated validation failed."
                break
            resolved_patch = repair_patch

        attempt_root = artifact_root / f"attempt_{attempt_number:02d}"
        attempt_root.mkdir(parents=True, exist_ok=True)
        normalized_patch_path = attempt_root / "normalized_input.diff"
        attempt_candidate_diff_path = attempt_root / "candidate.diff"
        patch_text = resolved_patch.read_text(encoding="utf-8", errors="replace")
        normalized_patch_text, patch_normalized = _repair_unified_diff_counts(patch_text)
        normalized_patch_path.write_text(normalized_patch_text, encoding="utf-8")
        changed_files = _patch_changed_files(normalized_patch_text)
        for path in changed_files:
            main_hashes_before.setdefault(path, _sha256(root / path))
        unsafe_patch_paths = [path for path in changed_files if not _path_is_safe(path)]
        outside_work_order = [path for path in changed_files if files_to_edit and path not in files_to_edit]
        if not normalized_patch_text.strip() or not changed_files or unsafe_patch_paths or outside_work_order:
            status = "blocked_patch_scope_violation"
            message = "Patch failed path/scope validation."
            attempt_payload = {
                "attempt": attempt_number,
                "status": status,
                "message": message,
                "patch_path": str(resolved_patch),
                "normalized_patch_path": str(normalized_patch_path),
                "changed_files": changed_files,
                "unsafe_paths": unsafe_patch_paths,
                "outside_work_order": outside_work_order,
                "code_agent": code_agent,
                "main_worktree_modified": False,
                "rollback_verified": True,
                "manifest_path": str(attempt_root / "manifest.json"),
            }
            _write_json(attempt_root / "manifest.json", attempt_payload)
            attempts.append(attempt_payload)
            break

        temp_parent = Path(tempfile.mkdtemp(prefix="evomind-engineering-"))
        worktree = temp_parent / "worktree"
        acceptance_results = []
        attempt_patch_applied = False
        attempt_cleanup_ok = False
        candidate_diff = ""
        try:
            added = _git(root, "worktree", "add", "--detach", str(worktree), "HEAD", timeout=180)
            if added.returncode != 0:
                raise RuntimeError(f"git worktree add failed: {_safe_text(added.stdout, limit=1000)}")
            checked = _git(worktree, "apply", "--recount", "--check", str(normalized_patch_path), timeout=120)
            if checked.returncode != 0:
                raise RuntimeError(f"git apply --recount --check failed: {_safe_text(checked.stdout, limit=1500)}")
            applied = _git(worktree, "apply", "--recount", str(normalized_patch_path), timeout=120)
            if applied.returncode != 0:
                raise RuntimeError(f"git apply failed: {_safe_text(applied.stdout, limit=1500)}")
            attempt_patch_applied = True
            actual_changed = [
                _normalize_repo_path(line)
                for line in _git(worktree, "diff", "--name-only").stdout.splitlines()
                if line.strip()
            ]
            unexpected = [path for path in actual_changed if path not in changed_files]
            if unexpected:
                raise RuntimeError(f"Applied patch changed unexpected files: {unexpected}")
            commands = [str(item) for item in work_order.get("acceptance_checks") or [] if str(item).strip()]
            acceptance_results, all_passed = _run_acceptance_checks(
                worktree=worktree,
                commands=commands or ["git diff --check"],
                logs_dir=attempt_root / "checks",
            )
            candidate_diff = _git(worktree, "diff", "--binary", "--").stdout
            attempt_candidate_diff_path.write_text(candidate_diff, encoding="utf-8")
            candidate_diff_path.write_text(candidate_diff, encoding="utf-8")
            if all_passed:
                status = "passed_review_candidate"
                message = "Patch passed isolated worktree checks and is ready for human review; it was not merged."
            else:
                status = "failed_rolled_back"
                message = "Patch failed one or more isolated acceptance checks; the temporary worktree is discarded."
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            status = "failed_rolled_back"
            message = _safe_text(exc, limit=1600)
        finally:
            if worktree.exists():
                removed = _git(root, "worktree", "remove", "--force", str(worktree), timeout=180)
                attempt_cleanup_ok = removed.returncode == 0
            _git(root, "worktree", "prune", timeout=60)
            shutil.rmtree(temp_parent, ignore_errors=True)

        attempt_main_head_after = _git(root, "rev-parse", "HEAD").stdout.strip()
        attempt_main_hashes_after = {path: _sha256(root / path) for path in main_hashes_before}
        attempt_main_unchanged = (
            main_head_before == attempt_main_head_after
            and main_hashes_before == attempt_main_hashes_after
        )
        if not attempt_main_unchanged:
            status = "failed_main_worktree_changed"
            message = "Engineering attempt changed the main worktree; automatic repair stopped."
        patch_applied = attempt_patch_applied
        cleanup_ok = attempt_cleanup_ok
        attempt_payload = {
            "attempt": attempt_number,
            "status": status,
            "message": message,
            "patch_path": str(resolved_patch),
            "patch_sha256": _sha256(resolved_patch),
            "normalized_patch_path": str(normalized_patch_path),
            "changed_files": changed_files,
            "patch_applied_in_isolated_worktree": attempt_patch_applied,
            "acceptance_checks": acceptance_results,
            "candidate_diff_path": str(attempt_candidate_diff_path),
            "cleanup_ok": attempt_cleanup_ok,
            "code_agent": code_agent,
            "main_head_before": main_head_before,
            "main_head_after": attempt_main_head_after,
            "main_file_hashes_before": main_hashes_before,
            "main_file_hashes_after": attempt_main_hashes_after,
            "main_worktree_modified": not attempt_main_unchanged,
            "rollback_verified": attempt_main_unchanged,
            "manifest_path": str(attempt_root / "manifest.json"),
        }
        _write_json(attempt_root / "manifest.json", attempt_payload)
        attempts.append(attempt_payload)
        if status == "passed_review_candidate":
            candidate_diff_path.write_text(candidate_diff, encoding="utf-8")
            break
        if not attempt_main_unchanged:
            break
        repair_evidence = _repair_context(
            attempt=attempt_number,
            changed_files=changed_files,
            candidate_diff=candidate_diff,
            acceptance_results=acceptance_results,
            message=message,
        )

    main_head_after = _git(root, "rev-parse", "HEAD").stdout.strip()
    main_hashes_after = {path: _sha256(root / path) for path in main_hashes_before}
    main_unchanged = main_head_before == main_head_after and main_hashes_before == main_hashes_after
    payload = {
        **base,
        "ok": status == "passed_review_candidate" and main_unchanged,
        "status": status if main_unchanged else "failed_main_worktree_changed",
        "message": message,
        "work_order": {
            "id": work_order.get("work_order_id") or work_order.get("issue_id"),
            "title": work_order.get("title"),
            "files_to_edit": files_to_edit,
            "rollback_condition": work_order.get("rollback_condition"),
            "human_gate": work_order.get("human_gate") or "review_patch_before_merge",
        },
        "patch_path": str(resolved_patch),
        "patch_sha256": _sha256(resolved_patch) if resolved_patch else "missing",
        "normalized_patch_path": str(normalized_patch_path),
        "normalized_patch_sha256": _sha256(normalized_patch_path),
        "patch_normalized": patch_normalized,
        "changed_files": changed_files,
        "unsafe_paths": attempts[-1].get("unsafe_paths", []) if attempts else [],
        "outside_work_order": attempts[-1].get("outside_work_order", []) if attempts else [],
        "patch_applied_in_isolated_worktree": patch_applied,
        "acceptance_checks": acceptance_results,
        "candidate_diff_path": str(candidate_diff_path),
        "candidate_diff_sha256": hashlib.sha256(candidate_diff.encode("utf-8")).hexdigest() if candidate_diff else "",
        "code_agent": code_agent,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "repair_attempted": len(attempts) > 1,
        "cleanup_ok": cleanup_ok,
        "main_head_before": main_head_before,
        "main_head_after": main_head_after,
        "main_file_hashes_before": main_hashes_before,
        "main_file_hashes_after": main_hashes_after,
        "main_worktree_modified": not main_unchanged,
        "merge_ready": status == "passed_review_candidate" and main_unchanged,
        "next_safe_command": (
            "review candidate diff before merge"
            if status == "passed_review_candidate"
            else "evomind engineer --generate" if generate_patch else "evomind patch-order"
        ),
        "epistemic_status": (
            "validated_in_isolated_worktree_not_merged"
            if status == "passed_review_candidate"
            else "failed_validation_not_applied"
        ),
    }
    _write_json(run_manifest_path, payload)
    _write_json(artifact_path, payload)
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    with trials_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "generated_at": generated_at,
            "run_id": run_id,
            "status": payload["status"],
            "work_order_id": payload["work_order"]["id"],
            "changed_files": changed_files,
            "acceptance_passed": all(item.get("passed") for item in acceptance_results) if acceptance_results else False,
            "main_worktree_modified": payload["main_worktree_modified"],
            "candidate_diff_path": str(candidate_diff_path),
            "attempt_count": len(attempts),
            "repair_attempted": len(attempts) > 1,
            "epistemic_status": payload["epistemic_status"],
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        }, ensure_ascii=False) + "\n")
    return payload
