from __future__ import annotations

import base64
import csv
import difflib
import hashlib
import inspect
import itertools
import json
import math
import mimetypes
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import threading
import textwrap
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

from .assistant_runs import (
    _derive_office_preview,
    _media_type,
    _preview_kind,
    _safe_name,
    _sha256_file,
    _snapshot_artifact_file,
    public_artifact,
)
from .models import ToolResult, ToolSpec, new_id, utc_now

Handler = Callable[[dict[str, Any], "ToolContext"], ToolResult]

ALLOWED_PROCESS_EXECUTABLES = {
    "git", "git.exe", "node", "node.exe", "python", "python.exe", "python3", "python3.exe",
    "pytest", "pytest.exe", "rg", "rg.exe", "uv", "uv.exe",
    "powershell", "powershell.exe", "pwsh", "pwsh.exe", "cmd", "cmd.exe",
}
ALLOWED_PYTHON_MODULES = {"evomind_runtime.cli", "pip", "pytest", "venv", "xsci.kaggle"}
HPC_PROFILE_LOCK = threading.RLock()
MEMORY_WRITEBACK_LOCK = threading.RLock()
TRAINING_HPC_THRESHOLDS = {
    "rows": 20_000,
    "features": 128,
    "bytes": 8 * 1024 * 1024,
}
HPC_ASSET_PROBE_SCOPE = "cure_mindgames"
HPC_ASSET_PROBE_PACKAGE_NAMES = ("textarena", "trueskill", "transformers", "torch")


@dataclass
class ToolContext:
    session_id: str
    workspace_root: Path
    project_root: Path
    runtime_root: Path
    artifact_root: Path
    store: Any
    metadata: dict[str, Any]
    super_agent: Any = None
    approval_verified: bool = False
    approval_id: str = ""
    approval_fingerprint: str = ""
    progress: Any = None
    managed_asset_adapter: bool = False
    managed_work_kind: str = "executing"
    trusted_script_path: Path | None = None
    execution_deadline: float | None = None
    remote_settlement_uncertain: bool = False
    reports: Any = None


_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_COMPETITION_RECEIPT_PROJECTION_FIELDS = (
    "schema",
    "competition",
    "status",
    "files",
    "bytes",
    "manifest_sha256",
    "archive_sha256",
    "worker_alive",
    "worker_exit_code",
    "failure_code",
    "partial_archive_bytes",
    "expected_file_count",
    "attempt_id",
    "phase",
    "remote_file_set_sha256",
    "skipped_directory_count",
    "adapter_listing_status",
    "adapter_listing_skipped_directories",
    "adapter_listing_timeout_exhaustions",
    "secret_values_logged",
)


def _public_result_artifact(row: dict[str, Any]) -> dict[str, Any]:
    """Project an oversized tool-result artifact without leaking its path."""

    return {
        "id": str(row.get("id") or ""),
        "sha256": str(row.get("sha256") or ""),
        "media_type": str(row.get("media_type") or "application/octet-stream"),
        "bytes": int(row.get("bytes") or 0),
        "created_at": str(row.get("created_at") or ""),
        "artifact_kind": "tool_result",
    }


def _structured_json_projection(value: Any) -> dict[str, Any]:
    """Return a bounded, valid JSON projection for an oversized tool result."""

    if not isinstance(value, dict):
        return {
            "value_type": "array" if isinstance(value, list) else type(value).__name__,
            "item_count": len(value) if isinstance(value, (dict, list)) else None,
        }
    if value.get("schema") != "evomind.competition_data_tool_result.v1":
        keys = sorted(str(key) for key in value)[:128]
        return {
            "value_type": "object",
            "key_count": len(value),
            "keys": keys,
            "keys_truncated": len(value) > len(keys),
        }

    receipt = value.get("receipt")
    if not isinstance(receipt, dict):
        receipt = {}
    projected_receipt = {
        key: receipt.get(key)
        for key in _COMPETITION_RECEIPT_PROJECTION_FIELDS
        if key in receipt
    }
    smoke = receipt.get("loader_smoke")
    smoke_items = tuple(smoke.values()) if isinstance(smoke, dict) else ()
    manifest_sha = str(receipt.get("manifest_sha256") or "")
    archive_sha = str(receipt.get("archive_sha256") or "")
    try:
        file_count = int(receipt.get("files") or 0)
    except (TypeError, ValueError):
        file_count = 0
    projected_receipt.update(
        {
            "loader_smoke_count": len(smoke_items),
            "loader_smoke_all_ok": bool(smoke_items)
            and all(isinstance(item, dict) and item.get("ok") is True for item in smoke_items),
            "manifest_sha_valid": _SHA256_PATTERN.fullmatch(manifest_sha) is not None,
            "archive_sha_valid": _SHA256_PATTERN.fullmatch(archive_sha) is not None,
            "file_sha_manifest_bound": bool(
                receipt.get("status") == "FULL_DATA_READY"
                and file_count > 0
                and _SHA256_PATTERN.fullmatch(manifest_sha) is not None
            ),
        }
    )
    return {
        "schema": str(value.get("schema") or ""),
        "action": str(value.get("action") or ""),
        "competition": str(value.get("competition") or ""),
        "status": str(value.get("status") or ""),
        "receipt": projected_receipt,
        "hpc_identity_gate": str(value.get("hpc_identity_gate") or ""),
        "local_fallback_used": value.get("local_fallback_used") is True,
        "secret_values_logged": value.get("secret_values_logged") is True,
    }


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Handler] = {}

    def register(self, spec: ToolSpec, handler: Handler) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool: {spec.name}")
        Draft202012Validator.check_schema(spec.input_schema)
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def specs(self) -> list[ToolSpec]:
        return [self._specs[name] for name in sorted(self._specs)]

    def specs_for_prompt(self, prompt: str) -> list[ToolSpec]:
        """Load stable base tools plus only the domain schemas relevant to this run."""
        folded = str(prompt or "").casefold()
        base = {
            "file_list", "file_search", "file_read", "file_write", "file_patch", "file_copy", "file_move", "file_delete",
            "shell_exec", "process_start", "process_list", "process_poll", "process_log", "process_stdin", "process_cancel",
            "attachment_list", "attachment_read", "artifact_publish", "artifact_list", "artifact_preview", "artifact_bundle",
            "runtime_health", "research_capabilities", "verified_context", "skill_list", "training_route",
            "super_agent_status", "capability_discover", "connector_health",
            "ecosystem_refresh",
            "directory_list", "directory_stat", "directory_read", "directory_hash",
            "directory_mkdir", "directory_write_atomic", "directory_copy", "directory_sync", "directory_delete",
            "tool_synthesize", "tool_test", "tool_canary", "tool_promote", "tool_reject",
            "transfer_fetch", "job_execute", "job_status", "job_cancel",
        }
        domains = {
            "literature": ("论文", "文献", "doi", "引用", "citation", "paper", "literature", "arxiv", "openalex", "crossref"),
            "kaggle": ("kaggle", "competition", "比赛", "竞赛", "submission", "提交"),
            "hpc": ("hpc", "gpu", "cuda", "显卡", "训练", "train", "finetune", "微调"),
            "memory": ("记忆", "memory", "经验", "lesson"),
            "evolution": ("进化", "evolution", "self-improv", "自进化", "反思"),
            "report": ("报告", "report", "pdf", "docx", "xlsx", "pptx", "图表", "chart"),
            "browser": ("网页", "浏览器", "browser", "website", "页面", "chrome"),
            "desktop": ("桌面", "窗口", "desktop", "click", "截图", "screenshot"),
            "mcp": ("mcp", "connector", "连接器", "plugin", "插件"),
            "competition": (
                "competition data", "dataset download", "比赛数据", "竞赛数据", "下载数据",
                "六赛", "六个比赛", "六项比赛", "六个竞赛", "六项竞赛",
                "cure-bench", "curebench", "e2lmc", "mindgames", "mind games", "ariel",
                "weather4cast", "open polymer",
            ),
        }
        selected_domains = {domain for domain, terms in domains.items() if any(term in folded for term in terms)}
        for name, spec in self._specs.items():
            if name in base:
                continue
            if name == "memory_writeback" and {"memory", "evolution"} & selected_domains:
                base.add(name)
                continue
            domain = name.split("_", 1)[0]
            if name.startswith("literature_") or name.startswith("citation_"):
                domain = "literature"
            elif name.startswith("capability_"):
                domain = "evolution"
            elif name.startswith("competition_data_"):
                domain = "competition"
            if domain in selected_domains:
                base.add(name)
        return [self._specs[name] for name in sorted(base) if name in self._specs and self._specs[name].available]

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def invoke(self, name: str, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult("", False, {}, f"unknown tool {name}", error="unknown_tool")
        if not spec.available:
            return ToolResult("", False, {"available": False}, spec.unavailable_reason, error="capability_unavailable")
        errors = sorted(Draft202012Validator(spec.input_schema).iter_errors(arguments), key=lambda item: list(item.path))
        if errors:
            return ToolResult("", False, {}, "invalid tool arguments", error="; ".join(error.message for error in errors[:8]))
        try:
            if "official_calibration" in context.metadata:
                from .ev_calibration_control import guard_tool
                guard_tool(name, arguments, context)
            if "siim_calibration" in context.metadata:
                from .siim_calibration_control import guard_tool as guard_siim
                guard_siim(name, arguments, context)
            result = self._handlers[name](arguments, context)
        except Exception as exc:  # tool failures become observations
            return ToolResult("", False, {}, f"{name} failed", error=f"{type(exc).__name__}: {exc}")
        encoded = json.dumps(result.content, ensure_ascii=False, default=str).encode("utf-8")
        if len(encoded) > spec.max_result_bytes:
            projection = _structured_json_projection(result.content)
            artifact = context.store.add_artifact(context.session_id, encoded, ".json", "application/json", context.artifact_root)
            public_artifact_row = _public_result_artifact(artifact)
            result.content = {
                "truncated": True,
                "preview_complete": False,
                "preview_encoding": "structured_json_projection",
                "preview": json.dumps(projection, ensure_ascii=False, separators=(",", ":")),
                "content_bytes": len(encoded),
                "content_sha256": str(artifact["sha256"]),
                "artifact": public_artifact_row,
            }
            result.artifacts.append(public_artifact_row)
        return result


def _path(args: dict[str, Any], key: str, context: ToolContext) -> Path:
    value = str(args.get(key) or "")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = context.workspace_root / candidate
    root = context.workspace_root.resolve(strict=False)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{key} escapes the configured workspace") from exc
    return resolved


def _workspace_path(value: str, context: ToolContext, *, label: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = context.workspace_root / candidate
    root = context.workspace_root.resolve(strict=False)
    resolved = candidate.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the configured workspace") from exc
    if not relative.parts:
        raise ValueError(f"{label} must be below the configured workspace")
    return resolved


def _guard_workspace_python_module(
    module: str,
    module_args: list[str],
    executable: Path,
    context: ToolContext,
) -> None:
    if module == "venv":
        if len(module_args) != 1 or module_args[0].startswith("-"):
            raise ValueError("venv requires one workspace-local destination and no mutation flags")
        destination = _workspace_path(module_args[0], context, label="venv destination")
        if destination.is_symlink():
            raise ValueError("venv destination cannot be a symbolic link")
        return
    if module != "pip":
        return

    workspace = context.workspace_root.resolve(strict=False)
    try:
        executable.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("pip is allowed only through a workspace-local virtual environment") from exc
    venv_root = next(
        (
            parent
            for parent in executable.parents
            if parent != workspace
            and (parent / "pyvenv.cfg").is_file()
            and not parent.is_symlink()
            and parent.is_relative_to(workspace)
        ),
        None,
    )
    if venv_root is None:
        raise ValueError("pip interpreter is not bound to a verified workspace virtual environment")
    if not module_args:
        raise ValueError("pip requires an allowlisted subcommand")

    command = module_args[0].casefold()
    if command == "install":
        safe_flags = {
            "--disable-pip-version-check", "--no-input", "--no-deps",
            "--prefer-binary", "--quiet", "-q",
        }
        requirements = []
        for value in module_args[1:]:
            lowered = value.casefold()
            if lowered in safe_flags:
                continue
            bounded_network_option = re.fullmatch(r"--(timeout|retries|resume-retries)=([0-9]{1,3})", lowered)
            if bounded_network_option:
                option, raw_limit = bounded_network_option.groups()
                limit = int(raw_limit)
                maximum = 300 if option == "timeout" else 10
                if 1 <= limit <= maximum:
                    continue
                raise ValueError(f"pip install network option is outside the allowed range: {value}")
            if lowered.startswith("-"):
                raise ValueError(f"pip install option is not allowlisted: {value}")
            if "://" in lowered or lowered.startswith(("git+", "hg+", "svn+", "bzr+")):
                raise ValueError("pip direct URL and VCS requirements are not allowed")
            if not re.fullmatch(r"[A-Za-z0-9_.\-\[\],<>=!~]+", value):
                raise ValueError(f"pip requirement is not allowlisted: {value}")
            requirements.append(value)
        if not requirements:
            raise ValueError("pip install requires at least one package requirement")
        return
    if command == "freeze" and all(value in {"--all", "--exclude-editable", "--local"} for value in module_args[1:]):
        return
    if command == "check" and len(module_args) == 1:
        return
    if command == "show" and all(re.fullmatch(r"[A-Za-z0-9_.-]+", value) for value in module_args[1:]):
        return
    if command == "list" and all(value in {"--format=json", "--not-required", "--uptodate", "--outdated"} for value in module_args[1:]):
        return
    raise ValueError(f"pip subcommand or arguments are not allowlisted: {command}")


def _trusted_argv(args: dict[str, Any], context: ToolContext, key: str = "argv") -> list[str]:
    raw = args.get(key)
    if not isinstance(raw, list) or not raw or len(raw) > 128:
        raise ValueError(f"{key} must be a non-empty bounded argument array")
    argv = [str(value) for value in raw]
    if any(not value or "\x00" in value or len(value) > 8192 for value in argv):
        raise ValueError(f"{key} contains an invalid argument")
    if sum(len(value) for value in argv) > 65536:
        raise ValueError(f"{key} exceeds the command-size limit")

    requested = argv[0]
    basename = Path(requested).name.lower()
    if basename not in ALLOWED_PROCESS_EXECUTABLES:
        raise ValueError(f"executable is not allowlisted: {basename or '<empty>'}")

    python_aliases = {"python", "python.exe", "python3", "python3.exe"}
    if basename in python_aliases and not Path(requested).is_absolute():
        requested_path = Path(requested).expanduser()
        if requested_path.parent != Path("."):
            executable = _workspace_path(requested, context, label="Python executable")
        else:
            configured = os.environ.get("WORKSTATION_PYTHON", "").strip()
            executable = Path(configured).expanduser().resolve(strict=False) if configured else Path(sys.executable).resolve()
    else:
        located = requested if Path(requested).is_absolute() else shutil.which(requested)
        if not located:
            raise ValueError(f"allowlisted executable is unavailable: {basename}")
        executable = Path(located).expanduser().resolve(strict=False)
    if not executable.is_file() or executable.name.lower() not in ALLOWED_PROCESS_EXECUTABLES:
        raise ValueError("resolved executable is not a trusted allowlisted file")

    lowered = [value.lower() for value in argv[1:]]
    if basename in python_aliases:
        if "-c" in lowered:
            raise ValueError("inline Python code is not allowed; execute a workspace script or allowlisted module")
        if "-m" in lowered:
            index = lowered.index("-m")
            module = argv[index + 2] if index + 2 < len(argv) else ""
            if module not in ALLOWED_PYTHON_MODULES:
                raise ValueError(f"Python module is not allowlisted: {module or '<missing>'}")
            _guard_workspace_python_module(module, argv[index + 3 :], executable, context)
    if basename in {"node", "node.exe"} and any(value in {"-e", "--eval", "-p", "--print"} for value in lowered):
        raise ValueError("inline Node.js code is not allowed; execute a workspace script")
    if basename in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"} and not any(
        value in {"-command", "-file"} for value in lowered
    ):
        raise ValueError("PowerShell invocation is not allowlisted without an explicit -Command or -File argument")
    if basename in {"cmd", "cmd.exe"} and not any(value in {"/c", "/k"} for value in lowered):
        raise ValueError("cmd invocation is not allowlisted without an explicit /c or /k argument")
    if basename in {"git", "git.exe"} and any(value == "-c" or value.startswith("--exec-path") for value in lowered):
        raise ValueError("Git configuration and executable overrides are not allowed")

    # Script entrypoints must remain inside the workspace. Options and ordinary
    # scalar arguments are left to the invoked program's own parser.
    if basename in python_aliases | {"node", "node.exe"}:
        for value in argv[1:]:
            candidate = Path(value).expanduser()
            if candidate.suffix.lower() not in {".py", ".js", ".mjs", ".cjs"}:
                continue
            resolved = candidate if candidate.is_absolute() else context.workspace_root / candidate
            try:
                resolved.resolve(strict=False).relative_to(context.workspace_root.resolve(strict=False))
            except ValueError as exc:
                raise ValueError("script entrypoint escapes the configured workspace") from exc
    return [str(executable), *argv[1:]]


def _safe_glob(value: str) -> str:
    pattern = value or "*"
    if "\x00" in pattern or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError("glob pattern escapes the configured search root")
    return pattern


def _assert_tree_contained(root: Path, workspace_root: Path) -> None:
    boundary = workspace_root.resolve(strict=False)
    for entry in itertools.chain((root,), root.rglob("*")):
        attributes = getattr(entry.lstat(), "st_file_attributes", 0)
        if entry.is_symlink() or attributes & 0x400:
            raise ValueError(f"source tree contains a symbolic link or reparse point: {entry.name}")
        try:
            entry.resolve(strict=False).relative_to(boundary)
        except ValueError as exc:
            raise ValueError("source tree escapes the configured workspace") from exc


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _backup(path: Path, context: ToolContext) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = path.read_bytes()
    suffix = path.suffix or ".bin"
    media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return context.store.add_artifact(context.session_id, data, suffix, media, context.artifact_root)


def _file_list(args: dict[str, Any], context: ToolContext) -> ToolResult:
    root = _path(args, "path", context)
    pattern = _safe_glob(str(args.get("glob") or "*"))
    recursive = bool(args.get("recursive", False))
    limit = max(1, min(int(args.get("limit", 500)), 5000))
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    rows = []
    for path in iterator:
        try:
            stat = path.stat()
            rows.append({"path": str(path), "kind": "directory" if path.is_dir() else "file", "bytes": stat.st_size, "modified": stat.st_mtime})
        except OSError:
            continue
        if len(rows) >= limit:
            break
    return ToolResult("", True, {"root": str(root), "entries": rows, "truncated": len(rows) >= limit}, f"listed {len(rows)} entries")


def _file_search(args: dict[str, Any], context: ToolContext) -> ToolResult:
    root = _path(args, "path", context)
    query = str(args["query"])
    glob = _safe_glob(str(args.get("glob") or "*"))
    limit = max(1, min(int(args.get("limit", 200)), 2000))
    rg = shutil.which("rg")
    if rg:
        command = [
            rg,
            "--line-number",
            "--column",
            "--no-heading",
            "--no-messages",
            "--color",
            "never",
            "--fixed-strings",
            "--max-filesize",
            "8M",
            "--max-depth",
            "12",
            "-g",
            glob,
            "-g",
            "!**/node_modules/**",
            "-g",
            "!**/.next/**",
            "-g",
            "!**/.git/**",
            "-g",
            "!**/.claude/worktrees/**",
            "-e",
            query,
            "--",
            str(root),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", "replace")
            lines = str(stdout).splitlines()
            return ToolResult(
                "",
                True,
                {
                    "matches": lines[:limit],
                    "match_count": len(lines),
                    "truncated": True,
                    "timed_out": True,
                    "root": str(root),
                },
                f"search reached the time limit; returned {len(lines[:limit])} partial matches",
            )
        lines = completed.stdout.splitlines()
        return ToolResult("", completed.returncode in {0, 1}, {"matches": lines[:limit], "match_count": len(lines), "truncated": len(lines) > limit}, f"found {len(lines)} matches", error=completed.stderr[:2000])
    matches = []
    for path in root.rglob(glob):
        if not path.is_file():
            continue
        try:
            for index, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if query in line:
                    matches.append(f"{path}:{index}:{line[:500]}")
                    if len(matches) >= limit:
                        return ToolResult("", True, {"matches": matches, "truncated": True}, f"found at least {limit} matches")
        except OSError:
            continue
    return ToolResult("", True, {"matches": matches, "truncated": False}, f"found {len(matches)} matches")


def _file_read(args: dict[str, Any], context: ToolContext) -> ToolResult:
    path = _path(args, "path", context)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if b"\x00" in data[:4096]:
        return ToolResult("", True, {"path": str(path), "bytes": len(data), "sha256": digest, "base64_preview": base64.b64encode(data[:4096]).decode("ascii"), "binary": True}, f"read binary file {path.name}")
    text = data.decode(str(args.get("encoding") or "utf-8"), errors="replace")
    start = max(1, int(args.get("start_line", 1)))
    end = max(start, int(args.get("end_line", start + 399)))
    lines = text.splitlines()
    selected = [{"line": index, "text": lines[index - 1]} for index in range(start, min(end, len(lines)) + 1)]
    return ToolResult("", True, {"path": str(path), "bytes": len(data), "sha256": digest, "lines": selected, "line_count": len(lines), "ends_with_newline": text.endswith(("\n", "\r"))}, f"read {len(selected)} lines from {path.name}")


def _file_write(args: dict[str, Any], context: ToolContext) -> ToolResult:
    path = _path(args, "path", context)
    expected = str(args.get("expected_sha256") or "")
    current = path.read_bytes() if path.exists() else b""
    current_hash = hashlib.sha256(current).hexdigest() if path.exists() else ""
    if expected and expected != current_hash:
        return ToolResult("", False, {"current_sha256": current_hash}, "preimage hash mismatch", error="preimage_mismatch")
    backup = _backup(path, context)
    content = str(args.get("content") or "")
    data = content.encode(str(args.get("encoding") or "utf-8"))
    _atomic_write(path, data)
    artifact = context.store.add_artifact(context.session_id, data, path.suffix or ".txt", mimetypes.guess_type(path.name)[0] or "text/plain", context.artifact_root)
    return ToolResult("", True, {"path": str(path), "sha256": artifact["sha256"], "bytes": len(data), "backup": backup}, f"wrote {len(data)} bytes to {path.name}", artifacts=[artifact, *([backup] if backup else [])])


def _file_patch(args: dict[str, Any], context: ToolContext) -> ToolResult:
    path = _path(args, "path", context)
    old = path.read_text(encoding="utf-8", errors="strict")
    expected = str(args.get("expected_sha256") or "")
    current_hash = hashlib.sha256(old.encode("utf-8")).hexdigest()
    if expected and expected != current_hash:
        return ToolResult("", False, {"current_sha256": current_hash}, "preimage hash mismatch", error="preimage_mismatch")
    find = str(args["find"])
    replace = str(args.get("replace") or "")
    count = old.count(find)
    expected_count = int(args.get("expected_count", 1))
    if count != expected_count:
        return ToolResult("", False, {"actual_count": count, "expected_count": expected_count}, "patch match count mismatch", error="match_count_mismatch")
    new = old.replace(find, replace, expected_count)
    backup = _backup(path, context)
    _atomic_write(path, new.encode("utf-8"))
    diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=str(path), tofile=str(path)))
    return ToolResult("", True, {"path": str(path), "sha256": hashlib.sha256(new.encode("utf-8")).hexdigest(), "diff": diff, "backup": backup}, f"patched {path.name}", artifacts=[backup] if backup else [])


def _file_copy(args: dict[str, Any], context: ToolContext) -> ToolResult:
    source, destination = _path(args, "source", context), _path(args, "destination", context)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = _backup(destination, context)
    if source.is_dir():
        _assert_tree_contained(source, context.workspace_root)
        shutil.copytree(source, destination, dirs_exist_ok=bool(args.get("overwrite", False)))
    else:
        shutil.copy2(source, destination)
    return ToolResult("", True, {"source": str(source), "destination": str(destination), "backup": backup}, f"copied {source.name}")


def _file_move(args: dict[str, Any], context: ToolContext) -> ToolResult:
    source, destination = _path(args, "source", context), _path(args, "destination", context)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = _backup(destination, context)
    if source.is_dir():
        _assert_tree_contained(source, context.workspace_root)
    shutil.move(str(source), str(destination))
    return ToolResult("", True, {"source": str(source), "destination": str(destination), "backup": backup}, f"moved {source.name}")


def _file_delete(args: dict[str, Any], context: ToolContext) -> ToolResult:
    path = _path(args, "path", context)
    if path.is_dir() and any(path.iterdir()) and not bool(args.get("recursive", False)):
        return ToolResult("", False, {"path": str(path)}, "directory is not empty", error="recursive_required")
    backup = _backup(path, context)
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return ToolResult("", True, {"path": str(path), "backup": backup}, f"deleted {path.name}", artifacts=[backup] if backup else [])


@dataclass
class ManagedProcess:
    id: str
    session_id: str
    process: subprocess.Popen[bytes]
    argv: tuple[str, ...]
    cwd: str
    log_path: Path
    stdin_enabled: bool
    created_at: float
    log_stream: Any
    log_thread: threading.Thread
    log_closed: bool = False


def _redact_process_log(value: str) -> str:
    redacted = re.sub(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", "[credential redacted]", value, flags=re.I)
    redacted = re.sub(
        r"\b([a-z][a-z0-9+.-]*://)[^\s/@:]+:[^\s/@]+@",
        r"\1[credential redacted]@",
        redacted,
        flags=re.I,
    )
    redacted = re.sub(
        r"\b(api[_-]?key|token|access[_-]?token|refresh[_-]?token|session[_-]?token|password|passwd|secret|cookie|authorization)\s*[:=]\s*([^\s,;]+)",
        r"\1=[redacted]",
        redacted,
        flags=re.I,
    )
    redacted = re.sub(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b", "[jwt redacted]", redacted)
    redacted = re.sub(
        r"\b[A-Za-z]:[\\/][^\r\n]*?(?:\.ssh|credentials?|secrets?|profiles?)[^\s\r\n]*",
        "[credential path redacted]",
        redacted,
        flags=re.I,
    )
    redacted = re.sub(r"\bdpapi\s*[:=]\s*[A-Za-z0-9+/=_-]{16,}", "dpapi=[redacted]", redacted, flags=re.I)
    return redacted


class ProcessManager:
    def __init__(self) -> None:
        self._items: dict[str, ManagedProcess] = {}
        self._lock = threading.RLock()
        self._start_lock = threading.Lock()

    def start(self, session_id: str, argv: list[str], cwd: Path, log_path: Path) -> ManagedProcess:
        # Serialize the equivalence check and Popen creation so concurrent
        # continuations cannot race past the de-duplication boundary.
        with self._start_lock:
            return self._start_serialized(session_id, argv, cwd, log_path)

    def _start_serialized(self, session_id: str, argv: list[str], cwd: Path, log_path: Path) -> ManagedProcess:
        normalized_argv = tuple(argv)
        normalized_cwd = str(cwd.resolve())
        # Model continuations can legitimately repeat the same process_start
        # request after a tool-step pause.  Reuse the exact live process for
        # this run so a continuation cannot launch concurrent installers or
        # trainers against the same workspace.
        with self._lock:
            for existing in self._items.values():
                if (
                    existing.session_id == session_id
                    and existing.argv == normalized_argv
                    and existing.cwd == normalized_cwd
                    and existing.process.poll() is None
                ):
                    return existing
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("ab")
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process_env = os.environ.copy()
        for name in ("WORKSTATION_SESSION_SECRET", "WORKSTATION_BOOTSTRAP_TOKEN_HASH", "HTTP_COOKIE", "COOKIE"):
            process_env.pop(name, None)
        try:
            process = subprocess.Popen(
                argv,
                cwd=str(cwd),
                env=process_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                shell=False,
            )
        except BaseException:
            stream.close()
            raise

        def pump() -> None:
            assert process.stdout is not None
            try:
                while True:
                    chunk = process.stdout.readline(65536)
                    if not chunk:
                        break
                    scrubbed = _redact_process_log(chunk.decode("utf-8", errors="replace")).encode("utf-8")
                    stream.write(scrubbed)
                    stream.flush()
            finally:
                process.stdout.close()
                stream.close()

        log_thread = threading.Thread(target=pump, name=f"evomind-log-{process.pid}", daemon=True)
        log_thread.start()
        item = ManagedProcess(
            f"proc_{process.pid}_{time.time_ns()}",
            session_id,
            process,
            normalized_argv,
            normalized_cwd,
            log_path,
            True,
            time.time(),
            stream,
            log_thread,
        )
        with self._lock:
            self._items[item.id] = item
        return item

    def get(self, process_id: str, session_id: str) -> ManagedProcess:
        with self._lock:
            item = self._items.get(process_id)
            if item is None or item.session_id != session_id:
                raise KeyError(process_id)
            return item

    def list(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            items = [item for item in self._items.values() if item.session_id == session_id]
        for item in items:
            self.close_log_if_exited(item)
        return [{"id": item.id, "pid": item.process.pid, "status": "running" if item.process.poll() is None else "exited", "exit_code": item.process.poll(), "log_path": str(item.log_path), "age_seconds": round(time.time() - item.created_at, 3)} for item in items]

    def close_log_if_exited(self, item: ManagedProcess) -> None:
        if item.process.poll() is not None and not item.log_closed:
            item.log_thread.join(timeout=5)
            if not item.log_thread.is_alive():
                if not item.log_stream.closed:
                    item.log_stream.close()
                item.log_closed = True

    def cancel(self, item: ManagedProcess) -> None:
        if item.process.poll() is not None:
            return
        # Popen retains a handle to the exact process object on Windows, so
        # terminate/kill cannot target a newly reused numeric PID.
        item.process.terminate()
        try:
            item.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            item.process.kill()
            item.process.wait(timeout=5)
        self.close_log_if_exited(item)


PROCESSES = ProcessManager()


def _process_start(args: dict[str, Any], context: ToolContext) -> ToolResult:
    cwd = _path(args, "cwd", context) if args.get("cwd") else context.workspace_root
    log_path = context.runtime_root / "processes" / context.session_id / f"{time.time_ns()}.log"
    item = PROCESSES.start(context.session_id, _trusted_argv(args, context), cwd, log_path)
    reused = item.log_path != log_path
    return ToolResult(
        "",
        True,
        {
            "process_id": item.id,
            "pid": item.process.pid,
            "log_path": str(item.log_path),
            "reused_running_equivalent": reused,
        },
        f"reused running equivalent process {item.process.pid}" if reused else f"started process {item.process.pid}",
    )


def _shell_exec(args: dict[str, Any], context: ToolContext) -> ToolResult:
    item_result = _process_start(args, context)
    item = PROCESSES.get(item_result.content["process_id"], context.session_id)
    timeout = max(1, min(int(args.get("timeout_seconds", 120)), 3600))
    raw_argv = args.get("argv")
    executable = Path(str(raw_argv[0])).name.casefold() if isinstance(raw_argv, list) and raw_argv else ""
    is_rg = executable in {"rg", "rg.exe"}
    try:
        code = item.process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        PROCESSES.cancel(item)
        output = item.log_path.read_text(encoding="utf-8", errors="replace") if item.log_path.exists() else ""
        return ToolResult(
            "",
            is_rg,
            {
                **item_result.content,
                "running": False,
                "timed_out": True,
                "exit_code": item.process.poll(),
                "output": output,
                "partial_result": is_rg,
            },
            "search reached the time limit and returned partial output" if is_rg else "command timed out and was terminated",
            error="" if is_rg else "timeout",
        )
    PROCESSES.close_log_if_exited(item)
    output = item.log_path.read_text(encoding="utf-8", errors="replace") if item.log_path.exists() else ""
    ok = code == 0 or (is_rg and code == 1)
    summary = "search completed with no matches" if is_rg and code == 1 else f"command exited {code}"
    return ToolResult("", ok, {**item_result.content, "exit_code": code, "output": output}, summary, error="" if ok else "nonzero_exit")


def _process_list(_args: dict[str, Any], context: ToolContext) -> ToolResult:
    return ToolResult("", True, {"processes": PROCESSES.list(context.session_id)}, "listed managed processes")


def _process_poll(args: dict[str, Any], context: ToolContext) -> ToolResult:
    item = PROCESSES.get(str(args["process_id"]), context.session_id)
    code = item.process.poll()
    PROCESSES.close_log_if_exited(item)
    return ToolResult("", True, {"process_id": item.id, "running": code is None, "exit_code": code}, "process is running" if code is None else f"process exited {code}")


def _process_log(args: dict[str, Any], context: ToolContext) -> ToolResult:
    item = PROCESSES.get(str(args["process_id"]), context.session_id)
    lines = item.log_path.read_text(encoding="utf-8", errors="replace").splitlines() if item.log_path.exists() else []
    limit = max(1, min(int(args.get("lines", 200)), 5000))
    return ToolResult("", True, {"process_id": item.id, "lines": lines[-limit:], "total_lines": len(lines)}, f"read {min(limit, len(lines))} log lines")


def _process_stdin(args: dict[str, Any], context: ToolContext) -> ToolResult:
    item = PROCESSES.get(str(args["process_id"]), context.session_id)
    if item.process.stdin is None or item.process.poll() is not None:
        return ToolResult("", False, {}, "process stdin unavailable", error="stdin_unavailable")
    data = str(args.get("data") or "").encode("utf-8")
    item.process.stdin.write(data)
    item.process.stdin.flush()
    return ToolResult("", True, {"bytes": len(data)}, f"wrote {len(data)} bytes")


def _process_cancel(args: dict[str, Any], context: ToolContext) -> ToolResult:
    item = PROCESSES.get(str(args["process_id"]), context.session_id)
    PROCESSES.cancel(item)
    return ToolResult("", True, {"process_id": item.id, "exit_code": item.process.poll()}, "process tree terminated")


def _chrome_path() -> str:
    candidates = [
        os.getenv("CHROME_PATH", ""),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    return next((path for path in candidates if path and Path(path).is_file()), "")


def _browser_health(_args: dict[str, Any], _context: ToolContext) -> ToolResult:
    chrome = _chrome_path()
    return ToolResult("", bool(chrome), {"available": bool(chrome), "executable": chrome, "modes": ["open", "headless_dom", "headless_screenshot"] if chrome else []}, "browser adapter ready" if chrome else "Chrome or Edge not found", error="" if chrome else "browser_missing")


def _browser_open(args: dict[str, Any], _context: ToolContext) -> ToolResult:
    chrome = _chrome_path()
    if not chrome:
        return ToolResult("", False, {}, "browser missing", error="browser_missing")
    process = subprocess.Popen([chrome, str(args["url"])], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ToolResult("", True, {"pid": process.pid, "url": str(args["url"])}, "opened URL in browser")


def _browser_dom(args: dict[str, Any], _context: ToolContext) -> ToolResult:
    chrome = _chrome_path()
    if not chrome:
        return ToolResult("", False, {}, "browser missing", error="browser_missing")
    completed = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--dump-dom", str(args["url"])], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=False)
    return ToolResult("", completed.returncode == 0, {"url": str(args["url"]), "dom": completed.stdout}, "captured browser DOM", error=completed.stderr[:2000])


def _desktop_windows(_args: dict[str, Any], _context: ToolContext) -> ToolResult:
    command = "Get-Process | Where-Object {$_.MainWindowHandle -ne 0} | Select-Object Id,ProcessName,MainWindowTitle | ConvertTo-Json -Compress"
    completed = subprocess.run([shutil.which("pwsh") or "powershell.exe", "-NoProfile", "-Command", command], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20, check=False)
    try:
        windows = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
        windows = []
    return ToolResult("", completed.returncode == 0, {"windows": windows}, "listed desktop windows", error=completed.stderr[:1000])


def _desktop_screenshot(_args: dict[str, Any], context: ToolContext) -> ToolResult:
    target = context.runtime_root / "screenshots" / context.session_id / f"{time.time_ns()}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    escaped = str(target).replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; "
        "$b=[System.Windows.Forms.SystemInformation]::VirtualScreen; "
        "$i=New-Object System.Drawing.Bitmap $b.Width,$b.Height; $g=[System.Drawing.Graphics]::FromImage($i); "
        "$g.CopyFromScreen($b.Left,$b.Top,0,0,$i.Size); $i.Save('" + escaped + "',[System.Drawing.Imaging.ImageFormat]::Png); $g.Dispose(); $i.Dispose()"
    )
    completed = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=30, check=False)
    if completed.returncode or not target.exists():
        return ToolResult("", False, {}, "desktop screenshot failed", error=completed.stderr[:2000])
    artifact = context.store.add_artifact(context.session_id, target.read_bytes(), ".png", "image/png", context.artifact_root)
    return ToolResult("", True, {"path": str(target), "artifact": artifact}, "captured desktop screenshot", artifacts=[artifact])


def _desktop_click(args: dict[str, Any], _context: ToolContext) -> ToolResult:
    x, y = int(args["x"]), int(args["y"])
    script = f"Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class M{{[DllImport(\"user32.dll\")] public static extern bool SetCursorPos(int X,int Y); [DllImport(\"user32.dll\")] public static extern void mouse_event(uint f,uint x,uint y,uint d,UIntPtr e);}}'; [M]::SetCursorPos({x},{y}); [M]::mouse_event(2,0,0,0,[UIntPtr]::Zero); [M]::mouse_event(4,0,0,0,[UIntPtr]::Zero)"
    completed = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=15, check=False)
    return ToolResult("", completed.returncode == 0, {"x": x, "y": y}, f"clicked desktop at {x},{y}", error=completed.stderr[:1000])


def _desktop_type(args: dict[str, Any], _context: ToolContext) -> ToolResult:
    encoded = base64.b64encode(str(args["text"]).encode("utf-16le")).decode("ascii")
    script = f"Add-Type -AssemblyName System.Windows.Forms; $t=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('{encoded}')); [System.Windows.Forms.SendKeys]::SendWait($t)"
    completed = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=15, check=False)
    return ToolResult("", completed.returncode == 0, {"characters": len(str(args["text"]))}, "typed desktop text", error=completed.stderr[:1000])


def _skill_list(_args: dict[str, Any], _context: ToolContext) -> ToolResult:
    roots = [Path.home() / ".codex" / "skills", Path.home() / ".agents" / "skills"]
    items = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("SKILL.md"):
            try:
                head = path.read_text(encoding="utf-8-sig", errors="replace")[:2000]
                name = re.search(r"(?m)^name:\s*(.+)$", head)
                description = re.search(r"(?m)^description:\s*(.+)$", head)
                items.append({"name": name.group(1).strip() if name else path.parent.name, "description": description.group(1).strip() if description else "", "path": str(path)})
            except OSError:
                continue
    return ToolResult("", True, {"skills": items}, f"found {len(items)} skills")


def _runtime_health(_args: dict[str, Any], context: ToolContext) -> ToolResult:
    return ToolResult("", True, {"status": "ready", "workspace_root": str(context.workspace_root), "runtime_root": str(context.runtime_root), "managed_processes": len(PROCESSES.list(context.session_id))}, "runtime ready")


def _attachment_list(_args: dict[str, Any], context: ToolContext) -> ToolResult:
    rows = context.store.list_attachments(context.session_id)
    safe_rows = []
    for item in rows:
        path = Path(str(item["path"]))
        if not path.is_file():
            continue
        try:
            relative = path.resolve(strict=True).relative_to(context.workspace_root.resolve(strict=True))
        except (OSError, ValueError):
            continue
        safe_rows.append(
            {
                "id": item["id"],
                "name": item["name"],
                "path": relative.as_posix(),
                "media_type": item["media_type"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
        )
    return ToolResult("", True, {"attachments": safe_rows, "count": len(safe_rows)}, f"listed {len(safe_rows)} attachment(s)")


def _attachment_read(args: dict[str, Any], context: ToolContext) -> ToolResult:
    attachment_id = str(args.get("attachment_id") or "")
    row = context.store.get_attachment(attachment_id)
    if not row or row.get("run_id") != context.session_id:
        return ToolResult("", False, {}, "attachment is not bound to this run", error="attachment_not_found")
    resolved = Path(str(row["path"])).resolve(strict=True)
    resolved.relative_to(context.workspace_root.resolve(strict=True))
    if not resolved.is_file() or resolved.is_symlink():
        return ToolResult("", False, {}, "attachment is not a regular file", error="attachment_invalid")
    max_bytes = max(1024, min(int(args.get("max_bytes", 131072)), 2 * 1024 * 1024))
    raw = resolved.read_bytes()[:max_bytes]
    binary = b"\x00" in raw
    content: dict[str, Any] = {
        "id": attachment_id,
        "name": row["name"],
        "bytes": row["bytes"],
        "sha256": row["sha256"],
        "media_type": row["media_type"],
        "truncated": int(row["bytes"]) > len(raw),
        "binary": binary,
    }
    if binary:
        content["base64_preview"] = base64.b64encode(raw[:4096]).decode("ascii")
    else:
        content["text"] = raw.decode(str(args.get("encoding") or "utf-8"), errors="replace")
    return ToolResult("", True, content, f"read attachment {row['name']}")


def _artifact_publish(args: dict[str, Any], context: ToolContext) -> ToolResult:
    path = _path(args, "path", context)
    outputs = (context.workspace_root / "outputs").resolve(strict=False)
    try:
        path.resolve(strict=False).relative_to(outputs)
    except ValueError:
        return ToolResult("", False, {"path": str(path)}, "deliverables must be published from outputs/", error="artifact_outside_outputs")
    if not path.is_file() or path.is_symlink():
        return ToolResult("", False, {"path": str(path)}, "deliverable must be a regular file", error="artifact_invalid")
    artifact_id = new_id("artifact")
    display_name = _safe_name(str(args.get("name") or path.name))
    snapshot = _snapshot_artifact_file(
        path,
        outputs,
        artifact_id,
        display_name,
    )
    digest = _sha256_file(snapshot)
    size = snapshot.stat().st_size
    media = _media_type(snapshot, str(args.get("media_type") or ""))
    preview_kind = _preview_kind(snapshot, media)
    if preview_kind == "office":
        _derive_office_preview(snapshot, digest)
    payload = {
        "id": artifact_id,
        "run_id": context.session_id,
        "session_id": context.session_id,
        "name": display_name,
        "path": str(snapshot),
        "media_type": media,
        "bytes": size,
        "sha256": digest,
        "source_tool_call": str(args.get("source_tool_call") or "model_publish")[:180],
        "preview_kind": preview_kind,
        "verified_at": utc_now(),
    }
    row = context.store.publish_deliverable(payload)
    if str(row.get("id") or "") != artifact_id:
        snapshot.unlink(missing_ok=True)
    public = public_artifact(row)
    context.store.append_event(context.session_id, "artifact_published", {"run_id": context.session_id, "artifact": public})
    return ToolResult("", True, {"artifact": public}, f"published {row['name']}", artifacts=[public])


def _artifact_list(_args: dict[str, Any], context: ToolContext) -> ToolResult:
    rows = [public_artifact(item) for item in context.store.list_deliverables(context.session_id)]
    return ToolResult("", True, {"artifacts": rows, "count": len(rows)}, f"listed {len(rows)} published artifact(s)")


def _artifact_import(args: dict[str, Any], context: ToolContext) -> ToolResult:
    """Import an exact, SHA-verified allowlist from a same-tenant parent Run."""

    source_run_id = str(args.get("source_run_id") or "")
    artifact_ids = list(dict.fromkeys(str(item) for item in (args.get("artifact_ids") or []) if str(item)))
    if (
        set(args) != {"source_run_id", "artifact_ids"}
        or not re.fullmatch(r"run_[0-9a-f]{32}", source_run_id)
        or source_run_id == context.session_id
        or not artifact_ids
        or len(artifact_ids) > 20
        or any(not re.fullmatch(r"artifact_[0-9a-f]{32}", item) for item in artifact_ids)
    ):
        return ToolResult("", False, {}, "artifact import arguments are invalid", error="artifact_import_arguments_invalid")
    if not context.approval_verified:
        return ToolResult("", False, {}, "artifact import requires exact approval", error="exact_approval_required")

    current_session = context.store.get_session(context.session_id)
    source_session = context.store.get_session(source_run_id)

    def tenant_id(session: Any) -> str:
        if not isinstance(session, dict):
            return ""
        metadata = session.get("metadata")
        identity = metadata.get("managed_hpc_identity") if isinstance(metadata, dict) else None
        value = str(identity.get("tenant_id") or "") if isinstance(identity, dict) else ""
        return value if re.fullmatch(r"tenant_[A-Za-z0-9_.-]{1,120}", value) else ""

    current_tenant = tenant_id(current_session)
    source_tenant = tenant_id(source_session)
    if not current_tenant or current_tenant != source_tenant:
        return ToolResult("", False, {}, "source Run is not bound to the current tenant", error="artifact_import_tenant_mismatch")
    try:
        source_workspace = Path(str(source_session.get("workspace_root") or "")).resolve(strict=True)
        current_workspace = context.workspace_root.resolve(strict=True)
        if (
            source_workspace.name != source_run_id
            or current_workspace.name != context.session_id
            or source_workspace.parent != current_workspace.parent
        ):
            raise ValueError("assistant task roots do not share the managed parent")
    except (OSError, ValueError):
        return ToolResult("", False, {}, "source Run workspace boundary is invalid", error="artifact_import_boundary")

    verified: list[tuple[dict[str, Any], Path]] = []
    total_bytes = 0
    for artifact_id in artifact_ids:
        row = context.store.get_deliverable(artifact_id)
        if not row or str(row.get("run_id") or "") != source_run_id:
            return ToolResult("", False, {}, "source artifact allowlist is incomplete", error="artifact_import_not_found")
        try:
            path = Path(str(row.get("path") or "")).resolve(strict=True)
            allowed_roots = (source_workspace, (context.artifact_root / source_run_id).resolve(strict=False))
            if not any(path.is_relative_to(root) for root in allowed_roots):
                raise ValueError("source artifact path is outside its Run")
            if path.is_symlink() or not path.is_file():
                raise ValueError("source artifact is not a regular file")
            size = path.stat().st_size
            if size != int(row.get("bytes") or 0) or _sha256_file(path) != str(row.get("sha256") or ""):
                raise ValueError("source artifact integrity mismatch")
        except (OSError, ValueError):
            return ToolResult("", False, {}, "source artifact integrity verification failed", error="artifact_import_integrity")
        total_bytes += size
        if total_bytes > 8 * 1024 * 1024 * 1024:
            return ToolResult("", False, {}, "source artifact allowlist is too large", error="artifact_import_size_limit")
        verified.append((row, path))

    outputs = (current_workspace / "outputs").resolve(strict=False)
    outputs.mkdir(parents=True, exist_ok=True)
    imported: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    for source_row, source_path in verified:
        artifact_id = new_id("artifact")
        display_name = _safe_name(str(source_row.get("name") or source_path.name))
        snapshot_root = (outputs / ".a").resolve(strict=False)
        snapshot_root.relative_to(outputs)
        snapshot_root.mkdir(parents=True, exist_ok=True)
        suffix = Path(display_name).suffix[:20]
        snapshot = snapshot_root / f"{artifact_id.rsplit('_', 1)[-1][:16]}{suffix}"
        temporary = snapshot.with_name(snapshot.name + ".tmp")
        try:
            shutil.copyfile(source_path, temporary)
            os.replace(temporary, snapshot)
            snapshot = snapshot.resolve(strict=True)
        except Exception:
            temporary.unlink(missing_ok=True)
            snapshot.unlink(missing_ok=True)
            raise
        payload = {
            "id": artifact_id,
            "run_id": context.session_id,
            "session_id": context.session_id,
            "name": display_name,
            "path": str(snapshot),
            "media_type": str(source_row.get("media_type") or _media_type(snapshot, "")),
            "bytes": snapshot.stat().st_size,
            "sha256": _sha256_file(snapshot),
            "source_tool_call": f"artifact_import:{source_run_id}",
            "preview_kind": str(source_row.get("preview_kind") or _preview_kind(snapshot, str(source_row.get("media_type") or ""))),
            "verified_at": utc_now(),
        }
        if payload["bytes"] != int(source_row["bytes"]) or payload["sha256"] != str(source_row["sha256"]):
            snapshot.unlink(missing_ok=True)
            return ToolResult("", False, {}, "imported artifact integrity verification failed", error="artifact_import_integrity")
        published = context.store.publish_deliverable(payload)
        if str(published.get("id") or "") != artifact_id:
            snapshot.unlink(missing_ok=True)
        public = public_artifact(published)
        imported.append(public)
        source_manifest.append({
            "source_artifact_id": str(source_row["id"]),
            "name": display_name,
            "bytes": int(source_row["bytes"]),
            "sha256": str(source_row["sha256"]),
        })
        context.store.append_event(
            context.session_id,
            "artifact_published",
            {"run_id": context.session_id, "artifact": public, "imported_from_run_id": source_run_id},
        )
    return ToolResult(
        "",
        True,
        {
            "source_run_id": source_run_id,
            "same_tenant_verified": True,
            "exact_approval_verified": True,
            "source_manifest": source_manifest,
            "artifacts": imported,
            "count": len(imported),
            "total_bytes": total_bytes,
        },
        f"imported {len(imported)} SHA-verified artifact(s) from the approved parent Run",
        artifacts=imported,
    )


def _verified_artifact_path(row: dict[str, Any], context: ToolContext) -> Path:
    path = Path(str(row["path"])).resolve(strict=True)
    allowed_roots = (
        context.workspace_root.resolve(strict=True),
        (context.artifact_root / context.session_id).resolve(strict=False),
    )
    allowed = False
    for root in allowed_roots:
        try:
            path.relative_to(root)
            allowed = True
            break
        except ValueError:
            continue
    if not allowed:
        raise ValueError("artifact path is outside the run boundary")
    if not path.is_file() or path.is_symlink():
        raise ValueError("artifact is not a regular file")
    return path


def _artifact_preview(args: dict[str, Any], context: ToolContext) -> ToolResult:
    artifact_id = str(args.get("artifact_id") or "")
    row = context.store.get_deliverable(artifact_id)
    if row and row.get("run_id") == context.session_id:
        path = _verified_artifact_path(row, context)
        if path.stat().st_size != int(row["bytes"]) or _sha256_file(path) != row["sha256"]:
            return ToolResult("", False, {}, "artifact integrity verification failed", error="artifact_integrity")
        kind = str(row["preview_kind"])
        preview: Any = None
        preview_complete = True
        if kind in {"text", "markdown", "structured_text", "table"}:
            preview = path.read_text(encoding="utf-8", errors="replace")[:131072]
            preview_complete = path.stat().st_size <= 131072
        elif kind == "archive":
            with zipfile.ZipFile(path) as archive:
                preview = [
                    {"name": item.filename, "bytes": item.file_size}
                    for item in archive.infolist()[:500]
                    if not item.is_dir() and not Path(item.filename).is_absolute() and ".." not in Path(item.filename).parts
                ]
                preview_complete = len(archive.infolist()) <= 500
        return ToolResult(
            "",
            True,
            {
                "artifact": public_artifact(row),
                "preview": preview,
                "preview_complete": preview_complete,
                "artifact_hash_verified": True,
            },
            f"previewed {row['name']}",
        )

    generic = context.store.get_artifact(artifact_id)
    if not generic or generic.get("session_id") != context.session_id:
        return ToolResult("", False, {}, "artifact is not bound to this run", error="artifact_not_found")
    try:
        path = Path(str(generic.get("path") or "")).resolve(strict=True)
        artifact_root = context.artifact_root.resolve(strict=True)
    except OSError:
        return ToolResult("", False, {}, "artifact integrity verification failed", error="artifact_integrity")
    try:
        path.relative_to(artifact_root)
    except ValueError:
        return ToolResult("", False, {}, "artifact path is outside the run boundary", error="artifact_integrity")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(generic.get("bytes") or 0)
        or _sha256_file(path) != str(generic.get("sha256") or "")
    ):
        return ToolResult("", False, {}, "artifact integrity verification failed", error="artifact_integrity")
    data = path.read_bytes()
    projection: dict[str, Any]
    json_valid = False
    try:
        decoded = json.loads(data.decode("utf-8"))
        projection = _structured_json_projection(decoded)
        json_valid = True
    except (UnicodeError, json.JSONDecodeError):
        projection = {"value_type": "unparseable"}
    return ToolResult(
        "",
        True,
        {
            "artifact": _public_result_artifact(generic),
            "preview": json.dumps(projection, ensure_ascii=False, separators=(",", ":")),
            "preview_complete": False,
            "preview_encoding": "structured_json_projection",
            "source_content_bytes": len(data),
            "source_content_sha256": hashlib.sha256(data).hexdigest(),
            "artifact_hash_verified": True,
            "json_valid": json_valid,
        },
        "previewed oversized tool result",
    )


def _artifact_bundle(args: dict[str, Any], context: ToolContext) -> ToolResult:
    rows = context.store.list_deliverables(context.session_id)
    requested = {str(value) for value in (args.get("artifact_ids") or [])}
    selected = [row for row in rows if not requested or row["id"] in requested]
    if not selected:
        return ToolResult("", False, {}, "no published artifacts selected", error="artifact_selection_empty")
    outputs = context.workspace_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    name = Path(str(args.get("name") or "evomind-deliverables.zip")).name
    if not name.lower().endswith(".zip"):
        name += ".zip"
    target = outputs / name
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        seen: set[str] = set()
        for row in selected:
            source = _verified_artifact_path(row, context)
            arcname = Path(str(row["name"])).name
            if arcname in seen:
                arcname = f"{row['id']}_{arcname}"
            seen.add(arcname)
            archive.write(source, arcname)
    os.replace(temporary, target)
    return _artifact_publish({"path": str(target), "name": name, "source_tool_call": "artifact_bundle"}, context)


def _research_capabilities(_args: dict[str, Any], context: ToolContext) -> ToolResult:
    capabilities = ["scientist", "ask", "workspace", "engineer", "memory", "evolution", "readiness-report", "causal-diagnosis", "strategy", "briefing", "self-audit", "upgrade-plan"]
    return ToolResult("", True, {"capabilities": capabilities, "human_gates": ["official_kaggle_submit", "merge", "deployment", "self_promotion"]}, "legacy Research OS adapter ready")


def _research_invoke(args: dict[str, Any], context: ToolContext) -> ToolResult:
    action = str(args["action"])
    allowed = {"scientist", "ask", "workspace", "engineer", "memory", "evolution", "readiness-report", "causal-diagnosis", "strategy", "briefing", "self-audit", "upgrade-plan"}
    if action not in allowed:
        return ToolResult("", False, {"allowed": sorted(allowed)}, "research action not allowlisted", error="action_not_allowed")
    argv = [os.environ.get("WORKSTATION_PYTHON") or os.sys.executable, "-X", "utf8", "-m", "xsci.kaggle", action]
    if args.get("input"):
        argv.append(str(args["input"]))
    completed = subprocess.run(argv, cwd=str(context.workspace_root), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=max(10, min(int(args.get("timeout_seconds", 180)), 3600)), check=False)
    return ToolResult("", completed.returncode == 0, {"action": action, "exit_code": completed.returncode, "output": completed.stdout, "stderr": completed.stderr}, f"research action {action} exited {completed.returncode}", error="" if completed.returncode == 0 else "nonzero_exit")


def _write_evidence(context: ToolContext, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    target = context.workspace_root / "evidence" / Path(name).name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary.write_bytes(encoded)
    os.replace(temporary, target)
    return {"path": target.relative_to(context.workspace_root).as_posix(), "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def _url_json(url: str, *, timeout: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "EvoMind/0.3 literature-verifier"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("external source returned a non-object response")
    return value


def _literature_search(args: dict[str, Any], context: ToolContext) -> ToolResult:
    query = " ".join(str(args.get("query") or "").split())
    if not query:
        return ToolResult("", False, {}, "literature query is required", error="invalid_query")
    limit = max(1, min(int(args.get("max_results", 12)), 30))
    papers: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    try:
        params = urllib.parse.urlencode({"search": query, "per-page": limit, "select": "id,doi,title,publication_year,authorships,primary_location"})
        payload = _url_json(f"https://api.openalex.org/works?{params}")
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            location = item.get("primary_location") if isinstance(item.get("primary_location"), dict) else {}
            authorships = item.get("authorships") if isinstance(item.get("authorships"), list) else []
            authors = [str((author.get("author") or {}).get("display_name") or "") for author in authorships if isinstance(author, dict)]
            doi = str(item.get("doi") or "").removeprefix("https://doi.org/")
            papers.append({
                "source": "openalex", "id": str(item.get("id") or ""), "title": str(item.get("title") or ""),
                "year": item.get("publication_year"), "authors": [value for value in authors if value][:12], "doi": doi,
                "url": str(location.get("landing_page_url") or item.get("id") or ""),
            })
    except Exception as exc:
        errors.append({"source": "openalex", "error_type": type(exc).__name__})

    try:
        params = urllib.parse.urlencode({"query.bibliographic": query, "rows": limit, "select": "DOI,title,author,published-print,published-online,URL"})
        payload = _url_json(f"https://api.crossref.org/works?{params}")
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        for item in message.get("items") or []:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            dates = item.get("published-print") or item.get("published-online") or {}
            date_parts = dates.get("date-parts") if isinstance(dates, dict) else []
            year = date_parts[0][0] if date_parts and date_parts[0] else None
            authors = [" ".join(filter(None, (str(author.get("given") or ""), str(author.get("family") or "")))) for author in item.get("author") or [] if isinstance(author, dict)]
            papers.append({
                "source": "crossref", "id": str(item.get("DOI") or ""),
                "title": str(title[0] if isinstance(title, list) and title else title or ""), "year": year,
                "authors": [value for value in authors if value][:12], "doi": str(item.get("DOI") or ""), "url": str(item.get("URL") or ""),
            })
    except Exception as exc:
        errors.append({"source": "crossref", "error_type": type(exc).__name__})

    try:
        params = urllib.parse.urlencode({"search_query": f"all:{query}", "start": 0, "max_results": limit})
        request = urllib.request.Request(f"https://export.arxiv.org/api/query?{params}", headers={"Accept": "application/atom+xml", "User-Agent": "EvoMind/0.3 literature-verifier"})
        with urllib.request.urlopen(request, timeout=30) as response:
            root = ET.fromstring(response.read())
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            identifier = (entry.findtext("atom:id", default="", namespaces=ns) or "").rsplit("/", 1)[-1]
            papers.append({
                "source": "arxiv", "id": identifier, "title": " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split()),
                "year": (entry.findtext("atom:published", default="", namespaces=ns) or "")[:4],
                "authors": [str(author.findtext("atom:name", default="", namespaces=ns) or "") for author in entry.findall("atom:author", ns)][:12],
                "doi": "", "url": f"https://arxiv.org/abs/{identifier}",
            })
    except Exception as exc:
        errors.append({"source": "arxiv", "error_type": type(exc).__name__})

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for paper in papers:
        key = str(paper.get("doi") or paper.get("title") or paper.get("id") or "").casefold().strip()
        if not key or key in seen or not paper.get("title"):
            continue
        seen.add(key)
        unique.append(paper)
        if len(unique) >= limit:
            break
    payload = {
        "schema": "evomind.literature.search.v1", "query": query, "papers": unique,
        "source_counts": {source: sum(item["source"] == source for item in unique) for source in ("arxiv", "openalex", "crossref")},
        "source_errors": errors, "fabricated": 0, "generated_at": utc_now(),
    }
    evidence = _write_evidence(context, f"literature_search_{hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]}.json", payload)
    ok = bool(unique)
    return ToolResult("", ok, {**payload, "evidence": evidence}, f"verified {len(unique)} external paper(s)", error="" if ok else "literature_sources_unavailable")


def _citation_audit(args: dict[str, Any], context: ToolContext) -> ToolResult:
    artifact_id = str(args.get("artifact_id") or "")
    row = context.store.get_deliverable(artifact_id)
    if not row or row.get("run_id") != context.session_id:
        return ToolResult("", False, {}, "artifact is not bound to this run", error="artifact_not_found")
    path = _verified_artifact_path(row, context)
    if _sha256_file(path) != row["sha256"]:
        return ToolResult("", False, {}, "artifact integrity verification failed", error="artifact_integrity")
    text = path.read_text(encoding="utf-8", errors="replace")[:2_000_000]
    dois = list(dict.fromkeys(match.rstrip(".,;:)]}") for match in re.findall(r"(?i)\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", text)))[:20]
    checks = []
    for doi in dois:
        try:
            payload = _url_json(f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}", timeout=20)
            message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
            title = message.get("title")
            checks.append({"doi": doi, "verified": True, "title": str(title[0] if isinstance(title, list) and title else title or "")})
        except Exception as exc:
            checks.append({"doi": doi, "verified": False, "error_type": type(exc).__name__})
    payload = {
        "schema": "evomind.citation.audit.v1", "artifact_id": artifact_id, "artifact_sha256": row["sha256"],
        "doi_count": len(dois), "verified_count": sum(item["verified"] for item in checks), "checks": checks, "generated_at": utc_now(),
    }
    evidence = _write_evidence(context, f"citation_audit_{artifact_id}.json", payload)
    ok = bool(dois) and all(item["verified"] for item in checks)
    return ToolResult("", ok, {**payload, "evidence": evidence}, f"verified {payload['verified_count']}/{len(dois)} DOI citation(s)", error="" if ok else "citation_audit_incomplete")


def _memory_search(args: dict[str, Any], context: ToolContext) -> ToolResult:
    query = " ".join(str(args.get("query") or "").split())
    if not query:
        return ToolResult("", False, {}, "memory query is required", error="invalid_query")
    limit = max(1, min(int(args.get("limit", 10)), 50))
    terms = [term.casefold() for term in re.findall(r"[\w\u3400-\u9fff-]+", query) if len(term) >= 2]
    roots = [context.project_root / ".xsci", context.project_root / "workspace" / "memory", context.project_root / "workspace" / "evolution"]
    matches: list[dict[str, Any]] = []
    scanned = 0
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if scanned >= 500 or len(matches) >= limit or not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".json", ".jsonl"}:
                continue
            scanned += 1
            if path.stat().st_size > 4 * 1024 * 1024 or any(part.lower() in {"secrets", "credentials"} for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            folded = text.casefold()
            score = sum(folded.count(term) for term in terms)
            if score <= 0:
                continue
            first = min((folded.find(term) for term in terms if term in folded), default=0)
            snippet = " ".join(text[max(0, first - 160):first + 360].split())
            match = {"path": path.relative_to(context.project_root).as_posix(), "score": score, "snippet": snippet, "sha256": _sha256_file(path)}
            if path.suffix.lower() == ".json":
                try:
                    payload = json.loads(text)
                except (TypeError, ValueError):
                    payload = None
                if isinstance(payload, dict) and payload.get("schema") == "evomind.memory.writeback.v1":
                    match.update({
                        "record_id": str(payload.get("record_id") or ""),
                        "record_type": str(payload.get("record_type") or ""),
                        "task_id": str(payload.get("task_id") or ""),
                        "source_run_id": str(payload.get("source_run_id") or ""),
                    })
            matches.append(match)
    matches.sort(key=lambda item: (-int(item["score"]), str(item["path"])))
    return ToolResult("", True, {"query": query, "matches": matches[:limit], "count": len(matches[:limit]), "files_scanned": scanned}, f"found {len(matches[:limit])} memory match(es)")


_MEMORY_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
_MEMORY_SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")
_MEMORY_SECRET_KEYS = {
    "access_key", "access_token", "api_key", "authorization", "client_secret", "cookie", "credential",
    "credentials", "dpapi", "password", "passwd", "private_key", "proxy_password", "refresh_token",
    "secret", "session_token", "token",
}
_MEMORY_SECRET_KEY_SUFFIXES = (
    "accesskey", "accesstoken", "apikey", "authorization", "clientsecret", "credential", "credentials",
    "password", "passwd", "privatekey", "proxypassword", "refreshtoken", "secret", "sessiontoken",
)


def _memory_record_guard(value: Any, *, depth: int = 0) -> None:
    if depth > 5:
        raise ValueError("memory record nesting exceeds the configured limit")
    if isinstance(value, dict):
        if len(value) > 64:
            raise ValueError("memory record has too many fields")
        for raw_key, child in value.items():
            key = str(raw_key).casefold().replace("-", "_")
            compact_key = re.sub(r"[^a-z0-9]", "", key)
            if key in _MEMORY_SECRET_KEYS or compact_key == "token" or compact_key.endswith(_MEMORY_SECRET_KEY_SUFFIXES):
                raise ValueError("memory record contains a forbidden secret field")
            _memory_record_guard(child, depth=depth + 1)
        return
    if isinstance(value, list):
        if len(value) > 64:
            raise ValueError("memory record list exceeds the configured limit")
        for child in value:
            _memory_record_guard(child, depth=depth + 1)
        return
    if isinstance(value, str):
        if len(value) > 8_000:
            raise ValueError("memory record text exceeds the configured limit")
        if re.search(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\)", value) or re.search(r"(?:^|[\s'\"(])/(?!/)", value):
            raise ValueError("memory record must not contain absolute filesystem paths")
        if "-----BEGIN " in value or _redact_process_log(value) != value:
            raise ValueError("memory record contains secret-like content")


def _memory_writeback(args: dict[str, Any], context: ToolContext) -> ToolResult:
    task_id = str(args.get("task_id") or "").strip()
    source_run_id = str(args.get("source_run_id") or "").strip()
    idempotency_key = str(args.get("idempotency_key") or "").strip()
    record_type = str(args.get("record_type") or "").strip()
    record = dict(args.get("record") or {})
    sources = list(args.get("source_evidence") or [])
    if not _MEMORY_SAFE_ID.fullmatch(task_id):
        return ToolResult("", False, {}, "memory task id is invalid", error="invalid_task_id")
    if source_run_id != context.session_id:
        return ToolResult("", False, {}, "memory source run must equal the current run", error="source_run_mismatch")
    required = {"retrospective": {"summary", "lessons", "strategy_changes"}, "experience": {"summary", "lessons", "applicable_conditions"}}
    missing = sorted(required[record_type] - set(record))
    if missing:
        return ToolResult("", False, {"missing_fields": missing}, "memory record is incomplete", error="record_incomplete")
    encoded_record = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded_record) > 32 * 1024:
        return ToolResult("", False, {}, "memory record exceeds 32 KiB", error="record_too_large")
    _memory_record_guard(record)

    normalized_sources: list[dict[str, str]] = []
    for source in sources:
        kind = str(source.get("kind") or "")
        source_id = str(source.get("id") or "")
        digest = str(source.get("sha256") or "").lower()
        if not _MEMORY_SAFE_ID.fullmatch(source_id) or not _MEMORY_SHA256.fullmatch(digest):
            return ToolResult("", False, {}, "memory evidence reference is invalid", error="invalid_evidence_reference")
        if kind == "artifact":
            artifact = context.store.get_deliverable(source_id)
            if not artifact or artifact.get("run_id") != source_run_id:
                return ToolResult("", False, {}, "memory source artifact is not bound to this run", error="artifact_not_found")
            artifact_path = Path(str(artifact["path"])).resolve(strict=True)
            artifact_path.relative_to(context.workspace_root.resolve(strict=True))
            if artifact.get("sha256") != digest or _sha256_file(artifact_path) != digest:
                return ToolResult("", False, {}, "memory source artifact integrity verification failed", error="artifact_integrity")
        else:
            evidence_path = (context.workspace_root / "evidence" / source_id).resolve(strict=False)
            evidence_path.relative_to((context.workspace_root / "evidence").resolve(strict=False))
            if not evidence_path.is_file() or evidence_path.is_symlink():
                return ToolResult("", False, {}, "memory source evidence is not bound to this run", error="evidence_not_found")
            if _sha256_file(evidence_path) != digest:
                return ToolResult("", False, {}, "memory source evidence integrity verification failed", error="evidence_integrity")
        normalized_sources.append({"kind": kind, "id": source_id, "sha256": digest})

    request = {
        "task_id": task_id,
        "record_type": record_type,
        "source_run_id": source_run_id,
        "source_evidence": normalized_sources,
        "record": record,
        "idempotency_key": idempotency_key,
    }
    request_sha256 = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    record_key = f"{task_id}\0{idempotency_key}".encode("utf-8")
    record_id = f"memory_{hashlib.sha256(record_key).hexdigest()[:24]}"
    root = (context.project_root / "workspace" / "evolution" / task_id / "memory").resolve(strict=False)
    root.relative_to(context.project_root.resolve(strict=True))
    target = root / f"{record_id}.json"

    with MEMORY_WRITEBACK_LOCK:
        if target.exists():
            existing_bytes = target.read_bytes()
            try:
                existing = json.loads(existing_bytes.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                return ToolResult("", False, {}, "existing memory record is unreadable", error="memory_integrity")
            if existing.get("request_sha256") != request_sha256:
                return ToolResult("", False, {"record_id": record_id}, "memory idempotency key is bound to different content", error="idempotency_conflict")
            digest = hashlib.sha256(existing_bytes).hexdigest()
            relative = target.relative_to(context.project_root).as_posix()
            return ToolResult("", True, {"record_id": record_id, "path": relative, "bytes": len(existing_bytes), "sha256": digest, "replayed": True}, f"replayed memory record {record_id}")

        payload = {
            "schema": "evomind.memory.writeback.v1",
            "record_id": record_id,
            **request,
            "request_sha256": request_sha256,
            "created_at": utc_now(),
        }
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        root.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, encoded)
        if target.read_bytes() != encoded:
            return ToolResult("", False, {}, "memory write verification failed", error="memory_integrity")
    relative = target.relative_to(context.project_root).as_posix()
    return ToolResult("", True, {"record_id": record_id, "path": relative, "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest(), "replayed": False}, f"persisted memory record {record_id}")


def _verified_context(args: dict[str, Any], context: ToolContext) -> ToolResult:
    """Load evidence-bound history for the selected existing research task."""

    from xsci.assistant_context import build_assistant_context

    selected_task = str(args.get("selected_task") or context.metadata.get("selected_task") or "").strip()
    if selected_task and not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", selected_task):
        raise ValueError("selected_task is invalid")
    packet = build_assistant_context(
        context.project_root,
        live_environment=False,
        selected_task=selected_task,
    )
    payload = packet.to_dict()
    run = payload.get("current_run") if isinstance(payload.get("current_run"), dict) else {}
    summary = (
        f"verified context loaded for run {run.get('run_id')}"
        if run.get("available")
        else "verified context loaded; no matching historical run was found"
    )
    return ToolResult("", True, payload, summary)


def _report_job_summary(job: dict[str, Any]) -> dict[str, Any]:
    value = {key: job.get(key) for key in ("id", "run_id", "status", "stage", "report_status", "evidence_status", "execution_status", "error_code", "document_sha256", "source_artifact_ids", "source_count", "missing")}
    value["artifacts"] = [{key: artifact.get(key) for key in ("id", "name", "sha256", "bytes", "preview_kind", "preview_url", "download_url")}
                          for artifact in job.get("artifacts", [])]
    value["attempt_count"] = len(job.get("attempts", []))
    return value


def _report_generate(args: dict[str, Any], context: ToolContext) -> ToolResult:
    if args.get("report_id"):
        if any(args.get(key) for key in ("formats", "title", "summary", "artifact_ids", "name", "report_kind", "language")):
            raise ValueError("report_resume_cannot_change_source")
        if context.reports is None:
            return ToolResult("", False, {}, "Report service unavailable", error="report_service_unavailable")
        job = context.reports.resume(context.session_id, str(args["report_id"]))
        return ToolResult("", True, {"report_job": _report_job_summary(job)}, "Existing report job retained; query report_status for its terminal state.")
    if args.get("formats"):
        if context.reports is None:
            return ToolResult("", False, {}, "Report service unavailable", error="report_service_unavailable")
        job = context.reports.enqueue(context.session_id, args)
        return ToolResult("", True, {"report_job": _report_job_summary(job)}, "Report generation is queued; use report_status before claiming delivery.")
    title = " ".join(str(args.get("title") or "EvoMind Assistant Report").split())[:200]
    requested = {str(value) for value in args.get("artifact_ids") or []}
    artifacts = [row for row in context.store.list_deliverables(context.session_id) if not requested or row["id"] in requested]
    attachments = context.store.list_attachments(context.session_id)
    lines = [f"# {title}", "", f"- Run ID: `{context.session_id}`", f"- Generated: `{utc_now()}`", "", "## Inputs", ""]
    lines.extend([f"- `{item['name']}` — {item['bytes']} bytes — SHA-256 `{item['sha256']}`" for item in attachments] or ["- No attached files."])
    lines.extend(["", "## Verified artifacts", ""])
    lines.extend([f"- `{item['name']}` — {item['media_type']} — {item['bytes']} bytes — SHA-256 `{item['sha256']}`" for item in artifacts] or ["- No prior published artifacts."])
    summary = str(args.get("summary") or "").strip()
    if summary:
        lines.extend(["", "## Summary", "", summary])
    lines.extend(["", "## Evidence boundary", "", "Only files re-read and SHA-256 verified by the durable run are listed above.", ""])
    name = Path(str(args.get("name") or "assistant_report.md")).name
    if not name.lower().endswith(".md"):
        name += ".md"
    target = context.workspace_root / "outputs" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return _artifact_publish({"path": str(target), "name": name, "media_type": "text/markdown", "source_tool_call": "report_generate"}, context)


def _report_status(args: dict[str, Any], context: ToolContext) -> ToolResult:
    if context.reports is None:
        return ToolResult("", False, {}, "Report service unavailable", error="report_service_unavailable")
    job = context.reports.get(context.session_id, str(args["report_id"]), include_document=False)
    return ToolResult("", True, {"report_job": _report_job_summary(job)}, "Report job status: " + job["status"])


def _evolution_evaluate(args: dict[str, Any], _context: ToolContext) -> ToolResult:
    from .evolution import evaluate_candidate
    metrics = dict(args.get("metrics") or {})
    decision = asdict(evaluate_candidate(metrics))
    return ToolResult("", True, {"decision": decision, "auto_promoted": False}, f"candidate status: {decision['status']}")


def _load_kaggle_api(context: ToolContext):
    from xsci.config import inject_engine_env, load_config
    inject_engine_env(load_config(context.project_root))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def _managed_kaggle_credential_payload() -> dict[str, str]:
    """Project an already-loaded managed Kaggle credential without logging it.

    The verified workstation launcher loads the DPAPI credential into the
    service process environment.  Competition-data Runs still prefer their
    tenant-bound one-time secret references; this projection is the safe
    fallback for the same managed connector already used by ``kaggle_list``
    and ``kaggle_download``.
    """

    api_token = str(os.getenv("KAGGLE_API_TOKEN") or "").strip()
    username = str(os.getenv("KAGGLE_USERNAME") or "").strip()
    legacy_key = str(os.getenv("KAGGLE_KEY") or "").strip()
    if api_token:
        if len(api_token) > 4096:
            raise ValueError("managed Kaggle API token is invalid")
        return {
            "schema": "evomind.run_secret_payload.v1",
            "purpose": "kaggle_api",
            "token": api_token,
        }
    if username and legacy_key:
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", username) is None or len(legacy_key) > 4096:
            raise ValueError("managed Kaggle username/key credential is invalid")
        return {
            "schema": "evomind.run_secret_payload.v1",
            "purpose": "kaggle_api",
            "username": username,
            "token": legacy_key,
        }
    raise FileNotFoundError("managed Kaggle credential is not loaded")


def _is_transient_kaggle_error(exc: BaseException) -> bool:
    transient_names = {
        "ConnectionError", "ConnectionResetError", "ProtocolError", "TimeoutError",
        "ReadTimeout", "ReadTimeoutError", "ConnectTimeout", "MaxRetryError",
    }
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in transient_names:
            return True
        response = getattr(current, "response", None)
        status = int(getattr(response, "status", 0) or getattr(response, "status_code", 0) or 0)
        if status == 429 or 500 <= status <= 599:
            return True
        reason = getattr(current, "reason", None)
        current = current.__cause__ or current.__context__ or (reason if isinstance(reason, BaseException) else None)
    return False


def _kaggle_list(args: dict[str, Any], context: ToolContext) -> ToolResult:
    base_delay = max(0.0, min(float(os.getenv("EVOMIND_KAGGLE_LIST_RETRY_BASE_SECONDS", "0.5")), 5.0))
    last_error: Exception | None = None
    api = None
    for attempt in range(1, 4):
        try:
            if api is None:
                api = _load_kaggle_api(context)
            response = api.competitions_list(page=max(1, int(args.get("page", 1))), search=str(args.get("query") or ""), sort_by="latestDeadline")
            # kaggle 2.2.x returns an ApiListCompetitionsResponse wrapper while
            # older clients returned the competition sequence directly.
            rows = getattr(response, "competitions", response)
            competitions = []
            for item in list(rows or [])[:50]:
                competitions.append({
                    "slug": str(getattr(item, "ref", "") or ""), "title": str(getattr(item, "title", "") or ""),
                    "deadline": str(getattr(item, "deadline", "") or ""), "metric": str(getattr(item, "evaluationMetric", "") or ""),
                    "url": f"https://www.kaggle.com/competitions/{getattr(item, 'ref', '')}",
                })
            return ToolResult(
                "", True,
                {"competitions": competitions, "count": len(competitions), "attempts": attempt},
                f"listed {len(competitions)} competition(s)",
            )
        except Exception as exc:
            last_error = exc
            if attempt >= 3 or not _is_transient_kaggle_error(exc):
                break
            time.sleep(base_delay * (2 ** (attempt - 1)))
    assert last_error is not None
    if _is_transient_kaggle_error(last_error):
        from .competition_data import catalog_projection, kaggle_catalog_projection

        query = str(args.get("query") or "").strip().casefold()
        fixed = kaggle_catalog_projection()
        if query:
            fixed = [
                item for item in fixed
                if query in str(item["slug"]).casefold()
                or query in str(item["title"]).casefold()
                or query in str(item["catalog_id"]).casefold()
            ]
        return ToolResult(
            "",
            True,
            {
                "competitions": fixed,
                "count": len(fixed),
                "attempts": attempt,
                "source": "managed_catalog_fallback",
                "live_listing_ok": False,
                "listing_error_class": type(last_error).__name__,
                "competition_data_catalog": catalog_projection(),
            },
            "live Kaggle listing unavailable; using the fixed managed competition catalog",
        )
    return ToolResult(
        "", False, {"attempts": attempt}, "Kaggle competition listing failed",
        error=type(last_error).__name__,
    )


def _safe_extract_zip(archive_path: Path, destination: Path, *, max_total_bytes: int) -> list[Path]:
    extracted: list[Path] = []
    total = 0
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            relative = Path(item.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Kaggle archive contains an unsafe path")
            total += int(item.file_size)
            if total > max_total_bytes:
                raise ValueError("Kaggle archive exceeds the configured extraction limit")
            target = (destination / relative).resolve(strict=False)
            target.relative_to(destination.resolve(strict=True))
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted.append(target)
    return extracted


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _is_transient_kaggle_worker_error(detail: str) -> bool:
    lowered = detail.casefold()
    if any(term in lowered for term in (
        "401", "403", "404", "unauthorized", "forbidden", "not found",
        "accept the competition rules", "must accept", "join the competition",
    )):
        return False
    if re.search(r"(?:http(?: status)?\s*[:=]?\s*|status(?: code)?\s*[:=]?\s*)(429|502|503|504)\b", lowered):
        return True
    return any(term in lowered for term in (
        "connectionerror", "connection reset", "connectionreseterror",
        "winerror 10054", "error 10054", "remote host forcibly closed",
        "readtimeout", "connecttimeout", "timeout", "timed out",
        "temporarily unavailable", "too many requests", "bad gateway",
        "service unavailable", "gateway timeout",
    ))


def _download_kaggle_archive_bounded(slug: str, target: Path, workspace_root: Path) -> Path:
    total_timeout = _bounded_env_int("EVOMIND_KAGGLE_TOTAL_TIMEOUT_SECONDS", 300, 60, 1800)
    retry_base_delay = max(
        0.0,
        min(float(os.getenv("EVOMIND_KAGGLE_DOWNLOAD_RETRY_BASE_SECONDS", "1")), 5.0),
    )
    max_attempts = 3
    deadline = time.monotonic() + total_timeout
    archive = target / f"{slug}.zip"
    if archive.is_file() and archive.stat().st_size > 0 and zipfile.is_zipfile(archive):
        return archive
    archive.unlink(missing_ok=True)
    staging = target.parent / f".{target.name}.download"
    staging.mkdir(parents=True, exist_ok=True)
    expected_names = {archive.name, f"{archive.name}.kaggle-partial"}
    staged_items = list(staging.iterdir())
    if any(item.is_symlink() or not item.is_file() or item.name not in expected_names for item in staged_items):
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=False)
    worker = workspace_root / "work" / "kaggle_download_worker.py"
    worker.parent.mkdir(parents=True, exist_ok=True)
    worker.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "from kaggle.api.kaggle_api_extended import KaggleApi\n"
        "slug, destination = sys.argv[1], Path(sys.argv[2])\n"
        "destination.mkdir(parents=True, exist_ok=True)\n"
        "api = KaggleApi()\n"
        "api.authenticate()\n"
        "api.competition_download_files(slug, path=str(destination), quiet=True)\n",
        encoding="utf-8",
    )
    allowed_environment = {
        "PATH", "SYSTEMROOT", "WINDIR", "APPDATA", "LOCALAPPDATA", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
        "PROGRAMDATA", "TEMP", "TMP", "PYTHONPATH", "KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY",
        "KAGGLE_CONFIG_DIR", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    }
    child_environment = {name: value for name, value in os.environ.items() if name.upper() in allowed_environment}
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    preserve_staging = False
    try:
        staged_archive = staging / archive.name
        if staged_archive.is_file() and staged_archive.stat().st_size > 0 and zipfile.is_zipfile(staged_archive):
            os.replace(staged_archive, archive)
            return archive
        for attempt in range(1, max_attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                preserve_staging = True
                raise TimeoutError("Kaggle download worker exceeded the shared total timeout")
            attempts_left = max_attempts - attempt + 1
            attempt_timeout = max(1, int(remaining / attempts_left))
            worker_error = ""
            try:
                completed = subprocess.run(
                    [sys.executable, "-X", "utf8", str(worker), slug, str(staging)],
                    cwd=str(workspace_root),
                    env=child_environment,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=attempt_timeout,
                    check=False,
                    creationflags=creationflags,
                )
            except subprocess.TimeoutExpired as exc:
                failure: Exception = TimeoutError(
                    f"Kaggle download worker attempt {attempt}/{max_attempts} timed out"
                )
                failure.__cause__ = exc
                transient = True
            else:
                if completed.returncode == 0:
                    candidates = sorted(staging.glob("*.zip"))
                    staged_archive = next(
                        (item for item in candidates if item.name == archive.name),
                        candidates[0] if len(candidates) == 1 else None,
                    )
                    if staged_archive is None or staged_archive.stat().st_size <= 0 or not zipfile.is_zipfile(staged_archive):
                        raise ValueError("Kaggle download is not a valid ZIP archive")
                    os.replace(staged_archive, archive)
                    return archive
                worker_error = _redact_process_log(
                    (completed.stderr or b"").decode("utf-8", errors="replace")
                ).strip()
                if len(worker_error) > 2000:
                    worker_error = worker_error[-2000:]
                failure = RuntimeError(
                    f"Kaggle download worker exited {completed.returncode} on attempt "
                    f"{attempt}/{max_attempts}: {worker_error or 'no stderr was returned'}"
                )
                transient = _is_transient_kaggle_worker_error(worker_error)
            if not transient:
                raise failure
            if attempt >= max_attempts:
                preserve_staging = True
                raise failure
            delay = retry_base_delay * (2 ** (attempt - 1))
            if deadline - time.monotonic() <= delay:
                preserve_staging = True
                raise TimeoutError("Kaggle download worker exceeded the shared total timeout") from failure
            time.sleep(delay)
    finally:
        if not preserve_staging:
            shutil.rmtree(staging, ignore_errors=True)


def _kaggle_download(args: dict[str, Any], context: ToolContext) -> ToolResult:
    slug = str(args.get("competition") or "").strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,120}", slug):
        return ToolResult("", False, {}, "invalid Kaggle competition slug", error="invalid_competition")
    target = context.workspace_root / "inputs" / "kaggle" / slug
    target.mkdir(parents=True, exist_ok=True)
    try:
        from xsci.config import inject_engine_env, load_config
        inject_engine_env(load_config(context.project_root), include_gpu=False)
        _download_kaggle_archive_bounded(slug, target, context.workspace_root)
        for archive in list(target.glob("*.zip")):
            _safe_extract_zip(archive, target, max_total_bytes=int(os.environ.get("EVOMIND_KAGGLE_EXTRACT_LIMIT_BYTES", 20 * 1024**3)))
        files = []
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            files.append({"path": path.relative_to(context.workspace_root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
        manifest = {
            "schema": "evomind.kaggle.download.v2",
            "competition": slug,
            "source": f"https://www.kaggle.com/competitions/{slug}/data",
            "download_mode": "bounded_atomic",
            "resume_supported": True,
            "files": files,
            "generated_at": utc_now(),
        }
        evidence = _write_evidence(context, f"kaggle_download_{slug}.json", manifest)
        return ToolResult("", bool(files), {**manifest, "evidence": evidence}, f"downloaded and hashed {len(files)} Kaggle file(s)", error="" if files else "kaggle_download_empty")
    except Exception as exc:
        detail = _redact_process_log(str(exc)).strip()[:2400]
        lowered = detail.casefold()
        if any(term in lowered for term in (
            "403", "forbidden", "permission", "unauthorized", "401",
            "accept the competition rules", "must accept", "join the competition",
        )):
            category = "human_gate_or_auth"
            human_gate_required = True
        elif any(term in lowered for term in ("404", "not found", "no such competition")):
            category = "competition_not_found"
            human_gate_required = False
        elif _is_transient_kaggle_error(exc) or any(term in lowered for term in (
            "timeout", "timed out", "connection", "temporarily unavailable", "429", "502", "503", "504",
        )):
            category = "transient_network"
            human_gate_required = False
        else:
            category = "runtime_or_client"
            human_gate_required = False
        return ToolResult(
            "",
            False,
            {
                "competition": slug,
                "error_class": type(exc).__name__,
                "error_category": category,
                "error_detail": detail,
                "human_gate_required": human_gate_required,
            },
            f"Kaggle download failed ({category})",
            error=type(exc).__name__,
        )


def _kaggle_submit(args: dict[str, Any], context: ToolContext) -> ToolResult:
    slug = str(args.get("competition") or "").strip()
    path = Path(str(args.get("submission_path") or "")).resolve(strict=True)
    path.relative_to((context.workspace_root / "outputs").resolve(strict=True))
    digest = _sha256_file(path)
    try:
        api = _load_kaggle_api(context)
        result = api.competition_submit(str(path), str(args.get("message") or "EvoMind verified submission")[:200], slug)
        reference = str(getattr(result, "ref", "") or getattr(result, "id", "") or "submitted")
        return ToolResult("", True, {"competition": slug, "submission_name": path.name, "sha256": digest, "reference": reference}, "official Kaggle submission accepted")
    except Exception as exc:
        return ToolResult("", False, {"competition": slug, "submission_name": path.name, "sha256": digest}, "Kaggle submission failed", error=type(exc).__name__)


def _managed_competition_training_source(
    competition_value: str,
    args: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    """Resolve a FULL managed competition root without copying it into the Run.

    Competition adapters intentionally keep large datasets on the managed HPC
    filesystem.  Training must bind the current receipt and manifest instead of
    guessing a historical path or asking the user to upload the data again.
    """

    from .competition_data import normalize_competition, persistent_root

    competition = normalize_competition(competition_value)
    status_result = _competition_data_status(
        {
            "competition": competition,
            "timeout_seconds": int(args.get("timeout_seconds", 1800)),
        },
        context,
    )
    status = str(status_result.content.get("status") or "") if isinstance(status_result.content, dict) else ""
    receipt = status_result.content.get("receipt") if isinstance(status_result.content, dict) else None
    if not status_result.ok or status != "FULL_DATA_READY" or not isinstance(receipt, dict):
        raise ValueError(f"competition {competition} is not FULL_DATA_READY")
    manifest_sha256 = str(receipt.get("manifest_sha256") or "").casefold()
    if re.fullmatch(r"[a-f0-9]{64}", manifest_sha256) is None:
        raise ValueError(f"competition {competition} FULL_DATA_READY receipt lacks a valid manifest SHA-256")
    root = str(receipt.get("persistent_root") or "")
    if root != persistent_root(competition):
        raise ValueError(f"competition {competition} persistent root does not match the managed catalog")
    file_count = int(receipt.get("files") or 0)
    byte_count = int(receipt.get("bytes") or 0)
    if file_count <= 0 or byte_count <= 0:
        raise ValueError(f"competition {competition} FULL_DATA_READY receipt lacks verified files")
    artifact = status_result.content.get("artifact") if isinstance(status_result.content, dict) else None
    return {
        "attachment_id": "",
        "name": f"{competition}_persistent_data",
        "media_type": "application/x-evomind-managed-competition-data",
        "sha256": manifest_sha256,
        "bytes": byte_count,
        "file_count": file_count,
        "competition": competition,
        "source_kind": "managed_competition_data",
        "data_status": status,
        "persistent_root": root,
        "manifest_sha256": manifest_sha256,
        "status_artifact_id": str(artifact.get("id") or "") if isinstance(artifact, dict) else "",
        "local_copy_created": False,
    }


def _training_source(args: dict[str, Any], context: ToolContext) -> tuple[Path | None, dict[str, Any]]:
    attachment_id = str(args.get("attachment_id") or "").strip()
    path_value = str(args.get("path") or "").strip()
    competition_value = str(args.get("competition") or "").strip()
    if competition_value:
        if attachment_id or path_value:
            raise ValueError("competition cannot be combined with attachment_id or path")
        return None, _managed_competition_training_source(competition_value, args, context)
    if attachment_id:
        row = context.store.get_attachment(attachment_id)
        if not row or row.get("run_id") != context.session_id:
            raise ValueError("attachment is not bound to this run")
        source = Path(str(row["path"])).resolve(strict=True)
        source.relative_to(context.workspace_root.resolve(strict=True))
        if path_value:
            supplied = _path(args, "path", context).resolve(strict=True)
            if supplied != source:
                raise ValueError("attachment_id and path refer to different files")
        metadata = {
            "attachment_id": attachment_id,
            "name": str(row.get("name") or source.name),
            "media_type": str(row.get("media_type") or ""),
            "sha256": str(row.get("sha256") or ""),
        }
    elif path_value:
        source = _path(args, "path", context).resolve(strict=True)
        metadata = {"attachment_id": "", "name": source.name, "media_type": mimetypes.guess_type(source.name)[0] or ""}
    else:
        return None, {"attachment_id": "", "name": "", "media_type": "", "sha256": ""}
    if not source.is_file() or source.is_symlink():
        raise ValueError("training source must be a regular file")
    metadata.update({
        "path": source.relative_to(context.workspace_root.resolve(strict=True)).as_posix(),
        "bytes": source.stat().st_size,
        "sha256": metadata.get("sha256") or _sha256_file(source),
    })
    return source, metadata


def _tabular_profile(path: Path, target_column: str) -> dict[str, Any]:
    suffix = path.suffix.casefold()
    delimiter = "\t" if suffix in {".tsv", ".tab"} else ","
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(64 * 1024)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|") if sample else None
            delimiter = dialect.delimiter if dialect else delimiter
        except csv.Error:
            pass
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, [])
        rows = 0
        complete = True
        for _row in reader:
            rows += 1
            if rows >= TRAINING_HPC_THRESHOLDS["rows"]:
                complete = next(reader, None) is None
                break
    columns = [str(item).strip() for item in header]
    feature_count = max(0, len(columns) - (1 if target_column and target_column in columns else 0))
    return {
        "format": "tabular",
        "delimiter": delimiter,
        "columns": len(columns),
        "feature_count": feature_count,
        "target_column_present": bool(target_column and target_column in columns),
        "row_count": rows if complete else None,
        "row_count_lower_bound": rows,
        "row_count_complete": complete,
        "header": columns[:64],
        "header_truncated": len(columns) > 64,
    }


_TASK_NEGATION_MARKERS = (
    "do not ", "don't ", "not use ", "no ", "without ", "disabled ",
    "禁止", "不得", "不使用", "不用", "不调用", "不训练", "禁用", "无需", "不要",
)


def _task_term_requested(task: str, term: str) -> bool:
    """Return True only when at least one occurrence is not locally negated.

    Compute-route prompts routinely describe forbidden fallback paths (for
    example, ``CPU-only; do not use GPU``).  Treating those safety clauses as
    positive accelerator requests incorrectly forces a small CPU job onto HPC.
    Keep the check deliberately local so a separate positive clause such as
    ``do not use local GPU; use managed HPC`` still selects HPC.
    """
    start = 0
    while True:
        index = task.find(term, start)
        if index < 0:
            return False
        prefix = task[max(0, index - 32) : index]
        local_prefix = re.split(r"[;；,.，。!?！？\n]", prefix)[-1]
        suffix = task[index + len(term) : index + len(term) + 24]
        locally_negated = any(marker in local_prefix for marker in _TASK_NEGATION_MARKERS)
        if term == "gpu" and (
            suffix.startswith("_allowed=false")
            or suffix.lstrip().startswith(("disabled", "禁用"))
        ):
            locally_negated = True
        if not locally_negated:
            return True
        start = index + len(term)


def _training_route(args: dict[str, Any], context: ToolContext) -> ToolResult:
    if "siim_calibration" in context.metadata:
        from .siim_calibration_control import training_route as siim_route
        return siim_route(args, context)
    if "official_calibration" in context.metadata:
        from .ev_calibration_control import training_route
        return training_route(args, context)
    source, source_metadata = _training_source(args, context)
    task = " ".join(str(args.get("task_description") or "").casefold().split())
    target_column = str(args.get("target_column") or "").strip()
    suffix = source.suffix.casefold() if source else ""
    media_type = str(source_metadata.get("media_type") or "").casefold()
    tabular = bool(source and (suffix in {".csv", ".tsv", ".tab"} or media_type in {"text/csv", "text/tab-separated-values"}))
    managed_competition = source_metadata.get("source_kind") == "managed_competition_data"
    profile: dict[str, Any] = _tabular_profile(source, target_column) if source and tabular else {
        "format": "managed_competition_data" if managed_competition else ("non_tabular" if source else "task_only"),
        "feature_count": None,
        "row_count": None,
        "row_count_lower_bound": None,
        "row_count_complete": False,
    }
    if managed_competition:
        profile.update({
            "competition": source_metadata["competition"],
            "data_status": source_metadata["data_status"],
            "file_count": source_metadata["file_count"],
            "bytes": source_metadata["bytes"],
            "manifest_sha256": source_metadata["manifest_sha256"],
            "local_copy_created": False,
        })

    modality_terms = (
        "image", "vision", "audio", "video", "图像", "图片", "视觉", "音频", "视频",
    )
    deep_learning_terms = (
        "transformer", "finetune", "fine-tune", "fine tune", "deep learning", "neural network",
        "cnn", "rnn", "lstm", "bert", "llm", "lm_eval", "mmlu", "huggingface", "pretrained",
        "checkpoint", "language model", "微调", "深度学习", "神经网络", "大模型", "语言模型",
        "模型评估", "模型推理", "检查点",
    )
    search_terms = (
        "hyperparameter", "grid search", "optuna", "multi-seed", "multiseed", "multi seed",
        "cross-validation", "cross validation", "multi-fold", "multifold", "多折", "多种子", "超参数",
    )
    explicit_gpu_terms = ("hpc", "gpu", "cuda", "显卡", "加速训练")
    reasons: list[dict[str, Any]] = []
    if managed_competition:
        reasons.append({
            "code": "managed_competition_data",
            "observed": source_metadata["competition"],
        })
    if any(_task_term_requested(task, term) for term in modality_terms):
        reasons.append({"code": "accelerator_modality", "observed": True})
    if any(_task_term_requested(task, term) for term in deep_learning_terms):
        reasons.append({"code": "deep_learning_requested", "observed": True})
    if any(_task_term_requested(task, term) for term in search_terms):
        reasons.append({"code": "search_or_repeated_fit_requested", "observed": True})
    if any(_task_term_requested(task, term) for term in explicit_gpu_terms):
        reasons.append({"code": "gpu_explicitly_requested", "observed": True})
    row_lower_bound = profile.get("row_count_lower_bound")
    feature_count = profile.get("feature_count")
    byte_count = int(source_metadata.get("bytes") or 0)
    if isinstance(row_lower_bound, int) and row_lower_bound >= TRAINING_HPC_THRESHOLDS["rows"]:
        reasons.append({"code": "row_threshold", "observed": row_lower_bound, "threshold": TRAINING_HPC_THRESHOLDS["rows"]})
    if isinstance(feature_count, int) and feature_count >= TRAINING_HPC_THRESHOLDS["features"]:
        reasons.append({"code": "feature_threshold", "observed": feature_count, "threshold": TRAINING_HPC_THRESHOLDS["features"]})
    if byte_count >= TRAINING_HPC_THRESHOLDS["bytes"]:
        reasons.append({"code": "byte_threshold", "observed": byte_count, "threshold": TRAINING_HPC_THRESHOLDS["bytes"]})

    decision = "hpc_gpu" if reasons else "local_cpu"
    required_steps = (
        ["training_route", "hpc_verify_5_of_5", "hpc_execute_solution", "artifact_collect", "independent_verify"]
        if decision == "hpc_gpu"
        else ["training_route", "local_cpu_train", "artifact_publish", "independent_verify"]
    )
    payload = {
        "schema": "evomind.training.compute_route.v1",
        "decision": decision,
        "source": source_metadata,
        "profile": profile,
        "reasons": reasons or [{"code": "bounded_tabular_cpu_fit", "observed": True}],
        "thresholds": dict(TRAINING_HPC_THRESHOLDS),
        "local_gpu_allowed": False,
        "hpc_required": decision == "hpc_gpu",
        "hpc_fallback": "blocked_recoverable_no_local_fallback" if decision == "hpc_gpu" else "not_applicable",
        "required_steps": required_steps,
        "generated_at": utc_now(),
    }
    evidence_key = str(source_metadata.get("sha256") or hashlib.sha256(task.encode("utf-8")).hexdigest())[:16]
    payload["evidence"] = _write_evidence(context, f"training_route_{evidence_key}.json", payload)
    context.store.append_event(context.session_id, "data_profiled", {
        "run_id": context.session_id,
        "source": source_metadata,
        "profile": profile,
    })
    context.store.append_event(context.session_id, "compute_route_selected", {
        "run_id": context.session_id,
        "decision": decision,
        "reasons": payload["reasons"],
        "required_steps": required_steps,
    })
    summary = "selected managed HPC GPU" if decision == "hpc_gpu" else "selected local CPU"
    return ToolResult("", True, payload, summary)


def _hpc_verify(args: dict[str, Any], context: ToolContext) -> ToolResult:
    try:
        job_id, profile = _required_managed_hpc_identity(args, context)
    except ValueError:
        return ToolResult("", False, {}, "managed job/profile binding is invalid", error="profile_job_binding_invalid")
    from xsci.terminal_tools import (
        _live_hpc_connection_probe,
        hpc_identity_evidence_complete,
    )
    with HPC_PROFILE_LOCK:
        evidence = _live_hpc_connection_probe(profile, job_id, sample_count=5)
    ok = hpc_identity_evidence_complete(
        evidence,
        expected_sample_count=5,
        expected_profile=profile,
        expected_job_id=job_id,
    )
    error = "" if ok else str(evidence.get("status") or "hpc_identity_evidence_incomplete")
    if not ok and error == "job_container_verified":
        error = "hpc_identity_evidence_incomplete"
    return ToolResult("", ok, evidence, "HPC job container passed the complete 5/5 identity gate" if ok else "HPC identity evidence is incomplete or not verified", error=error)


def _hpc_asset_probe_remote_source() -> str:
    """Return the fixed metadata-only program for the verified job container."""

    return textwrap.dedent(
        r'''
        import hashlib
        import importlib.machinery
        import json
        import os
        import re
        import stat
        import sys

        root = os.path.realpath(sys.argv[1])
        if not os.path.isabs(root) or root == "/" or ".." in root.split("/"):
            raise SystemExit(20)

        def within(path):
            value = os.path.realpath(path)
            try:
                return os.path.normcase(os.path.commonpath((value, root))) == os.path.normcase(root)
            except ValueError:
                return False

        def safe(path):
            current = root
            relative = os.path.relpath(path, root)
            if relative == ".." or relative.startswith("../"):
                raise SystemExit(21)
            for part in (() if relative == "." else relative.split(os.sep)):
                current = os.path.join(current, part)
                try:
                    mode = os.lstat(current).st_mode
                except FileNotFoundError:
                    continue
                if stat.S_ISLNK(mode):
                    raise SystemExit(22)
            resolved = os.path.realpath(path)
            if not within(resolved):
                raise SystemExit(23)
            return resolved

        def relative(path):
            return os.path.relpath(path, root).replace(os.sep, "/")

        skipped_optional_paths = []

        def safe_optional(path):
            try:
                return safe(path)
            except SystemExit as exc:
                if int(exc.code or 0) in (22, 23):
                    skipped_optional_paths.append(int(exc.code or 0))
                    return None
                raise

        def metadata(path, base):
            row = os.stat(path, follow_symlinks=False)
            if not stat.S_ISREG(row.st_mode):
                raise SystemExit(24)
            leaf = os.path.basename(path).casefold()
            credential_name = re.compile(
                r"^(?:\.env(?:\..*)?|credentials?(?:\..*)?|passwords?(?:\..*)?|passwds?(?:\..*)?|"
                r"secrets?(?:\..*)?|tokens?(?:\.(?:txt|json|key))?|api[_-]?keys?(?:\..*)?|"
                r"id_(?:rsa|ed25519)(?:\..*)?|.*private[_-]?keys?.*|cookies?(?:\..*)?)$"
            )
            if credential_name.fullmatch(leaf):
                raise SystemExit(25)
            return {
                "path": os.path.relpath(path, base).replace(os.sep, "/"),
                "bytes": int(row.st_size),
                "mtime_ns": int(row.st_mtime_ns),
                "regular_file": True,
                "content_read": False,
            }

        runtime = safe(os.path.join(root, "competition_data", ".runtime", "mindgames"))
        site_candidates = [
            os.path.join(runtime, "site-packages"),
            os.path.join(runtime, "Lib", "site-packages"),
        ]
        for library in (os.path.join(runtime, "lib"), os.path.join(runtime, "lib64")):
            if os.path.isdir(library):
                for name in sorted(os.listdir(library)):
                    if name.casefold().startswith("python"):
                        site_candidates.append(os.path.join(library, name, "site-packages"))
        sites = []
        for path in site_candidates:
            if os.path.isdir(path):
                bounded = safe_optional(path)
                if bounded is not None:
                    sites.append(bounded)
        packages = []
        for package_name in ("textarena", "trueskill", "transformers", "torch"):
            item = {"name": package_name, "available": False, "version": None, "origin": []}
            for site in sites:
                try:
                    spec = importlib.machinery.PathFinder.find_spec(package_name, [site])
                except (AttributeError, ImportError, ValueError):
                    continue
                if spec is None:
                    continue
                origins = list(spec.submodule_search_locations or [])
                if spec.origin and spec.origin not in ("built-in", "frozen"):
                    origins.append(spec.origin)
                bounded = [safe(path) for path in origins]
                version = None
                prefix = package_name.casefold().replace("-", "_") + "_"
                for entry in sorted(os.listdir(site)):
                    folded = entry.casefold().replace("-", "_")
                    if folded.startswith(prefix) and folded.endswith(".dist_info"):
                        candidate_version = entry[: -len(".dist-info")][len(package_name) + 1 :]
                        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+!-]{0,127}", candidate_version):
                            version = candidate_version
                            break
                item = {"name": package_name, "available": True, "version": version, "origin": [relative(path) for path in bounded]}
                break
            packages.append(item)

        models = []
        model_paths = (
            "competition_data/.runtime/mindgames/models/Qwen3-8B",
            "competition_data/.runtime/mindgames/models/STARS",
            "competition_data/.runtime/mindgames/models/tungsten",
            "competition_data/cure_bench/managed_runtime/models/Qwen3-8B",
            "competition_data/cure_bench/managed_runtime/models/gpt-oss-20b",
            "competition_data/cure_bench/managed_runtime/models/GPT-OSS-20B",
            "competition_data/.runtime/cure_bench/models/gpt-oss-20b",
            "models/gpt-oss-20b",
        )
        for candidate in model_paths:
            path = safe_optional(os.path.join(root, *candidate.split("/")))
            if path is None or not os.path.isdir(path):
                continue
            try:
                file_count = 0
                total_bytes = 0
                closure = hashlib.sha256()
                closure.update(b"[")
                first_file = True
                for directory, names, leaves in os.walk(path, followlinks=False):
                    safe(directory)
                    names[:] = sorted(names)
                    for name in sorted(leaves):
                        child = safe(os.path.join(directory, name))
                        row = metadata(child, path)
                        file_count += 1
                        if file_count > 100000:
                            raise SystemExit(26)
                        total_bytes += row["bytes"]
                        if not first_file:
                            closure.update(b",")
                        closure.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
                        first_file = False
                closure.update(b"]")
            except SystemExit as exc:
                if int(exc.code or 0) in (22, 23):
                    skipped_optional_paths.append(int(exc.code or 0))
                    continue
                raise
            models.append({
                "path": relative(path),
                "file_count": file_count,
                "total_bytes": total_bytes,
                "metadata_closure_sha256": closure.hexdigest(),
                "metadata_closure_complete": True,
                "content_sha256_computed": False,
                "file_details_included": False,
            })

        manifest_candidates = []
        fixed_manifests = (
            "competition_data/.runtime/mindgames/formal-protocol.json",
            "competition_data/.runtime/mindgames/primary-baseline.json",
            "competition_data/.runtime/mindgames/holdout-ledger.json",
            "competition_data/cure_bench/.evomind/cure-bench/frozen-encoder.json",
            "competition_data/cure_bench/managed_runtime/frozen-encoder.json",
            "competition_data/cure_bench/.evomind/cure-bench/holdout-ledger.json",
            "competition_data/cure_bench/managed_runtime/cure-bench-primary-baseline.json",
            "competition_data/cure_bench/.evomind/cure-bench/primary-baseline-manifest.json",
        )
        for candidate in fixed_manifests:
            path = safe_optional(os.path.join(root, *candidate.split("/")))
            if path is not None and os.path.isfile(path):
                row = metadata(path, root)
                row["role"] = (
                    "formal" if candidate.endswith("formal-protocol.json") else
                    "encoder" if candidate.endswith("frozen-encoder.json") else
                    "ledger" if candidate.endswith("holdout-ledger.json") else
                    "primary"
                )
                manifest_candidates.append(row)

        print(json.dumps({
            "schema": "evomind.g21_hpc_asset_probe.v1",
            "scope": "cure_mindgames",
            "packages": packages,
            "models": models,
            "manifest_candidates": manifest_candidates,
            "skipped_optional_paths": len(skipped_optional_paths),
            "allowed_remote_root_match": True,
            "read_only": True,
            "network_access": False,
            "file_contents_read": False,
            "test_labels_used": False,
            "secret_values_logged": False,
            "signals_sent": 0,
            "remote_writes": 0,
            "other_processes_modified": False,
            "training_started": False,
            "worker_control": 0,
        }, sort_keys=True, separators=(",", ":")))
        '''
    ).strip()


def _parse_hpc_asset_probe_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != "evomind.g21_hpc_asset_probe.v1":
        raise ValueError("asset probe schema is invalid")
    expected_keys = {
        "schema", "scope", "packages", "models", "manifest_candidates", "skipped_optional_paths",
        "allowed_remote_root_match", "read_only", "network_access", "file_contents_read",
        "test_labels_used", "secret_values_logged", "signals_sent", "remote_writes",
        "other_processes_modified",
        "training_started", "worker_control",
    }
    if set(value) != expected_keys:
        raise ValueError("asset probe output fields are invalid")
    if value.get("scope") != HPC_ASSET_PROBE_SCOPE:
        raise ValueError("asset probe scope is invalid")
    if value.get("allowed_remote_root_match") is not True or value.get("read_only") is not True:
        raise ValueError("asset probe remote-root boundary is invalid")
    for key in ("network_access", "file_contents_read", "test_labels_used", "secret_values_logged", "other_processes_modified"):
        if value.get(key) is not False:
            raise ValueError("asset probe safety boundary is invalid")
    if value.get("signals_sent") != 0 or value.get("remote_writes") != 0 or value.get("worker_control") != 0 or value.get("training_started") is not False:
        raise ValueError("asset probe side-effect boundary is invalid")
    if not isinstance(value.get("skipped_optional_paths"), int) or not 0 <= int(value["skipped_optional_paths"]) <= 64:
        raise ValueError("asset probe skipped-path count is invalid")
    packages = value.get("packages")
    if not isinstance(packages, list) or [row.get("name") for row in packages if isinstance(row, dict)] != list(HPC_ASSET_PROBE_PACKAGE_NAMES):
        raise ValueError("asset probe package projection is invalid")
    for row in packages:
        if set(row) != {"name", "available", "version", "origin"} or not isinstance(row.get("available"), bool):
            raise ValueError("asset probe package row is invalid")
        if row.get("version") is not None and not isinstance(row.get("version"), str):
            raise ValueError("asset probe package version is invalid")
        origins = row.get("origin")
        if not isinstance(origins, list) or any(not isinstance(path, str) or path.startswith("/") or ".." in Path(path).parts for path in origins):
            raise ValueError("asset probe package origin is invalid")
    models = value.get("models")
    manifests = value.get("manifest_candidates")
    if not isinstance(models, list) or not isinstance(manifests, list) or len(models) > 8 or len(manifests) > 16:
        raise ValueError("asset probe collection is invalid")

    def safe_relative(path: Any) -> bool:
        return isinstance(path, str) and bool(path) and not path.startswith("/") and ".." not in Path(path).parts and len(path) <= 4096

    for model in models:
        if (
            not isinstance(model, dict)
            or set(model) != {
                "path", "file_count", "total_bytes", "metadata_closure_sha256",
                "metadata_closure_complete", "content_sha256_computed", "file_details_included",
            }
            or not safe_relative(model.get("path"))
            or model.get("metadata_closure_complete") is not True
            or model.get("content_sha256_computed") is not False
            or model.get("file_details_included") is not False
            or not isinstance(model.get("file_count"), int)
            or not isinstance(model.get("total_bytes"), int)
            or model.get("file_count") < 0
            or model.get("file_count") > 100_000
            or model.get("total_bytes") < 0
        ):
            raise ValueError("asset probe model closure is invalid")
        if not _SHA256_PATTERN.fullmatch(str(model.get("metadata_closure_sha256") or "")):
            raise ValueError("asset probe model metadata SHA is invalid")
        if model["file_count"] == 0 and model["total_bytes"] != 0:
            raise ValueError("asset probe model totals are invalid")
    for row in manifests:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "bytes", "mtime_ns", "regular_file", "content_read", "role"}
            or not safe_relative(row.get("path"))
            or not isinstance(row.get("bytes"), int)
            or not isinstance(row.get("mtime_ns"), int)
            or row.get("bytes") < 0
            or row.get("regular_file") is not True
            or row.get("content_read") is not False
        ):
            raise ValueError("asset probe manifest row is invalid")
        if row.get("role") not in {"formal", "encoder", "primary", "ledger"}:
            raise ValueError("asset probe manifest role is invalid")
    return dict(value)


def _hpc_asset_probe(args: dict[str, Any], context: ToolContext) -> ToolResult:
    safe = {
        "schema": "evomind.g21_hpc_asset_probe.v1",
        "scope": HPC_ASSET_PROBE_SCOPE,
        "read_only": True,
        "network_access": False,
        "file_contents_read": False,
        "test_labels_used": False,
        "secret_values_logged": False,
        "signals_sent": 0,
        "remote_writes": 0,
        "other_processes_modified": False,
        "training_started": False,
        "worker_control": 0,
        "binding_unchanged": False,
        "same_connection_identity_verified": False,
        "remote_commands_attempted": 0,
        "local_fallback_used": False,
        "failure_stage": "arguments",
        "failure_code": "",
        "remote_exit_code": None,
        "stdout_bytes": 0,
        "stderr_present": False,
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
    }
    if set(args) - {"scope"} or str(args.get("scope") or HPC_ASSET_PROBE_SCOPE) != HPC_ASSET_PROBE_SCOPE:
        return ToolResult("", False, safe, "HPC asset probe arguments are fixed", error="asset_probe_arguments_invalid")
    stage = "managed_binding"
    try:
        binding_before = _competition_progress_binding_fingerprint(context)
        job_id, profile = _required_managed_hpc_identity({}, context)
    except ValueError:
        return ToolResult("", False, safe, "managed job/profile binding is invalid", error="profile_job_binding_invalid")
    stage = "hpc_verify"
    verified = _hpc_verify({}, context)
    try:
        from xsci.terminal_tools import hpc_identity_evidence_complete

        samples = verified.content.get("samples") if isinstance(verified.content, dict) else None
        complete_5_of_5 = bool(
            verified.ok
            and isinstance(samples, list)
            and len(samples) == 5
            and all(isinstance(sample, dict) and sample.get("complete") is True for sample in samples)
            and hpc_identity_evidence_complete(
                verified.content,
                expected_sample_count=5,
                expected_profile=profile,
                expected_job_id=job_id,
            )
        )
    except Exception:
        complete_5_of_5 = False
    if not complete_5_of_5:
        safe["hpc_identity_gate"] = "failed_closed"
        return ToolResult("", False, safe, "HPC asset probe requires the complete current 5/5 identity gate", error="hpc_identity_evidence_incomplete")
    client = None
    try:
        from research_agent_workstation.server.core.gpu_credentials import (
            ALLOWED_GPU_REMOTE_ROOT,
            connect_ssh,
            verify_job_container_identity,
        )

        stage = "binding_recheck"
        binding_after = _competition_progress_binding_fingerprint(context)
        current_job, current_profile = _required_managed_hpc_identity({}, context)
        if binding_after != binding_before or current_job != job_id or current_profile != profile:
            raise ValueError("managed HPC run binding changed during asset verification")
        safe["binding_unchanged"] = True
        config = _load_bound_hpc_config(profile, job_id, context.session_id)
        remote_root = str(getattr(config, "remote_workspace", "") or "").rstrip("/")
        if (
            not re.fullmatch(r"/[A-Za-z0-9._/-]{3,240}", remote_root)
            or ".." in Path(remote_root).parts
            or remote_root == "/"
            or remote_root.rstrip("/") != ALLOWED_GPU_REMOTE_ROOT.rstrip("/")
        ):
            raise ValueError("allowed remote root is invalid")
        stage = "ssh_connect"
        client = connect_ssh(config, timeout=30)
        stage = "same_connection_identity"
        same = verify_job_container_identity(client, config, expected_job_id=job_id)
        required = (
            "designated_proxy_path_verified", "pinned_gateway_host_key_verified",
            "allocation_role_authenticated", "expected_host_uuid_match", "expected_gpu_uuid_match",
            "expected_gpu_model_and_memory_match", "allowed_remote_root_match", "job_container_verified",
        )
        if (
            not all(same.get(name) is True for name in required)
            or same.get("read_only") is not True
            or int(same.get("signals_sent") or 0) != 0
            or int(same.get("remote_writes") or 0) != 0
            or same.get("other_processes_modified") is not False
        ):
            raise ValueError("same-connection identity evidence is incomplete")
        if (
            int(same.get("job_id") or 0) != job_id
            or str(same.get("credential_profile") or "") != profile
            or str(same.get("remote_root") or "").rstrip("/") != remote_root
            or str(same.get("host_uuid") or "") != str(verified.content.get("host_uuid") or "")
            or list(same.get("gpu_uuids") or []) != list(verified.content.get("gpu_uuids") or [])
            or str(same.get("gpu_name") or "") != str(verified.content.get("gpu_name") or "")
            or int(same.get("gpu_memory_total_mib") or 0) != int(verified.content.get("gpu_memory_total_mib") or 0)
            or str(getattr(config, "expected_host_uuid", "") or "") != str(same.get("host_uuid") or "")
            or str(getattr(config, "expected_gpu_uuid", "") or "") not in list(same.get("gpu_uuids") or [])
            or int(getattr(config, "job_id", 0) or 0) != job_id
            or str(getattr(config, "credential_profile", "") or "") != profile
        ):
            raise ValueError("same-connection identity binding changed")
        safe["same_connection_identity_verified"] = True
        stage = "remote_execute"
        source = _hpc_asset_probe_remote_source()
        encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
        remote_code = f"import base64;exec(base64.b64decode({encoded!r}))"
        command = "python3 -I -S -c " + shlex.quote(remote_code) + " " + shlex.quote(remote_root)
        safe["remote_commands_attempted"] = 1
        _stdin, stdout, stderr = client.exec_command(command, timeout=45)
        raw_output = stdout.read(262_145)
        raw_error = stderr.read(4_097)
        exit_code = stdout.channel.recv_exit_status()
        if isinstance(raw_output, str):
            raw_output = raw_output.encode("utf-8", "replace")
        if isinstance(raw_error, str):
            raw_error = raw_error.encode("utf-8", "replace")
        safe.update({
            "remote_exit_code": int(exit_code),
            "stdout_bytes": len(raw_output),
            "stderr_present": bool(raw_error),
            "stderr_sha256": hashlib.sha256(raw_error).hexdigest(),
        })
        stage = "remote_output_validation"
        if len(raw_output) > 262_144 or raw_error or exit_code != 0 or not raw_output.strip():
            raise ValueError("asset probe remote output is invalid")
        lines = raw_output.decode("utf-8", "strict").strip().splitlines()
        if len(lines) != 1:
            raise ValueError("asset probe remote output line count is invalid")
        stage = "payload_validation"
        payload = _parse_hpc_asset_probe_payload(json.loads(lines[0]))
        payload.update({
            "hpc_identity_gate": "passed_5_of_5",
            "samples_requested": 5,
            "samples_passed": 5,
            "same_connection_identity_verified": True,
            "binding_unchanged": True,
            "remote_commands_attempted": 1,
            "local_fallback_used": False,
            "stdout_bytes": len(raw_output),
            "stderr_present": bool(raw_error),
            "stderr_sha256": hashlib.sha256(raw_error).hexdigest(),
        })
        return ToolResult("", True, payload, "G21 managed asset metadata sampled read-only")
    except Exception as exc:
        safe["hpc_identity_gate"] = "passed_5_of_5"
        safe["failure_stage"] = stage
        known_failure_codes = {
            "managed HPC run binding changed during asset verification": "binding_changed",
            "allowed remote root is invalid": "remote_root_invalid",
            "same-connection identity evidence is incomplete": "identity_incomplete",
            "same-connection identity binding changed": "identity_changed",
            "asset probe remote output is invalid": "remote_output_invalid",
            "asset probe remote output line count is invalid": "remote_output_line_count",
        }
        safe["failure_code"] = known_failure_codes.get(str(exc), f"{type(exc).__name__.lower()}_fail_closed")
        return ToolResult("", False, safe, "HPC asset probe failed closed", error=f"asset_probe:{type(exc).__name__}")
    finally:
        if client is not None:
            client.close()


def _profile_environment(profile: str, job_id: int, run_id: str):
    names = {
        "EVOMIND_HPC_CREDENTIAL_PROFILE": profile,
        "EVOMIND_SIIM_HPC_JOB_ID": str(job_id),
        "EVOMIND_SIIM_RUN_ID": run_id,
    }
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update(names)
    return previous


def _required_managed_hpc_identity(args: dict[str, Any], context: ToolContext | Any | None = None) -> tuple[int, str]:
    """Resolve the service-bound allocation without allowing tool-call retargeting."""
    metadata = getattr(context, "metadata", None)
    generation_alias = ""
    if isinstance(metadata, dict) and "managed_hpc_identity" in metadata:
        identity = metadata.get("managed_hpc_identity")
        if not isinstance(identity, dict):
            raise ValueError("managed HPC run binding is unavailable")
        raw_job = str(identity.get("job_id") or "").strip()
        bound_profile = str(identity.get("credential_profile") or "").strip()
        tenant_id = str(identity.get("tenant_id") or "").strip()
        owner = str(identity.get("owner_principal_id") or "").strip()
        generation_text = str(identity.get("allocation_generation") or "").strip()
        profile_instance_id = str(identity.get("profile_instance_id") or "").strip()
        allocation_binding_id = str(identity.get("allocation_binding_id") or "").strip()
        if (
            not re.fullmatch(r"tenant_[a-f0-9]{24}", tenant_id)
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", owner)
            or not generation_text.isdigit()
            or int(generation_text) <= 0
            or not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", profile_instance_id, re.IGNORECASE)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}", allocation_binding_id)
        ):
            raise ValueError("managed HPC run binding is invalid")
        if bound_profile != f"{tenant_id}_job{raw_job}_g{generation_text}":
            raise ValueError("managed HPC run profile identity is invalid")
        generation_alias = f"g{generation_text}"
    else:
        raw_job = str(os.environ.get("EVOMIND_SIIM_HPC_JOB_ID") or "").strip()
        bound_profile = str(os.environ.get("EVOMIND_HPC_CREDENTIAL_PROFILE") or "").strip()
    if not raw_job.isdigit() or int(raw_job) <= 0:
        raise ValueError("managed HPC job is unavailable")
    bound_job = int(raw_job)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", bound_profile):
        raise ValueError("managed HPC profile is unavailable")
    if not re.search(rf"(?:^|_)job{bound_job}(?:_|$)", bound_profile):
        raise ValueError("managed HPC profile does not match the bound job")

    requested_job = int(args.get("job_id") or bound_job)
    requested_profile = str(args.get("credential_profile") or bound_profile).strip()
    profile_matches = requested_profile == bound_profile or (
        bool(generation_alias) and requested_profile.lower() == generation_alias
    )
    if requested_job != bound_job or not profile_matches:
        raise ValueError("tool call attempted to retarget the managed HPC allocation")
    return bound_job, bound_profile


def _restore_environment(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _load_bound_hpc_config(profile: str, job_id: int, run_id: str):
    from research_agent_workstation.server.core.gpu_credentials import load_gpu_ssh_config
    with HPC_PROFILE_LOCK:
        previous = _profile_environment(profile, job_id, run_id)
        try:
            return load_gpu_ssh_config(strict_named_profile=True)
        finally:
            _restore_environment(previous)


_COMPETITION_PROGRESS_NAMES = {
    "cure_bench", "e2lmc", "mindgames", "ariel_2025", "weather4cast", "open_polymer",
}
_COMPETITION_PROGRESS_FAILURES = {
    "", "SFTP_AUTH_FAILED", "SFTP_TIMEOUT", "SFTP_CONNECTION_FAILED",
    "SFTP_ASKPASS_FAILED", "SFTP_BATCH_FAILED", "SFTP_FAILED",
    "SFTP_BINARY_MISSING", "SFTP_CONTROLLED_PAYLOAD_INVALID", "SFTP_LISTING_REJECTED",
    "SFTP_LISTING_FAILED",
    "SFTP_PARALLEL_RETRY_EXHAUSTED",
    "WORKER_START_FAILED", "WEATHER_FILE_SET_INVALID", "WEATHER_LISTING_INCOMPLETE",
}
_COMPETITION_LISTING_REJECT_REASONS = {
    "total_like_format", "field_count", "symlink_type", "special_type",
    "unknown_type", "path", "leading_dash", "colon", "bracket", "percent",
    "ampersand", "hash", "other_leaf_charset", "leaf_length", "apostrophe",
    "double_quote", "semicolon", "exclamation", "tilde", "dollar", "backtick",
    "brace", "asterisk", "question", "control_character", "unicode_character",
    "other_ascii", "duplicate",
}
_COMPETITION_PROGRESS_REQUIRED_GATE = (
    "designated_proxy_path_verified",
    "pinned_gateway_host_key_verified",
    "allocation_role_authenticated",
    "expected_host_uuid_match",
    "expected_gpu_uuid_match",
    "expected_gpu_model_and_memory_match",
    "allowed_remote_root_match",
    "job_container_verified",
    "identity_consistent",
)


def _progress_child_pids(proc_root: Path, pid: int) -> list[int]:
    """Collect children created by every thread in one Linux process group."""

    task_root = proc_root / str(int(pid)) / "task"
    try:
        task_directories = tuple(task_root.iterdir())
    except OSError:
        return []
    children: set[int] = set()
    for task_directory in task_directories:
        if not task_directory.name.isdigit():
            continue
        try:
            value = (task_directory / "children").read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        for item in value.split():
            if item.isdigit():
                child = int(item)
                if child > 0 and child != int(pid):
                    children.add(child)
    return sorted(children)


def _competition_data_progress_remote_source(competition: str) -> str:
    """Return the frozen read-only sampler executed inside the verified container.

    The competition value is validated server-side but deliberately is not
    interpolated into the source.  The only remote argument is the fixed root
    derived by :func:`persistent_root`; callers cannot provide a path or a
    command.  This program never opens an SFTP client, sends a signal, or writes
    a file.  It emits aggregate numbers only.
    """

    if competition not in _COMPETITION_PROGRESS_NAMES:
        raise ValueError("competition progress source is not allowlisted")
    source = r'''from __future__ import annotations
import datetime
import hashlib
import json
import os
import pathlib
import re
import stat
import sys

root = pathlib.Path(sys.argv[1])
metadata_root = root / ".evomind" / "download-job"

def utc(value):
    if not value:
        return None
    return datetime.datetime.fromtimestamp(float(value), datetime.timezone.utc).isoformat()

def bounded_json(path):
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}

selector_path = metadata_root / "current-attempt.json"
selector = bounded_json(selector_path)
attempt_id = str(selector.get("attempt_id") or "")
selector_pid = selector.get("pid")
selector_valid = (
    selector.get("schema") == "evomind.weather4cast_current_attempt.v1"
    and re.fullmatch(r"[a-f0-9]{32}", attempt_id) is not None
    and isinstance(selector_pid, int)
    and selector_pid > 0
)
attempt_root = metadata_root / "attempts" / attempt_id if selector_valid else metadata_root
pid_path = attempt_root / "pid"
state_path = attempt_root / "state.json"
exit_path = attempt_root / "exit-code"
pid_value = None
try:
    if not pid_path.is_symlink() and pid_path.is_file() and pid_path.stat().st_size <= 32:
        pid_value = int(pid_path.read_text(encoding="utf-8").strip())
except Exception:
    pid_value = None
pid_records_match = selector_valid and pid_value == selector_pid and isinstance(pid_value, int)
state = bounded_json(state_path)
attempt_binding_valid = (
    selector_valid
    and pid_records_match
    and state.get("schema") == "evomind.weather4cast_download_state.v2"
    and state.get("attempt_id") == attempt_id
    and state.get("rules_accepted") is False
    and state.get("secret_values_logged") is False
)
failure_code = str(state.get("failure_code") or "")
allowed_failures = {
    "", "SFTP_AUTH_FAILED", "SFTP_TIMEOUT", "SFTP_CONNECTION_FAILED",
    "SFTP_ASKPASS_FAILED", "SFTP_BATCH_FAILED", "SFTP_FAILED",
    "SFTP_BINARY_MISSING", "SFTP_CONTROLLED_PAYLOAD_INVALID",
    "SFTP_LISTING_REJECTED", "SFTP_LISTING_FAILED", "SFTP_PARALLEL_RETRY_EXHAUSTED",
    "WORKER_START_FAILED", "WEATHER_FILE_SET_INVALID", "WEATHER_LISTING_INCOMPLETE",
}
if failure_code not in allowed_failures:
    failure_code = "OTHER_REDACTED"
exit_code = None
try:
    if not exit_path.is_symlink() and exit_path.is_file() and exit_path.stat().st_size <= 32:
        exit_code = int(exit_path.read_text(encoding="utf-8").strip())
except Exception:
    exit_code = None

def proc_stat(pid):
    raw = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close = raw.rfind(")")
    if close < 0:
        raise ValueError("invalid proc stat")
    fields = raw[close + 2:].split()
    return {
        "state": fields[0],
        "utime": int(fields[11]),
        "stime": int(fields[12]),
        "starttime": int(fields[19]),
    }

__EVOMIND_PROGRESS_CHILD_PIDS__

worker_stat = None
cmdline_matches = False
if pid_records_match:
    try:
        worker_stat = proc_stat(pid_value)
        tokens = [item for item in pathlib.Path(f"/proc/{pid_value}/cmdline").read_bytes().split(b"\0") if item]
        cmdline_matches = any(b"worker.sh" in item for item in tokens) and attempt_id.encode("ascii") in tokens
    except Exception:
        worker_stat = None
worker_alive = (
    attempt_binding_valid
    and worker_stat is not None
    and worker_stat.get("state") != "Z"
    and cmdline_matches
)

tree = []
seen = set()
if worker_alive:
    pending = [pid_value]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            proc_stat(current)
        except Exception:
            continue
        tree.append(current)
        pending.extend(_progress_child_pids(pathlib.Path("/proc"), current))

clock_ticks = int(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
cpu_ticks = 0
read_bytes = 0
write_bytes = 0
io_readable = bool(tree)
transfer_backend = "unknown"
transport_process_count = 0
sftp_buffer_argv_present = False
sftp_requests_argv_present = False
socket_inodes = set()
for process_id in tree:
    try:
        info = proc_stat(process_id)
        cpu_ticks += info["utime"] + info["stime"]
        try:
            tokens = [item for item in pathlib.Path(f"/proc/{process_id}/cmdline").read_bytes().split(b"\0") if item]
            executable = pathlib.PurePosixPath(tokens[0].decode("utf-8", "ignore")).name.casefold() if tokens else ""
            if executable == "lftp":
                transfer_backend = "lftp"
                transport_process_count += 1
            elif executable in {"sftp", "ssh", "sshpass"}:
                if transfer_backend == "unknown":
                    transfer_backend = "openssh_sftp"
                transport_process_count += 1
                if executable == "sftp":
                    for index, token in enumerate(tokens[:-1]):
                        if token == b"-B" and tokens[index + 1] == b"262144":
                            sftp_buffer_argv_present = True
                        if token == b"-R" and tokens[index + 1] == b"256":
                            sftp_requests_argv_present = True
        except Exception:
            pass
        try:
            for descriptor in pathlib.Path(f"/proc/{process_id}/fd").iterdir():
                matched = re.fullmatch(r"socket:\[(\d+)\]", os.readlink(descriptor))
                if matched:
                    socket_inodes.add(matched.group(1))
        except Exception:
            pass
        values = {}
        for line in pathlib.Path(f"/proc/{process_id}/io").read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                if key in {"read_bytes", "write_bytes"}:
                    values[key] = int(value.strip())
        if set(values) != {"read_bytes", "write_bytes"}:
            io_readable = False
        read_bytes += int(values.get("read_bytes", 0))
        write_bytes += int(values.get("write_bytes", 0))
    except Exception:
        io_readable = False

tcp_socket_count = 0
established_socket_count = 0
tcp_send_queue_bytes = 0
tcp_receive_queue_bytes = 0
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        rows = pathlib.Path(table).read_text(encoding="ascii").splitlines()[1:]
    except Exception:
        continue
    for row in rows:
        fields = row.split()
        if len(fields) < 10 or fields[9] not in socket_inodes:
            continue
        tcp_socket_count += 1
        if fields[3] == "01":
            established_socket_count += 1
        try:
            send_text, receive_text = fields[4].split(":", 1)
            tcp_send_queue_bytes += int(send_text, 16)
            tcp_receive_queue_bytes += int(receive_text, 16)
        except Exception:
            pass

worker_elapsed = None
if worker_stat is not None:
    try:
        uptime = float(pathlib.Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
        worker_elapsed = max(0.0, uptime - worker_stat["starttime"] / clock_ticks)
    except Exception:
        worker_elapsed = None

regular_files = 0
nonempty_files = 0
total_bytes = 0
newest_mtime = 0.0
partial_count = 0
partial_bytes = 0
partial_suffixes = (".part", ".partial", ".tmp", ".download", ".incomplete", ".lftp-pget-status")
stack = [root] if root.is_dir() and not root.is_symlink() else []
while stack:
    current = stack.pop()
    try:
        entries = list(os.scandir(current))
    except Exception:
        continue
    for entry in entries:
        try:
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                if entry.name != ".evomind":
                    stack.append(pathlib.Path(entry.path))
                continue
            info = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                continue
            size = int(info.st_size)
            regular_files += 1
            total_bytes += size
            if size > 0:
                nonempty_files += 1
            newest_mtime = max(newest_mtime, float(info.st_mtime))
            lower = entry.name.casefold()
            if lower.endswith(partial_suffixes) or lower.startswith(".lftp-") or ".part." in lower:
                partial_count += 1
                partial_bytes += size
        except Exception:
            continue

root_real = os.path.realpath(str(root))
open_files = set()
largest_open = 0
if worker_alive:
    for process_id in tree:
        try:
            descriptors = list(pathlib.Path(f"/proc/{process_id}/fd").iterdir())
        except Exception:
            continue
        for descriptor in descriptors:
            try:
                target = os.path.realpath(os.readlink(descriptor))
                if os.path.commonpath([root_real, target]) != root_real:
                    continue
                info = os.stat(target, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    continue
                identity = (int(info.st_dev), int(info.st_ino))
                if identity in open_files:
                    continue
                open_files.add(identity)
                largest_open = max(largest_open, int(info.st_size))
            except Exception:
                continue

state_mtime = 0.0
try:
    if not state_path.is_symlink() and state_path.is_file():
        state_mtime = float(state_path.stat().st_mtime)
except Exception:
    state_mtime = 0.0

worker_status = str(state.get("status") or "")
if worker_status not in {
    "", "RUNNING", "FULL_DATA_READY", "DOWNLOAD_FAILED",
    "VALIDATION_FAILED", "START_FAILED",
}:
    worker_status = "OTHER_REDACTED"
worker_phase = str(state.get("phase") or "")
if worker_phase not in {"", "downloading", "validating", "complete", "failed"}:
    worker_phase = "OTHER_REDACTED"

def adapter_count(key):
    value = state.get(key)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

print(json.dumps({
    "schema": "evomind.competition_data_progress.remote.v1",
    "sampled_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "regular_files": regular_files,
    "nonempty_files": nonempty_files,
    "total_bytes": total_bytes,
    "newest_mtime_utc": utc(newest_mtime),
    "partial_count": partial_count,
    "partial_bytes": partial_bytes,
    "largest_open_data_file_bytes": largest_open,
    "worker_alive": worker_alive,
    "worker_elapsed_seconds": worker_elapsed,
    "worker_status": worker_status,
    "worker_phase": worker_phase,
    "failure_code": failure_code,
    "worker_exit_code": exit_code,
    "worker_binding_valid": attempt_binding_valid,
    "process_tree_count": len(tree),
    "transfer_backend": transfer_backend,
    "adapter_transfer_mode": str(state.get("adapter_transfer_mode") or "legacy_unknown"),
    "adapter_requested_parallelism": adapter_count("adapter_requested_parallelism"),
    "adapter_listing_status": str(state.get("adapter_listing_status") or "legacy_unknown"),
    "adapter_listing_entries": adapter_count("adapter_listing_entries"),
    "adapter_listing_files": adapter_count("adapter_listing_files"),
    "adapter_listing_directories": adapter_count("adapter_listing_directories"),
    "adapter_listing_archive_files": adapter_count("adapter_listing_archive_files"),
    "adapter_listing_scientific_files": adapter_count("adapter_listing_scientific_files"),
    "adapter_listing_metadata_files": adapter_count("adapter_listing_metadata_files"),
    "adapter_listing_other_files": adapter_count("adapter_listing_other_files"),
    "adapter_listing_max_depth": adapter_count("adapter_listing_max_depth"),
    "adapter_listing_raw_lines": adapter_count("adapter_listing_raw_lines"),
    "adapter_listing_control_lines": adapter_count("adapter_listing_control_lines"),
    "adapter_listing_rejected_lines": adapter_count("adapter_listing_rejected_lines"),
    "adapter_listing_skipped_directories": adapter_count("adapter_listing_skipped_directories"),
    "adapter_listing_timeout_exhaustions": adapter_count("adapter_listing_timeout_exhaustions"),
    "adapter_parallel_failures": adapter_count("adapter_parallel_failures"),
    "adapter_retry_rounds": adapter_count("adapter_retry_rounds"),
    "adapter_pending_batches": adapter_count("adapter_pending_batches"),
    "adapter_fallback_reason": str(state.get("adapter_fallback_reason") or ""),
    "adapter_sftp_buffer_bytes": adapter_count("adapter_sftp_buffer_bytes"),
    "adapter_sftp_num_requests": adapter_count("adapter_sftp_num_requests"),
    "adapter_tuning_requested": (
        state.get("adapter_tuning_requested")
        if isinstance(state.get("adapter_tuning_requested"), bool)
        else False
    ),
    "adapter_tuning_argv_present": (
        sftp_buffer_argv_present and sftp_requests_argv_present
    ),
    "transport_process_count": transport_process_count,
    "tcp_socket_count": tcp_socket_count,
    "established_socket_count": established_socket_count,
    "tcp_send_queue_bytes": tcp_send_queue_bytes,
    "tcp_receive_queue_bytes": tcp_receive_queue_bytes,
    "cpu_ticks": cpu_ticks,
    "cpu_seconds": cpu_ticks / clock_ticks,
    "read_bytes": read_bytes,
    "write_bytes": write_bytes,
    "io_readable": io_readable,
    "open_data_files": len(open_files),
    "metadata_last_activity_utc": utc(state_mtime),
    "log_available": False,
    "read_only": True,
    "remote_writes": 0,
    "signals_sent": 0,
    "other_processes_modified": False,
}, separators=(",", ":")))
'''
    helper_source = textwrap.dedent(inspect.getsource(_progress_child_pids)).strip()
    return source.replace("__EVOMIND_PROGRESS_CHILD_PIDS__", helper_source)


def _competition_progress_binding_fingerprint(context: ToolContext) -> str:
    metadata = getattr(context, "metadata", None)
    identity = metadata.get("managed_hpc_identity") if isinstance(metadata, dict) else None
    if not isinstance(identity, dict):
        raise ValueError("managed HPC run binding is unavailable")
    selected = {
        key: identity.get(key)
        for key in (
            "tenant_id", "owner_principal_id", "job_id", "credential_profile",
            "allocation_generation", "profile_instance_id", "allocation_binding_id",
        )
    }
    return hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def _competition_progress_gate_complete(evidence: Any) -> bool:
    source = evidence if isinstance(evidence, dict) else {}
    return bool(
        all(source.get(name) is True for name in _COMPETITION_PROGRESS_REQUIRED_GATE)
        and int(source.get("samples_requested") or 0) == 5
        and int(source.get("samples_passed") or 0) == 5
        and source.get("read_only") is True
        and int(source.get("signals_sent") or 0) == 0
        and source.get("other_processes_modified") is False
    )


def _parse_competition_progress_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("competition progress payload must be an object")
    allowed = {
        "schema", "sampled_at_utc", "regular_files", "nonempty_files", "total_bytes",
        "newest_mtime_utc", "partial_count", "partial_bytes",
        "largest_open_data_file_bytes", "worker_alive", "worker_elapsed_seconds",
        "worker_status", "worker_phase", "failure_code", "worker_exit_code",
        "worker_binding_valid", "process_tree_count", "transfer_backend",
        "adapter_transfer_mode", "adapter_requested_parallelism",
        "adapter_listing_status", "adapter_listing_entries",
        "adapter_listing_files", "adapter_listing_directories",
        "adapter_listing_archive_files", "adapter_listing_scientific_files",
        "adapter_listing_metadata_files", "adapter_listing_other_files",
        "adapter_listing_max_depth",
        "adapter_listing_raw_lines", "adapter_listing_control_lines",
        "adapter_listing_rejected_lines", "adapter_listing_skipped_directories",
        "adapter_listing_timeout_exhaustions",
        "adapter_parallel_failures", "adapter_retry_rounds",
        "adapter_pending_batches", "adapter_fallback_reason",
        "adapter_sftp_buffer_bytes", "adapter_sftp_num_requests",
        "adapter_tuning_requested", "adapter_tuning_argv_present",
        "transport_process_count", "tcp_socket_count", "established_socket_count",
        "tcp_send_queue_bytes", "tcp_receive_queue_bytes", "cpu_ticks", "cpu_seconds",
        "read_bytes", "write_bytes", "io_readable", "open_data_files",
        "metadata_last_activity_utc", "log_available", "read_only", "remote_writes",
        "signals_sent", "other_processes_modified",
    }
    if set(payload) != allowed or payload.get("schema") != "evomind.competition_data_progress.remote.v1":
        raise ValueError("competition progress payload contract mismatch")
    integers = (
        "regular_files", "nonempty_files", "total_bytes", "partial_count", "partial_bytes",
        "largest_open_data_file_bytes", "process_tree_count", "cpu_ticks", "read_bytes",
        "write_bytes", "open_data_files", "transport_process_count", "tcp_socket_count",
        "established_socket_count", "tcp_send_queue_bytes", "tcp_receive_queue_bytes",
        "adapter_requested_parallelism", "adapter_listing_entries",
        "adapter_listing_files", "adapter_listing_directories",
        "adapter_listing_archive_files", "adapter_listing_scientific_files",
        "adapter_listing_metadata_files", "adapter_listing_other_files",
        "adapter_listing_max_depth",
        "adapter_listing_raw_lines", "adapter_listing_control_lines",
        "adapter_listing_rejected_lines", "adapter_listing_skipped_directories",
        "adapter_listing_timeout_exhaustions",
        "adapter_parallel_failures", "adapter_retry_rounds",
        "adapter_pending_batches", "adapter_sftp_buffer_bytes",
        "adapter_sftp_num_requests",
        "remote_writes", "signals_sent",
    )
    if any(
        isinstance(payload.get(key), bool)
        or not isinstance(payload.get(key), int)
        or int(payload[key]) < 0
        for key in integers
    ):
        raise ValueError("competition progress payload contains an invalid count")
    if int(payload["nonempty_files"]) > int(payload["regular_files"]):
        raise ValueError("competition progress nonempty count exceeds the file count")
    if (
        isinstance(payload.get("cpu_seconds"), bool)
        or not isinstance(payload.get("cpu_seconds"), (int, float))
        or not math.isfinite(float(payload["cpu_seconds"]))
        or float(payload["cpu_seconds"]) < 0
    ):
        raise ValueError("competition progress CPU total is invalid")
    elapsed = payload.get("worker_elapsed_seconds")
    if elapsed is not None and (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0
    ):
        raise ValueError("competition progress worker elapsed time is invalid")
    exit_code = payload.get("worker_exit_code")
    if exit_code is not None and (isinstance(exit_code, bool) or not isinstance(exit_code, int)):
        raise ValueError("competition progress worker exit code is invalid")
    if payload.get("worker_status") not in {
        "", "RUNNING", "FULL_DATA_READY", "DOWNLOAD_FAILED",
        "VALIDATION_FAILED", "START_FAILED", "OTHER_REDACTED",
    }:
        raise ValueError("competition progress worker status is invalid")
    if payload.get("worker_phase") not in {
        "", "downloading", "validating", "complete", "failed", "OTHER_REDACTED",
    }:
        raise ValueError("competition progress worker phase is invalid")
    for key in (
        "worker_alive", "worker_binding_valid", "io_readable", "log_available",
        "read_only", "other_processes_modified", "adapter_tuning_requested",
        "adapter_tuning_argv_present",
    ):
        if not isinstance(payload.get(key), bool):
            raise ValueError("competition progress payload contains an invalid boolean")
    if payload.get("failure_code") not in _COMPETITION_PROGRESS_FAILURES | {"OTHER_REDACTED"}:
        raise ValueError("competition progress failure class is invalid")
    if payload.get("transfer_backend") not in {"lftp", "openssh_sftp", "unknown"}:
        raise ValueError("competition progress transfer backend is invalid")
    if payload.get("adapter_transfer_mode") not in {
        "legacy_unknown", "starting", "lftp_parallel", "openssh_listing",
        "openssh_parallel", "openssh_parallel_files", "openssh_recursive_fallback",
        "openssh_recursive_get",
    }:
        raise ValueError("competition progress adapter transfer mode is invalid")
    if payload.get("adapter_listing_status") not in {
        "legacy_unknown", "not_run", "ok", "ok_with_skips", "timeout", "nonzero", "parse_rejected",
    }:
        raise ValueError("competition progress adapter listing status is invalid")
    if payload.get("adapter_fallback_reason") not in {
        "", "listing_timeout", "listing_nonzero", "listing_parse_rejected",
        "parallel_batch_failed", "fatal_parallel_failure", "parallel_retry_exhausted",
        "listing_structure_duplicate_directory", "listing_structure_depth",
        "listing_structure_invalid_relative", "listing_structure_duplicate_file",
        "listing_structure_entry_cap",
    } | {
        f"listing_parse_rejected_{reason}"
        for reason in _COMPETITION_LISTING_REJECT_REASONS
    }:
        raise ValueError("competition progress adapter fallback reason is invalid")
    if int(payload["adapter_requested_parallelism"]) > 16:
        raise ValueError("competition progress adapter parallelism exceeds the bound")
    if any(int(payload[key]) > 10_000 for key in (
        "adapter_listing_entries", "adapter_listing_files",
        "adapter_listing_directories", "adapter_listing_raw_lines",
        "adapter_listing_archive_files", "adapter_listing_scientific_files",
        "adapter_listing_metadata_files", "adapter_listing_other_files",
        "adapter_listing_max_depth", "adapter_listing_control_lines",
        "adapter_listing_rejected_lines", "adapter_listing_skipped_directories",
        "adapter_listing_timeout_exhaustions",
    )):
        raise ValueError("competition progress adapter listing count exceeds the bound")
    if int(payload["adapter_listing_control_lines"]) + int(payload["adapter_listing_rejected_lines"]) > int(
        payload["adapter_listing_raw_lines"]
    ):
        raise ValueError("competition progress adapter listing diagnostics are inconsistent")
    if int(payload["adapter_listing_files"]) + int(payload["adapter_listing_directories"]) != int(
        payload["adapter_listing_entries"]
    ):
        raise ValueError("competition progress adapter listing totals are inconsistent")
    if int(payload["adapter_listing_skipped_directories"]) > int(
        payload["adapter_listing_directories"]
    ):
        raise ValueError("competition progress skipped directory count is inconsistent")
    if int(payload["adapter_listing_timeout_exhaustions"]) != int(
        payload["adapter_listing_skipped_directories"]
    ):
        raise ValueError("competition progress listing timeout count is inconsistent")
    if (
        payload["adapter_listing_status"] == "ok_with_skips"
        and int(payload["adapter_listing_skipped_directories"]) == 0
    ):
        raise ValueError("competition progress skipped listing status lacks a skipped directory")
    if (
        payload["adapter_listing_status"] == "ok"
        and int(payload["adapter_listing_skipped_directories"]) != 0
    ):
        raise ValueError("competition progress complete listing status contradicts skipped directories")
    listing_profile_total = sum(
        int(payload[key])
        for key in (
            "adapter_listing_archive_files", "adapter_listing_scientific_files",
            "adapter_listing_metadata_files", "adapter_listing_other_files",
        )
    )
    # A zero total with a non-zero legacy file count is accepted because
    # pre-R110 worker state does not contain the aggregate profile fields.
    if listing_profile_total not in {0, int(payload["adapter_listing_files"])}:
        raise ValueError("competition progress adapter listing profile is inconsistent")
    if int(payload["adapter_parallel_failures"]) > 16:
        raise ValueError("competition progress adapter failure count exceeds the bound")
    if int(payload["adapter_retry_rounds"]) > 4:
        raise ValueError("competition progress adapter retry count exceeds the bound")
    if int(payload["adapter_pending_batches"]) > 16:
        raise ValueError("competition progress adapter pending batch count exceeds the bound")
    if int(payload["adapter_sftp_buffer_bytes"]) not in {0, 262144}:
        raise ValueError("competition progress adapter buffer tuning is invalid")
    if int(payload["adapter_sftp_num_requests"]) not in {0, 256}:
        raise ValueError("competition progress adapter request tuning is invalid")
    tuning_complete = (
        int(payload["adapter_sftp_buffer_bytes"]) == 262144
        and int(payload["adapter_sftp_num_requests"]) == 256
    )
    if payload["adapter_tuning_requested"] is not tuning_complete:
        raise ValueError("competition progress adapter tuning evidence is inconsistent")
    if payload.get("read_only") is not True or payload.get("remote_writes") != 0:
        raise ValueError("competition progress probe violated the read-only boundary")
    if payload.get("signals_sent") != 0 or payload.get("other_processes_modified") is not False:
        raise ValueError("competition progress probe violated the process boundary")
    for key in ("sampled_at_utc", "newest_mtime_utc", "metadata_last_activity_utc"):
        value = payload.get(key)
        if value is not None and re.fullmatch(r"[0-9T:+.\-Z]{20,40}", str(value)) is None:
            raise ValueError("competition progress timestamp is invalid")
    return dict(payload)


def _competition_data_progress(args: dict[str, Any], context: ToolContext) -> ToolResult:
    try:
        from .competition_data import normalize_competition, persistent_root
        from .hpc_runtime_overlay import HpcRuntime
        from research_agent_workstation.server.core.gpu_credentials import (
            connect_ssh,
            verify_job_container_identity,
        )

        competition = normalize_competition(str(args.get("competition") or ""))
        binding_before = _competition_progress_binding_fingerprint(context)
        job_id, profile = _required_managed_hpc_identity({}, context)
    except Exception as exc:
        return ToolResult(
            "", False,
            {"read_only": True, "remote_commands_attempted": 0, "remote_writes": 0,
             "signals_sent": 0, "other_processes_modified": False},
            "competition progress binding is invalid",
            error=f"progress_preflight:{type(exc).__name__}",
        )

    verified = _hpc_verify({}, context)
    if not verified.ok or not _competition_progress_gate_complete(verified.content):
        return ToolResult(
            "", False,
            {"read_only": True, "remote_commands_attempted": 0, "remote_writes": 0,
             "signals_sent": 0, "other_processes_modified": False,
             "hpc_identity_gate": "failed_closed"},
            "competition progress requires the complete current 5/5 identity gate",
            error="hpc_identity_evidence_incomplete",
        )

    client = None
    aggregation_commands = 0
    try:
        binding_after = _competition_progress_binding_fingerprint(context)
        current_job, current_profile = _required_managed_hpc_identity({}, context)
        if binding_after != binding_before or current_job != job_id or current_profile != profile:
            raise ValueError("managed HPC run binding changed during progress verification")
        config = _load_bound_hpc_config(profile, job_id, context.session_id)
        target = HpcRuntime._persistent_competition_root(persistent_root(competition))
        client = connect_ssh(config, timeout=30)
        same_connection = verify_job_container_identity(client, config, expected_job_id=job_id)
        same_connection_fields = (
            "designated_proxy_path_verified", "pinned_gateway_host_key_verified",
            "allocation_role_authenticated", "expected_host_uuid_match",
            "expected_gpu_uuid_match", "expected_gpu_model_and_memory_match",
            "allowed_remote_root_match", "job_container_verified",
        )
        if (
            not all(same_connection.get(name) is True for name in same_connection_fields)
            or same_connection.get("read_only") is not True
            or int(same_connection.get("signals_sent") or 0) != 0
            or same_connection.get("other_processes_modified") is not False
        ):
            raise ValueError("same-connection identity evidence is incomplete")
        source = _competition_data_progress_remote_source(competition)
        encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
        remote_code = f"import base64;exec(base64.b64decode({encoded!r}))"
        command = "python3 -c " + shlex.quote(remote_code) + " " + shlex.quote(target)
        aggregation_commands = 1
        _stdin, stdout, stderr = client.exec_command(command, timeout=54)
        raw_output = stdout.read(131_073)
        raw_error = stderr.read(4_097)
        exit_code = stdout.channel.recv_exit_status()
        if isinstance(raw_output, str):
            raw_output = raw_output.encode("utf-8", "replace")
        if isinstance(raw_error, str):
            raw_error = raw_error.encode("utf-8", "replace")
        if len(raw_output) > 131_072:
            raise ValueError("competition progress output exceeds the bounded size")
        if exit_code != 0 or not raw_output.strip():
            return ToolResult(
                "", False,
                {"read_only": True, "remote_commands_attempted": aggregation_commands,
                 "remote_writes": 0, "signals_sent": 0, "other_processes_modified": False,
                 "stderr_present": bool(raw_error.strip()),
                 "stderr_sha256": hashlib.sha256(raw_error).hexdigest() if raw_error else ""},
                "competition progress probe failed",
                error="remote_progress_probe_failed",
            )
        lines = raw_output.decode("utf-8", "strict").strip().splitlines()
        progress = _parse_competition_progress_payload(json.loads(lines[-1]))
        payload = {
            "schema": "evomind.competition_data_progress.v1",
            "competition": competition,
            **{key: value for key, value in progress.items() if key != "schema"},
            "hpc_identity_gate": "passed_5_of_5",
            "samples_requested": 5,
            "samples_passed": 5,
            "same_connection_identity_verified": True,
            "remote_commands_attempted": aggregation_commands,
            "local_fallback_used": False,
            "secret_values_logged": False,
        }
        return ToolResult("", True, payload, f"{competition} progress sampled read-only")
    except Exception as exc:
        return ToolResult(
            "", False,
            {"read_only": True, "remote_commands_attempted": aggregation_commands,
             "remote_writes": 0, "signals_sent": 0, "other_processes_modified": False,
             "hpc_identity_gate": "passed_5_of_5"},
            "competition progress probe failed closed",
            error=f"progress_probe:{type(exc).__name__}",
        )
    finally:
        if client is not None:
            client.close()


def _validate_hpc_solution_inputs(
    args: dict[str, Any],
    context: ToolContext,
) -> tuple[Path, Path | None, str]:
    """Resolve the ordinary-user execution inputs without touching HPC state."""

    workspace = context.workspace_root.resolve(strict=True)

    def resolve_workspace_input(key: str, label: str) -> tuple[Path, Path]:
        raw = str(args.get(key) or "").strip()
        if not raw:
            raise ValueError(f"{label} is required")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = workspace / candidate
        if candidate.is_symlink():
            raise ValueError(f"{label} must not be a symbolic link")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(f"{label} escapes the configured workspace") from exc
        return candidate, resolved

    script_candidate, script = resolve_workspace_input("script_path", "script_path")
    if script_candidate.suffix.casefold() != ".py" or not script_candidate.is_file():
        raise ValueError("script_path must be a regular non-symlink .py file")

    competition_value = str(args.get("competition") or "").strip()
    data_value = str(args.get("data_dir") or "").strip()
    if competition_value:
        if data_value:
            raise ValueError("competition cannot be combined with data_dir")
        from .competition_data import normalize_competition

        return script, None, normalize_competition(competition_value)

    data_candidate, data_dir = resolve_workspace_input("data_dir", "data_dir")
    if not data_candidate.is_dir():
        raise ValueError("data_dir must be a regular non-symlink directory")
    try:
        has_regular_input = any(path.is_file() and not path.is_symlink() for path in data_candidate.rglob("*"))
    except OSError as exc:
        raise ValueError("data_dir could not be inspected safely") from exc
    if not has_regular_input:
        raise ValueError("data_dir must contain at least one regular non-symlink input file")
    return script, data_dir, ""


def _managed_model_prepare(args: dict[str, Any], context: ToolContext) -> ToolResult:
    """Only the signed-off official adapter may provision the persistent model."""
    from . import managed_model_download

    if set(args) - {"model", "revision", "official_cdn_descriptor_sha256"} or args.get("model") != managed_model_download.REPO or args.get("revision") != managed_model_download.REVISION:
        return ToolResult("", False, {}, "model download must match the fixed official snapshot", error="managed_model_source_rejected")
    if not context.approval_verified:
        return ToolResult("", False, {}, "model provisioning requires exact approval", error="exact_approval_required")
    descriptor_sha256 = str(args.get("official_cdn_descriptor_sha256") or "")
    secret_files = None
    if descriptor_sha256:
        from . import model_descriptor_store
        try:
            payload = model_descriptor_store.load_descriptor(context.workspace_root, descriptor_sha256)
        except (OSError, ValueError, managed_model_download.DownloadRejected):
            return ToolResult("", False, {}, "The approved private model descriptor is invalid or expired", error="model_descriptor_rejected")
        secret_files = {managed_model_download.CDN_SECRET_ENV: payload}
    work = (context.workspace_root / "work").resolve(strict=False)
    work.relative_to(context.workspace_root.resolve(strict=True))
    work.mkdir(parents=True, exist_ok=True)
    script = work / "managed_model_download.py"
    if script.is_symlink():
        return ToolResult("", False, {}, "adapter path cannot be a symlink", error="managed_adapter_path")
    shutil.copyfile(Path(managed_model_download.__file__), script)
    context.managed_asset_adapter = True
    context.managed_work_kind = "downloading"
    context.trusted_script_path = Path(managed_model_download.__file__)
    result = None
    try:
        extra = {"secret_files": secret_files} if secret_files else {}
        result = _hpc_execute_solution({"solution_id": "managed-gptoss-download", "script_path": str(script), "competition": "cure_bench", "timeout_seconds": 1200}, context, **extra)
        if descriptor_sha256:
            result.content["official_cdn_descriptor_sha256"] = descriptor_sha256
        return result
    finally:
        context.managed_asset_adapter = False
        context.trusted_script_path = None
        context.managed_work_kind = "executing"
        if descriptor_sha256:
            try:
                model_descriptor_store.remove_descriptor(context.workspace_root, descriptor_sha256)
            except (OSError, ValueError):
                if result is None:
                    raise ValueError("local_model_descriptor_cleanup_failed") from None
                result.ok = False
                result.error = "local_model_descriptor_cleanup_failed"
                result.summary = "Private descriptor cleanup requires attention"
                result.content["local_descriptor_cleanup_confirmed"] = False
            else:
                if result is not None:
                    result.content["local_descriptor_cleanup_confirmed"] = True


def _hpc_execution_budget_status(args: dict[str, Any], context: ToolContext) -> ToolResult:
    from .training_control import controls, load_policy
    from .research_budget import read_execution_budget
    try:
        policy = load_policy(context.runtime_root)
        if not controls(policy, context.metadata):
            return ToolResult('', True, {'configured': False, 'scope': 'general_hpc_execution', 'execution_authorized': False}, 'No general study budget is configured for this caller; this is not execution authorization')
        summary = read_execution_budget(context.runtime_root / 'gpu_budget.sqlite3', policy['study_id'])
        summary['configured'] = True
        return ToolResult('', True, summary, 'General GPU execution budget only. SIIM has a separate study ledger. Rebinding does not reset either budget. Preserve all historical charges: do not clear the ledger, switch study IDs, or reclassify engineering work to bypass a cap. An explicit administrator-authorized budget amendment is required before additional blocked execution.')
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        return ToolResult('', False, {}, 'Unable to read the authorized GPU execution budget', error=type(error).__name__)


def _hpc_execute_solution(args: dict[str, Any], context: ToolContext, *, secret_files: dict[str, bytes] | None = None) -> ToolResult:
    if "siim_calibration" in context.metadata:
        if secret_files:
            return ToolResult("", False, {}, "SIIM model workers cannot receive secrets", error="calibration_secrets_forbidden")
        from .siim_calibration_control import execute as siim_execute
        return siim_execute(args, context, _hpc_execute_solution_impl)
    if "official_calibration" in context.metadata:
        if secret_files:
            return ToolResult("", False, {}, "Calibration model runs cannot receive secrets", error="calibration_secrets_forbidden")
        from .ev_calibration_control import execute
        return execute(args, context, _hpc_execute_solution_impl)
    from .training_control import controls, load_policy
    from .research_budget import GpuExecutionBudget
    extra = {"secret_files": secret_files} if secret_files else {}
    if secret_files and not (getattr(context, "approval_verified", False) and getattr(context, "managed_asset_adapter", False) and getattr(context, "trusted_script_path", None) and Path(context.trusted_script_path).name == "managed_model_download.py"):
        return ToolResult("", False, {}, "Private model inputs require the exact managed approval", error="managed_secret_approval_required")
    try:
        _validate_hpc_solution_inputs(args, context)
    except (OSError, ValueError) as error:
        return ToolResult("", False, {}, str(error), error="invalid_hpc_solution_inputs")
    try:
        policy = load_policy(context.runtime_root)
        controlled = controls(policy, context.metadata)
    except (OSError, ValueError) as error:
        return ToolResult("", False, {}, "HPC input or control-policy gate rejected the request", error=type(error).__name__)
    if not controlled:
        return _hpc_execute_solution_impl(args, context, **extra)
    ledger = GpuExecutionBudget(str(context.runtime_root / "gpu_budget.sqlite3"), policy["study_id"])
    seconds = None
    try:
        seconds = min(1200, max(60, int(args.get("timeout_seconds", 1200))))
        reservation = ledger.reserve(context.session_id, seconds)
    except (ValueError, RuntimeError) as error:
        return ToolResult("", False, {"gpu_budget": ledger.summary(), "requested_seconds": seconds,
            "remote_dispatched": False, "retryable": False, "next_action": "administrator_budget_review"},
            "The general GPU study budget rejected this request. Rebinding cannot reset it; SIIM calibration status uses a separate study and is not the cause. Preserve historical charges and request administrator budget review.", error=str(error))
    started = time.monotonic()
    context.execution_deadline = started + seconds
    context.remote_settlement_uncertain = False
    result = None
    try:
        if context.progress:
            context.progress(budget=reservation)
        result = _hpc_execute_solution_impl({**args, "timeout_seconds": seconds}, context, **extra)
        return result
    finally:
        ledger.settle(reservation["id"], time.monotonic()-started, uncertain=context.remote_settlement_uncertain, success=bool(result and result.ok))
        if result is not None:
            result.content["gpu_budget"] = ledger.summary()
        if context.progress:
            context.progress(budget=ledger.summary())
        context.execution_deadline = None


def _hpc_execute_solution_impl(args: dict[str, Any], context: ToolContext, *, secret_files: dict[str, bytes] | None = None) -> ToolResult:
    try:
        script, data_dir, competition = _validate_hpc_solution_inputs(args, context)
    except (OSError, ValueError) as exc:
        return ToolResult("", False, {}, str(exc), error="invalid_hpc_solution_inputs")
    competition_binding: dict[str, Any] | None = None
    if competition:
        try:
            competition_binding = _managed_competition_training_source(competition, args, context)
        except (OSError, ValueError) as exc:
            return ToolResult("", False, {"competition": competition}, str(exc), error="competition_data_not_ready")
    else:
        verified = _hpc_verify(args, context)
        if not verified.ok:
            return verified
    try:
        job_id, profile = _required_managed_hpc_identity(args, context)
    except ValueError:
        return ToolResult("", False, {}, "managed job/profile binding is invalid", error="profile_job_binding_invalid")
    try:
        from .hpc_runtime_overlay import HpcRuntime
        from research_agent_workstation.server.core.gpu_credentials import connect_ssh
        config = _load_bound_hpc_config(profile, job_id, context.session_id)
        local_dir = context.workspace_root / "work" / "hpc"
        runtime_class=HpcRuntime
        if "siim_calibration" in context.metadata:
            from .siim_hpc_runtime import SiimHpcRuntime
            runtime_class=SiimHpcRuntime
        runtime = runtime_class(
            run_id=context.session_id,
            local_run_dir=local_dir,
            timeout_seconds=int(args.get("timeout_seconds", 3600)),
            connector=lambda: connect_ssh(config, timeout=30),
        )
        runtime.progress_callback = context.progress
        runtime.trusted_progress = context.managed_asset_adapter
        runtime.deadline = context.execution_deadline
        if "siim_calibration" in context.metadata:
            from .siim_calibration_control import policy_for as siim_policy
            runtime.verified_python=siim_policy(context)[0]['python_executable']
        if context.managed_asset_adapter:
            runtime.managed_work_kind = context.managed_work_kind
            trusted = context.trusted_script_path.resolve() if context.trusted_script_path else None
            allowed = {(Path(__file__).parent / name).resolve() for name in ("managed_model_download.py", "managed_tensor_train.py")}
            if trusted not in allowed:
                raise ValueError("managed_adapter_source_rejected")
            script = trusted
        execute_args: dict[str, Any] = {
            "solution_id": str(args.get("solution_id") or "solution"),
            "script_path": script,
        }
        if secret_files:
            execute_args["secret_files"] = secret_files
        if competition_binding:
            execute_args.update({
                "persistent_data_root": competition_binding["persistent_root"],
                "expected_manifest_sha256": competition_binding["manifest_sha256"],
            })
        else:
            execute_args["data_dir"] = data_dir
        from .execution_progress import queued_resource
        resource = "gpu:" + str(getattr(config, "expected_gpu_uuid", "") or f"{job_id}:{profile}")
        waiting_reported = False
        def waiting() -> None:
            nonlocal waiting_reported
            if context.progress and not waiting_reported:
                context.progress(source="managed_adapter", work_kind="preparing", phase="waiting_for_compute", worker_state="queued", detail="Waiting for the shared compute slot")
                waiting_reported = True
        with queued_resource(context.runtime_root / "resource_leases", resource, context.session_id + ":" + new_id("lease"), notify=waiting):
            if context.progress:
                context.progress(source="executor", work_kind="executing", phase="tool_execution", worker_state="unverified")
            result = runtime.execute_solution(**execute_args)
            if "remote_settlement_unconfirmed" in str(result.error):
                context.remote_settlement_uncertain = True
                raise RuntimeError("remote_settlement_unconfirmed")
        payload = asdict(result)
        if secret_files:
            payload["private_inputs"] = {"count": len(secret_files), "cleanup_confirmed": getattr(runtime, "last_solution_secret_cleanup_ok", False), "values_logged": False}
        if competition_binding:
            payload["competition_data_binding"] = competition_binding
        published = []
        published_output_count = 0
        output_root = context.workspace_root / "outputs" / "hpc"
        solution_root = (output_root / result.solution_id).resolve(strict=False)
        solution_root.relative_to(context.workspace_root.resolve(strict=True))
        solution_root.mkdir(parents=True, exist_ok=True)
        for item in result.local_artifacts:
            source = Path(str(item.get("path") or ""))
            if not source.is_file() or source.is_symlink():
                continue
            relative_text = str(item.get("relative_path") or source.name).replace("\\", "/")
            relative = Path(relative_text)
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                continue
            target = (solution_root / relative).resolve(strict=False)
            try:
                target.relative_to(solution_root)
            except ValueError:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            artifact = _artifact_publish({"path": str(target), "source_tool_call": "hpc_execute_solution"}, context).content.get("artifact")
            published.append(artifact)
            if artifact and relative.as_posix() != "training.log":
                published_output_count += 1
        payload["published_artifacts"] = [item for item in published if item]
        payload["published_output_artifact_count"] = published_output_count
        if result.exit_code == 0 and published_output_count == 0:
            payload["runtime_status"] = result.status
            payload["status"] = "failed"
            payload["failure_type"] = "artifact"
            payload["error"] = "solution produced no publishable output artifacts"
            return ToolResult(
                "",
                False,
                payload,
                "HPC solution failed output artifact evidence",
                error="hpc_output_artifact_missing",
            )
        ok = result.status == "completed" and result.exit_code == 0 and published_output_count > 0
        return ToolResult("", ok, payload, f"HPC solution {result.status}", error="" if ok else result.failure_type or "hpc_execution_failed")
    except Exception as exc:
        return ToolResult("", False, {}, "HPC execution failed", error=type(exc).__name__)


def _model_protocols(args: dict[str, Any], context: ToolContext) -> ToolResult:
    from .training_control import list_protocols
    try:
        protocols = list_protocols(context.runtime_root, context.metadata)
        return ToolResult("", True, {"protocols": protocols}, "Authorized model evaluation protocols")
    except (OSError, ValueError) as error:
        return ToolResult("", False, {}, "Model protocols are not available", error=str(error) if re.fullmatch(r"[A-Za-z0-9_]{1,120}", str(error)) else type(error).__name__)


def _managed_tensor_train(args: dict[str, Any], context: ToolContext) -> ToolResult:
    from . import managed_tensor_train, tensor_inference
    from .training_control import digest, load_protocol, read_json, training_config
    if not context.approval_verified:
        return ToolResult("", False, {}, "Registered GPU training requires exact approval", error="exact_approval_required")
    try:
        protocol = load_protocol(context.runtime_root, context.metadata, str(args.get("protocol_id", "")))
        if protocol["policy"].get("enabled") is not True:
            raise ValueError("managed_training_disabled")
        candidate = str(args.get("solution_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", candidate):
            raise ValueError("managed_candidate_id_invalid")
        config_path = Path(str(args.get("configuration_path", "")))
        if not config_path.is_absolute():
            config_path = context.workspace_root / config_path
        config_path.resolve(strict=True).relative_to(context.workspace_root.resolve(strict=True))
        if not re.fullmatch(r"[a-f0-9]{64}", str(args.get("configuration_sha256", ""))) or digest(config_path) != args["configuration_sha256"]:
            raise ValueError("approved_training_configuration_changed")
        config = training_config(read_json(config_path), protocol)
        import numpy as np
        work = context.workspace_root / "work" / ("managed-fit-" + new_id("input"))
        work.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(protocol["training_path"], work / "training.npz")
        np.save(work / "evaluation-features.npy", protocol["evaluation"]["x"].astype(np.float32), allow_pickle=False)
        config.update(training_sha256=digest(work / "training.npz"), evaluation_features_sha256=digest(work / "evaluation-features.npy"))
        (work / "training-config.json").write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        expected_config = digest(work / "training-config.json")
        trace_script = work / "managed_tensor_train.py"
        shutil.copyfile(Path(managed_tensor_train.__file__), trace_script)
    except (OSError, ValueError, TypeError) as error:
        return ToolResult("", False, {}, "Registered training inputs were rejected", error=str(error) if re.fullmatch(r"[A-Za-z0-9_]{1,120}", str(error)) else type(error).__name__)
    context.managed_asset_adapter = True
    context.managed_work_kind = "training"
    context.trusted_script_path = Path(managed_tensor_train.__file__)
    try:
        result = _hpc_execute_solution({"solution_id": candidate, "script_path": str(trace_script), "data_dir": str(work), "timeout_seconds": 1200}, context)
    finally:
        context.managed_asset_adapter = False
        context.managed_work_kind = "executing"
        context.trusted_script_path = None
    if not result.ok:
        return result
    output = context.workspace_root / "outputs/hpc" / candidate
    try:
        required = ("model.safetensors", "model-config.json", "predictions.npz", "training-receipt.json", "environment.lock.json", "training-log.jsonl")
        if any(not (output / name).is_file() or (output / name).is_symlink() for name in required):
            raise ValueError("managed_training_deliverables_missing")
        receipt = read_json(output / "training-receipt.json")
        if (receipt.get("schema") != "evomind.managed_tensor_training.v1" or receipt.get("protocol_id") != protocol["id"]
                or receipt.get("config_sha256") != expected_config or digest(output / "model-config.json") != expected_config
                or receipt.get("checkpoint_sha256") != digest(output / "model.safetensors")
                or receipt.get("predictions_sha256") != digest(output / "predictions.npz")
                or receipt.get("device") != "cuda:0" or type(receipt.get("fit_steps")) is not int or receipt["fit_steps"] < 1
                or receipt.get("evaluation_labels_used") is not False):
            raise ValueError("managed_training_receipt_mismatch")
        shutil.copyfile(Path(tensor_inference.__file__), output / "predict.py")
        public_protocol = {"protocol_id": protocol["id"], **{key: protocol["contract"][key] for key in ("scope", "task", "metric", "source")}, "official_score": False}
        (output / "data-protocol.json").write_text(json.dumps(public_protocol, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        for name in ("predict.py", "data-protocol.json"):
            published = _artifact_publish({"path": str(output / name)}, context).content.get("artifact")
            if published:
                result.content.setdefault("published_artifacts", []).append(published)
        files = {name: digest(output / name) for name in (*required, "predict.py", "data-protocol.json")}
        result.content["managed_training"] = {"schema": "evomind.managed_training_evidence.v1", "protocol_id": protocol["id"], "candidate": candidate, "relative_root": output.relative_to(context.workspace_root).as_posix(), "files": files, "adapter_sha256": digest(Path(managed_tensor_train.__file__)), "fit_steps": receipt["fit_steps"], "scope": protocol["contract"]["scope"], "model_acceptance": "pending_independent_reload"}
        return result
    except (OSError, ValueError, TypeError) as error:
        return ToolResult("", False, result.content, "Training executed but independent model delivery remains unverified", error=str(error) if re.fullmatch(r"[A-Za-z0-9_]{1,120}", str(error)) else type(error).__name__)


def _competition_data_hpc_identity(
    evidence: Any,
    *,
    job_id: int,
    credential_profile: str,
    context: ToolContext | Any,
) -> dict[str, Any]:
    """Project a complete, non-secret HPC identity receipt for data tools."""

    source = evidence if isinstance(evidence, dict) else {}
    metadata = getattr(context, "metadata", None)
    managed = metadata.get("managed_hpc_identity") if isinstance(metadata, dict) else {}
    if not isinstance(managed, dict):
        managed = {}
    generation_text = str(managed.get("allocation_generation") or "").strip()
    generation = int(generation_text) if generation_text.isdigit() and int(generation_text) > 0 else 0
    gpu_uuids = source.get("gpu_uuids") if isinstance(source.get("gpu_uuids"), list) else []
    return {
        "schema": "evomind.competition_data_hpc_identity.v1",
        "status": str(source.get("status") or ""),
        "job_id": int(job_id),
        "credential_profile": credential_profile,
        "allocation_generation": generation,
        "profile_alias": f"G{generation}" if generation else "",
        "designated_proxy_path_verified": source.get("designated_proxy_path_verified") is True,
        "pinned_gateway_host_key_verified": source.get("pinned_gateway_host_key_verified") is True,
        "allocation_role_authenticated": source.get("allocation_role_authenticated") is True,
        "expected_host_uuid_match": source.get("expected_host_uuid_match") is True,
        "expected_gpu_uuid_match": source.get("expected_gpu_uuid_match") is True,
        "expected_gpu_model_and_memory_match": source.get("expected_gpu_model_and_memory_match") is True,
        "allowed_remote_root_match": source.get("allowed_remote_root_match") is True,
        "job_container_verified": source.get("job_container_verified") is True,
        "identity_consistent": source.get("identity_consistent") is True,
        "samples_requested": int(source.get("samples_requested") or 0),
        "samples_passed": int(source.get("samples_passed") or 0),
        "host_uuid_present": bool(str(source.get("host_uuid") or "").strip()),
        "gpu_uuid_count": len([value for value in gpu_uuids if str(value).strip()]),
        "gpu_name": str(source.get("gpu_name") or ""),
        "gpu_memory_total_mib": int(source.get("gpu_memory_total_mib") or 0),
        "remote_root": str(source.get("remote_root") or ""),
        "read_only": source.get("read_only") is True,
        "signals_sent": int(source.get("signals_sent") or 0),
        "other_processes_modified": source.get("other_processes_modified") is True,
        "secrets_returned": False,
    }


def _competition_data_operation(args: dict[str, Any], context: ToolContext, *, action: str) -> ToolResult:
    verified = _hpc_verify({}, context)
    if not verified.ok:
        return verified
    try:
        from .competition_data import (
            normalize_competition,
            persistent_root,
            required_secret_purpose,
            validate_receipt,
            write_adapter_script,
        )
        from .hpc_runtime_overlay import HpcRuntime
        from research_agent_workstation.server.core.gpu_credentials import connect_ssh

        competition = normalize_competition(str(args.get("competition") or ""))
        job_id, profile = _required_managed_hpc_identity({}, context)
        hpc_identity = _competition_data_hpc_identity(
            verified.content,
            job_id=job_id,
            credential_profile=profile,
            context=context,
        )
        config = _load_bound_hpc_config(profile, job_id, context.session_id)
        secret_files: dict[str, bytes] = {}
        credential_source = "not_required"
        purpose = required_secret_purpose(competition)
        if purpose and action in {"prepare", "accelerate"}:
            from .run_secrets import RunSecretStore

            try:
                secret_payload = RunSecretStore(context.runtime_root).consume_for_run(
                    context.session_id,
                    purpose=purpose,
                    managed_hpc_identity=dict(context.metadata.get("managed_hpc_identity") or {}),
                )
                credential_source = "tenant_run_secret_ref"
            except FileNotFoundError:
                if purpose != "kaggle_api":
                    raise
                secret_payload = _managed_kaggle_credential_payload()
                credential_source = "managed_kaggle_connector"
            secret_env_name = "EVOMIND_SECRET_WEATHER4CAST_FILE" if purpose == "weather4cast_sftp" else "EVOMIND_SECRET_KAGGLE_FILE"
            secret_files[secret_env_name] = (
                json.dumps(secret_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            secret_payload = None
        script = write_adapter_script(context.workspace_root, competition, action)
        runtime = HpcRuntime(
            run_id=context.session_id,
            local_run_dir=context.workspace_root / "work" / "hpc",
            timeout_seconds=int(args.get("timeout_seconds", 1800)),
            connector=lambda: connect_ssh(config, timeout=30),
        )
        execution = runtime.execute_managed_data_task(
            task_id=f"{competition}-{action}",
            receipt_competition=competition,
            script_path=script,
            persistent_data_root=persistent_root(competition),
            secret_files=secret_files,
            timeout_seconds=int(args.get("timeout_seconds", 1800)),
        )
        receipt = validate_receipt(competition, execution.get("receipt"))
        output = context.workspace_root / "outputs" / "competition-data" / competition / f"{action}-receipt.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        published = _artifact_publish(
            {"path": str(output), "source_tool_call": f"competition_data_{action}"},
            context,
        ).content.get("artifact")
        status = str(receipt["status"])
        accepted = (
            {"FULL_DATA_READY", "RUNNING", "RUNNING_OR_PARTIAL", "PARTIAL", "NOT_STARTED"}
            if action == "status"
            else {"FULL_DATA_READY", "RUNNING"}
        )
        ok = status in accepted
        payload = {
            "schema": "evomind.competition_data_tool_result.v1",
            "action": action,
            "competition": competition,
            "status": status,
            "receipt": receipt,
            "execution": {
                key: value
                for key, value in execution.items()
                if key not in {"stdout_tail", "stderr_tail", "local_log"}
            },
            "artifact": published,
            "hpc_identity_gate": "passed_5_of_5",
            "hpc_identity": hpc_identity,
            "local_fallback_used": False,
            "credential_source": credential_source,
            "secret_values_logged": False,
        }
        error = "" if ok else (
            "controlled_secret_required" if status == "SECRET_REQUIRED"
            else "human_gate_or_auth" if status == "HUMAN_GATE_OR_AUTH"
            else status.casefold()
        )
        return ToolResult(
            "",
            ok,
            payload,
            f"{competition} data {status}",
            error=error,
            artifacts=[published] if published else [],
        )
    except FileNotFoundError:
        return ToolResult("", False, {}, "controlled competition credential is required", error="controlled_secret_required")
    except Exception as exc:
        safe_error = str(exc)
        if not re.fullmatch(r"(?:consumer_process_exit|consumer_host_exit):[A-Za-z][A-Za-z0-9_.-]{0,64}|consumer_invalid_base64|controlled_secret_required", safe_error):
            safe_error = type(exc).__name__
        return ToolResult("", False, {}, "competition data operation failed", error=safe_error)


def _competition_data_prepare(args: dict[str, Any], context: ToolContext) -> ToolResult:
    return _competition_data_operation(args, context, action="prepare")


def _competition_data_accelerate(args: dict[str, Any], context: ToolContext) -> ToolResult:
    return _competition_data_operation(args, context, action="accelerate")


def _competition_data_status(args: dict[str, Any], context: ToolContext) -> ToolResult:
    return _competition_data_operation(args, context, action="status")


def _hpc_cancel(args: dict[str, Any], context: ToolContext) -> ToolResult:
    try:
        job_id, profile = _required_managed_hpc_identity(args, context)
    except ValueError:
        return ToolResult("", False, {}, "managed job/profile binding is invalid", error="profile_job_binding_invalid")
    try:
        from .managed_cancel import require_fresh_identity
        require_fresh_identity(context.runtime_root, context.session_id, job_id)
        from .hpc_runtime_overlay import HpcRuntime
        from research_agent_workstation.server.core.gpu_credentials import connect_ssh
        config = _load_bound_hpc_config(profile, job_id, context.session_id)
        runtime = HpcRuntime(
            run_id=context.session_id,
            local_run_dir=context.workspace_root / "work" / "hpc",
            connector=lambda: connect_ssh(config, timeout=30),
        )
        cancelled = runtime.cancel(solution_id=str(args.get("solution_id") or "solution"))
        return ToolResult("", cancelled, {"job_id": job_id, "solution_id": str(args.get("solution_id") or "solution"), "cancelled": cancelled, "receipt": runtime.last_cancel_receipt}, "Managed process exit independently observed" if cancelled else "HPC cancellation unconfirmed", error="" if cancelled else "hpc_cancel_failed")
    except Exception as exc:
        return ToolResult("", False, {}, "HPC cancel failed", error=type(exc).__name__)


def _mcp_list_tools(args: dict[str, Any], context: ToolContext) -> ToolResult:
    from .mcp import MCPClient
    with MCPClient(_trusted_argv(args, context), timeout=float(args.get("timeout_seconds", 30))) as client:
        result = client.request("tools/list", {})
    return ToolResult("", True, result, "listed MCP tools")


def _mcp_call(args: dict[str, Any], context: ToolContext) -> ToolResult:
    from .mcp import MCPClient
    with MCPClient(_trusted_argv(args, context), timeout=float(args.get("timeout_seconds", 60))) as client:
        result = client.request("tools/call", {"name": str(args["name"]), "arguments": dict(args.get("arguments") or {})})
    return ToolResult("", True, result, f"called MCP tool {args['name']}")


OBJECT = {"type": "object", "additionalProperties": False}


def build_default_registry() -> ToolRegistry:
    # Compatibility bridge for the legacy name-based policy engine.  Keep the
    # focused R84 change in this module while ensuring the declared read-only
    # ToolSpec and the runtime policy agree before any invocation is evaluated.
    from .policy import ALWAYS_APPROVAL_TOOLS, READ_ONLY_TOOLS, WORKSPACE_WRITE_TOOLS

    READ_ONLY_TOOLS.update({"competition_data_progress", "hpc_asset_probe"})
    WORKSPACE_WRITE_TOOLS.add("competition_data_accelerate")
    ALWAYS_APPROVAL_TOOLS.update({"competition_data_accelerate", "hpc_calibration_run"})
    READ_ONLY_TOOLS.update({"hpc_calibration_status", "hpc_execution_budget_status"})
    registry = ToolRegistry()
    from . import super_agent_tools as v2_tools

    def add(name: str, description: str, properties: dict[str, Any], required: list[str], capability: str, handler: Handler, *, read_only: bool = True, cancel: bool = False, available: bool = True, reason: str = "") -> None:
        registry.register(ToolSpec(name, description, {**OBJECT, "properties": properties, "required": required}, capability, read_only=read_only, supports_cancel=cancel, available=available, unavailable_reason=reason), handler)
    directory_ref = {
        "directory_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"},
        "relative_path": {"type": "string", "maxLength": 4096},
    }
    add(
        "super_agent_status",
        "Return the additive Super Agent V1 kernel, catalog, mount, migration, and durable-run status.",
        {}, [], "agent.read", v2_tools.super_agent_status,
    )
    add(
        "capability_discover",
        "Discover healthy capability descriptors from metadata and operation requirements; this does not use a fixed domain keyword table.",
        {
            "objective": {"type": "string", "maxLength": 8000},
            "required_operations": {"type": "array", "items": {"type": "string", "maxLength": 128}, "maxItems": 32, "uniqueItems": True},
            "healthy_only": {"type": "boolean"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        }, [], "capability.read", v2_tools.capability_discover,
    )
    add(
        "ecosystem_refresh",
        "Boundedly rescan installed Codex/Agent SKILL front matter and refresh declarative capability descriptors; instruction bodies are not executed.",
        {}, [], "capability.read", v2_tools.ecosystem_refresh,
    )
    add(
        "connector_health",
        "Read the current health receipt for one authorized directory capability.",
        {"directory_id": directory_ref["directory_id"]}, ["directory_id"], "connector.read", v2_tools.connector_health,
    )
    add(
        "directory_list", "List entries inside an authorized directory capability.",
        {**directory_ref, "recursive": {"type": "boolean"}, "limit": {"type": "integer", "minimum": 1, "maximum": 100000}},
        ["directory_id"], "directory.read", v2_tools.directory_list,
    )
    add(
        "directory_stat", "Stat one capability-relative path without exposing the connector root.",
        directory_ref, ["directory_id"], "directory.read", v2_tools.directory_stat,
    )
    add(
        "directory_read", "Read a bounded capability-relative file range with a structured operation receipt.",
        {**directory_ref, "max_bytes": {"type": "integer", "minimum": 1, "maximum": 2097152}, "encoding": {"type": "string", "maxLength": 64}},
        ["directory_id", "relative_path"], "directory.read", v2_tools.directory_read,
    )
    add(
        "directory_hash", "Hash one capability-relative file without revealing its connector root.",
        directory_ref, ["directory_id", "relative_path"], "directory.read", v2_tools.directory_hash,
    )
    add(
        "directory_mkdir", "Create a directory inside an authorized writable capability.",
        {**directory_ref, "parents": {"type": "boolean"}, "exist_ok": {"type": "boolean"}},
        ["directory_id", "relative_path"], "directory.write", v2_tools.directory_mkdir, read_only=False,
    )
    add(
        "directory_write_atomic", "Atomically write bounded text or base64 bytes inside an authorized capability.",
        {
            **directory_ref,
            "content": {"type": "string", "maxLength": 2097152},
            "base64_content": {"type": "string", "maxLength": 2796204},
            "encoding": {"type": "string", "maxLength": 64},
            "create_parents": {"type": "boolean"},
            "expected_sha256": {"type": "string", "pattern": "^$|^[a-fA-F0-9]{64}$"},
        },
        ["directory_id", "relative_path"], "directory.write", v2_tools.directory_write_atomic, read_only=False,
    )
    transfer_ref = {
        "source_directory_id": directory_ref["directory_id"],
        "source_relative_path": directory_ref["relative_path"],
        "destination_directory_id": directory_ref["directory_id"],
        "destination_relative_path": directory_ref["relative_path"],
    }
    add(
        "directory_copy", "Copy one file between authorized directory capabilities with hash verification.",
        transfer_ref,
        ["source_directory_id", "source_relative_path", "destination_directory_id", "destination_relative_path"],
        "directory.write", v2_tools.directory_copy, read_only=False,
    )
    add(
        "directory_sync", "Synchronize an authorized directory tree; deletion requires exact approval.",
        {**transfer_ref, "delete_extraneous": {"type": "boolean"}},
        ["source_directory_id", "destination_directory_id"], "directory.write", v2_tools.directory_sync, read_only=False,
    )
    add(
        "directory_delete", "Delete one exact capability-relative path after exact approval.",
        {**directory_ref, "recursive": {"type": "boolean"}},
        ["directory_id", "relative_path"], "directory.delete", v2_tools.directory_delete, read_only=False,
    )
    package_id = {"package_id": {"type": "string", "minLength": 1, "maxLength": 128}}
    add(
        "tool_synthesize",
        "Stage a run-scoped declarative tool package manifest; this never installs or executes package code.",
        {"manifest": {"type": "object", "maxProperties": 64}}, ["manifest"],
        "tool-package.write", v2_tools.tool_synthesize, read_only=False,
    )
    package_evidence = {
        **package_id,
        "passed": {"type": "boolean"},
        "evidence_artifact_id": {"type": "string", "minLength": 1, "maxLength": 160},
        "evidence_sha256": {"type": "string", "pattern": "^[a-fA-F0-9]{64}$"},
    }
    add(
        "tool_test", "Record hash-bound published test evidence for a run-scoped tool package.",
        package_evidence, ["package_id", "passed", "evidence_artifact_id", "evidence_sha256"],
        "tool-package.write", v2_tools.tool_test, read_only=False,
    )
    add(
        "tool_canary", "Record hash-bound published canary evidence after declared tests pass.",
        package_evidence, ["package_id", "passed", "evidence_artifact_id", "evidence_sha256"],
        "tool-package.write", v2_tools.tool_canary, read_only=False,
    )
    add(
        "tool_promote", "Promote verified metadata for this Run only; sensitive or irreversible packages return an exact approval Gate.",
        package_id, ["package_id"], "tool-package.write", v2_tools.tool_promote, read_only=False,
    )
    add(
        "tool_reject", "Irreversibly reject a non-promoted run-scoped tool package.",
        package_id, ["package_id"], "tool-package.write", v2_tools.tool_reject, read_only=False,
    )
    add(
        "transfer_fetch",
        "Stream an HTTP/HTTPS source directly into an authorized directory connector with resume, quota, and hash verification.",
        {
            "source_connector_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "source_url": {"type": "string", "format": "uri", "maxLength": 8192},
            "directory_id": directory_ref["directory_id"],
            "relative_path": directory_ref["relative_path"],
            "resume": {"type": "boolean"},
            "expected_sha256": {"type": "string", "pattern": "^$|^[a-fA-F0-9]{64}$"},
        },
        ["source_connector_id", "source_url", "directory_id", "relative_path"],
        "transfer.write", v2_tools.transfer_fetch, read_only=False, cancel=True,
    )
    job_ref = {
        "directory_id": directory_ref["directory_id"],
        "job_ref": {"type": "string", "minLength": 1, "maxLength": 256},
    }
    add(
        "job_execute", "Execute one structured job through an injected execution connector after its live identity gate.",
        {
            "directory_id": directory_ref["directory_id"],
            "relative_path": directory_ref["relative_path"],
            "job_spec": {"type": "object", "maxProperties": 128},
        },
        ["directory_id", "job_spec"], "job.execute", v2_tools.job_execute, read_only=False, cancel=True,
    )
    add(
        "job_status", "Read one injected execution connector job receipt by opaque job reference.",
        job_ref, ["directory_id", "job_ref"], "job.read", v2_tools.job_status,
    )
    add(
        "job_cancel", "Cancel one exact connector job after exact approval.",
        job_ref, ["directory_id", "job_ref"], "job.cancel", v2_tools.job_cancel, read_only=False,
    )
    path = {"path": {"type": "string", "minLength": 1}}
    add("file_list", "List files and directories.", {**path, "glob": {"type": "string"}, "recursive": {"type": "boolean"}, "limit": {"type": "integer"}}, ["path"], "filesystem.read", _file_list)
    add("file_search", "Search file text.", {**path, "query": {"type": "string"}, "glob": {"type": "string"}, "limit": {"type": "integer"}}, ["path", "query"], "filesystem.read", _file_search)
    add("file_read", "Read a bounded file range with hash metadata.", {**path, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}, "encoding": {"type": "string"}}, ["path"], "filesystem.read", _file_read)
    add("file_write", "Atomically write a file with preimage checking and backup.", {**path, "content": {"type": "string"}, "encoding": {"type": "string"}, "expected_sha256": {"type": "string"}}, ["path", "content"], "filesystem.write", _file_write, read_only=False)
    add("file_patch", "Replace an exact text fragment with preimage checking and diff.", {**path, "find": {"type": "string"}, "replace": {"type": "string"}, "expected_count": {"type": "integer"}, "expected_sha256": {"type": "string"}}, ["path", "find", "replace"], "filesystem.write", _file_patch, read_only=False)
    transfer = {"source": {"type": "string"}, "destination": {"type": "string"}, "overwrite": {"type": "boolean"}}
    add("file_copy", "Copy a file or directory.", transfer, ["source", "destination"], "filesystem.write", _file_copy, read_only=False)
    add("file_move", "Move a file or directory.", transfer, ["source", "destination"], "filesystem.write", _file_move, read_only=False)
    add("file_delete", "Delete an exact path after approval.", {**path, "recursive": {"type": "boolean"}}, ["path"], "filesystem.delete", _file_delete, read_only=False)
    command = {"argv": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 8192}, "minItems": 1, "maxItems": 128}, "cwd": {"type": "string"}, "timeout_seconds": {"type": "integer"}}
    add("shell_exec", "Execute an allowlisted argument-array command and capture complete logs.", command, ["argv"], "shell.execute", _shell_exec, read_only=False, cancel=True)
    add("process_start", "Start an allowlisted managed background process from an argument array.", command, ["argv"], "process.manage", _process_start, read_only=False, cancel=True)
    add("process_list", "List runtime-managed processes.", {}, [], "process.read", _process_list)
    pid = {"process_id": {"type": "string"}}
    add("process_poll", "Poll a managed process.", pid, ["process_id"], "process.read", _process_poll)
    add("process_log", "Read a managed process log tail.", {**pid, "lines": {"type": "integer"}}, ["process_id"], "process.read", _process_log)
    add("process_stdin", "Write to managed process stdin.", {**pid, "data": {"type": "string"}}, ["process_id", "data"], "process.manage", _process_stdin, read_only=False)
    add("process_cancel", "Terminate a managed process tree.", pid, ["process_id"], "process.cancel", _process_cancel, read_only=False)
    add("browser_health", "Inspect browser adapter availability.", {}, [], "browser.read", _browser_health)
    add("browser_open", "Open a URL in the installed browser.", {"url": {"type": "string", "format": "uri"}}, ["url"], "browser.control", _browser_open, read_only=False)
    add("browser_dom", "Capture DOM from a headless browser navigation.", {"url": {"type": "string", "format": "uri"}}, ["url"], "browser.read", _browser_dom)
    add("desktop_windows", "List visible desktop windows.", {}, [], "desktop.read", _desktop_windows)
    add("desktop_screenshot", "Capture the Windows virtual desktop.", {}, [], "desktop.read", _desktop_screenshot)
    add("desktop_click", "Click an approved desktop coordinate.", {"x": {"type": "integer"}, "y": {"type": "integer"}}, ["x", "y"], "desktop.control", _desktop_click, read_only=False)
    add("desktop_type", "Type approved text into the focused window.", {"text": {"type": "string"}}, ["text"], "desktop.control", _desktop_type, read_only=False)
    add("skill_list", "Discover installed Codex and Agent skills.", {}, [], "skills.read", _skill_list)
    mcp = {"argv": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 8192}, "minItems": 1, "maxItems": 128}, "timeout_seconds": {"type": "integer"}}
    add("mcp_list_tools", "Initialize an allowlisted stdio MCP server and list its tools.", mcp, ["argv"], "mcp.read", _mcp_list_tools)
    add("mcp_call", "Call an exact MCP tool after policy approval.", {**mcp, "name": {"type": "string"}, "arguments": {"type": "object"}}, ["argv", "name"], "mcp.call", _mcp_call, read_only=False)
    add("research_capabilities", "List integrated Research OS capabilities and gates.", {}, [], "research.read", _research_capabilities)
    add("verified_context", "Load evidence-bound historical run, metrics, artifacts, memory, literature and governance for the selected research task before answering.", {"selected_task": {"type": "string", "pattern": "^[A-Za-z0-9_.-]{1,160}$"}}, [], "research.read", _verified_context)
    add("research_invoke", "Invoke an allowlisted legacy Research OS action.", {"action": {"type": "string"}, "input": {"type": "string"}, "timeout_seconds": {"type": "integer"}}, ["action"], "research.execute", _research_invoke, read_only=False)
    add("literature_search", "Search arXiv, OpenAlex and Crossref and persist a source evidence manifest.", {"query": {"type": "string", "minLength": 2, "maxLength": 1200}, "max_results": {"type": "integer", "minimum": 1, "maximum": 30}}, ["query"], "literature.search", _literature_search, read_only=False)
    add("citation_audit", "Resolve DOI citations in one published text artifact and persist an audit manifest.", {"artifact_id": {"type": "string", "minLength": 1}}, ["artifact_id"], "literature.audit", _citation_audit, read_only=False)
    add("memory_search", "Search project and experiment memory without exposing credential stores.", {"query": {"type": "string", "minLength": 2, "maxLength": 1000}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, ["query"], "memory.read", _memory_search)
    memory_text = {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 1000}, "maxItems": 50}
    memory_scalar = {"oneOf": [{"type": "number"}, {"type": "boolean"}, {"type": "string", "maxLength": 1000}, {"type": "null"}]}
    memory_record = {
        "type": "object", "additionalProperties": False, "maxProperties": 12,
        "properties": {
            "title": {"type": "string", "minLength": 1, "maxLength": 200},
            "summary": {"type": "string", "minLength": 1, "maxLength": 4000},
            "applicable_conditions": memory_text,
            "lessons": memory_text,
            "failure_modes": memory_text,
            "strategy_changes": memory_text,
            "metrics": {"type": "object", "maxProperties": 50, "additionalProperties": memory_scalar},
            "candidate_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$"},
            "parent_record_ids": {"type": "array", "items": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$"}, "maxItems": 20},
            "tags": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 80}, "maxItems": 30, "uniqueItems": True},
        },
        "required": ["summary", "lessons"],
    }
    memory_evidence = {
        "type": "array", "minItems": 1, "maxItems": 32,
        "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "kind": {"enum": ["artifact", "evidence"]},
                "id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$"},
                "sha256": {"type": "string", "pattern": "^[a-fA-F0-9]{64}$"},
            },
            "required": ["kind", "id", "sha256"],
        },
    }
    add(
        "memory_writeback",
        "Persist one SHA-bound retrospective or experience record for future runs after verified evidence exists.",
        {
            "task_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$"},
            "record_type": {"enum": ["retrospective", "experience"]},
            "record": memory_record,
            "source_run_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$"},
            "source_evidence": memory_evidence,
            "idempotency_key": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"},
        },
        ["task_id", "record_type", "record", "source_run_id", "source_evidence", "idempotency_key"],
        "memory.write",
        _memory_writeback,
        read_only=False,
    )
    add("report_generate", "Generate from a frozen snapshot of already published artifacts. Before multi-format generation, artifact_publish the intended inputs/results and pass their returned, existing IDs in artifact_ids. Merely writing, reading or hashing a file does not make it a report source. With formats, queue an idempotent job, then query report_status until terminal and inspect source_count/missing. After publishing missing sources, create a new report; resuming an old report cannot change its snapshot. Diagnostic tasks may produce task records without artifacts. Legacy calls publish Markdown. Rendering is not independent research validation.", {"report_id": {"type": "string", "pattern": "^report_[a-f0-9]{32}$"}, "title": {"type": "string"}, "summary": {"type": "string"}, "artifact_ids": {"type": "array", "items": {"type": "string"}}, "name": {"type": "string"}, "report_kind": {"enum": ["diagnostic", "training", "inference", "analysis"]}, "language": {"enum": ["zh-CN", "en-US"]}, "formats": {"type": "array", "minItems": 1, "items": {"enum": ["markdown", "html", "docx", "pdf"]}}}, [], "report.write", _report_generate, read_only=False)
    add("report_status", "Read an owned Run's report generation, evidence and output states. Does not start or resume work.", {"report_id": {"type": "string", "pattern": "^report_[a-f0-9]{32}$"}}, ["report_id"], "report.read", _report_status)
    add("evolution_evaluate", "Evaluate a candidate against regression and improvement gates without auto-promotion.", {"metrics": {"type": "object"}}, ["metrics"], "evolution.evaluate", _evolution_evaluate, read_only=False)
    add("kaggle_list", "Discover live Kaggle competitions. For the fixed six-source data workflow, do not depend on this listing: use competition_data_status/prepare directly; transient listing failures return the pinned managed catalog.", {"query": {"type": "string"}, "page": {"type": "integer", "minimum": 1}}, [], "kaggle.read", _kaggle_list)
    add("kaggle_download", "Download and hash one competition dataset into this run's isolated inputs directory.", {"competition": {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]{1,120}$"}}, ["competition"], "kaggle.download", _kaggle_download, read_only=False)
    add("kaggle_submit", "Submit one verified outputs/ file to Kaggle after exact one-shot approval.", {"competition": {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]{1,120}$"}, "submission_path": {"type": "string", "minLength": 1}, "message": {"type": "string", "maxLength": 200}}, ["competition", "submission_path", "message"], "kaggle.submit", _kaggle_submit, read_only=False)
    add(
        "training_route",
        "Profile an attached file or bind one FULL_DATA_READY managed competition dataset, then deterministically select local CPU or managed HPC GPU. For the six managed competitions pass competition instead of asking the user to upload data again. Always call this before fitting a model.",
        {
            "attachment_id": {"type": "string", "minLength": 1},
            "path": {"type": "string", "minLength": 1},
            "competition": {"enum": ["cure_bench", "e2lmc", "mindgames", "ariel_2025", "weather4cast", "open_polymer", "histopathologic_cancer"]},
            "task_description": {"type": "string", "minLength": 1, "maxLength": 4000},
            "target_column": {"type": "string", "maxLength": 256},
            "timeout_seconds": {"type": "integer", "minimum": 30, "maximum": 86400},
        },
        ["task_description"],
        "training.route",
        _training_route,
        read_only=False,
    )
    # The allocation identity is injected by the authenticated server route and
    # persisted in session metadata.  It is deliberately absent from model-
    # writable tool schemas: exposing optional job/profile fields encouraged
    # the model to invent a different allocation, which the fail-closed handler
    # correctly rejected.  The handlers retain their retarget checks as a
    # second boundary for direct/internal callers.
    add("hpc_verify", "Verify the server-bound designated proxy, allocation role route, host/GPU UUIDs, remote root and five read-only GPU samples. The model cannot select or retarget the job/profile.", {}, [], "hpc.read", _hpc_verify)
    add("hpc_execution_budget_status", "Read this caller's general hpc_execute_solution study budget, engineering cap, remaining seconds and reconciliation state. Local read only; no HPC connection or reservation. Use this to diagnose gpu_execution_budget_exhausted. Rebinding does not replenish the budget. SIIM calibration has a separate ledger.", {}, [], "hpc.read", _hpc_execution_budget_status)
    from .siim_calibration_web import run_tool as siim_web_run, status_tool as siim_web_status
    add("hpc_calibration_status", "Read only the already-registered SIIM MLE-bench comparison and its separate campaign budget. This is not the general hpc_execute_solution budget or current allocation health. Use hpc_execution_budget_status for general budget errors. No training, remote inspection, submission or hidden scoring is performed.", {}, [], "hpc.read", siim_web_status)
    add("hpc_calibration_run", "Start/resume the registered SIIM MLE-bench study through this visible Assistant Run. Exact approval required. Uses existing fixed baseline, independent EvoMind + DeepSeek-V4-Pro and AIDE-adapted controls, original seeds/folds/data, preserved checkpoints and unchanged total 24 GPU-hour ledger. The sealed coordinator dispatches native isolated sessions; this tool waits and reports progress until candidates freeze or a real blocker occurs. Pause/cancel prevents subsequent child calls; a current bounded fit drains before stopping. Never submits to Kaggle or reads hidden labels. Do not run separate training scripts or create replacement studies. After a verified fix and full settlement, an explicitly authorized retry of the same Run requires a fresh operation_id and fresh exact approval; old operations and their receipts are immutable. Constant interface probes cannot complete training. Returns a verifiable receipt, not an official score.", {"campaign": {"enum": ["siim_mlebench_calibration_20260908"]}, "operation_id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{4,40}$"}}, ["campaign"], "hpc.execute", siim_web_run, read_only=False)
    add("model_protocols", "List administrator-frozen model-data protocols for this exact user and project. Exposes schema and source identities, never evaluation labels or permission to submit a competition.", {}, [], "runtime.read", _model_protocols)
    add("managed_tensor_train", "Fit one registered CUDA-only dense PyTorch model using an authorized data protocol. Coder must first write and hash a JSON configuration containing only hidden_sizes, epochs, batch_size, learning_rate and seed. Approval binds configuration_sha256, not just its path. The immutable managed adapter, not generated Python, executes. Returns safetensors checkpoint, predictions, hash-locked environment and independent predict.py. Does not provide official competition scores. Requires exact approval, fresh 5/5 HPC gate, GPU queue and configured study budget.", {"protocol_id": {"type": "string", "pattern": "^[a-f0-9]{64}$"}, "solution_id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,100}$"}, "configuration_path": {"type": "string", "minLength": 1}, "configuration_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"}}, ["protocol_id", "solution_id", "configuration_path", "configuration_sha256"], "hpc.execute", _managed_tensor_train, read_only=False)
    add(
        "hpc_asset_probe",
        "Read metadata for the fixed CURE/MindGames managed package, model and manifest candidates after the complete current 5/5 identity gate. The tool cannot accept a path, command, job or profile; it never reads file contents or labels and never writes, signals, uploads, installs, downloads or starts training.",
        {"scope": {"type": "string", "enum": ["cure_mindgames"]}},
        [],
        "hpc.read",
        _hpc_asset_probe,
    )
    add(
        "hpc_execute_solution",
        "Execute one bounded Python .py solution on the server-bound allocation only after the strict 5/5 job-container identity gate. The script is invoked as Python SCRIPT --data-dir DATA_DIR --out-dir OUT_DIR. Supply exactly one input source: data_dir containing at least one regular non-symlink input file, or competition for a freshly revalidated FULL_DATA_READY persistent managed dataset. Managed competition data stays on HPC and is manifest-SHA-bound without being re-uploaded. Success requires at least one output artifact that can be published. The model cannot select or retarget the job/profile.",
        {
            "solution_id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,120}$"},
            "script_path": {"type": "string", "minLength": 1},
            "data_dir": {"type": "string", "minLength": 1},
            "competition": {"enum": ["cure_bench", "e2lmc", "mindgames", "ariel_2025", "weather4cast", "open_polymer", "histopathologic_cancer"]},
            "timeout_seconds": {"type": "integer", "minimum": 30, "maximum": 86400},
        },
        ["solution_id", "script_path"],
        "hpc.execute",
        _hpc_execute_solution,
        read_only=False,
    )
    competition_name = {
        "competition": {
            "type": "string",
            "enum": ["cure_bench", "e2lmc", "mindgames", "ariel_2025", "weather4cast", "open_polymer", "histopathologic_cancer"],
        },
        "timeout_seconds": {"type": "integer", "minimum": 30, "maximum": 86400},
    }
    add(
        "competition_data_progress",
        "Sample aggregate file growth and the existing worker's process-tree IO through the server-bound managed HPC route. The tool accepts no path, command, job, or profile and never writes, signals, uploads, publishes, restarts, or creates a worker.",
        {"competition": competition_name["competition"]},
        ["competition"],
        "hpc.read",
        _competition_data_progress,
    )
    add(
        "competition_data_accelerate",
        "Replace the single active Weather4cast worker with the high-throughput adapter after exact approval and a fresh controlled credential. The fixed adapter verifies the current worker identity, sends one TERM to that process group, and resumes the existing data directory without creating a concurrent worker.",
        {
            "competition": {"type": "string", "enum": ["weather4cast"]},
            "timeout_seconds": {"type": "integer", "minimum": 30, "maximum": 86400},
        },
        ["competition"],
        "hpc.execute",
        _competition_data_accelerate,
        read_only=False,
    )
    add(
        "competition_data_prepare",
        "Prepare one official competition source in the fixed managed HPC directory. Automatically enforces the current 5/5 job-container identity gate, resumable source adapter, real loader smoke, and a hash-bound receipt. Never accepts credentials in tool arguments.",
        competition_name,
        ["competition"],
        "hpc.execute",
        _competition_data_prepare,
        read_only=False,
    )
    add(
        "competition_data_status",
        "Read and validate one competition's persistent HPC download/readiness receipt. Use this instead of guessing from prior Run outputs or process exit codes.",
        competition_name,
        ["competition"],
        "hpc.read",
        _competition_data_status,
    )
    add("hpc_cancel", "Cancel one exact managed HPC solution on the server-bound allocation after exact approval. The model cannot select or retarget the job/profile.", {"solution_id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,120}$"}}, ["solution_id"], "hpc.cancel", _hpc_cancel, read_only=False)
    add("runtime_health", "Return unified runtime health.", {}, [], "runtime.read", _runtime_health)
    add("attachment_list", "List verified files explicitly attached to this run.", {}, [], "attachment.read", _attachment_list)
    add("attachment_read", "Read a bounded preview of one verified attachment.", {"attachment_id": {"type": "string"}, "max_bytes": {"type": "integer"}, "encoding": {"type": "string"}}, ["attachment_id"], "attachment.read", _attachment_read)
    add("artifact_publish", "Verify and publish one final user-facing file from this task workspace.", {**path, "name": {"type": "string"}, "media_type": {"type": "string"}, "source_tool_call": {"type": "string"}}, ["path"], "artifact.write", _artifact_publish, read_only=False)
    add("artifact_list", "List verified deliverables for this run with preview/download URLs.", {}, [], "artifact.read", _artifact_list)
    add(
        "managed_model_prepare",
        "Prepare the fixed official GPT-OSS-20B snapshot through the persistent, hash-verified managed downloader. Requires exact approval and a fresh HPC identity gate. Does not train, submit, install dependencies, or accept terms.",
        {"model": {"type": "string", "enum": ["openai/gpt-oss-20b"]}, "revision": {"type": "string", "enum": ["6cee5e81ee83917806bbde320786a8fb61efebee"]}, "official_cdn_descriptor_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"}},
        ["model", "revision"], "hpc.model.prepare", _managed_model_prepare, read_only=False,
    )
    add("artifact_preview", "Read a safe bounded preview for a published artifact.", {"artifact_id": {"type": "string"}}, ["artifact_id"], "artifact.read", _artifact_preview)
    add(
        "artifact_import",
        "After exact approval, copy an explicit allowlist of SHA-verified published artifacts from one same-tenant parent Run into the current Run. The source Run and every artifact ID are fingerprint-bound; no wildcard, cross-tenant access, deletion, credential access or source mutation is allowed.",
        {
            "source_run_id": {"type": "string", "pattern": "^run_[0-9a-f]{32}$"},
            "artifact_ids": {"type": "array", "minItems": 1, "maxItems": 20, "uniqueItems": True, "items": {"type": "string", "pattern": "^artifact_[0-9a-f]{32}$"}},
        },
        ["source_run_id", "artifact_ids"],
        "artifact.import",
        _artifact_import,
        read_only=False,
    )
    add("artifact_bundle", "Bundle selected published artifacts into a verified ZIP deliverable.", {"artifact_ids": {"type": "array", "items": {"type": "string"}}, "name": {"type": "string"}}, [], "artifact.write", _artifact_bundle, read_only=False)
    return registry
