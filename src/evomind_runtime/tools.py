from __future__ import annotations

import base64
import difflib
import hashlib
import itertools
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

from .models import ToolResult, ToolSpec

Handler = Callable[[dict[str, Any], "ToolContext"], ToolResult]

ALLOWED_PROCESS_EXECUTABLES = {
    "git", "git.exe", "node", "node.exe", "python", "python.exe", "python3", "python3.exe",
    "pytest", "pytest.exe", "rg", "rg.exe", "uv", "uv.exe",
}
ALLOWED_PYTHON_MODULES = {"evomind_runtime.cli", "pytest", "xsci.kaggle"}


@dataclass
class ToolContext:
    session_id: str
    workspace_root: Path
    runtime_root: Path
    artifact_root: Path
    store: Any


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
            result = self._handlers[name](arguments, context)
        except Exception as exc:  # tool failures become observations
            return ToolResult("", False, {}, f"{name} failed", error=f"{type(exc).__name__}: {exc}")
        encoded = json.dumps(result.content, ensure_ascii=False, default=str).encode("utf-8")
        if len(encoded) > spec.max_result_bytes:
            artifact = context.store.add_artifact(context.session_id, encoded, ".json", "application/json", context.artifact_root)
            result.content = {"truncated": True, "preview": encoded[: spec.max_result_bytes].decode("utf-8", "replace"), "artifact": artifact}
            result.artifacts.append(artifact)
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
    if basename in {"node", "node.exe"} and any(value in {"-e", "--eval", "-p", "--print"} for value in lowered):
        raise ValueError("inline Node.js code is not allowed; execute a workspace script")
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
        command = [rg, "--line-number", "--column", "--no-heading", "--color", "never", "--fixed-strings", "-g", glob, "-e", query, "--", str(root)]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=False)
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
    process: subprocess.Popen[bytes]
    log_path: Path
    stdin_enabled: bool
    created_at: float
    log_stream: Any
    log_thread: threading.Thread
    log_closed: bool = False


def _redact_process_log(value: str) -> str:
    redacted = re.sub(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", "[credential redacted]", value, flags=re.I)
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

    def start(self, argv: list[str], cwd: Path, log_path: Path) -> ManagedProcess:
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
        item = ManagedProcess(f"proc_{process.pid}_{time.time_ns()}", process, log_path, True, time.time(), stream, log_thread)
        with self._lock:
            self._items[item.id] = item
        return item

    def get(self, process_id: str) -> ManagedProcess:
        with self._lock:
            if process_id not in self._items:
                raise KeyError(process_id)
            return self._items[process_id]

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._items.values())
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
    item = PROCESSES.start(_trusted_argv(args, context), cwd, log_path)
    return ToolResult("", True, {"process_id": item.id, "pid": item.process.pid, "log_path": str(log_path)}, f"started process {item.process.pid}")


def _shell_exec(args: dict[str, Any], context: ToolContext) -> ToolResult:
    item_result = _process_start(args, context)
    item = PROCESSES.get(item_result.content["process_id"])
    timeout = max(1, min(int(args.get("timeout_seconds", 120)), 3600))
    try:
        code = item.process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return ToolResult("", False, {**item_result.content, "running": True}, "command continues in background", error="timeout")
    PROCESSES.close_log_if_exited(item)
    output = item.log_path.read_text(encoding="utf-8", errors="replace") if item.log_path.exists() else ""
    return ToolResult("", code == 0, {**item_result.content, "exit_code": code, "output": output}, f"command exited {code}", error="" if code == 0 else "nonzero_exit")


def _process_list(_args: dict[str, Any], _context: ToolContext) -> ToolResult:
    return ToolResult("", True, {"processes": PROCESSES.list()}, "listed managed processes")


def _process_poll(args: dict[str, Any], _context: ToolContext) -> ToolResult:
    item = PROCESSES.get(str(args["process_id"]))
    code = item.process.poll()
    PROCESSES.close_log_if_exited(item)
    return ToolResult("", True, {"process_id": item.id, "running": code is None, "exit_code": code}, "process is running" if code is None else f"process exited {code}")


def _process_log(args: dict[str, Any], _context: ToolContext) -> ToolResult:
    item = PROCESSES.get(str(args["process_id"]))
    lines = item.log_path.read_text(encoding="utf-8", errors="replace").splitlines() if item.log_path.exists() else []
    limit = max(1, min(int(args.get("lines", 200)), 5000))
    return ToolResult("", True, {"process_id": item.id, "lines": lines[-limit:], "total_lines": len(lines)}, f"read {min(limit, len(lines))} log lines")


def _process_stdin(args: dict[str, Any], _context: ToolContext) -> ToolResult:
    item = PROCESSES.get(str(args["process_id"]))
    if item.process.stdin is None or item.process.poll() is not None:
        return ToolResult("", False, {}, "process stdin unavailable", error="stdin_unavailable")
    data = str(args.get("data") or "").encode("utf-8")
    item.process.stdin.write(data)
    item.process.stdin.flush()
    return ToolResult("", True, {"bytes": len(data)}, f"wrote {len(data)} bytes")


def _process_cancel(args: dict[str, Any], _context: ToolContext) -> ToolResult:
    item = PROCESSES.get(str(args["process_id"]))
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
    return ToolResult("", True, {"status": "ready", "workspace_root": str(context.workspace_root), "runtime_root": str(context.runtime_root), "managed_processes": len(PROCESSES.list())}, "runtime ready")


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
    registry = ToolRegistry()
    def add(name: str, description: str, properties: dict[str, Any], required: list[str], capability: str, handler: Handler, *, read_only: bool = True, cancel: bool = False, available: bool = True, reason: str = "") -> None:
        registry.register(ToolSpec(name, description, {**OBJECT, "properties": properties, "required": required}, capability, read_only=read_only, supports_cancel=cancel, available=available, unavailable_reason=reason), handler)
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
    add("research_invoke", "Invoke an allowlisted legacy Research OS action.", {"action": {"type": "string"}, "input": {"type": "string"}, "timeout_seconds": {"type": "integer"}}, ["action"], "research.execute", _research_invoke, read_only=False)
    add("runtime_health", "Return unified runtime health.", {}, [], "runtime.read", _runtime_health)
    return registry
