from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:65068/v1"


def state_root() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "ResearchAgentWorkstation"
    return Path.home() / ".research-agent-workstation"


def workspace_root() -> Path:
    configured = os.environ.get("WORKSTATION_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if (ROOT / "app" / "server.js").is_file():
        return (ROOT / "user-data").resolve()
    return ROOT.resolve()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def validate_base_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value.rstrip("/"))
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.port != 65068:
        raise SystemExit("LOCAL_GATEWAY_FAILED: base URL must be the loopback gateway on port 65068")
    if parsed.path.rstrip("/") != "/v1":
        raise SystemExit("LOCAL_GATEWAY_FAILED: base URL path must be /v1")
    return value.rstrip("/")


def pid_file() -> Path:
    return state_root() / "local_gateway.pid"


def process_state_file() -> Path:
    return state_root() / "local_gateway.process.json"


def process_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid() -> int | None:
    try:
        return int(pid_file().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def read_process_state() -> dict[str, Any]:
    try:
        payload = json.loads(process_state_file().read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def process_command_line(pid: int) -> str:
    if not process_alive(pid):
        return ""
    try:
        if os.name == "nt":
            script = (
                f"$p=Get-CimInstance Win32_Process -Filter \"ProcessId = {pid}\" "
                "-ErrorAction SilentlyContinue; if($p){[Console]::Out.Write($p.CommandLine)}"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def managed_process_matches(pid: int | None) -> bool:
    if not pid or not process_alive(pid):
        return False
    state = read_process_state()
    if state.get("pid") != pid:
        return False
    expected = str(state.get("executable", "")).casefold()
    command = process_command_line(pid).casefold()
    return bool(expected and command and expected in command)


def probe(base_url: str, timeout: float = 2.5) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(base_url)
    tcp = False
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 65068), timeout=min(timeout, 1.5)):
            tcp = True
    except OSError:
        pass
    result: dict[str, Any] = {
        "base_url": base_url,
        "tcp_reachable": tcp,
        "http_status": None,
        "models_endpoint_ok": False,
        "credential_present": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
    }
    if not tcp:
        return result
    headers = {"Accept": "application/json"}
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(base_url + "/models", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(2 * 1024 * 1024)
            result["http_status"] = response.status
            payload = json.loads(body.decode("utf-8"))
            result["models_endpoint_ok"] = response.status == 200 and isinstance(payload, dict)
    except urllib.error.HTTPError as error:
        result["http_status"] = error.code
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        pass
    return result


def runtime_manifest() -> Path | None:
    configured = os.environ.get("WORKSTATION_GATEWAY_MANIFEST", "").strip()
    candidates = [Path(configured)] if configured else []
    candidates.extend([ROOT / "runtime" / "gateway" / "manifest.json", state_root() / "gateway_runtime.json"])
    return next((path.expanduser().resolve() for path in candidates if path.is_file()), None)


def start_runtime() -> tuple[int | None, str]:
    manifest_path = runtime_manifest()
    if not manifest_path:
        return None, "runtime_manifest_missing"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    executable = Path(str(payload.get("executable", ""))).expanduser()
    if not executable.is_absolute():
        executable = (manifest_path.parent / executable).resolve()
    if not executable.is_file():
        return None, "runtime_executable_missing"
    args = payload.get("args", [])
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        return None, "runtime_args_invalid"
    cwd_value = payload.get("cwd")
    cwd = Path(cwd_value).expanduser().resolve() if cwd_value else executable.parent
    logs = state_root() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout = (logs / "local-gateway.out.log").open("ab")
    stderr = (logs / "local-gateway.err.log").open("ab")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    try:
        process = subprocess.Popen([str(executable), *args], cwd=cwd, env=os.environ.copy(), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, creationflags=creationflags, close_fds=True)
    finally:
        stdout.close()
        stderr.close()
    pid_file().parent.mkdir(parents=True, exist_ok=True)
    pid_file().write_text(str(process.pid), encoding="utf-8")
    atomic_json(
        process_state_file(),
        {
            "format_version": 1,
            "pid": process.pid,
            "executable": str(executable.resolve()),
            "manifest": str(manifest_path),
            "cwd": str(cwd),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    return process.pid, "runtime_started"


def stop_runtime(timeout: float = 15.0) -> tuple[int | None, bool]:
    pid = read_pid()
    if not pid or not process_alive(pid):
        pid_file().unlink(missing_ok=True)
        process_state_file().unlink(missing_ok=True)
        return pid, True
    if not managed_process_matches(pid):
        # A stale PID may have been reused by another process. Drop only the
        # stale ownership marker; never terminate an unverified process.
        pid_file().unlink(missing_ok=True)
        process_state_file().unlink(missing_ok=True)
        return pid, True
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=20)
    else:
        import signal

        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and process_alive(pid):
        time.sleep(0.2)
    clean = not process_alive(pid)
    if clean:
        pid_file().unlink(missing_ok=True)
        process_state_file().unlink(missing_ok=True)
    return pid, clean


def publish_status(probe_result: dict[str, Any], runtime_action: str, managed_pid: int | None) -> dict[str, Any]:
    ready = bool(probe_result.get("models_endpoint_ok"))
    if ready:
        mode = "ready"
        state = "Local Gateway Ready"
        notes = "Loopback OpenAI-compatible gateway is healthy; streaming/tool calls can use it."
    elif probe_result.get("tcp_reachable"):
        mode = "degraded_auth" if probe_result.get("http_status") in {401, 403} else "degraded_http"
        state = "Local Gateway Degraded"
        notes = "Gateway process is reachable but model readiness is incomplete; deterministic local fallback remains available."
    else:
        mode = "degraded_unavailable"
        state = "Local Deterministic Fallback"
        notes = "Gateway runtime is unavailable; the workstation remains usable through its deterministic local fallback."
    report = {
        "status": mode,
        "ready": ready,
        "degraded": not ready,
        "local_fallback_available": True,
        "runtime_action": runtime_action,
        "managed_pid": managed_pid,
        "managed_pid_running": managed_process_matches(managed_pid),
        "probe": probe_result,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    root = workspace_root()
    atomic_json(root / "workspace" / "runtime" / "local_gateway_status.json", report)
    summary_path = root / "workspace" / "workstation_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                summary = loaded
        except (OSError, json.JSONDecodeError):
            summary = {}
    connectors = summary.setdefault("connector_status", {})
    if not isinstance(connectors, dict):
        connectors = {}
        summary["connector_status"] = connectors
    connectors["local_gateway"] = {
        "name": "Local Model Gateway",
        "state": state,
        "configured": ready,
        "notes": notes,
    }
    atomic_json(summary_path, summary)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover, start, probe and diagnose the loopback model gateway.")
    parser.add_argument("command", choices=["start", "status", "diagnose", "stop"])
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--strict", action="store_true", help="Return a failure exit code when the gateway is degraded.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base_url = validate_base_url(args.base_url)
    runtime_action = "status_only"
    managed_pid = read_pid()
    if args.command == "stop":
        stopped_pid, clean = stop_runtime(args.timeout)
        result = {"status": "stopped" if clean else "still_running", "managed_pid": stopped_pid, "clean": clean}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if clean else 1
    result = probe(base_url)
    if args.command == "start" and not result["models_endpoint_ok"] and not result["tcp_reachable"]:
        managed_pid, runtime_action = start_runtime()
        if managed_pid:
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                result = probe(base_url)
                if result["tcp_reachable"]:
                    break
                if not process_alive(managed_pid):
                    break
                time.sleep(0.4)
    report = publish_status(result, runtime_action, managed_pid)
    report["manifest"] = str(runtime_manifest()) if runtime_manifest() else None
    if args.output:
        atomic_json(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if args.strict and not report["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
