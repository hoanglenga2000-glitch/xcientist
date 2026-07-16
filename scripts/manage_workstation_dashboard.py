from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "web" / "research-agent-workstation"
RUNTIME_DIR = APP_DIR / ".runtime-logs"
PID_FILE = RUNTIME_DIR / "dashboard.pid"
OUT_LOG = RUNTIME_DIR / "dashboard.out.log"
ERR_LOG = RUNTIME_DIR / "dashboard.err.log"
STATE_FILE = RUNTIME_DIR / "dashboard.state.json"
PRISMA_PUSH_SCRIPT = APP_DIR / "scripts" / "prisma-db-push.mjs"
DEFAULT_DATABASE_PATH = APP_DIR / "prisma" / "workstation.db"
DATABASE_RUNTIME_SUFFIXES = (".db", ".db-journal", ".db-shm", ".db-wal")

SOURCE_FILES = {
    "package.json",
    "package-lock.json",
    "next.config.mjs",
    "postcss.config.mjs",
    "tailwind.config.ts",
    "tsconfig.json",
}
SOURCE_DIRECTORIES = ("src", "prisma", "public", "scripts")


def npx_command() -> str:
    candidate = shutil.which("npx.cmd") or shutil.which("npx")
    if candidate:
        return candidate
    local_node = Path("D:/下载/npx.cmd")
    if local_node.exists():
        return str(local_node)
    raise SystemExit("DASHBOARD_MANAGER_FAILED: npx/npx.cmd not found on PATH")


def node_command() -> str:
    candidate = shutil.which("node.exe") or shutil.which("node")
    if candidate:
        return candidate
    raise SystemExit("DASHBOARD_MANAGER_FAILED: node/node.exe not found on PATH")


def npm_command() -> str:
    candidate = shutil.which("npm.cmd") or shutil.which("npm")
    if candidate:
        return candidate
    raise SystemExit("DASHBOARD_MANAGER_FAILED: npm/npm.cmd not found on PATH")


def next_cli_path() -> str:
    candidate = APP_DIR / "node_modules" / "next" / "dist" / "bin" / "next"
    if candidate.exists():
        return str(candidate)
    raise SystemExit("DASHBOARD_MANAGER_FAILED: Next.js CLI not found under node_modules")


def has_production_build() -> bool:
    return (APP_DIR / ".next" / "BUILD_ID").is_file()


def production_build_id(app_dir: Path = APP_DIR) -> str | None:
    path = app_dir / ".next" / "BUILD_ID"
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def source_tree_digest(app_dir: Path = APP_DIR) -> str:
    files = [app_dir / name for name in SOURCE_FILES if (app_dir / name).is_file()]
    for directory in SOURCE_DIRECTORIES:
        root = app_dir / directory
        if root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and not path.name.lower().endswith(DATABASE_RUNTIME_SUFFIXES)
            )
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(app_dir).as_posix()):
        relative = path.relative_to(app_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def read_runtime_state() -> dict:
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def dashboard_env() -> dict[str, str]:
    env = os.environ.copy()
    env["WORKSTATION_ROOT"] = str(ROOT)
    env["DATABASE_URL"] = f"file:{DEFAULT_DATABASE_PATH.as_posix()}"
    env["WORKSTATION_PYTHON"] = sys.executable
    env.setdefault("NEXT_TELEMETRY_DISABLED", "1")
    return env


def redact_process_output(value: str, env: dict[str, str]) -> str:
    database_url = env.get("DATABASE_URL", "")
    if database_url:
        return value.replace(database_url, "<redacted DATABASE_URL>")
    return value


def ensure_database_schema(env: dict[str, str]) -> str:
    if not PRISMA_PUSH_SCRIPT.is_file():
        raise SystemExit(
            json.dumps(
                {
                    "status": "failed",
                    "stage": "database_schema",
                    "message": "Prisma database initialization script is missing.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    result = subprocess.run(
        [node_command(), str(PRISMA_PUSH_SCRIPT), "--skip-generate"],
        cwd=APP_DIR,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if result.returncode != 0:
        raise SystemExit(
            json.dumps(
                {
                    "status": "failed",
                    "stage": "database_schema",
                    "message": "Prisma database schema initialization failed.",
                    "stdout": redact_process_output(result.stdout, env),
                    "stderr": redact_process_output(result.stderr, env),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    generate = subprocess.run(
        [npm_command(), "run", "db:generate"],
        cwd=APP_DIR,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if generate.returncode != 0:
        raise SystemExit(
            json.dumps(
                {
                    "status": "failed",
                    "stage": "prisma_client",
                    "message": "Prisma client generation failed.",
                    "stdout": redact_process_output(generate.stdout, env),
                    "stderr": redact_process_output(generate.stderr, env),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return "synced"


def url_for(port: int, path: str = "") -> str:
    return f"http://127.0.0.1:{port}{path}"


def control_url_for(port: int) -> str:
    return url_for(port, "/?page=control")


def fetch_status(port: int, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(url_for(port, "/api/workstation-summary"), timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if response.status != 200 or not isinstance(payload, dict):
            return None
        return {
            "reachable": True,
            "http_status": response.status,
            "task_count": len(payload.get("tasks", [])),
            "has_runtime": bool(payload.get("runtime") or payload.get("runtime_by_task")),
        }
    except (urllib.error.URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def process_command_line(pid: int) -> str | None:
    if os.name == "nt":
        script = (
            "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
            f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}' -ErrorAction SilentlyContinue; "
            "if ($p) { $p.CommandLine }"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        value = result.stdout.strip()
        return value or None

    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        if proc_cmdline.is_file():
            value = proc_cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
            return value or None
    except OSError:
        pass
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    value = result.stdout.strip()
    return value or None


def process_matches_dashboard(pid: int, port: int) -> bool:
    command_line = process_command_line(pid)
    if not command_line:
        return False
    normalized = command_line.replace("\\", "/").lower()
    expected_cli = str(Path(next_cli_path()).resolve()).replace("\\", "/").lower()
    port_pattern = re.compile(rf"(?:^|\s)--port(?:=|\s+)[\"']?{port}(?:[\"']?(?:\s|$))")
    return expected_cli in normalized and bool(port_pattern.search(normalized))


def runtime_state_matches_process(pid: int, port: int, state: dict | None = None) -> bool:
    runtime_state = state if state is not None else read_runtime_state()
    return (
        runtime_state.get("schema") == "evomind.dashboard_runtime.v1"
        and runtime_state.get("pid") == pid
        and runtime_state.get("port") == port
        and process_matches_dashboard(pid, port)
    )


def port_processes(port: int, state: dict | None = None) -> tuple[list[int], list[int]]:
    runtime_state = state if state is not None else read_runtime_state()
    owned: list[int] = []
    unowned: list[int] = []
    for pid in pids_on_port(port):
        target = owned if runtime_state_matches_process(pid, port, runtime_state) else unowned
        target.append(pid)
    return sorted(set(owned)), sorted(set(unowned))


def fail_runtime(stage: str, message: str, **evidence: object) -> None:
    raise SystemExit(
        json.dumps(
            {
                "status": "failed",
                "stage": stage,
                "message": message,
                "evidence": evidence,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def pids_on_port(port: int) -> list[int]:
    if os.name == "nt":
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique",
        ]
        result = subprocess.run(command, text=True, capture_output=True, encoding="utf-8", errors="replace")
        pids: list[int] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        return pids

    result = subprocess.run(
        ["sh", "-c", f"ss -ltnp 'sport = :{port}' 2>/dev/null || true"],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    pids = []
    for token in result.stdout.replace(",", " ").split():
        if token.startswith("pid="):
            value = token.removeprefix("pid=").strip()
            if value.isdigit():
                pids.append(int(value))
    return sorted(set(pids))


def stop_pid(pid: int, timeout: float = 10.0) -> bool:
    try:
        if os.name == "nt":
            graceful = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if graceful.returncode != 0 and not pid_running(pid):
                return True
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        return not pid_running(pid)

    graceful_deadline = time.time() + max(1.0, timeout / 2)
    while time.time() < graceful_deadline:
        if not pid_running(pid):
            return True
        time.sleep(0.2)

    try:
        if os.name == "nt":
            forced = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if forced.returncode != 0 and not pid_running(pid):
                return True
        else:
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError:
        return not pid_running(pid)

    deadline = time.time() + max(1.0, timeout / 2)
    while time.time() < deadline:
        if not pid_running(pid):
            return True
        time.sleep(0.2)
    return not pid_running(pid)


def stop_port(port: int, state: dict | None = None) -> list[int]:
    owned, unowned = port_processes(port, state)
    if unowned:
        fail_runtime(
            "port_ownership",
            "Refusing to stop a listener that is not owned by this workstation runtime.",
            port=port,
            unowned_pids=unowned,
        )
    stopped = [pid for pid in owned if stop_pid(pid)]
    failed = sorted(set(owned) - set(stopped))
    if failed:
        fail_runtime(
            "port_cleanup",
            "One or more owned dashboard listeners could not be stopped; runtime metadata was preserved.",
            port=port,
            failed_pids=failed,
        )
    return stopped


def wait_ready(port: int, timeout: float, expected_pid: int | None = None) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        status = fetch_status(port, timeout=3)
        if status and (
            expected_pid is None
            or (expected_pid in pids_on_port(port) and runtime_state_matches_process(expected_pid, port))
        ):
            return status
        last = "not reachable or listener ownership did not match the launched process"
        time.sleep(0.7)
    raise SystemExit(
        json.dumps(
            {
                "status": "failed",
                "message": f"dashboard did not become ready on {url_for(port)}",
                "last_status": last,
                "stdout_log": str(OUT_LOG.relative_to(ROOT)),
                "stderr_log": str(ERR_LOG.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def start(args: argparse.Namespace) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    runtime_state = read_runtime_state()
    existing_status = fetch_status(args.port, timeout=2)
    existing_pid = read_pid()
    owned_listeners, unowned_listeners = port_processes(args.port, runtime_state)
    if unowned_listeners:
        fail_runtime(
            "port_ownership",
            "The requested port is occupied by a process that is not owned by this workstation runtime.",
            port=args.port,
            unowned_pids=unowned_listeners,
        )

    if owned_listeners and not args.force:
        current_source_digest = source_tree_digest()
        current_build_id = production_build_id()
        runtime_is_current = (
            runtime_state.get("source_digest") == current_source_digest
            and runtime_state.get("build_id") == current_build_id
        )
        if existing_status and runtime_is_current:
            print(
                json.dumps(
                    {
                        "status": "already_running",
                        "url": control_url_for(args.port),
                        "health_url": url_for(args.port, "/api/workstation-summary"),
                        "pid": owned_listeners[0],
                        **existing_status,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        fail_runtime(
            "existing_runtime",
            "An owned dashboard listener exists but is unhealthy or stale; use restart to replace it.",
            port=args.port,
            owned_pids=owned_listeners,
            reachable=bool(existing_status),
            source_matches=runtime_state.get("source_digest") == current_source_digest,
            build_matches=runtime_state.get("build_id") == current_build_id,
        )

    if owned_listeners:
        stop_port(args.port, runtime_state)

    if existing_pid and pid_running(existing_pid):
        if runtime_state_matches_process(existing_pid, args.port, runtime_state):
            if not args.force:
                fail_runtime(
                    "existing_runtime",
                    "An owned dashboard process is still running without a healthy listener; use restart to replace it.",
                    port=args.port,
                    pid=existing_pid,
                )
            if not stop_pid(existing_pid):
                fail_runtime(
                    "existing_runtime_cleanup",
                    "The owned dashboard process could not be stopped; runtime metadata was preserved.",
                    port=args.port,
                    pid=existing_pid,
                )
        else:
            fail_runtime(
                "pid_ownership",
                "PID metadata references a running process whose workstation identity cannot be proven.",
                port=args.port,
                pid=existing_pid,
            )
    elif existing_pid or runtime_state:
        PID_FILE.unlink(missing_ok=True)
        STATE_FILE.unlink(missing_ok=True)

    environment = dashboard_env()
    database_schema_status = ensure_database_schema(environment)
    if args.build:
        build = subprocess.run(
            [npm_command(), "run", "build"],
            cwd=APP_DIR,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        if build.returncode != 0:
            raise SystemExit(
                json.dumps(
                    {"status": "failed", "stage": "build", "stdout": build.stdout, "stderr": build.stderr},
                    ensure_ascii=False,
                    indent=2,
                )
            )
    mode = "start" if has_production_build() else "dev"
    if args.build and mode != "start":
        raise SystemExit("DASHBOARD_MANAGER_FAILED: npm build completed without a production BUILD_ID")
    build_id = production_build_id()
    source_digest = source_tree_digest()
    command = [node_command(), next_cli_path(), mode, "--hostname", "127.0.0.1", "--port", str(args.port)]

    stdout = OUT_LOG.open("ab")
    stderr = ERR_LOG.open("ab")
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        process = subprocess.Popen(
            command,
            cwd=APP_DIR,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    finally:
        stdout.close()
        stderr.close()
    PID_FILE.write_text(str(process.pid), encoding="utf-8")
    runtime_state = {
        "schema": "evomind.dashboard_runtime.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pid": process.pid,
        "port": args.port,
        "mode": mode,
        "build_id": build_id,
        "source_digest": source_digest,
        "build_requested": bool(args.build),
        "database_schema_status": database_schema_status,
    }
    STATE_FILE.write_text(json.dumps(runtime_state, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        ready = wait_ready(args.port, args.timeout, expected_pid=process.pid)
    except SystemExit as readiness_error:
        if stop_pid(process.pid):
            PID_FILE.unlink(missing_ok=True)
            STATE_FILE.unlink(missing_ok=True)
            raise
        fail_runtime(
            "readiness_cleanup",
            "Dashboard readiness failed and the launched process could not be stopped; runtime metadata was preserved.",
            pid=process.pid,
            port=args.port,
            readiness_error=str(readiness_error),
        )
    print(
        json.dumps(
            {
                "status": "started",
                "url": control_url_for(args.port),
                "health_url": url_for(args.port, "/api/workstation-summary"),
                **runtime_state,
                **ready,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def stop(args: argparse.Namespace, *, emit: bool = True) -> None:
    pid = read_pid()
    runtime_state = read_runtime_state()
    owned_listeners, unowned_listeners = port_processes(args.port, runtime_state)
    if unowned_listeners:
        fail_runtime(
            "port_ownership",
            "Refusing to stop a listener that is not owned by this workstation runtime.",
            port=args.port,
            unowned_pids=unowned_listeners,
        )

    candidates = set(owned_listeners)
    if pid and runtime_state_matches_process(pid, args.port, runtime_state):
        candidates.add(pid)
    elif pid and pid_running(pid):
        fail_runtime(
            "pid_ownership",
            "PID metadata references a running process whose workstation identity cannot be proven.",
            port=args.port,
            pid=pid,
        )
    stale_metadata = bool(pid or runtime_state) and not candidates
    stopped_pids = [candidate for candidate in sorted(candidates) if stop_pid(candidate)]
    stopped = len(stopped_pids) == len(candidates)
    if stopped:
        PID_FILE.unlink(missing_ok=True)
        STATE_FILE.unlink(missing_ok=True)
    if emit:
        print(
            json.dumps(
                {
                    "status": "stopped" if stopped and stopped_pids else "not_running" if stopped else "still_running",
                    "pid": pid,
                    "stopped_pids": stopped_pids,
                    "stale_metadata_removed": stale_metadata and stopped,
                    "force_requested": bool(args.force),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def status(args: argparse.Namespace) -> None:
    pid = read_pid()
    reachable = fetch_status(args.port, timeout=3)
    runtime_state = read_runtime_state()
    owned_listeners, unowned_listeners = port_processes(args.port, runtime_state)
    runtime_is_current = (
        runtime_state.get("source_digest") == source_tree_digest()
        and runtime_state.get("build_id") == production_build_id()
    )
    print(
        json.dumps(
            {
                "status": "running"
                if reachable and owned_listeners and runtime_is_current
                else "port_conflict"
                if unowned_listeners
                else "stale_runtime"
                if owned_listeners
                else "not_reachable",
                "url": control_url_for(args.port),
                "health_url": url_for(args.port, "/api/workstation-summary"),
                "pid": pid,
                "pid_running": pid_running(pid),
                "owned_listener_pids": owned_listeners,
                "unowned_listener_pids": unowned_listeners,
                "runtime_is_current": runtime_is_current,
                "runtime_state": runtime_state,
                "health": reachable,
                "logs": {
                    "stdout": str(OUT_LOG.relative_to(ROOT)),
                    "stderr": str(ERR_LOG.relative_to(ROOT)),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the local Next.js Research Agent Workstation dashboard.")
    parser.add_argument("command", choices=["start", "stop", "restart", "status"])
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--build", action="store_true", help="Run npm build before start/restart.")
    parser.add_argument("--force", action="store_true", help="Restart even if the dashboard is already reachable.")
    args = parser.parse_args()

    if args.command == "start":
        start(args)
    elif args.command == "stop":
        stop(args)
    elif args.command == "restart":
        args.force = True
        stop(args, emit=False)
        start(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
