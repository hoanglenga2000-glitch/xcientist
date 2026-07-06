from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "web" / "research-agent-workstation"
RUNTIME_DIR = APP_DIR / ".runtime-logs"
PID_FILE = RUNTIME_DIR / "dashboard.pid"
OUT_LOG = RUNTIME_DIR / "dashboard.out.log"
ERR_LOG = RUNTIME_DIR / "dashboard.err.log"


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


def next_cli_path() -> str:
    candidate = APP_DIR / "node_modules" / "next" / "dist" / "bin" / "next"
    if candidate.exists():
        return str(candidate)
    raise SystemExit("DASHBOARD_MANAGER_FAILED: Next.js CLI not found under node_modules")


def has_production_build() -> bool:
    return (APP_DIR / ".next" / "BUILD_ID").is_file()


def dashboard_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("WORKSTATION_ROOT", str(ROOT))
    env.setdefault("DATABASE_URL", f"file:{(APP_DIR / 'prisma' / 'workstation.db').as_posix()}")
    env.setdefault("NEXT_TELEMETRY_DISABLED", "1")
    return env


def url_for(port: int, path: str = "") -> str:
    return f"http://127.0.0.1:{port}{path}"


def control_url_for(port: int) -> str:
    return url_for(port, "/?page=control")


def fetch_status(port: int, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(url_for(port, "/api/workstation-summary"), timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {
            "reachable": True,
            "http_status": response.status,
            "task_count": len(payload.get("tasks", [])),
            "has_runtime": bool(payload.get("runtime") or payload.get("runtime_by_task")),
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
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

    result = subprocess.run(["sh", "-c", f"ss -ltnp 'sport = :{port}' 2>/dev/null || true"], text=True, capture_output=True, encoding="utf-8", errors="replace")
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
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        return True

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_running(pid):
            return True
        time.sleep(0.2)
    return not pid_running(pid)


def stop_port(port: int) -> list[int]:
    stopped = []
    for pid in pids_on_port(port):
        if stop_pid(pid):
            stopped.append(pid)
    return stopped


def wait_ready(port: int, timeout: float) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        status = fetch_status(port, timeout=3)
        if status:
            return status
        last = "not reachable"
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
    existing_status = fetch_status(args.port, timeout=2)
    existing_pid = read_pid()
    if existing_status and not args.force:
        print(json.dumps({"status": "already_running", "url": control_url_for(args.port), "health_url": url_for(args.port, "/api/workstation-summary"), "pid": existing_pid, **existing_status}, ensure_ascii=False, indent=2))
        return
    if existing_pid and pid_running(existing_pid):
        stop_pid(existing_pid)
    if existing_status and args.force:
        stop_port(args.port)

    if args.build:
        build = subprocess.run(["npm.cmd" if os.name == "nt" else "npm", "run", "build"], cwd=APP_DIR, text=True, capture_output=True, encoding="utf-8", errors="replace", env=dashboard_env())
        if build.returncode != 0:
            raise SystemExit(json.dumps({"status": "failed", "stage": "build", "stdout": build.stdout, "stderr": build.stderr}, ensure_ascii=False, indent=2))
    mode = "start" if has_production_build() else "dev"
    command = [node_command(), next_cli_path(), mode, "--hostname", "127.0.0.1", "--port", str(args.port)]

    stdout = OUT_LOG.open("ab")
    stderr = ERR_LOG.open("ab")
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        cwd=APP_DIR,
        env=dashboard_env(),
        stdout=stdout,
        stderr=stderr,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    PID_FILE.write_text(str(process.pid), encoding="utf-8")
    ready = wait_ready(args.port, args.timeout)
    print(json.dumps({"status": "started", "url": control_url_for(args.port), "health_url": url_for(args.port, "/api/workstation-summary"), "pid": process.pid, "mode": mode, **ready}, ensure_ascii=False, indent=2))


def stop(args: argparse.Namespace) -> None:
    pid = read_pid()
    stopped_by_port = stop_port(args.port) if args.force else []
    if not pid:
        print(
            json.dumps(
                {
                    "status": "stopped" if stopped_by_port else "not_running",
                    "pid_file": str(PID_FILE.relative_to(ROOT)),
                    "stopped_by_port": stopped_by_port,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    stopped = stop_pid(pid)
    if stopped and PID_FILE.exists():
        PID_FILE.unlink()
    print(json.dumps({"status": "stopped" if stopped else "still_running", "pid": pid, "stopped_by_port": stopped_by_port}, ensure_ascii=False, indent=2))


def status(args: argparse.Namespace) -> None:
    pid = read_pid()
    reachable = fetch_status(args.port, timeout=3)
    print(
        json.dumps(
            {
                "status": "running" if reachable else "not_reachable",
                "url": control_url_for(args.port),
                "health_url": url_for(args.port, "/api/workstation-summary"),
                "pid": pid,
                "pid_running": pid_running(pid),
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
        stop(args)
        start(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
