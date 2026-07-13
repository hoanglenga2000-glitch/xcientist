"""Start the local HPC SOCKS bridge as a detached, least-privilege process."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_SENSITIVE_ENV = re.compile(
    r"api[_-]?key|authorization|cookie|credential|password|passwd|secret|token",
    re.IGNORECASE,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-script", required=True)
    parser.add_argument("--working-directory", required=True)
    parser.add_argument("--listen-port", required=True, type=int)
    parser.add_argument("--upstream-host", required=True)
    parser.add_argument("--upstream-port", required=True, type=int)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    args = parser.parse_args()

    bridge_script = Path(args.bridge_script).resolve(strict=True)
    working_directory = Path(args.working_directory).resolve(strict=True)
    stdout_log = Path(args.stdout_log).resolve()
    stderr_log = Path(args.stderr_log).resolve()
    if bridge_script.parent != working_directory / "scripts":
        raise SystemExit("bridge script must be the workspace scripts/hpc_socks_bridge.py")
    if bridge_script.name != "hpc_socks_bridge.py":
        raise SystemExit("unexpected bridge script")
    if not 1 <= args.listen_port <= 65535 or not 1 <= args.upstream_port <= 65535:
        raise SystemExit("bridge ports must be between 1 and 65535")

    username = os.environ.get("HPC_SOCKS_USER", "")
    password = os.environ.get("HPC_SOCKS_PASSWORD", "")
    if not username or not password:
        raise SystemExit("HPC SOCKS credentials were not supplied to the launcher")
    child_env = {key: value for key, value in os.environ.items() if not _SENSITIVE_ENV.search(key)}
    child_env["HPC_SOCKS_USER"] = username
    child_env["HPC_SOCKS_PASSWORD"] = password
    child_env["PYTHONUNBUFFERED"] = "1"

    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NO_WINDOW
        )
    command = [
        sys.executable,
        str(bridge_script),
        "--listen-port",
        str(args.listen_port),
        "--upstream-host",
        args.upstream_host,
        "--upstream-port",
        str(args.upstream_port),
    ]
    with stdout_log.open("ab", buffering=0) as stdout_handle, stderr_log.open("ab", buffering=0) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            close_fds=True,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
    print(json.dumps({"status": "started", "pid": process.pid}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
