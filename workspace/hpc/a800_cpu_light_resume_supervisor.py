"""Resume one suspended CPU-light A800 run after the GPU queue finishes."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.research_agent_workstation.server.core.gpu_credentials import (  # noqa: E402
    ALLOWED_GPU_REMOTE_ROOT,
    connect_ssh,
)
from workspace.hpc.probe_hpc_gpu_profile import _load_profile  # noqa: E402

RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
DEFAULT_MARKER = PROJECT_ROOT / "workspace" / "hpc" / "a800_recovery_queue_current.json"
DEFAULT_EVIDENCE = (
    PROJECT_ROOT / "workspace" / "hpc" / "a800_cpu_light_resume_current.json"
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def queue_allows_resume(marker: dict[str, Any], run_id: str) -> bool:
    parallel_runs = marker.get("parallel_runs") or {}
    return bool(
        marker.get("schema") == "evomind.mlebench.a800_recovery_queue.v1"
        and marker.get("status") == "completed"
        and run_id in set(parallel_runs.values())
    )


def inspect_or_resume(profile_dir: Path, *, run_id: str, remote_pid: int) -> dict[str, Any]:
    quoted_run_id = shlex.quote(run_id)
    quoted_root = shlex.quote(ALLOWED_GPU_REMOTE_ROOT)
    command = f"""RUN_ID={quoted_run_id}
ROOT={quoted_root}
PID={remote_pid}
if ! kill -0 "$PID" 2>/dev/null; then
  printf 'RESULT=already_exited\n'
  exit 0
fi
CMD=$(tr '\\000' ' ' < "/proc/$PID/cmdline")
USER_NAME=$(ps -o user= -p "$PID" | xargs)
STATE=$(ps -o state= -p "$PID" | xargs)
case "$CMD" in *"$ROOT"*"$RUN_ID"*) ;; *) printf 'RESULT=pid_contract_mismatch\n'; exit 42;; esac
[ "$USER_NAME" = "aimslab" ] || {{ printf 'RESULT=pid_owner_mismatch\n'; exit 43; }}
OTHER_RUNNERS=$(ps -eo pid=,ppid=,state=,args= | awk -v pid="$PID" -v root="$ROOT" '
  index($0, root) && index($0, "run_mlebench_lite_full.py") && $1 != pid {{ print }}')
COMPUTE_APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits 2>/dev/null || true)
if [ -n "$OTHER_RUNNERS" ] || [ -n "$COMPUTE_APPS" ]; then
  printf 'RESULT=busy\nSTATE=%s\nOTHER_RUNNERS=%s\nCOMPUTE_APPS=%s\n' \
    "$STATE" "$(printf '%s' "$OTHER_RUNNERS" | wc -l)" \
    "$(printf '%s' "$COMPUTE_APPS" | wc -l)"
  exit 0
fi
if [ "$STATE" = "T" ]; then
  kill -CONT "$PID"
  sleep 1
fi
NEW_STATE=$(ps -o state= -p "$PID" | xargs)
printf 'RESULT=resumed\nPREVIOUS_STATE=%s\nSTATE=%s\nPID=%s\n' "$STATE" "$NEW_STATE" "$PID"
"""
    config = _load_profile(profile_dir.resolve())
    client = connect_ssh(config, timeout=25)
    try:
        _stdin, stdout, stderr = client.exec_command(command, timeout=30)
        output = stdout.read().decode("utf-8", "replace").strip()
        error = stderr.read().decode("utf-8", "replace").strip()
        exit_code = stdout.channel.recv_exit_status()
    finally:
        client.close()
    fields = {
        key: value
        for line in output.splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }
    return {
        "exit_code": exit_code,
        "result": fields.get("RESULT", "unknown"),
        "fields": fields,
        "stderr_present": bool(error),
        "stderr_type": error.splitlines()[-1] if error else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--remote-pid", type=int, required=True)
    parser.add_argument("--queue-marker", type=Path, default=DEFAULT_MARKER)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("Run id is malformed")
    if args.remote_pid <= 1:
        raise ValueError("Remote PID is invalid")

    while True:
        marker = json.loads(args.queue_marker.read_text(encoding="utf-8-sig"))
        evidence: dict[str, Any] = {
            "schema": "evomind.hpc.cpu_light_resume_supervisor.v1",
            "created_at": now_iso(),
            "pid": os.getpid(),
            "run_id": args.run_id,
            "remote_pid": args.remote_pid,
            "queue_status": marker.get("status"),
            "human_gate_preserved": True,
            "kaggle_submission_enabled": False,
        }
        if not queue_allows_resume(marker, args.run_id):
            evidence["status"] = "waiting_for_gpu_queue"
            write_json_atomic(args.evidence.resolve(), evidence)
            if args.once:
                print(json.dumps(evidence, ensure_ascii=False, indent=2))
                return 0
            time.sleep(max(10, args.poll_seconds))
            continue

        remote = inspect_or_resume(
            args.profile_dir,
            run_id=args.run_id,
            remote_pid=args.remote_pid,
        )
        evidence["remote"] = remote
        if remote["exit_code"]:
            evidence["status"] = "remote_contract_failed"
            write_json_atomic(args.evidence.resolve(), evidence)
            return 1
        if remote["result"] == "busy":
            evidence["status"] = "waiting_for_remote_idle"
            write_json_atomic(args.evidence.resolve(), evidence)
            if args.once:
                print(json.dumps(evidence, ensure_ascii=False, indent=2))
                return 0
            time.sleep(max(10, args.poll_seconds))
            continue
        evidence["status"] = remote["result"]
        write_json_atomic(args.evidence.resolve(), evidence)
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
