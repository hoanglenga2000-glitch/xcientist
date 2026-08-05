#!/usr/bin/env python3
"""Capture read-only process progress for the job89941 Taxi CPU candidate."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deploy_job89941_taxi_cpu_candidate import (  # noqa: E402
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_PLAN,
    connect_job89941,
    exec_json,
    validate_status_plan,
    write_json_atomic,
)

DEFAULT_OUTPUT = DEFAULT_EVIDENCE_DIR / "progress_current.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def encoded_python(source: str) -> str:
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    return f"python3 -c {shlex.quote('import base64;exec(base64.b64decode(' + repr(encoded) + '))')}"


def compute_delta(current: Mapping[str, Any], previous: Mapping[str, Any] | None) -> dict[str, Any]:
    if not previous:
        return {"available": False, "making_progress": None}
    fields = {
        "utime_ticks": int(current.get("utime_ticks") or 0) - int(previous.get("utime_ticks") or 0),
        "stime_ticks": int(current.get("stime_ticks") or 0) - int(previous.get("stime_ticks") or 0),
        "rchar": int((current.get("io") or {}).get("rchar") or 0) - int((previous.get("io") or {}).get("rchar") or 0),
        "wchar": int((current.get("io") or {}).get("wchar") or 0) - int((previous.get("io") or {}).get("wchar") or 0),
        "log_bytes": int(current.get("log_bytes") or 0) - int(previous.get("log_bytes") or 0),
    }
    fields["available"] = True
    fields["making_progress"] = bool(
        fields["utime_ticks"] > 0
        or fields["stime_ticks"] > 0
        or fields["rchar"] > 0
        or fields["wchar"] > 0
        or fields["log_bytes"] > 0
        or current.get("result_exists") is True
    )
    return fields


def probe_once(plan_path: Path, output: Path) -> dict[str, Any]:
    plan = validate_status_plan(plan_path)
    source = f'''import json,os,pathlib,time
state_path=pathlib.Path({str(plan['remote_state'])!r}); state=json.loads(state_path.read_text())
pid=int(state.get("pid") or 0); proc=pathlib.Path("/proc")/str(pid); exists=bool(pid and proc.is_dir())
payload={{"captured_at_epoch":time.time(),"pid":pid,"process_exists":exists}}
if exists:
 s=(proc/"stat").read_text().split(); payload.update({{"process_state":s[2],"utime_ticks":int(s[13]),"stime_ticks":int(s[14])}})
 status={{}}
 for line in (proc/"status").read_text().splitlines():
  if line.startswith(("VmRSS:","VmSize:","Threads:")): k,v=line.split(":",1); status[k]=v.strip()
 payload["process_status"]=status
 io={{}}
 for line in (proc/"io").read_text().splitlines(): k,v=line.split(":",1); io[k]=int(v.strip())
 payload["io"]=io
log=pathlib.Path({str(plan['remote_log'])!r}); result=pathlib.Path({str(PurePosixPath(str(plan['remote_output_dir'])) / 'result.json')!r})
payload.update({{"log_bytes":log.stat().st_size if log.is_file() else 0,"result_exists":result.is_file(),"loadavg":os.getloadavg()}})
print(json.dumps(payload))
'''
    client = connect_job89941()
    try:
        remote = exec_json(client, encoded_python(source), timeout=300)
    finally:
        client.close()
    previous = None
    if output.is_file():
        try:
            previous_report = json.loads(output.read_text(encoding="utf-8"))
            previous = previous_report.get("sample")
        except (OSError, json.JSONDecodeError):
            previous = None
    delta = compute_delta(remote, previous if isinstance(previous, dict) else None)
    report = {
        "schema": "evomind.hpc.job89941_taxi_cpu_progress.v1",
        "created_at": now_iso(),
        "status": "running" if remote.get("process_exists") else "terminal_or_missing",
        "plan_path": plan["_path"],
        "plan_sha256": plan["_sha256"],
        "sample": remote,
        "delta_from_previous": delta,
        "stalled": delta.get("making_progress") is False,
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    write_json_atomic(output, report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--deadline-hours", type=float, default=72.0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    while True:
        report = probe_once(args.plan, args.output.resolve())
        if not args.watch or report.get("status") != "running":
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if datetime.now().astimezone() >= deadline:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
