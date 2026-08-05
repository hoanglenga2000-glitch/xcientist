#!/usr/bin/env python3
"""Persistently monitor the job89941 Taxi CPU candidate without signalling it."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deploy_job89941_taxi_cpu_candidate import (
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_PLAN,
    collect_status,
    validate_plan,
    write_json_atomic,
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def classify_status(value: Mapping[str, Any]) -> str:
    process = value.get("process") if isinstance(value.get("process"), dict) else {}
    result = value.get("result") if isinstance(value.get("result"), dict) else None
    if result is not None:
        if result.get("status") in {
            "candidate_complete",
            "verification_complete_gate_failed",
        }:
            return "terminal_result"
        return "invalid_result"
    if process.get("running") is True and process.get("cmdline_matches") is True:
        return "running"
    if process.get("running") is True:
        return "process_identity_mismatch"
    return "stopped_without_result"


def watch(
    *,
    plan_path: Path,
    evidence_dir: Path,
    poll_seconds: int,
    deadline_hours: float,
) -> dict[str, Any]:
    plan = validate_plan(plan_path)
    deadline = datetime.now().astimezone() + timedelta(hours=deadline_hours)
    watcher_path = evidence_dir / "watcher_current.json"
    while True:
        observed = collect_status(plan, evidence_dir)
        classification = classify_status(observed)
        report = {
            "schema": "evomind.hpc.job89941_taxi_cpu_candidate_watcher.v1",
            "created_at": now_iso(),
            "status": classification,
            "plan_path": plan["_path"],
            "plan_sha256": plan["_sha256"],
            "remote_status": observed,
            "process_signals_sent": 0,
            "other_processes_modified": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        }
        write_json_atomic(watcher_path, report)
        if classification != "running":
            return report
        if datetime.now().astimezone() >= deadline:
            report["status"] = "deadline_reached_process_left_running"
            report["created_at"] = now_iso()
            write_json_atomic(watcher_path, report)
            return report
        time.sleep(poll_seconds)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--deadline-hours", type=float, default=72.0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = watch(
        plan_path=args.plan,
        evidence_dir=args.evidence_dir.resolve(),
        poll_seconds=args.poll_seconds,
        deadline_hours=args.deadline_hours,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
