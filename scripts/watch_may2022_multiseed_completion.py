#!/usr/bin/env python3
"""Collect, verify, aggregate, and Human-Gate-stage May-2022 multiseed results."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import aggregate_may2022_multiseed_candidate as aggregate
from scripts import mlebench_remote_ops as ops
from scripts import queue_hpc88240_may2022_cached as queue
from scripts import stage_mlebench_human_gate_candidate as stage
from scripts import watch_taxi_multiseed_completion as common_watch

DEFAULT_QUEUE_STATUS = PROJECT_ROOT / "workspace" / "hpc" / "job88240_may2022_nested_queue_v8_multiseed_cache_taxi_route" / "status_current.json"
DEFAULT_EVIDENCE_DIR = PROJECT_ROOT / "workspace" / "hpc" / "job88240_may2022_multiseed_postrun"
DEFAULT_STAGED_RUN_ID = "hg_may2022_nested_s424344_20260728"


def run_once(plan, *, queue_status_path: Path, evidence_dir: Path, package_dir: Path, sample_path: Path, staged_run_id: str):
    status_path = evidence_dir / "status_current.json"
    base = {
        "schema": "evomind.hpc88240.may2022_multiseed_postrun.v1", "created_at": common_watch.now_iso(),
        "plan_path": plan["_path"], "plan_sha256": plan["_sha256"], "process_signals_sent": 0,
        "other_processes_modified": False, "private_labels_used": False,
        "official_grader_executed": False, "kaggle_submission_executed": False,
    }
    if not queue_status_path.is_file():
        result = {**base, "status": "waiting_for_may_queue"}
    else:
        current = json.loads(queue_status_path.read_text(encoding="utf-8"))
        if current.get("plan_sha256") != plan["_sha256"]:
            result = {**base, "status": "waiting_for_matching_may_plan", "observed_plan_sha256": current.get("plan_sha256")}
        elif current.get("status") != "all_seeds_terminal":
            result = {**base, "status": "waiting_for_may_seeds_terminal", "may_queue_status": current.get("status")}
        else:
            completed = [str(item.get("run_id")) for item in current.get("completed_seeds") or []]
            if completed != list(plan["run_ids"]):
                raise RuntimeError("May terminal seed inventory changed")
            collections = [ops.collect_run(run_id) for run_id in plan["run_ids"]]
            sample = common_watch.stage_public_sample(plan, sample_path)
            if package_dir.exists():
                verified = stage.verify_human_gate_package(package_dir)
                aggregation = {"status": "existing_package_verified", "candidate_sha256": verified.files["candidate_submission_withheld.csv"].sha256}
            else:
                aggregation = aggregate.aggregate(plan_path=Path(plan["_path"]), collected_root=ops.LOCAL_CONTROL_ROOT / "collected", sample_path=sample_path, output_dir=package_dir)
                stage.verify_human_gate_package(package_dir)
            staged_root = stage.ALLOWED_OUTPUT_ROOT / staged_run_id
            staging = stage.verify_staged_run(staged_root) if staged_root.exists() else stage.stage_candidate_run(package_dir, run_id=staged_run_id)
            result = {**base, "status": "verified_human_approval_pending", "collections": collections, "sample_submission": sample, "aggregation": aggregation, "staging": staging, "approved": False, "automatic_approval": False}
    common_watch.write_json_atomic(status_path, result)
    return result


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=aggregate.DEFAULT_PLAN)
    parser.add_argument("--queue-status", type=Path, default=DEFAULT_QUEUE_STATUS)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--package-dir", type=Path, default=aggregate.DEFAULT_OUTPUT)
    parser.add_argument("--sample-submission", type=Path, default=aggregate.DEFAULT_SAMPLE)
    parser.add_argument("--staged-run-id", default=DEFAULT_STAGED_RUN_ID)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--deadline-hours", type=float, default=504.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.poll_seconds < 30 or args.deadline_hours <= 0:
        raise ValueError("May post-run watcher timing contract is invalid")
    evidence = args.evidence_dir.resolve(); evidence.mkdir(parents=True, exist_ok=True)
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    while True:
        try:
            plan = queue.validate_execution_plan(args.plan)
            result = run_once(plan, queue_status_path=args.queue_status.resolve(), evidence_dir=evidence, package_dir=args.package_dir.resolve(), sample_path=args.sample_submission.resolve(), staged_run_id=stage.validate_run_id(args.staged_run_id))
        except Exception as exc:
            result = {"schema": "evomind.hpc88240.may2022_multiseed_postrun_failure.v1", "created_at": common_watch.now_iso(), "status": "transient_or_contract_failure", "error_type": type(exc).__name__, "error": str(exc), "process_signals_sent": 0, "other_processes_modified": False, "private_labels_used": False, "official_grader_executed": False, "kaggle_submission_executed": False}
            common_watch.write_json_atomic(evidence / "failure_current.json", result)
            if args.once: print(json.dumps(result, ensure_ascii=False, indent=2)); return 1
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if result["status"] == "verified_human_approval_pending": return 0
            if args.once: return 0
        if datetime.now().astimezone() >= deadline: return 4
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
