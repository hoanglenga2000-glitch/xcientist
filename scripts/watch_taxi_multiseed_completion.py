#!/usr/bin/env python3
"""Collect, verify, aggregate, and stage Taxi after all three GPU seeds finish."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import aggregate_taxi_multiseed_candidate as aggregate  # noqa: E402
from scripts import mlebench_remote_ops as ops  # noqa: E402
from scripts import queue_hpc88240_taxi_after_may as queue  # noqa: E402
from scripts import stage_mlebench_human_gate_candidate as stage  # noqa: E402

DEFAULT_QUEUE_STATUS = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "job88240_taxi_source_audited_queue_v5_may_multiseed_route_cache"
    / "status_current.json"
)
DEFAULT_EVIDENCE_DIR = (
    PROJECT_ROOT / "workspace" / "hpc" / "job88240_taxi_multiseed_postrun_v2_may_multiseed"
)
DEFAULT_STAGED_RUN_ID = "hg_taxi_route_cache_s434445_20260728"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _base(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "evomind.hpc88240.taxi_multiseed_postrun.v1",
        "created_at": now_iso(),
        "plan_path": plan["_path"],
        "plan_sha256": plan["_sha256"],
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def stage_public_sample(plan: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    record = plan.get("public_sample_submission") or {}
    expected_bytes = int(record.get("bytes", -1))
    expected_sha = str(record.get("sha256") or "")
    remote = ops.ensure_remote_path(str(record.get("remote_path") or ""))
    if not remote.startswith(ops.REMOTE_DATA_ROOT + "/") or not remote.endswith(
        "/prepared/public/sample_submission.csv"
    ):
        raise RuntimeError("Taxi public sample remote path drifted")
    destination = Path(destination).resolve()
    allowed = (PROJECT_ROOT / "workspace" / "local_gpu" / "mlebench_official_data").resolve()
    try:
        destination.relative_to(allowed)
    except ValueError as exc:
        raise RuntimeError("Taxi public sample destination escaped local data root") from exc
    if (
        destination.is_file()
        and destination.stat().st_size == expected_bytes
        and aggregate.sha256_file(destination) == expected_sha
    ):
        return {**aggregate.file_record(destination), "downloaded": False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    client = ops._connect()
    try:
        with client.open_sftp() as sftp:
            item = sftp.lstat(remote)
            if int(item.st_size) != expected_bytes:
                raise RuntimeError("Taxi public sample remote size drifted")
            sftp.get(remote, str(temporary))
    finally:
        client.close()
    if (
        temporary.stat().st_size != expected_bytes
        or aggregate.sha256_file(temporary) != expected_sha
    ):
        raise RuntimeError("Taxi public sample hash verification failed")
    os.replace(temporary, destination)
    return {**aggregate.file_record(destination), "downloaded": True}


def run_once(
    plan: Mapping[str, Any],
    *,
    queue_status_path: Path,
    evidence_dir: Path,
    package_dir: Path,
    sample_path: Path,
    staged_run_id: str,
) -> dict[str, Any]:
    status_path = evidence_dir / "status_current.json"
    if not queue_status_path.is_file():
        result = {**_base(plan), "status": "waiting_for_taxi_queue"}
        write_json_atomic(status_path, result)
        return result
    taxi_status = json.loads(queue_status_path.read_text(encoding="utf-8"))
    if taxi_status.get("plan_sha256") != plan["_sha256"]:
        result = {
            **_base(plan),
            "status": "waiting_for_matching_taxi_plan",
            "observed_plan_sha256": taxi_status.get("plan_sha256"),
        }
        write_json_atomic(status_path, result)
        return result
    if taxi_status.get("status") != "all_seeds_terminal":
        result = {
            **_base(plan),
            "status": "waiting_for_taxi_seeds_terminal",
            "taxi_queue_status": taxi_status.get("status"),
        }
        write_json_atomic(status_path, result)
        return result

    collections = [ops.collect_run(str(run_id)) for run_id in plan["run_ids"]]
    sample = stage_public_sample(plan, sample_path)
    if package_dir.exists():
        verified = stage.verify_human_gate_package(package_dir)
        aggregation = {
            "status": "existing_package_verified",
            "package": str(package_dir),
            "candidate_sha256": verified.files["candidate_submission_withheld.csv"].sha256,
        }
    else:
        aggregation = aggregate.aggregate(
            plan_path=Path(str(plan["_path"])),
            collected_root=ops.LOCAL_CONTROL_ROOT / "collected",
            sample_path=sample_path,
            output_dir=package_dir,
        )
        stage.verify_human_gate_package(package_dir)
    staged_root = stage.ALLOWED_OUTPUT_ROOT / staged_run_id
    if staged_root.exists():
        staging = stage.verify_staged_run(staged_root)
    else:
        staging = stage.stage_candidate_run(package_dir, run_id=staged_run_id)
    result = {
        **_base(plan),
        "status": "verified_human_approval_pending",
        "collections": collections,
        "sample_submission": sample,
        "aggregation": aggregation,
        "staging": staging,
        "approved": False,
        "automatic_approval": False,
    }
    write_json_atomic(status_path, result)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=aggregate.DEFAULT_PLAN)
    parser.add_argument("--queue-status", type=Path, default=DEFAULT_QUEUE_STATUS)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--package-dir", type=Path, default=aggregate.DEFAULT_OUTPUT)
    parser.add_argument("--sample-submission", type=Path, default=aggregate.DEFAULT_SAMPLE)
    parser.add_argument("--staged-run-id", default=DEFAULT_STAGED_RUN_ID)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--deadline-hours", type=float, default=336.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.poll_seconds < 30 or args.deadline_hours <= 0:
        raise ValueError("Taxi post-run watcher timing contract is invalid")
    evidence_dir = args.evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    while True:
        try:
            plan = queue.validate_plan(args.plan)
            result = run_once(
                plan,
                queue_status_path=args.queue_status.resolve(),
                evidence_dir=evidence_dir,
                package_dir=args.package_dir.resolve(),
                sample_path=args.sample_submission.resolve(),
                staged_run_id=stage.validate_run_id(args.staged_run_id),
            )
        except Exception as exc:
            result = {
                "schema": "evomind.hpc88240.taxi_multiseed_postrun_failure.v1",
                "created_at": now_iso(),
                "status": "transient_or_contract_failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "process_signals_sent": 0,
                "other_processes_modified": False,
                "private_labels_used": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            }
            write_json_atomic(evidence_dir / "failure_current.json", result)
            if args.once:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 1
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if result["status"] == "verified_human_approval_pending":
                return 0
            if args.once:
                return 0
        if datetime.now().astimezone() >= deadline:
            return 4
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
