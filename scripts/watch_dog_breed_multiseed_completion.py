#!/usr/bin/env python3
"""Collect and Human-Gate-stage verified Dog Breed frozen-head multiseed results."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import aggregate_dog_breed_multiseed_candidate as aggregate  # noqa: E402
from scripts import mlebench_remote_ops as ops  # noqa: E402
from scripts import queue_hpc88240_dog_frozen_head_after_taxi as queue  # noqa: E402
from scripts import stage_mlebench_human_gate_candidate as stage  # noqa: E402
from scripts import watch_taxi_multiseed_completion as common_watch  # noqa: E402

DEFAULT_QUEUE_STATUS = queue.DEFAULT_EVIDENCE_DIR / "status_current.json"
DEFAULT_EVIDENCE_DIR = (
    PROJECT_ROOT / "workspace" / "hpc" / "job88240_dog_frozen_head_postrun"
)
DEFAULT_STAGED_RUN_ID = "hg_dog_breed_frozen_head_s4647_20260728"
UPSTREAM_GATE_FAILURES = {
    "diagnostic_gate_failed",
    "full_seed_gate_failed",
    "blocked_by_dependency_integrity_failure",
    "launch_claim_already_exists",
}


def _base(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "evomind.hpc88240.dog_breed_frozen_head_postrun.v1",
        "created_at": common_watch.now_iso(),
        "plan_path": plan["_path"],
        "plan_sha256": plan["_sha256"],
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def stage_public_sample(
    plan: Mapping[str, Any], destination: Path, *, evidence_dir: Path
) -> dict[str, Any]:
    record = plan.get("public_sample_submission") or {}
    expected_bytes = int(record.get("bytes", -1))
    expected_sha = str(record.get("sha256") or "")
    remote = ops.ensure_remote_path(str(record.get("remote_path") or ""))
    if not remote.startswith(ops.REMOTE_DATA_ROOT + "/") or not remote.endswith(
        "/prepared/public/sample_submission.csv"
    ):
        raise RuntimeError("Dog public sample remote path drifted")
    destination = Path(destination).resolve()
    allowed = stage.DEFAULT_PUBLIC_DATA_ROOT.resolve()
    try:
        destination.relative_to(allowed)
    except ValueError as exc:
        raise RuntimeError("Dog public sample destination escaped local data root") from exc
    if (
        destination.is_file()
        and destination.stat().st_size == expected_bytes
        and aggregate.common.sha256_file(destination) == expected_sha
    ):
        return {**aggregate.common.file_record(destination), "downloaded": False, "backup": None}

    backup_record = None
    if destination.is_file():
        backup_root = Path(evidence_dir).resolve() / "public_sample_backups"
        backup_root.mkdir(parents=True, exist_ok=True)
        old_sha = aggregate.common.sha256_file(destination)
        backup = backup_root / f"sample_submission.{old_sha}.csv"
        if not backup.exists():
            shutil.copyfile(destination, backup)
        if (
            backup.stat().st_size != destination.stat().st_size
            or aggregate.common.sha256_file(backup) != old_sha
        ):
            raise RuntimeError("Dog public sample backup verification failed")
        backup_record = aggregate.common.file_record(backup)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.part")
    if temporary.exists():
        temporary.unlink()
    client = ops._connect()
    try:
        with client.open_sftp() as sftp:
            item = sftp.lstat(remote)
            if int(item.st_size) != expected_bytes:
                raise RuntimeError("Dog public sample remote size drifted")
            sftp.get(remote, str(temporary))
    finally:
        client.close()
    if (
        temporary.stat().st_size != expected_bytes
        or aggregate.common.sha256_file(temporary) != expected_sha
    ):
        raise RuntimeError("Dog public sample hash verification failed")
    os.replace(temporary, destination)
    return {
        **aggregate.common.file_record(destination),
        "downloaded": True,
        "backup": backup_record,
    }


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
        result = {**_base(plan), "status": "waiting_for_dog_queue"}
        common_watch.write_json_atomic(status_path, result)
        return result
    current = json.loads(queue_status_path.read_text(encoding="utf-8"))
    if current.get("plan_sha256") != plan["_sha256"]:
        result = {
            **_base(plan),
            "status": "waiting_for_matching_dog_plan",
            "observed_plan_sha256": current.get("plan_sha256"),
        }
        common_watch.write_json_atomic(status_path, result)
        return result
    if current.get("status") in UPSTREAM_GATE_FAILURES:
        result = {
            **_base(plan),
            "status": "upstream_gate_failed",
            "dog_queue_status": current.get("status"),
            "candidate_ready_for_human_gate": False,
        }
        common_watch.write_json_atomic(status_path, result)
        return result
    if current.get("status") != "all_full_seeds_terminal":
        result = {
            **_base(plan),
            "status": "waiting_for_dog_seeds_terminal",
            "dog_queue_status": current.get("status"),
        }
        common_watch.write_json_atomic(status_path, result)
        return result

    expected_run_ids = [str(value) for value in plan["confirmation"]["run_ids"]]
    completed = [str(item.get("run_id")) for item in current.get("completed_seeds") or []]
    if completed != expected_run_ids:
        raise RuntimeError("Dog terminal seed inventory changed")
    collections = [ops.collect_run(run_id) for run_id in expected_run_ids]
    sample = stage_public_sample(plan, sample_path, evidence_dir=evidence_dir)
    if package_dir.exists():
        verified = stage.verify_human_gate_package(package_dir)
        aggregation = {
            "status": "existing_package_verified",
            "candidate_sha256": verified.files[
                "candidate_submission_withheld.csv"
            ].sha256,
        }
    else:
        aggregation = aggregate.aggregate(
            plan_path=Path(plan["_path"]),
            collected_root=ops.LOCAL_CONTROL_ROOT / "collected",
            sample_path=sample_path,
            output_dir=package_dir,
        )
        stage.verify_human_gate_package(package_dir)
    staged_root = stage.ALLOWED_OUTPUT_ROOT / staged_run_id
    staging = (
        stage.verify_staged_run(staged_root)
        if staged_root.exists()
        else stage.stage_candidate_run(package_dir, run_id=staged_run_id)
    )
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
    common_watch.write_json_atomic(status_path, result)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=queue.DEFAULT_PLAN)
    parser.add_argument("--queue-status", type=Path, default=DEFAULT_QUEUE_STATUS)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--package-dir", type=Path, default=aggregate.DEFAULT_OUTPUT)
    parser.add_argument("--sample-submission", type=Path, default=aggregate.DEFAULT_SAMPLE)
    parser.add_argument("--staged-run-id", default=DEFAULT_STAGED_RUN_ID)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--deadline-hours", type=float, default=720.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.poll_seconds < 30 or args.deadline_hours <= 0:
        raise ValueError("Dog post-run watcher timing contract is invalid")
    evidence = args.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    while True:
        try:
            plan = queue.validate_plan(args.plan)
            result = run_once(
                plan,
                queue_status_path=args.queue_status.resolve(),
                evidence_dir=evidence,
                package_dir=args.package_dir.resolve(),
                sample_path=args.sample_submission.resolve(),
                staged_run_id=stage.validate_run_id(args.staged_run_id),
            )
        except Exception as exc:
            result = {
                "schema": "evomind.hpc88240.dog_breed_frozen_head_postrun_failure.v1",
                "created_at": common_watch.now_iso(),
                "status": "transient_or_contract_failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "process_signals_sent": 0,
                "other_processes_modified": False,
                "private_labels_used": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            }
            common_watch.write_json_atomic(evidence / "failure_current.json", result)
            if args.once:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 1
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if result["status"] in {
                "verified_human_approval_pending",
                "upstream_gate_failed",
            }:
                return 0
            if args.once:
                return 0
        if datetime.now().astimezone() >= deadline:
            return 4
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
