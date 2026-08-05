#!/usr/bin/env python3
"""Watch RANZCR terminal evidence and stage a verified Human Gate candidate."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from scripts import aggregate_taxi_multiseed_candidate as common  # noqa: E402
from scripts import mlebench_remote_ops as ops  # noqa: E402
from scripts import stage_mlebench_human_gate_candidate as stage  # noqa: E402
from scripts import stage_ranzcr_crossrun_candidate as ranzcr  # noqa: E402

REMOTE_BASE = ops.ALLOWED_GPU_REMOTE_ROOT + "/evomind_mle22/job88240_ranzcr_highres_s42_20260727"
REMOTE_STATUS = REMOTE_BASE + "/full_ranzcr_status.json"
REMOTE_CANDIDATE = REMOTE_BASE + "/crossrun_candidate"
REMOTE_SAMPLE = ops.REMOTE_DATA_ROOT + "/ranzcr-clip-catheter-line-classification/prepared/public/sample_submission.csv"
DEFAULT_EVIDENCE = PROJECT_ROOT / "workspace" / "hpc" / "job88240_ranzcr_human_gate_postrun"
DEFAULT_STAGED_RUN_ID = "hg_ranzcr_crossrun_s42_20260728"


def now_iso(): return datetime.now().astimezone().isoformat()


def remote_status():
    client = ops._connect()
    try:
        with client.open_sftp() as sftp:
            try: item = sftp.lstat(REMOTE_STATUS)
            except OSError: return None
            if not stat.S_ISREG(item.st_mode) or stat.S_ISLNK(item.st_mode) or item.st_size > 1024 * 1024: raise RuntimeError("RANZCR status is unsafe")
            with sftp.open(REMOTE_STATUS, "rb") as handle: return json.loads(handle.read().decode("utf-8"))
    finally: client.close()


def collect(destination: Path, source_plan: Path):
    destination = destination.resolve(); destination.mkdir(parents=True, exist_ok=True)
    plan = json.loads(source_plan.read_text(encoding="utf-8")); expected = plan["data_contract"]["core"]["sample_submission.csv"]
    files = {"aggregation_result.json": REMOTE_CANDIDATE + "/aggregation_result.json", "independent_verification.json": REMOTE_CANDIDATE + "/independent_verification.json", "ranzcr_crossrun_oof_and_test.npz": REMOTE_CANDIDATE + "/ranzcr_crossrun_oof_and_test.npz", "submission_withheld.csv": REMOTE_CANDIDATE + "/submission_withheld.csv", "sample_submission.csv": REMOTE_SAMPLE}
    records = []; client = ops._connect()
    try:
        with client.open_sftp() as sftp:
            for name, remote in files.items():
                remote = ops.ensure_remote_path(remote); item = sftp.lstat(remote)
                if not stat.S_ISREG(item.st_mode) or stat.S_ISLNK(item.st_mode): raise RuntimeError(f"RANZCR artifact unsafe: {name}")
                target = destination / name; part = target.with_suffix(target.suffix + ".part")
                if part.exists(): part.unlink()
                sftp.get(remote, str(part)); os.replace(part, target); records.append({"name": name, "remote": remote, **common.file_record(target)})
    finally: client.close()
    sample = destination / "sample_submission.csv"
    if sample.stat().st_size != int(expected["bytes"]) or common.sha256_file(sample) != expected["sha256"]: raise RuntimeError("RANZCR sample hash drifted")
    manifest = {"schema": "evomind.ranzcr.human_gate_collection.v1", "created_at": now_iso(), "status": "completed_and_verified", "files": records, "remote_writes_performed": False, "process_signals_sent": 0, "official_grader_executed": False, "kaggle_submission_executed": False}
    common.write_json_atomic(destination / "collection_manifest.json", manifest); return manifest


def run_once(*, evidence: Path, source_plan: Path, package_dir: Path, staged_run_id: str):
    base = {"schema": "evomind.hpc88240.ranzcr_human_gate_postrun.v1", "created_at": now_iso(), "process_signals_sent": 0, "other_processes_modified": False, "private_labels_used": False, "official_grader_executed": False, "kaggle_submission_executed": False}
    remote = remote_status()
    if remote is None or remote.get("status") not in {"verification_passed", "verification_complete_gate_failed"}: result = {**base, "status": "waiting_for_ranzcr_terminal", "remote_status": None if remote is None else remote.get("status")}
    elif remote.get("status") == "verification_complete_gate_failed": result = {**base, "status": "ranzcr_confirmation_gate_failed", "remote_evidence": remote, "candidate_ready": False}
    else:
        collection = collect(evidence / "collected", source_plan)
        if package_dir.exists():
            verified = stage.verify_human_gate_package(package_dir); package = {"status": "existing_package_verified", "candidate_sha256": verified.files["candidate_submission_withheld.csv"].sha256}
        else:
            package = ranzcr.build_package(collected_dir=evidence / "collected", source_plan_path=source_plan, output_dir=package_dir); stage.verify_human_gate_package(package_dir)
        staged_root = stage.ALLOWED_OUTPUT_ROOT / staged_run_id; staging = stage.verify_staged_run(staged_root) if staged_root.exists() else stage.stage_candidate_run(package_dir, run_id=staged_run_id)
        result = {**base, "status": "verified_human_approval_pending", "remote_evidence": remote, "collection": collection, "package": package, "staging": staging, "approved": False, "automatic_approval": False}
    common.write_json_atomic(evidence / "status_current.json", result); return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE); parser.add_argument("--source-plan", type=Path, default=ranzcr.DEFAULT_SOURCE_PLAN); parser.add_argument("--package-dir", type=Path, default=ranzcr.DEFAULT_OUTPUT); parser.add_argument("--staged-run-id", default=DEFAULT_STAGED_RUN_ID); parser.add_argument("--poll-seconds", type=int, default=120); parser.add_argument("--deadline-hours", type=float, default=672.0); parser.add_argument("--once", action="store_true"); args = parser.parse_args()
    if args.poll_seconds < 30 or args.deadline_hours <= 0: raise ValueError("RANZCR watcher timing invalid")
    evidence = args.evidence.resolve(); evidence.mkdir(parents=True, exist_ok=True); deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    while True:
        try: result = run_once(evidence=evidence, source_plan=args.source_plan.resolve(), package_dir=args.package_dir.resolve(), staged_run_id=stage.validate_run_id(args.staged_run_id))
        except Exception as exc:
            result = {"schema": "evomind.hpc88240.ranzcr_human_gate_postrun_failure.v1", "created_at": now_iso(), "status": "transient_or_contract_failure", "error_type": type(exc).__name__, "error": str(exc), "process_signals_sent": 0, "other_processes_modified": False, "private_labels_used": False, "official_grader_executed": False, "kaggle_submission_executed": False}; common.write_json_atomic(evidence / "failure_current.json", result)
            if args.once: print(json.dumps(result, ensure_ascii=False, indent=2)); return 1
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if result["status"] in {"verified_human_approval_pending", "ranzcr_confirmation_gate_failed"}: return 0
            if args.once: return 0
        if datetime.now().astimezone() >= deadline: return 4
        time.sleep(args.poll_seconds)


if __name__ == "__main__": raise SystemExit(main())
