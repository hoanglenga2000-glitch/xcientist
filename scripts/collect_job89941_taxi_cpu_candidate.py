#!/usr/bin/env python3
"""Collect and independently verify the job89941 Taxi CPU candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deploy_job89941_taxi_cpu_candidate import (  # noqa: E402
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_PLAN,
    collect_status,
    connect_job89941,
    confined_remote,
    write_json_atomic,
)


class TaxiCpuCollectionError(RuntimeError):
    """Raised when a remote candidate fails collection integrity."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_collection_plan(plan_path: Path) -> dict[str, Any]:
    """Validate immutable run identity without requiring a mutable source checkout."""
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema") != "evomind.hpc.job89941_taxi_cpu_candidate_plan.v1":
        raise TaxiCpuCollectionError("Taxi collection plan schema changed")
    if plan.get("status") != "approved_candidate_only" or plan.get("job_id") != 89941:
        raise TaxiCpuCollectionError("Taxi collection plan identity changed")
    boundary = plan.get("boundary") or {}
    for key, expected in {
        "visibility_mode": "PUBLIC_ONLY",
        "candidate_only": True,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "gpu_used": False,
        "process_signals_allowed": False,
        "other_processes_may_be_modified": False,
    }.items():
        if boundary.get(key) != expected:
            raise TaxiCpuCollectionError(f"Taxi collection boundary changed: {key}")
    for name in ("base_cache", "route_stat_cache"):
        record = plan.get(name) or {}
        local = Path(str(record.get("local_evidence_path") or "")).resolve()
        if not local.is_file() or local.stat().st_size != record.get("bytes") or sha256_file(local) != record.get("sha256"):
            raise TaxiCpuCollectionError(f"Taxi collection input evidence changed: {name}")
    plan["_path"] = str(plan_path)
    plan["_sha256"] = sha256_file(plan_path)
    return plan


def download_regular(sftp: Any, remote: str, local: Path) -> dict[str, Any]:
    remote = confined_remote(remote)
    item = sftp.lstat(remote)
    if stat.S_ISLNK(item.st_mode) or not stat.S_ISREG(item.st_mode):
        raise TaxiCpuCollectionError(f"remote artifact is unsafe: {remote}")
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_suffix(local.suffix + f".{os.getpid()}.tmp")
    digest = hashlib.sha256()
    total = 0
    with sftp.open(remote, "rb") as source, temporary.open("wb") as target:
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            target.write(block)
            digest.update(block)
            total += len(block)
    os.replace(temporary, local)
    return {
        "remote_path": remote,
        "local_path": str(local.resolve()),
        "bytes": total,
        "sha256": digest.hexdigest(),
    }


def remote_regular_record(sftp: Any, remote: str) -> dict[str, Any]:
    remote = confined_remote(remote)
    item = sftp.lstat(remote)
    if stat.S_ISLNK(item.st_mode) or not stat.S_ISREG(item.st_mode):
        raise TaxiCpuCollectionError(f"remote source is unsafe: {remote}")
    digest = hashlib.sha256()
    total = 0
    with sftp.open(remote, "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
    return {"path": remote, "bytes": total, "sha256": digest.hexdigest()}


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if truth.shape != prediction.shape or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise TaxiCpuCollectionError("Taxi OOF arrays are invalid")
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def verify_bundle(path: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    stress_verification: dict[str, float] = {}
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "truth",
            "oof_prediction",
            "fold_assignment",
            "test_key",
            "fold_test_prediction",
            "refit_test_prediction",
            "selected_test_prediction",
        }
        if required - set(archive.files):
            raise TaxiCpuCollectionError("Taxi candidate bundle arrays are incomplete")
        truth = np.asarray(archive["truth"], dtype=np.float64)
        oof = np.asarray(archive["oof_prediction"], dtype=np.float64)
        folds = np.asarray(archive["fold_assignment"], dtype=np.int16)
        test_key = np.asarray(archive["test_key"]).astype(str)
        fold_test = np.asarray(archive["fold_test_prediction"], dtype=np.float64)
        refit_test = np.asarray(archive["refit_test_prediction"], dtype=np.float64)
        selected_test = np.asarray(archive["selected_test_prediction"], dtype=np.float64)
        for context in ("temporal", "geographic"):
            keys = {
                f"{context}_valid_index",
                f"{context}_truth",
                f"{context}_prediction",
            }
            if keys.issubset(archive.files):
                index = np.asarray(archive[f"{context}_valid_index"], dtype=np.int64)
                stress_truth = np.asarray(archive[f"{context}_truth"], dtype=np.float64)
                stress_prediction = np.asarray(
                    archive[f"{context}_prediction"], dtype=np.float64
                )
                if len(index) != len(stress_truth) or len(set(index.tolist())) != len(index):
                    raise TaxiCpuCollectionError(f"Taxi {context} stress alignment changed")
                stress_verification[f"independent_{context}_rmse"] = rmse(
                    stress_truth, stress_prediction
                )
    if truth.shape != (5_000_000,) or folds.shape != truth.shape:
        raise TaxiCpuCollectionError("Taxi OOF cardinality changed")
    if set(np.unique(folds).tolist()) != {0, 1, 2}:
        raise TaxiCpuCollectionError("Taxi fold assignment changed")
    if test_key.shape != (9_914,) or len(set(test_key.tolist())) != len(test_key):
        raise TaxiCpuCollectionError("Taxi test key contract changed")
    if fold_test.shape != (9_914, 3) or refit_test.shape != (9_914,) or selected_test.shape != (9_914,):
        raise TaxiCpuCollectionError("Taxi test prediction shape changed")
    for values in (fold_test, refit_test, selected_test):
        if not np.isfinite(values).all():
            raise TaxiCpuCollectionError("Taxi test prediction is non-finite")
    independent_rmse = rmse(truth, oof)
    if abs(independent_rmse - float(result.get("random_oof_rmse"))) > 1e-6:
        raise TaxiCpuCollectionError("Taxi independent OOF RMSE differs")
    for context in ("temporal", "geographic"):
        key = f"independent_{context}_rmse"
        if key in stress_verification and abs(
            stress_verification[key] - float(result.get(f"{context}_rmse"))
        ) > 1e-6:
            raise TaxiCpuCollectionError(f"Taxi independent {context} RMSE differs")
    return {
        "train_rows": len(truth),
        "test_rows": len(test_key),
        "folds": sorted(np.unique(folds).astype(int).tolist()),
        "independent_random_oof_rmse": independent_rmse,
        "oof_finite": True,
        "test_predictions_finite": True,
        "test_keys_unique": True,
        **stress_verification,
    }


def validate_result(result: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    if result.get("schema") != "evomind.mlebench.taxi_cpu_lightgbm_candidate.v1":
        raise TaxiCpuCollectionError("Taxi result schema changed")
    if result.get("status") not in {
        "candidate_complete",
        "verification_complete_gate_failed",
    }:
        raise TaxiCpuCollectionError("Taxi result status is not terminal")
    for key, expected in {
        "candidate_only": True,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "gpu_used": False,
        "visibility_mode": "PUBLIC_ONLY",
    }.items():
        if result.get(key) != expected:
            raise TaxiCpuCollectionError(f"Taxi result boundary changed: {key}")
    if result.get("seed") != plan["runtime"]["seed"] or result.get("threads") != 60:
        raise TaxiCpuCollectionError("Taxi result runtime changed")
    for name, plan_key in (("cache_manifest", "base_cache"), ("route_stat_manifest", "route_stat_cache")):
        if (result.get(name) or {}).get("sha256") != plan[plan_key]["sha256"]:
            raise TaxiCpuCollectionError(f"Taxi result input hash changed: {name}")


def collect_validated_plan(plan: Mapping[str, Any], evidence_dir: Path) -> dict[str, Any]:
    client = connect_job89941()
    try:
        observed = collect_status(plan, evidence_dir)
        process = observed.get("process") or {}
        result = observed.get("result")
        if not isinstance(result, dict):
            report = {
                "schema": "evomind.hpc.job89941_taxi_cpu_candidate_collection.v1",
                "created_at": now_iso(),
                "status": "waiting_for_terminal_result" if process.get("running") else "stopped_without_result",
                "process": process,
                "plan_sha256": plan["_sha256"],
                "process_signals_sent": 0,
                "other_processes_modified": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            }
            write_json_atomic(evidence_dir / "collection_current.json", report)
            return report
        validate_result(result, plan)
        remote_result = str(PurePosixPath(str(plan["remote_output_dir"])) / "result.json")
        bundle_record = result.get("candidate_bundle") or {}
        remote_bundle = confined_remote(str(bundle_record.get("path") or ""))
        output_root = evidence_dir / "collected" / str(plan["run_id"])
        with client.open_sftp() as sftp:
            remote_source = remote_regular_record(sftp, str(plan["remote_source"]))
            expected_source = plan.get("candidate_source") or {}
            if (
                remote_source["bytes"] != expected_source.get("bytes")
                or remote_source["sha256"] != expected_source.get("sha256")
            ):
                raise TaxiCpuCollectionError("Taxi frozen remote source differs")
            result_download = download_regular(sftp, remote_result, output_root / "result.json")
            bundle_download = download_regular(sftp, remote_bundle, output_root / "taxi_cpu_candidate.npz")
        if bundle_download["bytes"] != bundle_record.get("bytes") or bundle_download["sha256"] != bundle_record.get("sha256"):
            raise TaxiCpuCollectionError("Taxi downloaded bundle differs from result")
        independent = verify_bundle(Path(bundle_download["local_path"]), result)
        report = {
            "schema": "evomind.hpc.job89941_taxi_cpu_candidate_collection.v1",
            "created_at": now_iso(),
            "status": "completed_and_verified",
            "plan_path": plan["_path"],
            "plan_sha256": plan["_sha256"],
            "result": result,
            "downloads": {"result": result_download, "candidate_bundle": bundle_download},
            "remote_frozen_source": remote_source,
            "independent_verification": independent,
            "candidate_ready_for_multiseed_confirmation": result.get("passed") is True,
            "official_medal_claimed": False,
            "process_signals_sent": 0,
            "other_processes_modified": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        }
        write_json_atomic(evidence_dir / "collection_current.json", report)
        return report
    finally:
        client.close()


def collect(plan_path: Path, evidence_dir: Path) -> dict[str, Any]:
    return collect_validated_plan(validate_collection_plan(plan_path), evidence_dir)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = collect(args.plan, args.evidence_dir.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
