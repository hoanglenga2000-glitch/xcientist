#!/usr/bin/env python3
"""Aggregate verified Taxi CPU seeds into a frozen Human Gate package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPETITION_ID = "new-york-city-taxi-fare-prediction"
DEFAULT_PLAN = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "taxi_cpu_multiseed_human_gate_plan_20260728.json"
)
DEFAULT_COLLECTIONS = (
    PROJECT_ROOT / "workspace" / "hpc" / "job89941_taxi_cpu_candidate_s43_v3" / "collection_current.json",
    PROJECT_ROOT / "workspace" / "hpc" / "job89941_taxi_cpu_confirmation_s44" / "collection_current.json",
    PROJECT_ROOT / "workspace" / "hpc" / "job89941_taxi_cpu_confirmation_s45" / "collection_current.json",
)
DEFAULT_SAMPLE = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "mlebench_official_data"
    / COMPETITION_ID
    / "prepared"
    / "public"
    / "sample_submission.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "workspace" / "human_gate" / "taxi_cpu_multiseed_s43_s44_s45_20260728"
)


class TaxiCpuAggregationError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TaxiCpuAggregationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    require(path.is_file() and not path.is_symlink(), f"Unsafe artifact: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    require(truth.shape == prediction.shape, "Taxi RMSE arrays are misaligned")
    require(np.isfinite(truth).all() and np.isfinite(prediction).all(), "Taxi RMSE arrays are non-finite")
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def load_seed_collection(
    path: Path,
    *,
    seed: int,
    expected_plan_sha256: str,
    expected_train_rows: int,
    expected_test_rows: int,
) -> dict[str, Any]:
    path = Path(path).resolve()
    value = read_json(path)
    result = value.get("result") if isinstance(value.get("result"), dict) else {}
    require(value.get("status") == "completed_and_verified", f"Taxi seed {seed} is not collected")
    require(value.get("plan_sha256") == expected_plan_sha256, f"Taxi seed {seed} plan differs")
    require(value.get("candidate_ready_for_multiseed_confirmation") is True, f"Taxi seed {seed} did not pass")
    require(
        result.get("seed") == seed
        and result.get("passed") is True
        and result.get("status") == "candidate_complete",
        f"Taxi seed {seed} result gate failed",
    )
    for key, expected in {
        "candidate_only": True,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "gpu_used": False,
    }.items():
        require(result.get(key) == expected, f"Taxi seed {seed} boundary changed: {key}")
    download = ((value.get("downloads") or {}).get("candidate_bundle") or {})
    bundle_path = Path(str(download.get("local_path") or "")).resolve()
    require(
        bundle_path.is_file()
        and bundle_path.stat().st_size == download.get("bytes")
        and sha256_file(bundle_path) == download.get("sha256"),
        f"Taxi seed {seed} bundle differs",
    )
    with np.load(bundle_path, allow_pickle=False) as archive:
        truth = np.asarray(archive["truth"], dtype=np.float64)
        oof = np.asarray(archive["oof_prediction"], dtype=np.float64)
        folds = np.asarray(archive["fold_assignment"], dtype=np.int16)
        test_key = np.asarray(archive["test_key"]).astype(str)
        test_prediction = np.asarray(archive["selected_test_prediction"], dtype=np.float64)
    require(truth.shape == (expected_train_rows,), f"Taxi seed {seed} train rows differ")
    require(oof.shape == truth.shape and folds.shape == truth.shape, f"Taxi seed {seed} OOF shape differs")
    require(test_key.shape == (expected_test_rows,), f"Taxi seed {seed} test rows differ")
    require(test_prediction.shape == test_key.shape, f"Taxi seed {seed} test predictions differ")
    require(len(set(test_key.tolist())) == len(test_key), f"Taxi seed {seed} test keys duplicate")
    score = rmse(truth, oof)
    independent = value.get("independent_verification") or {}
    require(abs(score - float(independent.get("independent_random_oof_rmse"))) <= 1e-6, f"Taxi seed {seed} OOF audit differs")
    return {
        "seed": seed,
        "collection_path": path,
        "collection": value,
        "result": result,
        "bundle_path": bundle_path,
        "truth": truth,
        "oof": oof,
        "folds": folds,
        "test_key": test_key,
        "test_prediction": test_prediction,
        "random_oof_rmse": score,
        "temporal_rmse": float(result["temporal_rmse"]),
        "geographic_rmse": float(result["geographic_rmse"]),
    }


def aggregate(
    *,
    plan_path: Path,
    collection_paths: list[Path],
    sample_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    sample_path = Path(sample_path).resolve()
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "Taxi CPU Human Gate package already exists")
    plan = read_json(plan_path)
    require(plan.get("schema") == "evomind.mlebench.taxi_cpu_multiseed_human_gate_plan.v1", "Taxi CPU aggregate plan changed")
    seeds = [int(value) for value in plan.get("seeds") or []]
    plan_hashes = [str(value) for value in plan.get("seed_plan_sha256s") or []]
    require(seeds == [43, 44, 45] and len(plan_hashes) == 3, "Taxi CPU seed contract changed")
    require(len(collection_paths) == 3, "Taxi CPU needs three collections")
    source = plan.get("aggregator_source") or {}
    require(
        Path(str(source.get("path") or "")).resolve() == Path(__file__).resolve()
        and Path(__file__).stat().st_size == source.get("bytes")
        and sha256_file(Path(__file__)) == source.get("sha256"),
        "Taxi CPU aggregator source changed",
    )
    sample_record = plan.get("sample_submission") or {}
    require(
        sample_path.is_file()
        and sample_path.stat().st_size == sample_record.get("bytes")
        and sha256_file(sample_path) == sample_record.get("sha256"),
        "Taxi CPU sample submission changed",
    )
    expected_train_rows = int(plan["expected_train_rows"])
    expected_test_rows = int(plan["expected_test_rows"])
    items = [
        load_seed_collection(
            path,
            seed=seed,
            expected_plan_sha256=plan_sha,
            expected_train_rows=expected_train_rows,
            expected_test_rows=expected_test_rows,
        )
        for path, seed, plan_sha in zip(collection_paths, seeds, plan_hashes, strict=True)
    ]
    reference = items[0]
    for item in items[1:]:
        require(np.array_equal(item["truth"], reference["truth"]), "Taxi CPU truth alignment differs")
        require(np.array_equal(item["folds"], reference["folds"]), "Taxi CPU fold alignment differs")
        require(np.array_equal(item["test_key"], reference["test_key"]), "Taxi CPU test key alignment differs")
    ensemble_oof = np.mean(np.vstack([item["oof"] for item in items]), axis=0)
    ensemble_test = np.mean(np.vstack([item["test_prediction"] for item in items]), axis=0)
    ensemble_rmse = rmse(reference["truth"], ensemble_oof)
    thresholds = plan.get("promotion_gate") or {}
    seed_rmse = [item["random_oof_rmse"] for item in items]
    temporal = [item["temporal_rmse"] for item in items]
    geographic = [item["geographic_rmse"] for item in items]
    checks = {
        "every_seed_random_oof": max(seed_rmse) <= float(thresholds["every_seed_random_oof_rmse_maximum"]),
        "every_seed_temporal": max(temporal) <= float(thresholds["every_seed_temporal_rmse_maximum"]),
        "every_seed_geographic": max(geographic) <= float(thresholds["every_seed_geographic_rmse_maximum"]),
        "mean_seed_random_oof": float(np.mean(seed_rmse)) <= float(thresholds["mean_random_oof_rmse_maximum"]),
        "ensemble_random_oof": ensemble_rmse <= float(thresholds["ensemble_random_oof_rmse_maximum"]),
    }
    require(all(checks.values()), "Taxi CPU multiseed gate did not pass")
    sample = pd.read_csv(sample_path)
    require(list(sample.columns) == ["key", "fare_amount"], "Taxi sample schema changed")
    require(sample["key"].astype(str).to_numpy().tolist() == reference["test_key"].tolist(), "Taxi sample keys differ")
    output_dir.mkdir(parents=True)
    candidate_path = output_dir / "candidate_submission_withheld.csv"
    candidate = sample.copy()
    candidate["fare_amount"] = ensemble_test
    candidate.to_csv(candidate_path, index=False)
    frozen_plan_path = output_dir / "frozen_plan.json"
    shutil.copyfile(plan_path, frozen_plan_path)
    result = {
        "schema": "evomind.mlebench.taxi_cpu_multiseed_confirmation.v1",
        "created_at": now_iso(),
        "competition_id": COMPETITION_ID,
        "status": "confirmation_passed_human_gate_pending",
        "candidate_ready_for_human_gate": True,
        "metrics": {
            "seed_random_oof_rmse": seed_rmse,
            "mean_seed_random_oof_rmse": float(np.mean(seed_rmse)),
            "maximum_seed_random_oof_rmse": max(seed_rmse),
            "ensemble_random_oof_rmse": ensemble_rmse,
            "ensemble_oof_rmse": ensemble_rmse,
            "seed_temporal_rmse": temporal,
            "seed_geographic_rmse": geographic,
        },
        "confirmation_gate": {
            "passed": True,
            "metric": "rmse",
            "direction": "minimize",
            "checks": checks,
            "thresholds": thresholds,
        },
        "seed_records": [
            {
                "model_seed": item["seed"],
                "collection": file_record(item["collection_path"]),
                "candidate_bundle": file_record(item["bundle_path"]),
                "random_oof_rmse": item["random_oof_rmse"],
                "temporal_rmse": item["temporal_rmse"],
                "geographic_rmse": item["geographic_rmse"],
            }
            for item in items
        ],
        "submission_withheld": file_record(candidate_path),
        "sample_submission": file_record(sample_path),
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "human_gate_preserved": True,
        "claim_boundary": "PUBLIC_ONLY multiseed OOF evidence; no official score or medal claimed.",
    }
    result_path = output_dir / "taxi_cpu_multiseed_confirmation_result.json"
    write_json_atomic(result_path, result)
    independent = {
        "schema": "evomind.mlebench.taxi_cpu_multiseed_independent_verification.v1",
        "created_at": now_iso(),
        "status": "verification_passed",
        "ok": True,
        "candidate_ready_for_human_gate": True,
        "errors": [],
        "plan_sha256": sha256_file(frozen_plan_path),
        "result_sha256": sha256_file(result_path),
        "recomputed_seed_random_oof_rmse": seed_rmse,
        "recomputed_ensemble_random_oof_rmse": ensemble_rmse,
        "recomputed_ensemble_oof_rmse": ensemble_rmse,
        "oof_rows": expected_train_rows,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "human_gate_preserved": True,
    }
    independent_path = output_dir / "independent_verification.json"
    write_json_atomic(independent_path, independent)
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "# Taxi CPU multiseed Human Gate package\n\nCandidate-only; no grader or Kaggle submission executed.\n",
        encoding="utf-8",
    )
    core_files = [
        file_record(candidate_path),
        file_record(frozen_plan_path),
        file_record(independent_path),
        file_record(result_path),
        file_record(readme_path),
    ]
    manifest = {
        "schema": "evomind.human_gate.candidate_package.v1",
        "created_at": now_iso(),
        "status": "ready_for_human_review_not_submitted",
        "competition_id": COMPETITION_ID,
        "public_oof_metrics": result["metrics"],
        "candidate_ready_for_human_gate": True,
        "files": core_files,
        "automatic_submission": False,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
    }
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    verification = {
        "schema": "evomind.human_gate.package_verification.v1",
        "created_at": now_iso(),
        "status": "verified",
        "files": core_files + [file_record(manifest_path)],
        "candidate_csv_present": True,
        "independent_verification_passed": True,
        "automatic_submission": False,
    }
    verification_path = output_dir / "package_verification.json"
    write_json_atomic(verification_path, verification)
    return {
        "status": "ready_for_human_review_not_submitted",
        "package": str(output_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "candidate_sha256": sha256_file(candidate_path),
        "metrics": result["metrics"],
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--collection", type=Path, action="append")
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = aggregate(
        plan_path=args.plan,
        collection_paths=list(args.collection or DEFAULT_COLLECTIONS),
        sample_path=args.sample_submission,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
