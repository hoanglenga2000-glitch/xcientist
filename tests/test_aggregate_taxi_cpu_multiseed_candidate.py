from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate_taxi_cpu_multiseed_candidate import (
    TaxiCpuAggregationError,
    aggregate,
)


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_collection(root: Path, seed: int, plan_sha: str, prediction: np.ndarray) -> Path:
    bundle = root / f"seed{seed}.npz"
    truth = np.array([10.0, 12.0, 20.0, 22.0], dtype=np.float32)
    np.savez_compressed(
        bundle,
        truth=truth,
        oof_prediction=prediction.astype(np.float32),
        fold_assignment=np.array([0, 1, 2, 0], dtype=np.int16),
        test_key=np.array(["a", "b"]),
        selected_test_prediction=np.array([11.0 + seed / 100, 21.0 + seed / 100]),
    )
    score = float(np.sqrt(np.mean(np.square(truth - prediction))))
    collection = {
        "status": "completed_and_verified",
        "plan_sha256": plan_sha,
        "candidate_ready_for_multiseed_confirmation": True,
        "result": {
            "seed": seed,
            "status": "candidate_complete",
            "passed": True,
            "candidate_only": True,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "gpu_used": False,
            "random_oof_rmse": score,
            "temporal_rmse": 2.7,
            "geographic_rmse": 2.8,
        },
        "downloads": {
            "candidate_bundle": {
                "local_path": str(bundle.resolve()),
                "bytes": bundle.stat().st_size,
                "sha256": sha(bundle),
            }
        },
        "independent_verification": {"independent_random_oof_rmse": score},
    }
    path = root / f"collection{seed}.json"
    path.write_text(json.dumps(collection), encoding="utf-8")
    return path


def make_fixture(tmp_path: Path) -> tuple[Path, list[Path], Path]:
    plan_hashes = [str(seed) * 64 for seed in (4, 5, 6)]
    collections = [
        write_collection(
            tmp_path,
            seed,
            plan_sha,
            np.array([10.1, 11.9, 20.1, 21.9]),
        )
        for seed, plan_sha in zip((43, 44, 45), plan_hashes, strict=True)
    ]
    sample = tmp_path / "sample.csv"
    pd.DataFrame({"key": ["a", "b"], "fare_amount": [0.0, 0.0]}).to_csv(sample, index=False)
    source = ROOT / "scripts" / "aggregate_taxi_cpu_multiseed_candidate.py"
    plan = {
        "schema": "evomind.mlebench.taxi_cpu_multiseed_human_gate_plan.v1",
        "seeds": [43, 44, 45],
        "seed_plan_sha256s": plan_hashes,
        "expected_train_rows": 4,
        "expected_test_rows": 2,
        "aggregator_source": {
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": sha(source),
        },
        "sample_submission": {
            "bytes": sample.stat().st_size,
            "sha256": sha(sample),
        },
        "promotion_gate": {
            "every_seed_random_oof_rmse_maximum": 2.85,
            "every_seed_temporal_rmse_maximum": 3.10,
            "every_seed_geographic_rmse_maximum": 3.35,
            "mean_random_oof_rmse_maximum": 2.82,
            "ensemble_random_oof_rmse_maximum": 2.82,
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path, collections, sample


def test_aggregate_builds_candidate_only_human_gate_package(tmp_path: Path) -> None:
    plan, collections, sample = make_fixture(tmp_path)
    output = tmp_path / "package"
    report = aggregate(
        plan_path=plan,
        collection_paths=collections,
        sample_path=sample,
        output_dir=output,
    )
    assert report["status"] == "ready_for_human_review_not_submitted"
    assert report["official_grader_executed"] is False
    assert (output / "candidate_submission_withheld.csv").is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "package_verification.json").is_file()


def test_aggregate_rejects_seed_plan_drift(tmp_path: Path) -> None:
    plan, collections, sample = make_fixture(tmp_path)
    value = json.loads(collections[1].read_text())
    value["plan_sha256"] = "f" * 64
    collections[1].write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(TaxiCpuAggregationError, match="plan differs"):
        aggregate(
            plan_path=plan,
            collection_paths=collections,
            sample_path=sample,
            output_dir=tmp_path / "package",
        )
