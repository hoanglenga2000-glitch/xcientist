from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import aggregate_taxi_multiseed_candidate as aggregate
from scripts import stage_mlebench_human_gate_candidate as stage


def _sha(path: Path) -> str:
    return aggregate.sha256_file(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_collected_run(root: Path, run_id: str, seed: int, offset: float) -> None:
    competition = root / run_id / aggregate.COMPETITION_ID
    attempts = competition / "attempts" / "attempt_001"
    attempts.mkdir(parents=True)
    target = [10.0, 20.0, 30.0, 40.0]
    prediction = [value + offset for value in target]
    oof = pd.DataFrame(
        {
            "key": ["a", "b", "c", "d"],
            "source_row": [0, 1, 2, 3],
            "fare_amount": target,
            "duplicate_group": ["g0", "g1", "g2", "g3"],
            "fold": [0, 1, 2, 0],
            "oof_prediction": prediction,
        }
    )
    oof_path = competition / "taxi_oof_manifest.csv"
    oof.to_csv(oof_path, index=False)
    folds_path = competition / "taxi_oof_fold_records.json"
    _write_json(folds_path, {"schema": "evomind.mlebench_lite.taxi_oof_folds.v1", "seed": seed})
    submission = pd.DataFrame(
        {"key": ["x", "y"], "fare_amount": [11.0 + offset, 22.0 + offset]}
    )
    submission_path = attempts / "submission.csv"
    submission.to_csv(submission_path, index=False)
    rmse = abs(offset)
    result = {
        "competition_id": aggregate.COMPETITION_ID,
        "status": "candidate_ready",
        "metric": "rmse",
        "direction": "minimize",
        "cv_score": rmse,
        "valid_submission": True,
        "submission_sha256": _sha(submission_path),
        "official_grader_executed": False,
        "promotion_gate": {
            "passed": True,
            "evidence": {
                "temporal_stress_rmse": 2.7,
                "geographic_stress_rmse": 2.8,
            },
        },
        "budget": {
            "seed": seed,
            "precomputed_public_cache": {"verified": True},
            "precomputed_route_stat_cache": {"verified": True},
            "feature_build_on_gpu_run": False,
            "route_stat_build_on_gpu_run": False,
        },
        "evidence_contract": {"private_labels_used": False},
    }
    result_path = competition / "result.json"
    _write_json(result_path, result)
    files = []
    for path in (oof_path, folds_path, submission_path, result_path):
        files.append(
            {
                "local": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    _write_json(
        root / run_id / "collection_manifest.json",
        {"passed": True, "run_id": run_id, "files": files},
    )


def test_aggregate_taxi_builds_stageable_regression_human_gate_package(tmp_path: Path):
    plan = json.loads(aggregate.DEFAULT_PLAN.read_text(encoding="utf-8"))
    sample = tmp_path / "public" / "sample_submission.csv"
    sample.parent.mkdir(parents=True)
    pd.DataFrame({"key": ["x", "y"], "fare_amount": [11.35, 11.35]}).to_csv(
        sample, index=False
    )
    plan["public_sample_submission"] = {
        "path": str(sample),
        "bytes": sample.stat().st_size,
        "sha256": _sha(sample),
    }
    plan_path = tmp_path / "taxi_plan.json"
    _write_json(plan_path, plan)
    collected = tmp_path / "collected"
    for seed, run_id, offset in zip(
        plan["seeds"], plan["run_ids"], [1.0, -1.0, 0.5], strict=True
    ):
        _build_collected_run(collected, run_id, seed, offset)
    package = tmp_path / "human_gate" / "taxi-package"

    report = aggregate.aggregate(
        plan_path=plan_path,
        collected_root=collected,
        sample_path=sample,
        output_dir=package,
        chunksize=2,
    )

    assert report["status"] == "ready_for_human_review_not_submitted"
    assert report["metrics"]["ensemble_oof_rmse"] < 1.0
    public_root = tmp_path / "public_root"
    staged_sample = (
        public_root
        / aggregate.COMPETITION_ID
        / "prepared"
        / "public"
        / "sample_submission.csv"
    )
    staged_sample.parent.mkdir(parents=True)
    staged_sample.write_bytes(sample.read_bytes())
    verified = stage.verify_human_gate_package(
        package,
        public_data_root=public_root,
        allowed_package_root=tmp_path / "human_gate",
    )
    assert verified.metric == "rmse"
    assert verified.direction == "minimize"
    assert verified.model_seeds == (43, 44, 45)
