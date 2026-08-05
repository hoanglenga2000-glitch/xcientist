from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import aggregate_siim_multiseed_candidate as aggregate
from scripts import aggregate_taxi_multiseed_candidate as common


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_collection(root: Path, run_id: str, seed: int, sample: pd.DataFrame) -> None:
    competition = root / run_id / aggregate.COMPETITION_ID
    attempt = competition / "attempts" / "attempt_001"
    attempt.mkdir(parents=True)
    train_rows = 28_984
    index = np.arange(train_rows)
    target = np.zeros(train_rows, dtype=np.int8)
    target[:513] = 1
    probability = np.where(target == 1, 0.78, 0.03).astype(np.float64)
    probability += ((index * (seed + 3)) % 101) / 20_000.0
    test_probability = 0.02 + ((np.arange(4_142) * (seed + 5)) % 997) / 1_100.0
    test_probability = np.clip(test_probability, 1e-5, 1 - 1e-5)
    train_id = np.asarray([f"tr-{value:05d}" for value in index])
    patient_id = np.asarray([f"patient-{value % 2056:04d}" for value in index])
    leakage_group = np.asarray([f"group-{value % 2056:04d}" for value in index])
    fold = (index % 5).astype(np.int16)
    bundle = attempt / "siim_fold_ensemble.npz"
    np.savez_compressed(
        bundle,
        train_id=train_id,
        patient_id=patient_id,
        leakage_group=leakage_group,
        target=target,
        fold=fold,
        crossfit_probability=probability,
        pure_image_oof=np.clip(probability * 0.91 + 0.01, 0, 1),
        lesion_focus_oof=np.clip(probability * 0.93 + 0.008, 0, 1),
        image_metadata_fusion_oof=np.clip(probability * 0.97 + 0.004, 0, 1),
        metadata_catboost_oof=np.clip(probability * 0.75 + 0.02, 0, 1),
        seed_final_fold_percentile_rank_oof=probability,
        pure_image_percentile_rank_oof=np.clip(probability * 0.91 + 0.01, 0, 1),
        image_metadata_fusion_percentile_rank_oof=np.clip(
            probability * 0.97 + 0.004, 0, 1
        ),
        test_id=sample["image_name"].astype(str).to_numpy(dtype=np.str_),
        test_probability=test_probability,
        seed_final_fold_percentile_rank_test=test_probability,
        pure_image_percentile_rank_test=np.clip(
            test_probability * 0.91 + 0.01, 0, 1
        ),
        image_metadata_fusion_percentile_rank_test=np.clip(
            test_probability * 0.97 + 0.004, 0, 1
        ),
    )
    oof = attempt / "siim_oof_predictions.csv"
    pd.DataFrame({
        "image_name": train_id,
        "patient_id": patient_id,
        "leakage_group": leakage_group,
        "target": target,
        "fold": fold,
        "blended_probability": probability,
    }).to_csv(oof, index=False)
    submission = attempt / "submission.csv"
    pd.DataFrame({
        "image_name": sample["image_name"],
        "target": test_probability,
    }).to_csv(submission, index=False)
    result = {
        "competition_id": aggregate.COMPETITION_ID,
        "metric": "roc_auc",
        "direction": "maximize",
        "cv_score": 1.0,
        "valid_submission": True,
        "submission_sha256": common.sha256_file(submission),
        "official_grader_executed": False,
        "mle_private_grader_score": None,
        "kaggle_public_score": None,
        "kaggle_private_score": None,
        "budget": {"seed": seed},
    }
    result_path = competition / "result.json"
    _write_json(result_path, result)
    files = [result_path, bundle, oof, submission]
    _write_json(
        root / run_id / "collection_manifest.json",
        {
            "passed": True,
            "run_id": run_id,
            "files": [common.file_record(path) | {"local": str(path.resolve())} for path in files],
        },
    )


def test_siim_three_seed_aggregation_freezes_hash_bound_candidate(tmp_path: Path):
    sample = pd.DataFrame({
        "image_name": [f"te-{value:05d}" for value in range(4_142)],
        "target": np.zeros(4_142),
    })
    sample_path = tmp_path / "sample_submission.csv"
    sample.to_csv(sample_path, index=False)
    collected = tmp_path / "collected"
    formal_runs = []
    for seed in aggregate.EXPECTED_FORMAL_SEEDS:
        run_id = f"siim-s{seed}"
        _build_collection(collected, run_id, seed, sample)
        formal_runs.append({"seed": seed, "run_id": run_id})
    plan_path = tmp_path / "plan.json"
    _write_json(
        plan_path,
        {
            "schema": "evomind.siim.hpc_campaign_plan.v1",
            "run_id": "siim-business-run",
            "formal_seeds": list(aggregate.EXPECTED_FORMAL_SEEDS),
            "ablation_seeds": list(aggregate.EXPECTED_ABLATION_SEEDS),
            "formal_runs": formal_runs,
            "evaluation": {"bootstrap_samples": 120},
            "historical_thresholds": {
                "historical_private_grader": 0.92165,
                "bronze": 0.937,
                "silver": 0.9401,
                "gold": 0.9455,
            },
        },
    )
    output = tmp_path / "frozen"
    result = aggregate.aggregate(
        plan_path=plan_path,
        collected_root=collected,
        sample_path=sample_path,
        output_dir=output,
    )
    assert result["status"] == "frozen_before_private_grader"
    assert result["official_grader_executed"] is False
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["private_grader_execution_count"] == 0
    assert len(metrics["fold_metrics"]) == 5
    assert set(metrics["model_components"]) == {
        "pure_image",
        "lesion_focus",
        "image_metadata_fusion",
        "metadata_catboost",
        "final_ensemble",
    }
    assert metrics["roc_curve"]["false_positive_rate"][0] == pytest.approx(0.0)
    assert metrics["roc_curve"]["true_positive_rate"][-1] == pytest.approx(1.0)
    assert metrics["precision_recall_curve"]["recall"][0] == pytest.approx(0.0)
    assert metrics["precision_recall_curve"]["recall"][-1] == pytest.approx(1.0)
    assert len(pd.read_csv(output / "candidate_submission_withheld.csv")) == 4_142
    verification = json.loads(
        (output / "independent_verification.json").read_text(encoding="utf-8")
    )
    assert verification["passed"] is True
    assert verification["formal_ablation_seed_separation_verified"] is True
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["all_sha256_bound"] is True
    assert all(Path(record["path"]).parent == output.resolve() for record in manifest["artifacts"])
    assert all(Path(record["path"]).is_file() for record in manifest["artifacts"])


def test_siim_aggregation_failure_leaves_no_partial_candidate_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sample = pd.DataFrame({
        "image_name": [f"te-{value:05d}" for value in range(4_142)],
        "target": np.zeros(4_142),
    })
    sample_path = tmp_path / "sample_submission.csv"
    sample.to_csv(sample_path, index=False)
    collected = tmp_path / "collected"
    formal_runs = []
    for seed in aggregate.EXPECTED_FORMAL_SEEDS:
        run_id = f"siim-s{seed}"
        _build_collection(collected, run_id, seed, sample)
        formal_runs.append({"seed": seed, "run_id": run_id})
    plan_path = tmp_path / "plan.json"
    _write_json(
        plan_path,
        {
            "schema": "evomind.siim.hpc_campaign_plan.v1",
            "run_id": "siim-transaction-run",
            "formal_seeds": list(aggregate.EXPECTED_FORMAL_SEEDS),
            "ablation_seeds": list(aggregate.EXPECTED_ABLATION_SEEDS),
            "formal_runs": formal_runs,
            "evaluation": {"bootstrap_samples": 120},
        },
    )
    output = tmp_path / "frozen"
    original_write = aggregate.common.write_json_atomic

    def fail_metrics(path: Path, payload: dict) -> None:
        if path.name == "metrics.json":
            raise OSError("fixture interruption")
        original_write(path, payload)

    monkeypatch.setattr(aggregate.common, "write_json_atomic", fail_metrics)
    with pytest.raises(OSError, match="fixture interruption"):
        aggregate.aggregate(
            plan_path=plan_path,
            collected_root=collected,
            sample_path=sample_path,
            output_dir=output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".frozen.staging.*"))

    monkeypatch.setattr(aggregate.common, "write_json_atomic", original_write)
    result = aggregate.aggregate(
        plan_path=plan_path,
        collected_root=collected,
        sample_path=sample_path,
        output_dir=output,
    )
    assert result["status"] == "frozen_before_private_grader"
    assert (output / "artifact_manifest.json").is_file()


def test_siim_aggregation_rejects_formal_seed_reused_by_ablation(tmp_path: Path):
    plan = tmp_path / "bad-plan.json"
    _write_json(
        plan,
        {
            "schema": "evomind.siim.hpc_campaign_plan.v1",
            "run_id": "bad",
            "formal_seeds": [40, 44, 45],
            "ablation_seeds": [40, 41, 42],
            "formal_runs": [],
        },
    )
    with pytest.raises(aggregate.SiimAggregationError, match="formal seeds changed"):
        aggregate.aggregate(
            plan_path=plan,
            collected_root=tmp_path,
            sample_path=tmp_path / "missing.csv",
            output_dir=tmp_path / "out",
        )


def test_siim_r2_accepts_fresh_seeds_and_applies_only_the_frozen_formula(
    tmp_path: Path,
):
    sample = pd.DataFrame({
        "image_name": [f"te-{value:05d}" for value in range(4_142)],
        "target": np.zeros(4_142),
    })
    sample_path = tmp_path / "sample_submission.csv"
    sample.to_csv(sample_path, index=False)
    collected = tmp_path / "collected"
    formal_runs = []
    for seed in aggregate.R2_FORMAL_SEEDS:
        run_id = f"siim-r2-s{seed}"
        _build_collection(collected, run_id, seed, sample)
        formal_runs.append({"seed": seed, "run_id": run_id})
    protocol = {
        "candidate_id": "r2_foldwise_rank_channel_consensus_v1",
        "score_definition": {
            "weights": {
                "seed_final_foldrank": 0.5,
                "pure_image": 0.3,
                "image_metadata_fusion": 0.2,
                "lesion_focus": 0.0,
                "metadata_catboost": 0.0,
            }
        },
        "adoption_gate": {
            "minimum_overall_roc_auc_delta": 0.0005,
            "minimum_pr_auc_delta": 0.0,
            "minimum_positive_fold_count": 4,
            "maximum_worst_fold_auc_regression": 0.001,
            "minimum_paired_group_bootstrap_probability_delta_gt_zero": 0.9,
        },
    }
    plan_path = tmp_path / "plan-r2.json"
    _write_json(
        plan_path,
        {
            "schema": "evomind.siim.hpc_campaign_plan.v1",
            "run_id": "siim-r2-business-run",
            "formal_seeds": list(aggregate.R2_FORMAL_SEEDS),
            "ablation_seeds": list(aggregate.EXPECTED_ABLATION_SEEDS),
            "formal_runs": formal_runs,
            "evaluation": {"bootstrap_samples": 120},
            "evolution_protocol": protocol,
        },
    )

    output = tmp_path / "frozen-r2"
    result = aggregate.aggregate(
        plan_path=plan_path,
        collected_root=collected,
        sample_path=sample_path,
        output_dir=output,
    )

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    freeze = json.loads((output / "candidate_freeze.json").read_text(encoding="utf-8"))
    assert result["status"] == "frozen_before_private_grader"
    assert metrics["evolution_decision"]["candidate_id"] == protocol["candidate_id"]
    assert freeze["evolution_protocol"]["score_definition"]["weights"][
        "seed_final_foldrank"
    ] == pytest.approx(0.5)
    assert freeze["formal_seeds"] == [46, 47, 48]
