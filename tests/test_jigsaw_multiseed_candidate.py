from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate_jigsaw_multiseed_candidate import (
    TARGET_COLUMNS,
    aggregate,
    load_numeric_bundle,
    model_seed_from_run_plan,
)
from scripts.verify_jigsaw_multiseed_candidate import verify


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    rows = 30
    test_rows = 9
    train_ids = np.asarray([f"train-{index}" for index in range(rows)], dtype=object)
    test_ids = np.asarray([f"test-{index}" for index in range(test_rows)], dtype=object)
    truth = (
        np.arange(rows)[:, None] + np.arange(len(TARGET_COLUMNS))[None, :]
    ) % 2
    truth = truth.astype(np.int8)
    fold = (np.arange(rows) % 5).astype(np.int16)
    train = pd.DataFrame({"id": train_ids.astype(str)})
    for index, target in enumerate(TARGET_COLUMNS):
        train[target] = truth[:, index]
    test = pd.DataFrame({"id": test_ids.astype(str), "comment_text": ["x"] * test_rows})
    sample = pd.DataFrame({"id": test_ids.astype(str)})
    for target in TARGET_COLUMNS:
        sample[target] = 0.5
    public = tmp_path / "public"
    public.mkdir()
    train_path = public / "train.csv"
    test_path = public / "test.csv"
    sample_path = public / "sample_submission.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    sample.to_csv(sample_path, index=False)

    seed_runs = []
    for seed in (40, 41, 42):
        run_dir = tmp_path / f"seed-{seed}"
        run_dir.mkdir()
        run_plan = run_dir / "plan.json"
        _write_json(
            run_plan,
            {
                "schema": "evomind.jigsaw.transformer_confirmation_plan.v2",
                "status": "frozen_before_training",
                "training": {"model_seed": seed},
            },
        )
        candidate_oof = np.clip(
            0.02 + 0.96 * truth + (seed - 41) * 0.0001, 0.0, 1.0
        )
        base = np.linspace(0.05, 0.95, test_rows)[:, None]
        candidate_test = np.clip(
            np.repeat(base, len(TARGET_COLUMNS), axis=1)
            + (seed - 41) * 0.0001,
            0.0,
            1.0,
        )
        bundle_path = run_dir / "bundle.npz"
        np.savez_compressed(
            bundle_path,
            truth=truth,
            fold=fold,
            candidate_oof=candidate_oof,
            candidate_test=candidate_test,
            candidate_write_counts=np.ones_like(truth, dtype=np.uint8),
            # Legacy object arrays must remain unread by the production aggregator.
            train_id=train_ids,
            test_id=test_ids,
        )
        summary_path = run_dir / "summary.json"
        _write_json(
            summary_path,
            {
                "schema": "evomind.jigsaw.transformer_run.v1",
                "run_id": f"seed-{seed}",
                "status": "promotion_gate_passed",
                "plan_sha256": _sha(run_plan),
                "candidate_oof_auc": 1.0,
                "promotion_gate": {
                    "passed": True,
                    "gain_over_strongest_base": 0.001,
                },
                "prediction_bundle": _record(bundle_path),
                "private_labels_used": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
                "human_gate_preserved": True,
            },
        )
        verification_path = run_dir / "independent_verification.json"
        _write_json(
            verification_path,
            {
                "schema": "evomind.jigsaw.transformer_independent_verification.v1",
                "run_id": f"seed-{seed}",
                "status": "promotion_gate_passed",
                "plan_sha256": _sha(run_plan),
                "full_contract_valid": True,
                "metrics": {"candidate_auc": 1.0},
                "private_labels_used": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            },
        )
        seed_runs.append(
            {
                "model_seed": seed,
                "run_id": f"seed-{seed}",
                "run_plan": _record(run_plan),
                "summary": _record(summary_path),
                "independent_verification": _record(verification_path),
                "prediction_bundle": _record(bundle_path),
            }
        )

    implementation = {}
    for name in ("aggregator", "verifier"):
        source = tmp_path / f"{name}.py"
        source.write_text(f"# {name}\n", encoding="utf-8")
        implementation[name] = _record(source)
    queue_status = tmp_path / "queue_status.json"
    _write_json(
        queue_status,
        {
            "status": "all_three_seed_sources_stable",
            "process_signals_sent": 0,
        },
    )
    output_dir = tmp_path / "output"
    plan_path = tmp_path / "aggregation_plan.json"
    result_path = output_dir / "result.json"
    _write_json(
        plan_path,
        {
            "schema": "evomind.jigsaw.multiseed_confirmation_plan.v1",
            "status": "frozen_before_aggregation",
            "competition_id": "jigsaw-toxic-comment-classification-challenge",
            "implementation": implementation,
            "public_inputs": {
                "train": _record(train_path),
                "test": _record(test_path),
                "sample_submission": _record(sample_path),
            },
            "seed_runs": seed_runs,
            "queue_status": _record(queue_status),
            "confirmation_gate": {
                "minimum_seed_auc": 0.987,
                "minimum_mean_auc": 0.987,
                "minimum_seed_gain": 0.0003,
                "maximum_population_std": 0.0025,
            },
            "output": {
                "directory": str(output_dir.resolve()),
                "result": str(result_path.resolve()),
            },
            "boundaries": {
                "private_labels_used": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
                "process_signals_sent": 0,
                "human_gate_preserved": True,
            },
        },
    )
    return plan_path, result_path


def test_numeric_loader_ignores_legacy_object_ids(tmp_path: Path) -> None:
    plan_path, _result_path = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    bundle = Path(plan["seed_runs"][0]["prediction_bundle"]["path"])
    numeric = load_numeric_bundle(bundle)
    assert set(numeric) == {
        "truth",
        "fold",
        "candidate_oof",
        "candidate_test",
        "candidate_write_counts",
    }


def test_run_plan_model_seed_supports_both_frozen_contract_versions() -> None:
    assert model_seed_from_run_plan({"training": {"model_seed": 41}}) == 41
    assert model_seed_from_run_plan({"training": {"seed": 42}}) == 42
    assert model_seed_from_run_plan({"training": {"model_seed": 40, "seed": 40}}) == 40
    with pytest.raises(ValueError, match="seed fields disagree"):
        model_seed_from_run_plan({"training": {"model_seed": 40, "seed": 41}})
    with pytest.raises(ValueError, match="does not freeze a model seed"):
        model_seed_from_run_plan({"training": {}})


def test_aggregate_and_independent_verify(tmp_path: Path) -> None:
    plan_path, result_path = _fixture(tmp_path)
    result = aggregate(plan_path)
    assert result["status"] == "confirmation_passed_human_gate_pending"
    assert result["candidate_ready_for_human_gate"] is True
    assert result["metrics"]["ensemble_oof_auc"] == 1.0

    report = verify(plan_path, result_path, tmp_path / "verification.json")
    assert report["ok"] is True
    assert report["errors"] == []

    with np.load(Path(result["prediction_bundle"]["path"]), allow_pickle=False) as archive:
        assert archive["train_id"].dtype.kind == "U"
        assert archive["test_id"].dtype.kind == "U"


def test_aggregate_accepts_hash_identical_relocated_seed_bundle(tmp_path: Path) -> None:
    plan_path, _result_path = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    seed41 = plan["seed_runs"][1]
    summary_path = Path(seed41["summary"]["path"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["prediction_bundle"]["path"] = (
        "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/remote-seed41/bundle.npz"
    )
    _write_json(summary_path, summary)
    seed41["summary"] = _record(summary_path)
    _write_json(plan_path, plan)

    result = aggregate(plan_path)

    provenance = result["seed_records"][1]["prediction_bundle_provenance"]
    assert result["status"] == "confirmation_passed_human_gate_pending"
    assert provenance["relocated"] is True
    assert provenance["identity"] == "sha256"
    assert provenance["summary_path"] == summary["prediction_bundle"]["path"]
    assert provenance["materialized_path"] == seed41["prediction_bundle"]["path"]


def test_aggregate_rejects_relocated_seed_bundle_with_different_hash(
    tmp_path: Path,
) -> None:
    plan_path, _result_path = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    seed41 = plan["seed_runs"][1]
    summary_path = Path(seed41["summary"]["path"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["prediction_bundle"] = {
        "path": "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/other/bundle.npz",
        "sha256": "0" * 64,
    }
    _write_json(summary_path, summary)
    seed41["summary"] = _record(summary_path)
    _write_json(plan_path, plan)

    with pytest.raises(ValueError, match="Seed 41 summary bundle record mismatch"):
        aggregate(plan_path)


def test_verifier_detects_tampered_submission(tmp_path: Path) -> None:
    plan_path, result_path = _fixture(tmp_path)
    result = aggregate(plan_path)
    submission_path = Path(result["submission_withheld"]["path"])
    submission = pd.read_csv(submission_path)
    submission.loc[0, TARGET_COLUMNS[0]] = 0.123
    submission.to_csv(submission_path, index=False)

    report = verify(plan_path, result_path, tmp_path / "tampered_verification.json")
    assert report["ok"] is False
    assert "submission_withheld" in report["errors"]
