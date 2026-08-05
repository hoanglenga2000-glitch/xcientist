"""Tests for the read-only partial Spooky DeBERTa quality audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import audit_spooky_deberta_partial as partial


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _probability(truth: np.ndarray, strength: float) -> np.ndarray:
    values = np.full((len(truth), 3), 1.0)
    values[np.arange(len(truth)), truth] += strength
    return values / values.sum(axis=1, keepdims=True)


def _fixture(tmp_path: Path, *, corrupt_probability: bool = False) -> partial.AuditConfig:
    run_dir = tmp_path / "run"
    public_dir = tmp_path / "public"
    run_dir.mkdir()
    public_dir.mkdir()
    rows = 30
    test_rows = 4
    truth = np.tile(np.arange(3, dtype=np.int64), rows // 3)
    folds = np.tile(np.arange(5, dtype=np.int16), rows // 5)
    ids = np.asarray([f"train-{index}" for index in range(rows)], dtype=np.str_)
    text = [f"unique public text {index}" for index in range(rows)]
    # One duplicate pair is deliberately kept inside the same outer fold.
    text[5] = text[0]
    truth[5] = truth[0]
    folds[5] = folds[0]
    train = pd.DataFrame(
        {
            "id": ids,
            "text": text,
            "author": [partial.CLASS_COLUMNS[value] for value in truth],
        }
    )
    train_path = public_dir / "train.csv"
    train.to_csv(train_path, index=False)

    np.savez_compressed(run_dir / "fold_assignments.npz", train_id=ids, truth=truth, fold=folds)
    duplicate = partial.duplicate_group_report(text, truth, folds, seed=42)
    duplicate.pop("groups_crossing_folds")
    plan = {
        "schema": partial.EXPECTED_PLAN_SCHEMA,
        "status": "frozen_before_training",
        "inputs": {
            "public_train_sha256": partial.sha256_file(train_path),
            "train_rows": rows,
            "test_rows": test_rows,
            "class_order": list(partial.CLASS_COLUMNS),
            "duplicate_group_report": duplicate,
        },
        "training": {"folds": 5, "seed": 42},
        "promotion_gate": {
            "bronze_threshold_reference": 0.29381,
            "single_seed_log_loss_threshold": 0.292,
        },
    }
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    plan_hash = partial.sha256_file(plan_path)
    runner_hash = "a" * 64
    manifest = {
        "schema": partial.EXPECTED_MANIFEST_SCHEMA,
        "run_id": "current-run",
        "plan_sha256": plan_hash,
        "source_contract": {"runner": {"actual_sha256": runner_hash}},
        "fold_assignment": {
            "sha256": partial.sha256_file(run_dir / "fold_assignments.npz")
        },
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(
        run_dir / "heartbeat.json",
        {
            "status": "running",
            "fold": 2,
            "epoch": 1,
            "update": 10,
            "total_updates": 100,
            "process_signals_sent": 0,
        },
    )

    prior_transformer = _probability(truth, 1.3)
    prior_sparse = _probability(truth, 1.0)
    prior_byte = _probability(truth, 0.7)
    prior_candidate = _probability(truth, 1.6)
    prior_bundle = tmp_path / "prior.npz"
    np.savez_compressed(
        prior_bundle,
        truth=truth,
        fold=folds,
        transformer_oof=prior_transformer,
        sparse_oof=prior_sparse,
        byte_oof=prior_byte,
        candidate_oof=prior_candidate,
        legacy_object_id=np.asarray([f"legacy-{index}" for index in range(rows)], dtype=object),
    )
    prior_summary = tmp_path / "prior_summary.json"
    _write_json(
        prior_summary,
        {"status": "single_seed_gate_failed", "candidate_oof_log_loss": 0.34},
    )
    cross_run_plan = {
        "prior": {
            "run_id": "prior-run",
            "summary": {
                "path": str(prior_summary),
                "sha256": partial.sha256_file(prior_summary),
            },
            "bundle": {
                "path": str(prior_bundle),
                "sha256": partial.sha256_file(prior_bundle),
            },
        },
        "current": {
            "run_id": "current-run",
            "plan": {"path": str(plan_path), "sha256": plan_hash},
        },
    }
    cross_run_plan_path = tmp_path / "cross_run_plan.json"
    _write_json(cross_run_plan_path, cross_run_plan)

    for fold in (0, 1):
        valid = np.flatnonzero(folds == fold).astype(np.int64)
        fit = np.flatnonzero(folds != fold).astype(np.int64)
        transformer = _probability(truth[valid], 1.5)
        sparse = _probability(truth[valid], 2.0)
        sparse_metadata_score = partial.multiclass_log_loss(truth[valid], sparse)
        if corrupt_probability and fold == 1:
            sparse[0, 0] += 0.25
        transformer_test = np.full((test_rows, 3), 1.0 / 3.0)
        sparse_test = np.full((test_rows, 3), 1.0 / 3.0)
        prediction_path = run_dir / f"fold_{fold}_predictions.npz"
        np.savez_compressed(
            prediction_path,
            valid_indices=valid,
            test_indices=np.arange(test_rows, dtype=np.int64),
            transformer_valid=transformer,
            transformer_test=transformer_test,
            sparse_valid=sparse,
            sparse_test=sparse_test,
        )
        metadata = {
            "status": "passed",
            "fold": fold,
            "plan_sha256": plan_hash,
            "runner_sha256": runner_hash,
            "prediction_sha256": partial.sha256_file(prediction_path),
            "fit_index_sha256": partial.index_sha256(fit),
            "valid_index_sha256": partial.index_sha256(valid),
            "fit_valid_disjoint": True,
            "component_log_loss": {
                "transformer": partial.multiclass_log_loss(truth[valid], transformer),
                "sparse": sparse_metadata_score,
            },
            "checkpoint_selection_used_outer_fold": False,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
        }
        _write_json(run_dir / f"fold_{fold}_result.json", metadata)

    return partial.AuditConfig(
        run_dir=run_dir,
        public_dir=public_dir,
        plan_path=plan_path,
        cross_run_plan_path=cross_run_plan_path,
        output_path=tmp_path / "audit.json",
    )


def test_partial_audit_recomputes_completed_public_oof_and_ignores_object_array(
    tmp_path: Path,
):
    report = partial.audit(_fixture(tmp_path))

    assert report["status"] == "passed_partial_audit"
    assert report["completed_folds"] == [0, 1]
    assert report["coverage"]["exact_once_completed"] is True
    assert report["coverage"]["completed_rows"] == 12
    assert report["reference"]["prior_alignment"]["allow_pickle"] is False
    assert all(record["passed"] for record in report["fold_records"])
    assert report["decision"]["meta_candidate_available"] is False
    assert (
        report["decision"]["training_action"]
        == "CONTINUE_TO_FULL_FIVE_FOLD_VERIFICATION"
    )
    assert report["private_labels_used"] is False
    assert report["official_grader_executed"] is False
    assert report["kaggle_submission_executed"] is False
    assert report["process_signals_sent"] == 0


def test_partial_audit_fails_closed_on_non_normalized_probability(tmp_path: Path):
    report = partial.audit(_fixture(tmp_path, corrupt_probability=True))

    assert report["status"] == "failed_partial_audit"
    fold_one = next(record for record in report["fold_records"] if record["fold"] == 1)
    assert fold_one["checks"]["sparse_valid_probability"] is False
    assert report["global_checks"]["completed_fold_artifacts"] is False


def test_duplicate_report_detects_cross_fold_duplicate():
    text = ["The same text", "the same text"]
    truth = np.asarray([0, 0], dtype=np.int64)
    folds = np.asarray([0, 1], dtype=np.int16)

    report = partial.duplicate_group_report(text, truth, folds, seed=42)

    assert report["duplicate_groups"] == 1
    assert report["groups_crossing_folds"] == 1
    assert report["group_isolation"] is False
