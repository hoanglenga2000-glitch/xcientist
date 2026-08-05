"""Independent verifier for the leakage-controlled Jigsaw transformer OOF run.

This verifier deliberately does not import ``run_jigsaw_transformer_oof``.  It
reconstructs every fold, blend weight, aggregate metric, candidate prediction,
and promotion-gate decision from immutable public inputs and completed fold
artifacts.  It can audit a live partial run without reading an in-progress fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGET_COLUMNS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def independent_fractional_rank(values: np.ndarray) -> np.ndarray:
    """Average ranks scaled by row count, implemented independently with pandas."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    require(len(array) > 0, "Rank input is empty")
    require(bool(np.isfinite(array).all()), "Rank input contains a non-finite value")
    ranked = pd.Series(array, copy=False).rank(method="average").to_numpy(dtype=np.float64)
    return ranked / len(array)


def independent_fold_rank_matrix(values: np.ndarray, folds: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    fold_vector = np.asarray(folds, dtype=np.int16).reshape(-1)
    require(matrix.ndim == 2, "Fold-rank input must be a matrix")
    require(len(matrix) == len(fold_vector), "Fold-rank rows do not match fold assignment")
    ranked = np.full(matrix.shape, np.nan, dtype=np.float64)
    for fold in sorted(int(value) for value in np.unique(fold_vector)):
        mask = fold_vector == fold
        for label in range(matrix.shape[1]):
            ranked[mask, label] = independent_fractional_rank(matrix[mask, label])
    require(bool(np.isfinite(ranked).all()), "Fold-rank reconstruction is incomplete")
    return ranked


def mean_columnwise_auc(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, list[float]]:
    labels = np.asarray(truth, dtype=np.int8)
    scores = np.asarray(prediction, dtype=np.float64)
    require(labels.shape == scores.shape and labels.ndim == 2, "Truth/prediction shape mismatch")
    per_label = [
        float(roc_auc_score(labels[:, label], scores[:, label]))
        for label in range(labels.shape[1])
    ]
    return float(np.mean(per_label)), per_label


def select_weight(
    truth: np.ndarray,
    sparse_rank: np.ndarray,
    transformer_rank: np.ndarray,
    grid: Sequence[float],
) -> tuple[float, list[dict[str, float]]]:
    candidates: list[dict[str, float]] = []
    for value in grid:
        weight = float(value)
        prediction = (1.0 - weight) * sparse_rank + weight * transformer_rank
        candidates.append(
            {"transformer_weight": weight, "auc": float(roc_auc_score(truth, prediction))}
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["auc"],
            -abs(item["transformer_weight"] - 0.5),
            -item["transformer_weight"],
        ),
    )
    return float(selected["transformer_weight"]), candidates


def reconstruct_cross_fit_blend(
    sparse_oof: np.ndarray,
    transformer_oof: np.ndarray,
    sparse_test: np.ndarray,
    transformer_test_by_fold: np.ndarray,
    truth: np.ndarray,
    fold_assignment: np.ndarray,
    weight_grid: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    sparse = np.asarray(sparse_oof, dtype=np.float64)
    transformer = np.asarray(transformer_oof, dtype=np.float64)
    sparse_test_matrix = np.asarray(sparse_test, dtype=np.float64)
    transformer_test = np.asarray(transformer_test_by_fold, dtype=np.float64)
    labels = np.asarray(truth, dtype=np.int8)
    folds = np.asarray(fold_assignment, dtype=np.int16).reshape(-1)
    grid = tuple(sorted({float(value) for value in weight_grid}))
    unique_folds = sorted(int(value) for value in np.unique(folds))

    require(sparse.shape == transformer.shape == labels.shape, "OOF matrix contract mismatch")
    require(unique_folds == list(range(5)), "Expected contiguous outer folds 0..4")
    require(bool(grid) and all(0.0 <= value <= 1.0 for value in grid), "Invalid weight grid")
    require(sparse_test_matrix.shape == (len(transformer_test[0]), labels.shape[1]), "Test shape mismatch")
    require(
        transformer_test.shape == (5, len(sparse_test_matrix), labels.shape[1]),
        "Transformer test-by-fold contract mismatch",
    )
    require(
        all(np.isfinite(value).all() for value in (sparse, transformer, sparse_test_matrix, transformer_test)),
        "Blend input contains non-finite values",
    )

    sparse_rank = independent_fold_rank_matrix(sparse, folds)
    transformer_rank = independent_fold_rank_matrix(transformer, folds)
    sparse_test_rank = np.column_stack(
        [independent_fractional_rank(sparse_test_matrix[:, label]) for label in range(labels.shape[1])]
    )
    transformer_test_rank = np.empty_like(transformer_test, dtype=np.float64)
    for fold in unique_folds:
        transformer_test_rank[fold] = np.column_stack(
            [
                independent_fractional_rank(transformer_test[fold, :, label])
                for label in range(labels.shape[1])
            ]
        )

    candidate_oof = np.full(sparse.shape, np.nan, dtype=np.float64)
    candidate_counts = np.zeros(labels.shape, dtype=np.uint8)
    test_components = np.zeros(transformer_test.shape, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in unique_folds:
        fit_mask = folds != fold
        held_mask = folds == fold
        for label, target in enumerate(TARGET_COLUMNS):
            weight, candidates = select_weight(
                labels[fit_mask, label],
                sparse_rank[fit_mask, label],
                transformer_rank[fit_mask, label],
                grid,
            )
            candidate_oof[held_mask, label] = (
                (1.0 - weight) * sparse_rank[held_mask, label]
                + weight * transformer_rank[held_mask, label]
            )
            candidate_counts[held_mask, label] += 1
            test_components[fold, :, label] = (
                (1.0 - weight) * sparse_test_rank[:, label]
                + weight * transformer_test_rank[fold, :, label]
            )
            records.append(
                {
                    "score_fold": fold,
                    "fit_folds": [value for value in unique_folds if value != fold],
                    "target": target,
                    "target_index": label,
                    "selected_transformer_weight": weight,
                    "candidate_scores": candidates,
                }
            )
    require(bool(np.all(candidate_counts == 1)), "Candidate OOF was not written exactly once")
    require(bool(np.isfinite(candidate_oof).all()), "Candidate OOF is incomplete")
    averaged_test = np.mean(test_components, axis=0)
    candidate_test = np.column_stack(
        [independent_fractional_rank(averaged_test[:, label]) for label in range(labels.shape[1])]
    )
    return candidate_oof, candidate_test, candidate_counts, records


def validate_driver_audit(plan: dict[str, Any]) -> dict[str, Any]:
    driver = plan.get("driver", {})
    audit_path = Path(str(driver.get("audit_path", "")))
    require(audit_path.is_file(), "GPT-5.6 audit artifact is missing")
    audit_sha = sha256_file(audit_path)
    require(audit_sha == driver.get("audit_sha256"), "GPT-5.6 audit SHA256 drifted")
    audit = read_json(audit_path)
    planner = audit.get("planner", {})
    served_model = (
        audit.get("served_model")
        or planner.get("served_model")
        or audit.get("response", {}).get("served_model")
    )
    requested_model = (
        audit.get("requested_model")
        or planner.get("requested_model")
        or audit.get("request", {}).get("model")
    )
    require(served_model == "gpt-5.6-sol", "Audit did not record gpt-5.6-sol as served model")
    require(requested_model == "gpt-5.6-sol", "Audit did not request gpt-5.6-sol")
    return {
        "path": str(audit_path),
        "sha256": audit_sha,
        "requested_model": requested_model,
        "served_model": served_model,
    }


def resolve_seed_contract(plan: dict[str, Any]) -> dict[str, Any]:
    """Return the explicit or legacy Jigsaw seed contract."""

    training = plan.get("training") or {}
    legacy_seed = training.get("seed")
    fold_seed = training.get("fold_seed", legacy_seed)
    model_seed = training.get("model_seed", legacy_seed)
    require(fold_seed is not None, "Frozen plan has no fold seed")
    require(model_seed is not None, "Frozen plan has no model seed")
    return {
        "fold_assignment_seed": int(fold_seed),
        "model_seed": int(model_seed),
        "fold_model_seeds": [int(model_seed + fold * 1009) for fold in range(5)],
        "explicit_split": "fold_seed" in training or "model_seed" in training,
    }


def verify_run(
    *,
    run_dir: Path,
    public_dir: Path,
    sparse_bundle: Path,
    frozen_plan: Path,
    require_complete: bool,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    public_dir = public_dir.resolve()
    sparse_bundle = sparse_bundle.resolve()
    frozen_plan = frozen_plan.resolve()
    manifest = read_json(run_dir / "manifest.json")
    plan = read_json(frozen_plan)
    plan_sha = sha256_file(frozen_plan)
    seed_contract = resolve_seed_contract(plan)
    require(manifest.get("plan_sha256") == plan_sha, "Manifest/plan SHA256 mismatch")
    require(manifest.get("diagnostic") is False, "Production verifier received a diagnostic run")
    require(manifest.get("private_labels_used") is False, "Manifest reports private-label use")
    require(manifest.get("official_grader_executed") is False, "Manifest reports grader execution")
    require(manifest.get("kaggle_submission_executed") is False, "Manifest reports Kaggle submission")
    require(manifest.get("human_gate_preserved") is True, "Manifest did not preserve Human Gate")
    require(manifest.get("requested_folds") == [0, 1, 2, 3, 4], "Run did not request all five folds")
    if seed_contract["explicit_split"]:
        require(manifest.get("process_signals_sent") == 0, "Manifest reports process signals")
        manifest_seed_contract = manifest.get("seed_contract") or {}
        require(
            manifest_seed_contract.get("fold_assignment_seed")
            == seed_contract["fold_assignment_seed"],
            "Manifest fold seed mismatch",
        )
        require(
            manifest_seed_contract.get("model_seed") == seed_contract["model_seed"],
            "Manifest model seed mismatch",
        )
        require(
            manifest_seed_contract.get("fold_model_seeds")
            == seed_contract["fold_model_seeds"],
            "Manifest per-fold model seeds mismatch",
        )

    inputs = plan.get("inputs", {})
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    sample_path = public_dir / "sample_submission.csv"
    input_hashes = {
        "public_train_sha256": sha256_file(train_path),
        "public_test_sha256": sha256_file(test_path),
        "public_sample_submission_sha256": sha256_file(sample_path),
        "sparse_bundle_sha256": sha256_file(sparse_bundle),
    }
    for key, value in input_hashes.items():
        require(value == inputs.get(key), f"Frozen input drift: {key}")
    driver_audit = validate_driver_audit(plan)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    require(tuple(str(value) for value in sample.columns[1:]) == TARGET_COLUMNS, "Target order drifted")
    require(len(sample) == len(test), "Sample/test row count mismatch")
    truth = train[list(TARGET_COLUMNS)].to_numpy(dtype=np.int8)
    require(truth.shape == (len(train), 6), "Public truth shape mismatch")

    required_sparse = {"truth", "fold", "oof_stacker", "test_stacker"}
    with np.load(sparse_bundle, allow_pickle=False) as archive:
        require(required_sparse.issubset(archive.files), "Sparse bundle is missing numeric arrays")
        sparse_truth = np.asarray(archive["truth"], dtype=np.int8)
        fold_assignment = np.asarray(archive["fold"], dtype=np.int16)
        sparse_oof = np.asarray(archive["oof_stacker"], dtype=np.float64)
        sparse_test = np.asarray(archive["test_stacker"], dtype=np.float64)
    require(bool(np.array_equal(sparse_truth, truth)), "Sparse/public truth mismatch")
    require(fold_assignment.shape == (len(train),), "Sparse fold shape mismatch")
    require(sorted(int(value) for value in np.unique(fold_assignment)) == list(range(5)), "Fold set mismatch")
    require(sparse_oof.shape == truth.shape, "Sparse OOF shape mismatch")
    require(sparse_test.shape == (len(test), 6), "Sparse test shape mismatch")
    require(bool(np.isfinite(sparse_oof).all() and np.isfinite(sparse_test).all()), "Sparse prediction is non-finite")

    completed_folds: list[int] = []
    fold_checks: list[dict[str, Any]] = []
    transformer_oof = np.full(truth.shape, np.nan, dtype=np.float64)
    transformer_test_by_fold = np.full((5, len(test), 6), np.nan, dtype=np.float64)
    transformer_counts = np.zeros(truth.shape, dtype=np.uint8)
    for fold in range(5):
        result_path = run_dir / f"fold_{fold}_result.json"
        prediction_path = run_dir / f"fold_{fold}_predictions.npz"
        if not result_path.is_file() and not prediction_path.is_file():
            continue
        require(result_path.is_file() and prediction_path.is_file(), f"Fold {fold} has a torn artifact pair")
        result = read_json(result_path)
        require(result.get("status") == "passed", f"Fold {fold} did not pass")
        require(result.get("diagnostic") is False, f"Fold {fold} is diagnostic")
        require(result.get("fold") == fold, f"Fold {fold} result identity mismatch")
        require(result.get("plan_sha256") == plan_sha, f"Fold {fold} plan SHA mismatch")
        require(
            result.get("seed") == seed_contract["fold_model_seeds"][fold],
            f"Fold {fold} model seed mismatch",
        )
        require(result.get("private_labels_used") is False, f"Fold {fold} reports private labels")
        require(result.get("official_grader_executed") is False, f"Fold {fold} reports grader execution")
        if seed_contract["explicit_split"]:
            require(result.get("process_signals_sent") == 0, f"Fold {fold} reports process signals")
        require(result.get("train_rows") == int(np.sum(fold_assignment != fold)), f"Fold {fold} train rows mismatch")
        require(result.get("valid_rows") == int(np.sum(fold_assignment == fold)), f"Fold {fold} valid rows mismatch")
        require(result.get("test_rows") == len(test), f"Fold {fold} test rows mismatch")
        require(result.get("mixed_precision") == "bf16", f"Fold {fold} did not use BF16")
        require(result.get("prediction_sha256") == sha256_file(prediction_path), f"Fold {fold} SHA mismatch")
        epochs = result.get("epochs", [])
        expected_epochs = int(plan["training"]["epochs_per_fold"])
        require(len(epochs) == expected_epochs, f"Fold {fold} epoch count mismatch")
        require([item.get("epoch") for item in epochs] == list(range(1, expected_epochs + 1)), f"Fold {fold} epoch order mismatch")

        with np.load(prediction_path, allow_pickle=False) as archive:
            required_prediction = {"valid_indices", "valid_prediction", "test_indices", "test_prediction"}
            require(required_prediction.issubset(archive.files), f"Fold {fold} prediction arrays missing")
            valid_indices = np.asarray(archive["valid_indices"], dtype=np.int64)
            valid_prediction = np.asarray(archive["valid_prediction"], dtype=np.float64)
            test_indices = np.asarray(archive["test_indices"], dtype=np.int64)
            test_prediction = np.asarray(archive["test_prediction"], dtype=np.float64)
        expected_valid_indices = np.flatnonzero(fold_assignment == fold)
        require(bool(np.array_equal(valid_indices, expected_valid_indices)), f"Fold {fold} valid indices drifted")
        require(bool(np.array_equal(test_indices, np.arange(len(test)))), f"Fold {fold} test order drifted")
        require(valid_prediction.shape == (len(valid_indices), 6), f"Fold {fold} valid shape mismatch")
        require(test_prediction.shape == (len(test), 6), f"Fold {fold} test shape mismatch")
        require(bool(np.isfinite(valid_prediction).all()), f"Fold {fold} valid prediction is non-finite")
        require(bool(np.isfinite(test_prediction).all()), f"Fold {fold} test prediction is non-finite")
        recomputed_auc, per_label_auc = mean_columnwise_auc(truth[valid_indices], valid_prediction)
        require(close(recomputed_auc, epochs[-1]["validation_auc"]), f"Fold {fold} reported AUC mismatch")
        transformer_oof[valid_indices] = valid_prediction
        transformer_counts[valid_indices] += 1
        transformer_test_by_fold[fold] = test_prediction
        completed_folds.append(fold)
        fold_checks.append(
            {
                "fold": fold,
                "prediction_sha256": sha256_file(prediction_path),
                "valid_rows": len(valid_indices),
                "test_rows": len(test),
                "recomputed_final_epoch_auc": recomputed_auc,
                "per_label_auc": dict(zip(TARGET_COLUMNS, per_label_auc, strict=True)),
                "exact_expected_valid_indices": True,
                "test_order_exact": True,
            }
        )

    report: dict[str, Any] = {
        "schema": "evomind.jigsaw.transformer_independent_verification.v1",
        "created_at": now_iso(),
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id"),
        "status": "in_progress",
        "plan_sha256": plan_sha,
        "driver_audit": driver_audit,
        "seed_contract": seed_contract,
        "input_hashes": input_hashes,
        "completed_folds": completed_folds,
        "fold_checks": fold_checks,
        "partial_contract_valid": True,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "human_gate_preserved": True,
        "process_signals_sent": 0,
    }
    if len(completed_folds) < 5:
        require(not require_complete, f"Run is incomplete: {len(completed_folds)}/5 folds")
        return report

    require(bool(np.all(transformer_counts == 1)), "Transformer OOF exact-once check failed")
    require(bool(np.isfinite(transformer_oof).all()), "Transformer OOF assembly is non-finite")
    require(bool(np.isfinite(transformer_test_by_fold).all()), "Transformer test assembly is non-finite")
    summary_path = run_dir / "summary.json"
    require(summary_path.is_file(), "Complete fold set has no summary.json")
    summary = read_json(summary_path)
    require(summary.get("plan_sha256") == plan_sha, "Summary/plan SHA mismatch")
    require(summary.get("private_labels_used") is False, "Summary reports private-label use")
    require(summary.get("official_grader_executed") is False, "Summary reports grader execution")
    require(summary.get("kaggle_submission_executed") is False, "Summary reports Kaggle submission")
    require(summary.get("human_gate_preserved") is True, "Summary did not preserve Human Gate")

    weight_grid = plan["ensemble"]["weight_grid"]
    candidate_oof, candidate_test, candidate_counts, reconstructed_records = reconstruct_cross_fit_blend(
        sparse_oof,
        transformer_oof,
        sparse_test,
        transformer_test_by_fold,
        truth,
        fold_assignment,
        weight_grid,
    )
    sparse_rank = independent_fold_rank_matrix(sparse_oof, fold_assignment)
    transformer_rank = independent_fold_rank_matrix(transformer_oof, fold_assignment)
    sparse_auc, sparse_per_label = mean_columnwise_auc(truth, sparse_rank)
    transformer_auc, transformer_per_label = mean_columnwise_auc(truth, transformer_rank)
    candidate_auc, candidate_per_label = mean_columnwise_auc(truth, candidate_oof)
    strongest = max(sparse_auc, transformer_auc)
    gain = candidate_auc - strongest
    threshold = float(plan["promotion_gate"]["threshold"])
    minimum_gain = float(plan["promotion_gate"]["minimum_gain_over_strongest_public_oof_base"])
    checks = {
        "aggregate_threshold": candidate_auc >= threshold,
        "minimum_gain": gain >= minimum_gain,
        "exact_once_transformer_oof": bool(np.all(transformer_counts == 1)),
        "exact_once_candidate_oof": bool(np.all(candidate_counts == 1)),
        "all_six_labels_scoreable": all(np.unique(truth[:, label]).size == 2 for label in range(6)),
        "prediction_provenance_validated": True,
        "private_labels_unused": True,
        "human_gate_preserved": True,
    }
    passed = all(checks.values())
    require(close(summary["transformer_oof_auc"], transformer_auc), "Summary transformer AUC mismatch")
    require(close(summary["candidate_oof_auc"], candidate_auc), "Summary candidate AUC mismatch")
    promotion = summary["promotion_gate"]
    require(close(promotion["sparse_auc"], sparse_auc), "Summary sparse AUC mismatch")
    require(close(promotion["gain_over_strongest_base"], gain), "Summary gain mismatch")
    require(bool(promotion["passed"]) == passed, "Summary promotion decision mismatch")
    require(promotion["checks"] == {key: checks[key] for key in promotion["checks"]}, "Summary gate checks mismatch")

    recorded_records = summary["blend_contract"]["records"]
    require(len(recorded_records) == len(reconstructed_records) == 30, "Blend record count mismatch")
    for recorded, reconstructed in zip(recorded_records, reconstructed_records, strict=True):
        require(recorded["score_fold"] == reconstructed["score_fold"], "Blend score-fold mismatch")
        require(recorded["target"] == reconstructed["target"], "Blend target mismatch")
        require(
            close(recorded["selected_transformer_weight"], reconstructed["selected_transformer_weight"]),
            "Blend selected-weight mismatch",
        )
        require(len(recorded["candidate_scores"]) == len(reconstructed["candidate_scores"]), "Blend grid mismatch")
        for left, right in zip(recorded["candidate_scores"], reconstructed["candidate_scores"], strict=True):
            require(close(left["transformer_weight"], right["transformer_weight"]), "Blend grid weight mismatch")
            require(close(left["auc"], right["auc"]), "Blend candidate AUC mismatch")

    bundle_path = Path(summary["prediction_bundle"]["path"])
    submission_path = Path(summary["submission_withheld"]["path"])
    require(bundle_path.is_file(), "Final prediction bundle is missing")
    require(submission_path.is_file(), "Withheld submission is missing")
    require(sha256_file(bundle_path) == summary["prediction_bundle"]["sha256"], "Bundle SHA mismatch")
    require(sha256_file(submission_path) == summary["submission_withheld"]["sha256"], "Submission SHA mismatch")
    with np.load(bundle_path, allow_pickle=True) as archive:
        require(bool(np.array_equal(archive["truth"], truth)), "Final bundle truth mismatch")
        require(bool(np.array_equal(archive["fold"], fold_assignment)), "Final bundle fold mismatch")
        require(bool(np.allclose(archive["transformer_oof"], transformer_oof, rtol=0.0, atol=0.0)), "Final transformer OOF mismatch")
        require(bool(np.allclose(archive["transformer_test_by_fold"], transformer_test_by_fold, rtol=0.0, atol=0.0)), "Final transformer test mismatch")
        require(bool(np.allclose(archive["candidate_oof"], candidate_oof, rtol=0.0, atol=1e-15)), "Final candidate OOF mismatch")
        require(bool(np.allclose(archive["candidate_test"], candidate_test, rtol=0.0, atol=1e-15)), "Final candidate test mismatch")
        require(bool(np.array_equal(archive["transformer_write_counts"], transformer_counts)), "Transformer counts mismatch")
        require(bool(np.array_equal(archive["candidate_write_counts"], candidate_counts)), "Candidate counts mismatch")
        require(bool(np.array_equal(archive["train_id"].astype(str), train["id"].astype(str).to_numpy())), "Train ID order mismatch")
        require(bool(np.array_equal(archive["test_id"].astype(str), test["id"].astype(str).to_numpy())), "Test ID order mismatch")

    submission = pd.read_csv(submission_path)
    require(list(submission.columns) == list(sample.columns), "Submission columns mismatch")
    require(bool(np.array_equal(submission.iloc[:, 0].astype(str), sample.iloc[:, 0].astype(str))), "Submission ID order mismatch")
    require(
        bool(np.allclose(submission[list(TARGET_COLUMNS)].to_numpy(), candidate_test, rtol=0.0, atol=5e-16)),
        "Submission predictions mismatch",
    )

    report.update(
        {
            "status": "promotion_gate_passed" if passed else "promotion_gate_failed",
            "completed_folds": completed_folds,
            "transformer_exact_once": True,
            "candidate_exact_once": True,
            "metrics": {
                "sparse_auc": sparse_auc,
                "transformer_auc": transformer_auc,
                "candidate_auc": candidate_auc,
                "gain_over_strongest_base": gain,
                "sparse_per_label_auc": dict(zip(TARGET_COLUMNS, sparse_per_label, strict=True)),
                "transformer_per_label_auc": dict(zip(TARGET_COLUMNS, transformer_per_label, strict=True)),
                "candidate_per_label_auc": dict(zip(TARGET_COLUMNS, candidate_per_label, strict=True)),
            },
            "promotion_gate": {
                "threshold": threshold,
                "minimum_gain_over_strongest_base": minimum_gain,
                "checks": checks,
                "passed": passed,
            },
            "prediction_bundle": {
                "path": str(bundle_path),
                "sha256": sha256_file(bundle_path),
            },
            "submission_withheld": {
                "path": str(submission_path),
                "sha256": sha256_file(submission_path),
            },
            "reconstructed_blend_records": reconstructed_records,
            "full_contract_valid": True,
        }
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--sparse-bundle", type=Path, required=True)
    parser.add_argument("--frozen-plan", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = args.report_path or args.run_dir / "independent_verification.json"
    try:
        report = verify_run(
            run_dir=args.run_dir,
            public_dir=args.public_dir,
            sparse_bundle=args.sparse_bundle,
            frozen_plan=args.frozen_plan,
            require_complete=args.require_complete,
        )
    except Exception as exc:
        failure = {
            "schema": "evomind.jigsaw.transformer_independent_verification.v1",
            "created_at": now_iso(),
            "status": "verification_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json_atomic(report_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 2
    write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "promotion_gate_failed":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
