#!/usr/bin/env python3
"""Cross-fitted May-2022 OOF meta-stacker diagnostic.

The candidate grid is frozen in this file.  Every candidate is fitted without
the held outer fold, and no private labels, official grader, or submission path
is touched.  The output is diagnostic evidence for deciding whether further
post-processing is credible before training a new base model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

COMPONENT_KEYS = ("mlp_oof", "xgboost_oof", "catboost_oof")
TEST_COMPONENT_KEYS = ("mlp_test", "xgboost_test", "catboost_test")
REQUIRED_KEYS = {
    "id",
    "target",
    "fold_assignment",
    *COMPONENT_KEYS,
    "oof_probability",
    *TEST_COMPONENT_KEYS,
    "test_probability",
    "test_id",
}
AGGREGATE_PROMOTION_AUC = 0.9985
EVERY_FOLD_PROMOTION_AUC = 0.99818
POSTPROCESSING_CONTINUE_AUC = 0.99818

# Predeclared before inspecting candidate results.  C values are deliberately
# small because sklearn's logistic objective scales regularization against a
# very large (800k-row) fitting set.
CANDIDATE_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "ridge_raw_c1e-4", "space": "raw", "c": 1e-4},
    {"name": "ridge_raw_c1e-3", "space": "raw", "c": 1e-3},
    {"name": "ridge_raw_c1e-2", "space": "raw", "c": 1e-2},
    {"name": "ridge_logit_c1e-4", "space": "logit", "c": 1e-4},
    {"name": "ridge_logit_c1e-3", "space": "logit", "c": 1e-3},
    {"name": "ridge_logit_c1e-2", "space": "logit", "c": 1e-2},
    {"name": "ridge_rank_c1e-4", "space": "rank", "c": 1e-4},
    {"name": "ridge_poly2_logit_c1e-5", "space": "poly2_logit", "c": 1e-5},
    {"name": "ridge_poly2_logit_c1e-4", "space": "poly2_logit", "c": 1e-4},
    {"name": "ridge_poly2_logit_c1e-3", "space": "poly2_logit", "c": 1e-3},
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def candidate_grid_sha256(
    candidate_specs: Sequence[dict[str, Any]] = CANDIDATE_SPECS,
) -> str:
    canonical = json.dumps(
        list(candidate_specs), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_and_validate_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        missing = REQUIRED_KEYS - set(bundle.files)
        if missing:
            raise RuntimeError(f"May-2022 OOF bundle is missing keys: {sorted(missing)}")
        arrays = {key: np.asarray(bundle[key]) for key in REQUIRED_KEYS}
    row_count = len(arrays["target"])
    test_count = len(arrays["test_id"])
    if row_count < 100 or test_count < 1:
        raise RuntimeError("May-2022 OOF bundle has implausible row counts")
    one_dimensional = {
        "id",
        "target",
        "fold_assignment",
        *COMPONENT_KEYS,
        "oof_probability",
        *TEST_COMPONENT_KEYS,
        "test_probability",
        "test_id",
    }
    if any(arrays[key].ndim != 1 for key in one_dimensional):
        raise RuntimeError("May-2022 OOF bundle arrays must be one-dimensional")
    if any(len(arrays[key]) != row_count for key in (*COMPONENT_KEYS, "oof_probability", "id")):
        raise RuntimeError("May-2022 OOF arrays do not align with the labels")
    if any(len(arrays[key]) != test_count for key in (*TEST_COMPONENT_KEYS, "test_probability")):
        raise RuntimeError("May-2022 test arrays do not align with test_id")
    target = np.asarray(arrays["target"], dtype=np.int8)
    folds = np.asarray(arrays["fold_assignment"])
    if set(np.unique(target).tolist()) != {0, 1}:
        raise RuntimeError("May-2022 OOF target is not binary")
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2:
        raise RuntimeError("May-2022 OOF bundle needs at least two folds")
    for fold in unique_folds:
        if set(np.unique(target[folds == fold]).tolist()) != {0, 1}:
            raise RuntimeError("A May-2022 outer fold does not contain both classes")
    if len(np.unique(arrays["id"])) != row_count or len(np.unique(arrays["test_id"])) != test_count:
        raise RuntimeError("May-2022 train or test IDs are not unique")
    numeric_keys = (*COMPONENT_KEYS, "oof_probability", *TEST_COMPONENT_KEYS, "test_probability")
    if any(not np.isfinite(np.asarray(arrays[key], dtype=np.float64)).all() for key in numeric_keys):
        raise RuntimeError("May-2022 OOF bundle contains non-finite predictions")
    arrays["target"] = target
    arrays["fold_assignment"] = folds
    return arrays


def _rank_columns(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    ranked = np.empty_like(values, dtype=np.float64)
    denominator = len(values) + 1.0
    for column in range(values.shape[1]):
        ranked[:, column] = rankdata(values[:, column], method="average") / denominator
    return ranked


def transform_features(values: np.ndarray, space: str) -> np.ndarray:
    from sklearn.preprocessing import PolynomialFeatures

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(COMPONENT_KEYS):
        raise ValueError("May-2022 meta features must have exactly three columns")
    clipped = np.clip(matrix, 1e-6, 1.0 - 1e-6)
    if space == "raw":
        return clipped
    if space == "logit":
        return np.log(clipped / (1.0 - clipped))
    if space == "rank":
        return _rank_columns(clipped)
    if space == "poly2_logit":
        logits = np.log(clipped / (1.0 - clipped))
        return PolynomialFeatures(degree=2, include_bias=False).fit_transform(logits)
    raise ValueError(f"Unknown May-2022 meta feature space: {space}")


def _fit_predict(
    fit_values: np.ndarray,
    fit_target: np.ndarray,
    predict_values: np.ndarray,
    *,
    c_value: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            solver="lbfgs",
            max_iter=300,
            random_state=42,
        ),
    )
    with threadpool_limits(limits=1):
        model.fit(fit_values, fit_target)
        prediction = model.predict_proba(predict_values)[:, 1]
    classifier = model.named_steps["logisticregression"]
    return np.asarray(prediction, dtype=np.float64), {
        "coefficient": classifier.coef_.ravel().astype(float).tolist(),
        "intercept": classifier.intercept_.ravel().astype(float).tolist(),
        "iterations": classifier.n_iter_.ravel().astype(int).tolist(),
    }


def evaluate_candidate(
    values: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    spec: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.metrics import roc_auc_score

    prediction = np.full(len(target), np.nan, dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds).tolist()):
        validation = folds == fold
        fitting = ~validation
        fit_features = transform_features(values[fitting], str(spec["space"]))
        validation_features = transform_features(values[validation], str(spec["space"]))
        fold_prediction, fit_record = _fit_predict(
            fit_features,
            target[fitting],
            validation_features,
            c_value=float(spec["c"]),
        )
        prediction[validation] = fold_prediction
        fold_records.append(
            {
                "fold": int(fold),
                "fit_rows": int(fitting.sum()),
                "validation_rows": int(validation.sum()),
                "outer_auc": float(roc_auc_score(target[validation], fold_prediction)),
                **fit_record,
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("May-2022 meta candidate did not predict every OOF row")
    aggregate_auc = float(roc_auc_score(target, prediction))
    fold_auc = [float(record["outer_auc"]) for record in fold_records]
    return prediction, {
        "name": str(spec["name"]),
        "space": str(spec["space"]),
        "c": float(spec["c"]),
        "aggregate_oof_auc": aggregate_auc,
        "minimum_fold_auc": min(fold_auc),
        "maximum_fold_auc": max(fold_auc),
        "folds": fold_records,
    }


def evaluate_candidates(
    values: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    candidate_specs: Sequence[dict[str, Any]] = CANDIDATE_SPECS,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    predictions: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in candidate_specs:
        name = str(spec["name"])
        if name in seen:
            raise ValueError(f"Duplicate May-2022 meta candidate: {name}")
        seen.add(name)
        prediction, record = evaluate_candidate(values, target, folds, spec)
        predictions[name] = prediction
        records.append(record)
    records.sort(
        key=lambda item: (
            -float(item["aggregate_oof_auc"]),
            -float(item["minimum_fold_auc"]),
            str(item["name"]),
        )
    )
    best = records[0]
    return predictions[str(best["name"])], best, records


def fit_full_candidate_and_predict_test(
    oof_values: np.ndarray,
    target: np.ndarray,
    test_values: np.ndarray,
    spec: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    fit_features = transform_features(oof_values, str(spec["space"]))
    test_features = transform_features(test_values, str(spec["space"]))
    return _fit_predict(
        fit_features,
        target,
        test_features,
        c_value=float(spec["c"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "workspace"
        / "mlebench_plans"
        / "may2022_oof_meta_stacker_diagnostic_current.json",
    )
    parser.add_argument("--predictions", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from sklearn.metrics import roc_auc_score

    args = build_parser().parse_args(argv)
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    predictions_path = (
        args.predictions.resolve()
        if args.predictions
        else output_path.with_suffix(".npz")
    )
    arrays = load_and_validate_bundle(input_path)
    target = arrays["target"]
    folds = arrays["fold_assignment"]
    oof_values = np.column_stack([arrays[key] for key in COMPONENT_KEYS])
    test_values = np.column_stack([arrays[key] for key in TEST_COMPONENT_KEYS])
    best_oof, best, records = evaluate_candidates(oof_values, target, folds)
    best_spec = next(spec for spec in CANDIDATE_SPECS if spec["name"] == best["name"])
    diagnostic_test, full_fit = fit_full_candidate_and_predict_test(
        oof_values, target, test_values, best_spec
    )
    fold_values = sorted(np.unique(folds).tolist())
    baseline_fold_auc = {
        str(int(fold)): float(
            roc_auc_score(target[folds == fold], arrays["oof_probability"][folds == fold])
        )
        for fold in fold_values
    }
    component_auc = {
        key: float(roc_auc_score(target, arrays[key])) for key in COMPONENT_KEYS
    }
    aggregate_passed = float(best["aggregate_oof_auc"]) >= AGGREGATE_PROMOTION_AUC
    folds_passed = float(best["minimum_fold_auc"]) >= EVERY_FOLD_PROMOTION_AUC
    continue_postprocessing = (
        float(best["aggregate_oof_auc"]) >= POSTPROCESSING_CONTINUE_AUC
        and folds_passed
    )
    write_npz_atomic(
        predictions_path,
        id=arrays["id"],
        target=target,
        fold_assignment=folds,
        best_crossfit_oof_probability=best_oof.astype(np.float32),
        test_id=arrays["test_id"],
        diagnostic_test_probability=diagnostic_test.astype(np.float32),
    )
    report = {
        "schema": "evomind.mlebench.may2022_oof_meta_stacker_diagnostic.v1",
        "created_at": now_iso(),
        "input_bundle": str(input_path),
        "input_bundle_sha256": sha256_file(input_path),
        "predictions_bundle": str(predictions_path),
        "predictions_bundle_sha256": sha256_file(predictions_path),
        "row_count": int(len(target)),
        "test_row_count": int(len(arrays["test_id"])),
        "fold_values": [int(value) for value in fold_values],
        "component_oof_auc": component_auc,
        "existing_crossfit_oof_auc": float(
            roc_auc_score(target, arrays["oof_probability"])
        ),
        "existing_crossfit_fold_auc": baseline_fold_auc,
        "candidate_grid": list(CANDIDATE_SPECS),
        "candidate_grid_sha256": candidate_grid_sha256(),
        "candidate_count": len(CANDIDATE_SPECS),
        "candidate_records": records,
        "best_candidate": best,
        "best_gain_over_existing": float(best["aggregate_oof_auc"])
        - float(roc_auc_score(target, arrays["oof_probability"])),
        "full_oof_fit_for_withheld_test_prediction": full_fit,
        "promotion_gate": {
            "aggregate_oof_auc_threshold": AGGREGATE_PROMOTION_AUC,
            "every_fold_oof_auc_threshold": EVERY_FOLD_PROMOTION_AUC,
            "aggregate_passed": aggregate_passed,
            "all_folds_passed": folds_passed,
            "passed": bool(aggregate_passed and folds_passed),
        },
        "postprocessing_decision": {
            "continue_threshold": POSTPROCESSING_CONTINUE_AUC,
            "continue_pure_postprocessing": bool(continue_postprocessing),
            "recommended_next_route": (
                "additional_fixed_validation"
                if continue_postprocessing
                else "new_fold_safe_base_model"
            ),
        },
        "private_labels_used": False,
        "private_paths_requested": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "submission_withheld": True,
        "gpu_used": False,
        "process_signals_sent": 0,
        "claim_boundary": "Public OOF diagnostic evidence is not an official medal.",
    }
    write_json_atomic(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
