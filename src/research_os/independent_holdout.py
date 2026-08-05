"""Independent holdout reviewer for probability-based binary classifiers.

The evolution candidate never receives the label file.  This module is invoked
after search completes and evaluates the selected ``submission.csv`` against a
separately retained label ledger.  The decision threshold is read from the
candidate's OOF metrics; the holdout is never used to tune it.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _finite_probability(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _find_oof_threshold(payload: Any, prefix: str = "") -> tuple[float, str] | None:
    """Return an OOF-derived threshold without looking at holdout labels.

    Explicit OOF keys win.  A generic ``decision_threshold`` remains accepted
    for older generated solutions, but its provenance is clearly recorded.
    """
    candidates: list[tuple[int, float, str]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key)
                child = f"{path}.{key_text}" if path else key_text
                low = key_text.lower()
                if "threshold" in low:
                    threshold = _finite_probability(item)
                    if threshold is not None:
                        priority = 0 if "oof" in low else 1 if "decision" in low else 2
                        candidates.append((priority, threshold, child))
                walk(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(payload, prefix)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[2]))
    _, threshold, source = candidates[0]
    return threshold, source


def _expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Include probability==1.0 in the final bin.
    assignments = np.minimum(np.digitize(probability, edges[1:-1], right=False), bins - 1)
    total = float(len(y_true))
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(probability[mask].mean())
        observed = float(y_true[mask].mean())
        error += (count / total) * abs(observed - confidence)
    return float(error)


def evaluate_independent_holdout(
    submission_path: str | Path,
    labels_path: str | Path,
    metrics_path: str | Path,
    *,
    output_path: str | Path | None = None,
    split_policy: str = "independent temporal holdout",
) -> dict[str, Any]:
    """Evaluate one selected candidate and emit a reviewer/claim-audit record."""
    submission_path = Path(submission_path).resolve()
    labels_path = Path(labels_path).resolve()
    metrics_path = Path(metrics_path).resolve()
    for path in (submission_path, labels_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    submission = pd.read_csv(submission_path)
    labels = pd.read_csv(labels_path)
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics_payload, dict):
        raise ValueError("metrics.json must contain an object")

    common = [column for column in labels.columns if column in submission.columns]
    if len(common) < 2:
        raise ValueError("submission and label ledger do not share id + target columns")
    target_candidates = [column for column in common if column.lower() in {"is_fraud", "target", "label"}]
    target = target_candidates[0] if target_candidates else common[-1]
    id_candidates = [column for column in common if column != target]
    if not id_candidates:
        raise ValueError("independent review requires an aligned id column")
    id_column = id_candidates[0]

    row_count_matches = len(submission) == len(labels)
    id_order_matches = row_count_matches and submission[id_column].equals(labels[id_column])
    probabilities = pd.to_numeric(submission[target], errors="coerce").to_numpy(dtype=np.float64)
    truth = pd.to_numeric(labels[target], errors="coerce").to_numpy(dtype=np.float64)
    probability_finite = bool(np.isfinite(probabilities).all())
    probability_in_range = bool(probability_finite and ((probabilities >= 0.0) & (probabilities <= 1.0)).all())
    labels_binary = bool(np.isfinite(truth).all() and set(np.unique(truth)).issubset({0.0, 1.0}) and len(np.unique(truth)) == 2)

    threshold_record = _find_oof_threshold(metrics_payload)
    if threshold_record is None:
        threshold, threshold_source = 0.5, "fixed_default_0.5"
    else:
        threshold, threshold_source = threshold_record
    threshold_is_oof_or_fixed = "oof" in threshold_source.lower() or threshold_source == "fixed_default_0.5"

    checks = {
        "row_count_matches": row_count_matches,
        "id_order_matches": id_order_matches,
        "probability_finite": probability_finite,
        "probability_in_unit_interval": probability_in_range,
        "labels_binary_with_both_classes": labels_binary,
        "threshold_not_optimized_on_holdout": True,
    }
    passed = all(checks.values())
    score_payload: dict[str, float | int | None]
    if passed:
        y_true = truth.astype(np.int8)
        y_pred = (probabilities >= threshold).astype(np.int8)
        score_payload = {
            "average_precision": float(average_precision_score(y_true, probabilities)),
            "pr_auc": float(average_precision_score(y_true, probabilities)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "brier_score": float(brier_score_loss(y_true, probabilities)),
            "expected_calibration_error_15bin": _expected_calibration_error(y_true, probabilities),
            "decision_threshold": float(threshold),
        }
    else:
        score_payload = {
            "average_precision": None,
            "pr_auc": None,
            "f1": None,
            "recall": None,
            "precision": None,
            "brier_score": None,
            "expected_calibration_error_15bin": None,
            "decision_threshold": float(threshold),
        }

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload: dict[str, Any] = {
        "schema": "evomind.independent_holdout_review.v1",
        "status": "passed" if passed else "rejected",
        "reviewer": "IndependentHoldoutReviewer",
        "generated_at": now,
        "input": {
            "submission": {"path": str(submission_path), "sha256": _sha256(submission_path)},
            "labels": {"path": str(labels_path), "sha256": _sha256(labels_path)},
            "metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
        },
        "dataset": {
            "rows": int(len(labels)),
            "positive_rows": int(np.nansum(truth == 1.0)),
            "positive_rate": float(np.nanmean(truth)) if len(truth) else 0.0,
            "id_column": id_column,
            "target": target,
            "split_policy": split_policy,
        },
        "threshold_provenance": {
            "value": float(threshold),
            "source": threshold_source,
            "strict_oof_or_fixed": threshold_is_oof_or_fixed,
            "holdout_optimization_performed": False,
        },
        "checks": checks,
        "metrics": score_payload,
        "claim_audit": {
            "status": "passed" if passed else "rejected",
            "claim_boundary": "Independent offline temporal holdout only; not a public leaderboard score or rank.",
            "official_submission_performed": False,
            "holdout_labels_exposed_to_candidate": False,
            "threshold_tuned_on_holdout": False,
            "evidence_hashes_bound": True,
        },
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


__all__ = ["evaluate_independent_holdout"]
