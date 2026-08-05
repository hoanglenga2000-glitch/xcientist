#!/usr/bin/env python3
"""Evaluate leakage-safe Jigsaw meta-features on an immutable OOF artifact.

This is a diagnostic sweep, not an official grade and not a promotion result.
Every held-out row is predicted by a stacker trained on other outer folds.
Hyperparameter selection across the reported candidates remains diagnostic; a
chosen candidate must be rerun through a nested or frozen validation contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


TARGET_COLUMNS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)
THRESHOLD = 0.987


def fractional_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranked = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranked[order[start:end]] = (start + end - 1) / 2.0
        start = end
    if len(values) > 1:
        ranked /= len(values) - 1
    return ranked


def fold_rank_matrix(values: np.ndarray, fold: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    folds = np.asarray(fold, dtype=np.int16).reshape(-1)
    if values.ndim != 2 or len(values) != len(folds):
        raise ValueError("values and fold do not share a row contract")
    ranked = np.empty_like(values, dtype=np.float64)
    for current in np.unique(folds):
        held_out = folds == current
        for column in range(values.shape[1]):
            ranked[held_out, column] = fractional_rank(values[held_out, column])
    return ranked


def logit_matrix(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def build_features(
    oof_word: np.ndarray,
    oof_char: np.ndarray,
    oof_blend: np.ndarray,
    fold: np.ndarray,
    feature_set: str,
) -> np.ndarray:
    bases = np.concatenate([oof_word, oof_char], axis=1)
    ranked_bases = fold_rank_matrix(bases, fold)
    ranked_blend = fold_rank_matrix(oof_blend, fold)
    if feature_set == "rank12":
        return ranked_bases
    if feature_set == "rank18":
        return np.concatenate([ranked_bases, ranked_blend], axis=1)
    if feature_set == "logit12":
        return logit_matrix(bases)
    if feature_set == "hybrid30":
        return np.concatenate([ranked_bases, ranked_blend, logit_matrix(bases)], axis=1)
    raise ValueError(f"Unsupported feature set: {feature_set}")


def mean_columnwise_auc(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, list[float]]:
    scores = [
        float(roc_auc_score(truth[:, index], prediction[:, index]))
        for index in range(truth.shape[1])
    ]
    return float(np.mean(scores)), scores


def cross_fit_logistic_stacker(
    features: np.ndarray,
    truth: np.ndarray,
    fold: np.ndarray,
    *,
    c_value: float,
    class_weight: str | None,
    max_iter: int,
    seed: int,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int8)
    folds = np.asarray(fold, dtype=np.int16).reshape(-1)
    if features.ndim != 2 or truth.ndim != 2 or len(features) != len(truth) or len(folds) != len(truth):
        raise ValueError("stacker arrays do not share a row contract")
    if not np.isfinite(features).all():
        raise ValueError("stacker features must be finite")
    prediction = np.zeros_like(truth, dtype=np.float64)
    for current in np.unique(folds):
        fit = folds != current
        held_out = folds == current
        for label in range(truth.shape[1]):
            model = LogisticRegression(
                C=float(c_value),
                class_weight=class_weight,
                max_iter=int(max_iter),
                random_state=int(seed + 1009 * int(current) + label),
                solver="liblinear",
            )
            model.fit(features[fit], truth[fit, label])
            prediction[held_out, label] = model.predict_proba(features[held_out])[:, 1]
    return prediction


def nested_cross_fit_logistic_stacker(
    features: np.ndarray,
    truth: np.ndarray,
    fold: np.ndarray,
    *,
    c_values: Iterable[float],
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Select C inside each outer meta-training partition and score its held fold."""

    features = np.asarray(features, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int8)
    folds = np.asarray(fold, dtype=np.int16).reshape(-1)
    grid = tuple(float(value) for value in c_values)
    unique_folds = sorted(int(value) for value in np.unique(folds))
    if unique_folds != list(range(len(unique_folds))) or len(unique_folds) < 3:
        raise ValueError("nested stacker requires at least three contiguous zero-based folds")
    if features.ndim != 2 or truth.ndim != 2 or len(features) != len(truth) or len(folds) != len(truth):
        raise ValueError("nested stacker arrays do not share a row contract")
    if not grid or any(not math.isfinite(value) or value <= 0 for value in grid):
        raise ValueError("nested stacker C grid must be finite and positive")
    prediction = np.full_like(truth, np.nan, dtype=np.float64)
    write_counts = np.zeros_like(truth, dtype=np.uint8)
    records: list[dict] = []
    for outer_fold in unique_folds:
        outer_fit = folds != outer_fold
        outer_score = folds == outer_fold
        inner_folds = [value for value in unique_folds if value != outer_fold]
        for label in range(truth.shape[1]):
            candidates: list[dict] = []
            for candidate_index, c_value in enumerate(grid):
                inner_prediction = np.full(len(truth), np.nan, dtype=np.float64)
                for inner_fold in inner_folds:
                    inner_score = outer_fit & (folds == inner_fold)
                    inner_fit = outer_fit & (folds != inner_fold)
                    model = LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=max_iter,
                        random_state=seed + 100_000 * outer_fold + 1_000 * inner_fold + 10 * label + candidate_index,
                        solver="liblinear",
                    )
                    model.fit(features[inner_fit], truth[inner_fit, label])
                    inner_prediction[inner_score] = model.predict_proba(features[inner_score])[:, 1]
                if not np.isfinite(inner_prediction[outer_fit]).all():
                    raise RuntimeError("nested stacker inner OOF coverage is incomplete")
                candidates.append(
                    {
                        "c_value": c_value,
                        "inner_auc": float(
                            roc_auc_score(
                                truth[outer_fit, label], inner_prediction[outer_fit]
                            )
                        ),
                    }
                )
            selected = max(
                candidates,
                key=lambda item: (
                    item["inner_auc"],
                    -abs(math.log(item["c_value"] / 0.1)),
                    -item["c_value"],
                ),
            )
            final = LogisticRegression(
                C=float(selected["c_value"]),
                class_weight="balanced",
                max_iter=max_iter,
                random_state=seed + 1_000_000 + 10_000 * outer_fold + label,
                solver="liblinear",
            )
            final.fit(features[outer_fit], truth[outer_fit, label])
            prediction[outer_score, label] = final.predict_proba(features[outer_score])[:, 1]
            write_counts[outer_score, label] += 1
            records.append(
                {
                    "score_fold": outer_fold,
                    "target": TARGET_COLUMNS[label],
                    "fit_folds": inner_folds,
                    "selected_c": float(selected["c_value"]),
                    "selected_inner_auc": float(selected["inner_auc"]),
                    "candidate_scores": candidates,
                }
            )
    if not np.all(write_counts == 1) or not np.isfinite(prediction).all():
        raise RuntimeError("nested stacker OOF prediction was not written exactly once")
    return prediction, write_counts, records


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    parsed = [float(item) for item in parse_csv(value)]
    if not parsed or any(not math.isfinite(item) or item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("C values must be finite and positive")
    return parsed


def evaluate(
    artifact: Path,
    *,
    feature_sets: Iterable[str],
    c_values: Iterable[float],
    class_weights: Iterable[str],
    max_iter: int,
    seed: int,
) -> dict:
    started = time.perf_counter()
    with np.load(artifact, allow_pickle=False) as arrays:
        required = {"oof_word", "oof_char", "oof_blend", "truth", "fold"}
        if not required <= set(arrays.files):
            raise ValueError(f"OOF artifact is missing arrays: {sorted(required - set(arrays.files))}")
        oof_word = np.asarray(arrays["oof_word"], dtype=np.float64)
        oof_char = np.asarray(arrays["oof_char"], dtype=np.float64)
        oof_blend = np.asarray(arrays["oof_blend"], dtype=np.float64)
        truth = np.asarray(arrays["truth"], dtype=np.int8)
        fold = np.asarray(arrays["fold"], dtype=np.int16)
    if oof_word.shape != oof_char.shape or oof_word.shape != oof_blend.shape or oof_word.shape != truth.shape:
        raise ValueError("OOF arrays do not share the six-label contract")
    baseline_mean, baseline_labels = mean_columnwise_auc(truth, oof_blend)
    candidates: list[dict] = []
    feature_cache: dict[str, np.ndarray] = {}
    for feature_set in feature_sets:
        feature_cache[feature_set] = build_features(oof_word, oof_char, oof_blend, fold, feature_set)
        for c_value in c_values:
            for weight_name in class_weights:
                class_weight = None if weight_name == "none" else weight_name
                candidate_started = time.perf_counter()
                prediction = cross_fit_logistic_stacker(
                    feature_cache[feature_set],
                    truth,
                    fold,
                    c_value=c_value,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    seed=seed,
                )
                score, per_label = mean_columnwise_auc(truth, prediction)
                candidates.append(
                    {
                        "feature_set": feature_set,
                        "feature_count": int(feature_cache[feature_set].shape[1]),
                        "c_value": float(c_value),
                        "class_weight": weight_name,
                        "mean_columnwise_auc": score,
                        "per_label_auc": dict(zip(TARGET_COLUMNS, per_label, strict=True)),
                        "gain_over_existing_blend": score - baseline_mean,
                        "crosses_threshold": score >= THRESHOLD,
                        "runtime_seconds": time.perf_counter() - candidate_started,
                    }
                )
    candidates.sort(key=lambda item: item["mean_columnwise_auc"], reverse=True)
    return {
        "schema": "evomind.mlebench_lite.jigsaw_oof_stacking_diagnostic.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": str(artifact.resolve()),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "rows": int(len(truth)),
        "labels": list(TARGET_COLUMNS),
        "folds": int(len(np.unique(fold))),
        "existing_blend": {
            "mean_columnwise_auc": baseline_mean,
            "per_label_auc": dict(zip(TARGET_COLUMNS, baseline_labels, strict=True)),
        },
        "promotion_threshold": THRESHOLD,
        "candidates": candidates,
        "best_candidate": candidates[0] if candidates else None,
        "claim_boundary": (
            "Diagnostic cross-fitted OOF only; candidate selection is not nested and this is not "
            "an official private grade or medal."
        ),
        "runtime_seconds": time.perf_counter() - started,
    }


def evaluate_nested(
    artifact: Path,
    *,
    c_values: Iterable[float],
    max_iter: int,
    seed: int,
) -> dict:
    """Run the frozen Hybrid30 feature set under the production nested-C contract."""

    started = time.perf_counter()
    with np.load(artifact, allow_pickle=False) as arrays:
        required = {"oof_word", "oof_char", "oof_blend", "truth", "fold"}
        if not required <= set(arrays.files):
            raise ValueError(f"OOF artifact is missing arrays: {sorted(required - set(arrays.files))}")
        word = np.asarray(arrays["oof_word"], dtype=np.float64)
        char = np.asarray(arrays["oof_char"], dtype=np.float64)
        blend = np.asarray(arrays["oof_blend"], dtype=np.float64)
        truth = np.asarray(arrays["truth"], dtype=np.int8)
        fold = np.asarray(arrays["fold"], dtype=np.int16)
    if word.shape != char.shape or word.shape != blend.shape or word.shape != truth.shape:
        raise ValueError("OOF arrays do not share the six-label contract")
    features = build_features(word, char, blend, fold, "hybrid30")
    prediction, write_counts, records = nested_cross_fit_logistic_stacker(
        features,
        truth,
        fold,
        c_values=c_values,
        max_iter=max_iter,
        seed=seed,
    )
    baseline_mean, baseline_labels = mean_columnwise_auc(truth, blend)
    candidate_mean, candidate_labels = mean_columnwise_auc(truth, prediction)
    return {
        "schema": "evomind.mlebench_lite.jigsaw_nested_stacker_validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": str(artifact.resolve()),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "rows": int(len(truth)),
        "labels": list(TARGET_COLUMNS),
        "folds": int(len(np.unique(fold))),
        "feature_set": "hybrid30",
        "feature_count": int(features.shape[1]),
        "c_grid": [float(value) for value in c_values],
        "class_weight": "balanced",
        "existing_blend": {
            "mean_columnwise_auc": baseline_mean,
            "per_label_auc": dict(zip(TARGET_COLUMNS, baseline_labels, strict=True)),
        },
        "nested_candidate": {
            "mean_columnwise_auc": candidate_mean,
            "per_label_auc": dict(zip(TARGET_COLUMNS, candidate_labels, strict=True)),
            "gain_over_existing_blend": candidate_mean - baseline_mean,
            "crosses_threshold": candidate_mean >= THRESHOLD,
        },
        "exact_once_oof": bool(np.all(write_counts == 1)),
        "selected_c_records": records,
        "promotion_threshold": THRESHOLD,
        "claim_boundary": (
            "Nested-C cross-fitted public OOF confirmation with a previously frozen feature set; "
            "this is not an official private grade or medal."
        ),
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-sets", default="rank12,rank18,logit12,hybrid30")
    parser.add_argument("--c-values", type=parse_floats, default=[0.03, 0.1, 0.3])
    parser.add_argument("--class-weights", default="none,balanced")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--nested-production-contract",
        action="store_true",
        help="Run frozen Hybrid30 with per-outer-fold nested C selection.",
    )
    args = parser.parse_args()
    feature_sets = parse_csv(args.feature_sets)
    class_weights = parse_csv(args.class_weights)
    if any(value not in {"none", "balanced"} for value in class_weights):
        raise SystemExit("--class-weights supports only none,balanced")
    if args.nested_production_contract:
        report = evaluate_nested(
            args.artifact,
            c_values=args.c_values,
            max_iter=args.max_iter,
            seed=args.seed,
        )
    else:
        report = evaluate(
            args.artifact,
            feature_sets=feature_sets,
            c_values=args.c_values,
            class_weights=class_weights,
            max_iter=args.max_iter,
            seed=args.seed,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "existing_blend": report["existing_blend"]["mean_columnwise_auc"],
                "best_candidate": report.get("best_candidate") or report.get("nested_candidate"),
                "runtime_seconds": report["runtime_seconds"],
                "claim_boundary": report["claim_boundary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
