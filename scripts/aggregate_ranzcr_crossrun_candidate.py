#!/usr/bin/env python3
"""Build a leakage-safe cross-run RANZCR ensemble candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGET_COLUMNS = (
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def normalize_probability(value: np.ndarray) -> np.ndarray:
    probability = np.asarray(value, dtype=np.float64)
    require(probability.ndim == 2, "RANZCR probability is not a matrix")
    require(np.isfinite(probability).all(), "RANZCR probability is non-finite")
    return np.clip(probability, 1e-7, 1.0 - 1e-7)


def mean_column_auc(truth: np.ndarray, probability: np.ndarray) -> tuple[float, list[float]]:
    scores = [
        float(roc_auc_score(truth[:, index], probability[:, index]))
        for index in range(truth.shape[1])
    ]
    return float(np.mean(scores)), scores


def select_weight(
    truth: np.ndarray,
    baseline: np.ndarray,
    highres: np.ndarray,
    grid: np.ndarray,
) -> tuple[float, float]:
    best_weight = 0.0
    best_score = -np.inf
    for highres_weight in grid:
        candidate = (1.0 - highres_weight) * baseline + highres_weight * highres
        score = float(roc_auc_score(truth, candidate))
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and highres_weight < best_weight
        ):
            best_weight = float(highres_weight)
            best_score = score
    return best_weight, best_score


def crossfit_blend(
    truth: np.ndarray,
    baseline: np.ndarray,
    highres: np.ndarray,
    folds: np.ndarray,
    *,
    grid_step: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    unique_folds = sorted(np.unique(folds).tolist())
    require(unique_folds == list(range(len(unique_folds))), "RANZCR folds are not contiguous")
    grid = np.arange(0.0, 1.0 + grid_step / 2.0, grid_step, dtype=np.float64)
    blended = np.full_like(baseline, np.nan, dtype=np.float64)
    weights = np.zeros((len(unique_folds), truth.shape[1]), dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in unique_folds:
        fit = folds != fold
        valid = folds == fold
        label_records = []
        for label in range(truth.shape[1]):
            weight, fit_auc = select_weight(
                truth[fit, label],
                baseline[fit, label],
                highres[fit, label],
                grid,
            )
            weights[fold, label] = weight
            blended[valid, label] = (
                (1.0 - weight) * baseline[valid, label]
                + weight * highres[valid, label]
            )
            label_records.append(
                {
                    "label": TARGET_COLUMNS[label],
                    "highres_weight": weight,
                    "fit_auc": fit_auc,
                    "valid_auc": float(
                        roc_auc_score(truth[valid, label], blended[valid, label])
                    ),
                }
            )
        records.append(
            {
                "fold": int(fold),
                "fit_rows": int(fit.sum()),
                "valid_rows": int(valid.sum()),
                "labels": label_records,
            }
        )
    require(np.isfinite(blended).all(), "RANZCR cross-fitted blend is incomplete")
    return blended, weights.mean(axis=0), records


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        required = {"truth", "selected_oof_probability", "selected_test_probability", "fold"}
        require(required <= set(bundle.files), f"RANZCR bundle is incomplete: {path}")
        return {name: np.asarray(bundle[name]) for name in required}


def aggregate(
    *,
    baseline_path: Path,
    highres_path: Path,
    public_dir: Path,
    output_dir: Path,
    grid_step: float,
    promotion_auc: float,
) -> dict[str, Any]:
    require(0.0 < grid_step <= 0.25, "RANZCR blend grid step is invalid")
    baseline = load_bundle(baseline_path)
    highres = load_bundle(highres_path)
    truth = np.asarray(baseline["truth"], dtype=np.float64)
    folds = np.asarray(baseline["fold"], dtype=np.int16)
    require(np.array_equal(truth, highres["truth"]), "RANZCR truth differs across runs")
    require(np.array_equal(folds, highres["fold"]), "RANZCR folds differ across runs")
    baseline_oof = normalize_probability(baseline["selected_oof_probability"])
    baseline_test = normalize_probability(baseline["selected_test_probability"])
    highres_oof = normalize_probability(highres["selected_oof_probability"])
    highres_test = normalize_probability(highres["selected_test_probability"])
    require(baseline_oof.shape == highres_oof.shape == truth.shape, "RANZCR OOF shapes differ")
    require(baseline_test.shape == highres_test.shape, "RANZCR test shapes differ")
    require(truth.shape[1] == len(TARGET_COLUMNS), "RANZCR label width differs")

    candidate_oof, deployment_weights, records = crossfit_blend(
        truth,
        baseline_oof,
        highres_oof,
        folds,
        grid_step=grid_step,
    )
    candidate_test = (
        (1.0 - deployment_weights[None, :]) * baseline_test
        + deployment_weights[None, :] * highres_test
    )
    baseline_auc, baseline_labels = mean_column_auc(truth, baseline_oof)
    highres_auc, highres_labels = mean_column_auc(truth, highres_oof)
    candidate_auc, candidate_labels = mean_column_auc(truth, candidate_oof)

    train = pd.read_csv(public_dir / "train.csv")
    sample = pd.read_csv(public_dir / "sample_submission.csv")
    require(list(sample.columns[1:]) == list(TARGET_COLUMNS), "RANZCR sample columns differ")
    require(len(train) == len(candidate_oof), "RANZCR train rows differ")
    require(len(sample) == len(candidate_test), "RANZCR test rows differ")
    submission = sample.copy()
    submission.loc[:, TARGET_COLUMNS] = candidate_test

    output_dir.mkdir(parents=True, exist_ok=False)
    bundle_path = output_dir / "ranzcr_crossrun_oof_and_test.npz"
    np.savez_compressed(
        bundle_path,
        truth=truth.astype(np.float32),
        fold=folds,
        baseline_oof=baseline_oof,
        highres_oof=highres_oof,
        candidate_oof=candidate_oof,
        baseline_test=baseline_test,
        highres_test=highres_test,
        candidate_test=candidate_test,
        deployment_highres_weights=deployment_weights,
    )
    submission_path = output_dir / "submission_withheld.csv"
    submission.to_csv(submission_path, index=False)
    passed = candidate_auc >= promotion_auc
    report = {
        "schema": "evomind.ranzcr.crossrun_candidate.v1",
        "created_at": now_iso(),
        "status": "promotion_gate_passed" if passed else "promotion_gate_failed",
        "competition_id": "ranzcr-clip-catheter-line-classification",
        "metric": "mean_columnwise_roc_auc",
        "baseline": {
            "path": str(baseline_path.resolve()),
            "sha256": sha256_file(baseline_path),
            "oof_auc": baseline_auc,
            "label_auc": dict(zip(TARGET_COLUMNS, baseline_labels, strict=True)),
        },
        "highres": {
            "path": str(highres_path.resolve()),
            "sha256": sha256_file(highres_path),
            "oof_auc": highres_auc,
            "label_auc": dict(zip(TARGET_COLUMNS, highres_labels, strict=True)),
        },
        "candidate": {
            "oof_auc": candidate_auc,
            "label_auc": dict(zip(TARGET_COLUMNS, candidate_labels, strict=True)),
            "deployment_highres_weights": dict(
                zip(TARGET_COLUMNS, deployment_weights.tolist(), strict=True)
            ),
            "crossfit_records": records,
            "promotion_auc": promotion_auc,
            "passed": passed,
        },
        "prediction_bundle": {
            "path": str(bundle_path.resolve()),
            "bytes": bundle_path.stat().st_size,
            "sha256": sha256_file(bundle_path),
        },
        "submission": {
            "path": str(submission_path.resolve()),
            "bytes": submission_path.stat().st_size,
            "sha256": sha256_file(submission_path),
        },
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "human_gate_preserved": True,
        "claim_boundary": "Cross-fitted public OOF evidence is not an official medal.",
    }
    report_path = output_dir / "aggregation_result.json"
    write_json_atomic(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--highres", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-step", type=float, default=0.025)
    parser.add_argument("--promotion-auc", type=float, default=0.9725)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = aggregate(
        baseline_path=args.baseline.resolve(),
        highres_path=args.highres.resolve(),
        public_dir=args.public_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        grid_step=args.grid_step,
        promotion_auc=args.promotion_auc,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
