#!/usr/bin/env python3
"""Leakage-safe nested cross-fit class-bias diagnostic for Dog Breed OOF.

The diagnostic fits ``softmax(log(p) + b)`` on rows outside each outer fold.
The L2 penalty is selected on a deterministic inner fold that is also outside
the outer fold.  It never emits a submission or invokes an official grader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def fit_class_bias(
    probability: np.ndarray, truth: np.ndarray, l2: float
) -> tuple[np.ndarray, dict[str, Any]]:
    if l2 <= 0:
        raise ValueError("Class-bias L2 penalty must be positive")
    logits = np.log(np.clip(probability, 1e-12, 1.0))
    rows, classes = logits.shape
    class_frequency = np.bincount(truth, minlength=classes) / rows

    def objective(raw_bias: np.ndarray) -> tuple[float, np.ndarray]:
        bias = raw_bias - raw_bias.mean()
        calibrated = softmax(logits + bias)
        loss = -np.log(
            np.clip(calibrated[np.arange(rows), truth], 1e-15, 1.0)
        ).mean() + 0.5 * l2 * np.mean(bias * bias)
        gradient = calibrated.mean(axis=0) - class_frequency + (l2 / classes) * bias
        gradient -= gradient.mean()
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(classes, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 120, "ftol": 1e-11, "gtol": 1e-8},
    )
    bias = np.asarray(result.x, dtype=np.float64)
    bias -= bias.mean()
    return bias, {
        "success": bool(result.success),
        "status": int(result.status),
        "iterations": int(result.nit),
        "bias_l2_norm": float(np.linalg.norm(bias)),
    }


def cross_fit_class_bias(
    probability: np.ndarray,
    truth: np.ndarray,
    folds: np.ndarray,
    *,
    l2_grid: Sequence[float],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    probability = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    if probability.ndim != 2 or len(truth) != len(probability) or len(folds) != len(truth):
        raise ValueError("Dog class-bias input shapes are inconsistent")
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("Dog class-bias probabilities are invalid")
    probability /= probability.sum(axis=1, keepdims=True)
    unique_folds = np.unique(folds)
    if len(unique_folds) < 3 or not np.array_equal(
        unique_folds, np.arange(len(unique_folds))
    ):
        raise ValueError("Dog class-bias folds must be contiguous and contain at least three folds")
    if not l2_grid or any(value <= 0 for value in l2_grid):
        raise ValueError("Dog class-bias L2 grid must contain positive values")

    labels = np.arange(probability.shape[1])
    calibrated = np.empty_like(probability)
    records: list[dict[str, Any]] = []
    for outer_fold in unique_folds:
        outer_validation = folds == outer_fold
        outer_fit = ~outer_validation
        inner_fold = int((outer_fold + 1) % len(unique_folds))
        inner_validation = folds == inner_fold
        inner_fit = outer_fit & ~inner_validation
        candidates: list[dict[str, Any]] = []
        for l2 in l2_grid:
            bias, optimizer = fit_class_bias(probability[inner_fit], truth[inner_fit], l2)
            prediction = softmax(
                np.log(np.clip(probability[inner_validation], 1e-12, 1.0)) + bias
            )
            candidates.append(
                {
                    "l2": float(l2),
                    "inner_log_loss": float(
                        log_loss(truth[inner_validation], prediction, labels=labels)
                    ),
                    "optimizer": optimizer,
                }
            )
        selected = min(candidates, key=lambda item: (item["inner_log_loss"], item["l2"]))
        bias, optimizer = fit_class_bias(
            probability[outer_fit], truth[outer_fit], selected["l2"]
        )
        calibrated[outer_validation] = softmax(
            np.log(np.clip(probability[outer_validation], 1e-12, 1.0)) + bias
        )
        records.append(
            {
                "outer_fold": int(outer_fold),
                "outer_fit_rows": int(outer_fit.sum()),
                "outer_validation_rows": int(outer_validation.sum()),
                "inner_validation_fold": inner_fold,
                "selected_l2": selected["l2"],
                "inner_candidates": candidates,
                "outer_incumbent_log_loss": float(
                    log_loss(
                        truth[outer_validation],
                        probability[outer_validation],
                        labels=labels,
                    )
                ),
                "outer_candidate_log_loss": float(
                    log_loss(
                        truth[outer_validation],
                        calibrated[outer_validation],
                        labels=labels,
                    )
                ),
                "optimizer": optimizer,
            }
        )
    return calibrated, records


def run(bundle: Path, output: Path, l2_grid: Sequence[float]) -> dict[str, Any]:
    with np.load(bundle, allow_pickle=False) as arrays:
        required = {"truth", "fold", "selected_oof_probability"}
        if not required <= set(arrays.files):
            raise RuntimeError("Dog OOF bundle is missing class-bias inputs")
        truth = np.asarray(arrays["truth"], dtype=np.int64)
        folds = np.asarray(arrays["fold"], dtype=np.int64)
        incumbent = np.asarray(arrays["selected_oof_probability"], dtype=np.float64)
    candidate, fold_records = cross_fit_class_bias(
        incumbent, truth, folds, l2_grid=l2_grid
    )
    labels = np.arange(incumbent.shape[1])
    incumbent_loss = float(log_loss(truth, incumbent, labels=labels))
    candidate_loss = float(log_loss(truth, candidate, labels=labels))
    target = 0.04
    source_path = Path(__file__).resolve()
    payload = {
        "schema": "evomind.dog_breed.class_bias_diagnostic.v1",
        "created_at": now_iso(),
        "status": "completed",
        "source": {
            "path": str(source_path),
            "sha256": sha256(source_path),
        },
        "input": {
            "path": str(bundle.resolve()),
            "sha256": sha256(bundle),
            "allow_pickle": False,
            "rows": len(truth),
            "classes": incumbent.shape[1],
            "folds": int(len(np.unique(folds))),
        },
        "method": {
            "transform": "softmax(log(p)+centered_class_bias)",
            "l2_grid": [float(value) for value in l2_grid],
            "outer_fold_labels_used_for_outer_prediction": False,
            "inner_penalty_selection_excludes_outer_fold": True,
        },
        "incumbent_oof_log_loss": incumbent_loss,
        "candidate_oof_log_loss": candidate_loss,
        "improvement": incumbent_loss - candidate_loss,
        "target_oof_log_loss": target,
        "target_passed": candidate_loss <= target,
        "fold_records": fold_records,
        "decision": (
            "RETAIN_AS_MINOR_NON_PROMOTABLE_IMPROVEMENT"
            if candidate_loss < incumbent_loss
            else "RETAIN_INCUMBENT"
        ),
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "human_gate_preserved": True,
        "claim_boundary": "Public OOF diagnostic only; not an official medal or rank.",
    }
    write_json_atomic(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--l2-grid", default="1,10,100")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    l2_grid = tuple(float(value) for value in args.l2_grid.split(","))
    payload = run(args.bundle.resolve(), args.output.resolve(), l2_grid)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
