"""Ensemble / stacking engine — the gold-medal-critical combination layer.

Pure numpy + scikit-learn so every method is unit-testable offline (no GPU, no
Kaggle). The GPU trainer supplies per-model OOF/test predictions; this engine
combines them: weighted blending, OOF stacking, multi-seed averaging, and
pseudo-label selection.

Design:
  * Deterministic given inputs (seeds are explicit).
  * Never mutates caller arrays.
  * Works for both regression (raw values) and classification (probabilities).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class BlendResult:
    weights: list[float]
    blended: np.ndarray
    per_model_scores: list[float] = field(default_factory=list)
    blended_score: Optional[float] = None


def _as_2d_stack(model_outputs: Sequence[np.ndarray]) -> np.ndarray:
    """Stack a list of (n,) or (n, k) arrays into (n_models, n, ...) safely."""
    if not model_outputs:
        raise ValueError("model_outputs is empty")
    arrays = [np.asarray(a, dtype=float) for a in model_outputs]
    shape = arrays[0].shape
    for i, a in enumerate(arrays):
        if a.shape != shape:
            raise ValueError(f"model_outputs[{i}] shape {a.shape} != {shape}")
    return np.stack(arrays, axis=0)


def blend(
    model_outputs: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Weighted average of model predictions.

    weights default to equal; they are normalized to sum to 1. Negative weights
    are rejected (they make blends unstable and are almost never intended).
    """
    stack = _as_2d_stack(model_outputs)
    n_models = stack.shape[0]
    if weights is None:
        w = np.full(n_models, 1.0 / n_models)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n_models,):
            raise ValueError(f"weights length {w.shape} != n_models {n_models}")
        if np.any(w < 0):
            raise ValueError("negative weights are not allowed")
        total = w.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        w = w / total
    # Broadcast weights over the leading model axis.
    extra_dims = (1,) * (stack.ndim - 1)
    return np.sum(stack * w.reshape((n_models, *extra_dims)), axis=0)


def search_blend_weights(
    oof_outputs: Sequence[np.ndarray],
    y_true: np.ndarray,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    higher_is_better: bool = True,
    step: float = 0.1,
) -> BlendResult:
    """Grid-search convex blend weights over the simplex to optimize OOF score.

    Practical for 2-3 models (the common Kaggle blend). ``score_fn(y_true, pred)``
    returns the metric; ``higher_is_better`` controls the direction.
    """
    stack = _as_2d_stack(oof_outputs)
    n_models = stack.shape[0]
    y_true = np.asarray(y_true)

    per_model = [float(score_fn(y_true, stack[i])) for i in range(n_models)]

    best_weights: Optional[np.ndarray] = None
    best_score: Optional[float] = None
    for combo in _simplex_grid(n_models, step):
        pred = blend(oof_outputs, combo)
        score = float(score_fn(y_true, pred))
        if best_score is None or (score > best_score if higher_is_better else score < best_score):
            best_score = score
            best_weights = combo
    assert best_weights is not None
    return BlendResult(
        weights=[float(x) for x in best_weights],
        blended=blend(oof_outputs, best_weights),
        per_model_scores=per_model,
        blended_score=best_score,
    )


def _simplex_grid(n: int, step: float):
    """Yield weight vectors of length n on the simplex (sum≈1) at a grid step."""
    levels = int(round(1.0 / step))

    def rec(remaining_levels: int, slots: int):
        if slots == 1:
            yield [remaining_levels]
            return
        for i in range(remaining_levels + 1):
            for rest in rec(remaining_levels - i, slots - 1):
                yield [i, *rest]

    for combo in rec(levels, n):
        yield np.array(combo, dtype=float) / levels


def stack(
    oof_predictions: Sequence[np.ndarray],
    y_true: np.ndarray,
    test_predictions: Optional[Sequence[np.ndarray]] = None,
    *,
    task_type: str = "classification",
    meta_model: str = "ridge",
):
    """OOF stacking: train a meta-model on out-of-fold predictions.

    Returns (meta_oof_pred, meta_test_pred). ``meta_test_pred`` is None when
    ``test_predictions`` is not supplied. Uses a linear meta-model (Ridge for
    regression, LogisticRegression for classification) — the standard, robust
    choice that resists overfitting the meta layer.
    """
    from sklearn.linear_model import LogisticRegression, Ridge

    X_meta = np.column_stack([np.asarray(p, dtype=float).ravel() if np.asarray(p).ndim == 1
                              else np.asarray(p, dtype=float) for p in oof_predictions])
    y_true = np.asarray(y_true)

    if task_type == "regression" or meta_model == "ridge":
        model = Ridge(alpha=1.0)
        model.fit(X_meta, y_true)
        oof_pred = model.predict(X_meta)
        test_pred = None
        if test_predictions is not None:
            X_test = np.column_stack([np.asarray(p, dtype=float) for p in test_predictions])
            test_pred = model.predict(X_test)
        return oof_pred, test_pred

    model = LogisticRegression(max_iter=1000)
    model.fit(X_meta, y_true)
    oof_pred = model.predict_proba(X_meta)
    if oof_pred.shape[1] == 2:
        oof_pred = oof_pred[:, 1]
    test_pred = None
    if test_predictions is not None:
        X_test = np.column_stack([np.asarray(p, dtype=float) for p in test_predictions])
        test_pred = model.predict_proba(X_test)
        if test_pred.shape[1] == 2:
            test_pred = test_pred[:, 1]
    return oof_pred, test_pred


def multi_seed_average(
    train_predict_fn: Callable[[int], np.ndarray],
    seeds: Sequence[int] = (42, 123, 456, 789, 1024),
) -> np.ndarray:
    """Average predictions across seeds to reduce variance (small-data gold trick)."""
    if not seeds:
        raise ValueError("seeds must be non-empty")
    preds = [np.asarray(train_predict_fn(int(s)), dtype=float) for s in seeds]
    return blend(preds)  # equal-weight average


def select_pseudo_labels(
    test_probabilities: np.ndarray,
    *,
    threshold: float = 0.95,
) -> dict:
    """Select high-confidence test rows for pseudo-labeling.

    For binary/multiclass probabilities, a row qualifies when its max class
    probability >= threshold. Returns indices, pseudo-labels, and coverage.
    """
    probs = np.asarray(test_probabilities, dtype=float)
    if probs.ndim == 1:
        # Binary given as P(class=1): confidence is distance from 0.5.
        confidence = np.maximum(probs, 1.0 - probs)
        labels = (probs >= 0.5).astype(int)
    else:
        confidence = probs.max(axis=1)
        labels = probs.argmax(axis=1)
    mask = confidence >= threshold
    idx = np.where(mask)[0]
    return {
        "indices": idx,
        "pseudo_labels": labels[idx],
        "n_selected": int(mask.sum()),
        "coverage": float(mask.mean()) if probs.shape[0] else 0.0,
        "threshold": threshold,
    }
