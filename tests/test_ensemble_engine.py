"""Tests for the ensemble/stacking engine (gold-medal combination layer)."""
from __future__ import annotations

import numpy as np
import pytest

from research_agent_workstation.server.training.ensemble_engine import (
    BlendResult,
    blend,
    multi_seed_average,
    search_blend_weights,
    select_pseudo_labels,
    stack,
)


def test_blend_equal_weights_is_mean():
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(blend([a, b]), [1.0, 2.0, 3.0])


def test_blend_custom_weights_normalized():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    # weights [3,1] -> 0.75*b ... wait: 0.25*a + 0.75*b = 0.75
    np.testing.assert_allclose(blend([a, b], [1.0, 3.0]), [0.75, 0.75])


def test_blend_2d_probabilities():
    a = np.array([[0.2, 0.8], [0.6, 0.4]])
    b = np.array([[0.4, 0.6], [0.8, 0.2]])
    out = blend([a, b])
    np.testing.assert_allclose(out, [[0.3, 0.7], [0.7, 0.3]])


def test_blend_rejects_negative_weights():
    with pytest.raises(ValueError):
        blend([np.array([1.0]), np.array([2.0])], [-1.0, 2.0])


def test_blend_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        blend([np.array([1.0, 2.0]), np.array([1.0])])


def test_blend_empty_raises():
    with pytest.raises(ValueError):
        blend([])


def test_search_blend_weights_improves_or_matches():
    # Two models; the blend should score no worse than the best single model.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200).astype(float)
    good = np.clip(y + rng.normal(0, 0.2, 200), 0, 1)   # correlated with y
    noisy = rng.random(200)                              # uncorrelated

    def neg_mse(yt, yp):
        return -float(np.mean((yt - yp) ** 2))

    result = search_blend_weights([good, noisy], y, neg_mse, higher_is_better=True, step=0.1)
    assert isinstance(result, BlendResult)
    assert result.blended_score >= max(result.per_model_scores) - 1e-9
    # Should weight the good model more heavily.
    assert result.weights[0] >= result.weights[1]
    assert abs(sum(result.weights) - 1.0) < 1e-9


def test_stack_regression_returns_predictions():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 100)
    m1 = y + rng.normal(0, 0.3, 100)
    m2 = y + rng.normal(0, 0.5, 100)
    test1, test2 = rng.normal(0, 1, 20), rng.normal(0, 1, 20)
    oof_pred, test_pred = stack([m1, m2], y, [test1, test2], task_type="regression")
    assert oof_pred.shape == (100,)
    assert test_pred is not None and test_pred.shape == (20,)


def test_stack_classification_probabilities():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 120)
    m1 = np.clip(y + rng.normal(0, 0.2, 120), 0, 1)
    m2 = np.clip(y + rng.normal(0, 0.4, 120), 0, 1)
    oof_pred, _ = stack([m1, m2], y, task_type="classification", meta_model="logistic")
    assert oof_pred.shape == (120,)
    assert np.all((oof_pred >= 0) & (oof_pred <= 1))


def test_multi_seed_average_reduces_to_mean():
    # Deterministic fake trainer: prediction depends on seed in a known way.
    def fake_train(seed):
        return np.array([seed, seed * 2], dtype=float)

    out = multi_seed_average(fake_train, seeds=[1, 3])
    np.testing.assert_allclose(out, [2.0, 4.0])  # mean of [1,3] and [2,6]


def test_multi_seed_empty_raises():
    with pytest.raises(ValueError):
        multi_seed_average(lambda s: np.array([1.0]), seeds=[])


def test_pseudo_label_binary_selection():
    probs = np.array([0.99, 0.5, 0.02, 0.80])
    out = select_pseudo_labels(probs, threshold=0.95)
    assert out["n_selected"] == 2  # 0.99 and 0.02 (confidence 0.98)
    assert set(out["indices"].tolist()) == {0, 2}
    assert out["pseudo_labels"].tolist() == [1, 0]


def test_pseudo_label_multiclass_selection():
    probs = np.array([[0.97, 0.02, 0.01], [0.4, 0.35, 0.25], [0.1, 0.1, 0.8]])
    out = select_pseudo_labels(probs, threshold=0.9)
    assert out["n_selected"] == 1
    assert out["indices"].tolist() == [0]
    assert out["pseudo_labels"].tolist() == [0]


def test_pseudo_label_coverage():
    probs = np.array([0.99, 0.99, 0.5, 0.5])
    out = select_pseudo_labels(probs, threshold=0.9)
    assert out["coverage"] == 0.5
