from __future__ import annotations

import numpy as np
import pytest

from scripts import aggregate_spooky_cross_run_xgb_confirmation as aggregate


def test_probability_aggregation_is_normalized_and_scores() -> None:
    first = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=np.float64)
    second = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]], dtype=np.float64)

    mean = aggregate.aggregate_probability_matrices([first, second])

    assert np.allclose(mean.sum(axis=1), 1.0)
    assert aggregate.multiclass_log_loss(np.array([0, 1]), mean) == pytest.approx(
        -np.log(0.75)
    )


def test_confirmation_gate_requires_exact_seeds_and_all_thresholds() -> None:
    passed = aggregate.confirmation_gate(
        model_seeds=[40, 41, 42],
        seed_scores=[0.2816, 0.2817, 0.2818],
        ensemble_score=0.2815,
        threshold=0.292,
    )
    failed = aggregate.confirmation_gate(
        model_seeds=[40, 42],
        seed_scores=[0.2816, 0.293],
        ensemble_score=0.2815,
        threshold=0.292,
    )

    assert passed["passed"] is True
    assert failed["passed"] is False


def test_probability_aggregation_rejects_shape_drift() -> None:
    valid = np.array([[0.8, 0.1, 0.1]], dtype=np.float64)
    other = np.array([[0.5, 0.4, 0.1], [0.1, 0.2, 0.7]], dtype=np.float64)

    with pytest.raises(ValueError, match="shapes differ"):
        aggregate.aggregate_probability_matrices([valid, other])
