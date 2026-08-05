from __future__ import annotations

import numpy as np

from scripts import aggregate_ranzcr_crossrun_candidate as aggregate


def test_crossfit_blend_uses_only_other_folds() -> None:
    rng = np.random.default_rng(7)
    folds = np.repeat(np.arange(3), 40)
    truth = rng.integers(0, 2, size=(120, 2)).astype(np.float64)
    baseline = np.clip(0.25 + 0.50 * truth + rng.normal(0, 0.18, truth.shape), 0.01, 0.99)
    highres = np.clip(0.15 + 0.70 * truth + rng.normal(0, 0.12, truth.shape), 0.01, 0.99)
    blended, weights, records = aggregate.crossfit_blend(
        truth, baseline, highres, folds, grid_step=0.1
    )
    assert blended.shape == truth.shape
    assert weights.shape == (2,)
    assert len(records) == 3
    assert np.isfinite(blended).all()
    assert np.all((weights >= 0.0) & (weights <= 1.0))


def test_mean_column_auc_reports_every_label() -> None:
    truth = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.float64)
    probability = np.asarray(
        [[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]], dtype=np.float64
    )
    mean, labels = aggregate.mean_column_auc(truth, probability)
    assert mean == 1.0
    assert labels == [1.0, 1.0]
