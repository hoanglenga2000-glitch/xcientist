from __future__ import annotations

import numpy as np

from scripts import diagnose_dog_class_bias_calibration as diagnostic


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    folds = np.repeat(np.arange(5), 60)
    truth = np.tile(np.arange(3), 100)
    logits = rng.normal(0.0, 0.35, size=(len(truth), 3))
    logits[np.arange(len(truth)), truth] += 1.5
    logits[:, 0] += 0.25
    return diagnostic.softmax(logits), truth, folds


def test_cross_fit_probabilities_are_finite_normalized_and_complete() -> None:
    probability, truth, folds = fixture()
    calibrated, records = diagnostic.cross_fit_class_bias(
        probability, truth, folds, l2_grid=(1.0, 10.0, 100.0)
    )

    assert calibrated.shape == probability.shape
    assert np.isfinite(calibrated).all()
    assert (calibrated >= 0).all()
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert [record["outer_fold"] for record in records] == list(range(5))
    assert all(record["outer_fit_rows"] == 240 for record in records)
    assert all(record["outer_validation_rows"] == 60 for record in records)


def test_outer_predictions_do_not_depend_on_outer_fold_labels() -> None:
    probability, truth, folds = fixture()
    changed = truth.copy()
    changed[folds == 0] = (changed[folds == 0] + 1) % 3

    original, _ = diagnostic.cross_fit_class_bias(
        probability, truth, folds, l2_grid=(1.0, 10.0)
    )
    perturbed, _ = diagnostic.cross_fit_class_bias(
        probability, changed, folds, l2_grid=(1.0, 10.0)
    )

    assert np.allclose(
        original[folds == 0], perturbed[folds == 0], rtol=0.0, atol=1e-15
    )
