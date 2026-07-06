"""Regression tests for the GPU trainer's label-space handling.

These lock in the dec-2021 fix: integer class labels (e.g. Cover_Type 1-7) must be
LabelEncoded to 0..K-1 so that argmax(predict_proba) lines up, and multi-class test
predictions must average PROBABILITIES across folds (not per-fold argmax indices).

The trainer lives in ``scripts/`` and imports catboost lazily, so we load it by path
and exercise the real shipped helpers with a tiny fake model.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

_TRAINER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gpu_batch_trainer_v1.py"


def _load_trainer():
    spec = importlib.util.spec_from_file_location("gpu_batch_trainer_v1", _TRAINER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


trainer = _load_trainer()


class _FakeModel:
    """Mimics a fitted CatBoost model's predict_proba/classes_ contract."""

    def __init__(self, classes, proba):
        self.classes_ = np.asarray(classes)
        self._proba = np.asarray(proba, dtype=float)

    def predict_proba(self, _pool):
        return self._proba


def test_full_width_proba_passthrough_when_already_full():
    # classes 0..2 in order -> returned unchanged.
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    model = _FakeModel([0, 1, 2], proba)
    out = trainer._full_width_proba(model, None, n_classes=3)
    np.testing.assert_allclose(out, proba)


def test_full_width_proba_places_columns_by_class_id():
    # A fold that only saw classes {0, 2} (missing class 1). Columns must land at 0 and 2.
    proba = np.array([[0.6, 0.4], [0.3, 0.7]])  # cols correspond to classes [0, 2]
    model = _FakeModel([0, 2], proba)
    out = trainer._full_width_proba(model, None, n_classes=3)
    expected = np.array([[0.6, 0.0, 0.4], [0.3, 0.0, 0.7]])
    np.testing.assert_allclose(out, expected)
    # argmax still valid across the full class space.
    assert out.shape == (2, 3)


def test_full_width_proba_handles_float_class_labels():
    proba = np.array([[0.1, 0.9]])
    model = _FakeModel([0.0, 1.0], proba)  # some backends return float classes_
    out = trainer._full_width_proba(model, None, n_classes=2)
    np.testing.assert_allclose(out, proba)


def test_encode_argmax_decode_roundtrip_fixes_off_by_one():
    # Reproduce dec-2021: integer labels 1..7, NOT object dtype -> old code skipped encoding.
    rng = np.random.RandomState(0)
    y_raw = pd.Series(rng.randint(1, 8, size=500), name="Cover_Type")
    assert y_raw.dtype != object  # the exact condition that skipped the old encoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    # A perfect model puts all mass on the true encoded class, columns ordered 0..6.
    proba = np.zeros((len(y_enc), 7))
    proba[np.arange(len(y_enc)), y_enc] = 1.0
    argmax_enc = np.argmax(proba, axis=1)

    # Old (buggy) comparison: 0-indexed argmax vs raw 1..7 labels -> ~0 accuracy.
    old_acc = float((argmax_enc == y_raw.values).mean())
    assert old_acc < 0.05

    # New: score in encoded space is perfect, and decode round-trips exactly.
    new_acc = float((argmax_enc == y_enc).mean())
    assert new_acc == pytest.approx(1.0)
    decoded = le.inverse_transform(argmax_enc)
    assert np.array_equal(decoded, y_raw.values)


def test_binary_labelencoder_is_identity_on_zero_one():
    # The always-encode change must be a no-op for existing binary 0/1 targets.
    y = pd.Series([0, 1, 1, 0, 1], name="target")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    np.testing.assert_array_equal(y_enc, y.values)
    np.testing.assert_array_equal(le.inverse_transform(y_enc), y.values)


def test_multiclass_probability_averaging_beats_index_averaging():
    # Two folds, 3 classes. Averaging argmax indices (old bug) can pick a class that
    # neither fold's probabilities support; averaging probabilities is correct.
    fold1 = np.array([[0.45, 0.10, 0.45]])  # argmax -> 0 (ties to 0)
    fold2 = np.array([[0.40, 0.15, 0.45]])  # argmax -> 2
    # Old bug: mean of indices (0, 2) = 1.0 -> rounds to class 1, which is the LEAST likely.
    old_index_avg = round((np.argmax(fold1) + np.argmax(fold2)) / 2)
    assert old_index_avg == 1
    # New: average probabilities then argmax -> class 2 (highest combined mass).
    prob_avg_argmax = int(np.argmax((fold1 + fold2) / 2, axis=1)[0])
    assert prob_avg_argmax == 2
    assert prob_avg_argmax != old_index_avg


def test_compute_metric_accuracy_multiclass_encoded_space():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4], dtype=float)  # argmax indices as floats
    acc = trainer.compute_metric(y_true, y_pred, "accuracy", "classification")
    assert acc == pytest.approx(1.0)
