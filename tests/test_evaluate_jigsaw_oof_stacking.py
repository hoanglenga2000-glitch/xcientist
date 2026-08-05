from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_jigsaw_oof_stacking",
    ROOT / "scripts" / "evaluate_jigsaw_oof_stacking.py",
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_fold_rank_matrix_normalizes_each_fold_independently() -> None:
    values = np.array([[1.0], [2.0], [100.0], [200.0]])
    fold = np.array([0, 0, 1, 1])
    ranked = module.fold_rank_matrix(values, fold)
    assert ranked[:, 0].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_cross_fit_stacker_never_fits_held_out_rows(monkeypatch) -> None:
    rows = 24
    fold = np.repeat(np.arange(3), rows // 3)
    truth = np.zeros((rows, 2), dtype=np.int8)
    truth[::2, 0] = 1
    truth[1::2, 1] = 1
    features = np.column_stack([np.linspace(0, 1, rows), np.linspace(1, 0, rows)])
    seen_fit_sizes: list[int] = []
    real = module.LogisticRegression

    class RecordingLogistic(real):
        def fit(self, x, y):
            seen_fit_sizes.append(len(x))
            return super().fit(x, y)

    monkeypatch.setattr(module, "LogisticRegression", RecordingLogistic)
    prediction = module.cross_fit_logistic_stacker(
        features,
        truth,
        fold,
        c_value=0.1,
        class_weight=None,
        max_iter=100,
        seed=42,
    )
    assert prediction.shape == truth.shape
    assert seen_fit_sizes == [16] * 6
    assert np.all((prediction > 0.0) & (prediction < 1.0))


def test_build_features_contracts() -> None:
    rng = np.random.RandomState(42)
    word = rng.uniform(0.01, 0.99, size=(20, 6))
    char = rng.uniform(0.01, 0.99, size=(20, 6))
    blend = (word + char) / 2.0
    fold = np.repeat(np.arange(2), 10)
    assert module.build_features(word, char, blend, fold, "rank12").shape == (20, 12)
    assert module.build_features(word, char, blend, fold, "rank18").shape == (20, 18)
    assert module.build_features(word, char, blend, fold, "logit12").shape == (20, 12)
    assert module.build_features(word, char, blend, fold, "hybrid30").shape == (20, 30)


def test_nested_stacker_selects_c_without_outer_fold_truth() -> None:
    rng = np.random.default_rng(44)
    rows = np.arange(180)
    fold = np.repeat(np.arange(3), 60)
    truth = np.column_stack(
        [((rows + label) % (2 + label % 2) == 0).astype(np.int8) for label in range(6)]
    )
    features = rng.normal(size=(len(rows), 30))
    features[:, :6] += 0.8 * truth
    prediction, counts, records = module.nested_cross_fit_logistic_stacker(
        features,
        truth,
        fold,
        c_values=[0.03, 0.1],
        max_iter=100,
        seed=70043,
    )
    assert prediction.shape == truth.shape
    assert np.isfinite(prediction).all()
    assert np.all(counts == 1)
    assert len(records) == 18
    assert all(record["score_fold"] not in record["fit_folds"] for record in records)

    mutated = truth.copy()
    held_out = fold == 0
    mutated[held_out, 0] = 1 - mutated[held_out, 0]
    repeated, _, _ = module.nested_cross_fit_logistic_stacker(
        features,
        mutated,
        fold,
        c_values=[0.03, 0.1],
        max_iter=100,
        seed=70043,
    )
    np.testing.assert_allclose(prediction[held_out, 0], repeated[held_out, 0], atol=1e-12)
