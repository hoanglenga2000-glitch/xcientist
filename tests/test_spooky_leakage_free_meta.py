"""Tests for leakage-free Spooky meta-model utilities."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import spooky_leakage_free_meta as meta


def _synthetic(seed: int = 42):
    rng = np.random.default_rng(seed)
    rows = 150
    test_rows = 17
    classes = 3
    folds = np.tile(np.arange(5), rows // 5)
    truth = np.tile(np.arange(classes), rows // classes)
    primary = np.full((rows, classes), 0.12)
    primary[np.arange(rows), truth] = 0.76
    primary = meta.normalize_probability(primary + rng.uniform(0.0, 0.02, primary.shape))
    secondary = meta.normalize_probability(0.8 * primary + 0.2 / classes)
    style = rng.normal(size=(rows, 4))
    embeddings = rng.normal(size=(rows, 7))
    features, feature_contract = meta.assemble_meta_features(
        {"primary": primary, "secondary": secondary},
        style_features=style,
        embedding_features=embeddings,
    )
    test_by_fold = rng.normal(size=(5, test_rows, features.shape[1]))
    for fold in range(5):
        probability = meta.normalize_probability(
            rng.uniform(0.1, 1.0, size=(test_rows, 3))
        )
        test_by_fold[fold, :, :6] = np.concatenate([probability, probability], axis=1)
    return features, test_by_fold, truth, folds, feature_contract


def test_assemble_meta_features_records_exact_column_ranges():
    features, _, _, _, contract = _synthetic()
    assert features.shape == (150, 17)
    assert contract["columns"] == 17
    assert contract["blocks"] == [
        {"name": "primary", "kind": "probability", "start": 0, "stop": 3},
        {"name": "secondary", "kind": "probability", "start": 3, "stop": 6},
        {"name": "style", "kind": "dense", "start": 6, "stop": 10},
        {"name": "embedding", "kind": "dense", "start": 10, "stop": 17},
    ]
    assert contract["private_labels_used"] is False


def test_cross_fit_logistic_meta_is_exact_once_normalized_and_disjoint():
    features, test_by_fold, truth, folds, _ = _synthetic()
    oof, test, counts, contract = meta.cross_fit_logistic_meta(
        features,
        test_by_fold,
        truth,
        folds,
        c_value=0.3,
    )
    assert oof.shape == (150, 3)
    assert test.shape == (17, 3)
    assert np.all(counts == 1)
    assert np.allclose(oof.sum(axis=1), 1.0)
    assert np.allclose(test.sum(axis=1), 1.0)
    assert contract["exact_once_oof"] is True
    assert contract["private_labels_used"] is False
    assert contract["official_grader_executed"] is False
    assert contract["kaggle_submission_executed"] is False
    for record in contract["records"]:
        assert record["score_fold"] not in record["fit_folds"]
        assert record["fit_rows"] + record["score_rows"] == len(truth)
        assert record["fit_score_disjoint"] is True
        assert record["fit_index_sha256"] != record["score_index_sha256"]


def test_cross_fit_rejects_noncontiguous_folds():
    features, test_by_fold, truth, folds, _ = _synthetic()
    folds = folds.copy()
    folds[folds == 4] = 7
    with pytest.raises(ValueError, match="contiguous"):
        meta.cross_fit_logistic_meta(features, test_by_fold, truth, folds)


def test_cross_fit_rejects_nonfinite_features():
    features, test_by_fold, truth, folds, _ = _synthetic()
    features[3, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        meta.cross_fit_logistic_meta(features, test_by_fold, truth, folds)
