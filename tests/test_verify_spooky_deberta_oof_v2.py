"""Independent reconstruction tests for the Spooky DeBERTa OOF v2 verifier."""

from __future__ import annotations

import numpy as np

from scripts import verify_spooky_deberta_oof_v2 as verifier


def _fixture(seed: int = 21):
    rng = np.random.default_rng(seed)
    rows = 150
    test_rows = 11
    folds = np.tile(np.arange(5), rows // 5)
    truth = np.tile(np.arange(3), rows // 3)
    component_oof = {}
    component_test = {}
    for strength, name in ((1.6, "transformer"), (1.0, "sparse")):
        logits = rng.normal(size=(rows, 3))
        logits[np.arange(rows), truth] += strength
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        component_oof[name] = probability / probability.sum(axis=1, keepdims=True)
        raw = rng.uniform(0.1, 1.0, size=(5, test_rows, 3))
        component_test[name] = raw / raw.sum(axis=2, keepdims=True)
    train_style = rng.normal(size=(rows, 28))
    test_style = rng.normal(size=(test_rows, 28))
    return component_oof, component_test, train_style, test_style, truth, folds


def test_independent_meta_reconstruction_is_exact_once_and_normalized():
    component_oof, component_test, train_style, test_style, truth, folds = _fixture()
    oof, test, counts, feature_contract, meta_contract = verifier.reconstruct_candidate(
        component_oof,
        component_test,
        train_style,
        test_style,
        truth,
        folds,
        c_value=0.3,
        max_iter=1000,
        random_state=9042,
    )
    assert oof.shape == (150, 3)
    assert test.shape == (11, 3)
    assert np.all(counts == 1)
    assert np.allclose(oof.sum(axis=1), 1.0)
    assert np.allclose(test.sum(axis=1), 1.0)
    assert feature_contract["columns"] == 34
    assert meta_contract["exact_once_oof"] is True


def test_held_fold_truth_does_not_change_predictions_for_that_fold():
    component_oof, component_test, train_style, test_style, truth, folds = _fixture()
    baseline = verifier.reconstruct_candidate(
        component_oof,
        component_test,
        train_style,
        test_style,
        truth,
        folds,
        c_value=0.3,
        max_iter=1000,
        random_state=9042,
    )
    changed_truth = truth.copy()
    changed_truth[folds == 2] = (changed_truth[folds == 2] + 1) % 3
    changed = verifier.reconstruct_candidate(
        component_oof,
        component_test,
        train_style,
        test_style,
        changed_truth,
        folds,
        c_value=0.3,
        max_iter=1000,
        random_state=9042,
    )
    assert np.array_equal(baseline[0][folds == 2], changed[0][folds == 2])


def test_bundle_loader_accepts_unicode_ids_without_pickle(tmp_path):
    rows = 6
    test_rows = 2
    folds = 5
    probability = np.full((rows, 3), 1.0 / 3.0)
    test_probability = np.full((folds, test_rows, 3), 1.0 / 3.0)
    path = tmp_path / "bundle.npz"
    np.savez_compressed(
        path,
        truth=np.arange(rows) % 3,
        fold=np.arange(rows) % folds,
        transformer_oof=probability,
        sparse_oof=probability,
        transformer_test_by_fold=test_probability,
        sparse_test_by_fold=test_probability,
        candidate_oof=probability,
        candidate_test=np.full((test_rows, 3), 1.0 / 3.0),
        component_write_counts=np.ones(rows, dtype=np.uint8),
        candidate_write_counts=np.ones(rows, dtype=np.uint8),
        train_id=np.asarray([f"训练-{index}" for index in range(rows)], dtype=np.str_),
        test_id=np.asarray([f"测试-{index}" for index in range(test_rows)], dtype=np.str_),
    )
    arrays, report = verifier._load_bundle(path)
    assert report["allow_pickle"] is False
    assert report["id_arrays_safe"] is True
    assert arrays["train_id"].dtype.kind == "U"
