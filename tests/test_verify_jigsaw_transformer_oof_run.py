"""Contracts for the independent Jigsaw transformer OOF verifier."""

from __future__ import annotations

import numpy as np

from scripts import verify_jigsaw_transformer_oof_run as verifier


def _blend_fixture():
    rng = np.random.default_rng(20260727)
    folds = np.repeat(np.arange(5), 24)
    row = np.arange(len(folds))
    truth = np.column_stack(
        [((row + label * 2) % (3 + label % 3) == 0).astype(np.int8) for label in range(6)]
    )
    sparse_oof = np.clip(0.15 + 0.70 * truth + rng.normal(0, 0.23, truth.shape), 0.001, 0.999)
    transformer_oof = np.clip(0.12 + 0.76 * truth + rng.normal(0, 0.18, truth.shape), 0.001, 0.999)
    sparse_test = rng.uniform(0.01, 0.99, size=(19, 6))
    transformer_test = rng.uniform(0.01, 0.99, size=(5, 19, 6))
    return sparse_oof, transformer_oof, sparse_test, transformer_test, truth, folds


def test_independent_fractional_rank_uses_average_ties():
    values = np.array([9.0, 2.0, 2.0, 5.0])
    ranked = verifier.independent_fractional_rank(values)
    assert np.allclose(ranked, np.array([1.0, 0.375, 0.375, 0.75]))


def test_reconstructed_blend_is_exact_once_and_finite():
    sparse, transformer, sparse_test, transformer_test, truth, folds = _blend_fixture()
    oof, test, counts, records = verifier.reconstruct_cross_fit_blend(
        sparse,
        transformer,
        sparse_test,
        transformer_test,
        truth,
        folds,
        [0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
    )
    assert oof.shape == truth.shape
    assert test.shape == sparse_test.shape
    assert np.isfinite(oof).all()
    assert np.isfinite(test).all()
    assert np.all(counts == 1)
    assert len(records) == 30


def test_held_fold_truth_does_not_affect_its_prediction_or_weight():
    sparse, transformer, sparse_test, transformer_test, truth, folds = _blend_fixture()
    baseline = verifier.reconstruct_cross_fit_blend(
        sparse,
        transformer,
        sparse_test,
        transformer_test,
        truth,
        folds,
        [0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
    )
    changed_truth = truth.copy()
    changed_truth[folds == 3] = 1 - changed_truth[folds == 3]
    changed = verifier.reconstruct_cross_fit_blend(
        sparse,
        transformer,
        sparse_test,
        transformer_test,
        changed_truth,
        folds,
        [0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
    )
    assert np.array_equal(baseline[0][folds == 3], changed[0][folds == 3])
    baseline_records = [item for item in baseline[3] if item["score_fold"] == 3]
    changed_records = [item for item in changed[3] if item["score_fold"] == 3]
    assert baseline_records == changed_records


def test_mean_columnwise_auc_returns_per_label_evidence():
    _, transformer, _, _, truth, _ = _blend_fixture()
    aggregate, per_label = verifier.mean_columnwise_auc(truth, transformer)
    assert len(per_label) == 6
    assert np.isclose(aggregate, np.mean(per_label))
    assert all(0.5 < value <= 1.0 for value in per_label)


def test_explicit_confirmation_seed_contract_is_deterministic():
    contract = verifier.resolve_seed_contract(
        {"training": {"seed": 40, "fold_seed": 42, "model_seed": 40}}
    )
    assert contract == {
        "fold_assignment_seed": 42,
        "model_seed": 40,
        "fold_model_seeds": [40, 1049, 2058, 3067, 4076],
        "explicit_split": True,
    }


def test_legacy_seed_contract_remains_compatible():
    contract = verifier.resolve_seed_contract({"training": {"seed": 42}})
    assert contract["fold_assignment_seed"] == 42
    assert contract["model_seed"] == 42
    assert contract["fold_model_seeds"] == [42, 1051, 2060, 3069, 4078]
    assert contract["explicit_split"] is False
