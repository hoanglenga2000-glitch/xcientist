"""Offline contract tests for the May-2022 nested model-selection repair."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import mlebench_medal_recovery_adapters as recovery


def _balanced_outer_splits(
    *, row_count: int = 60, fold_count: int = 3
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Return deterministic folds whose train and validation sets both contain two classes."""

    if row_count % (2 * fold_count):
        raise ValueError("test fixture requires equal, class-balanced folds")
    target = np.tile(np.array([0, 1], dtype=np.int8), row_count // 2)
    all_rows = np.arange(row_count, dtype=np.int64)
    validation_folds = np.split(all_rows, fold_count)
    outer_splits = [
        (np.setdiff1d(all_rows, valid, assume_unique=True), valid.copy())
        for valid in validation_folds
    ]
    return target, outer_splits


def _nested_contract_fixture() -> tuple[list[dict[str, object]], np.ndarray, int, int]:
    target, outer_splits = _balanced_outer_splits()
    plans, coverage = recovery.build_may2022_nested_fold_plans(
        target,
        outer_splits,
        seed=42,
    )
    records: list[dict[str, object]] = []
    for plan in plans:
        nested = {
            **plan["evidence"],
            "all_budgets_frozen_before_outer_prediction": True,
            "refit_uses_complete_outer_train": True,
            "budgets": {
                "residual_mlp": {
                    "selected_iteration": 17,
                    "selected_iteration_zero_based": False,
                    "requested_budget": 80,
                    "frozen_refit_budget": 17,
                    "fallback_to_requested_budget": False,
                    "selection_inner_auc": 0.81,
                    "selection_seed": 143,
                    "outer_validation_labels_used_for_selection": False,
                },
                "xgboost": {
                    "selected_iteration": 310,
                    "selected_iteration_zero_based": True,
                    "requested_budget": 1_200,
                    "frozen_refit_budget": 311,
                    "fallback_to_requested_budget": False,
                    "selection_inner_auc": 0.82,
                    "selection_seed": 143,
                    "outer_validation_labels_used_for_selection": False,
                },
                "catboost": {
                    "selected_iteration": 408,
                    "selected_iteration_zero_based": True,
                    "requested_budget": 1_500,
                    "frozen_refit_budget": 409,
                    "fallback_to_requested_budget": False,
                    "selection_inner_auc": 0.83,
                    "selection_seed": 160,
                    "outer_validation_labels_used_for_selection": False,
                },
            },
        }
        records.append(
            {
                "fold": int(plan["outer_fold"]) + 1,
                "nested_selection": nested,
            }
        )
    return records, coverage, len(target), len(outer_splits)


def _parameter_args() -> SimpleNamespace:
    return SimpleNamespace(
        may_xgb_depth=7,
        may_catboost_depth=8,
        may_learning_rate=0.025,
        may_early_stopping=73,
        may_verbose_eval=125,
    )


def test_nested_fold_plans_are_deterministic_and_confined_to_outer_train():
    target, outer_splits = _balanced_outer_splits()

    first, first_coverage = recovery.build_may2022_nested_fold_plans(
        target,
        outer_splits,
        seed=91,
    )
    second, second_coverage = recovery.build_may2022_nested_fold_plans(
        target,
        outer_splits,
        seed=91,
    )

    np.testing.assert_array_equal(first_coverage, second_coverage)
    assert [plan["evidence"] for plan in first] == [plan["evidence"] for plan in second]
    assert len(first) == len(second) == len(outer_splits)
    all_rows = np.arange(len(target), dtype=np.int64)
    independently_reconstructed_coverage = np.zeros(len(target), dtype=np.int16)
    for left, right in zip(first, second, strict=True):
        for name in (
            "outer_train_index",
            "outer_valid_index",
            "inner_train_index",
            "inner_valid_index",
        ):
            np.testing.assert_array_equal(left[name], right[name])

        outer_train = left["outer_train_index"]
        outer_valid = left["outer_valid_index"]
        inner_train = left["inner_train_index"]
        inner_valid = left["inner_valid_index"]
        assert np.intersect1d(outer_train, outer_valid).size == 0
        assert np.intersect1d(inner_train, inner_valid).size == 0
        assert np.intersect1d(inner_train, outer_valid).size == 0
        assert np.intersect1d(inner_valid, outer_valid).size == 0
        np.testing.assert_array_equal(
            np.sort(np.concatenate((inner_train, inner_valid))),
            outer_train,
        )
        np.testing.assert_array_equal(
            np.sort(np.concatenate((outer_train, outer_valid))),
            all_rows,
        )
        assert set(target[inner_train]) == {0, 1}
        assert set(target[inner_valid]) == {0, 1}
        independently_reconstructed_coverage[outer_valid] += 1

    np.testing.assert_array_equal(first_coverage, independently_reconstructed_coverage)
    np.testing.assert_array_equal(first_coverage, np.ones(len(target), dtype=np.int16))


@pytest.mark.parametrize("malformation", ["overlap", "incomplete"])
def test_nested_fold_plans_fail_closed_when_an_outer_fold_is_malformed(malformation: str):
    target, outer_splits = _balanced_outer_splits()
    malformed = [(train.copy(), valid.copy()) for train, valid in outer_splits]
    train, valid = malformed[0]
    if malformation == "overlap":
        malformed[0] = (np.append(train, valid[0]), valid)
        message = "outer train and validation rows overlap"
    else:
        malformed[0] = (train[1:], valid)
        message = "does not partition every training row"

    with pytest.raises(RuntimeError, match=message):
        recovery.build_may2022_nested_fold_plans(target, malformed, seed=42)


def test_nested_fold_plans_fail_closed_for_inexact_global_outer_coverage():
    target, outer_splits = _balanced_outer_splits()
    duplicated_validation = outer_splits[0][1].copy()
    all_rows = np.arange(len(target), dtype=np.int64)
    malformed = list(outer_splits)
    malformed[-1] = (
        np.setdiff1d(all_rows, duplicated_validation, assume_unique=True),
        duplicated_validation,
    )

    with pytest.raises(RuntimeError, match="cover every row exactly once"):
        recovery.build_may2022_nested_fold_plans(target, malformed, seed=42)


def test_freeze_refit_budget_converts_fallbacks_and_clamps():
    zero_based = recovery.freeze_may2022_refit_budget(
        selected_iteration=4,
        requested_budget=10,
        zero_based_iteration=True,
    )
    fallback = recovery.freeze_may2022_refit_budget(
        selected_iteration=-1,
        requested_budget=23,
        zero_based_iteration=True,
    )
    clamped = recovery.freeze_may2022_refit_budget(
        selected_iteration=500,
        requested_budget=100,
        zero_based_iteration=True,
    )

    assert zero_based["frozen_refit_budget"] == 5
    assert zero_based["fallback_to_requested_budget"] is False
    assert fallback["frozen_refit_budget"] == 23
    assert fallback["fallback_to_requested_budget"] is True
    assert clamped["frozen_refit_budget"] == 100
    assert clamped["fallback_to_requested_budget"] is False


@pytest.mark.parametrize("requested_budget", [0, -1, -100])
def test_freeze_refit_budget_rejects_nonpositive_requested_budget(requested_budget: int):
    with pytest.raises(RuntimeError, match="requested iteration budget must be positive"):
        recovery.freeze_may2022_refit_budget(
            selected_iteration=1,
            requested_budget=requested_budget,
            zero_based_iteration=True,
        )


@pytest.mark.parametrize(
    ("selected_iteration", "zero_based_iteration", "message"),
    [
        (-2, True, "must be -1 or non-negative"),
        (0, False, "one-based selected iteration must be positive or -1"),
    ],
)
def test_freeze_refit_budget_rejects_invalid_selected_iteration_basis(
    selected_iteration: int,
    zero_based_iteration: bool,
    message: str,
):
    with pytest.raises(RuntimeError, match=message):
        recovery.freeze_may2022_refit_budget(
            selected_iteration=selected_iteration,
            requested_budget=10,
            zero_based_iteration=zero_based_iteration,
        )


def test_xgboost_early_stopping_exists_only_for_inner_selection():
    args = _parameter_args()
    selector = recovery.build_may2022_xgboost_parameters(
        args,
        seed=42,
        estimator_budget=1_200,
        selection=True,
    )
    refit = recovery.build_may2022_xgboost_parameters(
        args,
        seed=42,
        estimator_budget=311,
        selection=False,
    )

    assert selector["n_estimators"] == 1_200
    assert selector["early_stopping_rounds"] == args.may_early_stopping
    assert refit["n_estimators"] == 311
    assert "early_stopping_rounds" not in refit
    assert selector["random_state"] == refit["random_state"] == 42


def test_catboost_overfitting_detector_exists_only_for_inner_selection():
    args = _parameter_args()
    selector = recovery.build_may2022_catboost_parameters(
        args,
        seed=59,
        iteration_budget=1_500,
        selection=True,
    )
    refit = recovery.build_may2022_catboost_parameters(
        args,
        seed=59,
        iteration_budget=409,
        selection=False,
    )

    assert selector["iterations"] == 1_500
    assert selector["od_type"] == "Iter"
    assert selector["od_wait"] == args.may_early_stopping
    assert refit["iterations"] == 409
    assert "od_type" not in refit
    assert "od_wait" not in refit
    assert selector["random_seed"] == refit["random_seed"] == 59


def test_valid_nested_selection_contract_records_exact_oof_coverage():
    records, coverage, expected_rows, expected_folds = _nested_contract_fixture()

    contract = recovery.validate_may2022_nested_selection_contract(
        records,
        coverage,
        expected_rows=expected_rows,
        expected_folds=expected_folds,
    )

    assert contract["passed"] is True
    assert contract["fold_count"] == expected_folds
    assert contract["expected_rows"] == expected_rows
    assert contract["oof_coverage_min"] == 1
    assert contract["oof_coverage_max"] == 1
    assert len(contract["oof_coverage_sha256"]) == 64
    assert contract["outer_validation_labels_used_for_selection"] is False
    assert contract["all_budgets_frozen_before_outer_prediction"] is True
    assert contract["refit_uses_complete_outer_train"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda records: records[0]["nested_selection"].__setitem__(
                "outer_validation_labels_used_for_selection", True
            ),
            "outer validation labels entered budget selection",
        ),
        (
            lambda records: records[0]["nested_selection"]["budgets"]["xgboost"].__setitem__(
                "frozen_refit_budget", 1_201
            ),
            "xgboost frozen budget is invalid",
        ),
        (
            lambda records: records[0]["nested_selection"]["budgets"]["xgboost"].__setitem__(
                "frozen_refit_budget", 312
            ),
            "xgboost frozen budget provenance is inconsistent",
        ),
        (
            lambda records: records[0]["nested_selection"]["budgets"]["catboost"].__setitem__(
                "selection_inner_auc", float("nan")
            ),
            "catboost inner-selection AUC is invalid",
        ),
        (
            lambda records: records[0]["nested_selection"]["budgets"]["catboost"].__setitem__(
                "selection_inner_auc", 1.01
            ),
            "catboost inner-selection AUC is invalid",
        ),
        (
            lambda records: records[0]["nested_selection"].__setitem__(
                "inner_validation_fraction", 0.25
            ),
            "inner validation fraction provenance is invalid",
        ),
        (
            lambda records: records[0]["nested_selection"].__setitem__(
                "inner_valid_index_sha256", "g" * 64
            ),
            "nested split hash is invalid",
        ),
        (
            lambda records: records[0].__setitem__("fold", 99),
            "nested outer-fold provenance is inconsistent",
        ),
    ],
)
def test_nested_selection_contract_fails_closed_when_provenance_is_tampered(
    mutation,
    message: str,
):
    records, coverage, expected_rows, expected_folds = _nested_contract_fixture()
    tampered = copy.deepcopy(records)
    mutation(tampered)

    with pytest.raises(RuntimeError, match=message):
        recovery.validate_may2022_nested_selection_contract(
            tampered,
            coverage,
            expected_rows=expected_rows,
            expected_folds=expected_folds,
        )


@pytest.mark.parametrize(
    "coverage_mutation",
    [
        lambda coverage: coverage.__setitem__(0, 0),
        lambda coverage: coverage.__setitem__(0, 2),
    ],
)
def test_nested_selection_contract_fails_closed_for_inexact_oof_coverage(
    coverage_mutation,
):
    records, coverage, expected_rows, expected_folds = _nested_contract_fixture()
    tampered_coverage = coverage.copy()
    coverage_mutation(tampered_coverage)

    with pytest.raises(RuntimeError, match="OOF coverage is not exactly one"):
        recovery.validate_may2022_nested_selection_contract(
            records,
            tampered_coverage,
            expected_rows=expected_rows,
            expected_folds=expected_folds,
        )
