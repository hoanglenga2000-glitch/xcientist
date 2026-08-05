import numpy as np
import pytest

from scripts.diagnose_taxi_cpu_longhaul_expert import (
    TaxiLonghaulDiagnosticError,
    blend_longhaul_predictions,
    feature_indices,
    longhaul_mask,
    rmse,
)


def test_longhaul_mask_uses_distance_or_airport_only() -> None:
    names = ["haversine_km", "airport_trip", "year"]
    values = np.array(
        [
            [2.0, 0.0, 2012],
            [6.0, 0.0, 2013],
            [1.0, 1.0, 2014],
            [10.0, 0.0, 2015],
        ],
        dtype=np.float32,
    )

    result = longhaul_mask(values, names, distance_threshold_km=6.0)

    assert result.tolist() == [False, True, True, True]


def test_longhaul_mask_rejects_missing_or_degenerate_inputs() -> None:
    with pytest.raises(TaxiLonghaulDiagnosticError, match="lacks features"):
        longhaul_mask(
            np.ones((3, 1), dtype=np.float32),
            ["haversine_km"],
            distance_threshold_km=6.0,
        )
    with pytest.raises(TaxiLonghaulDiagnosticError, match="degenerate"):
        longhaul_mask(
            np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            ["haversine_km", "airport_trip"],
            distance_threshold_km=6.0,
        )


def test_blend_longhaul_predictions_changes_only_selected_rows() -> None:
    base = np.array([5.0, 10.0, 20.0])
    expert = np.array([50.0, 14.0, 28.0])
    mask = np.array([False, True, True])

    result = blend_longhaul_predictions(
        base,
        expert,
        mask,
        expert_weight=0.75,
    )

    assert result.tolist() == [5.0, 13.0, 26.0]


def test_blend_and_rmse_validate_contracts() -> None:
    assert rmse(np.array([1.0, 3.0]), np.array([1.0, 5.0])) == pytest.approx(
        np.sqrt(2.0)
    )
    with pytest.raises(TaxiLonghaulDiagnosticError, match="outside"):
        blend_longhaul_predictions(
            np.array([1.0]),
            np.array([2.0]),
            np.array([True]),
            expert_weight=1.1,
        )
    with pytest.raises(TaxiLonghaulDiagnosticError, match="misaligned"):
        blend_longhaul_predictions(
            np.array([1.0, 2.0]),
            np.array([2.0]),
            np.array([True, False]),
            expert_weight=0.5,
        )


def test_feature_indices_preserves_requested_order() -> None:
    assert feature_indices(
        ["year", "haversine_km", "airport_trip"],
        ["airport_trip", "year"],
    ) == {"airport_trip": 2, "year": 0}
