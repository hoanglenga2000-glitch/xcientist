import numpy as np
import pytest

from scripts.diagnose_taxi_cpu_airport_expert import airport_mask
from scripts.diagnose_taxi_cpu_longhaul_expert import TaxiLonghaulDiagnosticError


def test_airport_mask_selects_only_airport_feature() -> None:
    names = ["haversine_km", "airport_trip", "year"]
    values = np.array(
        [
            [20.0, 0.0, 2012],
            [2.0, 1.0, 2013],
            [0.5, 0.0, 2014],
            [15.0, 1.0, 2015],
        ],
        dtype=np.float32,
    )

    result = airport_mask(values, names)

    assert result.tolist() == [False, True, False, True]


def test_airport_mask_rejects_degenerate_input() -> None:
    with pytest.raises(TaxiLonghaulDiagnosticError, match="degenerate"):
        airport_mask(
            np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            ["haversine_km", "airport_trip"],
        )
