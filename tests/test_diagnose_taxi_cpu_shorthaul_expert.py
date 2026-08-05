import numpy as np
import pytest

from scripts.diagnose_taxi_cpu_longhaul_expert import TaxiLonghaulDiagnosticError
from scripts.diagnose_taxi_cpu_shorthaul_expert import shorthaul_mask


def test_shorthaul_mask_excludes_airport_rows() -> None:
    names = ["haversine_km", "airport_trip", "year"]
    values = np.array(
        [
            [0.0, 0.0, 2012],
            [0.5, 0.0, 2013],
            [0.5, 1.0, 2014],
            [1.0, 0.0, 2015],
            [2.0, 0.0, 2015],
        ],
        dtype=np.float32,
    )

    result = shorthaul_mask(values, names, maximum_distance_km=1.0)

    assert result.tolist() == [True, True, False, False, False]


def test_shorthaul_mask_rejects_invalid_threshold_and_degenerate_mask() -> None:
    values = np.array([[2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    names = ["haversine_km", "airport_trip"]
    with pytest.raises(TaxiLonghaulDiagnosticError, match="must be positive"):
        shorthaul_mask(values, names, maximum_distance_km=0.0)
    with pytest.raises(TaxiLonghaulDiagnosticError, match="degenerate"):
        shorthaul_mask(values, names, maximum_distance_km=1.0)
