import numpy as np
import pytest

from scripts.diagnose_taxi_cpu_distance_band_expert import distance_band_mask
from scripts.diagnose_taxi_cpu_longhaul_expert import TaxiLonghaulDiagnosticError


def test_distance_band_mask_can_exclude_airport_rows() -> None:
    names = ["haversine_km", "airport_trip"]
    values = np.array(
        [[2.9, 0.0], [3.0, 0.0], [4.0, 1.0], [5.9, 0.0], [6.0, 0.0]],
        dtype=np.float32,
    )

    result = distance_band_mask(
        values,
        names,
        minimum_distance_km=3.0,
        maximum_distance_km=6.0,
        exclude_airport=True,
    )

    assert result.tolist() == [False, True, False, True, False]


def test_distance_band_mask_rejects_invalid_bounds() -> None:
    with pytest.raises(TaxiLonghaulDiagnosticError, match="bounds are invalid"):
        distance_band_mask(
            np.ones((2, 2), dtype=np.float32),
            ["haversine_km", "airport_trip"],
            minimum_distance_km=6.0,
            maximum_distance_km=3.0,
            exclude_airport=False,
        )
