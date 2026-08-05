import numpy as np
import pytest

from scripts.diagnose_taxi_cpu_longhaul_expert import TaxiLonghaulDiagnosticError
from scripts.diagnose_taxi_cpu_weighted_midhaul_expert import fare_weights


def test_fare_weights_are_continuous_and_bounded() -> None:
    result = fare_weights(
        np.array([2.5, 20.0, 30.0, 80.0, 100.0]),
        pivot_fare=20.0,
        maximum_weight=4.0,
    )

    assert result.tolist() == [1.0, 1.0, 1.5, 4.0, 4.0]


def test_fare_weights_reject_invalid_contract() -> None:
    with pytest.raises(TaxiLonghaulDiagnosticError, match="parameters are invalid"):
        fare_weights(np.array([10.0]), pivot_fare=0.0, maximum_weight=4.0)
    with pytest.raises(TaxiLonghaulDiagnosticError, match="target is invalid"):
        fare_weights(
            np.array([np.nan]),
            pivot_fare=20.0,
            maximum_weight=4.0,
        )
