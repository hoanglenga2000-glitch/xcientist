from __future__ import annotations

import math

from scripts import verify_spooky_cross_run_xgb_confirmation as verifier


def test_close_metric_requires_finite_exact_tolerance() -> None:
    assert verifier.close_metric(0.2815, 0.2815 + 1e-13) is True
    assert verifier.close_metric(0.2815, 0.2815 + 1e-8) is False
    assert verifier.close_metric(math.nan, 0.2815) is False
