from __future__ import annotations

from scripts.benchmark_openai_gateway_profiles import profile_order, summarize


def test_profile_order_rotates_first_profile() -> None:
    names = ["baseline", "interactive", "research"]
    assert profile_order(0, names) == names
    assert profile_order(1, names) == ["interactive", "research", "baseline"]
    assert profile_order(2, names) == ["research", "baseline", "interactive"]


def test_summarize_reports_stable_median_and_bounds() -> None:
    assert summarize([]) == {
        "count": 0,
        "minimum_ms": None,
        "median_ms": None,
        "maximum_ms": None,
    }
    assert summarize([300.0, 100.0, 200.0]) == {
        "count": 3,
        "minimum_ms": 100.0,
        "median_ms": 200.0,
        "maximum_ms": 300.0,
    }
