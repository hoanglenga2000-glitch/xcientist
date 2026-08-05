from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.collect_job89941_taxi_cpu_candidate import (
    TaxiCpuCollectionError,
    rmse,
    validate_collection_plan,
    validate_result,
)

def test_rmse_recomputes_metric_and_rejects_misalignment() -> None:
    assert rmse(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == pytest.approx(2**0.5)
    with pytest.raises(TaxiCpuCollectionError, match="invalid"):
        rmse(np.ones(2), np.ones(3))


def test_validate_result_preserves_candidate_boundary() -> None:
    result = {
        "schema": "evomind.mlebench.taxi_cpu_lightgbm_candidate.v1",
        "status": "candidate_complete",
        "seed": 43,
        "threads": 60,
        "candidate_only": True,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "gpu_used": False,
        "visibility_mode": "PUBLIC_ONLY",
        "cache_manifest": {"sha256": "a" * 64},
        "route_stat_manifest": {"sha256": "b" * 64},
    }
    plan = {
        "runtime": {"seed": 43},
        "base_cache": {"sha256": "a" * 64},
        "route_stat_cache": {"sha256": "b" * 64},
    }
    validate_result(result, plan)
    result["official_grader_executed"] = True
    with pytest.raises(TaxiCpuCollectionError, match="boundary"):
        validate_result(result, plan)


def test_collection_plan_accepts_frozen_launched_source_after_checkout_advances() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = validate_collection_plan(
        root
        / "workspace"
        / "mlebench_plans"
        / "taxi_cpu_lightgbm_candidate_s43_job89941_v3_20260728.json"
    )
    assert plan["runtime"]["seed"] == 43
    assert len(plan["_sha256"]) == 64
