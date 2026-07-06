"""Tests for the benchmark manager (medal-rate / valid-submission accounting).

These calculations back the anti-overclaim rules: medal_rate and
valid_submission_rate must reflect only real, recorded results.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_os.benchmark_manager import (
    BenchmarkResult,
    compute_gap_to_target,
    compute_medal_rate,
    compute_valid_submission_rate,
    save_result,
    summarize_results,
)


def _result(task_id: str, *, medal: str = "none", valid: bool = True, failed: int = 0,
            top30: bool = False) -> BenchmarkResult:
    return BenchmarkResult(
        task_id=task_id,
        best_exp_id=f"exp_{task_id}",
        best_cv_score=0.5,
        best_public_score=0.5,
        best_private_score=None,
        valid_submission=valid,
        medal=medal,
        rank_percentile=None,
        runtime_hours=1.0,
        num_experiments=3,
        num_failed_runs=failed,
        num_recoveries=0,
        reproducibility_score=1.0,
        auditability_score=1.0,
        claim_drift_detected=False,
        final_report_path=None,
        artifacts_path=None,
        gap_to_mlevolve=None,
        next_improvement_plan="",
        top30_reached=top30,
    )


def test_empty_rates_are_zero():
    assert compute_medal_rate([]) == 0.0
    assert compute_valid_submission_rate([]) == 0.0


def test_medal_rate_counts_only_medals():
    results = [
        _result("a", medal="bronze"),
        _result("b", medal="silver"),
        _result("c", medal="gold"),
        _result("d", medal="none"),
    ]
    assert compute_medal_rate(results) == 0.75


def test_unknown_medal_does_not_count():
    results = [_result("a", medal="unknown"), _result("b", medal="bronze")]
    assert compute_medal_rate(results) == 0.5


def test_valid_submission_rate():
    results = [_result("a", valid=True), _result("b", valid=False), _result("c", valid=True)]
    assert abs(compute_valid_submission_rate(results) - 2 / 3) < 1e-9


def test_gap_to_target():
    assert compute_gap_to_target(0.2, 0.5) == pytest.approx(0.3)
    assert compute_gap_to_target(0.6, 0.5) == pytest.approx(-0.1)


def test_summarize_results_structure():
    results = [
        _result("a", medal="bronze", top30=True),
        _result("b", medal="none", valid=False, failed=1),
    ]
    summary = summarize_results(results)
    assert summary["evaluated_tasks"] == 2
    assert summary["medal_rate"] == 0.5
    assert summary["bronze_rate"] == 0.5
    assert summary["top30_rate"] == 0.5
    assert "b" in summary["failed_tasks"]
    assert summary["medal_counts"]["bronze"] == 1


def test_save_result_dedupes_by_task_id(tmp_path: Path):
    target = tmp_path / "results.json"
    save_result(target, _result("titanic", medal="none"))
    save_result(target, _result("titanic", medal="bronze"))  # overwrite same task
    save_result(target, _result("house_prices", medal="none"))

    data = json.loads(target.read_text(encoding="utf-8"))
    by_id = {row["task_id"]: row for row in data}
    assert len(data) == 2
    assert by_id["titanic"]["medal"] == "bronze"
