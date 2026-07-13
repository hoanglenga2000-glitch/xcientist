"""Tests for the read-only results reader, including a pass over the real
experiments/evolution artifacts in this repo (skipped if absent)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from xsci import results as xr

REPO_EVO = Path(__file__).resolve().parents[1] / "experiments" / "evolution"


def _write_run(base, run_id, *, task="titanic", best="exp_002", score=0.9, iters=None):
    d = base / run_id
    d.mkdir(parents=True)
    iters = iters if iters is not None else [
        {"exp_id": "exp_001", "mode": "draft", "success": True, "cv_score": 0.8,
         "promoted": False, "provider": "anthropic", "model": "opus"},
        {"exp_id": "exp_002", "mode": "improve", "success": True, "cv_score": 0.9,
         "promoted": True, "provider": "anthropic", "model": "opus"},
    ]
    summary = {
        "task": task, "best_exp_id": best, "best_cv_score": score,
        "metric": "accuracy", "metric_direction": "maximize",
        "n_iterations": len(iters), "n_promotions": 1,
        "iterations": iters,
    }
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return d


def test_load_result_parses_summary(tmp_path):
    run = _write_run(tmp_path / "experiments" / "evolution", "titanic_local_20260101_000000")
    r = xr.load_result(run)
    assert r is not None
    assert r.task == "titanic" and r.best_cv_score == 0.9
    assert r.n_iterations == 2 and len(r.iterations) == 2
    assert r.iterations[1].promoted is True
    assert r.success_rate == 1.0


def test_list_results_newest_first(tmp_path):
    base = tmp_path / "experiments" / "evolution"
    _write_run(base, "a_local_20260101_000000")
    _write_run(base, "b_local_20260202_000000")
    got = xr.list_results(project_root=tmp_path)
    assert [r.run_id for r in got] == [
        "b_local_20260202_000000", "a_local_20260101_000000"]


def test_missing_or_corrupt_summary_is_skipped(tmp_path):
    base = tmp_path / "experiments" / "evolution"
    (base / "in_progress").mkdir(parents=True)  # no summary.json yet
    bad = base / "corrupt"; bad.mkdir()
    (bad / "summary.json").write_text("{not json", encoding="utf-8")
    _write_run(base, "good_local_20260101_000000")
    got = xr.list_results(project_root=tmp_path)
    assert [r.run_id for r in got] == ["good_local_20260101_000000"]
    assert xr.load_result(base / "in_progress") is None
    assert xr.load_result(base / "corrupt") is None


def test_find_result_by_run_id_and_task(tmp_path):
    base = tmp_path / "experiments" / "evolution"
    _write_run(base, "titanic_local_20260101_000000", task="titanic")
    _write_run(base, "titanic_gpu_20260305_000000", task="titanic")
    by_id = xr.find_result("titanic_local_20260101_000000", project_root=tmp_path)
    assert by_id and by_id.run_id == "titanic_local_20260101_000000"
    by_task = xr.find_result("titanic", project_root=tmp_path)  # newest wins
    assert by_task and by_task.run_id == "titanic_gpu_20260305_000000"
    assert xr.find_result("nope", project_root=tmp_path) is None


@pytest.mark.skipif(not REPO_EVO.is_dir(), reason="no real experiments dir")
def test_reads_real_repo_artifacts():
    root = REPO_EVO.parents[1]
    results = xr.list_results(project_root=root)
    assert len(results) >= 1
    for r in results:  # every parsed run has the invariants we rely on
        assert r.task
        assert r.n_iterations >= 0
        assert 0.0 <= r.success_rate <= 1.0
        assert isinstance(r.iterations, list)
