"""CLI tests for `xsci report` (list + detail), read-only over artifacts."""
from __future__ import annotations

import json

import pytest

from xsci.__main__ import main


def _write_run(base, run_id, *, task="titanic", score=0.9):
    d = base / run_id
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({
        "task": task, "best_exp_id": "exp_002", "best_cv_score": score,
        "metric": "accuracy", "metric_direction": "maximize",
        "n_iterations": 2, "n_promotions": 1,
        "iterations": [
            {"exp_id": "exp_001", "mode": "draft", "success": True, "cv_score": 0.8,
             "promoted": False, "provider": "anthropic", "model": "opus"},
            {"exp_id": "exp_002", "mode": "improve", "success": False, "cv_score": None,
             "promoted": False, "provider": "deepseek", "model": "v3"},
        ],
    }), encoding="utf-8")
    return d


@pytest.fixture()
def project_with_runs(tmp_path, monkeypatch):
    from xsci.project import run_init
    monkeypatch.chdir(tmp_path)
    run_init()
    base = tmp_path / "experiments" / "evolution"
    _write_run(base, "titanic_local_20260101_000000", score=0.85)
    _write_run(base, "titanic_gpu_20260305_120000", score=0.91)
    return tmp_path


def test_report_list(project_with_runs, capsys):
    rc = main(["report"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "titanic_gpu_20260305_120000" in out
    assert "titanic_local_20260101_000000" in out
    # newest first
    assert out.index("20260305") < out.index("20260101")
    assert "0.9100" in out and "0.8500" in out


def test_report_detail_by_run_id(project_with_runs, capsys):
    rc = main(["report", "titanic_gpu_20260305_120000"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "best    : exp_002" in out
    assert "exp_001" in out and "exp_002" in out
    assert "anthropic/opus" in out and "deepseek/v3" in out
    assert "  -   " in out  # None cv rendered as dash


def test_report_detail_by_task_picks_newest(project_with_runs, capsys):
    rc = main(["report", "titanic"])
    out = capsys.readouterr().out
    assert rc == 0 and "titanic_gpu_20260305_120000" in out


def test_report_unknown_run(project_with_runs, capsys):
    rc = main(["report", "nope"])
    assert rc == 1 and "no run matching" in capsys.readouterr().out


def test_report_empty_project(tmp_path, monkeypatch, capsys):
    from xsci.project import run_init
    monkeypatch.chdir(tmp_path)
    run_init()
    rc = main(["report"])
    assert rc == 0 and "no completed runs yet" in capsys.readouterr().out
