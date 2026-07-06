"""Integration coverage for the engine-A -> workstation ``ingest_summary`` bridge.

This is the seam the web app crosses in the ``research_os`` engine path:
``runEvolutionEngineExperiment`` runs the loop, writes ``summary.json`` +
``search_graph.json`` under ``experiments/evolution/<run>/``, then
``ingestEvolutionSummary`` shells ``evolution_engine_cli.py --mode ingest_summary``
to re-apply the workstation promotion gate on top of engine A's own governance.

These tests exercise ``mode_ingest_summary`` directly against synthetic on-disk
engine-A artifacts (no GPU, no LLM, no network). They lock in the three outcomes
the web UI depends on:
  * eligible best node  -> promoted, gate passed, but official rank stays None
    and official submission stays disallowed (claim boundary preserved);
  * ineligible best node (run_success=False) -> held, never promoted;
  * missing/invalid summary -> rejected, never fabricated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# scripts/ is not a package on the default path; add it like the CLI expects.
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evolution_engine_cli as cli  # noqa: E402


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _seed_engine_a_run(
    root: Path,
    *,
    best_id: str = "EXP007",
    cv_score: float = 2.91,
    run_success: bool = True,
    artifacts_passed: bool = True,
    with_required_artifacts: bool = True,
) -> str:
    """Lay down a synthetic engine-A experiment dir under experiments/evolution/.

    Returns the ROOT-relative exp_dir string the web layer would pass in.
    """
    exp_rel = "experiments/evolution/nyc_taxi_gpu_20260703_120000"
    exp_dir = root / exp_rel

    node_artifacts = []
    if with_required_artifacts:
        node_artifacts = [
            {"path": f"{best_id}/metrics.json", "artifact_type": "metrics"},
            {"path": f"{best_id}/submission.csv", "artifact_type": "submission"},
        ]

    # search_graph.json: export_json writes nodes as a list.
    _write(exp_dir / "search_graph.json", {
        "best_exp_id": best_id,
        "metric_name": "rmse",
        "metric_direction": "minimize",
        "nodes": [{
            "exp_id": best_id,
            "parent_id": None,
            "branch_type": "research_os",
            "task_name": "new-york-city-taxi-fare-prediction",
            "hypothesis": "haversine + datetime parts + lightgbm",
            "implementation_summary": "engine A best solution",
            "cv_score": cv_score,
            "metric_name": "rmse",
            "metric_direction": "minimize",
            "run_success": run_success,
            "artifacts": node_artifacts,
            "metrics": {"rmse": cv_score},
        }],
        "edges": [],
    })
    _write(exp_dir / "summary.json", {
        "best_exp_id": best_id,
        "best_cv_score": cv_score,
        "metric": "rmse",
        "metric_direction": "minimize",
        "n_iterations": 6,
        "n_promotions": 1,
    })
    # Engine A's OWN recorded local governance for the best EXP.
    _write(exp_dir / best_id / "validation_contract.json", {
        "run_success": run_success,
        "artifact_check": {"passed": artifacts_passed},
        "cv_score": cv_score,
    })
    (exp_dir / "best_solution.py").write_text("# engine A best\n", encoding="utf-8")
    return exp_rel


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    """Redirect all CLI writes into tmp so tests never touch the real workspace."""
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "SHARED_MEMORY", tmp_path / "experiments" / "evolution" / "retrospective_memory.json")
    return tmp_path


def test_ingest_summary_promotes_eligible_best_but_preserves_claim_boundary(isolated_root):
    exp_rel = _seed_engine_a_run(isolated_root, cv_score=2.91)

    out = cli.mode_ingest_summary({
        "task_id": "nyc_taxi",
        "exp_dir": exp_rel,
        "metric_name": "rmse",
        "metric_direction": "minimize",
    })

    assert out["ok"] is True
    assert out["decision"] == "promoted"
    assert out["gate_status"] == "passed"
    assert out["cv_score"] == pytest.approx(2.91)
    assert out["run_success"] is True
    assert out["engine_a_exp_id"] == "EXP007"
    # Claim boundary: a promoted local run must NOT claim an official Kaggle rank
    # and must NOT unlock official submission.
    assert out["official_rank"] is None
    assert out["official_submit_allowed"] is False
    # The bridge actually wrote the workstation-side governance artifacts.
    tdir = isolated_root / "workspace" / "evolution" / "nyc_taxi"
    assert (tdir / "search_graph.json").exists()
    assert (tdir / "validation_contract.json").exists()
    assert (tdir / "claim_audit.json").exists()
    # And it imported engine A's real artifact references (no fabrication).
    assert any("submission.csv" in a for a in out["artifacts_found"])
    assert any("metrics.json" in a for a in out["artifacts_found"])


def test_ingest_summary_holds_when_engine_a_run_failed(isolated_root):
    exp_rel = _seed_engine_a_run(isolated_root, run_success=False)

    out = cli.mode_ingest_summary({
        "task_id": "nyc_taxi",
        "exp_dir": exp_rel,
        "metric_name": "rmse",
        "metric_direction": "minimize",
    })

    assert out["ok"] is True
    assert out["decision"] == "held"
    assert out["gate_status"] == "held"
    assert out["run_success"] is False
    assert out["official_submit_allowed"] is False


def test_ingest_summary_holds_when_artifacts_unverified(isolated_root):
    # run_success True but engine A's artifact_check did not pass -> not eligible.
    exp_rel = _seed_engine_a_run(isolated_root, run_success=True, artifacts_passed=False)

    out = cli.mode_ingest_summary({
        "task_id": "nyc_taxi",
        "exp_dir": exp_rel,
        "metric_name": "rmse",
        "metric_direction": "minimize",
    })

    assert out["ok"] is True
    assert out["decision"] == "held"
    assert out["official_submit_allowed"] is False


def test_ingest_summary_rejects_missing_summary(isolated_root):
    # Point at a dir under experiments/evolution/ that has no summary.json.
    empty_rel = "experiments/evolution/does_not_exist_run"
    (isolated_root / empty_rel).mkdir(parents=True, exist_ok=True)

    out = cli.mode_ingest_summary({"task_id": "nyc_taxi", "exp_dir": empty_rel})

    assert out["ok"] is False
    assert out["decision"] == "rejected_missing_summary"
    assert out["official_submit_allowed"] is False


def test_ingest_summary_rejects_exp_dir_outside_sandbox(isolated_root):
    # Path traversal / arbitrary location must be refused by _safe_exp_dir.
    with pytest.raises(ValueError):
        cli.mode_ingest_summary({"task_id": "nyc_taxi", "exp_dir": "../../etc/passwd"})
