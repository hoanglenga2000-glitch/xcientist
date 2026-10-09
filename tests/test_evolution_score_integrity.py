"""Score-integrity and acceptance-gate regressions for the evolution loop (P0-2 / P2-3).

These exercise the public seams (``_parse_cv_score``, ``read_metrics_json``,
``EvolutionLoop.run`` and the audit artifacts it writes) with deterministic
fake runners, so no subprocess, GPU or network is involved.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from research_os.evolution_loop import (
    EvolutionConfig,
    EvolutionLoop,
    RunResult,
    _parse_cv_score,
    read_metrics_json,
)
from research_os.retrospective_memory import RetrospectiveMemoryStore
from research_os.variation_generator import TaskContext, VariationGenerator
from tests.test_evolution_engine import FakeLLMClient, _full_script


# ---- P2-3: score parsing ---------------------------------------------------
@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "-inf", "Infinity", "1e999"])
def test_parse_cv_score_rejects_non_finite(raw):
    assert _parse_cv_score(f"CV_SCORE={raw}") is None


def test_parse_cv_score_final_non_finite_emission_is_not_masked_by_earlier_value():
    # Fail closed: when the last emitted score is NaN, an earlier (e.g. per-fold)
    # value must not silently become the run's score.
    assert _parse_cv_score("CV_SCORE=0.5\nCV_SCORE=nan") is None
    assert _parse_cv_score("CV_SCORE=nan\nCV_SCORE=0.61") == pytest.approx(0.61)


def _write(path: Path, payload) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


def test_metrics_without_cv_score_are_kept_and_score_is_missing(tmp_path):
    out = _write(tmp_path / "out", {"evaluator_version": "ev-1", "environment_hash": "a" * 64, "auc": 0.9})
    metrics, score, problem = read_metrics_json(out)
    assert metrics["evaluator_version"] == "ev-1"
    assert metrics["environment_hash"] == "a" * 64
    assert score is None
    assert problem == "metrics_cv_score_missing"


@pytest.mark.parametrize("value", ["nan", "inf", "-Infinity", "not-a-number", None, [0.5]])
def test_metrics_with_invalid_cv_score_keep_metrics_but_reject_score(tmp_path, value):
    out = _write(tmp_path / "out", {"cv_score": value, "evaluator_version": "ev-2"})
    metrics, score, problem = read_metrics_json(out)
    assert metrics["evaluator_version"] == "ev-2"
    assert score is None
    assert problem in {"metrics_cv_score_invalid", "metrics_cv_score_missing"}


def test_metrics_nan_literal_is_rejected(tmp_path):
    # json.loads accepts the bare NaN token; it must still not become a score.
    out = _write(tmp_path / "out", '{"cv_score": NaN, "evaluator_version": "ev-3"}')
    metrics, score, problem = read_metrics_json(out)
    assert score is None and problem == "metrics_cv_score_invalid"
    assert metrics["evaluator_version"] == "ev-3"


def test_invalid_metrics_json_is_reported_separately(tmp_path):
    out = _write(tmp_path / "out", "{not json")
    metrics, score, problem = read_metrics_json(out)
    assert metrics == {} and score is None and problem == "metrics_json_invalid"


def test_absent_metrics_json(tmp_path):
    metrics, score, problem = read_metrics_json(tmp_path / "nowhere")
    assert metrics == {} and score is None and problem == "metrics_json_absent"


def test_valid_metrics_score(tmp_path):
    out = _write(tmp_path / "out", {"cv_score": 0.75})
    _, score, problem = read_metrics_json(out)
    assert score == pytest.approx(0.75) and problem == ""


# ---- fakes for loop-level tests ---------------------------------------------
class ScriptedRunner:
    """Returns pre-scripted (success, score) outcomes and writes real artifacts."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def run(self, code, *, data_dir, out_dir, exp_id):
        success, score = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "submission.csv").write_text("id,y\n1,0\n", encoding="utf-8")
        (out / "metrics.json").write_text(json.dumps({"cv_score": score}), encoding="utf-8")
        return RunResult(success, score, out_dir=out_dir,
                         artifacts=[str(out / "submission.csv"), str(out / "metrics.json")],
                         exit_code=0 if success else 1)


def _loop(tmp_path, runner, *, iterations, direction="maximize", **config):
    ctx = TaskContext("integrity-task", "tabular", "classification", "accuracy", direction, target_column="y")
    gen = VariationGenerator(client=FakeLLMClient([_full_script(0.5)]))
    (tmp_path / "data").mkdir(exist_ok=True)
    return EvolutionLoop(
        ctx, data_dir=str(tmp_path / "data"), work_dir=tmp_path / "work", runner=runner,
        generator=gen, memory=RetrospectiveMemoryStore(tmp_path / "mem.json"),
        config=EvolutionConfig(max_iterations=iterations, stagnation_patience=99, **config),
    )


def _audit(tmp_path, exp_id):
    base = tmp_path / "work" / exp_id
    return (json.loads((base / "validation_contract.json").read_text(encoding="utf-8")),
            json.loads((base / "claim_audit.json").read_text(encoding="utf-8")))


# ---- P2-3 at loop level -----------------------------------------------------
def test_non_finite_runner_score_is_never_promoted(tmp_path):
    loop = _loop(tmp_path, ScriptedRunner([(True, float("nan")), (True, 0.6)]), iterations=2)
    summary = loop.run()
    first = summary["iterations"][0]
    assert first["success"] is False and first["promoted"] is False and first["cv_score"] is None
    # The NaN run must not become an incumbent that blocks every later candidate.
    assert summary["best_cv_score"] == pytest.approx(0.6)


# ---- P0-2: acceptance gate uses the incumbent, not the candidate -------------
def test_acceptance_gate_compares_against_incumbent_not_candidate(tmp_path):
    loop = _loop(tmp_path, ScriptedRunner([(True, 0.80), (True, 0.70)]), iterations=2)
    loop.run()
    exp_ids = [it.exp_id for it in loop.iterations]
    contract, _ = _audit(tmp_path, exp_ids[1])
    assert contract["acceptance_basis"] == "incumbent"
    assert contract["baseline_exp_id"] == exp_ids[0]
    assert contract["acceptance_criteria"]["cv_score"]["min"] == pytest.approx(0.80 + 1e-4)
    assert contract["acceptance"]["passed"] is False


def test_acceptance_gate_minimize_direction(tmp_path):
    loop = _loop(tmp_path, ScriptedRunner([(True, 0.30), (True, 0.20)]), iterations=2, direction="minimize")
    loop.run()
    contract, _ = _audit(tmp_path, loop.iterations[1].exp_id)
    assert contract["acceptance_criteria"]["cv_score"]["max"] == pytest.approx(0.30 - 1e-4)
    assert contract["acceptance"]["passed"] is True


def test_first_candidate_without_incumbent_or_threshold_is_not_accepted(tmp_path):
    loop = _loop(tmp_path, ScriptedRunner([(True, 0.9)]), iterations=1)
    loop.run()
    contract, _ = _audit(tmp_path, loop.iterations[0].exp_id)
    assert contract["acceptance_basis"] == "none"
    assert contract["acceptance"]["passed"] is False


def test_fixed_threshold_is_used_when_no_incumbent(tmp_path):
    loop = _loop(tmp_path, ScriptedRunner([(True, 0.9)]), iterations=1, acceptance_threshold=0.95)
    loop.run()
    contract, _ = _audit(tmp_path, loop.iterations[0].exp_id)
    assert contract["acceptance_basis"] == "fixed_threshold"
    assert contract["acceptance_criteria"]["cv_score"]["min"] == pytest.approx(0.95)
    assert contract["acceptance"]["passed"] is False


# ---- P0-2: claims are not self-certified -------------------------------------
def test_successful_run_does_not_claim_ablation_or_mechanism_evidence(tmp_path):
    loop = _loop(tmp_path, ScriptedRunner([(True, 0.80), (True, 0.90)]), iterations=2)
    loop.run()
    exp_id = loop.iterations[1].exp_id
    assert loop.iterations[1].promoted is True
    contract, audit = _audit(tmp_path, exp_id)
    assert audit["evidence"]["has_mechanistic_evidence"] is False
    assert audit["completed_ablations"] == []
    assert audit["score_verification"] == "self_reported_unverified"
    assert audit["promotion_claim_allowed"] is False
    assert contract["score_verification"] == "self_reported_unverified"


def test_summary_marks_scores_unverified(tmp_path):
    summary = _loop(tmp_path, ScriptedRunner([(True, 0.8)]), iterations=1).run()
    assert summary["score_verification"] == "self_reported_unverified"
    assert summary["promotion_claims_allowed"] is False
    assert math.isfinite(summary["best_cv_score"])
