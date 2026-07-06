"""Guardrail tests for the deep-agent toolbox — the no-fabrication invariants.

These are the load-bearing tests: they prove that the safety-critical rulings are
made by deterministic code, NOT by anything the model says. Specifically:
  * a crashed run is never promotable, even with a great score + artifacts on disk;
  * a successful, improving run with required artifacts IS promoted;
  * a conclusion on a failed run is audit-rejected;
  * Kaggle submission is always blocked behind the human gate;
  * run_experiment's success is taken from the Runner, not from tool args.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from research_os.agent.tools import ResearchToolbox
from research_os.evolution_loop import RunResult
from research_os.variation_generator import TaskContext


def _ctx() -> TaskContext:
    return TaskContext(
        task_name="titanic", modality="tabular", task_type="classification",
        metric="accuracy", metric_direction="maximize", n_train=891, n_test=418,
    )


class _FakeRunner:
    """Returns a scripted RunResult; the toolbox must trust IT, not the model."""

    def __init__(self, result: RunResult) -> None:
        self._result = result
        self.calls = 0

    def run(self, code, *, data_dir, out_dir, exp_id) -> RunResult:
        self.calls += 1
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return self._result


def _toolbox(tmp_path, result: RunResult) -> ResearchToolbox:
    return ResearchToolbox(
        _ctx(), data_dir=str(tmp_path / "data"), work_dir=tmp_path / "exp",
        runner=_FakeRunner(result),
    )


def _plan_and_run(tb: ResearchToolbox, **run_args):
    """Honor the mandatory-plan guardrail: plan (trivial in --no-mcgs mode) then run."""
    tb.dispatch("plan_next_experiment", {})
    return tb.dispatch("run_experiment", run_args)


def test_crashed_run_with_score_and_artifacts_is_never_promoted(tmp_path):
    """The classic remote-GPU-kill hole: a run exits non-zero (137=OOM) but had
    already flushed a high score + all artifacts to disk. It must NOT be promoted."""
    crashed = RunResult(
        success=False, cv_score=0.99, exit_code=137,
        artifacts=["/x/metrics.json", "/x/submission.csv"],
        error="Killed", out_dir=str(tmp_path / "exp"),
    )
    tb = _toolbox(tmp_path, crashed)
    run_out = _plan_and_run(tb, hypothesis="h", code="print('CV_SCORE=0.99')")
    assert run_out.ok is False  # the runner said it failed
    exp_id = next(iter(tb.graph.nodes))
    # Even though the node carries cv_score=0.99 and both artifacts exist, the gate
    # refuses because the RECORDED run_success is False.
    promo = tb.dispatch("evaluate_promotion", {"exp_id": exp_id})
    node = tb.graph.nodes[exp_id]
    assert node.run_success is False
    assert node.promoted is False
    assert tb.best_exp_id is None
    assert "not promotable" in node.promotion_reason or "not promot" in promo.content.lower()


def test_successful_improving_run_is_promoted(tmp_path):
    ok = RunResult(
        success=True, cv_score=0.83, exit_code=0,
        artifacts=["/x/metrics.json", "/x/submission.csv"],
        out_dir=str(tmp_path / "exp"), stdout_tail="CV_SCORE=0.83",
    )
    tb = _toolbox(tmp_path, ok)
    _plan_and_run(tb, hypothesis="baseline", code="print(1)")
    exp_id = next(iter(tb.graph.nodes))
    promo = tb.dispatch("evaluate_promotion", {"exp_id": exp_id})
    assert tb.graph.nodes[exp_id].promoted is True
    assert tb.best_exp_id == exp_id
    assert "PROMOTED" in promo.content


def test_promotion_requires_artifacts(tmp_path):
    """A successful run missing required artifacts is not promotable."""
    no_artifacts = RunResult(success=True, cv_score=0.9, exit_code=0,
                             artifacts=[], out_dir=str(tmp_path / "exp"))
    tb = _toolbox(tmp_path, no_artifacts)
    _plan_and_run(tb, hypothesis="h", code="print(1)")
    exp_id = next(iter(tb.graph.nodes))
    tb.dispatch("evaluate_promotion", {"exp_id": exp_id})
    assert tb.graph.nodes[exp_id].promoted is False


def test_audit_rejects_conclusion_on_failed_run(tmp_path):
    crashed = RunResult(success=False, cv_score=0.99, exit_code=124,
                        artifacts=[], error="timeout", out_dir=str(tmp_path / "exp"))
    tb = _toolbox(tmp_path, crashed)
    _plan_and_run(tb, hypothesis="h", code="print(1)")
    exp_id = next(iter(tb.graph.nodes))
    audit = tb.dispatch("audit_conclusion",
                        {"exp_id": exp_id, "claim": "this method reaches 0.99 accuracy"})
    assert audit.ok is False
    assert "REJECT" in audit.content.upper() or "证据不足" in audit.content


def test_kaggle_submit_is_always_blocked(tmp_path):
    ok = RunResult(success=True, cv_score=0.9, exit_code=0,
                   artifacts=["/x/metrics.json", "/x/submission.csv"], out_dir=str(tmp_path / "exp"))
    tb = _toolbox(tmp_path, ok)
    _plan_and_run(tb, hypothesis="h", code="print(1)")
    exp_id = next(iter(tb.graph.nodes))
    out = tb.dispatch("submit_to_kaggle", {"exp_id": exp_id})
    assert out.ok is False
    assert "BLOCKED" in out.content and "human" in out.content.lower()


def test_run_experiment_rejects_empty_code(tmp_path):
    tb = _toolbox(tmp_path, RunResult(success=True, cv_score=0.9, exit_code=0))
    out = tb.dispatch("run_experiment", {"hypothesis": "h", "code": "   "})
    assert out.ok is False
    assert tb.graph.nodes == {}  # nothing was run/recorded


def test_finish_sets_finished_flag(tmp_path):
    tb = _toolbox(tmp_path, RunResult(success=True, cv_score=0.9, exit_code=0))
    out = tb.dispatch("finish", {"summary": "done"})
    assert out.finished is True


def test_unknown_tool_is_soft_error(tmp_path):
    tb = _toolbox(tmp_path, RunResult(success=True, cv_score=0.9, exit_code=0))
    out = tb.dispatch("nonexistent_tool", {})
    assert out.ok is False
    assert "unknown tool" in out.content


def test_all_specs_have_valid_schema(tmp_path):
    """Every advertised tool must have a name, description, and object schema."""
    tb = _toolbox(tmp_path, RunResult(success=True, cv_score=0.9, exit_code=0))
    specs = tb.specs()
    assert {s.name for s in specs} == {
        "inspect_data", "recommend_strategies", "read_memory", "search_memory", "read_search_tree",
        "plan_next_experiment", "run_experiment", "evaluate_promotion", "record_lesson",
        "audit_conclusion", "request_audit", "submit_to_kaggle", "finish",
    }
    for s in specs:
        assert s.description and s.input_schema.get("type") == "object"
        # every advertised tool has a handler
        assert getattr(tb, f"_tool_{s.name}", None) is not None
