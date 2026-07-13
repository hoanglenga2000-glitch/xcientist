"""分层共治 tests: the MCGS brain owns topology, the model owns the science.

These prove the reshaped control flow honors the plan's innovation #1/#2:
  * run_experiment REFUSES without a pending plan (selection is non-bypassable);
  * the plan's node/mode — NOT the model — decide the node's parent_id/branch_type;
  * evaluate_promotion backpropagates, so visit_count increments (UCT lives) and
    failures drive stagnation (DIVERSIFY becomes reachable);
  * a failed run persists the cleaned traceback to run_error.txt (观测修复).
"""
from __future__ import annotations

from pathlib import Path

from research_os.agent.tools import ResearchToolbox
from research_os.evolution_loop import RunResult
from research_os.mcgs_selector import MCGSSelector
from research_os.variation_generator import TaskContext


def _ctx() -> TaskContext:
    return TaskContext(task_name="nomad", modality="tabular", task_type="regression",
                       metric="rmsle", metric_direction="minimize", n_train=2400, n_test=600)


class _FakeRunner:
    def __init__(self, result: RunResult) -> None:
        self._result = result

    def run(self, code, *, data_dir, out_dir, exp_id) -> RunResult:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return self._result


def _ok(score: float) -> RunResult:
    return RunResult(success=True, cv_score=score, exit_code=0,
                     artifacts=["/x/metrics.json", "/x/submission.csv"], stdout_tail=f"CV_SCORE={score}")


def _toolbox(tmp_path, result: RunResult, *, mcgs=True) -> ResearchToolbox:
    selector = MCGSSelector(total_steps=8) if mcgs else None
    return ResearchToolbox(_ctx(), data_dir=str(tmp_path / "data"), work_dir=tmp_path / "exp",
                           runner=_FakeRunner(result), selector=selector)


def test_run_experiment_refuses_without_a_plan(tmp_path):
    tb = _toolbox(tmp_path, _ok(0.06))
    out = tb.dispatch("run_experiment", {"hypothesis": "h", "code": "print(1)"})
    assert out.ok is False
    assert "plan_next_experiment" in out.content
    assert tb.graph.nodes == {}  # nothing ran


def test_plan_decides_topology_not_the_model(tmp_path):
    """The model cannot smuggle a parent/mode — they come from the ExpansionPlan."""
    tb = _toolbox(tmp_path, _ok(0.06))
    # first plan → seeds root EXP000 (primary/Base, no parent)
    plan_out = tb.dispatch("plan_next_experiment", {})
    assert "EXP000" in plan_out.content
    # model tries to pass parent_exp_id/mode — they must be IGNORED
    tb.dispatch("run_experiment", {"hypothesis": "h", "code": "print(1)",
                                   "parent_exp_id": "bogus", "mode": "Diff"})
    node = tb.graph.nodes["EXP000"]
    assert node.parent_id is None          # root, from the plan (not "bogus")
    assert node.branch_type == "Base"      # from the plan (not the model's "Diff")


def test_promotion_backpropagates_and_makes_uct_live(tmp_path):
    """After promotion, the selector's visit_count increments (the bug MLEvolve
    never fixed) and best is tracked."""
    tb = _toolbox(tmp_path, _ok(0.06))
    tb.dispatch("plan_next_experiment", {})
    tb.dispatch("run_experiment", {"hypothesis": "baseline", "code": "print(1)"})
    assert tb.selector.visits.get("EXP000", 0) == 0   # not yet backpropagated
    tb.dispatch("evaluate_promotion", {"exp_id": "EXP000"})
    assert tb.selector.visits.get("EXP000", 0) == 1   # backprop incremented it
    assert tb.best_exp_id == "EXP000"


def test_failed_run_drives_stagnation_and_persists_real_error(tmp_path):
    """A failed run: not promotable, stagnation increments, and the cleaned real
    traceback (progress-bar noise stripped) is persisted to run_error.txt."""
    noisy = RunResult(
        success=False, cv_score=None, exit_code=1, artifacts=[],
        error="Traceback (most recent call last):\n"
              "  100%|=====| 50/50 [00:03<00:00, 14.2it/s]\n"
              "ValueError: shape mismatch: (3,) vs (4,)",
        stdout_tail="",
    )
    tb = _toolbox(tmp_path, noisy)
    tb.dispatch("plan_next_experiment", {})
    out = tb.dispatch("run_experiment", {"hypothesis": "h", "code": "print(1)"})
    assert out.ok is False
    tb.dispatch("evaluate_promotion", {"exp_id": "EXP000"})
    assert tb.best_exp_id is None
    assert tb.selector.global_stagnation >= 1          # failure drove stagnation
    err_file = tmp_path / "exp" / "EXP000" / "run_error.txt"
    assert err_file.exists()
    body = err_file.read_text(encoding="utf-8")
    assert "ValueError: shape mismatch" in body        # real error kept
    assert "it/s]" not in body                          # progress-bar frame stripped


def test_cold_start_reaches_multiple_branches_via_diversify(tmp_path):
    """The DIVERSIFY fix (innovation #2): a single-branch world that keeps stalling
    must eventually open a second branch — otherwise cross_branch/aggregation are
    unreachable dead code. Drive several plan→run→hold rounds and assert >1 branch."""
    # every run succeeds but never improves the (minimize) best after the first,
    # so branch_0 stalls and the selector must DIVERSIFY.
    scores = iter([0.06, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07])

    class _Degrading:
        def run(self, code, *, data_dir, out_dir, exp_id):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            return _ok(next(scores, 0.07))

    tb = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "data"), work_dir=tmp_path / "exp",
                         runner=_Degrading(), selector=MCGSSelector(total_steps=8, max_branches=3))
    for _ in range(7):
        tb.dispatch("plan_next_experiment", {})
        exp_id = f"EXP{tb._exp_counter:03d}"
        tb.dispatch("run_experiment", {"hypothesis": "h", "code": "print(1)"})
        tb.dispatch("evaluate_promotion", {"exp_id": exp_id})
    branches = set(tb.selector.branch_of.values())
    assert len(branches) >= 2, f"DIVERSIFY never opened a second branch: {branches}"


def test_no_mcgs_fallback_still_plans_trivially(tmp_path):
    """--no-mcgs: no selector, but plan_next_experiment still yields a trivial plan
    so run_experiment's uniform 'must-plan' contract holds."""
    tb = _toolbox(tmp_path, _ok(0.06), mcgs=False)
    plan_out = tb.dispatch("plan_next_experiment", {})
    assert plan_out.ok is True and "mcgs off" in plan_out.summary.lower()
    out = tb.dispatch("run_experiment", {"hypothesis": "h", "code": "print(1)"})
    assert out.ok is True
    assert "EXP000" in tb.graph.nodes
