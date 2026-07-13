"""Tool-guardrail tests: loop-breaking without touching research outcomes.

Prove: identical repeated failures get blocked; truncated run_experiment code is
refused before running; consecutive failures halt the session; idempotent read
spin is nudged; and a SUCCESS resets the failure streak.
"""
from __future__ import annotations

from research_os.agent.guardrails import ToolGuardrailController, _looks_truncated


def test_identical_repeated_failure_is_blocked():
    g = ToolGuardrailController(repeat_failure_limit=3)
    args = {"exp_id": "EXP000"}
    for _ in range(3):
        assert g.before_call("evaluate_promotion", args).blocked is False
        g.after_call("evaluate_promotion", args, ok=False)
    # 4th identical call: blocked (already failed 3×)
    decision = g.before_call("evaluate_promotion", args)
    assert decision.blocked is True
    assert "already" in decision.content and "failed" in decision.content


def test_success_resets_failure_streak():
    g = ToolGuardrailController(repeat_failure_limit=2, consecutive_failure_halt=3)
    args = {"exp_id": "E"}
    g.after_call("t", args, ok=False)
    g.after_call("t", args, ok=True)   # success clears the count for this fingerprint
    assert g.before_call("t", args).blocked is False
    assert g._consecutive_failures == 0


def test_truncated_code_is_refused_before_running():
    g = ToolGuardrailController()
    # unbalanced fence
    d1 = g.before_call("run_experiment", {"code": "```python\nprint(1)"})
    assert d1.blocked is True and "fence" in d1.content
    # dangling operator
    d2 = g.before_call("run_experiment", {"code": "x = 1 +"})
    assert d2.blocked is True and "truncated" in d2.content.lower()
    # a complete script passes
    d3 = g.before_call("run_experiment", {"code": "import os\nprint('CV_SCORE=0.5')\n"})
    assert d3.blocked is False


def test_looks_truncated_helper():
    assert _looks_truncated("```\ncode") is not None       # odd fences
    assert _looks_truncated("a = [") is not None            # dangling bracket
    assert _looks_truncated("print('done')\n") is None      # clean
    assert _looks_truncated("```py\nx=1\n```") is None       # balanced fences


def test_consecutive_failures_halt():
    g = ToolGuardrailController(consecutive_failure_halt=3)
    assert g.halted is False
    for i in range(3):
        g.after_call(f"tool{i}", {"i": i}, ok=False)  # different calls, all fail
    assert g.halted is True
    assert "consecutive" in g.halt_reason


def test_idempotent_read_spin_nudge():
    g = ToolGuardrailController(idempotent_repeat_limit=3)
    for _ in range(3):
        g.after_call("read_search_tree", {}, ok=True, result_content="SAME OUTPUT")
    nudge = g.idempotent_spin_warning("read_search_tree")
    assert nudge is not None and "re-reading" in nudge
    # a NON-idempotent tool never nudges
    assert g.idempotent_spin_warning("run_experiment") is None


def test_changing_read_result_does_not_nudge():
    g = ToolGuardrailController(idempotent_repeat_limit=3)
    for i in range(4):
        g.after_call("read_memory", {}, ok=True, result_content=f"different {i}")
    # results kept changing -> the repeat counter never accumulated
    assert g.idempotent_spin_warning("read_memory") is None
