"""Tests for the monitor-and-submit gate core.

Most important invariant: the tool NEVER allows auto-submission, and only marks a
run "ready_for_human_gate" when OOF clears bronze AND the submission is valid.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "monitor_and_submit", _ROOT / "scripts" / "monitor_and_submit.py"
)
mas = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mas
_SPEC.loader.exec_module(mas)


def test_never_auto_submits():
    d = mas.evaluate_submission_gate("titanic", 0.99, 0.79, higher_is_better=True)
    assert d.auto_submit_allowed is False


def test_ready_when_beats_bronze_and_valid():
    d = mas.evaluate_submission_gate(
        "titanic", 0.82, 0.794, higher_is_better=True, submission_check={"valid": True}
    )
    assert d.decision == "ready_for_human_gate"
    assert d.beats_bronze is True
    assert d.auto_submit_allowed is False


def test_rejected_when_below_bronze():
    d = mas.evaluate_submission_gate("digit-recognizer", 0.968, 0.986, higher_is_better=True)
    assert d.decision == "rejected"
    assert d.beats_bronze is False


def test_regression_lower_is_better():
    # RMSLE: lower is better. 0.13 <= bronze 0.14 -> passes.
    d = mas.evaluate_submission_gate("house_prices", 0.13, 0.14, higher_is_better=False,
                                     submission_check={"valid": True})
    assert d.beats_bronze is True
    assert d.decision == "ready_for_human_gate"


def test_regression_worse_than_bronze_rejected():
    d = mas.evaluate_submission_gate("house_prices", 0.20, 0.14, higher_is_better=False)
    assert d.beats_bronze is False
    assert d.decision == "rejected"


def test_blocked_when_no_oof():
    d = mas.evaluate_submission_gate("running", None, 0.5)
    assert d.decision == "blocked"


def test_blocked_when_submission_invalid_even_if_score_good():
    d = mas.evaluate_submission_gate(
        "titanic", 0.82, 0.794, higher_is_better=True, submission_check={"valid": False}
    )
    assert d.decision == "blocked"
    assert d.submission_valid is False


def test_unknown_bronze_blocks_not_rejects():
    d = mas.evaluate_submission_gate("newcomp", 0.5, None)
    assert d.decision == "blocked"
    assert d.beats_bronze is None


def test_summarize_run_status():
    decisions = [
        {"task_id": "a", "decision": "ready_for_human_gate", "oof_score": 0.8},
        {"task_id": "b", "decision": "rejected", "oof_score": 0.1},
        {"task_id": "c", "decision": "blocked", "oof_score": None},
    ]
    summary = mas.summarize_run_status(decisions)
    assert summary["total_tasks"] == 3
    assert summary["ready_for_human_gate"] == 1
    assert summary["rejected"] == 1
    assert summary["blocked"] == 1
    assert summary["ready_task_ids"] == ["a"]


def test_decision_json_serializable():
    import json
    d = mas.evaluate_submission_gate("titanic", 0.82, 0.794, submission_check={"valid": True})
    json.dumps(d.to_dict())
