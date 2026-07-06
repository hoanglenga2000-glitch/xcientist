"""Tests for the runtime GateEngine.

The gate flow is safety-critical: a Kaggle submission (or any high-risk action)
must be blocked until a matching gate is explicitly approved by a human reviewer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_agent_workstation.server.core.gate_engine import GateEngine, RuntimeGate


def _engine() -> GateEngine:
    return GateEngine(task_id="titanic", run_id="run_001")


def test_create_gate_defaults():
    engine = _engine()
    gate = engine.create_gate(
        "kaggle_submission",
        triggered_by="trainer_agent",
        reason="OOF passed bronze threshold",
    )
    assert isinstance(gate, RuntimeGate)
    assert gate.status == "pending"
    assert gate.risk_level == "medium"
    assert gate.required_evidence == []
    assert gate.gate_id.startswith("gate_")
    assert gate.created_at  # populated
    assert gate.decided_at is None


def test_submission_blocked_until_approved():
    engine = _engine()
    engine.create_gate("kaggle_submission", triggered_by="agent", reason="ready")
    # Pending gate must block the action.
    with pytest.raises(RuntimeError, match="blocked"):
        engine.require_approved("kaggle_submission", "submit to kaggle")


def test_submission_allowed_after_approval():
    engine = _engine()
    gate = engine.create_gate("kaggle_submission", triggered_by="agent", reason="ready")
    engine.decide(gate.gate_id, "approved", reviewer="human", comment="looks good")
    # Should not raise once approved.
    engine.require_approved("kaggle_submission", "submit to kaggle")
    assert gate.status == "approved"
    assert gate.reviewer == "human"
    assert gate.decided_at is not None


def test_rejected_gate_still_blocks():
    engine = _engine()
    gate = engine.create_gate("kaggle_submission", triggered_by="agent", reason="ready")
    engine.decide(gate.gate_id, "rejected", reviewer="human", comment="CV-public gap too large")
    with pytest.raises(RuntimeError):
        engine.require_approved("kaggle_submission", "submit to kaggle")


def test_require_approved_uses_latest_gate():
    engine = _engine()
    g1 = engine.create_gate("kaggle_submission", triggered_by="agent", reason="first")
    engine.decide(g1.gate_id, "approved", reviewer="human", comment="ok")
    # A newer pending gate of the same type must re-block the action.
    engine.create_gate("kaggle_submission", triggered_by="agent", reason="second attempt")
    with pytest.raises(RuntimeError):
        engine.require_approved("kaggle_submission", "submit to kaggle")


def test_require_approved_with_no_gate_blocks():
    engine = _engine()
    with pytest.raises(RuntimeError):
        engine.require_approved("kaggle_submission", "submit to kaggle")


def test_get_unknown_gate_raises():
    engine = _engine()
    with pytest.raises(KeyError):
        engine.get("gate_does_not_exist")


def test_write_persists_gates_and_audit_log(tmp_path: Path):
    engine = _engine()
    gate = engine.create_gate(
        "kaggle_submission",
        triggered_by="agent",
        reason="ready",
        required_evidence=["oof_metrics.json", "submission.csv"],
        risk_level="high",
    )
    engine.decide(gate.gate_id, "approved", reviewer="human", comment="approved")
    engine.write(tmp_path)

    gate_file = tmp_path / "gate_engine.json"
    audit_file = tmp_path / "gate_audit_log.jsonl"
    assert gate_file.exists()
    assert audit_file.exists()

    data = json.loads(gate_file.read_text(encoding="utf-8"))
    assert len(data["gates"]) == 1
    assert data["gates"][0]["status"] == "approved"
    assert data["gates"][0]["risk_level"] == "high"

    audit_lines = audit_file.read_text(encoding="utf-8").splitlines()
    assert len(audit_lines) == 1
    record = json.loads(audit_lines[0])
    assert record["gate_id"] == gate.gate_id


def test_write_is_idempotent_audit_log(tmp_path: Path):
    engine = _engine()
    engine.create_gate("data_audit", triggered_by="agent", reason="check")
    engine.write(tmp_path)
    engine.write(tmp_path)  # second write must not duplicate audit rows
    audit_file = tmp_path / "gate_audit_log.jsonl"
    assert len(audit_file.read_text(encoding="utf-8").splitlines()) == 1
