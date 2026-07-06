from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from .json_utils import append_jsonl, write_json

GateStatus = Literal["pending", "approved", "rejected", "request_more_evidence"]


@dataclass(slots=True)
class RuntimeGate:
    gate_id: str
    task_id: str
    run_id: str | None
    gate_type: str
    triggered_by: str
    reason: str
    required_evidence: list[str]
    risk_level: str
    status: GateStatus = "pending"
    reviewer: str | None = None
    decision_comment: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    decided_at: str | None = None


class GateEngine:
    def __init__(self, task_id: str, run_id: str | None = None) -> None:
        self.task_id = task_id
        self.run_id = run_id
        self.gates: list[RuntimeGate] = []

    def create_gate(
        self,
        gate_type: str,
        *,
        triggered_by: str,
        reason: str,
        required_evidence: list[str] | None = None,
        risk_level: str = "medium",
    ) -> RuntimeGate:
        gate = RuntimeGate(
            gate_id=f"gate_{uuid4().hex[:10]}",
            task_id=self.task_id,
            run_id=self.run_id,
            gate_type=gate_type,
            triggered_by=triggered_by,
            reason=reason,
            required_evidence=required_evidence or [],
            risk_level=risk_level,
        )
        self.gates.append(gate)
        return gate

    def decide(self, gate_id: str, status: GateStatus, *, reviewer: str, comment: str) -> RuntimeGate:
        gate = self.get(gate_id)
        gate.status = status
        gate.reviewer = reviewer
        gate.decision_comment = comment
        gate.decided_at = datetime.now().isoformat(timespec="seconds")
        return gate

    def get(self, gate_id: str) -> RuntimeGate:
        for gate in self.gates:
            if gate.gate_id == gate_id:
                return gate
        raise KeyError(gate_id)

    def require_approved(self, gate_type: str, action: str) -> None:
        matching = [gate for gate in self.gates if gate.gate_type == gate_type]
        if not matching or matching[-1].status != "approved":
            raise RuntimeError(f"{action} is blocked until {gate_type} is approved.")

    def write(self, output_dir: Path) -> None:
        write_json(output_dir / "gate_engine.json", {"gates": self.gates})
        audit_log = output_dir / "gate_audit_log.jsonl"
        if audit_log.exists():
            audit_log.unlink()
        for gate in self.gates:
            append_jsonl(audit_log, gate)
