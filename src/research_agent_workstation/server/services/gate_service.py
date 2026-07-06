from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ..adapters.storage_adapter import StorageAdapter
from ..schemas.gate import GateRecord


class GateService:
    def __init__(self, storage: StorageAdapter) -> None:
        self.storage = storage

    def create_gate(self, task_id: str, gate_type: str, evidence_ids: list[str], output_dir: Path) -> GateRecord:
        gate = GateRecord(f"gate_{uuid4().hex[:10]}", task_id, gate_type, "pending", evidence_ids)
        self.storage.write_json(output_dir / f"{gate_type}_gate.json", gate)
        return gate

    def approve(self, gate: GateRecord, output_dir: Path, reviewer: str = "human") -> GateRecord:
        gate.decision = "approved"
        gate.reviewer = reviewer
        self.storage.write_json(output_dir / f"{gate.gate_type}_gate.json", gate)
        return gate

    def submission_allowed(self, submission_check: dict) -> bool:
        return bool(submission_check.get("valid"))

