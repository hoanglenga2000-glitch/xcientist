from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

GateDecision = Literal["pending", "approved", "rejected"]


@dataclass(slots=True)
class GateRecord:
    gate_id: str
    task_id: str
    gate_type: str
    decision: GateDecision = "pending"
    required_evidence: list[str] = field(default_factory=list)
    reviewer: str = "human"
    metadata: dict[str, Any] = field(default_factory=dict)

