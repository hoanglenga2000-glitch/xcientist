from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EvidenceRecord:
    evidence_id: str
    task_id: str
    artifact_path: Path
    artifact_type: str
    source: str
    bound_claims: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ArtifactManifest:
    task_id: str
    run_id: str | None
    artifacts: list[EvidenceRecord]

