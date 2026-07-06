from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from .artifact_registry import ArtifactRecord
from .json_utils import write_json

EvidenceStatus = Literal["verified", "pending", "outdated", "rejected"]


@dataclass(slots=True)
class EvidenceItem:
    evidence_id: str
    task_id: str
    run_id: str | None
    type: str
    path: str
    hash: str
    generated_by: str
    generated_at: str
    linked_claims: list[str] = field(default_factory=list)
    status: EvidenceStatus = "verified"


@dataclass(slots=True)
class Claim:
    claim_id: str
    text: str
    source: str
    evidence_ids: list[str]
    confidence: float
    reviewer_status: str
    risk_level: str


class EvidenceGraph:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.evidence: list[EvidenceItem] = []
        self.claims: list[Claim] = []

    def ingest_artifacts(self, artifacts: list[ArtifactRecord]) -> list[EvidenceItem]:
        for artifact in artifacts:
            self.evidence.append(
                EvidenceItem(
                    evidence_id=f"evidence_{uuid4().hex[:10]}",
                    task_id=artifact.task_id,
                    run_id=artifact.run_id,
                    type=artifact.type,
                    path=artifact.path,
                    hash=artifact.hash,
                    generated_by=artifact.created_by,
                    generated_at=artifact.created_at,
                    linked_claims=list(artifact.linked_claims),
                    status="verified" if artifact.hash else "pending",
                )
            )
        return self.evidence

    def bind_claim(self, text: str, *, source: str, evidence_ids: list[str], confidence: float, risk_level: str) -> Claim:
        claim = Claim(
            claim_id=f"claim_{uuid4().hex[:10]}",
            text=text,
            source=source,
            evidence_ids=evidence_ids,
            confidence=confidence,
            reviewer_status="bound" if evidence_ids else "needs_evidence",
            risk_level=risk_level,
        )
        self.claims.append(claim)
        for evidence in self.evidence:
            if evidence.evidence_id in evidence_ids and claim.claim_id not in evidence.linked_claims:
                evidence.linked_claims.append(claim.claim_id)
        return claim

    def needs_evidence(self) -> list[Claim]:
        return [claim for claim in self.claims if not claim.evidence_ids]

    def write(self, output_dir: Path) -> None:
        write_json(
            output_dir / "evidence_index.json",
            {
                "task_id": self.task_id,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "evidence": self.evidence,
                "claims": self.claims,
                "needs_evidence": [claim.claim_id for claim in self.needs_evidence()],
            },
        )
