from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ..adapters.storage_adapter import StorageAdapter
from ..schemas.evidence import ArtifactManifest, EvidenceRecord


class EvidenceService:
    def __init__(self, storage: StorageAdapter) -> None:
        self.storage = storage

    def collect_from_run(self, task_id: str, run_id: str, output_dir: Path) -> ArtifactManifest:
        artifacts = []
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                artifacts.append(
                    EvidenceRecord(
                        evidence_id=f"ev_{uuid4().hex[:10]}",
                        task_id=task_id,
                        artifact_path=path,
                        artifact_type=path.suffix.lstrip(".") or "file",
                        source="local_pipeline",
                        bound_claims=[],
                    )
                )
        manifest = ArtifactManifest(task_id, run_id, artifacts)
        self.storage.write_json(output_dir / "evidence_manifest.json", manifest)
        return manifest

    def bind_claim(self, evidence: EvidenceRecord, claim: str) -> EvidenceRecord:
        if claim not in evidence.bound_claims:
            evidence.bound_claims.append(claim)
        return evidence

