from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .json_utils import write_json


@dataclass(slots=True)
class ArtifactRecord:
    artifact_id: str
    task_id: str
    run_id: str | None
    name: str
    type: str
    path: str
    size: int
    hash: str
    created_by: str
    created_at: str
    linked_stage: str
    linked_claims: list[str] = field(default_factory=list)
    preview_available: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ArtifactRegistry:
    def __init__(self, task_id: str, run_id: str | None = None) -> None:
        self.task_id = task_id
        self.run_id = run_id
        self.artifacts: list[ArtifactRecord] = []

    def register(
        self,
        path: Path,
        *,
        artifact_type: str,
        created_by: str,
        linked_stage: str,
        linked_claims: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() and path.is_file() else ""
        record = ArtifactRecord(
            artifact_id=f"artifact_{uuid4().hex[:10]}",
            task_id=self.task_id,
            run_id=self.run_id,
            name=path.name,
            type=artifact_type,
            path=str(path),
            size=path.stat().st_size if path.exists() and path.is_file() else 0,
            hash=digest,
            created_by=created_by,
            created_at=datetime.now().isoformat(timespec="seconds"),
            linked_stage=linked_stage,
            linked_claims=linked_claims or [],
            preview_available=path.suffix.lower() in {".md", ".txt", ".json", ".csv", ".png"},
            metadata=metadata or {},
        )
        self.artifacts.append(record)
        return record

    def collect_directory(self, output_dir: Path, *, created_by: str = "local_pipeline", linked_stage: str = "training") -> list[ArtifactRecord]:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.name not in {"artifact_manifest.json"}:
                self.register(path, artifact_type=path.suffix.lstrip(".") or "file", created_by=created_by, linked_stage=linked_stage)
        return self.artifacts

    def write_manifest(self, output_dir: Path) -> Path:
        return write_json(output_dir / "artifact_manifest.json", {"task_id": self.task_id, "run_id": self.run_id, "artifacts": self.artifacts})
