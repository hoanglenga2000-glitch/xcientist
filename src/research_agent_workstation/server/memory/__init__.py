from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.json_utils import write_json


@dataclass(slots=True)
class ExperimentMemoryEntry:
    experiment_id: str
    date: str
    model: str
    features: str
    cv_scheme: str
    cv_score: float | None
    public_score: float | None
    seed: str
    decision: str
    notes: str
    artifact_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentMemory:
    """Loads and queries historical experiment data as evidence for new runs."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self._entries: list[ExperimentMemoryEntry] = []
        self._load_from_experiment_log()

    def _load_from_experiment_log(self) -> None:
        log_path = self.workspace_root / "experiments" / "EXPERIMENT_LOG.md"
        if not log_path.exists():
            return
        text = log_path.read_text(encoding="utf-8")
        in_table = False
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("| experiment_id"):
                in_table = True
                continue
            if in_table and line.startswith("|") and "EXP" in line:
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 12:
                    try:
                        cv_score = self._parse_score(parts[5])
                        public_score = self._parse_score(parts[6])
                        self._entries.append(ExperimentMemoryEntry(
                            experiment_id=parts[0],
                            date=parts[1],
                            model=parts[2],
                            features=parts[3],
                            cv_scheme=parts[4],
                            cv_score=cv_score,
                            public_score=public_score,
                            seed=parts[7],
                            notes=parts[8],
                            artifact_path=parts[9],
                            decision=parts[10],
                        ))
                    except (ValueError, IndexError):
                        continue

    @staticmethod
    def _parse_score(value: str) -> float | None:
        v = value.strip().strip("`")
        if not v or v.lower() in ("not submitted", "-", "n/a", ""):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    def best_cv_score(self) -> float | None:
        scores = [e.cv_score for e in self._entries if e.cv_score is not None]
        return max(scores) if scores else None

    def best_public_score(self) -> float | None:
        scores = [e.public_score for e in self._entries if e.public_score is not None]
        return max(scores) if scores else None

    def best_experiment(self) -> ExperimentMemoryEntry | None:
        return max(
            (e for e in self._entries if e.public_score is not None),
            key=lambda e: e.public_score,
            default=None,
        )

    def get_by_id(self, experiment_id: str) -> ExperimentMemoryEntry | None:
        for e in self._entries:
            if e.experiment_id == experiment_id:
                return e
        return None

    def get_submit_candidates(self) -> list[ExperimentMemoryEntry]:
        return [e for e in self._entries if e.decision == "submit_candidate"]

    def get_baseline_entries(self) -> list[ExperimentMemoryEntry]:
        return [e for e in self._entries if e.decision == "keep"]

    def summary_for_agent_context(self, task_id: str | None = None) -> dict[str, Any]:
        best = self.best_experiment()
        return {
            "total_entries": len(self._entries),
            "best_cv_score": self.best_cv_score(),
            "best_public_score": self.best_public_score(),
            "best_experiment": {
                "id": best.experiment_id,
                "model": best.model,
                "public_score": best.public_score,
                "decision": best.decision,
            } if best else None,
            "submit_candidates": [e.experiment_id for e in self.get_submit_candidates()],
            "baseline_count": len(self.get_baseline_entries()),
            "decision_distribution": self._decision_distribution(),
        }

    def _decision_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for e in self._entries:
            dist[e.decision] = dist.get(e.decision, 0) + 1
        return dist

    def write_memory_manifest(self, output_dir: Path) -> Path:
        data = {
            "source": "experiments/EXPERIMENT_LOG.md",
            "entries": [
                {
                    "experiment_id": e.experiment_id,
                    "date": e.date,
                    "model": e.model,
                    "cv_score": e.cv_score,
                    "public_score": e.public_score,
                    "decision": e.decision,
                }
                for e in self._entries
            ],
            "summary": self.summary_for_agent_context(),
        }
        return write_json(output_dir / "experiment_memory.json", data)
