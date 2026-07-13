from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .json_utils import write_json


@dataclass(slots=True)
class ExperimentNode:
    run_id: str
    parent_run_id: str | None
    branch_id: str
    stage: str
    hypothesis: str
    plan: dict[str, Any]
    code_snapshot: str | None
    model: str | None
    params: dict[str, Any]
    metric: str
    score: float | None
    is_buggy: bool
    status: str
    artifacts: list[str] = field(default_factory=list)
    reward: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass(slots=True)
class ExperimentEdge:
    source_run_id: str
    target_run_id: str
    relation: str


class ExperimentGraph:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.nodes: list[ExperimentNode] = []
        self.edges: list[ExperimentEdge] = []

    def add_run(self, node: ExperimentNode, relation: str = "baseline") -> ExperimentNode:
        if node.parent_run_id:
            self.edges.append(ExperimentEdge(node.parent_run_id, node.run_id, relation))
        self.nodes.append(node)
        return node

    def best_run(self) -> ExperimentNode | None:
        candidates = [node for node in self.nodes if node.score is not None and not node.is_buggy]
        return min(candidates, key=lambda node: node.score) if candidates else None

    def failed_runs(self) -> list[ExperimentNode]:
        return [node for node in self.nodes if node.is_buggy or node.status == "failed"]

    def write(self, output_dir: Path) -> Path:
        return write_json(
            output_dir / "experiment_graph.json",
            {
                "task_id": self.task_id,
                "nodes": self.nodes,
                "edges": self.edges,
                "best_run_id": self.best_run().run_id if self.best_run() else None,
                "failed_run_ids": [node.run_id for node in self.failed_runs()],
            },
        )
