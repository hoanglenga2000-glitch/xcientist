"""Search graph primitives for MLEvolve-style experiment planning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class ExperimentNode:
    exp_id: str
    parent_id: str | None
    branch_type: str
    task_name: str
    hypothesis: str
    implementation_summary: str
    code_path: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    cv_score: float | None = None
    public_score: float | None = None
    risk_flags: list[str] = field(default_factory=list)
    decision: str = "needs_review"
    next_actions: list[str] = field(default_factory=list)
    created_at: str = ""
    metric_name: str = "cv_score"
    metric_direction: str = "maximize"
    run_success: bool = True
    promoted: bool = False
    parent_score: float | None = None
    promotion_delta: float | None = None
    promotion_reason: str = ""
    official_rank: int | None = None
    leaderboard_team_count: int | None = None
    rank_percentile: float | None = None
    top30_reached: bool = False
    official_submission_ref: str | None = None


@dataclass
class SearchGraph:
    task_id: str
    root_exp_id: str
    nodes: dict[str, ExperimentNode] = field(default_factory=dict)
    edges: list[dict[str, str]] = field(default_factory=list)
    selected_next_branch: str | None = None
    exploration_stage: str = "exploration"
    metric_name: str = "cv_score"
    metric_direction: str = "maximize"
    best_exp_id: str | None = None
    promotion_history: list[dict[str, Any]] = field(default_factory=list)
    reference_edges: list[dict[str, Any]] = field(default_factory=list)


    def _node_score(self, node: ExperimentNode, metric: str | None = None) -> float | None:
        metric_name = metric or self.metric_name
        if metric_name == "cv_score":
            value = node.cv_score
        elif metric_name == "public_score":
            value = node.public_score
        else:
            value = node.metrics.get(metric_name)
        if value is None:
            return None
        return float(value)

    def _is_better(self, candidate: float, parent: float, min_delta: float = 0.0, direction: str | None = None) -> bool:
        metric_direction = (direction or self.metric_direction).lower()
        if metric_direction in {"minimize", "lower", "lower_is_better"}:
            return candidate < parent - min_delta
        return candidate > parent + min_delta

    def decide_promotion(
        self,
        candidate_exp_id: str,
        parent_exp_id: str | None = None,
        metric: str | None = None,
        direction: str | None = None,
        min_delta: float = 0.0,
        required_artifacts: list[str] | None = None,
        run_success: bool = True,
    ) -> dict[str, Any]:
        """Apply best-so-far promotion gate.

        This is the MLEvolve layer invariant: a candidate can be promoted only
        when (1) its run completed successfully, (2) its score improves under the
        declared metric direction, and (3) required evidence artifacts exist.
        Otherwise the parent/best-so-far is preserved and the candidate becomes a
        negative/hold node for retrospective memory.

        ``run_success`` is a hard precondition: a run that exits non-zero (crash,
        OOM, timeout) must never be promoted even if it flushed a valid score and
        artifacts to disk before dying. This closes the "failed-but-artifacts-
        exist" hole that a remote GPU kill can otherwise slip through.
        """
        if candidate_exp_id not in self.nodes:
            raise KeyError(f"Unknown candidate experiment: {candidate_exp_id}")
        candidate = self.nodes[candidate_exp_id]
        metric_name = metric or candidate.metric_name or self.metric_name
        metric_direction = direction or candidate.metric_direction or self.metric_direction
        parent_id = parent_exp_id or candidate.parent_id or self.best_exp_id
        if parent_id == candidate_exp_id:
            parent_id = None
        parent = self.nodes.get(parent_id) if parent_id else None
        candidate_score = self._node_score(candidate, metric_name)
        # A failed parent must not anchor the promotion baseline: a run that
        # crashed (even one that flushed a score before dying) is not a real
        # best-so-far, so its score is treated as absent. Otherwise a crashed
        # high score silently suppresses genuinely successful successors.
        parent_score = (self._node_score(parent, metric_name)
                        if parent and getattr(parent, "run_success", True) else None)
        required_artifacts = required_artifacts or []
        artifact_names = {str(item.get("path") or item.get("artifact_type") or item.get("name") or "") for item in candidate.artifacts}
        missing_artifacts = [item for item in required_artifacts if not any(item in name for name in artifact_names)]

        promoted = False
        reason = ""
        delta = None
        if not run_success:
            # Hard precondition: a failed run is never promotable. We still compute
            # delta (when scores exist) so retrospective memory sees how close it got.
            reason = "run did not complete successfully; not promotable"
            if candidate_score is not None and parent_score is not None:
                delta = (parent_score - candidate_score) if metric_direction.lower() in {"minimize", "lower", "lower_is_better"} else (candidate_score - parent_score)
        elif candidate_score is None:
            reason = f"candidate metric {metric_name} missing"
        elif parent_score is None:
            promoted = not missing_artifacts
            reason = "no parent score; candidate promoted as first scored node" if promoted else "required artifacts missing"
            delta = None
        elif missing_artifacts:
            reason = "required artifacts missing: " + ", ".join(missing_artifacts)
            delta = (parent_score - candidate_score) if metric_direction.lower() in {"minimize", "lower", "lower_is_better"} else (candidate_score - parent_score)
        else:
            promoted = self._is_better(candidate_score, parent_score, min_delta=min_delta, direction=metric_direction)
            delta = (parent_score - candidate_score) if metric_direction.lower() in {"minimize", "lower", "lower_is_better"} else (candidate_score - parent_score)
            reason = "candidate improves best-so-far" if promoted else "candidate does not improve best-so-far; preserve parent"

        candidate.parent_score = parent_score
        candidate.promotion_delta = delta
        candidate.metric_name = metric_name
        candidate.metric_direction = metric_direction
        candidate.promoted = promoted
        candidate.decision = "promote" if promoted else "hold"
        candidate.promotion_reason = reason
        if promoted:
            self.best_exp_id = candidate.exp_id
        elif self.best_exp_id is None and parent is not None and getattr(parent, "run_success", True):
            self.best_exp_id = parent.exp_id
        decision = {
            "candidate_exp_id": candidate.exp_id,
            "parent_exp_id": parent.exp_id if parent else None,
            "metric": metric_name,
            "direction": metric_direction,
            "candidate_score": candidate_score,
            "parent_score": parent_score,
            "promotion_delta": delta,
            "promoted": promoted,
            "decision": candidate.decision,
            "reason": reason,
            "missing_artifacts": missing_artifacts,
            "best_exp_id_after": self.best_exp_id,
        }
        self.promotion_history.append(decision)
        return decision

    def add_node(self, node: ExperimentNode) -> None:
        self.nodes[node.exp_id] = node

    def add_edge(self, source: str, target: str, reason: str = "") -> None:
        if source not in self.nodes:
            raise KeyError(f"Unknown source experiment: {source}")
        if target not in self.nodes:
            raise KeyError(f"Unknown target experiment: {target}")
        self.edges.append({"source": source, "target": target, "reason": reason})

    def add_reference_edge(
        self,
        source: str,
        target: str,
        reason: str,
        reference_type: str = "cross_branch_reference",
        reusable_strategy: str = "",
    ) -> None:
        """Record MLEvolve-style information flow without changing ancestry."""
        if source not in self.nodes:
            raise KeyError(f"Unknown source experiment: {source}")
        if target not in self.nodes:
            raise KeyError(f"Unknown target experiment: {target}")
        self.reference_edges.append(
            {
                "source": source,
                "target": target,
                "reason": reason,
                "reference_type": reference_type,
                "reusable_strategy": reusable_strategy,
            }
        )

    def get_top_candidates(self, limit: int = 3, metric: str = "cv_score", higher_is_better: bool = True) -> list[ExperimentNode]:
        def score(node: ExperimentNode) -> float:
            raw_value = getattr(node, metric, None)
            if raw_value is None:
                raw_value = node.metrics.get(metric)
            if raw_value is None:
                return float("-inf") if higher_is_better else float("inf")
            return float(raw_value)

        return sorted(self.nodes.values(), key=score, reverse=higher_is_better)[:limit]

    def get_branch_diverse_top_candidates(
        self,
        limit: int = 5,
        max_per_branch: int = 2,
        metric: str | None = None,
    ) -> list[ExperimentNode]:
        """Return top nodes while preventing one branch from monopolizing fusion input."""
        metric_name = metric or self.metric_name
        higher = self.metric_direction.lower() not in {"minimize", "lower", "lower_is_better"}
        ranked = self.get_top_candidates(limit=max(limit * 3, limit), metric=metric_name, higher_is_better=higher)
        selected: list[ExperimentNode] = []
        branch_counts: dict[str, int] = {}
        for node in ranked:
            count = branch_counts.get(node.branch_type, 0)
            if count >= max_per_branch:
                continue
            selected.append(node)
            branch_counts[node.branch_type] = count + 1
            if len(selected) >= limit:
                break
        return selected

    def detect_stagnation(self, min_delta: float = 0.0001, window: int = 2) -> list[str]:
        by_branch: dict[str, list[ExperimentNode]] = {}
        for node in self.nodes.values():
            by_branch.setdefault(node.branch_type, []).append(node)

        stagnant: list[str] = []
        for branch_type, nodes in by_branch.items():
            scored = [node for node in nodes if node.cv_score is not None]
            scored.sort(key=lambda node: node.created_at or node.exp_id)
            if len(scored) <= window:
                continue
            recent = scored[-(window + 1) :]
            gains = [
                abs(float(recent[index].cv_score) - float(recent[index - 1].cv_score))
                for index in range(1, len(recent))
            ]
            if gains and max(gains) < min_delta:
                stagnant.append(branch_type)
        return stagnant

    def detect_global_stagnation(self, min_delta: float = 0.0001, window: int = 4) -> bool:
        scored = [node for node in self.nodes.values() if self._node_score(node) is not None]
        scored.sort(key=lambda node: node.created_at or node.exp_id)
        if len(scored) <= window:
            return False
        recent = scored[-(window + 1) :]
        best_seen = self._node_score(recent[0])
        if best_seen is None:
            return False
        improved = False
        for node in recent[1:]:
            score = self._node_score(node)
            if score is None:
                continue
            if self._is_better(score, best_seen, min_delta=min_delta):
                improved = True
                best_seen = score
        return not improved

    def to_dict(self) -> dict[str, Any]:
        higher = self.metric_direction.lower() not in {"minimize", "lower", "lower_is_better"}
        top_candidates = [node.exp_id for node in self.get_top_candidates(metric=self.metric_name, higher_is_better=higher)]
        return {
            "task_id": self.task_id,
            "root_exp_id": self.root_exp_id,
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": self.edges,
            "reference_edges": self.reference_edges,
            "top_candidates": top_candidates,
            "branch_diverse_top_candidates": [
                node.exp_id for node in self.get_branch_diverse_top_candidates(metric=self.metric_name)
            ],
            "stagnation_branches": self.detect_stagnation(),
            "global_stagnation": self.detect_global_stagnation(),
            "selected_next_branch": self.selected_next_branch,
            "exploration_stage": self.exploration_stage,
            "metric_name": self.metric_name,
            "metric_direction": self.metric_direction,
            "best_exp_id": self.best_exp_id,
            "promotion_history": self.promotion_history,
        }

    def export_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchGraph":
        """Reconstruct a graph from the dict produced by :meth:`to_dict`.

        This is the resume counterpart of ``export_json``: it rebuilds the
        auditable research state (node lineage, edges, best-so-far, promotion
        history) so a crashed/interrupted run can continue WITHOUT overwriting or
        fabricating what prior experiments actually proved. Only fields that
        ``to_dict`` persisted are restored; the selector's private visit/branch
        side-table is NOT part of the audited schema and is re-seeded separately.

        Tolerant of unknown/legacy keys in a serialized node (only fields the
        current ``ExperimentNode`` declares are applied), so an older
        ``search_graph.json`` still loads.
        """
        graph = cls(
            task_id=str(data.get("task_id", "")),
            root_exp_id=str(data.get("root_exp_id", "EXP000")),
            metric_name=str(data.get("metric_name", "cv_score")),
            metric_direction=str(data.get("metric_direction", "maximize")),
        )
        allowed = {f.name for f in fields(ExperimentNode)}
        for raw in data.get("nodes", []) or []:
            if not isinstance(raw, dict) or "exp_id" not in raw:
                continue
            node = ExperimentNode(**{k: v for k, v in raw.items() if k in allowed})
            graph.nodes[node.exp_id] = node
        graph.edges = [e for e in (data.get("edges") or []) if isinstance(e, dict)]
        graph.reference_edges = [e for e in (data.get("reference_edges") or []) if isinstance(e, dict)]
        graph.promotion_history = [p for p in (data.get("promotion_history") or []) if isinstance(p, dict)]
        graph.best_exp_id = data.get("best_exp_id")
        graph.selected_next_branch = data.get("selected_next_branch")
        graph.exploration_stage = str(data.get("exploration_stage", "exploration"))
        return graph

    @classmethod
    def load_json(cls, path: str | Path) -> "SearchGraph":
        """Load and reconstruct a graph from a ``search_graph.json`` file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
