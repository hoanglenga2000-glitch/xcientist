"""
Four-Layer Architecture Integration Bridge

Layer 1 (bottom): Multi-Agent Workstation
  - Context partitioning, artifact-based workflow, multi-agent orchestration
  - Kaggle/HPC task execution, evidence tracking, gate management

Layer 2: MLEvolve Search Controller
  - Progressive MCGS with graph-based cross-branch search
  - Retrospective Memory (cold-start KB + dynamic experience)
  - Hierarchical planning with adaptive code generation

Layer 3: XCIENTIST Research Harness
  - Idea contracts with hypothesis/mechanism grounding
  - Validation contracts with pre/post-conditions
  - Claim audit to prevent drift between idea/code/conclusion

Layer 4: Island Model (HarnessEngine)
  - Parallel strategy islands for stagnation escape
  - feature_engineering / model_diversity / ensemble_blend tracks
  - Cross-task pattern injection for knowledge transfer

Architecture:
  XCIENTISTHarness (audit) + Island Model (parallel explore)
       |
       v
  MCEvolveSearchEngine (search) + RetrospectiveMemory
       |
       v
  AgentOrchestrator (execution - existing workstation)
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .mlevolve_search import (
    MCEvolveSearchEngine, ExpansionType, CodingMode, SearchPhase, SearchNode
)
from .retrospective_memory import RetrospectiveMemory, KnowledgeBase, GlobalExperience
from .harness_optimizer import HarnessEngine
from ..research_harness.contracts import (
    IdeaContract, ValidationContract, ExperimentRecord, ClaimAudit, ClaimStatus, RiskLevel
)
from ..research_harness.harness import XCIENTISTHarness


class MLEvolveHarnessBridge:
    """
    Three-layer integration bridge that coordinates:
    - MLEvolve search controller (how to search)
    - XCIENTIST research harness (how to validate)
    - Existing workstation orchestrator (how to execute)

    This bridge ensures:
    1. Every search expansion is validated by XCIENTIST contracts
    2. All experiments have auditable hypothesis->code->conclusion chains
    3. Memory accumulates across experiments for self-evolution
    """

    def __init__(self, task_id: str, metric: str = "accuracy",
                 metric_direction: str = "maximize",
                 total_budget_hours: float = 12.0,
                 workspace_root: Optional[Path] = None):
        self.task_id = task_id
        self.metric = metric
        self.metric_direction = metric_direction
        self.workspace_root = workspace_root or Path(".")

        # Initialize three layers
        self.search = MCEvolveSearchEngine(
            task_id=task_id, metric=metric,
            metric_direction=metric_direction,
            total_budget_hours=total_budget_hours,
            workspace_root=workspace_root
        )
        self.memory = RetrospectiveMemory(task_id=task_id, workspace_root=workspace_root)
        self.harness = XCIENTISTHarness(
            task_id=task_id, workspace_root=workspace_root,
            metric=metric, metric_direction=metric_direction
        )
        # Layer 4: Island Model for parallel exploration when stagnated
        self.island_model = HarnessEngine(task_id=task_id, n_islands=3)

        # Historical best for score gating
        self.protected_best_score: Optional[float] = None
        self.protected_best_run: Optional[str] = None

    def set_protected_baseline(self, score: float, run_id: str):
        self.protected_best_score = score
        self.protected_best_run = run_id
        self.harness.best_score = score
        self.search.global_best_score = score

    def get_search_context(self, query: str = "") -> dict:
        """Get memory context for planning the next search step."""
        return self.memory.get_context(query)

    def propose_search_idea(self, title: str, hypothesis: str,
                            mechanism: str, expected_effect: str = "",
                            risk_level: str = "medium",
                            component_changes: Optional[list[str]] = None,
                            literature_refs: Optional[list[str]] = None
                            ) -> IdeaContract:
        """Propose a new idea with both MLEvolve strategy and XCIENTIST grounding."""
        idea = self.harness.propose_idea(
            title=title, hypothesis=hypothesis, mechanism=mechanism,
            literature_grounding=literature_refs or ["MLEvolve knowledge base"],
            expected_effect=expected_effect, risk_level=risk_level,
            component_changes=component_changes or []
        )
        return idea

    def create_experiment_contract(self, idea: IdeaContract,
                                   baseline_score: Optional[float] = None,
                                   validation_plan: str = "",
                                   ablations: Optional[list[str]] = None
                                   ) -> ValidationContract:
        """Create a validation contract for an experiment."""
        return self.harness.create_validation_contract(
            idea=idea,
            baseline_score=baseline_score or self.protected_best_score,
            validation_plan=validation_plan,
            ablations=ablations or []
        )

    def record_and_audit(self, idea: IdeaContract, contract: ValidationContract,
                         baseline_value: float, experiment_value: float,
                         code_content: str = "", code_changes: Optional[list[str]] = None,
                         ablation_results: Optional[dict] = None,
                         implementation_summary: str = "",
                         conclusion: str = "",
                         evidence_artifacts: Optional[list[str]] = None
                         ) -> tuple[ExperimentRecord, ClaimAudit]:
        """Record experiment and audit for claim drift."""
        record = self.harness.record_experiment(
            idea=idea, contract=contract,
            baseline_value=baseline_value,
            experiment_value=experiment_value,
            implementation_summary=implementation_summary,
            code_changes=code_changes or [],
            ablation_results=ablation_results or {},
            conclusion=conclusion,
            evidence_artifacts=evidence_artifacts or []
        )

        audit = self.harness.audit_claim(record, code_content=code_content)

        # Record in memory for future search
        if experiment_value is not None:
            delta = experiment_value - baseline_value
            self.memory.record_experience(
                run_id=record.experiment_id,
                record_type="experiment",
                content=json.dumps({
                    "idea": idea.title,
                    "mechanism": idea.mechanism,
                    "baseline": baseline_value,
                    "experiment": experiment_value,
                    "delta": delta,
                    "audit_passed": audit.audit_passed,
                    "claim_status": record.claim_status.value
                }),
                score_delta=max(0, delta) if delta else None,
                tags=[self.task_id, idea.risk_level.value]
            )

            # Record errors if any
            if audit.claim_drift_detected:
                self.memory.record_experience(
                    run_id=record.experiment_id,
                    record_type="error",
                    content=f"Claim drift: {audit.drift_description}",
                    tags=["claim_drift", self.task_id]
                )

        return record, audit

    def accept_if_valid(self, idea: IdeaContract, record: ExperimentRecord,
                        audit: ClaimAudit, claim_boundary: str = "") -> bool:
        """Accept a claim only if audit passes."""
        if not audit.audit_passed:
            return False
        self.harness.accept_claim(idea, record, audit, claim_boundary)
        if record.experiment_value is not None:
            # Update search engine best
            self.search.global_best_score = record.experiment_value
        return True

    def trigger_island_exploration(self) -> dict:
        """Trigger island model parallel exploration when MCGS stagnates."""
        if not self.search._is_globally_stagnant():
            return {"triggered": False, "reason": "not_stagnant"}

        # Cross-task patterns for inspiration
        cross_patterns = self.memory.get_cross_task_patterns(
            source_task=self.task_id, min_delta=0.001
        )
        pattern_hints = [
            {"pattern": p.get("content","")[:200], "delta": p.get("score_delta",0)}
            for p in cross_patterns[:5]
        ]

        # Run one iteration of island model
        island_result = self.island_model.run_iteration()

        # Record island strategies in memory
        for island_id, strategy in island_result.get("active_strategies", {}).items():
            self.memory.record_experience(
                run_id=f"island_{island_id}_{int(time.time())}",
                record_type="island_strategy",
                content=json.dumps(strategy),
                tags=["island_model", self.task_id, island_id]
            )

        return island_result

    def get_next_action(self) -> dict:
        """Get the next recommended action based on search state."""
        phase = self.search.search_phase
        scheduler = self.search.scheduler

        recent_successes = self.memory.get_successful_patterns()
        recent_errors = self.memory.get_error_patterns()

        # Island model status
        island_status = self.island_model.get_status()

        # Trigger island model if globally stagnated
        should_trigger_islands = (
            self.search._is_globally_stagnant() and
            island_status.get("active_islands", 0) == 0
        )

        return {
            "search_phase": phase.value,
            "alpha": scheduler.alpha,
            "progress": scheduler.progress,
            "num_branches": len(self.search.graph.branches),
            "evaluated_nodes": self.search.graph.evaluated_nodes,
            "global_best": self.search.global_best_score,
            "protected_best": self.protected_best_score,
            "stagnation": self.search.global_stagnation_count,
            "should_explore": scheduler.should_explore_new_branch(),
            "should_trigger_island_model": should_trigger_islands,
            "top_branches": [
                {
                    "branch_id": bid,
                    "best_score": max(
                        (n.score or 0) for n in self.search.graph.get_branch_nodes(bid)
                        if n.score is not None
                    ) if any(n.score is not None for n in self.search.graph.get_branch_nodes(bid))
                    else None,
                    "node_count": len(nids)
                }
                for bid, nids in list(self.search.graph.branches.items())[-8:]
            ],
            "successful_patterns": [
                {"type": r.record_type, "delta": r.score_delta, "content": r.content[:200]}
                for r in recent_successes[:3]
            ],
            "recent_errors": [
                {"type": r.record_type, "content": r.content[:200]}
                for r in recent_errors[:3]
            ],
            "knowledge_context": self.memory.get_context(
                f"Improve {self.metric} for {self.task_id}"
            )["knowledge_base"][:500],
            "harness_summary": {
                "total_ideas": len(self.harness.ideas),
                "total_experiments": len(self.harness.experiments),
                "claim_drifts": self.harness.claim_drift_count,
                "validated_claims": sum(
                    1 for idea in self.harness.ideas.values()
                    if idea.status == ClaimStatus.BOUNDED
                )
            },
            "island_model": island_status
        }

    def full_status(self) -> dict:
        """Get complete status across all four layers."""
        return {
            "schema": "academic_research_os.four_layer_bridge_status.v2",
            "task_id": self.task_id,
            "metric": self.metric,
            "generated_at": datetime.now().isoformat(),
            "search_engine": self.search.to_manifest(),
            "harness": self.harness.to_summary(),
            "island_model": self.island_model.get_status(),
            "cross_task_patterns": len(self.memory.get_cross_task_patterns(self.task_id)),
            "next_action": self.get_next_action()
        }

    def save_full_status(self) -> Path:
        path = self.workspace_root / "workspace" / "three_layer" / f"status_{self.task_id}_{int(time.time())}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.full_status(), indent=2, ensure_ascii=False, default=str))
        return path
