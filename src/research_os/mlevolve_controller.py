"""MLEvolve-inspired control helpers for the workstation Search Controller.

This module is intentionally lightweight: it imports MLEvolve concepts into the
workstation contract without launching MLEvolve or bypassing AgentOrchestrator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


TOP30_TARGET_PERCENTILE = 0.30
DEFAULT_OFFICIAL_SUBMIT_BUDGET = 2


@dataclass
class RankGateResult:
    task_id: str
    run_id: str
    status: str
    rank_target_percentile: float = TOP30_TARGET_PERCENTILE
    official_rank: int | None = None
    leaderboard_team_count: int | None = None
    rank_percentile: float | None = None
    public_score: float | None = None
    official_submission_ref: str | None = None
    top30_reached: bool = False
    decision: str = "proxy_only"
    claim_boundary: str = "No official rank claim without Kaggle response artifact."
    blocker: str | None = None


@dataclass
class SearchControllerDecision:
    task_id: str
    run_id: str
    selected_branch: str
    exploration_stage: str
    code_generation_mode: str
    metric: str
    metric_direction: str
    rank_target_percentile: float = TOP30_TARGET_PERCENTILE
    official_submit_budget: int = DEFAULT_OFFICIAL_SUBMIT_BUDGET
    cross_branch_references: list[dict[str, Any]] = field(default_factory=list)
    memory_reuse_records: list[dict[str, Any]] = field(default_factory=list)
    stagnation_reason: str | None = None
    expected_delta: str = "positive local CV/proxy delta against best-so-far"
    rollback_condition: str = "hold candidate if it misses required artifacts, fails risk checks, or does not improve best-so-far"


def choose_code_generation_mode(
    *,
    has_parent: bool,
    branch_stagnant: bool = False,
    global_stagnant: bool = False,
    failure_count: int = 0,
) -> str:
    if not has_parent:
        return "Base"
    if branch_stagnant or global_stagnant or failure_count >= 2:
        return "Diff"
    return "Stepwise"


def classify_workstation_status(rank_gate: RankGateResult | dict[str, Any] | None = None, *, has_official_response: bool = False) -> str:
    top30_reached = False
    if isinstance(rank_gate, dict):
        top30_reached = bool(rank_gate.get("top30_reached"))
    elif rank_gate is not None:
        top30_reached = bool(rank_gate.top30_reached)
    if top30_reached:
        return "top30_reached"
    if has_official_response:
        return "top30_failed"
    return "proxy_only"


def evaluate_rank_gate(
    *,
    task_id: str,
    run_id: str,
    official_submission: dict[str, Any] | None,
    target_percentile: float = TOP30_TARGET_PERCENTILE,
) -> dict[str, Any]:
    if not official_submission:
        return asdict(
            RankGateResult(
                task_id=task_id,
                run_id=run_id,
                status="blocked_by_gate",
                blocker="Kaggle response artifact is required before official rank evaluation.",
            )
        )

    rank = official_submission.get("rank")
    total = official_submission.get("leaderboard_team_count")
    percentile = official_submission.get("rank_percentile")
    if percentile is None and isinstance(rank, int) and isinstance(total, int) and total > 0:
        percentile = rank / total
    top30 = isinstance(percentile, (int, float)) and float(percentile) <= target_percentile
    decision = "top30_reached" if top30 else "top30_failed"
    return asdict(
        RankGateResult(
            task_id=task_id,
            run_id=run_id,
            status="official_submitted",
            rank_target_percentile=target_percentile,
            official_rank=rank,
            leaderboard_team_count=total,
            rank_percentile=float(percentile) if isinstance(percentile, (int, float)) else None,
            public_score=official_submission.get("public_score"),
            official_submission_ref=str(official_submission.get("submission_ref") or ""),
            top30_reached=top30,
            decision=decision,
            claim_boundary="Official public leaderboard evidence only; private leaderboard or medal requires separate artifact.",
        )
    )


def build_benchmark_claim_gate(
    *,
    evaluated_tasks: int,
    total_target_tasks: int = 75,
    medal_rate: float | None = None,
    mlevolve_target_medal_rate: float = 0.653,
) -> dict[str, Any]:
    comparable = evaluated_tasks >= total_target_tasks
    reached = comparable and medal_rate is not None and medal_rate >= mlevolve_target_medal_rate
    return {
        "schema": "academic_research_os.benchmark_claim_gate.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluated_tasks": evaluated_tasks,
        "total_target_tasks": total_target_tasks,
        "mlevolve_target_medal_rate": mlevolve_target_medal_rate,
        "current_medal_rate": medal_rate,
        "mlebench_comparable": comparable,
        "claim_status": "allow" if reached else ("partial" if evaluated_tasks else "reject"),
        "allowed_conclusion": (
            "Comparable 75-task benchmark target reached."
            if reached
            else "Preliminary or insufficient benchmark evidence; do not claim MLEvolve-level performance."
        ),
    }


def build_search_controller_decision(
    *,
    task_id: str,
    run_id: str,
    selected_branch: str,
    exploration_stage: str,
    metric: str,
    metric_direction: str,
    has_parent: bool,
    branch_stagnant: bool = False,
    global_stagnant: bool = False,
    failure_count: int = 0,
    cross_branch_references: list[dict[str, Any]] | None = None,
    memory_reuse_records: list[dict[str, Any]] | None = None,
    official_submit_budget: int = DEFAULT_OFFICIAL_SUBMIT_BUDGET,
) -> dict[str, Any]:
    mode = choose_code_generation_mode(
        has_parent=has_parent,
        branch_stagnant=branch_stagnant,
        global_stagnant=global_stagnant,
        failure_count=failure_count,
    )
    stagnation_reason = None
    if global_stagnant:
        stagnation_reason = "global_stagnation_requires_cross_branch_fusion"
    elif branch_stagnant:
        stagnation_reason = "branch_stagnation_requires_reference_or_diff"
    elif failure_count:
        stagnation_reason = f"recent_failure_count={failure_count}"
    decision = SearchControllerDecision(
        task_id=task_id,
        run_id=run_id,
        selected_branch=selected_branch,
        exploration_stage=exploration_stage,
        code_generation_mode=mode,
        metric=metric,
        metric_direction=metric_direction,
        official_submit_budget=official_submit_budget,
        cross_branch_references=cross_branch_references or [],
        memory_reuse_records=memory_reuse_records or [],
        stagnation_reason=stagnation_reason,
    )
    payload = asdict(decision)
    payload.update(
        {
            "schema": "academic_research_os.search_controller_decision.v2",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "selected_task": task_id,
            "top30_target_status": "rank_gate_required_after_official_submission",
            "mlevolve_alignment": {
                "progressive_mcgs": True,
                "retrospective_memory": True,
                "adaptive_code_generation": True,
                "workstation_orchestrator_required": True,
            },
        }
    )
    return payload
