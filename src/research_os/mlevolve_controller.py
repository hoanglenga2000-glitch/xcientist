"""MLEvolve-inspired control helpers for the workstation Search Controller.

This module is intentionally lightweight: it imports MLEvolve concepts into the
workstation contract without launching MLEvolve or bypassing AgentOrchestrator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from .feature_flags import (
    ff_benchmark_claim_gate,
    ff_cross_branch_fusion,
    ff_mcgs_c_decay,
    ff_mcgs_uct,
    ff_memory_faiss,
    ff_memory_vector,
    ff_multi_branch,
    ff_retro_memory,
    ff_search_budget_500,
    ff_stagnation_detect,
    flags_snapshot,
)


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
    claim_text: str = "",
) -> dict[str, Any]:
    # W1/W5: claim gate defaults ON. Paths that would imply beating MLEvolve/AI2
    # without 75-task + budget evidence are reject/partial + labeled
    # preliminary/partial/no (P0-HN-03). Explicit OFF still emits non-allow labels.
    from .claim_audit import detect_benchmark_overclaim

    claim_gate_enabled = ff_benchmark_claim_gate()
    comparable = evaluated_tasks >= total_target_tasks
    reached = comparable and medal_rate is not None and medal_rate >= mlevolve_target_medal_rate
    overclaim = detect_benchmark_overclaim(claim_text)
    if claim_gate_enabled and overclaim and not reached:
        claim_status = "reject"
        claim_label = "no"
        allowed_conclusion = (
            "Overclaim refused: beating AI2/MLEvolve requires aligned 75-task + "
            "budget evidence. Status=no."
        )
    elif reached and claim_gate_enabled:
        claim_status = "allow"
        claim_label = "comparable"
        allowed_conclusion = "Comparable 75-task benchmark target reached."
    elif evaluated_tasks:
        claim_status = "partial"
        claim_label = "partial" if claim_gate_enabled else "preliminary"
        allowed_conclusion = (
            "Preliminary or partial benchmark evidence; do not claim MLEvolve-level performance."
        )
    else:
        claim_status = "reject"
        claim_label = "preliminary" if not claim_gate_enabled else "no"
        allowed_conclusion = (
            "Preliminary or insufficient benchmark evidence; do not claim MLEvolve-level performance."
        )
    if not claim_gate_enabled and claim_status == "allow":
        # Emergency OFF must not silently license medal-rate claims.
        claim_status = "partial"
        claim_label = "preliminary"
        allowed_conclusion = (
            "Claim gate flag off (emergency); treat results as preliminary only."
        )
    return {
        "schema": "academic_research_os.benchmark_claim_gate.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluated_tasks": evaluated_tasks,
        "total_target_tasks": total_target_tasks,
        "mlevolve_target_medal_rate": mlevolve_target_medal_rate,
        "current_medal_rate": medal_rate,
        "mlebench_comparable": comparable,
        "claim_status": claim_status,
        "claim_label": claim_label,
        "overclaim_detected": overclaim,
        "allowed_conclusion": allowed_conclusion,
        "feature_flag_enabled": claim_gate_enabled,
        "feature_flag": "EVOMIND_FF_BENCHMARK_CLAIM_GATE",
    }



def record_retrospective_strategy(
    store: Any,
    record: Any,
) -> dict[str, Any]:
    """P0-MEM-01: write success/failure strategy when RETRO_MEMORY on; no-op when off."""
    enabled = ff_retro_memory()
    if not enabled:
        return {
            "written": False,
            "enabled": False,
            "feature_flag": "EVOMIND_FF_RETRO_MEMORY",
            "memory_id": getattr(record, "memory_id", None),
        }
    # Prefer gated store API when present; fall back to plain add_memory.
    if hasattr(store, "add_memory_gated"):
        written = bool(store.add_memory_gated(record))
    else:
        store.add_memory(record)
        written = True
    return {
        "written": written,
        "enabled": True,
        "feature_flag": "EVOMIND_FF_RETRO_MEMORY",
        "memory_id": getattr(record, "memory_id", None),
    }


def reload_memory_reuse_records(
    store: Any,
    *,
    task_type: str | None = None,
    query: str | None = None,
    k: int = 8,
) -> list[dict[str, Any]]:
    """P0-MEM-01/02: reload strategies for controller reuse.

    When RETRO_MEMORY off → []. When MEMORY_VECTOR on and query given → TF-IDF
    k-NN (same API shape ready for later FAISS). Otherwise exact task_type filter.
    """
    from dataclasses import asdict

    if not ff_retro_memory():
        return []
    if ff_memory_vector() and query:
        from .memory_vector_index import MemoryVectorIndex

        index = MemoryVectorIndex(store)
        return index.query(query, k=k, task_type=task_type)
    if hasattr(store, "reload_for_reuse"):
        records = store.reload_for_reuse(task_type)
    elif task_type is not None:
        records = store.retrieve_by_task_type(task_type)
    else:
        records = store._load()
    out: list[dict[str, Any]] = []
    for rec in records[-k:]:
        if hasattr(rec, "__dataclass_fields__"):
            out.append(asdict(rec))
        elif isinstance(rec, dict):
            out.append(dict(rec))
    return out


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
    memory_store: Any | None = None,
    memory_task_type: str | None = None,
    memory_query: str | None = None,
    official_submit_budget: int = DEFAULT_OFFICIAL_SUBMIT_BUDGET,
) -> dict[str, Any]:
    stagn_on = ff_stagnation_detect()
    fusion_on = ff_cross_branch_fusion()
    # Diff mode still honors caller stagnant args (patience); reason strings are
    # gated so flag-off keeps prior behavior without fusion work-order labels.
    mode = choose_code_generation_mode(
        has_parent=has_parent,
        branch_stagnant=branch_stagnant,
        global_stagnant=global_stagnant if stagn_on else False,
        failure_count=failure_count,
    )
    stagnation_reason = None
    if stagn_on and global_stagnant:
        stagnation_reason = "global_stagnation_requires_cross_branch_fusion"
    elif stagn_on and branch_stagnant:
        stagnation_reason = "branch_stagnation_requires_reference_or_diff"
    elif failure_count:
        stagnation_reason = f"recent_failure_count={failure_count}"
    # W3: when RETRO_MEMORY on and a store is provided, reload reuse records
    # unless the caller already supplied them. Flag off → empty / no new path.
    reuse = list(memory_reuse_records) if memory_reuse_records is not None else None
    if reuse is None:
        if memory_store is not None and ff_retro_memory():
            reuse = reload_memory_reuse_records(
                memory_store,
                task_type=memory_task_type,
                query=memory_query,
            )
        else:
            reuse = []
    decision = SearchControllerDecision(
        task_id=task_id,
        run_id=run_id,
        selected_branch=selected_branch,
        exploration_stage=exploration_stage,
        code_generation_mode=mode,
        metric=metric,
        metric_direction=metric_direction,
        official_submit_budget=official_submit_budget,
        cross_branch_references=(cross_branch_references or []) if fusion_on else [],
        memory_reuse_records=reuse,
        stagnation_reason=stagnation_reason,
    )
    # W2–W4: record search/branch/memory/budget/stagnation/fusion flags.
    mcgs_uct = ff_mcgs_uct()
    mcgs_c_decay = ff_mcgs_c_decay()
    multi_branch = ff_multi_branch()
    retro_memory = ff_retro_memory()
    memory_vector = ff_memory_vector()
    memory_faiss = ff_memory_faiss()
    search_budget_500 = ff_search_budget_500()
    cited_ids = [
        str(r.get("memory_id"))
        for r in reuse
        if isinstance(r, dict) and r.get("memory_id")
    ]
    payload = asdict(decision)
    payload.update(
        {
            "schema": "academic_research_os.search_controller_decision.v2",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "selected_task": task_id,
            "top30_target_status": "rank_gate_required_after_official_submission",
            "mlevolve_alignment": {
                "progressive_mcgs": bool(mcgs_uct),
                "retrospective_memory": bool(retro_memory),
                "memory_vector": bool(memory_vector),
                "adaptive_code_generation": True,
                "workstation_orchestrator_required": True,
                "multi_branch": bool(multi_branch),
                "mcgs_c_decay": bool(mcgs_c_decay),
                "search_budget_500": bool(search_budget_500),
                "stagnation_detect": bool(stagn_on),
                "cross_branch_fusion": bool(fusion_on),
            },
            "stagnation": {
                "enabled": bool(stagn_on),
                "feature_flag": "EVOMIND_FF_STAGNATION_DETECT",
                "reason": stagnation_reason,
            },
            "cross_branch_fusion": {
                "enabled": bool(fusion_on),
                "feature_flag": "EVOMIND_FF_CROSS_BRANCH_FUSION",
                "reference_count": len(decision.cross_branch_references),
            },
            "search_budget": {
                "enabled_500": bool(search_budget_500),
                "feature_flag": "EVOMIND_FF_SEARCH_BUDGET_500",
            },
            "feature_flags": flags_snapshot(),
            "c_decay": {
                "enabled": bool(mcgs_c_decay),
                "feature_flag": "EVOMIND_FF_MCGS_C_DECAY",
                "schedule": "piecewise_1.414_to_0.5_at_30_70_pct" if mcgs_c_decay else "constant",
            },
            "retrospective_memory": {
                "enabled": bool(retro_memory),
                "feature_flag": "EVOMIND_FF_RETRO_MEMORY",
                "reuse_count": len(reuse),
                "cited_memory_ids": cited_ids,
                "vector_retrieval": bool(memory_vector),
                "vector_feature_flag": "EVOMIND_FF_MEMORY_VECTOR",
                "faiss_requested": bool(memory_faiss),
                "faiss_feature_flag": "EVOMIND_FF_MEMORY_FAISS",
                "faiss_status": "not_configured" if memory_faiss else "disabled",
            },
        }
    )
    return payload
