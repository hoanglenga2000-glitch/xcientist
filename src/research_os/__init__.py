"""Lightweight Research OS primitives for auditable MLE workflows."""

from .claim_audit import ClaimAudit, audit_claim, classify_drift_type, detect_benchmark_overclaim, detect_claim_drift
from .research_harness import run_research_os_harness
from .single_task_loop import run_single_task_closed_loop
from .debug_review_agents import build_debug_feedback, review_code_before_enqueue
from .feature_flags import FLAG_NAMES, flags_snapshot, is_enabled
from .benchmark_manager import (
    BenchmarkResult,
    BenchmarkTask,
    compute_gap_to_target,
    compute_medal_rate,
    compute_valid_submission_rate,
    export_benchmark_report,
    load_tasks,
    save_result,
    summarize_results,
)
from .retrospective_memory import MemoryRecord, RetrospectiveMemoryStore
from .search_graph import ExperimentNode, SearchGraph
from .mlevolve_controller import (
    DEFAULT_OFFICIAL_SUBMIT_BUDGET,
    TOP30_TARGET_PERCENTILE,
    build_benchmark_claim_gate,
    build_search_controller_decision,
    choose_code_generation_mode,
    classify_workstation_status,
    evaluate_rank_gate,
)
from .mlevolve_adapter import (
    MLEvolveSearchPolicy,
    build_workstation_alignment,
    extract_policy,
    load_mlevolve_config,
)
from .validation_contract import ValidationContract, check_required_artifacts, create_contract, evaluate_acceptance

__all__ = [
    "BenchmarkResult",
    "BenchmarkTask",
    "ClaimAudit",
    "ExperimentNode",
    "MemoryRecord",
    "MLEvolveSearchPolicy",
    "RetrospectiveMemoryStore",
    "SearchGraph",
    "TOP30_TARGET_PERCENTILE",
    "ValidationContract",
    "audit_claim",
    "build_debug_feedback",
    "detect_benchmark_overclaim",
    "build_benchmark_claim_gate",
    "build_search_controller_decision",
    "build_workstation_alignment",
    "check_required_artifacts",
    "choose_code_generation_mode",
    "classify_workstation_status",
    "classify_drift_type",
    "compute_gap_to_target",
    "compute_medal_rate",
    "compute_valid_submission_rate",
    "create_contract",
    "detect_claim_drift",
    "evaluate_acceptance",
    "evaluate_rank_gate",
    "extract_policy",
    "export_benchmark_report",
    "FLAG_NAMES",
    "flags_snapshot",
    "is_enabled",
    "review_code_before_enqueue",
    "run_research_os_harness",
    "run_single_task_closed_loop",
    "load_mlevolve_config",
    "load_tasks",
    "save_result",
    "summarize_results",
]
