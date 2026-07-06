"""Local benchmark manager for MLE-Bench-style task tracking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkTask:
    task_id: str
    competition_name: str
    task_type: str
    modality: str
    metric: str
    train_data_path: str | None
    test_data_path: str | None
    sample_submission_path: str | None
    time_budget_hours: float
    gpu_required: bool
    allowed_models: list[str]
    submission_limit: int | None
    evaluation_mode: str
    medal_thresholds: dict[str, Any]
    baseline_score: float | None
    target_score: float | None
    status: str
    notes: str = ""


@dataclass
class BenchmarkResult:
    task_id: str
    best_exp_id: str | None
    best_cv_score: float | None
    best_public_score: float | None
    best_private_score: float | None
    valid_submission: bool
    medal: str
    rank_percentile: float | None
    runtime_hours: float
    num_experiments: int
    num_failed_runs: int
    num_recoveries: int
    reproducibility_score: float
    auditability_score: float
    claim_drift_detected: bool
    final_report_path: str | None
    artifacts_path: str | None
    gap_to_mlevolve: str | float | None
    next_improvement_plan: str
    failure_reasons: list[str] = field(default_factory=list)
    top30_reached: bool = False
    official_rank: int | None = None
    leaderboard_team_count: int | None = None
    official_submission_ref: str | None = None
    mlebench_comparable: bool = False
    overclaim_risk: str = "insufficient_official_benchmark_evidence"


def load_tasks(path: str | Path) -> list[BenchmarkTask]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [BenchmarkTask(**item) for item in payload]


def _load_results(path: Path) -> list[BenchmarkResult]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("results", [])
    return [BenchmarkResult(**item) for item in records]


def _save_results(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2), encoding="utf-8")


def save_result(path: str | Path, result: BenchmarkResult) -> Path:
    output_path = Path(path)
    results = [item for item in _load_results(output_path) if item.task_id != result.task_id]
    results.append(result)
    _save_results(output_path, results)
    return output_path


def compute_valid_submission_rate(results: list[BenchmarkResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for item in results if item.valid_submission) / len(results)


def compute_medal_rate(results: list[BenchmarkResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for item in results if item.medal in {"bronze", "silver", "gold"}) / len(results)


def compute_gap_to_target(current_rate: float, target_rate: float) -> float:
    return target_rate - current_rate


# MLEvolve paper reference targets (for gap tracking only; not a comparability claim).
MLEVOLVE_REFERENCE_TARGETS = {
    "valid_submission_rate": 0.95,
    "medal_rate": 0.6133,
    "bronze_rate": 0.50,
}


def compute_mlevolve_gap_analysis(
    results: list[BenchmarkResult],
    *,
    total_target_tasks: int = 75,
    targets: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Per-metric gap vs MLEvolve reference targets, with an honest comparability guard.

    The gap numbers are informational. ``comparable`` is only True once at least
    ``total_target_tasks`` tasks have been evaluated, so the system never claims
    MLEvolve-level performance from a partial run (rules: no overclaiming).
    """
    targets = targets or MLEVOLVE_REFERENCE_TARGETS
    summary = summarize_results(results)
    evaluated = summary["evaluated_tasks"]
    comparable = evaluated >= total_target_tasks

    metrics: dict[str, Any] = {}
    for name, target in targets.items():
        current = float(summary.get(name, 0.0))
        metrics[name] = {
            "current": current,
            "target": target,
            "gap": compute_gap_to_target(current, target),
            "reached": current >= target,
        }

    return {
        "evaluated_tasks": evaluated,
        "total_target_tasks": total_target_tasks,
        "coverage": evaluated / total_target_tasks if total_target_tasks else 0.0,
        "comparable": comparable,
        "metrics": metrics,
        "claim_boundary": (
            "Comparable benchmark coverage reached; gaps are meaningful."
            if comparable
            else f"Only {evaluated}/{total_target_tasks} tasks evaluated - gaps are indicative, not a MLEvolve-level claim."
        ),
    }


def summarize_results(results: list[BenchmarkResult]) -> dict[str, Any]:
    medal_counts = {"none": 0, "bronze": 0, "silver": 0, "gold": 0, "unknown": 0}
    for result in results:
        medal_counts[result.medal] = medal_counts.get(result.medal, 0) + 1

    total = len(results)
    return {
        "evaluated_tasks": total,
        "valid_submission_rate": compute_valid_submission_rate(results),
        "medal_rate": compute_medal_rate(results),
        "top30_rate": sum(1 for item in results if item.top30_reached) / total if total else 0.0,
        "bronze_rate": medal_counts.get("bronze", 0) / total if total else 0.0,
        "silver_rate": medal_counts.get("silver", 0) / total if total else 0.0,
        "gold_rate": medal_counts.get("gold", 0) / total if total else 0.0,
        "average_runtime_hours": sum(item.runtime_hours for item in results) / total if total else 0.0,
        "average_experiments_per_task": sum(item.num_experiments for item in results) / total if total else 0.0,
        "claim_drift_rate": sum(1 for item in results if item.claim_drift_detected) / total if total else 0.0,
        "benchmark_overclaim_risk_count": sum(1 for item in results if item.overclaim_risk and item.overclaim_risk != "none"),
        "medal_counts": medal_counts,
        "failed_tasks": [item.task_id for item in results if not item.valid_submission or item.num_failed_runs > 0],
    }


def export_benchmark_report(
    path: str | Path,
    tasks: list[BenchmarkTask],
    results: list[BenchmarkResult],
    mlevolve_target_medal_rate: float | None = None,
    evidence_note: str = "local demo results only; not an official 75-task benchmark.",
) -> Path:
    summary = summarize_results(results)
    target_line = "not_configured"
    if mlevolve_target_medal_rate is not None:
        gap = compute_gap_to_target(summary["medal_rate"], mlevolve_target_medal_rate)
        target_line = f"target={mlevolve_target_medal_rate:.4f}, gap={gap:.4f}"

    failed_rows = []
    for result in results:
        if result.valid_submission and result.num_failed_runs == 0:
            continue
        reasons = ", ".join(result.failure_reasons) if result.failure_reasons else "not_classified"
        failed_rows.append(f"| {result.task_id} | benchmark | {reasons} | pending | {result.artifacts_path or ''} |")
    if not failed_rows:
        failed_rows.append("| none | none | none | none | none |")

    claim_status = "no_not_reached"
    if results and mlevolve_target_medal_rate is not None and summary["medal_rate"] >= mlevolve_target_medal_rate:
        claim_status = "partial_current_evaluated_tasks_only"

    body = [
        "# Benchmark Gap Report",
        "",
        "## 1. Current Task Coverage",
        "",
        f"- planned_tasks: {len(tasks)}",
        f"- evaluated_tasks: {summary['evaluated_tasks']}",
        f"- completed_tasks: {sum(1 for item in results if item.valid_submission)}",
        f"- failed_tasks: {len(summary['failed_tasks'])}",
        "",
        "## 2. Current valid_submission_rate",
        "",
        f"- valid_submission_rate: {summary['valid_submission_rate']:.4f}",
        f"- top30_rate: {summary['top30_rate']:.4f}",
        f"- evidence: {evidence_note}",
        "",
        "## 3. Current medal_rate",
        "",
        f"- medal_rate: {summary['medal_rate']:.4f}",
        f"- bronze_rate: {summary['bronze_rate']:.4f}",
        f"- silver_rate: {summary['silver_rate']:.4f}",
        f"- gold_rate: {summary['gold_rate']:.4f}",
        "- medal_judgement_mode: proxy / demo",
        "",
        "## 4. Gap to MLEvolve Target",
        "",
        f"- mlevolve_reference_setting: {target_line}",
        "- current_setting: local skeleton demo, not comparable to MLEvolve.",
        "- aligned_or_not: no",
        "- gap_summary: evidence insufficient; full 75-task benchmark not run.",
        "",
        "## 5. Failed Task List",
        "",
        "| task_id | stage | failure_reason | recovery_status | artifact |",
        "| --- | --- | --- | --- | --- |",
        *failed_rows,
        "",
        "## 6. Main Failure Reasons",
        "",
        "- insufficient search",
        "- missing ablation",
        "- claim drift",
        "",
        "## 7. Next Optimization Plan",
        "",
        "- priority_1: convert real historical experiments into benchmark results.",
        "- priority_2: connect SearchGraph and RetrospectiveMemory to workstation runs.",
        "- priority_3: add official/proxy medal judgement adapters.",
        "- owner_agent: benchmark_manager + search_controller + claim_audit_agent",
        "- expected_metric: higher valid_submission_rate before medal_rate.",
        "- rollback_condition: any benchmark overclaim or missing artifact blocks external claim.",
        "",
        "## 8. External Claim Permission",
        "",
        f"- claim_status: {claim_status}",
        "- allowed_external_claim: aligned with MLEvolve as a target and continuously verified through benchmark gates.",
        "- missing_evidence: complete 75-task results, official medal thresholds, private scores, full claim audits.",
        "",
    ]

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(body), encoding="utf-8")
    return output_path
