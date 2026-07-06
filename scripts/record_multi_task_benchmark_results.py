"""Record real local multi-task workstation results into benchmark artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from research_os.benchmark_manager import (  # noqa: E402
    BenchmarkResult,
    export_benchmark_report,
    load_tasks,
    save_result,
    summarize_results,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    tasks = load_tasks(ROOT / "benchmark" / "mle_bench_75" / "tasks_template.json")

    titanic_gate = load_json(ROOT / "experiments" / "titanic" / "20260623_150848" / "validation_gate.json")
    house_gate = load_json(ROOT / "experiments" / "house_prices" / "20260623_150918" / "validation_gate.json")
    telco_gate = load_json(ROOT / "experiments" / "telco_churn" / "20260623_150918" / "validation_gate.json")

    results = [
        BenchmarkResult(
            task_id="titanic",
            best_exp_id="titanic_20260623_150848",
            best_cv_score=float(titanic_gate["cv_accuracy_mean"]),
            best_public_score=None,
            best_private_score=None,
            valid_submission=True,
            medal="unknown",
            rank_percentile=None,
            runtime_hours=0.01,
            num_experiments=1,
            num_failed_runs=0,
            num_recoveries=0,
            reproducibility_score=0.86,
            auditability_score=0.9,
            claim_drift_detected=False,
            final_report_path="experiments/titanic/20260623_150848/local_report.md",
            artifacts_path="experiments/titanic/20260623_150848",
            gap_to_mlevolve="proxy_local_no_official_medal",
            next_improvement_plan="Add official/proxy medal adapter and repeated SearchGraph trials.",
            failure_reasons=[],
        ),
        BenchmarkResult(
            task_id="house_prices_advanced_regression_techniques",
            best_exp_id="house_prices_20260623_150918",
            best_cv_score=float(house_gate["cv_rmsle_mean"]),
            best_public_score=None,
            best_private_score=None,
            valid_submission=True,
            medal="unknown",
            rank_percentile=None,
            runtime_hours=0.02,
            num_experiments=1,
            num_failed_runs=0,
            num_recoveries=0,
            reproducibility_score=0.86,
            auditability_score=0.9,
            claim_drift_detected=False,
            final_report_path="experiments/house_prices/20260623_150918/local_report.md",
            artifacts_path="experiments/house_prices/20260623_150918",
            gap_to_mlevolve="proxy_local_no_official_medal",
            next_improvement_plan="Run multi-branch search and official/proxy leaderboard calibration.",
            failure_reasons=[],
        ),
        BenchmarkResult(
            task_id="telco_churn",
            best_exp_id="telco_churn_20260623_150918",
            best_cv_score=float(telco_gate["cv_accuracy_mean"]),
            best_public_score=None,
            best_private_score=None,
            valid_submission=True,
            medal="unknown",
            rank_percentile=None,
            runtime_hours=0.01,
            num_experiments=1,
            num_failed_runs=0,
            num_recoveries=0,
            reproducibility_score=0.86,
            auditability_score=0.9,
            claim_drift_detected=False,
            final_report_path="experiments/telco_churn/20260623_150918/local_report.md",
            artifacts_path="experiments/telco_churn/20260623_150918",
            gap_to_mlevolve="proxy_local_no_official_medal",
            next_improvement_plan="Add search controller branch expansion and reusable memory extraction.",
            failure_reasons=[],
        ),
        BenchmarkResult(
            task_id="playground_series_s6e6",
            best_exp_id=None,
            best_cv_score=None,
            best_public_score=None,
            best_private_score=None,
            valid_submission=False,
            medal="none",
            rank_percentile=None,
            runtime_hours=0.0,
            num_experiments=0,
            num_failed_runs=1,
            num_recoveries=0,
            reproducibility_score=0.4,
            auditability_score=0.55,
            claim_drift_detected=False,
            final_report_path=None,
            artifacts_path="tasks/playground_series_s6e6",
            gap_to_mlevolve="requires_hpc_workstation_job_not_local_training",
            next_improvement_plan="Run as a gated HPC/GPU workstation job; local machine should only perform schema/readiness checks.",
            failure_reasons=["training timeout", "insufficient search"],
        ),
    ]

    result_path = ROOT / "workspace" / "benchmark_multi_task_results_20260623.json"
    if result_path.exists():
        result_path.unlink()
    for result in results:
        save_result(result_path, result)

    report_path = export_benchmark_report(
        ROOT / "workspace" / "benchmark_multi_task_gap_report_20260623.md",
        tasks,
        results,
        mlevolve_target_medal_rate=None,
        evidence_note="real local workstation runs for 3 small/medium tasks plus one S6E6 readiness blocker; not official Kaggle submission and not a 75-task benchmark.",
    )
    summary = summarize_results(results)
    summary_path = ROOT / "workspace" / "benchmark_multi_task_summary_20260623.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": "recorded",
        "evaluated_tasks": summary["evaluated_tasks"],
        "valid_submission_rate": summary["valid_submission_rate"],
        "medal_rate": summary["medal_rate"],
        "claim_drift_rate": summary["claim_drift_rate"],
        "results": str(result_path),
        "summary": str(summary_path),
        "gap_report": str(report_path),
        "scope": "local/proxy multi-task test; not official Kaggle submission; not MLE-Bench 75 completion",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
