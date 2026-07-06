"""Demo the MLE-Bench-style benchmark manager without external services."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from research_os.benchmark_manager import (  # noqa: E402
    BenchmarkResult,
    compute_medal_rate,
    compute_valid_submission_rate,
    export_benchmark_report,
    load_tasks,
    save_result,
    summarize_results,
)


def main() -> int:
    tasks_path = ROOT / "benchmark" / "mle_bench_75" / "tasks_template.json"
    tasks = load_tasks(tasks_path)

    results = [
        BenchmarkResult(
            task_id="playground_series_s6e6",
            best_exp_id="EXP005",
            best_cv_score=0.7653,
            best_public_score=None,
            best_private_score=None,
            valid_submission=True,
            medal="unknown",
            rank_percentile=None,
            runtime_hours=1.5,
            num_experiments=6,
            num_failed_runs=1,
            num_recoveries=1,
            reproducibility_score=0.72,
            auditability_score=0.81,
            claim_drift_detected=True,
            final_report_path="reports/EXPERIMENT_EXP005.md",
            artifacts_path="examples/research_os_demo",
            gap_to_mlevolve="not_comparable_demo",
            next_improvement_plan="Run required blend ablations and official/proxy medal adapter.",
            failure_reasons=["missing ablation", "claim drift"],
        ),
        BenchmarkResult(
            task_id="titanic",
            best_exp_id="DEMO_BASELINE",
            best_cv_score=0.82,
            best_public_score=None,
            best_private_score=None,
            valid_submission=True,
            medal="unknown",
            rank_percentile=None,
            runtime_hours=0.2,
            num_experiments=2,
            num_failed_runs=0,
            num_recoveries=0,
            reproducibility_score=0.65,
            auditability_score=0.58,
            claim_drift_detected=False,
            final_report_path=None,
            artifacts_path="tasks/titanic",
            gap_to_mlevolve="proxy_only",
            next_improvement_plan="Attach validation contract and claim audit before counting as benchmark evidence.",
            failure_reasons=[],
        ),
        BenchmarkResult(
            task_id="digit_recognizer",
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
            reproducibility_score=0.0,
            auditability_score=0.2,
            claim_drift_detected=False,
            final_report_path=None,
            artifacts_path=None,
            gap_to_mlevolve="not_started",
            next_improvement_plan="Add image task baseline and GPU job template.",
            failure_reasons=["dependency/environment failure", "insufficient search"],
        ),
    ]

    results_path = ROOT / "workspace" / "benchmark_demo_results.json"
    for result in results:
        save_result(results_path, result)

    summary = summarize_results(results)
    report_path = export_benchmark_report(
        ROOT / "workspace" / "benchmark_demo_gap_report.md",
        tasks,
        results,
        mlevolve_target_medal_rate=None,
    )

    print("Benchmark manager demo")
    print(f"- planned template tasks: {len(tasks)}")
    print(f"- demo evaluated tasks: {summary['evaluated_tasks']}")
    print(f"- valid_submission_rate: {compute_valid_submission_rate(results):.4f}")
    print(f"- medal_rate: {compute_medal_rate(results):.4f}")
    print(f"- claim_drift_rate: {summary['claim_drift_rate']:.4f}")
    print(f"- results: {results_path}")
    print(f"- gap_report: {report_path}")
    print("当前只是 demo，不代表已经完成 75 个任务。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
