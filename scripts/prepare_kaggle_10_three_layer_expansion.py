from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = "20260623"
OUT_DIR = ROOT / "benchmark" / "kaggle_10_self_evolution"
TASKS_PATH = OUT_DIR / f"tasks_{TODAY}.json"
WORK_ORDERS_PATH = ROOT / "workspace" / f"kaggle_10_agent_work_orders_{TODAY}.json"
EXPANSION_SUMMARY_PATH = ROOT / "workspace" / f"kaggle_10_three_layer_expansion_{TODAY}.json"
REPORT_PATH = ROOT / "reports" / f"KAGGLE_10_THREE_LAYER_EXPANSION_PLAN_{TODAY}.md"

REQUIRED_TASK_FIELDS = [
    "task_id", "competition_name", "task_type", "modality", "metric", "train_data_path",
    "test_data_path", "sample_submission_path", "time_budget_hours", "gpu_required",
    "allowed_models", "submission_limit", "evaluation_mode", "medal_thresholds",
    "baseline_score", "target_score", "status", "notes"
]

# The list is a benchmark expansion registry, not a claim that all ten are already completed.
TASKS: list[dict[str, Any]] = [
    {
        "task_id": "digit_recognizer",
        "competition_name": "digit-recognizer",
        "task_type": "classification",
        "modality": "image_flat_pixels",
        "metric": "accuracy",
        "train_data_path": "datasets/kaggle/digit_recognizer/train.csv",
        "test_data_path": "datasets/kaggle/digit_recognizer/test.csv",
        "sample_submission_path": "datasets/kaggle/digit_recognizer/sample_submission.csv",
        "time_budget_hours": 4,
        "gpu_required": False,
        "allowed_models": ["LogisticRegression+PCA", "ExtraTrees", "RandomForest", "SVM", "CNN_when_GPU_returns", "Ensemble"],
        "submission_limit": None,
        "evaluation_mode": "local_cv_proxy_no_submit",
        "medal_thresholds": {"bronze": None, "silver": None, "gold": None},
        "baseline_score": 0.9128809523809523,
        "target_score": 0.975,
        "status": "round2_completed_local_cv",
        "notes": "Actual local CPU run completed by workstation artifacts; no official submit or leaderboard claim."
    },
    {
        "task_id": "titanic",
        "competition_name": "titanic",
        "task_type": "classification",
        "modality": "tabular_mixed",
        "metric": "accuracy",
        "train_data_path": "datasets/kaggle/titanic/train.csv",
        "test_data_path": "datasets/kaggle/titanic/test.csv",
        "sample_submission_path": "datasets/kaggle/titanic/gender_submission.csv",
        "time_budget_hours": 2,
        "gpu_required": False,
        "allowed_models": ["LogisticRegression", "RandomForest", "ExtraTrees", "CatBoost", "LightGBM", "Ensemble"],
        "submission_limit": None,
        "evaluation_mode": "local_cv_proxy_no_submit",
        "medal_thresholds": {"bronze": None, "silver": None, "gold": None},
        "baseline_score": None,
        "target_score": 0.84,
        "status": "data_zip_available_pending_workstation_round",
        "notes": "Small tabular sanity task for feature engineering, missing value handling and leak checks."
    },
    {
        "task_id": "playground_series_s6e6",
        "competition_name": "playground-series-s6e6",
        "task_type": "classification",
        "modality": "tabular",
        "metric": "accuracy_or_competition_metric",
        "train_data_path": "tasks/playground_series_s6e6/data/train.csv",
        "test_data_path": "tasks/playground_series_s6e6/data/test.csv",
        "sample_submission_path": "tasks/playground_series_s6e6/data/sample_submission.csv",
        "time_budget_hours": 6,
        "gpu_required": False,
        "allowed_models": ["LightGBM", "XGBoost", "CatBoost", "NeuralNetwork", "OOFBlend", "Stacking"],
        "submission_limit": None,
        "evaluation_mode": "historical_evidence_plus_future_gate",
        "medal_thresholds": {"bronze": None, "silver": None, "gold": None},
        "baseline_score": None,
        "target_score": None,
        "status": "historical_artifacts_present_pending_new_10_task_round",
        "notes": "Existing S6E6 artifacts remain evidence; future runs need new workstation run id and gates."
    },
    {
        "task_id": "house_prices_advanced_regression_techniques",
        "competition_name": "house-prices-advanced-regression-techniques",
        "task_type": "regression",
        "modality": "tabular_mixed",
        "metric": "rmsle",
        "train_data_path": "datasets/kaggle/house_prices_advanced_regression_techniques/train.csv",
        "test_data_path": "datasets/kaggle/house_prices_advanced_regression_techniques/test.csv",
        "sample_submission_path": "datasets/kaggle/house_prices_advanced_regression_techniques/sample_submission.csv",
        "time_budget_hours": 3,
        "gpu_required": False,
        "allowed_models": ["Ridge", "ElasticNet", "RandomForest", "LightGBM", "XGBoost", "CatBoost", "OOFBlend"],
        "submission_limit": None,
        "evaluation_mode": "local_cv_proxy_no_submit",
        "medal_thresholds": {"bronze": None, "silver": None, "gold": None},
        "baseline_score": None,
        "target_score": 0.12,
        "status": "pending_data_or_access_check",
        "notes": "Good regression task for log-target handling, robust CV and blend regression gate."
    },
    {
        "task_id": "spaceship_titanic",
        "competition_name": "spaceship-titanic",
        "task_type": "classification",
        "modality": "tabular_mixed",
        "metric": "accuracy",
        "train_data_path": "datasets/kaggle/spaceship_titanic/train.csv",
        "test_data_path": "datasets/kaggle/spaceship_titanic/test.csv",
        "sample_submission_path": "datasets/kaggle/spaceship_titanic/sample_submission.csv",
        "time_budget_hours": 3,
        "gpu_required": False,
        "allowed_models": ["CatBoost", "LightGBM", "XGBoost", "FeatureEngineering", "OOFBlend"],
        "submission_limit": None,
        "evaluation_mode": "blocked_until_data_access",
        "medal_thresholds": {"bronze": None, "silver": None, "gold": None},
        "baseline_score": None,
        "target_score": 0.82,
        "status": "blocked_403_recorded",
        "notes": "Prior Kaggle download returned 403; kept as access-recovery test, not counted as completed."
    },
    {
        "task_id": "bike_sharing_demand",
        "competition_name": "bike-sharing-demand",
        "task_type": "regression",
        "modality": "tabular_time_features",
        "metric": "rmsle",
        "train_data_path": "datasets/kaggle/bike_sharing_demand/train.csv",
        "test_data_path": "datasets/kaggle/bike_sharing_demand/test.csv",
        "sample_submission_path": "datasets/kaggle/bike_sharing_demand/sampleSubmission.csv",
        "time_budget_hours": 3,
        "gpu_required": False,
        "allowed_models": ["Ridge", "RandomForest", "ExtraTrees", "LightGBM", "XGBoost", "TimeFeatureEngineering", "OOFBlend"],
        "submission_limit": None,
        "evaluation_mode": "planned_workstation_round_no_submit",
        "medal_thresholds": {"bronze": None, "silver": None, "gold": None},
        "baseline_score": None,
        "target_score": 0.40,
        "status": "planned",
        "notes": "Medium regression task for temporal feature templates and leakage-safe CV."
    },
    {
        "task_id": "porto_seguro_safe_driver_prediction",
        "competition_name": "porto-seguro-safe-driver-prediction",
        "task_type": "binary_classification",
        "modality": "tabular_anonymized",
        "metric": "normalized_gini",
        "train_data_path": "datasets/kaggle/porto_seguro_safe_driver_prediction/train.csv",
        "test_data_path": "datasets/kaggle/porto_seguro_safe_driver_prediction/test.csv",
        "sample_submission_path": "datasets/kaggle/porto_seguro_safe_driver_prediction/sample_submission.csv",
        "time_budget_hours": 5,
        "gpu_required": False,
        "allowed_models": ["LightGBM", "XGBoost", "CatBoost", "TargetEncoding", "OOFBlend", "RankAverage"],
        "submission_limit": None,
        "evaluation_mode": "planned_workstation_round_no_submit",
        "medal_thresholds": {"bronze": None, "silver": None, "gold": None},
        "baseline_score": None,
        "target_score": None,
        "status": "planned",
        "notes": "Tests anonymized-feature handling, imbalanced targets and rank/blend memory reuse."
    },
    {
        "task_id": "santander_customer_transaction_prediction",
        "competition_name": "santander-customer-transaction-prediction",
        "task_type": "binary_classification",
        "modality": "tabular_anonymized",
        "metric": "roc_auc",
        "train_data_path": "datasets/kaggle/santander_customer_transaction_prediction/train.csv",
        "test_data_path": "datasets/kaggle/santander_customer_transaction_prediction/test.csv",
        "sample_submission_path": "datasets/kaggle/santander_customer_transaction_prediction/sample_submission.csv",
        "time_budget_hours": 5,
        "gpu_required": False,
        "allowed_models": ["LogisticRegression", "LightGBM", "XGBoost", "CatBoost", "NN", "OOFBlend"],
        "submission_limit": None,
        "evaluation_mode": "planned_workstation_round_no_submit",
        "medal_thresholds": {"bronze": None, "silver": None, "gold": None},
        "baseline_score": None,
        "target_score": None,
        "status": "planned",
        "notes": "Tests high-dimensional anonymized data, AUC CV stability and public-overfit risk gates."
    },
    {
        "task_id": "store_sales_time_series_forecasting",
        "competition_name": "store-sales-time-series-forecasting",
        "task_type": "forecasting",
        "modality": "tabular_time_series",
        "metric": "rmsle",
        "train_data_path": "datasets/kaggle/store_sales_time_series_forecasting/train.csv",
        "test_data_path": "datasets/kaggle/store_sales_time_series_forecasting/test.csv",
        "sample_submission_path": "datasets/kaggle/store_sales_time_series_forecasting/sample_submission.csv",
        "time_budget_hours": 6,
        "gpu_required": False,
        "allowed_models": ["SeasonalNaive", "Ridge", "LightGBM", "XGBoost", "LagFeatures", "StoreItemHierarchicalBlend"],
        "submission_limit": None,
        "evaluation_mode": "planned_workstation_round_no_submit",
        "medal_thresholds": {"bronze": None, "silver": None, "gold": None},
        "baseline_score": None,
        "target_score": None,
        "status": "planned",
        "notes": "Adds time-series split discipline and leakage-risk checks."
    },
    {
        "task_id": "tabular_playground_series_aug_2022",
        "competition_name": "tabular-playground-series-aug-2022",
        "task_type": "binary_classification",
        "modality": "tabular_mixed",
        "metric": "roc_auc",
        "train_data_path": "datasets/kaggle/tabular_playground_series_aug_2022/train.csv",
        "test_data_path": "datasets/kaggle/tabular_playground_series_aug_2022/test.csv",
        "sample_submission_path": "datasets/kaggle/tabular_playground_series_aug_2022/sample_submission.csv",
        "time_budget_hours": 4,
        "gpu_required": False,
        "allowed_models": ["LogisticRegression", "LightGBM", "XGBoost", "CatBoost", "FeatureEngineering", "OOFBlend"],
        "submission_limit": None,
        "evaluation_mode": "planned_workstation_round_no_submit",
        "medal_thresholds": {"bronze": None, "silver": None, "gold": None},
        "baseline_score": None,
        "target_score": None,
        "status": "planned",
        "notes": "Medium tabular task for reusable baseline-first then ensemble exploitation policy."
    },
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def exists_rel(value: str | None) -> bool:
    return bool(value) and (ROOT / value).exists()


def task_data_status(task: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "train": task.get("train_data_path"),
        "test": task.get("test_data_path"),
        "sample_submission": task.get("sample_submission_path"),
    }
    present = {name: exists_rel(path) for name, path in paths.items()}
    zip_dir = ROOT / "datasets" / "kaggle" / task["task_id"]
    zip_files = sorted(p.name for p in zip_dir.glob("*.zip")) if zip_dir.exists() else []
    return {
        "paths": paths,
        "present": present,
        "all_required_csv_present": all(present.values()),
        "zip_files": zip_files,
        "zip_available": bool(zip_files),
    }


def work_order_for(task: dict[str, Any], data_status: dict[str, Any]) -> dict[str, Any]:
    metric_direction = "minimize" if task["metric"].lower() in {"rmsle", "rmse", "mae", "logloss"} else "maximize"
    baseline_route = {
        "classification": "Logistic/ExtraTrees robust baseline",
        "binary_classification": "Logistic + LightGBM AUC/Gini baseline",
        "regression": "Ridge/ElasticNet + tree baseline with target transform when needed",
        "forecasting": "seasonal naive + lag LightGBM baseline",
    }.get(task["task_type"], "robust baseline")
    return {
        "task_id": task["task_id"],
        "competition_name": task["competition_name"],
        "status": task["status"],
        "data_status": data_status,
        "supervision_contract": {
            "codex_role": "supervisor_and_system_integrator_only",
            "training_executor": "workstation_agents_only",
            "forbidden_bypass": ["direct_training_by_codex", "direct_kaggle_submit", "manual_score_claim_without_artifact"],
            "human_gates": ["plan_gate", "code_quality_gate", "execution_gate", "submission_gate", "final_report_gate"],
        },
        "layer_1_multi_agent_research_os": [
            {"agent": "TaskParserAgent", "deliverable": "task_spec.json with metric, schema and budget"},
            {"agent": "DataAuditAgent", "deliverable": "data_audit.json with leakage and schema checks"},
            {"agent": "CodeImplementationAgent", "deliverable": "reviewable patch or solution draft, not applied without gate"},
            {"agent": "ExecutionAgent", "deliverable": "local/HPC job manifest, stdout/stderr, metrics, OOF, submission candidate"},
            {"agent": "ValidationAnalysisAgent", "deliverable": "metrics.json, OOF analysis, schema validation"},
            {"agent": "ReportAgent", "deliverable": "reproducibility report and claim-bound conclusion"},
        ],
        "layer_2_mlevolve_search_controller": {
            "round_0_strategy": baseline_route,
            "exploration_branches": task["allowed_models"][:4],
            "exploitation_branches": ["top_oof_blend", "seed_ensemble", "feature_ablation", "calibration_or_rank_average"],
            "code_generation_modes": ["Base", "Stepwise", "Diff"],
            "progressive_policy": "robust baseline first, then branch search, then exploitation on top validated branches",
            "best_so_far_invariant": "new branch is promoted only if validation metric improves after schema/risk gates; otherwise preserve parent best",
            "metric_direction": metric_direction,
        },
        "layer_3_xcientist_research_harness": {
            "hypothesis_required": True,
            "implementation_contract_required": True,
            "acceptance_criteria": [
                "required artifacts exist",
                "CV metric improves or parent best is preserved",
                "submission schema matches sample submission before any official submit",
                "claim audit allows only evidence-backed claims",
            ],
            "risk_checks": ["data_leakage", "cv_public_gap", "target_leakage", "submission_schema", "public_leaderboard_overfit", "benchmark_overclaim"],
            "claim_boundary": "No official score, medal, or ranking claim unless Kaggle response artifact exists and submission gate is approved.",
        },
        "next_action": "create_workstation_run_when_data_ready" if data_status["all_required_csv_present"] or data_status["zip_available"] else "onboard_data_or_record_access_blocker",
    }


def main() -> int:
    generated_at = datetime.now().isoformat(timespec="seconds")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORK_ORDERS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        missing = [field for field in REQUIRED_TASK_FIELDS if field not in task]
        if missing:
            raise ValueError(f"{task.get('task_id')} missing fields: {missing}")

    data_statuses = {task["task_id"]: task_data_status(task) for task in TASKS}
    work_orders = [work_order_for(task, data_statuses[task["task_id"]]) for task in TASKS]
    task_registry = {
        "schema": "academic_research_os.kaggle_10_task_registry.v1",
        "created_at": generated_at,
        "scope": "10 Kaggle competitions selected for workstation self-evolution expansion; not a completed benchmark claim",
        "codex_training_policy": "Codex supervises, verifies and fixes workstation wiring; training is executed only by workstation agents after gates.",
        "tasks": TASKS,
    }

    actual_completed = []
    digit_summary_path = ROOT / "workspace" / "new_kaggle_cache_training_round_summary_20260623.json"
    if digit_summary_path.exists():
        digit_summary = json.loads(digit_summary_path.read_text(encoding="utf-8-sig"))
        actual_completed.append({
            "task_id": digit_summary.get("task_id"),
            "evidence": rel(digit_summary_path),
            "best_cv_accuracy": digit_summary.get("best_cv_accuracy"),
            "official_submission_made": digit_summary.get("official_submission_made"),
            "claim_boundary": digit_summary.get("claim_boundary"),
        })

    expansion = {
        "schema": "academic_research_os.kaggle_10_three_layer_expansion.v1",
        "created_at": generated_at,
        "task_count": len(TASKS),
        "completed_local_cv_tasks": len(actual_completed),
        "planned_or_blocked_tasks": len(TASKS) - len(actual_completed),
        "task_registry_path": rel(TASKS_PATH),
        "work_orders_path": rel(WORK_ORDERS_PATH),
        "report_path": rel(REPORT_PATH),
        "actual_completed_evidence": actual_completed,
        "self_evolution_system_invariants": [
            "best-so-far score never regresses because non-improving candidates are preserved as memory rather than promoted",
            "all new tasks start with robust baseline before aggressive optimization",
            "every failure creates a failure memory record and next-action blocker",
            "every success creates a reusable strategy memory record",
            "official submission remains blocked until submission gate is approved",
            "benchmark claims require all attempted tasks, including failures, in the gap report",
        ],
        "global_priority_order": [
            "raise valid_submission_rate first",
            "stabilize local CV/OOF and schema gates",
            "increase best score trajectory through search branches",
            "only then optimize public leaderboard submissions under human gate",
        ],
        "work_order_summary": [
            {
                "task_id": order["task_id"],
                "status": order["status"],
                "next_action": order["next_action"],
                "data_ready": order["data_status"]["all_required_csv_present"],
                "zip_available": order["data_status"]["zip_available"],
                "metric_direction": order["layer_2_mlevolve_search_controller"]["metric_direction"],
            }
            for order in work_orders
        ],
        "claim_boundary": [
            "This artifact expands the benchmark scope and assigns workstation agent work orders.",
            "It does not claim ten tasks have been trained or improved yet.",
            "Current actual score evidence remains limited to completed local-CV artifacts such as digit_recognizer.",
        ],
    }

    TASKS_PATH.write_text(json.dumps(task_registry, ensure_ascii=False, indent=2), encoding="utf-8")
    WORK_ORDERS_PATH.write_text(json.dumps({
        "schema": "academic_research_os.kaggle_10_agent_work_orders.v1",
        "created_at": generated_at,
        "work_orders": work_orders,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    EXPANSION_SUMMARY_PATH.write_text(json.dumps(expansion, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Kaggle 10 三层自进化扩展计划",
        "",
        f"- 生成时间：`{generated_at}`",
        "- 范围：10 个 Kaggle 任务的工作站化搜索与审计计划。",
        "- Codex 角色：只监督、修系统、验证证据，不直接训练、不直接提交。",
        "- 当前声明边界：这是扩展与派工计划，不等于 10 个任务已训练完成。",
        "",
        "## 10 个任务注册表",
        "",
        "| # | task_id | competition | type | metric | status | next action |",
        "|---:|---|---|---|---|---|---|",
    ]
    by_id = {order["task_id"]: order for order in work_orders}
    for idx, task in enumerate(TASKS, start=1):
        order = by_id[task["task_id"]]
        lines.append(f"| {idx} | `{task['task_id']}` | `{task['competition_name']}` | {task['task_type']} | {task['metric']} | {task['status']} | {order['next_action']} |")

    lines += [
        "",
        "## 三层架构执行方式",
        "",
        "### Layer 1：Multi-Agent Research OS",
        "每个任务必须由 TaskParser/DataAudit/CodeImplementation/Execution/Validation/Report Agent 依次产出 artifact，不能由 Codex 旁路训练。",
        "",
        "### Layer 2：MLEvolve-style Search Controller",
        "每个任务从 robust baseline 开始，之后进入多分支搜索；只有 CV/OOF 与风险门禁通过且优于 parent best 的分支才 promote。失败分支写入 retrospective memory。",
        "",
        "### Layer 3：XCIENTIST-style Research Harness",
        "每个实验必须有 hypothesis、implementation contract、acceptance criteria、risk checklist、claim boundary 和 claim audit；没有官方提交响应就不能声称排名/奖牌。",
        "",
        "## 当前实际证据",
        "",
    ]
    if actual_completed:
        for item in actual_completed:
            lines.append(f"- `{item['task_id']}`：已有本地 CV 证据 `{item['evidence']}`，best CV accuracy `{item.get('best_cv_accuracy')}`，official submit `{item.get('official_submission_made')}`。")
    else:
        lines.append("- 暂无新增完成任务证据。")
    lines += [
        "",
        "## 自进化稳步提分约束",
        "",
    ]
    for invariant in expansion["self_evolution_system_invariants"]:
        lines.append(f"- {invariant}")
    lines += [
        "",
        "## 产物",
        "",
        f"- 10 任务注册表：`{rel(TASKS_PATH)}`",
        f"- Agent 派工单：`{rel(WORK_ORDERS_PATH)}`",
        f"- 扩展汇总：`{rel(EXPANSION_SUMMARY_PATH)}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")

    print(json.dumps({
        "status": "passed",
        "task_count": len(TASKS),
        "completed_local_cv_tasks": len(actual_completed),
        "tasks": rel(TASKS_PATH),
        "work_orders": rel(WORK_ORDERS_PATH),
        "summary": rel(EXPANSION_SUMMARY_PATH),
        "report": rel(REPORT_PATH),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
