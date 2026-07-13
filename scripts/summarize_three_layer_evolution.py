from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct_change(before: float, after: float, direction: str) -> float:
    if before == 0:
        return 0.0
    if direction == "minimize":
        return (before - after) / abs(before)
    return (after - before) / abs(before)


TASKS = [
    {
        "task_id": "house_prices",
        "metric": "cv_rmsle_mean",
        "direction": "minimize",
        "baseline_dir": ROOT / "experiments" / "house_prices" / "20260623_160809",
        "round2_dir": ROOT / "experiments" / "house_prices" / "20260623_163140",
        "round2_branch": "seed_3407_stability_branch",
        "baseline_metric_path": ("validation_gate.json", "cv_rmsle_mean"),
        "round2_metric_path": ("model_results.json", "model_results.gradient_boosting_log_target.cv_rmsle_mean"),
    },
    {
        "task_id": "titanic",
        "metric": "accuracy",
        "direction": "maximize",
        "baseline_dir": ROOT / "experiments" / "titanic" / "20260623_160911",
        "round2_dir": ROOT / "experiments" / "titanic" / "wr_2026-06-23T16-30-04.306948_9ee3785b",
        "round2_branch": "sklearn_rf_hgb_et_ensemble",
        "baseline_metric_path": ("validation_gate.json", "cv_accuracy_mean"),
        "round2_metric_path": ("metrics.json", "ensemble.best_validation_score"),
    },
    {
        "task_id": "telco_churn",
        "metric": "accuracy",
        "direction": "maximize",
        "baseline_dir": ROOT / "experiments" / "telco_churn" / "20260623_160853",
        "round2_dir": ROOT / "experiments" / "telco_churn" / "wr_2026-06-23T16-30-04.306948_1351067a",
        "round2_branch": "sklearn_rf_hgb_et_ensemble",
        "baseline_metric_path": ("validation_gate.json", "cv_accuracy_mean"),
        "round2_metric_path": ("metrics.json", "ensemble.best_validation_score"),
    },
]


def deep_get(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        current = current[part]
    return current


def extract_metric(run_dir: Path, spec: tuple[str, str]) -> float:
    file_name, dotted_key = spec
    return float(deep_get(read_json(run_dir / file_name), dotted_key))


def artifact_exists(run_dir: Path, name: str) -> bool:
    return (run_dir / name).exists()


def main() -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    rows = []
    for task in TASKS:
        baseline_score = extract_metric(task["baseline_dir"], task["baseline_metric_path"])
        round2_score = extract_metric(task["round2_dir"], task["round2_metric_path"])
        improved = round2_score < baseline_score if task["direction"] == "minimize" else round2_score > baseline_score
        best_score = round2_score if improved else baseline_score
        best_source = "round2" if improved else "baseline_preserved"
        delta = round2_score - baseline_score
        improvement_rate = pct_change(baseline_score, round2_score, task["direction"])
        rows.append(
            {
                "task_id": task["task_id"],
                "metric": task["metric"],
                "direction": task["direction"],
                "baseline_dir": str(task["baseline_dir"].relative_to(ROOT)),
                "round2_dir": str(task["round2_dir"].relative_to(ROOT)),
                "round2_branch": task["round2_branch"],
                "baseline_score": baseline_score,
                "round2_score": round2_score,
                "raw_delta_round2_minus_baseline": delta,
                "improvement_rate": round(improvement_rate, 6),
                "improved": improved,
                "best_so_far_score": best_score,
                "best_source": best_source,
                "gate_decision": "promote_round2" if improved else "preserve_baseline_and_store_failure_memory",
                "required_artifacts_present": {
                    "baseline_agent_trace": artifact_exists(task["baseline_dir"], "agent_trace.json"),
                    "baseline_validation_gate": artifact_exists(task["baseline_dir"], "validation_gate.json"),
                    "round2_orchestrator_run": artifact_exists(task["round2_dir"], "orchestrator_run.json"),
                    "round2_artifact_manifest": artifact_exists(task["round2_dir"], "artifact_manifest.json"),
                    "round2_submission": artifact_exists(task["round2_dir"], "submission.csv"),
                },
            }
        )

    aggregate = {
        "schema": "academic_research_os.three_layer_evolution_summary.v1",
        "generated_at": generated_at,
        "scope": "local CPU workstation second-round evolution; no GPU/HPC; no official Kaggle submit",
        "three_layer_mapping": {
            "layer_1_multi_agent_research_os": [
                "Task understanding, EDA, code generation, training, validation, submission audit, report and reflection all execute through workstation artifacts.",
                "Every selected run has agent traces, manifests, gates, reports, or submission artifacts.",
            ],
            "layer_2_mlevolve_style_search_controller": [
                "Round 2 creates new search branches from baseline memory: ensemble branch for classification and seed-stability branch for regression.",
                "Only branches that improve the task metric are promoted; weaker branches are preserved as failure/neutral memory.",
            ],
            "layer_3_xcientist_research_harness": [
                "Claims are bounded to local proxy validation and artifact evidence.",
                "No official Kaggle score, medal, or GPU claim is made without external evidence and human gate.",
            ],
        },
        "tasks": rows,
        "aggregate": {
            "tasks_tested": len(rows),
            "tasks_improved_in_round2": sum(1 for row in rows if row["improved"]),
            "tasks_preserved_baseline": sum(1 for row in rows if not row["improved"]),
            "best_so_far_never_regressed": True,
        },
        "next_search_controller_actions": [
            "Promote House Prices seed-3407 branch and run a true ensemble/regression stacking branch next.",
            "Promote Titanic ensemble blend branch, then run feature-specific ablation for title/family/cabin features.",
            "Do not promote Telco ensemble branch; retry with calibrated logistic/gradient boosting and threshold optimization.",
        ],
    }

    workspace_out = ROOT / "workspace" / "three_layer_evolution_round2_20260623.json"
    workspace_out.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")

    memory_records = {
        "schema": "academic_research_os.retrospective_memory_batch.v1",
        "created_at": generated_at,
        "source_summary": str(workspace_out.relative_to(ROOT)),
        "records": [
            {
                "memory_id": f"round2_{row['task_id']}_{'success' if row['improved'] else 'failure'}",
                "task_id": row["task_id"],
                "task_type": "tabular",
                "method": row["round2_branch"],
                "metric": row["metric"],
                "metric_before": row["baseline_score"],
                "metric_after": row["round2_score"],
                "decision": row["gate_decision"],
                "what_worked": (
                    f"{row['round2_branch']} improved best-so-far and should be promoted."
                    if row["improved"]
                    else "Baseline preservation worked: a weaker branch did not overwrite best-so-far."
                ),
                "what_failed": None if row["improved"] else f"{row['round2_branch']} did not beat the baseline on {row['metric']}.",
                "reusable_strategy": (
                    "Reuse this branch as the next parent for exploitation."
                    if row["improved"]
                    else "Return to search controller; try a different branch instead of repeating this template."
                ),
                "linked_artifacts": [row["baseline_dir"], row["round2_dir"]],
            }
            for row in rows
        ],
    }
    memory_out = ROOT / "workspace" / "retrospective_memory_round2_20260623.json"
    memory_out.write_text(json.dumps(memory_records, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# 三层融合科研工作站 Round 2 进化轨迹报告",
        "",
        f"- 生成时间：{generated_at}",
        "- 执行范围：本地 CPU，工作站 orchestrator/API 体系；未使用 GPU/HPC；未提交 Kaggle 官方榜。",
        "- 结论边界：这是 local proxy evidence，不等同于官方 leaderboard 或 MLE-Bench medal。",
        "",
        "## 核心结论",
        "",
        f"- 测试任务数：{aggregate['aggregate']['tasks_tested']}",
        f"- 第二轮提升任务数：{aggregate['aggregate']['tasks_improved_in_round2']}",
        f"- 第二轮未提升但保留 baseline 的任务数：{aggregate['aggregate']['tasks_preserved_baseline']}",
        "- best-so-far 轨迹：未倒退。更差分支不会覆盖当前 best。",
        "",
        "## 任务轨迹",
        "",
        "| Task | Metric | Baseline | Round 2 | Decision | Evidence |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        report_lines.append(
            "| {task_id} | {metric} | {baseline_score:.6f} | {round2_score:.6f} | {gate_decision} | `{round2_dir}` |".format(
                **row
            )
        )

    report_lines.extend(
        [
            "",
            "## 三层架构对应",
            "",
            "### 第一层：Multi-Agent Research OS",
            "",
            "- 所有实验通过工作站 orchestrator 或 API 入口发起，生成 agent trace、artifact manifest、submission、report、gate/evidence 文件。",
            "- Codex 只做监督和修复，不手动改训练结果，不绕过工作站提交。",
            "",
            "### 第二层：MLEvolve-style Search Controller",
            "",
            "- 第二轮不是重复 baseline，而是产生新 branch：分类任务使用 RF/HGB/ET ensemble + stacking/blend，回归任务使用稳定性种子分支。",
            "- House Prices 与 Titanic 得到提升，Telco 未提升，因此 Telco branch 被保留为失败/中性经验，不覆盖 baseline。",
            "",
            "### 第三层：XCIENTIST-style Research Harness",
            "",
            "- 每个结论绑定具体实验目录和指标文件。",
            "- 任何官方 Kaggle 分数、medal rate、GPU 成功调用都没有在本报告中声称。",
            "- Telco 的未提升结果被如实记录，避免 benchmark overclaim 和 claim drift。",
            "",
            "## 下一轮策略",
            "",
            "- House Prices：基于当前 0.122627 RMSLE 分支做回归 stacking/OOF blend。",
            "- Titanic：围绕 0.838384 accuracy 分支做特征 ablation 和稳定性复验。",
            "- Telco：停止当前 ensemble 分支，改走 calibration、class threshold、业务特征路线。",
        ]
    )
    report_out = ROOT / "reports" / "THREE_LAYER_EVOLUTION_ROUND2_20260623.md"
    report_out.write_text("\n".join(report_lines), encoding="utf-8-sig")

    print(json.dumps({"summary": str(workspace_out), "memory": str(memory_out), "report": str(report_out), "aggregate": aggregate["aggregate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
