from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "workspace" / "paper_evidence_bundle_20260623.json"
PROTOCOL_PATH = ROOT / "workspace" / "steady_improvement_protocol_20260623.json"
ROUND4_PLAN_PATH = ROOT / "workspace" / "round4_search_plan_20260623.json"
REPORT_PATH = ROOT / "reports" / "STEADY_IMPROVEMENT_PROTOCOL_20260623.md"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def is_better(direction: str, candidate: float, reference: float, eps: float = 1e-12) -> bool:
    if direction == "minimize":
        return candidate < reference - eps
    return candidate > reference + eps


def is_not_worse(direction: str, candidate: float, reference: float, eps: float = 1e-12) -> bool:
    if direction == "minimize":
        return candidate <= reference + eps
    return candidate >= reference - eps


def signed_delta(direction: str, candidate: float, reference: float) -> float:
    raw = candidate - reference
    return -raw if direction == "minimize" else raw


def next_branch_plan(row: dict[str, Any]) -> dict[str, Any]:
    task_id = row["task_id"]
    direction = row["direction"]
    parent_score = row["final_best_so_far"]
    metric = row["metric"]

    plans = {
        "house_prices": {
            "search_stage": "exploitation",
            "branch_type": "repeated_seed_oof_stack_and_residual_blend",
            "code_generation_mode": "Stepwise",
            "hypothesis": "在已提升的 OOF stacking 父节点上增加 repeated seed 与 residual correction，可以降低 CV RMSLE，同时保持 submission schema 稳定。",
            "implementation_requirement": [
                "继承 Round3 promoted 父节点作为唯一 best parent。",
                "新增 repeated seed OOF 预测与 residual blend，不允许覆盖 Round3 artifact。",
                "输出 per-fold RMSLE、OOF、submission、artifact_manifest、claim_audit。",
            ],
        },
        "titanic": {
            "search_stage": "exploration_to_exploitation",
            "branch_type": "new_feature_route_plus_model_diversity",
            "code_generation_mode": "Diff",
            "hypothesis": "Round3 ablation 已证明未提升，应切换到 Title/Cabin/Ticket group 与模型多样性路线，而不是重复同一 ablation。",
            "implementation_requirement": [
                "保留 Round2 parent best，不允许把持平分支宣传为提升。",
                "新增特征族并对 Logistic/ExtraTrees/GBDT 做轻量模型选择。",
                "若 CV 未超过 parent best，只沉淀 memory，不 promote。",
            ],
        },
        "telco_churn": {
            "search_stage": "exploitation_with_stability_check",
            "branch_type": "calibration_threshold_stability_and_class_weight",
            "code_generation_mode": "Stepwise",
            "hypothesis": "Round3 calibration/threshold recovery 有效，下一轮应验证跨折稳定性并加入 class-weight 搜索，以提高 accuracy 且降低阈值过拟合风险。",
            "implementation_requirement": [
                "继承 Round3 calibration 父节点。",
                "记录每折最佳阈值分布和 OOF confusion matrix。",
                "若阈值方差过大或 CV 提升不足，则只作为 weak evidence。",
            ],
        },
    }
    base = plans.get(task_id, {
        "search_stage": "exploration",
        "branch_type": "robust_baseline_then_light_ensemble",
        "code_generation_mode": "Base",
        "hypothesis": "先建立稳定 baseline，再进入 MLEvolve-style 分支搜索。",
        "implementation_requirement": ["输出 metrics、OOF、submission、artifact_manifest、claim_audit。"],
    })
    return {
        "task_id": task_id,
        "parent_branch_id": row["branch_id"],
        "parent_artifact": row["output_dir"],
        "parent_best_score": parent_score,
        "metric": metric,
        "direction": direction,
        "metric_floor_or_ceiling": {
            "rule": "new branch must be strictly better than parent best to promote; otherwise preserve parent best",
            "parent_best": parent_score,
        },
        "acceptance_criteria": [
            f"Primary metric {metric} must improve over {parent_score:.6f} under {direction} direction.",
            "Submission schema must match sample submission exactly.",
            "Validation contract and claim audit must be generated before report conclusion.",
            "Generated code must be attached as artifact and pass the code quality gate.",
        ],
        "rollback_condition": [
            "CV is worse than parent best or numerically tied without new evidence.",
            "OOF/submission artifact missing or schema check fails.",
            "Claim audit returns revise/reject for improvement claim.",
            "Runtime/dependency failure repeats after configured retries.",
        ],
        **base,
    }


def main() -> None:
    bundle = read_json(BUNDLE_PATH)
    generated_at = datetime.now().isoformat(timespec="seconds")
    rows = bundle.get("trajectory", [])

    trajectory_checks = []
    for row in rows:
        direction = row["direction"]
        r1 = float(row["round1_baseline"])
        r2 = float(row["round2_best_so_far"])
        r3 = float(row["round3_score"])
        final_best = float(row["final_best_so_far"])
        trajectory_checks.append({
            "task_id": row["task_id"],
            "metric": row["metric"],
            "direction": direction,
            "round1_baseline": r1,
            "round2_best_so_far": r2,
            "round3_score": r3,
            "final_best_so_far": final_best,
            "round2_not_worse_than_round1": is_not_worse(direction, r2, r1),
            "final_not_worse_than_round2_best": is_not_worse(direction, final_best, r2),
            "final_improved_over_round1": is_better(direction, final_best, r1),
            "round3_promoted": row["round3_decision"] == "promote_round3",
            "best_so_far_delta_vs_round1_positive_means_better": signed_delta(direction, final_best, r1),
            "claim_allowed": "best-so-far improved" if is_better(direction, final_best, r1) else "best-so-far preserved only",
            "artifact_binding": {
                "experiment_dir": row["output_dir"],
                "validation_contract": row["validation_contract"],
                "claim_audit": row["claim_audit"],
            },
        })

    all_monotonic = all(item["round2_not_worse_than_round1"] and item["final_not_worse_than_round2_best"] for item in trajectory_checks)
    improved_tasks = sum(1 for item in trajectory_checks if item["final_improved_over_round1"])
    promoted_round3 = sum(1 for item in trajectory_checks if item["round3_promoted"])

    round4_plan = {
        "schema": "academic_research_os.round4_search_plan.v1",
        "generated_at": generated_at,
        "purpose": "下一轮只通过工作站 Agent/Search Controller 发起，用于继续验证三层架构的自进化提分能力。",
        "execution_boundary": [
            "Codex 只监督、修系统和审计证据，不直接训练。",
            "GPU/HPC 恢复前可使用本地轻量资源；GPU/HPC 恢复后必须走 job manifest 与 gate。",
            "任何官方 Kaggle submission 必须经过人工 submission gate。",
        ],
        "branches": [next_branch_plan(row) for row in rows],
    }
    write_json(ROUND4_PLAN_PATH, round4_plan)

    protocol = {
        "schema": "academic_research_os.steady_improvement_protocol.v1",
        "generated_at": generated_at,
        "paper_core_claim": "三层架构保证的是 best-so-far 单调保护与失败可学习，而不是每个单独实验都必然提分。",
        "evidence_scope": "local_proxy_three_task_round1_to_round3",
        "steady_improvement_definition": {
            "strictly_guaranteed": [
                "低于 parent best 的实验不得 promote。",
                "每次实验必须产生 artifact、metrics、validation contract、claim audit。",
                "失败、持平、提升三类结果都写入 retrospective memory。",
                "最终报告只能声明 claim audit 允许的结论。",
            ],
            "not_guaranteed": [
                "不保证每一个分支实验都提分。",
                "不把 local proxy 结果宣称为官方 Kaggle leaderboard 结果。",
                "不把三任务 proxy 结果宣称为 MLE-Bench 75 任务 medal rate。",
            ],
        },
        "three_layer_runtime_contract": {
            "layer_1_multi_agent_research_os": [
                "所有训练由工作站 run/action 发起并记录 agent trace。",
                "每个任务绑定 task_id、run_id、artifact_manifest、metrics、submission。",
                "运行失败触发 retry/rollback artifact，而不是静默覆盖。",
            ],
            "layer_2_mlevolve_style_search_controller": [
                "根据上一轮 memory 选择下一轮 branch。",
                "根据阶段在 Base / Stepwise / Diff 代码生成模式间切换。",
                "探索期优先 valid submission rate，利用期优化 best score trajectory。",
                "成功策略抽象为 reusable strategy，失败路径抽象为 failure pattern。",
            ],
            "layer_3_xcientist_style_research_harness": [
                "实验前生成 hypothesis 与 validation contract。",
                "实验后执行 claim audit，检查 unsupported improvement、benchmark overclaim、CV-public gap 风险。",
                "结论必须绑定 exp_id、metric、artifact、ablation/risk evidence。",
            ],
        },
        "monotonicity_certificate": {
            "all_tasks_best_so_far_never_regressed": all_monotonic,
            "tasks_total": len(trajectory_checks),
            "tasks_final_improved_over_round1": improved_tasks,
            "round3_promoted_tasks": promoted_round3,
            "round3_preserved_parent_tasks": len(trajectory_checks) - promoted_round3,
        },
        "trajectory_checks": trajectory_checks,
        "next_round_plan": str(ROUND4_PLAN_PATH.relative_to(ROOT)),
        "claim_boundary": bundle.get("claim_boundary", {}),
    }
    write_json(PROTOCOL_PATH, protocol)

    lines = [
        "# 稳步提升协议：三层 AI 科研工作站的论文核心证明",
        "",
        f"- 生成时间：{generated_at}",
        "- 证据范围：本地三任务 Round1→Round2→Round3 代理验证。",
        "- 核心定义：系统保证 best-so-far 单调保护、失败可学习、成功可复用；不承诺每个单独实验必然提分。",
        "",
        "## 1. 为什么第二轮/后续轮次体现核心优势",
        "",
        "第一轮 baseline 只提供起点；从第二轮开始，Search Controller 才能读取上一轮 metrics、OOF、错误日志、artifact 与 retrospective memory，决定下一轮是 exploration、exploitation 还是 recovery。分数不理想时，系统不会把失败视为终点，而是把失败转化为下一轮搜索约束。",
        "",
        "## 2. 三层架构与稳步提升机制",
        "",
        "| 层级 | 论文角色 | 对稳步提升的贡献 |",
        "|---|---|---|",
        "| Multi-Agent Research OS | 执行与留痕 | 确保每轮实验都有 task/run/artifact/metrics/report，不让训练变成旁路脚本 |",
        "| MLEvolve-style Search Controller | 自进化搜索 | 根据 memory 选择 branch，成功 promote，失败 rollback，保持 best-so-far 不倒退 |",
        "| XCIENTIST-style Research Harness | 验证与审计 | 用 validation contract 和 claim audit 防止把持平/失败写成提升 |",
        "",
        "## 3. 当前轨迹证明",
        "",
        "| Task | Metric | Direction | Round1 | Round2 best | Round3 score | Final best | Round3 decision | Final vs Round1 |",
        "|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for row, check in zip(rows, trajectory_checks):
        lines.append(
            f"| {row['task_id']} | {row['metric']} | {row['direction']} | {row['round1_baseline']:.6f} | {row['round2_best_so_far']:.6f} | {row['round3_score']:.6f} | {row['final_best_so_far']:.6f} | {row['round3_decision']} | {check['best_so_far_delta_vs_round1_positive_means_better']:.6f} |"
        )
    lines.extend([
        "",
        "## 4. Monotonicity Certificate",
        "",
        f"- all_tasks_best_so_far_never_regressed: `{str(all_monotonic).lower()}`",
        f"- tasks_total: `{len(trajectory_checks)}`",
        f"- tasks_final_improved_over_round1: `{improved_tasks}`",
        f"- round3_promoted_tasks: `{promoted_round3}`",
        f"- round3_preserved_parent_tasks: `{len(trajectory_checks) - promoted_round3}`",
        "",
        "解释：Titanic Round3 没有超过 Round2 parent，所以系统保留 Round2 best，并通过 claim audit 阻止把 Round3 写成提升；这正是 XCIENTIST-style Harness 的价值。",
        "",
        "## 5. Round4 搜索计划入口",
        "",
        f"- JSON: `{ROUND4_PLAN_PATH.relative_to(ROOT)}`",
        "- Round4 必须从工作站 Agent/Search Controller 发起。",
        "- 每个分支必须先生成 validation contract，再执行代码生成/训练，再执行 claim audit。",
        "- 新分支只有严格超过 parent best 才能 promote，否则 preserve parent best 并沉淀 memory。",
        "",
        "## 6. 论文中建议采用的严谨表述",
        "",
        "> 本系统不假设每个候选实验都能提升分数，而是通过三层架构实现 best-so-far protection、failure-to-memory conversion 和 claim-bounded reporting。由此，失败实验不会破坏当前最优解，反而为下一轮 MLE search 提供可复用约束。",
        "",
        "## 7. 证据文件",
        "",
        f"- Steady improvement protocol JSON: `{PROTOCOL_PATH.relative_to(ROOT)}`",
        f"- Round4 search plan JSON: `{ROUND4_PLAN_PATH.relative_to(ROOT)}`",
        "- Paper evidence bundle: `workspace/paper_evidence_bundle_20260623.json`",
        "- Round3 retrospective memory: `workspace/retrospective_memory_round3_20260623.json`",
        "- Figure manifest: `reports/figures/three_layer_evidence_20260623/figure_manifest.json`",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")

    print(json.dumps({
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "round4_plan": str(ROUND4_PLAN_PATH.relative_to(ROOT)),
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "monotonic": all_monotonic,
        "tasks_final_improved_over_round1": improved_tasks,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
