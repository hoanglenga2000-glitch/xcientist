from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
INVENTORY = ROOT / "workspace" / "kaggle_experiment_inventory_20260624.json"
KAGGLE4_SUMMARY = ROOT / "workspace" / "kaggle4_self_evolution_rounds_20260624.json"
OUT_JSON = ROOT / "workspace" / "top30_next_evolution_orders_20260625.json"
OUT_MD = ROOT / "reports" / "TOP30_NEXT_EVOLUTION_ORDERS_20260625.md"

TOP30_TARGET = 0.30
OFFICIAL_SUBMIT_BUDGET = 2


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def task_by_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("task_id")): item for item in items if item.get("task_id")}


def official_for_task(inventory: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    for item in inventory.get("official_submission_records", []):
        if item.get("task_id") == task_id:
            return item
    return None


def make_spaceship_order(task: dict[str, Any], official: dict[str, Any] | None) -> dict[str, Any]:
    percentile = official.get("rank_percentile") if official else None
    current_gap = float(percentile) - TOP30_TARGET if isinstance(percentile, (int, float)) else None
    return {
        "task_id": "spaceship_titanic",
        "priority": "P0",
        "current_official_status": "top30_failed" if official and not official.get("top30_reached") else "proxy_only",
        "official_rank": official.get("rank") if official else None,
        "leaderboard_team_count": official.get("leaderboard_team_count") if official else None,
        "rank_percentile": percentile,
        "top30_target_percentile": TOP30_TARGET,
        "top30_gap": current_gap,
        "current_best_run": task.get("best_run_id") or task.get("best_run"),
        "current_best_score": task.get("best_score"),
        "metric": task.get("latest_metric") or task.get("metric") or "accuracy",
        "selected_branches": [
            {
                "branch_id": "spaceship_feature_interaction_exploitation",
                "branch_type": "feature_engineering",
                "code_generation_mode": "Stepwise",
                "hypothesis": "Group, cabin, home-planet and spending interaction features can improve local CV without changing the protected best-so-far.",
                "cross_branch_references": [
                    "reuse successful feature-common pattern from kaggle4 feature engineering branches",
                    "compare with held missing-indicator ablations to avoid repeated degradation",
                ],
                "expected_delta": "accuracy +0.001~0.004 local CV before official submit candidate",
                "rollback_condition": "hold if CV <= protected best, submission audit fails, or prediction distribution drifts",
            },
            {
                "branch_id": "spaceship_model_family_diversity",
                "branch_type": "model_family",
                "code_generation_mode": "Diff",
                "hypothesis": "CatBoost-like categorical treatment or calibrated HGB/ET blend may reduce CV-public mismatch versus pure one-hot ensembles.",
                "cross_branch_references": [
                    "use Porto Seguro normalized-gini branch memory for imbalanced/ordinal categorical stability",
                    "use Titanic family/group feature memory for passenger-group signals",
                ],
                "expected_delta": "accuracy +0.001~0.003 with lower CV-public gap risk",
                "rollback_condition": "hold if OOF stability worsens or claim audit marks public-score risk high",
            },
            {
                "branch_id": "spaceship_oof_blend_rank_gate_candidate",
                "branch_type": "ensemble_blend",
                "code_generation_mode": "Diff",
                "hypothesis": "Blend only top local candidates with different error profiles and submit at most one audited official candidate.",
                "cross_branch_references": [
                    "aggregate promoted spaceship nodes only",
                    "do not include held low-score candidates except as negative memory",
                ],
                "expected_delta": "official rank percentile should move from 0.368 toward <=0.30 if CV-public alignment holds",
                "rollback_condition": "no official submit if rank_promotion_gate prerequisites are missing",
            },
        ],
        "official_submit_budget": OFFICIAL_SUBMIT_BUDGET,
        "submit_policy": "Generate submission candidate only after CV/OOF, submission_audit, validation_contract, claim_audit and human approval gate pass.",
    }


def make_generic_order(task: dict[str, Any], priority: str) -> dict[str, Any]:
    task_id = task.get("task_id")
    metric = task.get("metric") or "task_metric"
    return {
        "task_id": task_id,
        "priority": priority,
        "current_official_status": "proxy_only",
        "official_rank": None,
        "leaderboard_team_count": None,
        "rank_percentile": None,
        "top30_target_percentile": TOP30_TARGET,
        "current_best_run": task.get("best_run"),
        "current_best_score": task.get("best_score"),
        "metric": metric,
        "selected_branches": [
            {
                "branch_id": f"{task_id}_robust_baseline_to_submit_candidate",
                "branch_type": "robust_baseline",
                "code_generation_mode": "Stepwise",
                "hypothesis": "Convert the strongest local-CV branch into a fully audited workstation candidate before considering official rank evidence.",
                "cross_branch_references": ["reuse promoted Kaggle4 patterns; avoid held branch failure modes"],
                "expected_delta": "preserve or improve local CV; prioritize valid submission rate before rank optimization",
                "rollback_condition": "hold if local CV does not beat best-so-far or required artifacts are missing",
            },
            {
                "branch_id": f"{task_id}_branch_diverse_ensemble",
                "branch_type": "ensemble_blend",
                "code_generation_mode": "Diff",
                "hypothesis": "Use branch-diverse top candidates to improve robustness and reduce single-model leaderboard variance.",
                "cross_branch_references": ["best promoted branch", "most stable held negative ablation"],
                "expected_delta": f"{metric} small positive delta with improved stability",
                "rollback_condition": "hold if validation_contract or claim_audit rejects the improvement claim",
            },
        ],
        "official_submit_budget": 1,
        "submit_policy": "Official submit remains blocked until human approval and rank gate prerequisites are present.",
    }


def main() -> None:
    inventory = read_json(INVENTORY)
    kaggle4 = read_json(KAGGLE4_SUMMARY)
    task_summary = task_by_id(inventory.get("task_summary", []))
    kaggle4_tasks = task_by_id(kaggle4.get("tasks", []))

    orders: list[dict[str, Any]] = []
    spaceship = kaggle4_tasks.get("spaceship_titanic") or task_summary.get("spaceship_titanic") or {}
    orders.append(make_spaceship_order(spaceship, official_for_task(inventory, "spaceship_titanic")))

    for task_id, priority in [
        ("house_prices", "P1"),
        ("titanic", "P1"),
        ("digit_recognizer", "P2"),
        ("porto_seguro_safe_driver_prediction", "P2"),
    ]:
        if task_id in task_summary:
            orders.append(make_generic_order(task_summary[task_id], priority))

    payload = {
        "schema": "academic_research_os.top30_next_evolution_orders.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_inventory": INVENTORY.relative_to(ROOT).as_posix(),
        "source_kaggle4_summary": KAGGLE4_SUMMARY.relative_to(ROOT).as_posix(),
        "controller": "MLEvolve-style Search Controller supervised by workstation AgentOrchestrator",
        "codex_role": "system_engineer_supervisor_only_no_direct_training_no_direct_submit",
        "rank_target_percentile": TOP30_TARGET,
        "official_submit_budget_policy": "conservative: <=2 official submits for P0 task batch and <=1 for proxy-only calibration tasks",
        "claim_boundary": "This is an execution order, not score evidence. Top30 requires rank_promotion_gate with top30_reached=true.",
        "orders": orders,
        "required_workstation_artifacts": [
            "agent_trace.json",
            "search_controller_decision.json",
            "metrics.json",
            "oof_predictions.csv",
            "submission.csv",
            "submission_audit.json",
            "validation_contract.json",
            "claim_audit.json",
            "rank_promotion_gate.json",
            "benchmark_claim_gate.json",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Top30 下一轮自进化任务单",
        "",
        f"- Created at: `{payload['created_at']}`",
        f"- Source inventory: `{payload['source_inventory']}`",
        f"- Rank target: `<= {TOP30_TARGET:.2f}`",
        f"- Codex role: `{payload['codex_role']}`",
        f"- Claim boundary: {payload['claim_boundary']}",
        "",
        "## Orders",
        "",
        "| priority | task | official status | rank | best local/proxy | branches | submit budget |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for order in orders:
        rank = (
            f"{order.get('official_rank')}/{order.get('leaderboard_team_count')}"
            if order.get("official_rank")
            else "n/a"
        )
        lines.append(
            f"| `{order['priority']}` | `{order['task_id']}` | `{order['current_official_status']}` | {rank} | {order.get('current_best_score')} | {len(order['selected_branches'])} | {order['official_submit_budget']} |"
        )
    lines.extend(
        [
            "",
            "## P0 Spaceship Titanic Execution Boundary",
            "",
            "- 当前官方证据：`spaceship_titanic` 仅达到前 36.8%，未达前 30%。",
            "- 下一轮必须由工作站 AgentOrchestrator 发起，Codex 不旁路写训练、不旁路提交。",
            "- 至少生成 feature_engineering、model_family、ensemble_blend 三条分支，并记录 cross-branch reference。",
            "- 只有通过 CV/OOF、submission audit、validation contract、claim audit、人审 Gate 后，才允许消耗官方提交预算。",
            "",
            "## MLEvolve Alignment",
            "",
            "- Progressive MCGS：按 P0/P1/P2 和 exploration/exploitation 选择下一分支。",
            "- Retrospective Memory：所有 hold、timeout、top30_failed 都进入失败记忆。",
            "- Adaptive Code Generation：Base/Stepwise/Diff 由分支状态和停滞状态决定。",
            "- Cross-Branch Fusion：P0 blend 分支只能引用 promoted top nodes，held nodes 只作为负证据。",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.relative_to(ROOT).as_posix(), "md": OUT_MD.relative_to(ROOT).as_posix(), "orders": len(orders)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
