from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
INVENTORY_PATH = ROOT / "workspace" / "kaggle_experiment_inventory_20260624.json"
S6E6_RUN_ROOT = ROOT / "workspace" / "workstation_runs" / "playground_series_s6e6"
OUT_JSON = ROOT / "workspace" / "mlevolve_next_orders_20260625.json"
OUT_MD = ROOT / "reports" / "MLEVOLVE_NEXT_ORDERS_20260625.md"
MEMORY_PATH = ROOT / "workspace" / "top30_retrospective_memory_20260625.json"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def latest_s6e6_recovery_plan() -> tuple[Path | None, dict[str, Any]]:
    if not S6E6_RUN_ROOT.exists():
        return None, {}
    candidates = sorted(
        S6E6_RUN_ROOT.glob("*/score_regression_recovery_plan.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None, {}
    return candidates[0], read_json(candidates[0])


def task_summary_by_id(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("task_id")): item
        for item in inventory.get("task_summary", [])
        if isinstance(item, dict) and item.get("task_id")
    }


def official_by_task(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in inventory.get("official_submission_records", []):
        if not isinstance(item, dict) or not item.get("task_id"):
            continue
        current = result.get(str(item["task_id"]))
        if current is None:
            result[str(item["task_id"])] = item
            continue
        current_score = current.get("public_score")
        item_score = item.get("public_score")
        if isinstance(item_score, (int, float)) and (
            not isinstance(current_score, (int, float)) or item_score > current_score
        ):
            result[str(item["task_id"])] = item
    return result


def append_memory(records: list[dict[str, Any]]) -> None:
    payload = read_json(MEMORY_PATH)
    existing = payload.get("records", []) if isinstance(payload.get("records"), list) else []
    by_id = {str(item.get("memory_id")): item for item in existing if isinstance(item, dict)}
    for record in records:
        by_id[str(record["memory_id"])] = record
    write_json(
        MEMORY_PATH,
        {
            "schema": "academic_research_os.top30_retrospective_memory.v1",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "records": list(by_id.values()),
        },
    )


def s6e6_order(task: dict[str, Any], official: dict[str, Any] | None, recovery: dict[str, Any], recovery_path: Path | None) -> dict[str, Any]:
    failed = recovery.get("failed_candidate", {}) if isinstance(recovery.get("failed_candidate"), dict) else {}
    rollback = recovery.get("rollback_baseline", {}) if isinstance(recovery.get("rollback_baseline"), dict) else {}
    blocked_reasons = failed.get("blocked_reasons") if isinstance(failed.get("blocked_reasons"), list) else []
    return {
        "task_id": "playground_series_s6e6",
        "priority": "P0",
        "controller_pattern": "progressive_mcgs_with_recovery_memory",
        "current_official_status": "official-known-rank-unknown" if official else "proxy_only",
        "official_public_score": official.get("public_score") if official else None,
        "official_rank": official.get("rank") if official else None,
        "rank_percentile": official.get("rank_percentile") if official else None,
        "current_best_run": task.get("best_run"),
        "current_best_score": task.get("best_score"),
        "recent_failed_run": recovery.get("workstation_run_id"),
        "recent_failed_score": failed.get("validation_score"),
        "recent_blocked_reasons": blocked_reasons,
        "recovery_plan_path": recovery_path.relative_to(ROOT).as_posix() if recovery_path else None,
        "protected_rollback_baseline": {
            "experiment_id": rollback.get("experiment_id") or "EXP007",
            "validation_balanced_accuracy": rollback.get("validation_balanced_accuracy"),
            "public_score": rollback.get("public_score"),
            "policy": rollback.get("policy") or "preserve_as_safe_submission_baseline",
        },
        "selected_branches": [
            {
                "branch_id": "s6e6_probability_contract_replay",
                "branch_type": "artifact_replay",
                "code_generation_mode": "Diff",
                "hypothesis": "Re-bind or reproduce the EXP003/EXP004/EXP006 probability contract before any new official candidate.",
                "workstation_action": "run_s6e6_artifact_replay_candidate",
                "resource_mode": "artifact_only_no_training_no_submit",
                "cross_branch_references": ["EXP003_LightGBM", "EXP004_XGBoost", "EXP006_CatBoost", "EXP007 rollback baseline"],
                "expected_delta": "recover safe EXP007-level local OOF and prevent weaker one-shot boosting from replacing best-so-far",
                "rollback_condition": "hold if probability assets are missing or score gate does not pass",
            },
            {
                "branch_id": "s6e6_calibration_repair_blend",
                "branch_type": "calibration_repair",
                "code_generation_mode": "Stepwise",
                "hypothesis": "Repair the three suspicious calibration bins before considering a fresh submit candidate.",
                "workstation_action": "run_s6e6_exp024_multi_asset_frontier_blend",
                "resource_mode": "artifact_blend_then_gate",
                "cross_branch_references": ["latest A800/A40 boosting artifact", "EXP017 current official best", "EXP007 rollback baseline"],
                "expected_delta": "balanced_accuracy >= EXP007 and calibration suspicious bins <= 0",
                "rollback_condition": "hold if validation margin remains negative or calibration bins remain suspicious",
            },
            {
                "branch_id": "s6e6_diverse_gpu_single_model_then_fusion",
                "branch_type": "model_family_diversity",
                "code_generation_mode": "Diff",
                "hypothesis": "Run a single-model diversity branch only if it creates a new probability asset for later fusion, not immediate submission.",
                "workstation_action": "run_s6e6_exp025_single_model_diversity",
                "resource_mode": "hpc_gpu_after_gate",
                "cross_branch_references": ["XGBoost had high raw accuracy but poor balanced accuracy", "CatBoost improved minority balance but weak logloss"],
                "expected_delta": "produce branch-diverse OOF asset for fusion; no official submit from this branch alone",
                "rollback_condition": "hold if OOF balanced accuracy or logloss fails recovery frontier",
            },
        ],
        "official_submit_budget": 0,
        "submit_policy": "No official submit for these recovery branches until score_improvement_gate, submission_audit, claim_audit, and human submission gate pass.",
    }


def generic_order(task: dict[str, Any], official: dict[str, Any] | None, priority: str) -> dict[str, Any]:
    task_id = str(task.get("task_id"))
    official_status = "top30_failed" if official and not official.get("rank_unknown") else "proxy_only"
    return {
        "task_id": task_id,
        "priority": priority,
        "controller_pattern": "robust_baseline_then_branch_diverse_fusion",
        "current_official_status": official_status,
        "official_public_score": official.get("public_score") if official else None,
        "official_rank": official.get("rank") if official else None,
        "rank_percentile": official.get("rank_percentile") if official else None,
        "current_best_run": task.get("best_run"),
        "current_best_score": task.get("best_score"),
        "metric": task.get("metric"),
        "selected_branches": [
            {
                "branch_id": f"{task_id}_best_so_far_protection",
                "branch_type": "best_so_far_guard",
                "code_generation_mode": "Stepwise",
                "hypothesis": "First protect the strongest local/proxy candidate with full gates before leaderboard optimization.",
                "workstation_action": "run_workstation_ensemble",
                "resource_mode": "workstation_agent_orchestrator",
                "expected_delta": "valid submission and reproducible local CV baseline",
                "rollback_condition": "hold if required artifacts are missing or local score regresses",
            },
            {
                "branch_id": f"{task_id}_branch_diverse_fusion",
                "branch_type": "ensemble_blend",
                "code_generation_mode": "Diff",
                "hypothesis": "Fuse branch-diverse promoted nodes and reject held candidates except as negative memory.",
                "workstation_action": "run_workstation_ensemble",
                "resource_mode": "workstation_agent_orchestrator",
                "expected_delta": "small local/proxy improvement with lower public mismatch risk",
                "rollback_condition": "hold if claim_audit marks unsupported improvement or submission schema fails",
            },
        ],
        "official_submit_budget": 1 if priority in {"P0", "P1"} else 0,
        "submit_policy": "Official submit requires current-turn human approval and rank gate; proxy evidence is not rank evidence.",
    }


def main() -> None:
    inventory = read_json(INVENTORY_PATH)
    tasks = task_summary_by_id(inventory)
    official = official_by_task(inventory)
    recovery_path, recovery = latest_s6e6_recovery_plan()

    orders = []
    if "playground_series_s6e6" in tasks:
        orders.append(s6e6_order(tasks["playground_series_s6e6"], official.get("playground_series_s6e6"), recovery, recovery_path))
    for task_id, priority in [
        ("spaceship_titanic", "P0"),
        ("house_prices", "P1"),
        ("titanic", "P1"),
        ("digit_recognizer", "P2"),
    ]:
        if task_id in tasks:
            orders.append(generic_order(tasks[task_id], official.get(task_id), priority))

    memory_records = []
    if recovery:
        failed = recovery.get("failed_candidate", {}) if isinstance(recovery.get("failed_candidate"), dict) else {}
        memory_records.append(
            {
                "memory_id": f"s6e6::{recovery.get('workstation_run_id')}::score_gate_blocked",
                "task_type": "kaggle_tabular",
                "dataset_profile": {"task_id": "playground_series_s6e6", "competition": "playground-series-s6e6"},
                "method": "hpc_boosting_ensemble_lgb_xgb_cat",
                "what_worked": "workstation GPU run completed and produced metrics, probabilities, submission, report, stdout and stderr artifacts",
                "what_failed": "; ".join(failed.get("blocked_reasons", [])) if isinstance(failed.get("blocked_reasons"), list) else "score gate blocked",
                "metric_delta": failed.get("validation_margin_vs_exp007_oof"),
                "reusable_strategy": "Do not submit one-shot boosting below EXP007; return to probability contract replay or calibration repair before fresh official candidate.",
                "failure_pattern": "score_gate_blocked_calibration_and_baseline_gap",
                "linked_exp_ids": [str(recovery.get("workstation_run_id"))],
            }
        )
        append_memory(memory_records)

    payload = {
        "schema": "academic_research_os.mlevolve_next_orders.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_inventory": INVENTORY_PATH.relative_to(ROOT).as_posix(),
        "source_recovery_plan": recovery_path.relative_to(ROOT).as_posix() if recovery_path else None,
        "controller": "MLEvolve-style Search Controller with XCIENTIST gates",
        "codex_role": "supervisor_only_no_direct_training_no_direct_submit",
        "claim_boundary": "These are next execution orders, not score evidence. Top30 requires official rank artifact and rank_promotion_gate.",
        "orders": orders,
        "memory_records_added": memory_records,
    }
    write_json(OUT_JSON, payload)

    lines = [
        "# MLEvolve Next Orders",
        "",
        f"- Created at: `{payload['created_at']}`",
        f"- Source inventory: `{payload['source_inventory']}`",
        f"- Source recovery plan: `{payload['source_recovery_plan']}`",
        f"- Codex role: `{payload['codex_role']}`",
        f"- Claim boundary: {payload['claim_boundary']}",
        "",
        "## Orders",
        "",
        "| priority | task | controller | best/proxy | official score | branches | submit budget |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for order in orders:
        lines.append(
            f"| `{order['priority']}` | `{order['task_id']}` | `{order['controller_pattern']}` | {order.get('current_best_score')} | {order.get('official_public_score')} | {len(order['selected_branches'])} | {order['official_submit_budget']} |"
        )
    lines.extend(
        [
            "",
            "## Immediate S6E6 Recovery Rule",
            "",
            "- Do not repeat the one-shot boosting template as an official candidate while it remains below EXP007 and has suspicious calibration bins.",
            "- First replay or reproduce the EXP003/EXP004/EXP006 probability contract, then run calibration repair/fusion gates.",
            "- Official submission budget for S6E6 recovery branches is `0` until score gate and human gate both pass.",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.relative_to(ROOT).as_posix(), "md": OUT_MD.relative_to(ROOT).as_posix(), "orders": len(orders), "memory_records_added": len(memory_records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
