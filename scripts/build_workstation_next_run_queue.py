from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKFILL_JSON = ROOT / "workspace" / "workstation_evidence_backfill_plan_20260630.json"
TASK_MATRIX_JSON = ROOT / "workspace" / "workstation_task_api_matrix_20260630.json"
FIGMA_GATE_JSON = ROOT / "workspace" / "workstation_figma_design_gate_20260630.json"
OUT_JSON = ROOT / "workspace" / "workstation_next_run_queue_20260630.json"
OUT_MD = ROOT / "reports" / "WORKSTATION_NEXT_RUN_QUEUE_20260630.md"


TASK_PRIORITIES = {
    "kaggle_new_competition_smoke": {
        "priority": 1,
        "reason": "资源恢复后的最小端到端新任务烟测，验证 create run -> gates -> evidence -> report。",
        "suggested_mode": "smoke_closed_loop",
        "time_budget_minutes": 20,
    },
    "tabular_playground_series_aug_2022": {
        "priority": 2,
        "reason": "中等复杂表格任务，适合验证 MLEvolve-style branch search 与 submission audit。",
        "suggested_mode": "robust_baseline_first",
        "time_budget_minutes": 60,
    },
    "telco_churn": {
        "priority": 3,
        "reason": "运行时间较短的分类任务，适合验证低成本闭环和报告稳定性。",
        "suggested_mode": "local_or_cpu_baseline",
        "time_budget_minutes": 30,
    },
    "ps3e1": {
        "priority": 4,
        "reason": "Playground 系列表格任务，可用于补齐 benchmark registry 到 workstation run。",
        "suggested_mode": "robust_baseline_first",
        "time_budget_minutes": 45,
    },
    "ps3e7": {
        "priority": 5,
        "reason": "Playground 系列表格任务，可用于批量任务 API/报告闭环回归。",
        "suggested_mode": "robust_baseline_first",
        "time_budget_minutes": 45,
    },
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def global_blockers() -> list[str]:
    blockers = ["gpu_resource_blocked"]
    figma_gate = read_json(FIGMA_GATE_JSON)
    if figma_gate.get("blocked") is True:
        blockers.append("figma_auth_blocked_for_design_parity_only")
    return blockers


def task_lookup() -> dict[str, dict[str, Any]]:
    matrix = read_json(TASK_MATRIX_JSON)
    tasks = matrix.get("tasks") if isinstance(matrix.get("tasks"), list) else []
    return {str(task.get("task_id")): task for task in tasks if task.get("task_id")}


def classify_task(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    priority_info = TASK_PRIORITIES.get(task_id, {})
    fallback_priority = 6 if task_id.startswith("tps_") else 10
    priority = int(priority_info.get("priority", fallback_priority))
    task_status = task.get("status") or "unknown"
    metric = task.get("metric")
    resource_mode = "workstation_resource_mode"
    if priority_info.get("suggested_mode") == "local_or_cpu_baseline":
        resource_mode = "workstation_cpu_or_remote_gpu"
    return {
        "task_id": task_id,
        "task_status": task_status,
        "priority": priority,
        "metric": metric,
        "suggested_mode": priority_info.get("suggested_mode", "robust_baseline_first"),
        "resource_mode": resource_mode,
        "time_budget_minutes": priority_info.get("time_budget_minutes", 45),
        "reason": priority_info.get(
            "reason",
            "注册任务尚未形成 workstation run；GPU/resource gate 恢复后再进入闭环。",
        ),
        "required_preflight": [
            "workstation summary ready",
            "DeepSeek/code-agent cache hit target >= 80% before batch code generation",
            "GPU/HPC resource gate ready or explicit CPU-safe mode",
            "Kaggle credential read-only smoke if official competition data is needed",
        ],
        "required_artifacts": [
            "workstation_run_id",
            "agent_trace",
            "search_controller_decision.json",
            "validation_contract.json",
            "metrics.json or failure_review.json",
            "artifact_manifest.json",
            "claim_audit.json",
            "report.md",
        ],
        "blocked_by": ["gpu_resource_blocked"],
        "claim_boundary": (
            "Queued task is not a completed experiment and must not be counted as valid "
            "submission, rank, top30 or medal evidence."
        ),
    }


def build_queue() -> dict[str, Any]:
    backfill = read_json(BACKFILL_JSON)
    buckets = backfill.get("buckets") if isinstance(backfill.get("buckets"), dict) else {}
    registry_only = buckets.get("not_started_or_registry_only")
    if not isinstance(registry_only, list):
        registry_only = []
    lookup = task_lookup()
    entries = [
        classify_task(str(item.get("task_id")), lookup.get(str(item.get("task_id")), {}))
        for item in registry_only
        if item.get("task_id")
    ]
    entries.sort(key=lambda item: (item["priority"], item["task_id"]))
    return {
        "schema": "academic_research_os.workstation_next_run_queue.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_backfill_plan": str(BACKFILL_JSON.relative_to(ROOT)).replace("\\", "/"),
        "source_task_matrix": str(TASK_MATRIX_JSON.relative_to(ROOT)).replace("\\", "/"),
        "queue_count": len(entries),
        "ready_to_start_now": False,
        "global_blockers": global_blockers(),
        "queue": entries,
        "recommended_first_batch": [entry["task_id"] for entry in entries[:3]],
        "claim_boundary": (
            "This queue is a launch plan only. It does not prove new training, official "
            "Kaggle submission, rank, medal, or MLE-Bench parity."
        ),
    }


def write_markdown(queue: dict[str, Any]) -> None:
    lines = [
        "# 工作站下一轮自动学习 Run 队列",
        "",
        f"- 生成时间：`{queue['created_at']}`",
        f"- 队列任务数：`{queue['queue_count']}`",
        f"- 当前是否可立即启动：`{queue['ready_to_start_now']}`",
        f"- 全局阻断：`{', '.join(queue['global_blockers']) or 'none'}`",
        f"- 推荐第一批：`{', '.join(queue['recommended_first_batch']) or 'none'}`",
        "",
        "## 队列",
        "",
        "| priority | task | status | mode | resource | budget(min) | reason |",
        "| ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    for item in queue["queue"]:
        lines.append(
            "| {priority} | `{task}` | `{status}` | `{mode}` | `{resource}` | {budget} | {reason} |".format(
                priority=item["priority"],
                task=item["task_id"],
                status=item["task_status"],
                mode=item["suggested_mode"],
                resource=item["resource_mode"],
                budget=item["time_budget_minutes"],
                reason=item["reason"],
            )
        )
    lines.extend(
        [
            "",
            "## 启动前硬门禁",
            "",
            "1. GPU/HPC resource gate 必须恢复，或任务被明确标记为 CPU-safe mode。",
            "2. DeepSeek/Code Agent 批量生成前必须确认缓存命中率目标仍然满足。",
            "3. 官方 Kaggle 提交仍必须经过 submission audit、claim audit 和人工 approval gate。",
            "4. 队列任务不得计入 valid submission、rank、top30 或 medal 统计。",
            "",
            "## Claim Boundary",
            "",
            queue["claim_boundary"],
        ]
    )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the next workstation run queue for registry-only tasks.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    queue = build_queue()
    if args.write_report:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(queue)

    print(
        json.dumps(
            {
                "queue_count": queue["queue_count"],
                "ready_to_start_now": queue["ready_to_start_now"],
                "recommended_first_batch": queue["recommended_first_batch"],
                "global_blockers": queue["global_blockers"],
                "json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
                "md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
