from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK_MATRIX_JSON = ROOT / "workspace" / "workstation_task_api_matrix_20260630.json"
OUT_JSON = ROOT / "workspace" / "workstation_evidence_backfill_plan_20260630.json"
OUT_MD = ROOT / "reports" / "WORKSTATION_EVIDENCE_BACKFILL_PLAN_20260630.md"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def classify_task(record: dict[str, Any]) -> dict[str, Any]:
    loop = record.get("closed_loop_evidence") or {}
    endpoints = record.get("endpoint_results") or {}
    counts = {
        key: (endpoints.get(key) or {}).get("count")
        for key in ("runs", "gates", "evidence", "workflow", "report", "figures")
    }
    task_id = str(record.get("task_id") or "")
    has_runs = bool(loop.get("has_runs"))
    has_gates = bool(loop.get("has_gates"))
    has_evidence = bool(loop.get("has_evidence"))
    has_workflow = bool(loop.get("has_workflow"))
    has_report = bool(loop.get("has_report"))

    if loop.get("full_reportable_loop_visible"):
        category = "complete_reportable_loop"
        priority = 5
        recommended_action = "keep_current_evidence; include in teacher-facing stable examples"
    elif has_runs and has_gates and has_evidence and has_workflow and not has_report:
        category = "report_only_backfill"
        priority = 1
        recommended_action = "use workstation report agent to generate draft report and submit final_report gate"
    elif has_runs and has_gates and not (has_evidence and has_workflow):
        category = "evidence_workflow_backfill"
        priority = 2
        recommended_action = "replay existing run artifacts into evidence ledger, workflow trace, validation contract and claim audit"
    elif has_runs and not has_gates:
        category = "governance_backfill"
        priority = 3
        recommended_action = "create gate records, failure review and minimal evidence before any new training"
    else:
        category = "not_started_or_registry_only"
        priority = 4
        recommended_action = "do not claim closed loop; schedule future workstation run after GPU/resource gate is ready"

    return {
        "task_id": task_id,
        "name": record.get("name"),
        "status": record.get("status"),
        "metric": record.get("metric"),
        "priority": priority,
        "category": category,
        "counts": counts,
        "recommended_action": recommended_action,
        "claim_boundary": (
            "This remediation item is based on read-only API evidence. "
            "It does not prove new training, official score, rank, or medal."
        ),
    }


def build_plan() -> dict[str, Any]:
    matrix = read_json(TASK_MATRIX_JSON)
    tasks = matrix.get("tasks") if isinstance(matrix.get("tasks"), list) else []
    items = [classify_task(task) for task in tasks]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        buckets.setdefault(item["category"], []).append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (item["priority"], item["task_id"]))

    return {
        "schema": "academic_research_os.workstation_evidence_backfill_plan.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_matrix": str(TASK_MATRIX_JSON.relative_to(ROOT)).replace("\\", "/"),
        "task_count": len(items),
        "complete_reportable_loop_count": len(buckets.get("complete_reportable_loop", [])),
        "report_only_backfill_count": len(buckets.get("report_only_backfill", [])),
        "evidence_workflow_backfill_count": len(buckets.get("evidence_workflow_backfill", [])),
        "governance_backfill_count": len(buckets.get("governance_backfill", [])),
        "not_started_or_registry_only_count": len(buckets.get("not_started_or_registry_only", [])),
        "next_best_actions": [
            "Backfill reports for report_only_backfill tasks first; this improves teacher-facing completeness without training.",
            "Replay evidence/workflow for existing failed or partial runs before starting new experiments.",
            "Keep not_started_or_registry_only tasks out of success-rate claims until a workstation run exists.",
            "Do not use this plan as official Kaggle score, rank, medal, or MLE-Bench parity evidence.",
        ],
        "buckets": buckets,
    }


def write_markdown(plan: dict[str, Any]) -> None:
    lines = [
        "# 工作站 Evidence / Report 回填计划",
        "",
        f"- 生成时间：`{plan['created_at']}`",
        f"- 来源矩阵：`{plan['source_matrix']}`",
        f"- 任务数：`{plan['task_count']}`",
        f"- 完整可汇报闭环：`{plan['complete_reportable_loop_count']}`",
        f"- 只缺报告：`{plan['report_only_backfill_count']}`",
        f"- 缺 evidence/workflow：`{plan['evidence_workflow_backfill_count']}`",
        f"- 缺 governance gate：`{plan['governance_backfill_count']}`",
        f"- 尚未形成 run：`{plan['not_started_or_registry_only_count']}`",
        "",
        "## 下一步优先级",
        "",
    ]
    lines.extend(f"{index}. {item}" for index, item in enumerate(plan["next_best_actions"], start=1))
    lines.extend([
        "",
        "## 分桶明细",
        "",
    ])
    for category, items in plan["buckets"].items():
        lines.extend([
            f"### {category}",
            "",
            "| task | status | runs | gates | evidence | workflow | report | recommended action |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for item in items:
            counts = item["counts"]
            lines.append(
                "| `{task}` | `{status}` | {runs} | {gates} | {evidence} | {workflow} | {report} | {action} |".format(
                    task=item["task_id"],
                    status=item.get("status"),
                    runs=counts.get("runs"),
                    gates=counts.get("gates"),
                    evidence=counts.get("evidence"),
                    workflow=counts.get("workflow"),
                    report=counts.get("report"),
                    action=item["recommended_action"],
                )
            )
        lines.append("")

    lines.extend([
        "## Claim Boundary",
        "",
        "本计划只说明现有工作站 artifact 的证据完整度，不证明新训练、官方 Kaggle 分数、排名、奖牌或 MLE-Bench 75 对齐完成。",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a read-only remediation plan for task evidence/report gaps.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    plan = build_plan()
    if args.write_report:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(plan)

    print(json.dumps({
        "task_count": plan["task_count"],
        "complete_reportable_loop_count": plan["complete_reportable_loop_count"],
        "report_only_backfill_count": plan["report_only_backfill_count"],
        "evidence_workflow_backfill_count": plan["evidence_workflow_backfill_count"],
        "governance_backfill_count": plan["governance_backfill_count"],
        "not_started_or_registry_only_count": plan["not_started_or_registry_only_count"],
        "json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
        "md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
