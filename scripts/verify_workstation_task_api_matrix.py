from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "workspace" / "workstation_task_api_matrix_20260630.json"
OUT_MD = ROOT / "reports" / "WORKSTATION_TASK_API_MATRIX_20260630.md"


READ_ONLY_ENDPOINTS = {
    "runs": "/api/tasks/{task_id}/runs",
    "gates": "/api/tasks/{task_id}/gates",
    "evidence": "/api/tasks/{task_id}/evidence",
    "workflow": "/api/tasks/{task_id}/workflow",
    "report": "/api/tasks/{task_id}/report",
    "figures": "/api/tasks/{task_id}/figures",
}

SIDE_EFFECT_ENDPOINTS = [
    "/api/tasks/{task_id}/code-agent-draft",
    "/api/tasks/{task_id}/export-code-agent-context",
    "/api/tasks/{task_id}/generate-figures",
    "/api/tasks/{task_id}/generate-report-draft",
    "/api/tasks/{task_id}/import-agent-patch",
    "/api/tasks/{task_id}/run-local-experiment",
    "/api/tasks/{task_id}/run-mcgs-experiment",
    "/api/tasks/{task_id}/run-ensemble-experiment",
]


def get_json(base_url: str, path: str) -> tuple[bool, int | str, dict[str, Any]]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        with urlopen(url, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status == 200, response.status, payload
    except Exception as exc:  # pragma: no cover - smoke utility
        return False, "error", {"error": str(exc)}


def count_payload(endpoint: str, payload: dict[str, Any]) -> int | None:
    if endpoint in {"runs", "gates", "evidence", "figures"}:
        value = payload.get(endpoint)
        return len(value) if isinstance(value, list) else None
    if endpoint == "workflow":
        return 1 if payload.get("workflow") else 0
    if endpoint == "report":
        return 1 if payload.get("report") else 0
    return None


def inspect_task(base_url: str, task: dict[str, Any]) -> dict[str, Any]:
    task_id = str(task.get("id") or "")
    endpoint_results: dict[str, Any] = {}
    for name, template in READ_ONLY_ENDPOINTS.items():
        path = template.format(task_id=task_id)
        ok, status, payload = get_json(base_url, path)
        endpoint_results[name] = {
            "path": path,
            "status": status,
            "ok": ok and payload.get("ok") is True,
            "count": count_payload(name, payload),
            "error": payload.get("error"),
        }

    has_runs = (endpoint_results["runs"].get("count") or 0) > 0
    has_gates = (endpoint_results["gates"].get("count") or 0) > 0
    has_evidence = (endpoint_results["evidence"].get("count") or 0) > 0
    has_workflow = (endpoint_results["workflow"].get("count") or 0) > 0
    has_report = (endpoint_results["report"].get("count") or 0) > 0

    return {
        "task_id": task_id,
        "name": task.get("name"),
        "status": task.get("status"),
        "metric": task.get("metric"),
        "priority": task.get("priority"),
        "endpoint_results": endpoint_results,
        "all_read_only_endpoints_ok": all(item["ok"] for item in endpoint_results.values()),
        "closed_loop_evidence": {
            "has_runs": has_runs,
            "has_gates": has_gates,
            "has_evidence": has_evidence,
            "has_workflow": has_workflow,
            "has_report": has_report,
            "minimum_closed_loop_visible": has_runs and has_gates and has_workflow,
            "full_reportable_loop_visible": has_runs and has_gates and has_evidence and has_workflow and has_report,
        },
    }


def build_report(base_url: str) -> dict[str, Any]:
    ok, status, payload = get_json(base_url, "/api/tasks")
    if not ok or payload.get("ok") is not True:
        return {
            "schema": "academic_research_os.workstation_task_api_matrix.v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "base_url": base_url,
            "status": "failed",
            "task_index_status": status,
            "error": payload.get("error") or "GET /api/tasks failed",
            "tasks": [],
        }

    tasks = payload.get("tasks") if isinstance(payload.get("tasks"), list) else []
    records = [inspect_task(base_url, task) for task in tasks if task.get("id")]
    endpoint_ok = all(record["all_read_only_endpoints_ok"] for record in records)
    min_loop_count = sum(1 for record in records if record["closed_loop_evidence"]["minimum_closed_loop_visible"])
    full_loop_count = sum(1 for record in records if record["closed_loop_evidence"]["full_reportable_loop_visible"])
    evidence_missing = [
        record["task_id"]
        for record in records
        if not record["closed_loop_evidence"]["has_evidence"]
    ]
    report_missing = [
        record["task_id"]
        for record in records
        if not record["closed_loop_evidence"]["has_report"]
    ]

    return {
        "schema": "academic_research_os.workstation_task_api_matrix.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "status": "passed" if endpoint_ok else "failed",
        "task_count": len(records),
        "all_read_only_endpoints_ok": endpoint_ok,
        "minimum_closed_loop_visible_count": min_loop_count,
        "full_reportable_loop_visible_count": full_loop_count,
        "evidence_missing_tasks": evidence_missing,
        "report_missing_tasks": report_missing,
        "read_only_endpoints_checked": READ_ONLY_ENDPOINTS,
        "side_effect_endpoints_not_called": SIDE_EFFECT_ENDPOINTS,
        "tasks": records,
        "claim_boundary": "只读 API 矩阵只证明路由和数据可见，不证明新训练、官方 Kaggle 分数质量或奖牌结果。",
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# 工作站任务 API 闭环矩阵",
        "",
        f"- 生成时间：`{report['created_at']}`",
        f"- 工作站地址：`{report['base_url']}`",
        f"- 总状态：`{report['status']}`",
        f"- 任务数：`{report.get('task_count', 0)}`",
        f"- 只读接口全部通过：`{report.get('all_read_only_endpoints_ok')}`",
        f"- 最小闭环可见任务：`{report.get('minimum_closed_loop_visible_count', 0)}`",
        f"- 完整可汇报闭环任务：`{report.get('full_reportable_loop_visible_count', 0)}`",
        "",
        "## 说明",
        "",
        "本检查只调用 GET 只读接口，不触发训练、代码生成、报告重写、Kaggle 提交或 GPU 作业。",
        "",
        "## 任务矩阵",
        "",
        "| task | status | runs | gates | evidence | workflow | report | figures | full loop |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in report.get("tasks", []):
        endpoints = record["endpoint_results"]
        loop = record["closed_loop_evidence"]
        lines.append(
            "| `{task}` | `{status}` | {runs} | {gates} | {evidence} | {workflow} | {report} | {figures} | `{full}` |".format(
                task=record["task_id"],
                status=record.get("status"),
                runs=endpoints["runs"].get("count"),
                gates=endpoints["gates"].get("count"),
                evidence=endpoints["evidence"].get("count"),
                workflow=endpoints["workflow"].get("count"),
                report=endpoints["report"].get("count"),
                figures=endpoints["figures"].get("count"),
                full=loop["full_reportable_loop_visible"],
            )
        )

    lines.extend([
        "",
        "## 仍需补齐",
        "",
        f"- evidence 缺失任务：`{', '.join(report.get('evidence_missing_tasks', [])) or 'none'}`",
        f"- report 缺失任务：`{', '.join(report.get('report_missing_tasks', [])) or 'none'}`",
        "",
        "## 未调用的副作用接口",
        "",
    ])
    for endpoint in report.get("side_effect_endpoints_not_called", []):
        lines.append(f"- `{endpoint}`")
    lines.extend(["", "## Claim Boundary", "", report.get("claim_boundary", "")])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify read-only task API matrix for the workstation.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = build_report(args.base_url)
    if args.write_report:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(report)
    print(json.dumps({
        "status": report["status"],
        "task_count": report.get("task_count", 0),
        "minimum_closed_loop_visible_count": report.get("minimum_closed_loop_visible_count", 0),
        "full_reportable_loop_visible_count": report.get("full_reportable_loop_visible_count", 0),
        "evidence_missing_tasks": report.get("evidence_missing_tasks", []),
        "report_missing_tasks": report.get("report_missing_tasks", []),
        "json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
        "md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
