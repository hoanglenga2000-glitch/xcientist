from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "workspace" / "workstation_online_smoke_summary_20260701.json"
OUT_MD = ROOT / "reports" / "WORKSTATION_ONLINE_SMOKE_SUMMARY_20260701.md"

SOURCE_FILES = {
    "server_health": ROOT / "workspace" / "workstation_server_health_20260630.json",
    "runtime_navigation": ROOT / "workspace" / "workstation_runtime_navigation_20260630.json",
    "frontend_api_contract": ROOT / "workspace" / "workstation_frontend_api_contract_20260630.json",
    "ui_action_contract": ROOT / "workspace" / "workstation_ui_action_contract_20260630.json",
    "browser_render_smoke": ROOT / "workspace" / "workstation_browser_render_smoke_20260630.json",
    "interactive_controls": ROOT / "workspace" / "workstation_interactive_controls_20260701.json",
    "responsive_smoke": ROOT / "workspace" / "workstation_responsive_smoke_20260701.json",
    "code_agent_file_navigation": ROOT / "workspace" / "code_agent_file_navigation_20260701.json",
    "stateful_interactions": ROOT / "workspace" / "workstation_stateful_interactions_20260701.json",
    "task_api_matrix": ROOT / "workspace" / "workstation_task_api_matrix_20260630.json",
    "learning_loop_readiness": ROOT / "workspace" / "workstation_learning_loop_readiness_20260630.json",
    "settings_secret_redaction": ROOT / "workspace" / "workstation_settings_secret_redaction_20260701.json",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_summary(base_url: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base_url.rstrip()}/api/workstation-summary", timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def compact_source_status(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {
        "name": name,
        "status": payload.get("status", "unknown"),
        "missing": bool(payload.get("missing")),
    }
    if name == "ui_action_contract":
        item.update({
            "data_ui_action_count": payload.get("data_ui_action_count"),
            "failed_data_ui_action_count": len(payload.get("failed_data_ui_actions", [])),
            "telemetry_only_data_ui_action_count": payload.get("telemetry_only_data_ui_action_count"),
        })
    if name == "frontend_api_contract":
        item.update({
            "fetch_contract_count": payload.get("fetch_contract_count"),
            "failed_fetch_contract_count": len(payload.get("failed_fetch_contracts", [])),
            "navigation_ok": payload.get("navigation_contract", {}).get("ok"),
        })
    if name == "task_api_matrix":
        item.update({
            "task_count": payload.get("task_count"),
            "minimum_closed_loop_visible_count": payload.get("minimum_closed_loop_visible_count"),
            "full_reportable_loop_visible_count": payload.get("full_reportable_loop_visible_count"),
        })
    if name == "interactive_controls":
        item.update({
            "missing_control_count": payload.get("missing_control_count"),
            "runtime_error_count": payload.get("runtime_error_count"),
        })
    if name == "responsive_smoke":
        item.update({
            "check_count": payload.get("check_count"),
            "failed_count": payload.get("failed_count"),
            "runtime_error_count": payload.get("runtime_error_count"),
        })
    if name == "code_agent_file_navigation":
        item.update({
            "file_check_count": len(payload.get("results", [])),
            "failed_file_count": len(payload.get("failed_files", [])),
            "runtime_error_count": payload.get("runtime_error_count"),
        })
    if name == "stateful_interactions":
        item.update({
            "stateful_check_count": len(payload.get("results", [])),
            "failed_stateful_check_count": len(payload.get("failed_checks", [])),
            "runtime_error_count": payload.get("runtime_error_count"),
        })
    if name == "learning_loop_readiness":
        item.update({
            "memory_records": payload.get("memory_records"),
            "search_order_records": payload.get("search_order_records"),
            "observed_runs": payload.get("observed_runs"),
            "next_run_ready": payload.get("next_run_ready"),
            "resource_blockers": payload.get("resource_blockers", []),
        })
    if name == "settings_secret_redaction":
        item.update({"finding_count": payload.get("finding_count")})
    if name == "server_health":
        item.update({
            "port_pids": payload.get("port_pids"),
            "warnings": payload.get("warnings", []),
        })
    return item


def build_report(base_url: str) -> dict[str, Any]:
    sources = {name: read_json(path) for name, path in SOURCE_FILES.items()}
    source_statuses = [compact_source_status(name, payload) for name, payload in sources.items()]
    failed = [item for item in source_statuses if item["status"] != "passed"]
    summary = fetch_summary(base_url)
    learning = summary.get("learning_loop_readiness", {})
    audit = summary.get("verified_launch_audit", {})
    connector_status = summary.get("connector_status", {})
    blockers = audit.get("blockers") or []
    return {
        "schema": "academic_research_os.workstation_online_smoke_summary.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "status": "passed_with_resource_blockers" if not failed and blockers else "passed" if not failed else "failed",
        "source_statuses": source_statuses,
        "failed_sources": failed,
        "runtime": {
            "launch_state": audit.get("launch_state"),
            "blockers": blockers,
            "kaggle_state": connector_status.get("kaggle", {}).get("state"),
            "gpu_state": connector_status.get("gpu", {}).get("state"),
            "deepseek_state": connector_status.get("deepseek", {}).get("state"),
            "memory_records": learning.get("memory", {}).get("record_count"),
            "search_order_records": learning.get("search_orders", {}).get("record_count"),
            "observed_runs": learning.get("training_progress", {}).get("observed_runs"),
            "scored_runs": learning.get("training_progress", {}).get("scored_runs"),
            "promoted_runs": learning.get("training_progress", {}).get("promoted_runs"),
            "next_run_ready": learning.get("next_run_queue", {}).get("ready_to_start_now"),
        },
        "claim_boundary": (
            "This summary proves local workstation online smoke coverage only. Resource blockers may still prevent "
            "new GPU training, batch code-agent generation, or official Kaggle submission. It does not prove Kaggle medal rate."
        ),
    }


def write_markdown(report: dict[str, Any]) -> None:
    runtime = report["runtime"]
    lines = [
        "# AI 科研工作站上线 Smoke 汇总",
        "",
        f"- 生成时间：`{report['created_at']}`",
        f"- 工作站地址：`{report['base_url']}`",
        f"- 汇总状态：`{report['status']}`",
        f"- Launch state：`{runtime.get('launch_state')}`",
        f"- Blockers：`{', '.join(runtime.get('blockers') or []) or 'none'}`",
        "",
        "## 验证项",
        "",
        "| 检查项 | 状态 | 关键数字 |",
        "| --- | --- | --- |",
    ]
    for item in report["source_statuses"]:
        extras = []
        for key in [
            "fetch_contract_count",
            "data_ui_action_count",
            "task_count",
            "minimum_closed_loop_visible_count",
            "full_reportable_loop_visible_count",
            "missing_control_count",
            "check_count",
            "failed_count",
            "file_check_count",
            "failed_file_count",
            "stateful_check_count",
            "failed_stateful_check_count",
            "runtime_error_count",
            "memory_records",
            "search_order_records",
            "observed_runs",
            "finding_count",
        ]:
            if item.get(key) is not None:
                extras.append(f"{key}={item[key]}")
        lines.append(f"| `{item['name']}` | `{item['status']}` | {', '.join(extras) or '-'} |")

    lines.extend([
        "",
        "## 运行状态",
        "",
        f"- Kaggle：`{runtime.get('kaggle_state')}`",
        f"- GPU：`{runtime.get('gpu_state')}`",
        f"- DeepSeek：`{runtime.get('deepseek_state')}`",
        f"- memory records：`{runtime.get('memory_records')}`",
        f"- search/evolution orders：`{runtime.get('search_order_records')}`",
        f"- observed runs：`{runtime.get('observed_runs')}`",
        f"- scored runs：`{runtime.get('scored_runs')}`",
        f"- promoted runs：`{runtime.get('promoted_runs')}`",
        f"- next run ready：`{runtime.get('next_run_ready')}`",
        "",
        "## 声明边界",
        "",
        report["claim_boundary"],
        "",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a consolidated workstation online smoke summary.")
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
        "failed_source_count": len(report["failed_sources"]),
        "blockers": report["runtime"].get("blockers"),
        "json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
        "md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0 if not report["failed_sources"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
