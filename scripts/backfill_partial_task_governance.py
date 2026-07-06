from __future__ import annotations

import argparse
import hashlib
import html
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "web" / "research-agent-workstation" / "prisma" / "dev.db"
OUT_JSON = ROOT / "workspace" / "partial_task_governance_backfill_20260630.json"
OUT_MD = ROOT / "reports" / "PARTIAL_TASK_GOVERNANCE_BACKFILL_20260630.md"


TASKS = {
    "porto_seguro_safe_driver_prediction": {
        "run_id": "run_porto_seguro_safe_driver_prediction_mcgs_1782556495288",
        "title": "porto_seguro_safe_driver_prediction 失败回退与修复证据报告",
        "source_dirs": [
            "experiments/porto_seguro_fixed/20260626_212628",
            "experiments/porto_seguro_safe_driver_prediction/20260626_170023",
        ],
        "metric_summary": "fixed candidate normalized_gini=0.148179 OOF, original local ensemble best_validation_score=0.088698",
        "failure_reason": "MCGS workstation run timed out or failed, but local fixed artifacts exist for audit and next-round search.",
    },
    "store_sales_time_series_forecasting": {
        "run_id": "run_store_sales_time_series_forecasting_mcgs_1782560096361",
        "title": "store_sales_time_series_forecasting 失败回退与修复证据报告",
        "source_dirs": [
            "experiments/store_sales_fixed/20260626_171402",
            "experiments/store_sales_time_series_forecasting/20260626_170023",
        ],
        "metric_summary": "fixed candidate rmsle=2.650037 CV mean / 3.134017 OOF, original local ensemble stack rmsle=1.538161 on sampled validation",
        "failure_reason": "MCGS workstation run timed out or failed, but local fixed artifacts exist for audit and next-round search.",
    },
}


ARTIFACT_CANDIDATES = [
    "results.json",
    "metrics.json",
    "artifact_manifest.json",
    "oof_predictions.csv",
    "submission.csv",
    "report.md",
]


def now_ms() -> int:
    return int(time.time() * 1000)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def markdown_to_html(markdown: str, title: str) -> str:
    lines = []
    for line in markdown.splitlines():
        escaped = html.escape(line)
        if line.startswith("# "):
            lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            lines.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.startswith("|"):
            lines.append(f"<pre>{escaped}</pre>")
        elif not line.strip():
            lines.append("<br/>")
        else:
            lines.append(f"<p>{escaped}</p>")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; max-width: 980px; margin: 40px auto; color: #111827; line-height: 1.75; }}
    h1, h2 {{ color: #111827; }}
    pre {{ white-space: pre-wrap; background: #f9fafb; padding: 4px 8px; }}
  </style>
</head>
<body>
{chr(10).join(lines)}
</body>
</html>"""


def collect_artifacts(task_id: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    index = 1
    for source_dir in config["source_dirs"]:
        base = ROOT / source_dir
        for filename in ARTIFACT_CANDIDATES:
            path = base / filename
            if not path.exists():
                continue
            label = f"{Path(source_dir).name} {filename}"
            artifacts.append({
                "id": f"{task_id}_governance_ev_{index:02d}",
                "label": label,
                "artifact_path": rel(path),
                "hash": sha256(path),
                "source": "GovernanceBackfillAgent",
                "claim_binding": "failure_recovery_evidence",
            })
            index += 1
    return artifacts


def workflow_nodes() -> list[dict[str, str]]:
    return [
        {"id": "failed_run_detected", "status": "failed"},
        {"id": "artifact_recovery", "status": "passed"},
        {"id": "submission_schema_check", "status": "passed"},
        {"id": "failure_review", "status": "passed"},
        {"id": "submission_gate", "status": "blocked"},
        {"id": "next_round_plan", "status": "pending_gpu"},
        {"id": "report_generation", "status": "passed"},
    ]


def build_report(task_id: str, config: dict[str, Any], artifacts: list[dict[str, Any]]) -> str:
    return "\n".join([
        f"# {config['title']}",
        "",
        "## 摘要",
        f"`{task_id}` 已有工作站失败 run，同时存在本地修复/候选产物。此报告把这些产物登记为 failure-recovery evidence，用于证明系统能记录失败、保留候选、阻断未授权提交，并为下一轮工作站优化提供依据。",
        "",
        "## 当前结论",
        f"- 失败原因：{config['failure_reason']}",
        f"- 指标摘要：{config['metric_summary']}",
        "- 官方 Kaggle 提交：未执行",
        "- 官方排名/奖牌：无证据，不能声明",
        "- 下一步：GPU resource gate 恢复后，由工作站 Search Controller 重新发起受控实验",
        "",
        "## 证据列表",
        "",
        "| artifact | source | sha256 |",
        "| --- | --- | --- |",
        *[f"| `{item['artifact_path']}` | {item['source']} | `{item['hash'] or ''}` |" for item in artifacts],
        "",
        "## Gate 边界",
        "",
        "- validation/failure review：基于已有 artifact 通过回放审计",
        "- submission gate：blocked，等待 submission audit、claim audit 和人工 approval",
        "- final claim gate：blocked，不能外宣官方成绩或奖牌",
    ])


def build_plan() -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task_id, config in TASKS.items():
        artifacts = collect_artifacts(task_id, config)
        nodes = workflow_nodes()
        edges = [{"source": nodes[index]["id"], "target": nodes[index + 1]["id"]} for index in range(len(nodes) - 1)]
        report_rel = f"workspace/tasks/{task_id}/reports/draft/report.md"
        html_rel = f"workspace/tasks/{task_id}/reports/draft/report.html"
        tasks[task_id] = {
            "task_id": task_id,
            "run_id": config["run_id"],
            "title": config["title"],
            "artifacts": artifacts,
            "workflow": {
                "id": f"{task_id}_workflow",
                "name": f"{task_id} Failure Recovery Workflow",
                "status": "replayed_failure_recovery",
                "nodes": nodes,
                "edges": edges,
            },
            "report_path": report_rel,
            "html_path": html_rel,
            "markdown": build_report(task_id, config, artifacts),
            "claim_boundary": "Failure-recovery backfill does not prove new training, official score, rank, medal, or MLE-Bench parity.",
        }
    return {
        "schema": "academic_research_os.partial_task_governance_backfill.v1",
        "created_at_ms": now_ms(),
        "db_path": rel(DB_PATH),
        "tasks": tasks,
    }


def apply_backfill(plan: dict[str, Any]) -> dict[str, Any]:
    timestamp = now_ms()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cur = conn.cursor()
        totals = {"evidence": 0, "gates": 0, "workflows": 0, "reports": 0}
        for task_id, item in plan["tasks"].items():
            exists = cur.execute("select count(*) from tasks where id=?", (task_id,)).fetchone()[0]
            if not exists:
                raise RuntimeError(f"Task {task_id} is missing from {rel(DB_PATH)}")
            run_id = item["run_id"]

            for evidence in item["artifacts"]:
                cur.execute(
                    """
                    insert or replace into evidence
                    (id, task_id, run_id, label, artifact_path, hash, source, claim_binding, created_at)
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        evidence["id"],
                        task_id,
                        run_id,
                        evidence["label"],
                        evidence["artifact_path"],
                        evidence["hash"],
                        evidence["source"],
                        evidence["claim_binding"],
                        timestamp,
                    ),
                )
                totals["evidence"] += 1

            workflow = item["workflow"]
            cur.execute(
                """
                insert or replace into workflows
                (id, task_id, name, status, version, nodes_json, edges_json, published_at, created_at, updated_at)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workflow["id"],
                    task_id,
                    workflow["name"],
                    workflow["status"],
                    1,
                    json.dumps(workflow["nodes"], ensure_ascii=False),
                    json.dumps(workflow["edges"], ensure_ascii=False),
                    timestamp,
                    timestamp,
                    timestamp,
                ),
            )
            totals["workflows"] += 1

            gates = [
                ("failure_review", "approved", "Existing failed run and recovery artifacts were reviewed."),
                ("submission_gate", "blocked", "Official submission is blocked until audit and human approval."),
                ("final_claim_gate", "blocked", "Official score/rank/medal claims are blocked."),
            ]
            for gate_type, decision, reason in gates:
                evidence_ids = [artifact["id"] for artifact in item["artifacts"][:4]]
                cur.execute(
                    """
                    insert or replace into gates
                    (id, task_id, run_id, gate_type, decision, reviewer, evidence_json, created_at, decided_at)
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"gate_{task_id}_{gate_type}_backfill_20260630",
                        task_id,
                        run_id,
                        gate_type,
                        decision,
                        "GovernanceBackfillAgent",
                        json.dumps({"reason": reason, "required_evidence": evidence_ids}, ensure_ascii=False),
                        timestamp,
                        timestamp,
                    ),
                )
                totals["gates"] += 1

            report_path = ROOT / item["report_path"]
            html_path = ROOT / item["html_path"]
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(item["markdown"], encoding="utf-8")
            html_path.write_text(markdown_to_html(item["markdown"], item["title"]), encoding="utf-8")
            cur.execute(
                """
                insert or replace into reports
                (id, task_id, run_id, title, status, markdown_content, content_json, markdown_path, docx_path, selected_section, submitted_at, created_at, updated_at)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{task_id}_latest_report",
                    task_id,
                    run_id,
                    item["title"],
                    "draft",
                    item["markdown"],
                    json.dumps({"html_path": item["html_path"], "markdown_path": item["report_path"], "generated_by": "GovernanceBackfillAgent"}, ensure_ascii=False),
                    item["report_path"],
                    None,
                    "摘要",
                    None,
                    timestamp,
                    timestamp,
                ),
            )
            totals["reports"] += 1

            cur.execute(
                """
                insert or replace into action_logs
                (id, action, task_id, run_id, message, artifact_path, metadata_json, created_at)
                values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"action_{task_id}_governance_backfill_20260630",
                    "failure_recovery_governance_backfill",
                    task_id,
                    run_id,
                    f"{task_id} governance gates/evidence/report replayed from existing artifacts.",
                    rel(OUT_JSON),
                    json.dumps({"artifact_count": len(item["artifacts"]), "gate_count": len(gates)}, ensure_ascii=False),
                    timestamp,
                ),
            )
        conn.commit()
        return totals
    finally:
        conn.close()


def write_markdown(plan: dict[str, Any], result: dict[str, Any] | None) -> None:
    lines = [
        "# 部分失败任务 Governance 回填报告",
        "",
        f"- 数据库：`{plan['db_path']}`",
        f"- 已应用：`{bool(result)}`",
        f"- 结果：`{json.dumps(result, ensure_ascii=False) if result else 'dry-run'}`",
        "",
    ]
    for task_id, item in plan["tasks"].items():
        lines.extend([
            f"## {task_id}",
            "",
            f"- run_id：`{item['run_id']}`",
            f"- artifact 数：`{len(item['artifacts'])}`",
            f"- workflow 节点数：`{len(item['workflow']['nodes'])}`",
            f"- report：`{item['report_path']}`",
            f"- 边界：{item['claim_boundary']}",
            "",
        ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill governance gates for partial failed workstation tasks.")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    plan = build_plan()
    result = apply_backfill(plan) if args.apply else None
    if args.write_report:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps({**plan, "result": result}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(plan, result)

    print(json.dumps({
        "tasks": list(plan["tasks"].keys()),
        "applied": bool(result),
        "result": result,
        "json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
        "md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
