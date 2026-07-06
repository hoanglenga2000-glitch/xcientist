from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "workspace" / "workstation_training_progress_20260630.json"
OUT_MD = ROOT / "reports" / "WORKSTATION_TRAINING_PROGRESS_20260630.md"

INVENTORY_PATH = ROOT / "workspace" / "kaggle_experiment_inventory_20260624.json"
LEADERBOARD_PATH = ROOT / "workspace" / "mlebench_style_current_leaderboard_20260625.json"
TASK_MATRIX_PATH = ROOT / "workspace" / "workstation_task_api_matrix_20260630.json"
READINESS_PATH = ROOT / "workspace" / "workstation_launch_readiness_20260630.json"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {"missing": True, "path": str(path)}
    except Exception as exc:
        return {"error": str(exc), "path": str(path)}


def as_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def matrix_by_task(task_matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("task_id")): item
        for item in task_matrix.get("tasks", [])
        if item.get("task_id")
    }


def leaderboard_by_task(leaderboard: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("task_id")): item
        for item in leaderboard.get("leaderboard_rows", [])
        if item.get("task_id")
    }


def classify_task(task: dict[str, Any], matrix_item: dict[str, Any] | None, lb_item: dict[str, Any] | None) -> dict[str, Any]:
    loop = (matrix_item or {}).get("closed_loop_evidence") or {}
    official_score_known = bool((lb_item or {}).get("official_score_known"))
    official_rank = (lb_item or {}).get("official_rank")
    top30_reached = (lb_item or {}).get("top30_reached")
    medal = (lb_item or {}).get("medal")

    if loop.get("full_reportable_loop_visible"):
        evidence_status = "full_reportable"
    elif loop.get("minimum_closed_loop_visible"):
        evidence_status = "minimum_loop_visible"
    else:
        evidence_status = "needs_evidence_backfill"

    if official_score_known and top30_reached is True:
        official_status = "official_top30_reached"
    elif official_score_known:
        official_status = "official_submitted_not_top30"
    else:
        official_status = "proxy_only"

    return {
        "task_id": task.get("task_id"),
        "metric": task.get("metric"),
        "run_count": task.get("run_count", 0),
        "scored_runs": task.get("scored_runs", 0),
        "promoted_runs": task.get("promoted_runs", 0),
        "held_runs": task.get("held_runs", 0),
        "timeout_or_failed_runs": task.get("timeout_or_failed_runs", 0),
        "best_score": task.get("best_score"),
        "best_run": task.get("best_run"),
        "improvement": task.get("improvement"),
        "agent_count_observed": task.get("agent_count_observed", 0),
        "agents_observed": task.get("agents_observed", []),
        "evidence_status": evidence_status,
        "official_status": official_status,
        "official_public_score": (lb_item or {}).get("official_public_score"),
        "official_rank": official_rank,
        "leaderboard_team_count": (lb_item or {}).get("leaderboard_team_count"),
        "rank_percentile": (lb_item or {}).get("rank_percentile"),
        "top30_reached": top30_reached,
        "medal": medal or "unknown",
        "medal_evidence": (lb_item or {}).get("medal_evidence"),
        "needs_report": not bool(loop.get("has_report")),
        "needs_evidence": not bool(loop.get("has_evidence")),
    }


def build_priority_list(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priorities: list[dict[str, Any]] = []
    for item in tasks:
        reasons: list[str] = []
        score = 0
        if item["official_status"] == "official_submitted_not_top30":
            score += 100
            reasons.append("已有官方提交但未达 top30，适合下一轮自进化校准")
        if item["evidence_status"] == "minimum_loop_visible":
            score += 40
            reasons.append("已有最小闭环，但缺完整报告或证据")
        if item["evidence_status"] == "full_reportable":
            score += 30
            reasons.append("证据链较完整，可作为稳定样板任务")
        if as_number(item.get("promoted_runs")) > 0:
            score += 20
            reasons.append("存在 promoted run，可做 best-so-far 保护与融合")
        if as_number(item.get("held_runs")) > 0:
            score += 10
            reasons.append("存在 held run，可沉淀失败记忆")
        if item.get("needs_report"):
            score += 8
            reasons.append("需要补齐 report artifact")
        if item.get("needs_evidence"):
            score += 5
            reasons.append("需要补齐 evidence artifact")
        if score:
            priorities.append({
                "task_id": item["task_id"],
                "priority_score": score,
                "official_status": item["official_status"],
                "evidence_status": item["evidence_status"],
                "reasons": reasons,
            })
    return sorted(priorities, key=lambda row: (-row["priority_score"], row["task_id"]))[:10]


def build_report() -> dict[str, Any]:
    inventory = read_json(INVENTORY_PATH)
    leaderboard = read_json(LEADERBOARD_PATH)
    task_matrix = read_json(TASK_MATRIX_PATH)
    readiness = read_json(READINESS_PATH)

    matrix_items = matrix_by_task(task_matrix)
    leaderboard_items = leaderboard_by_task(leaderboard)
    task_rows = [
        classify_task(task, matrix_items.get(str(task.get("task_id"))), leaderboard_items.get(str(task.get("task_id"))))
        for task in inventory.get("task_summary", [])
        if task.get("task_id")
    ]

    lb_summary = leaderboard.get("summary") or {}
    official_tasks = [item for item in task_rows if item["official_status"] != "proxy_only"]
    full_reportable = [item for item in task_rows if item["evidence_status"] == "full_reportable"]
    minimum_loop = [item for item in task_rows if item["evidence_status"] in {"minimum_loop_visible", "full_reportable"}]
    top30_tasks = [item for item in task_rows if item.get("top30_reached") is True]
    medal_tasks = [item for item in task_rows if item.get("medal") not in {None, "unknown", "none"}]

    return {
        "schema": "academic_research_os.workstation_training_progress.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "inventory": str(INVENTORY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "leaderboard": str(LEADERBOARD_PATH.relative_to(ROOT)).replace("\\", "/"),
            "task_api_matrix": str(TASK_MATRIX_PATH.relative_to(ROOT)).replace("\\", "/"),
            "readiness": str(READINESS_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
        "status": "passed" if task_rows else "failed",
        "summary": {
            "tasks_with_experiments": inventory.get("task_count_with_experiments"),
            "observed_runs": inventory.get("total_runs_observed"),
            "scored_runs": inventory.get("total_scored_runs"),
            "promoted_runs": inventory.get("total_promoted_runs"),
            "held_runs": inventory.get("total_held_runs"),
            "timeout_or_failed_runs": inventory.get("total_timeout_or_failed_runs"),
            "kaggle10_completion_status": inventory.get("kaggle10_completion_status"),
            "task_api_total": task_matrix.get("task_count"),
            "minimum_closed_loop_visible_count": len(minimum_loop),
            "full_reportable_loop_visible_count": len(full_reportable),
            "official_submission_tasks": len(official_tasks),
            "official_top30_tasks": len(top30_tasks),
            "official_top30_rate_among_rank_known": lb_summary.get("official_top30_rate_among_rank_known_official_submissions"),
            "official_top30_rate_among_all_observed": lb_summary.get("official_top30_rate_among_all_observed_tasks"),
            "medal_count": len(medal_tasks),
            "medal_rate": lb_summary.get("medal_rate"),
            "benchmark_claim_status": lb_summary.get("benchmark_claim_status"),
            "launch_state": readiness.get("launch_state"),
            "readiness_blockers": readiness.get("blockers", []),
        },
        "tasks": task_rows,
        "priority_remediation": build_priority_list(task_rows),
        "claim_boundary": (
            "本报告统计已有工作站 artifact 和官方 Kaggle response。没有官方 response 的任务只能视为 proxy/local 结果；"
            "当前不能宣称 MLE-Bench 75 达标，也不能把 CV/proxy 分数当作奖牌。"
        ),
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# AI 科研工作站训练进度统计",
        "",
        f"- 生成时间：`{report['created_at']}`",
        f"- 总状态：`{report['status']}`",
        f"- 上线状态：`{summary.get('launch_state')}`",
        f"- 阻断项：`{', '.join(summary.get('readiness_blockers') or []) or 'none'}`",
        "",
        "## 核心统计",
        "",
        f"- 已有实验任务：`{summary.get('tasks_with_experiments')}`",
        f"- 观测 run：`{summary.get('observed_runs')}`",
        f"- 有分数 run：`{summary.get('scored_runs')}`",
        f"- promoted / held / timeout-or-failed：`{summary.get('promoted_runs')}` / `{summary.get('held_runs')}` / `{summary.get('timeout_or_failed_runs')}`",
        f"- 工作站任务 API 总数：`{summary.get('task_api_total')}`",
        f"- 最小闭环可见任务：`{summary.get('minimum_closed_loop_visible_count')}`",
        f"- 完整可汇报闭环任务：`{summary.get('full_reportable_loop_visible_count')}`",
        f"- 官方提交任务：`{summary.get('official_submission_tasks')}`",
        f"- 官方 top30 任务：`{summary.get('official_top30_tasks')}`",
        f"- medal count / medal rate：`{summary.get('medal_count')}` / `{summary.get('medal_rate')}`",
        f"- benchmark claim：`{summary.get('benchmark_claim_status')}`",
        "",
        "## 任务明细",
        "",
        "| task | metric | runs | scored | promoted | held | best score | evidence | official | rank % | medal |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for task in report["tasks"]:
        lines.append(
            "| `{task}` | `{metric}` | {runs} | {scored} | {promoted} | {held} | {best} | `{evidence}` | `{official}` | {rank} | `{medal}` |".format(
                task=task["task_id"],
                metric=task.get("metric"),
                runs=task.get("run_count"),
                scored=task.get("scored_runs"),
                promoted=task.get("promoted_runs"),
                held=task.get("held_runs"),
                best=fmt(task.get("best_score")),
                evidence=task.get("evidence_status"),
                official=task.get("official_status"),
                rank=fmt(task.get("rank_percentile")),
                medal=task.get("medal"),
            )
        )

    lines.extend([
        "",
        "## 下一轮优先修复/提升队列",
        "",
        "| priority | task | official | evidence | reasons |",
        "| ---: | --- | --- | --- | --- |",
    ])
    for row in report["priority_remediation"]:
        lines.append(
            f"| {row['priority_score']} | `{row['task_id']}` | `{row['official_status']}` | `{row['evidence_status']}` | {'；'.join(row['reasons'])} |"
        )

    lines.extend([
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
        "## 建议动作",
        "",
        "1. 先补齐 `minimum_loop_visible` 任务的 report/evidence，把更多任务升级为完整可汇报闭环。",
        "2. 对已有官方提交但未达 top30 的任务，走工作站 Search Controller 第二轮自进化，不手工旁路训练。",
        "3. GPU/HPC gate 恢复前，不启动大规模训练；恢复后仍必须通过工作站 resource mode。",
        "4. 奖牌率展示只能绑定官方 response 和 medal evidence；当前 medal rate 仍按真实证据统计。",
    ])

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a consolidated training progress report for the workstation.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    report = build_report()
    if args.write_report:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(report)

    print(json.dumps({
        "status": report["status"],
        "tasks_with_experiments": report["summary"].get("tasks_with_experiments"),
        "observed_runs": report["summary"].get("observed_runs"),
        "official_submission_tasks": report["summary"].get("official_submission_tasks"),
        "official_top30_tasks": report["summary"].get("official_top30_tasks"),
        "medal_count": report["summary"].get("medal_count"),
        "benchmark_claim_status": report["summary"].get("benchmark_claim_status"),
        "json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
        "md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
