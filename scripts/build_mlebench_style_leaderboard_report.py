from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
INVENTORY = ROOT / "workspace" / "kaggle_experiment_inventory_20260624.json"
ALIGNMENT = ROOT / "workspace" / "mlevolve_alignment_matrix_20260625.json"
TOP30_ORDERS = ROOT / "workspace" / "top30_next_evolution_orders_20260625.json"
OUT_JSON = ROOT / "workspace" / "mlebench_style_current_leaderboard_20260625.json"
OUT_MD = ROOT / "reports" / "MLEBENCH_STYLE_CURRENT_LEADERBOARD_20260625.md"

MLEVOLVE_MEDAL_RATE = 0.653
TARGET_TASKS = 75


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def status_for_task(task: dict[str, Any], official: dict[str, Any] | None) -> str:
    if official and official.get("top30_reached") is True:
        return "top30_reached"
    if official and official.get("rank_unknown") is False:
        return "top30_failed"
    if official and official.get("official_score_known"):
        return "official-known-rank-unknown"
    if task.get("best_score") is not None:
        return "proxy_only"
    return "not_evaluated"


def metric_direction(metric: Any) -> str:
    if str(metric).lower() in {"rmsle", "rmse", "mae", "mse", "log_loss"}:
        return "minimize"
    return "maximize"


def with_latest_rank(item: dict[str, Any]) -> dict[str, Any]:
    snapshot = item.get("latest_leaderboard_snapshot")
    if not isinstance(snapshot, dict):
        return item
    rank = snapshot.get("rank")
    teams = snapshot.get("leaderboard_team_count")
    percentile = snapshot.get("rank_percentile")
    if not isinstance(rank, int) or not isinstance(teams, int):
        return item
    updated = dict(item)
    updated["rank"] = rank
    updated["leaderboard_team_count"] = teams
    updated["rank_percentile"] = percentile if isinstance(percentile, (int, float)) else rank / teams
    updated["rank_unknown"] = False
    updated["top30_reached"] = updated["rank_percentile"] <= 0.30
    return updated


def official_sort_key(item: dict[str, Any]) -> tuple[int, float, int, int]:
    score = item.get("public_score")
    score_value = float(score) if isinstance(score, (int, float)) else float("-inf")
    if metric_direction(item.get("metric")) == "minimize":
        score_value = -score_value
    top30 = 1 if item.get("top30_reached") is True else 0
    rank_known = 1 if item.get("rank_unknown") is False else 0
    evidence_priority = 2 if item.get("evidence_source") in {"kaggle_submission_response", "kaggle_official_submission"} else 1
    return (score_value != float("-inf"), score_value, top30 + rank_known, evidence_priority)


def choose_official_by_task(records: list[dict[str, Any]], task_metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for item in records:
        task_id = item.get("task_id")
        if not task_id:
            continue
        enriched = with_latest_rank({**item, "metric": task_metrics.get(task_id)})
        current = selected.get(task_id)
        if current is None or official_sort_key(enriched) > official_sort_key(current):
            selected[task_id] = enriched
    return selected


def main() -> None:
    inventory = read_json(INVENTORY)
    alignment = read_json(ALIGNMENT)
    orders = read_json(TOP30_ORDERS)
    task_metrics = {
        item.get("task_id"): item.get("metric")
        for item in inventory.get("task_summary", [])
        if isinstance(item, dict) and item.get("task_id")
    }
    official_by_task = choose_official_by_task(
        [item for item in inventory.get("official_submission_records", []) if isinstance(item, dict)],
        task_metrics,
    )

    rows = []
    for task in inventory.get("task_summary", []):
        task_id = task.get("task_id")
        official = official_by_task.get(task_id)
        rows.append(
            {
                "task_id": task_id,
                "status": status_for_task(task, official),
                "metric": task.get("metric"),
                "best_local_or_proxy_score": task.get("best_score"),
                "best_run": task.get("best_run"),
                "official_public_score": official.get("public_score") if official else None,
                "official_score_known": bool(official.get("official_score_known")) if official else False,
                "official_rank": official.get("rank") if official else None,
                "leaderboard_team_count": official.get("leaderboard_team_count") if official else None,
                "rank_percentile": official.get("rank_percentile") if official else None,
                "rank_unknown": bool(official.get("rank_unknown")) if official else False,
                "top30_reached": official.get("top30_reached") if official else None,
                "official_evidence_source": official.get("evidence_source") if official else None,
                "medal": "unknown",
                "medal_evidence": "missing_private_or_medal_threshold",
                "run_count": task.get("run_count"),
                "promoted_runs": task.get("promoted_runs"),
                "held_runs": task.get("held_runs"),
                "agent_count_observed": task.get("agent_count_observed"),
            }
        )

    official_score_known_count = len([row for row in rows if row["official_score_known"]])
    official_rank_known_count = len([row for row in rows if row["official_score_known"] and not row["rank_unknown"]])
    official_rank_unknown_count = len([row for row in rows if row["official_score_known"] and row["rank_unknown"]])
    top30_count = len([row for row in rows if row["top30_reached"] is True])
    medal_count = len([row for row in rows if row["medal"] in {"bronze", "silver", "gold"}])
    medal_rate = medal_count / len(rows) if rows else 0.0
    payload = {
        "schema": "academic_research_os.mlebench_style_current_leaderboard.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_inventory": INVENTORY.relative_to(ROOT).as_posix(),
        "source_alignment": ALIGNMENT.relative_to(ROOT).as_posix() if ALIGNMENT.exists() else None,
        "source_top30_orders": TOP30_ORDERS.relative_to(ROOT).as_posix() if TOP30_ORDERS.exists() else None,
        "target_reference": {
            "benchmark": "MLE-Bench 75 Kaggle tasks",
            "mlevolve_medal_rate_reference": MLEVOLVE_MEDAL_RATE,
            "target_tasks": TARGET_TASKS,
            "comparable": False,
        },
        "summary": {
            "tasks_with_experiments": len(rows),
            "official_submission_tasks": official_score_known_count,
            "official_score_known_tasks": official_score_known_count,
            "official_rank_known_tasks": official_rank_known_count,
            "official_rank_unknown_tasks": official_rank_unknown_count,
            "official_top30_count": top30_count,
            "official_top30_rate_among_rank_known_official_submissions": top30_count / official_rank_known_count if official_rank_known_count else 0.0,
            "official_top30_rate_among_all_observed_tasks": top30_count / len(rows) if rows else 0.0,
            "medal_count": medal_count,
            "medal_rate": medal_rate,
            "gap_to_mlevolve_medal_rate": MLEVOLVE_MEDAL_RATE - medal_rate,
            "benchmark_claim_status": "not_comparable_not_reached",
        },
        "leaderboard_rows": rows,
        "mlevolve_policy_snapshot": (alignment.get("mlevolve_reference_policy") or {}) if isinstance(alignment, dict) else {},
        "next_orders_snapshot": orders.get("orders", []) if isinstance(orders, dict) else [],
        "claim_boundary": "Preliminary workstation evidence only. Do not claim MLEvolve parity, top30, rank, or medal unless official artifacts prove it.",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = payload["summary"]
    lines = [
        "# MLE-Bench Style Current Leaderboard",
        "",
        f"- Created at: `{payload['created_at']}`",
        "- Benchmark target: `MLE-Bench 75 Kaggle tasks`",
        f"- MLEvolve reference medal rate: `{MLEVOLVE_MEDAL_RATE:.3f}`",
        f"- Comparable to MLEvolve: `{payload['target_reference']['comparable']}`",
        f"- Claim boundary: {payload['claim_boundary']}",
        "",
        "## Summary",
        "",
        f"- tasks_with_experiments: `{summary['tasks_with_experiments']}`",
        f"- official_submission_tasks: `{summary['official_submission_tasks']}`",
        f"- official_score_known_tasks: `{summary['official_score_known_tasks']}`",
        f"- official_rank_known_tasks: `{summary['official_rank_known_tasks']}`",
        f"- official_rank_unknown_tasks: `{summary['official_rank_unknown_tasks']}`",
        f"- official_top30_count: `{summary['official_top30_count']}`",
        f"- official_top30_rate_among_rank_known_official_submissions: `{summary['official_top30_rate_among_rank_known_official_submissions']:.4f}`",
        f"- medal_rate: `{summary['medal_rate']:.4f}`",
        f"- gap_to_mlevolve_medal_rate: `{summary['gap_to_mlevolve_medal_rate']:.4f}`",
        f"- benchmark_claim_status: `{summary['benchmark_claim_status']}`",
        "",
        "## Current Ranking Table",
        "",
        "| task | status | metric | best local/proxy | official score | rank | percentile | top30 | medal |",
        "|---|---|---|---:|---:|---|---:|---|---|",
    ]
    for row in rows:
        rank = f"{row['official_rank']}/{row['leaderboard_team_count']}" if row["official_rank"] else ("rank_unknown" if row["official_score_known"] else "n/a")
        top30 = row["top30_reached"] if row["top30_reached"] is not None else ("rank_unknown" if row["official_score_known"] else "n/a")
        lines.append(
            f"| `{row['task_id']}` | `{row['status']}` | `{row['metric']}` | {row['best_local_or_proxy_score']} | {row['official_public_score']} | {rank} | {row['rank_percentile']} | `{top30}` | `{row['medal']}` |"
        )
    lines.extend(
        [
            "",
            "## Gap Analysis",
            "",
            f"- 当前有 `{summary['official_score_known_tasks']}` 个任务有官方 public score 证据，其中 `{summary['official_rank_unknown_tasks']}` 个只有 public score、rank 未知。",
            "- 当前没有 private leaderboard 或 medal threshold artifact，因此 medal_rate 必须记为 0。",
            "- 当前不能对外宣称达到 MLEvolve 75-task 水平；只能说系统正在按 MLEvolve-style 搜索控制与 XCIENTIST-style 审计框架对齐。",
            "",
            "## Next Workstation Action",
            "",
            "- P0: 由工作站继续优化 `spaceship_titanic`，目标从 36.8% 推进到前 30%。",
            "- P1: 将 `house_prices` 与 `titanic` 从 proxy-only 转成有官方 rank artifact 的校准任务。",
            "- P2: 继续扩展任务覆盖率，先提高 valid submission rate，再追求 medal/top30 rate。",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.relative_to(ROOT).as_posix(), "md": OUT_MD.relative_to(ROOT).as_posix(), "tasks": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
