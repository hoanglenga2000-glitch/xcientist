#!/usr/bin/env python3
"""Build an official-metric MLE-Bench Lite progress report from grading summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--low-split", type=Path, required=True)
    parser.add_argument("--leaderboard-readme", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args(argv)


def main_leaderboard_lite_top(readme: Path) -> float:
    text = readme.read_text(encoding="utf-8")
    main = text.split("### Additional Leaderboard Submissions", 1)[0]
    values: list[float] = []
    for line in main.splitlines():
        if not line.startswith("|") or "±" not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        match = re.search(r"(\d+(?:\.\d+)?)\s*±", cells[2])
        if match:
            values.append(float(match.group(1)))
    if not values:
        raise RuntimeError("No main leaderboard Lite values found")
    return max(values)


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    grader = result.get("private_grader") or {}
    upstream = grader.get("upstream_report") or {}
    score = result.get("mle_private_grader_score")
    bronze = upstream.get("bronze_threshold")
    lower = bool(upstream.get("is_lower_better"))
    margin = None
    if score is not None and bronze is not None:
        margin = float(bronze) - float(score) if lower else float(score) - float(bronze)
    return {
        "competition_id": result["competition_id"],
        "status": result.get("status"),
        "metric": result.get("metric"),
        "direction": result.get("direction"),
        "cv_score": result.get("cv_score"),
        "mle_private_grader_score": score,
        "valid_submission": result.get("valid_submission"),
        "official_grader_executed": result.get("official_grader_executed"),
        "gold_threshold": upstream.get("gold_threshold"),
        "silver_threshold": upstream.get("silver_threshold"),
        "bronze_threshold": bronze,
        "median_threshold": upstream.get("median_threshold"),
        "any_medal": bool(upstream.get("any_medal")),
        "gold_medal": bool(upstream.get("gold_medal")),
        "silver_medal": bool(upstream.get("silver_medal")),
        "bronze_medal": bool(upstream.get("bronze_medal")),
        "above_median": bool(upstream.get("above_median")),
        "margin_to_bronze_positive_is_medal": margin,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    lite_ids = [line.strip() for line in args.low_split.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lite_ids) != 22 or len(set(lite_ids)) != 22:
        raise RuntimeError(f"Expected 22 unique Lite competitions, found {len(lite_ids)}")
    by_id: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for path in args.summary:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append({"path": path.as_posix(), "sha256": sha256(path), "run_id": payload.get("run_id")})
        for result in payload.get("results", []):
            competition_id = str(result["competition_id"])
            if competition_id not in lite_ids:
                raise RuntimeError(f"Non-Lite competition in summary: {competition_id}")
            by_id[competition_id] = result_row(result)

    rows = [by_id[item] for item in lite_ids if item in by_id]
    missing = [item for item in lite_ids if item not in by_id]
    total = len(lite_ids)
    medal_count = sum(row["any_medal"] for row in rows)
    gold_count = sum(row["gold_medal"] for row in rows)
    above_median = sum(row["above_median"] for row in rows)
    top = main_leaderboard_lite_top(args.leaderboard_readme)
    required = math.floor(top * total / 100.0) + 1
    maximum_if_all_missing_medal = medal_count + len(missing)
    scored_non_medals = len(rows) - medal_count
    minimum_existing_non_medals_to_convert = max(0, required - maximum_if_all_missing_medal)
    report = {
        "schema": "evomind.mlebench_lite.progress.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_metric": "any_medal_percentage",
        "lite_total_competitions": total,
        "scored_competitions": len(rows),
        "remaining_competitions": len(missing),
        "valid_official_grades": sum(
            bool(row["valid_submission"] and row["official_grader_executed"]) for row in rows
        ),
        "any_medal_count": medal_count,
        "gold_medal_count": gold_count,
        "above_median_count": above_median,
        "scored_subset_any_medal_percentage": medal_count / len(rows) * 100 if rows else 0.0,
        "padded_full_lite_any_medal_percentage": medal_count / total * 100,
        "main_leaderboard_lite_top_percentage": top,
        "medals_required_to_strictly_exceed_top": required,
        "medal_deficit": required - medal_count,
        "maximum_medals_if_all_unscored_tasks_medal": maximum_if_all_missing_medal,
        "maximum_percentage_if_all_unscored_tasks_medal": maximum_if_all_missing_medal / total * 100,
        "minimum_scored_non_medals_that_must_also_be_converted": minimum_existing_non_medals_to_convert,
        "scored_non_medal_count": scored_non_medals,
        "remaining_competition_ids": missing,
        "results": rows,
        "sources": sources,
        "leaderboard_source": {
            "path": args.leaderboard_readme.as_posix(),
            "sha256": sha256(args.leaderboard_readme),
        },
        "low_split_source": {"path": args.low_split.as_posix(), "sha256": sha256(args.low_split)},
        "claim_boundary": (
            "This is deterministic official MLE-Bench private grading progress for one seed. "
            "Leaderboard-comparable completion requires all 22 tasks and at least three seeds; "
            "no Kaggle submission was made."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# EvoMind MLE-Bench Lite Progress",
        "",
        f"- Official graded coverage: **{len(rows)}/22**",
        f"- Any-medal count: **{medal_count}/22**",
        f"- Current padded Any Medal: **{report['padded_full_lite_any_medal_percentage']:.2f}%**",
        f"- Main leaderboard top reference: **{top:.2f}%**",
        f"- Strictly exceed target: **{required}/22 medals ({required / total * 100:.2f}%)**",
        f"- Even if every unscored task medals, at least **{minimum_existing_non_medals_to_convert}** scored non-medals must be converted.",
        "",
        "| Competition | Score | Bronze threshold | Medal | Above median | Margin to bronze |",
        "|---|---:|---:|:---:|:---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['competition_id']} | {fmt(row['mle_private_grader_score'])} | "
            f"{fmt(row['bronze_threshold'])} | {'YES' if row['any_medal'] else 'NO'} | "
            f"{'YES' if row['above_median'] else 'NO'} | {fmt(row['margin_to_bronze_positive_is_medal'])} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"results", "sources"}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
