#!/usr/bin/env python3
"""Build a truthful Lite-22 training and grading coverage report.

The report intentionally keeps three states separate: a completed training
attempt, a valid submission artifact, and an executed MLE-Bench private grade.
It never treats offline grading or medal-threshold equivalence as a Kaggle
online submission or leaderboard rank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def medal_label(row: dict[str, Any]) -> str:
    if row.get("gold_medal"):
        return "Gold-equivalent"
    if row.get("silver_medal"):
        return "Silver-equivalent"
    if row.get("bronze_medal"):
        return "Bronze-equivalent"
    return "No medal-equivalent"


def build(progress: dict[str, Any], dog_summary: dict[str, Any]) -> dict[str, Any]:
    total = int(progress.get("lite_total_competitions") or 0)
    if total != 22:
        raise ValueError(f"Expected Lite total=22, got {total}")
    results = list(progress.get("results") or [])
    scored_ids = {str(row.get("competition_id")) for row in results}
    if len(scored_ids) != 21 or len(results) != 21:
        raise ValueError("The official progress input must contain 21 unique graded tasks")
    if "dog-breed-identification" in scored_ids:
        raise ValueError("Dog Breed is already graded in the official progress input")

    dog_results = list(dog_summary.get("results") or [])
    if len(dog_results) != 1:
        raise ValueError("Dog summary must contain exactly one result")
    dog = dog_results[0]
    if dog.get("competition_id") != "dog-breed-identification":
        raise ValueError("Unexpected Dog summary competition")
    if dog.get("valid_submission") is not True:
        raise ValueError("Dog Breed does not have a valid submission artifact")
    gate = dog.get("promotion_gate") or {}
    if not isinstance(gate, dict):
        raise ValueError("Dog Breed promotion gate is missing")

    rows: list[dict[str, Any]] = []
    for row in results:
        rows.append(
            {
                "competition_id": row["competition_id"],
                "training_attempt_completed": True,
                "valid_submission": bool(row.get("valid_submission")),
                "private_grader_executed": bool(row.get("official_grader_executed")),
                "metric": row.get("metric"),
                "cv_score": row.get("cv_score"),
                "mle_private_grader_score": row.get("mle_private_grader_score"),
                "medal_threshold_equivalence": medal_label(row),
                "status": row.get("status"),
            }
        )
    rows.append(
        {
            "competition_id": dog["competition_id"],
            "training_attempt_completed": True,
            "valid_submission": True,
            "private_grader_executed": False,
            "metric": dog.get("metric"),
            "cv_score": dog.get("cv_score"),
            "mle_private_grader_score": None,
            "medal_threshold_equivalence": "Not evaluated",
            "status": dog.get("status"),
        }
    )
    rows.sort(key=lambda item: item["competition_id"])

    dec = next(row for row in results if row["competition_id"] == "tabular-playground-series-dec-2021")
    training_completed = sum(bool(row["training_attempt_completed"]) for row in rows)
    valid_submissions = sum(bool(row["valid_submission"]) for row in rows)
    private_grades = sum(bool(row["private_grader_executed"]) for row in rows)
    return {
        "schema": "evomind.mlebench_lite.training_coverage.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lite_total_competitions": total,
        "training_attempts_completed": training_completed,
        "valid_submission_artifacts": valid_submissions,
        "private_grades_completed": private_grades,
        "all_competitions_trained": training_completed == total,
        "all_competitions_have_valid_submission": valid_submissions == total,
        "all_competitions_private_graded": private_grades == total,
        "medal_threshold_equivalence": {
            "any_medal_count": int(progress.get("any_medal_count") or 0),
            "gold_count": int(progress.get("gold_medal_count") or 0),
            "full_lite_percentage": float(progress.get("padded_full_lite_any_medal_percentage") or 0.0),
        },
        "demo_highlight": {
            "competition_id": dec["competition_id"],
            "metric": dec["metric"],
            "cv_score": dec["cv_score"],
            "mle_private_grader_score": dec["mle_private_grader_score"],
            "gold_threshold": dec["gold_threshold"],
            "medal_threshold_equivalence": medal_label(dec),
            "valid_submission": dec["valid_submission"],
            "official_grader_executed": dec["official_grader_executed"],
        },
        "remaining_grade_blocker": {
            "competition_id": dog["competition_id"],
            "status": dog.get("status"),
            "metric": dog.get("metric"),
            "cv_score": dog.get("cv_score"),
            "promotion_operator": gate.get("operator"),
            "promotion_threshold": gate.get("threshold"),
            "promotion_passed": gate.get("passed"),
            "valid_submission": dog.get("valid_submission"),
            "private_grader_withheld": dog.get("official_grader_withheld"),
            "reason": (dog.get("private_grader") or {}).get("reason"),
        },
        "results": rows,
        "claim_boundary": (
            "22/22 means every Lite competition has a completed single-seed training attempt and a valid "
            "submission artifact. Only 21/22 have an executed deterministic MLE-Bench private grade. "
            "Medal labels are offline threshold equivalence, not Kaggle online submissions, ranks, or medals."
        ),
        "kaggle_submission_executed": False,
        "human_gate_preserved": True,
    }


def render_markdown(report: dict[str, Any]) -> str:
    coverage = report["training_attempts_completed"]
    total = report["lite_total_competitions"]
    grades = report["private_grades_completed"]
    medals = report["medal_threshold_equivalence"]
    dec = report["demo_highlight"]
    dog = report["remaining_grade_blocker"]
    lines = [
        "# EvoMind MLE-Bench Lite 22 训练覆盖与评分总览",
        "",
        f"- 训练尝试完成：**{coverage}/{total}**",
        f"- 合法 submission 产物：**{report['valid_submission_artifacts']}/{total}**",
        f"- MLE-Bench private grader 完成：**{grades}/{total}**",
        f"- 奖牌阈值等价：**{medals['any_medal_count']}/{total}**（其中 Gold-equivalent {medals['gold_count']}）",
        "- Kaggle 官方提交：**未执行（Human Gate 保持）**",
        "",
        "## 演示任务：Tabular Playground Dec 2021",
        "",
        f"- 指标：{dec['metric']}",
        f"- 交叉验证：{dec['cv_score']:.6f}",
        f"- MLE-Bench private grader：{dec['mle_private_grader_score']:.6f}",
        f"- Gold 阈值：{dec['gold_threshold']:.6f}",
        f"- 结果：**{dec['medal_threshold_equivalence']}**",
        "",
        "## 尚未完成 private grader 的任务",
        "",
        f"- {dog['competition_id']} 已完成训练并生成合法 submission。",
        f"- OOF {dog['metric']}：{dog['cv_score']:.6f}",
        f"- 严格晋级门槛：{dog['promotion_operator']} {dog['promotion_threshold']:.6f}",
        "- 结论：晋级门禁未过，private grader 按设计暂扣。",
        "",
        "## 22 项明细",
        "",
        "| 比赛 | 训练 | Submission | Private grader | 得分 | 阈值等价 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["results"]:
        score = row["mle_private_grader_score"]
        score_text = "—" if score is None else f"{float(score):.6f}"
        lines.append(
            f"| {row['competition_id']} | ✓ | {'✓' if row['valid_submission'] else '✗'} | "
            f"{'✓' if row['private_grader_executed'] else '—'} | {score_text} | "
            f"{row['medal_threshold_equivalence']} |"
        )
    lines.extend(["", "## 口径边界", "", report["claim_boundary"], ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--dog-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    progress = load_json(args.progress)
    dog_summary = load_json(args.dog_summary)
    report = build(progress, dog_summary)
    report["sources"] = {
        "official_progress": {"path": str(args.progress), "sha256": sha256(args.progress)},
        "dog_summary": {"path": str(args.dog_summary), "sha256": sha256(args.dog_summary)},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output_json": str(args.output_json.resolve()),
        "output_md": str(args.output_md.resolve()),
        "all_competitions_trained": report["all_competitions_trained"],
        "private_grades_completed": report["private_grades_completed"],
        "lite_total_competitions": report["lite_total_competitions"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
