from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "workspace" / "kaggle_submissions"
REPORT_DIR = ROOT / "reports"
OFFICIAL_BEST = {
    "experiment_id": "EXP007",
    "public_score": 0.96659,
    "submission_ref": "53680150",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_rank(rank: str) -> tuple[int | None, int | None]:
    if not rank:
        return None, None
    cleaned = rank.replace(" ", "")
    if "/" not in cleaned:
        return None, None
    left, right = cleaned.split("/", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None, None


def compute_rank_percentile(rank_position: int | None, rank_total: int | None) -> float | None:
    if not isinstance(rank_position, int) or not isinstance(rank_total, int) or rank_total <= 0:
        return None
    return rank_position / rank_total


def build_report(record: dict[str, Any]) -> str:
    delta = record["score_delta_vs_previous_best"]
    rank_text = record.get("public_rank") or "not recorded"
    verdict = "beats previous official best" if delta > 0 else "does not beat previous official best"
    return "\n".join(
        [
            f"# Submission Score Record: {record['experiment_id']}",
            "",
            f"Recorded: `{record['recorded_at']}`",
            "",
            "## Official Result",
            "",
            f"- Experiment: `{record['experiment_id']}`",
            f"- Submission ref: `{record['submission_ref']}`",
            f"- Public score: `{record['public_score']}`",
            f"- Public rank: `{rank_text}`",
            f"- Rank percentile: `{record.get('rank_percentile')}`",
            f"- Top 30% reached: `{record.get('top30_reached')}`",
            f"- Previous official best: `{record['previous_best']['experiment_id']}` public `{record['previous_best']['public_score']}`, ref `{record['previous_best']['submission_ref']}`",
            f"- Score delta vs previous best: `{delta:+.8f}`",
            f"- Verdict: `{verdict}`",
            "",
            "## Governance Note",
            "",
            "This file records the public-score feedback only. It does not change the model-selection rule: local CV, stability, error analysis, and leakage controls remain the source of truth for deciding the next experiment.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a manually completed Kaggle public score into the audit trail.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--submission-ref", required=True)
    parser.add_argument("--public-score", type=float, required=True)
    parser.add_argument("--public-rank", default="", help="Optional rank like 527/1600.")
    parser.add_argument("--submission-file", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    rank_position, rank_total = parse_rank(args.public_rank)
    rank_percentile = compute_rank_percentile(rank_position, rank_total)
    delta = args.public_score - OFFICIAL_BEST["public_score"]
    record = {
        "experiment_id": args.experiment_id,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "submission_ref": args.submission_ref,
        "public_score": args.public_score,
        "public_rank": args.public_rank,
        "public_rank_position": rank_position,
        "public_rank_total": rank_total,
        "rank_percentile": rank_percentile,
        "top30_reached": bool(rank_percentile is not None and rank_percentile <= 0.30),
        "submission_file": args.submission_file,
        "previous_best": OFFICIAL_BEST,
        "score_delta_vs_previous_best": delta,
        "beats_previous_best": delta > 0,
        "notes": args.notes,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    slug = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.experiment_id.lower()}_score_record"
    json_path = OUT_DIR / f"{slug}.json"
    report_path = REPORT_DIR / f"SUBMISSION_SCORE_{args.experiment_id}.md"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_report(record), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "recorded",
                "json_path": rel(json_path),
                "report_path": rel(report_path),
                "score_delta_vs_previous_best": delta,
                "beats_previous_best": delta > 0,
                "top30_reached": record["top30_reached"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
