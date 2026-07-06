from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
WORKSPACE_DIR = ROOT / "workspace"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_direction(metrics: dict[str, Any]) -> str:
    return str(metrics.get("ensemble", {}).get("metric_direction") or metrics.get("metric_direction") or "maximize")


def is_better(candidate: float | None, current: float | None, direction: str) -> bool:
    if not isinstance(candidate, (int, float)):
        return False
    if not isinstance(current, (int, float)):
        return True
    return candidate < current if direction in {"minimize", "lower", "lower_is_better"} else candidate > current


def run_record(result: dict[str, Any]) -> dict[str, Any]:
    run_dir_raw = result.get("run_dir")
    run_dir = ROOT / run_dir_raw if run_dir_raw else None
    metrics = read_json(run_dir / "metrics.json") if run_dir else {}
    score_gate = read_json(run_dir / "score_promotion_gate.json") if run_dir else {}
    search = read_json(run_dir / "search_controller_decision.json") if run_dir else {}
    claim = read_json(run_dir / "claim_audit.json") if run_dir else {}
    rank_gate = read_json(run_dir / "rank_promotion_gate.json") if run_dir else {}
    decision = score_gate.get("decision", {}) if isinstance(score_gate.get("decision"), dict) else {}
    return {
        "task_id": result.get("task_id"),
        "branch_id": result.get("branch_id"),
        "branch_type": result.get("branch_type"),
        "run_dir": run_dir_raw,
        "status": result.get("status"),
        "candidate_score": result.get("best_score"),
        "metric": metrics.get("metric"),
        "metric_direction": metric_direction(metrics),
        "features_after_encoding": metrics.get("features_after_encoding"),
        "best_method": metrics.get("ensemble", {}).get("best_method"),
        "score_gate": result.get("score_gate_decision"),
        "rank_gate": result.get("rank_gate_decision"),
        "parent_score": decision.get("parent_score"),
        "promotion_delta": decision.get("promotion_delta"),
        "claim_audit_result": claim.get("audit_result"),
        "rank_gate_status": rank_gate.get("status"),
        "top30_reached": bool(rank_gate.get("top30_reached")),
        "search_controller": {
            "selected_branch": search.get("selected_branch"),
            "code_generation_mode": search.get("code_generation_mode"),
            "code_generation_mode_requested": search.get("code_generation_mode_requested"),
            "stagnation_reason": search.get("stagnation_reason"),
            "cross_branch_references": search.get("cross_branch_references", []),
            "memory_reuse_records": search.get("memory_reuse_records", []),
        },
    }


def recommend_next(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No branch rows available; rerun workstation batch with max_branches > 0."]
    held = [row for row in rows if row.get("score_gate") == "hold"]
    promoted = [row for row in rows if row.get("score_gate") == "promote"]
    best = None
    for row in rows:
        if is_better(row.get("candidate_score"), best.get("candidate_score") if best else None, row.get("metric_direction", "maximize")):
            best = row
    actions = []
    if promoted:
        actions.append("A promoted candidate exists: run submission_audit and human submission gate before any official Kaggle submit.")
    else:
        actions.append("No branch promoted; keep official submit blocked and use this batch as negative retrospective memory.")
    if best:
        actions.append(
            f"Best held branch is {best.get('branch_id')} with {best.get('metric')}={best.get('candidate_score')}; use it as the next parent only if it closes the gap on a full-data run."
        )
    if held:
        actions.append("Run a non-fast calibration batch because fast sampled smoke underestimates the current best-so-far and should not drive official rank decisions.")
    actions.append("Switch to official submission gate only after full-data local CV beats protected best-so-far and claim_audit remains allow/revise.")
    return actions


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Top30 Branch Batch Audit",
        "",
        f"- Batch: `{payload['batch_id']}`",
        f"- Created at: `{payload['created_at']}`",
        f"- Source: `{payload['source_batch']}`",
        f"- Official submit allowed: `{payload['official_submit_allowed']}`",
        f"- Conclusion: `{payload['conclusion']}`",
        "",
        "## Branches",
        "",
        "| branch | type | status | score | parent | delta | score gate | rank gate | features |",
        "|---|---|---|---:|---:|---:|---|---|---:|",
    ]
    for row in payload["branches"]:
        lines.append(
            f"| `{row['branch_id']}` | `{row['branch_type']}` | `{row['status']}` | {row['candidate_score']} | {row['parent_score']} | {row['promotion_delta']} | `{row['score_gate']}` | `{row['rank_gate']}` | {row['features_after_encoding']} |"
        )
    lines.extend(["", "## Next Actions", ""])
    for action in payload["next_actions"]:
        lines.append(f"- {action}")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- This audit is local/proxy evidence only.",
            "- Top30 remains unproven until a Kaggle response artifact updates rank_promotion_gate with top30_reached=true.",
            "- MLE-Bench parity remains unproven until the full 75-task protocol is evaluated.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a Top30 workstation branch batch.")
    parser.add_argument("--batch-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_path = Path(args.batch_json)
    if not batch_path.is_absolute():
        batch_path = ROOT / batch_path
    batch = read_json(batch_path)
    rows = [run_record(result) for result in batch.get("results", [])]
    promoted = [row for row in rows if row.get("score_gate") == "promote"]
    top30 = [row for row in rows if row.get("top30_reached")]
    payload = {
        "schema": "academic_research_os.top30_branch_batch_audit.v1",
        "batch_id": batch.get("batch_id"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_batch": batch_path.relative_to(ROOT).as_posix() if batch_path.is_relative_to(ROOT) else str(batch_path),
        "execution_subject": batch.get("execution_subject"),
        "official_submit_allowed": bool(promoted and top30),
        "conclusion": "promoted_candidate_available" if promoted else "all_candidates_held",
        "branches": rows,
        "next_actions": recommend_next(rows),
    }
    safe_batch_id = str(payload["batch_id"] or "unknown")
    out_json = WORKSPACE_DIR / "top30_branch_audits" / f"{safe_batch_id}.json"
    out_md = REPORTS_DIR / f"TOP30_BRANCH_BATCH_AUDIT_{safe_batch_id}.md"
    write_json(out_json, payload)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json": out_json.relative_to(ROOT).as_posix(), "md": out_md.relative_to(ROOT).as_posix(), "branches": len(rows), "conclusion": payload["conclusion"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
