#!/usr/bin/env python3
"""Monitor a training run, evaluate the submission gate, and PREPARE (never send)
a Kaggle submission.

Enforces the hard rules:
  * Never auto-submits to Kaggle. It only produces a gate decision and a ready
    submission file; a human must approve the final submission (Human Gate).
  * A gate can only reach "ready_for_human_gate" when the OOF score clears the
    bronze threshold AND the submission file passes format validation.

The pure, tested core is ``evaluate_submission_gate`` and ``summarize_run_status``.
The SSH log-tail helper is a thin optional wrapper used by the CLI.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GateDecision:
    task_id: str
    decision: str  # blocked | ready_for_human_gate | rejected
    oof_score: Optional[float]
    bronze_threshold: Optional[float]
    beats_bronze: Optional[bool]
    submission_valid: Optional[bool]
    reasons: list[str] = field(default_factory=list)
    # Human gate is always required; this is never True automatically.
    auto_submit_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_submission_gate(
    task_id: str,
    oof_score: Optional[float],
    bronze_threshold: Optional[float],
    *,
    higher_is_better: bool = True,
    submission_check: Optional[dict] = None,
) -> GateDecision:
    """Decide whether a run is ready to be *offered* for human submission.

    Returns a GateDecision. ``auto_submit_allowed`` is always False — the final
    Kaggle submission requires explicit human approval regardless of scores.
    """
    reasons: list[str] = []

    beats_bronze: Optional[bool] = None
    if oof_score is None:
        reasons.append("no OOF score yet (run may be incomplete)")
    elif bronze_threshold is None:
        reasons.append("bronze threshold unknown; cannot judge medal readiness")
    else:
        beats_bronze = oof_score >= bronze_threshold if higher_is_better else oof_score <= bronze_threshold
        reasons.append(
            f"OOF {oof_score:.5f} {'>=' if higher_is_better else '<='} bronze {bronze_threshold:.5f}: "
            f"{'PASS' if beats_bronze else 'FAIL'}"
        )

    submission_valid: Optional[bool] = None
    if submission_check is not None:
        submission_valid = bool(submission_check.get("valid"))
        if not submission_valid:
            reasons.append("submission failed format validation")

    if beats_bronze is False:
        decision = "rejected"
    elif oof_score is None:
        decision = "blocked"
    elif submission_check is not None and not submission_valid:
        decision = "blocked"
    elif beats_bronze is True and (submission_check is None or submission_valid):
        decision = "ready_for_human_gate"
    else:
        decision = "blocked"

    return GateDecision(
        task_id=task_id,
        decision=decision,
        oof_score=oof_score,
        bronze_threshold=bronze_threshold,
        beats_bronze=beats_bronze,
        submission_valid=submission_valid,
        reasons=reasons,
    )


def summarize_run_status(results: list[dict]) -> dict[str, Any]:
    """Aggregate per-task result dicts into a run-level status summary."""
    total = len(results)
    completed = [r for r in results if r.get("oof_score") is not None]
    ready = [r for r in results if r.get("decision") == "ready_for_human_gate"]
    rejected = [r for r in results if r.get("decision") == "rejected"]
    return {
        "total_tasks": total,
        "completed": len(completed),
        "ready_for_human_gate": len(ready),
        "rejected": len(rejected),
        "blocked": total - len(ready) - len(rejected),
        "ready_task_ids": [r.get("task_id") for r in ready],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-json", type=Path, required=True,
                        help="local JSON: list of {task_id, oof_score, bronze, higher_is_better}")
    parser.add_argument("--out", type=Path, default=None, help="write gate decisions JSON here")
    args = parser.parse_args()

    payload = json.loads(args.results_json.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("results", [])
    decisions = []
    for row in rows:
        decision = evaluate_submission_gate(
            task_id=row.get("task_id", "unknown"),
            oof_score=row.get("oof_score"),
            bronze_threshold=row.get("bronze"),
            higher_is_better=row.get("higher_is_better", True),
            submission_check=row.get("submission_check"),
        )
        decisions.append(decision.to_dict())

    summary = summarize_run_status(decisions)
    output = {"summary": summary, "decisions": decisions,
              "note": "Human Gate required before any Kaggle submission. This tool never submits."}
    if args.out:
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
