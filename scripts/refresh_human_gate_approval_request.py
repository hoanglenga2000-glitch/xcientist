#!/usr/bin/env python3
"""Rebuild the approval request from every verified, unapproved staged candidate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import stage_mlebench_human_gate_candidate as stage  # noqa: E402

DEFAULT_OUTPUT = stage.ALLOWED_OUTPUT_ROOT / "approval_request_current.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def record(path: Path) -> dict[str, Any]:
    item = stage.file_record(path)
    return {"path": str(item.path), "bytes": item.bytes, "sha256": item.sha256}


def build_request(root: Path = stage.ALLOWED_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    candidates = []
    for run_root in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "approvals"):
        verification_path = run_root / "verification.json"
        if not verification_path.is_file():
            continue
        try:
            verified = stage.verify_staged_run(run_root)
        except Exception:
            continue
        if verified.get("status") != "verified_human_approval_pending" or verified.get("approved") is not False:
            continue
        competition_dirs = [path for path in run_root.iterdir() if path.is_dir() and path.name != "approvals"]
        if len(competition_dirs) != 1:
            continue
        competition = competition_dirs[0]
        approval = run_root / "approvals" / f"{competition.name}.approval_template.json"
        manifest = competition / "frozen_candidate_manifest.json"
        submission = competition / "submission.csv"
        if not all(path.is_file() for path in (approval, manifest, submission)):
            continue
        candidates.append({
            "run_id": run_root.name, "competition_id": competition.name,
            "candidate_manifest": record(manifest), "submission": record(submission),
            "staging_verification": record(verification_path), "approval_template": record(approval),
            "approved": False, "official_grader_executed": False, "kaggle_submission_executed": False,
        })
    labels = " + ".join(item["competition_id"] for item in candidates)
    return {
        "schema": "evomind.mlebench_lite.human_gate_approval_request.v1", "created_at": now_iso(),
        "status": "awaiting_explicit_human_approval" if candidates else "no_verified_candidates_pending",
        "requested_action": "one_shot_official_mlebench_private_grade", "candidates": candidates,
        "exact_approval_phrase": f"批准执行一次 MLE-Bench 官方 private grader：{labels}；仅使用冻结候选；不提交 Kaggle。" if candidates else None,
        "on_approval": "Create new approval records from the frozen templates, bind manifest and submission SHA256, then execute each one-shot grader exactly once.",
        "automatic_approval": False, "official_grader_executed": False, "kaggle_submission_executed": False,
        "claim_boundary": "These are verified candidate-only packages, not official scores or medals.",
    }


def write_if_changed(output: Path, payload: dict[str, Any]) -> bool:
    output = Path(output).resolve()
    if output.is_file():
        prior = json.loads(output.read_text(encoding="utf-8"))
        comparable_prior = {key: value for key, value in prior.items() if key != "created_at"}
        comparable_new = {key: value for key, value in payload.items() if key != "created_at"}
        if comparable_prior == comparable_new:
            return False
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return True


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=stage.ALLOWED_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv); payload = build_request(args.root); changed = write_if_changed(args.output, payload)
    print(json.dumps({**payload, "changed": changed}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
