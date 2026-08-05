#!/usr/bin/env python3
"""Continuously refresh the verified staged-candidate approval index."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import refresh_human_gate_approval_request as refresh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=refresh.stage.ALLOWED_OUTPUT_ROOT / "approval_index_watcher.json")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--deadline-hours", type=float, default=720.0)
    args = parser.parse_args()
    if args.poll_seconds < 30 or args.deadline_hours <= 0: raise ValueError("Approval watcher timing invalid")
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    while datetime.now().astimezone() < deadline:
        payload = refresh.build_request(); changed = refresh.write_if_changed(refresh.DEFAULT_OUTPUT, payload)
        status = {"schema": "evomind.human_gate.approval_index_watcher.v1", "created_at": refresh.now_iso(), "status": "watching", "candidate_count": len(payload["candidates"]), "approval_request_changed": changed, "automatic_approval": False, "official_grader_executed": False, "kaggle_submission_executed": False}
        refresh.write_if_changed(args.evidence, status)
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__": raise SystemExit(main())
