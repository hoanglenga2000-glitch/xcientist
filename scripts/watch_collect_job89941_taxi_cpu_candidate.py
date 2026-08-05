#!/usr/bin/env python3
"""Wait for and collect the job89941 Taxi CPU candidate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_job89941_taxi_cpu_candidate import collect
from scripts.deploy_job89941_taxi_cpu_candidate import (
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_PLAN,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--deadline-hours", type=float, default=72.0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    while True:
        report = collect(args.plan, args.evidence_dir.resolve())
        if report.get("status") != "waiting_for_terminal_result":
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if datetime.now().astimezone() >= deadline:
            print(json.dumps({"status": "deadline_reached_process_left_running"}))
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
