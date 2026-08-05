#!/usr/bin/env python3
"""Run the versioned EvoMind novice-agent quality gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from xsci.assistant_quality_evaluation import (
    DEFAULT_SUITE,
    build_live_report,
    build_recorded_pair_report,
    load_suite,
    render_markdown,
    write_report,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "workspace" / "evaluation" / "assistant_novice_quality_current.json"
DEFAULT_MARKDOWN = ROOT / "workspace" / "evaluation" / "assistant_novice_quality_current.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--live", action="store_true", help="Run selected cases through the production assistant bridge.")
    parser.add_argument("--case", action="append", dest="cases", help="Case id; repeat to select multiple live cases.")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--baseline-session")
    parser.add_argument("--treatment-session")
    parser.add_argument("--pair-case", default="siim_results_for_novice")
    parser.add_argument("--strict", action="store_true", help="Return non-zero unless the report passes.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    suite = load_suite(args.suite)
    if args.live:
        report = build_live_report(root, suite, case_ids=args.cases, timeout_seconds=args.timeout)
    else:
        if not args.baseline_session or not args.treatment_session:
            raise SystemExit("recorded-pair mode requires --baseline-session and --treatment-session")
        report = build_recorded_pair_report(
            root,
            suite,
            case_id=args.pair_case,
            baseline_session_id=args.baseline_session,
            treatment_session_id=args.treatment_session,
        )
    output = write_report(args.output, report)
    markdown = args.markdown_output.resolve()
    markdown.parent.mkdir(parents=True, exist_ok=True)
    temporary = markdown.with_name(f".{markdown.name}.{os.getpid()}.tmp")
    temporary.write_text(render_markdown(report), encoding="utf-8")
    os.replace(temporary, markdown)
    print(json.dumps({
        "status": report["status"],
        "mode": report["mode"],
        "output": str(output),
        "markdown_output": str(markdown),
        "aggregate": report.get("aggregate"),
        "comparison": report.get("comparison"),
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
