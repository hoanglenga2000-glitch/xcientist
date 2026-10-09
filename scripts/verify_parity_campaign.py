"""Verify an externally generated EvoMind/Codex/Claude Code campaign."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xsci.parity_campaign_harness import (  # noqa: E402
    ParityEvidenceError,
    deterministic_json,
    load_suite_manifest,
    load_trial_rows,
    verify_campaign,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a complete external 100-task x 3-repeat x 3-agent evidence matrix. "
            "This command never launches paid agents."
        )
    )
    parser.add_argument("--suite-manifest", required=True, type=Path)
    parser.add_argument("--trials", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        suite, suite_sha256 = load_suite_manifest(args.suite_manifest)
        rows = load_trial_rows(args.trials)
        result = verify_campaign(suite, suite_sha256, rows, evidence_root=args.evidence_root)
    except (OSError, ParityEvidenceError, ValueError) as exc:
        print(f"parity campaign evidence rejected: {exc}", file=sys.stderr)
        return 2
    rendered = deterministic_json(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    # Local evidence readiness is intentionally not a release certificate.
    return 0 if result["status"] == "evidence_projection_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
