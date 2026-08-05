#!/usr/bin/env python3
"""CLI for preregistering, freezing, grading and aggregating MLE Lite A/B evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from research_os.mle_ab_campaign import (  # noqa: E402
    SCREEN_TASKS,
    aggregate,
    build_config,
    freeze_candidate,
    invalidate_campaign,
    preregister,
    record_blind_local_grade,
)
from research_os.mlebench_phase_a import LITE22_SPECS  # noqa: E402


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preregister")
    pre.add_argument("--root", type=Path, default=ROOT / "workspace" / "mle_ab_campaigns")
    pre.add_argument("--namespace", required=True)
    pre.add_argument("--phase", choices=("screen", "formal"), required=True)
    pre.add_argument("--provider", required=True)
    pre.add_argument("--canonical-manifest", type=Path, required=True)
    pre.add_argument("--source", type=Path, action="append", required=True)

    invalidate = sub.add_parser("invalidate")
    invalidate.add_argument("--campaign", type=Path, required=True)
    invalidate.add_argument("--reason-code", required=True)
    invalidate.add_argument("--reason", required=True)
    invalidate.add_argument("--replacement-namespace")
    invalidate.add_argument("--allow-existing-run-evidence", action="store_true")

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--campaign", type=Path, required=True)
    freeze.add_argument("--run-id", required=True)
    freeze.add_argument("--allowed-root", type=Path, required=True)
    freeze.add_argument("--candidate", type=Path, action="append", required=True)
    freeze.add_argument("--public-validation", type=Path, required=True)

    grade = sub.add_parser("record-grade")
    grade.add_argument("--campaign", type=Path, required=True)
    grade.add_argument("--run-id", required=True)
    grade.add_argument("--freeze-sha256", required=True)
    grade.add_argument("--grade", type=Path, required=True)

    report = sub.add_parser("aggregate")
    report.add_argument("--campaign", type=Path, required=True)
    report.add_argument("--bootstrap-samples", type=int, default=10_000)

    args = parser.parse_args()
    if args.command == "preregister":
        import hashlib

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        manifest_hash = digest(args.canonical_manifest)
        source_hashes = {str(path.resolve().relative_to(ROOT)): digest(path) for path in args.source}
        tasks = SCREEN_TASKS if args.phase == "screen" else tuple(spec.competition_id for spec in LITE22_SPECS)
        result = preregister(
            args.root,
            build_config(
                namespace=args.namespace,
                phase=args.phase,
                tasks=tasks,
                provider=args.provider,
                source_hashes=source_hashes,
                canonical_manifest_sha256=manifest_hash,
            ),
        )
    elif args.command == "invalidate":
        result = invalidate_campaign(
            args.campaign,
            reason_code=args.reason_code,
            reason=args.reason,
            replacement_namespace=args.replacement_namespace,
            require_zero_task_runs=not args.allow_existing_run_evidence,
        )
    elif args.command == "freeze":
        result = freeze_candidate(
            args.campaign,
            target_run_id=args.run_id,
            candidate_files=args.candidate,
            allowed_root=args.allowed_root,
            public_validation=_json(args.public_validation),
        )
    elif args.command == "record-grade":
        result = record_blind_local_grade(
            args.campaign,
            target_run_id=args.run_id,
            freeze_sha256=args.freeze_sha256,
            grade=_json(args.grade),
        )
    else:
        result = aggregate(args.campaign, bootstrap_samples=args.bootstrap_samples)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") != "failed_closed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
