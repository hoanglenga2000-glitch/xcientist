"""Verify an externally issued EvoMind capability certificate."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from xsci.capability_certification import (  # noqa: E402
    RESULT_SCHEMA,
    CertificationInputError,
    CertificationPolicy,
    deterministic_json,
    verify_capability_certification,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail closed unless an external held-out capability report, exact source, "
            "release artifacts, named baselines, and frozen-evaluator upgrade campaign all verify."
        )
    )
    parser.add_argument("report", type=Path, help="external capability evidence JSON")
    parser.add_argument("--repo-root", type=Path, default=ROOT, help="exact clean Git worktree being released")
    parser.add_argument("--artifact-root", type=Path, help="root for report-relative artifact paths")
    parser.add_argument("--expected-report-sha256", required=True, help="out-of-band external report SHA-256")
    parser.add_argument("--expected-suite-id", required=True)
    parser.add_argument("--expected-suite-sha256", required=True)
    parser.add_argument("--expected-evaluator-id", required=True)
    parser.add_argument("--expected-evaluator-sha256", required=True)
    parser.add_argument(
        "--baseline-agent",
        action="append",
        required=True,
        dest="baseline_agents",
        help="required named baseline; repeat for every release baseline",
    )
    parser.add_argument("--candidate-name", default="EvoMind")
    parser.add_argument("--minimum-hidden-tasks", type=int, default=100)
    parser.add_argument("--minimum-domains", type=int, default=8)
    parser.add_argument("--minimum-tasks-per-domain", type=int, default=3)
    parser.add_argument("--minimum-repeats", type=int, default=3)
    parser.add_argument("--minimum-success-rate", type=float, default=0.80)
    parser.add_argument("--minimum-wilson-lower-bound", type=float, default=0.75)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--baseline-noninferiority-margin", type=float, default=0.05)
    parser.add_argument("--maximum-timeout-rate", type=float, default=0.0)
    parser.add_argument("--minimum-upgrade-candidates", type=int, default=2)
    parser.add_argument("--maximum-evidence-age-days", type=int, default=30)
    parser.add_argument(
        "--required-artifact-role",
        action="append",
        dest="artifact_roles",
        help="override required artifact roles; repeat for multiple roles",
    )
    parser.add_argument(
        "--as-of",
        help="canonical UTC verification time (YYYY-MM-DDTHH:MM:SSZ); intended for reproducible audits",
    )
    parser.add_argument("--output", type=Path, help="also write the deterministic result JSON to this path")
    return parser


def _parse_as_of(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise CertificationInputError("--as-of must use YYYY-MM-DDTHH:MM:SSZ") from exc


def _input_error(message: str) -> dict[str, object]:
    return {
        "schema": RESULT_SCHEMA,
        "status": "FAIL",
        "release_allowed": False,
        "error": {"code": "invalid_verifier_input", "detail": message},
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        required_roles = (
            tuple(args.artifact_roles)
            if args.artifact_roles
            else ("wheel", "sdist", "source_archive", "benchmark_raw_results", "self_upgrade_campaign")
        )
        policy = CertificationPolicy(
            expected_report_sha256=args.expected_report_sha256,
            expected_suite_id=args.expected_suite_id,
            expected_suite_sha256=args.expected_suite_sha256,
            expected_evaluator_id=args.expected_evaluator_id,
            expected_evaluator_sha256=args.expected_evaluator_sha256,
            required_baseline_agents=tuple(args.baseline_agents),
            expected_candidate_name=args.candidate_name,
            minimum_hidden_tasks=args.minimum_hidden_tasks,
            minimum_domains=args.minimum_domains,
            minimum_tasks_per_domain=args.minimum_tasks_per_domain,
            minimum_repeats=args.minimum_repeats,
            minimum_candidate_success_rate=args.minimum_success_rate,
            minimum_candidate_wilson_lower_bound=args.minimum_wilson_lower_bound,
            confidence_level=args.confidence_level,
            baseline_noninferiority_margin=args.baseline_noninferiority_margin,
            maximum_timeout_rate=args.maximum_timeout_rate,
            minimum_upgrade_candidates=args.minimum_upgrade_candidates,
            maximum_evidence_age_days=args.maximum_evidence_age_days,
            required_artifact_roles=required_roles,
        )
        result = verify_capability_certification(
            args.report,
            repository=args.repo_root,
            artifact_root=args.artifact_root,
            policy=policy,
            now=_parse_as_of(args.as_of),
        )
        exit_code = 0 if result["release_allowed"] else 1
    except CertificationInputError as exc:
        result = _input_error(str(exc))
        exit_code = 2

    rendered = deterministic_json(result)
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8", newline="\n")
        except OSError as exc:
            result = _input_error(f"output could not be written: {exc.__class__.__name__}")
            rendered = deterministic_json(result)
            exit_code = 2
    sys.stdout.write(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
