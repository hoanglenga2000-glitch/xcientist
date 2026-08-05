#!/usr/bin/env python3
"""Fail-closed verifier for the real EvoMind novice-agent quality report."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xsci.assistant_quality_evaluation import (
    SCHEMA,
    implementation_hashes,
    load_suite,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "workspace" / "evaluation" / "assistant_novice_quality_gpt56_current.json"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _age_hours(value: str) -> float:
    generated = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - generated).total_seconds() / 3600.0)


def verify(report_path: Path, suite_path: Path, *, max_age_hours: float, max_p95_seconds: float) -> dict[str, Any]:
    suite = load_suite(suite_path)
    report = _read_object(report_path)
    expected_cases = [str(item["case_id"]) for item in suite["cases"]]
    results = report.get("results") if isinstance(report.get("results"), list) else []
    aggregate = report.get("aggregate") if isinstance(report.get("aggregate"), dict) else {}
    governance = report.get("governance_invariants") if isinstance(report.get("governance_invariants"), dict) else {}
    current_hashes = implementation_hashes(ROOT)
    recorded_hashes = report.get("implementation_hashes") if isinstance(report.get("implementation_hashes"), dict) else {}
    age = _age_hours(str(report.get("generated_at") or "1970-01-01T00:00:00+00:00"))

    checks: dict[str, bool] = {
        "schema": report.get("schema") == SCHEMA,
        "mode": report.get("mode") == "live_production_tool_loop",
        "status": report.get("status") == "passed",
        "suite_id": report.get("suite_id") == suite["suite_id"],
        "suite_version": report.get("suite_version") == suite["version"],
        "suite_sha256": report.get("suite_sha256") == suite["suite_sha256"],
        "implementation_hashes": recorded_hashes == current_hashes and all(current_hashes.values()),
        "all_cases_selected": report.get("selected_cases") == expected_cases,
        "all_cases_passed": len(results) == len(expected_cases) and all(bool(item.get("passed")) for item in results),
        "aggregate": (
            int(aggregate.get("case_count") or 0) == len(expected_cases)
            and int(aggregate.get("passed_cases") or 0) == len(expected_cases)
            and float(aggregate.get("pass_rate") or 0) == 1.0
            and float(aggregate.get("mean_score") or 0) >= float(suite["pass_threshold"])
        ),
        "provider_model": all(
            (item.get("execution") or {}).get("provider") == suite.get("required_provider")
            and (item.get("execution") or {}).get("model") == suite.get("required_model")
            and bool((item.get("fatal_gate") or {}).get("provider_ok"))
            and bool((item.get("fatal_gate") or {}).get("model_ok"))
            for item in results
        ),
        "prompt_and_tools": all(
            bool(item.get("prompt_matches_case"))
            and not (item.get("fatal_gate") or {}).get("missing_tools")
            and not (item.get("fatal_gate") or {}).get("unexpected_tools")
            and not (item.get("fatal_gate") or {}).get("forbidden_hits")
            and not (item.get("fatal_gate") or {}).get("forbidden_regex_hits")
            for item in results
        ),
        "governance_unchanged": bool(governance.get("passed")) and governance.get("before") == governance.get("after"),
        "grader_exactly_once": (
            (governance.get("after") or {}).get("grader_execution_count") == 1
            and (governance.get("after") or {}).get("grader_outcome") == "failed_closed"
            and (governance.get("after") or {}).get("grader_score") is None
            and (governance.get("after") or {}).get("official_submission_executed") is False
        ),
        "freshness": age <= max_age_hours,
        "latency": float(aggregate.get("latency_p95_seconds") or 10**9) <= max_p95_seconds,
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema": "evomind.assistant_quality_gate.v1",
        "status": "passed" if not failed else "failed",
        "report": str(report_path.resolve()),
        "suite": str(suite_path.resolve()),
        "report_age_hours": round(age, 3),
        "max_age_hours": max_age_hours,
        "latency_p95_seconds": aggregate.get("latency_p95_seconds"),
        "max_p95_seconds": max_p95_seconds,
        "failed_checks": failed,
        "checks": checks,
        "claim_scope": "local_representative_novice_agent_gate_not_official_kaggle_or_mle_bench_proof",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--suite", type=Path, default=ROOT / "configs" / "evaluation" / "assistant_novice_v1.json")
    parser.add_argument("--max-age-hours", type=float, default=24.0)
    parser.add_argument("--max-p95-seconds", type=float, default=90.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(
            args.report.resolve(),
            args.suite.resolve(),
            max_age_hours=max(0.1, args.max_age_hours),
            max_p95_seconds=max(1.0, args.max_p95_seconds),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema": "evomind.assistant_quality_gate.v1",
            "status": "failed",
            "failed_checks": ["report_read"],
            "error_type": type(exc).__name__,
        }
    if args.output:
        target = args.output.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
