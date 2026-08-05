#!/usr/bin/env python3
"""Validate and merge one completed local MLE-Bench run into official progress."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from scripts import mlebench_campaign_supervisor as campaign
except ModuleNotFoundError:  # executed as ``python scripts/integrate_local_mlebench_run.py``
    import mlebench_campaign_supervisor as campaign


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNS_ROOT = PROJECT_ROOT / "workspace" / "local_gpu" / "mlebench_lite_runs"
DEFAULT_PROGRESS = PROJECT_ROOT / "workspace" / "mlebench_progress" / "lite11_current.json"
CAMPAIGN_LOCK_BUSY_MESSAGE = (
    "A live Job 89441 campaign supervisor already owns the lock"
)
DEFAULT_LOCK_WAIT_SECONDS = 6 * 60 * 60
DEFAULT_LOCK_POLL_SECONDS = 30.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def ensure_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    allowed = root.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"local run escapes the dedicated local root: {resolved}") from exc
    return resolved


def acquire_campaign_lock_with_retry(
    *,
    timeout_seconds: float = DEFAULT_LOCK_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_LOCK_POLL_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Acquire the shared campaign lock without treating live ownership as failure.

    Only the campaign supervisor's explicit live-owner condition is retried. Other
    errors still fail closed so filesystem and JSON faults are not hidden.
    """
    if timeout_seconds < 0:
        raise ValueError("lock timeout must be non-negative")
    if poll_seconds <= 0:
        raise ValueError("lock poll interval must be positive")

    started = monotonic()
    attempts = 0
    while True:
        attempts += 1
        try:
            campaign.acquire_lock()
        except RuntimeError as exc:
            if str(exc) != CAMPAIGN_LOCK_BUSY_MESSAGE:
                raise
            elapsed = max(0.0, monotonic() - started)
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                raise TimeoutError(
                    "timed out waiting for the live campaign lock owner to release"
                ) from exc
            sleep(min(poll_seconds, remaining))
            continue

        return {
            "attempts": attempts,
            "waited_seconds": max(0.0, monotonic() - started),
            "timeout_seconds": float(timeout_seconds),
            "poll_seconds": float(poll_seconds),
        }


def validate_completed_local_run(
    run_dir: Path,
    expected_competition: str,
    *,
    allowed_root: Path = LOCAL_RUNS_ROOT,
) -> dict[str, Any]:
    run_dir = ensure_within(run_dir, allowed_root)
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("completed run requires manifest.json and summary.json")

    manifest = read_json(manifest_path)
    summary = read_json(summary_path)
    run_id = str(manifest.get("run_id") or "")
    if not run_id or run_id != run_dir.name or summary.get("run_id") != run_id:
        raise ValueError("run identity mismatch")
    if manifest.get("training_started") is not True:
        raise ValueError("manifest does not prove training started")
    if manifest.get("human_gate_preserved") is not True:
        raise ValueError("human gate is not preserved")
    if manifest.get("kaggle_submission_enabled") is not False:
        raise ValueError("Kaggle auto-submission must remain disabled")
    if manifest.get("status") != "passed" or summary.get("status") != "passed":
        raise ValueError("run is not in the passed terminal state")
    if summary.get("competition_count") != 1 or summary.get("passed") != 1:
        raise ValueError("local integration requires exactly one passed competition")
    if summary.get("failed") != 0:
        raise ValueError("summary contains a failed competition")

    official = campaign.official_results_from_summary(summary)
    if len(official) != 1:
        raise ValueError("summary must contain exactly one valid official private grade")
    result = official[0]
    if result.get("competition_id") != expected_competition:
        raise ValueError("official result does not match the expected competition")
    score = result.get("mle_private_grader_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("official score is not numeric")
    if not math.isfinite(float(score)):
        raise ValueError("official score is not finite")

    upstream = ((result.get("private_grader") or {}).get("upstream_report") or {})
    any_medal = upstream.get("any_medal")
    if not isinstance(any_medal, bool):
        raise ValueError("official upstream grader did not return a boolean any_medal")

    return {
        "run_id": run_id,
        "competition_id": expected_competition,
        "summary_path": summary_path,
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": sha256_file(manifest_path),
        "score": float(score),
        "any_medal": any_medal,
    }


def integrate(
    run_dir: Path,
    expected_competition: str,
    *,
    output: Path,
    allowed_root: Path = LOCAL_RUNS_ROOT,
    progress_path: Path = DEFAULT_PROGRESS,
    lock_wait_seconds: float = DEFAULT_LOCK_WAIT_SECONDS,
    lock_poll_seconds: float = DEFAULT_LOCK_POLL_SECONDS,
) -> dict[str, Any]:
    validated = validate_completed_local_run(
        run_dir,
        expected_competition,
        allowed_root=allowed_root,
    )
    summary_path = Path(validated["summary_path"])
    relative_summary = summary_path.resolve().relative_to(PROJECT_ROOT)
    lock_evidence = acquire_campaign_lock_with_retry(
        timeout_seconds=lock_wait_seconds,
        poll_seconds=lock_poll_seconds,
    )
    try:
        progress_before = read_json(progress_path)
        progress_before_sha256 = sha256_file(progress_path)
        source_paths = {
            Path(str(item.get("path")))
            for item in progress_before.get("sources", [])
            if isinstance(item, dict) and item.get("path")
        }

        if relative_summary in source_paths:
            refresh = {
                "updated": False,
                "reason": "already_integrated",
                "summary": str(relative_summary),
            }
        else:
            refresh = campaign.refresh_official_progress(summary_path)

        progress_after = read_json(progress_path)
    finally:
        campaign.release_lock()
    report = {
        "schema": "evomind.mlebench.local_run_integration.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "ok": True,
        "run_id": validated["run_id"],
        "competition_id": validated["competition_id"],
        "official_score": validated["score"],
        "any_medal": validated["any_medal"],
        "summary": str(relative_summary),
        "summary_sha256": validated["summary_sha256"],
        "manifest_sha256": validated["manifest_sha256"],
        "campaign_lock": lock_evidence,
        "progress_before": {
            "scored_competitions": progress_before.get("scored_competitions"),
            "any_medal_count": progress_before.get("any_medal_count"),
            "sha256": progress_before_sha256,
        },
        "refresh": refresh,
        "progress_after": {
            "scored_competitions": progress_after.get("scored_competitions"),
            "remaining_competitions": progress_after.get("remaining_competitions"),
            "any_medal_count": progress_after.get("any_medal_count"),
            "padded_full_lite_any_medal_percentage": progress_after.get(
                "padded_full_lite_any_medal_percentage"
            ),
            "target_medals": progress_after.get(
                "medals_required_to_strictly_exceed_top"
            ),
            "sha256": sha256_file(progress_path),
        },
        "human_gate_preserved": True,
        "kaggle_submission_enabled": False,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--competition", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--lock-wait-seconds",
        type=float,
        default=DEFAULT_LOCK_WAIT_SECONDS,
        help="maximum time to wait for a live campaign lock owner",
    )
    parser.add_argument(
        "--lock-poll-seconds",
        type=float,
        default=DEFAULT_LOCK_POLL_SECONDS,
        help="poll interval while another live campaign supervisor owns the lock",
    )
    args = parser.parse_args()
    report = integrate(
        args.run_dir,
        args.competition,
        output=args.output,
        lock_wait_seconds=args.lock_wait_seconds,
        lock_poll_seconds=args.lock_poll_seconds,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
