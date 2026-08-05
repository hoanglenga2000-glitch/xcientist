#!/usr/bin/env python3
"""One-shot, human-approved private grading of frozen MLE-Bench candidates.

The command never trains a model and never contacts Kaggle.  A grade is
eligible only when a candidate-only result has passed its promotion gate and a
separate approval file binds both the frozen candidate manifest and the exact
submission bytes.  Grading evidence is append-only under ``regrades``; the
candidate result, checkpoint, results_current, and summary files are not
rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import run_mlebench_lite_wave0 as wave0
except ModuleNotFoundError:
    from scripts import run_mlebench_lite_wave0 as wave0

from research_os.mlebench_phase_a import (
    DEFAULT_OFFICIAL_DATA_ROOT,
    DEFAULT_REMOTE_ROOT,
    grade_private_submission,
    validate_submission_file,
)

APPROVAL_SCHEMA = "evomind.mlebench_lite.human_grader_approval.v1"
CANDIDATE_MANIFEST_SCHEMA = "evomind.mlebench_lite.frozen_candidate_manifest.v1"
CLAIM_SCHEMA = "evomind.mlebench_lite.private_grade_claim.v1"
RECEIPT_SCHEMA = "evomind.mlebench_lite.private_grade_receipt.v1"
GRADE_RESULT_SCHEMA = "evomind.mlebench_lite.private_grade_result.v1"
RUN_REPORT_SCHEMA = "evomind.mlebench_lite.regrade_report.v2"
CONFIRMATION_STATUS = "promotion_gate_passed_confirmation_pending"
CONFIRMATION_REASON = "explicit_candidate_only_human_confirmation_required"


@dataclass(frozen=True)
class PreparedCandidate:
    run_id: str
    competition_id: str
    task_root: Path
    source_result_path: Path
    source_result: dict[str, Any]
    source_result_bytes: bytes
    source_result_sha256: str
    submission_path: Path
    submission_bytes: bytes
    submission_sha256: str
    candidate_manifest_path: Path
    candidate_manifest: dict[str, Any]
    candidate_manifest_bytes: bytes
    candidate_manifest_sha256: str
    approval_path: Path
    approval: dict[str, Any]
    approval_bytes: bytes
    approval_sha256: str
    preflight_validation: dict[str, Any]
    claim_path: Path
    receipt_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bytes(path: Path) -> tuple[bytes, str]:
    payload = path.read_bytes()
    return payload, _sha256_bytes(payload)


def _read_json_evidence(path: Path, *, label: str) -> tuple[dict[str, Any], bytes, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    raw, digest = _read_bytes(path)
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid {label} JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return payload, raw, digest


def _write_bytes_exclusive(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Immutable evidence already exists: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n").encode(
        "utf-8"
    )


def _write_json_exclusive(path: Path, payload: Any) -> Path:
    return _write_bytes_exclusive(path, _json_bytes(payload))


def _next_monotonic_dir(root: Path, prefix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        path = root / f"{prefix}_{number:03d}"
        try:
            path.mkdir(parents=False, exist_ok=False)
            return path
        except FileExistsError:
            number += 1


def next_regrade_dir(task_root: Path) -> Path:
    return _next_monotonic_dir(task_root / "regrades", "regrade")


def next_run_report_dir(run_dir: Path) -> Path:
    return _next_monotonic_dir(run_dir / "regrade_reports", "regrade")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REMOTE_ROOT / "mlebench_lite_runs")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OFFICIAL_DATA_ROOT)
    parser.add_argument("--official-source-root", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, default=DEFAULT_REMOTE_ROOT)
    parser.add_argument(
        "--approval-file",
        type=Path,
        action="append",
        required=True,
        help=(
            "Path to a JSON human-approval record. Repeat once per requested competition; "
            "the file must bind the frozen candidate manifest and submission SHA256."
        ),
    )
    return parser.parse_args(argv)


def load_requested(run_dir: Path) -> list[str]:
    checkpoint_path = run_dir / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    requested = [str(item) for item in checkpoint.get("requested", [])]
    if not requested:
        raise RuntimeError("Checkpoint contains no requested competitions")
    if len(requested) != len(set(requested)):
        raise RuntimeError("Checkpoint contains duplicate requested competitions")
    return requested


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _normalized_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError(f"{label} must be a lowercase-compatible SHA256 hex digest")
    return digest


def _resolve_declared_path(value: Any, *, relative_to: Path, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise RuntimeError(f"{label} path is missing")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _validate_source_result(current: dict[str, Any], competition_id: str) -> None:
    _require(current.get("competition_id") == competition_id, "Source result competition mismatch")
    _require(current.get("status") == CONFIRMATION_STATUS, "Source result is not confirmation-pending")
    promotion_gate = current.get("promotion_gate")
    _require(isinstance(promotion_gate, dict), "Source result promotion gate is missing")
    _require(promotion_gate.get("passed") is True, "Source result promotion gate did not pass")
    _require(current.get("candidate_only") is True, "Source result is not candidate-only")
    _require(
        current.get("official_grader_confirmation_pending") is True,
        "Source result has no pending human grader confirmation",
    )
    _require(current.get("official_grader_withheld") is True, "Source result did not withhold grader")
    _require(current.get("official_grader_executed") is False, "Source result was already graded")
    _require(current.get("valid_submission") is True, "Source result submission is not valid")
    private_grader = current.get("private_grader")
    _require(isinstance(private_grader, dict), "Source result private grader evidence is missing")
    _require(
        private_grader.get("reason") == CONFIRMATION_REASON,
        "Source result was not withheld for explicit human confirmation",
    )


def _assert_one_shot_available(claim_path: Path, receipt_path: Path) -> None:
    if receipt_path.exists():
        raise RuntimeError(f"Official private grade receipt already exists: {receipt_path}")
    if claim_path.exists():
        raise RuntimeError(f"Official private grade was already claimed: {claim_path}")


def prepare_candidate(
    *,
    run_id: str,
    competition_id: str,
    run_dir: Path,
    data_root: Path,
    allowed_root: Path,
    approval_path: Path,
) -> PreparedCandidate:
    task_root = wave0.ensure_within(run_dir / competition_id, allowed_root)
    source_result_path = task_root / "result.json"
    current, source_result_bytes, source_result_sha256 = _read_json_evidence(
        source_result_path, label="source result"
    )
    _validate_source_result(current, competition_id)

    resolved_approval_path = wave0.ensure_within(approval_path, allowed_root)
    approval, approval_bytes, approval_sha256 = _read_json_evidence(
        resolved_approval_path, label="human approval"
    )
    _require(approval.get("schema") == APPROVAL_SCHEMA, "Human approval schema mismatch")
    _require(approval.get("approved") is True, "Human approval record is not approved")
    _require(bool(str(approval.get("approval_id") or "").strip()), "Human approval ID is missing")
    _require(approval.get("run_id") == run_id, "Human approval run ID mismatch")
    _require(approval.get("competition_id") == competition_id, "Human approval competition mismatch")

    manifest_path = _resolve_declared_path(
        approval.get("candidate_manifest_path"),
        relative_to=resolved_approval_path.parent,
        label="candidate manifest",
    )
    manifest_path = wave0.ensure_within(manifest_path, task_root)
    manifest, manifest_bytes, manifest_sha256 = _read_json_evidence(
        manifest_path, label="candidate manifest"
    )
    approved_manifest_sha256 = _normalized_sha256(
        approval.get("candidate_manifest_sha256"), label="Approved candidate manifest SHA256"
    )
    _require(
        manifest_sha256 == approved_manifest_sha256,
        "Candidate manifest SHA256 does not match human approval",
    )
    _require(
        manifest.get("schema") == CANDIDATE_MANIFEST_SCHEMA,
        "Candidate manifest schema mismatch",
    )
    _require(manifest.get("frozen") is True, "Candidate manifest is not frozen")
    _require(manifest.get("run_id") == run_id, "Candidate manifest run ID mismatch")
    _require(manifest.get("competition_id") == competition_id, "Candidate manifest competition mismatch")
    _require(
        _normalized_sha256(
            manifest.get("source_result_sha256"), label="Candidate source result SHA256"
        )
        == source_result_sha256,
        "Source result SHA256 does not match candidate manifest",
    )

    manifest_submission = manifest.get("submission")
    _require(isinstance(manifest_submission, dict), "Candidate manifest submission record is missing")
    result_submission_path = _resolve_declared_path(
        current.get("submission_path"), relative_to=task_root, label="source result submission"
    )
    result_submission_path = wave0.ensure_within(result_submission_path, task_root)
    manifest_submission_path = _resolve_declared_path(
        manifest_submission.get("path"), relative_to=manifest_path.parent, label="manifest submission"
    )
    manifest_submission_path = wave0.ensure_within(manifest_submission_path, task_root)
    _require(
        manifest_submission_path == result_submission_path,
        "Candidate manifest and source result point to different submissions",
    )
    if not result_submission_path.is_file():
        raise FileNotFoundError(f"Missing submission: {result_submission_path}")
    submission_bytes, submission_sha256 = _read_bytes(result_submission_path)

    result_sha256 = _normalized_sha256(
        current.get("submission_sha256"), label="Source result submission SHA256"
    )
    manifest_submission_sha256 = _normalized_sha256(
        manifest_submission.get("sha256"), label="Candidate manifest submission SHA256"
    )
    approval_submission_sha256 = _normalized_sha256(
        approval.get("submission_sha256"), label="Human approval submission SHA256"
    )
    _require(
        len({submission_sha256, result_sha256, manifest_submission_sha256, approval_submission_sha256})
        == 1,
        "Submission SHA256 drift across bytes, source result, manifest, or approval",
    )
    try:
        declared_size = int(manifest_submission.get("size_bytes"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Candidate manifest submission size is invalid") from exc
    _require(declared_size == len(submission_bytes), "Submission byte length does not match manifest")

    regrade_root = task_root / "regrades"
    claim_path = regrade_root / "official_private_grader.claim.json"
    receipt_path = regrade_root / "official_private_grader.receipt.json"
    _assert_one_shot_available(claim_path, receipt_path)

    preflight_validation = validate_submission_file(result_submission_path, competition_id, data_root)
    _require(preflight_validation.get("valid") is True, "Frozen submission failed preflight validation")

    return PreparedCandidate(
        run_id=run_id,
        competition_id=competition_id,
        task_root=task_root,
        source_result_path=source_result_path,
        source_result=current,
        source_result_bytes=source_result_bytes,
        source_result_sha256=source_result_sha256,
        submission_path=result_submission_path,
        submission_bytes=submission_bytes,
        submission_sha256=submission_sha256,
        candidate_manifest_path=manifest_path,
        candidate_manifest=manifest,
        candidate_manifest_bytes=manifest_bytes,
        candidate_manifest_sha256=manifest_sha256,
        approval_path=resolved_approval_path,
        approval=approval,
        approval_bytes=approval_bytes,
        approval_sha256=approval_sha256,
        preflight_validation=preflight_validation,
        claim_path=claim_path,
        receipt_path=receipt_path,
    )


def _assert_prepared_evidence_unchanged(prepared: PreparedCandidate) -> None:
    checks = (
        (prepared.source_result_path, prepared.source_result_sha256, "source result"),
        (prepared.submission_path, prepared.submission_sha256, "submission"),
        (
            prepared.candidate_manifest_path,
            prepared.candidate_manifest_sha256,
            "candidate manifest",
        ),
        (prepared.approval_path, prepared.approval_sha256, "human approval"),
    )
    for path, expected, label in checks:
        _, actual = _read_bytes(path)
        _require(actual == expected, f"{label.capitalize()} changed after preflight")


def _write_failure_receipt(
    *,
    prepared: PreparedCandidate,
    claim_sha256: str,
    regrade_dir: Path,
    failure_result_path: Path,
    exc: Exception,
    grader: dict[str, Any] | None,
) -> None:
    failure_result_sha256 = _sha256_bytes(failure_result_path.read_bytes())
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "created_at": utc_now(),
        "status": "grader_failed",
        "one_shot": True,
        "run_id": prepared.run_id,
        "competition_id": prepared.competition_id,
        "approval_id": prepared.approval["approval_id"],
        "approval_sha256": prepared.approval_sha256,
        "candidate_manifest_sha256": prepared.candidate_manifest_sha256,
        "source_result_sha256": prepared.source_result_sha256,
        "submission_sha256": prepared.submission_sha256,
        "claim_sha256": claim_sha256,
        "regrade_dir": str(regrade_dir),
        "grade_result_sha256": failure_result_sha256,
        "official_grader_executed": (
            None
            if grader is None
            else bool(grader.get("official_mlebench_grader_executed"))
        ),
        "mle_private_grader_score": None if grader is None else grader.get("score"),
        "grader_execution_state": (
            "unknown_after_exception"
            if grader is None
            else "executed"
            if grader.get("official_mlebench_grader_executed") is True
            else "not_executed"
        ),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "frozen_candidate_artifacts_mutated": False,
        "candidate_task_directory_appended": True,
    }
    _write_json_exclusive(prepared.receipt_path, receipt)


def execute_prepared_regrade(
    *,
    prepared: PreparedCandidate,
    data_root: Path,
    official_source_root: Path,
) -> dict[str, Any]:
    _assert_one_shot_available(prepared.claim_path, prepared.receipt_path)
    _assert_prepared_evidence_unchanged(prepared)

    regrade_dir = next_regrade_dir(prepared.task_root)
    _write_bytes_exclusive(regrade_dir / "source_result.json", prepared.source_result_bytes)
    _write_bytes_exclusive(regrade_dir / "human_approval.json", prepared.approval_bytes)
    _write_bytes_exclusive(
        regrade_dir / "candidate_manifest.json", prepared.candidate_manifest_bytes
    )
    _write_json_exclusive(
        regrade_dir / "source_submission_validation.json", prepared.preflight_validation
    )
    frozen_submission_path = _write_bytes_exclusive(
        regrade_dir / "frozen_submission.csv", prepared.submission_bytes
    )
    _, frozen_submission_sha256 = _read_bytes(frozen_submission_path)
    _require(
        frozen_submission_sha256 == prepared.submission_sha256,
        "Frozen submission snapshot SHA256 mismatch",
    )
    validation = validate_submission_file(
        frozen_submission_path, prepared.competition_id, data_root
    )
    _write_json_exclusive(regrade_dir / "submission_validation.json", validation)
    _require(validation.get("valid") is True, "Frozen submission snapshot failed validation")

    _assert_prepared_evidence_unchanged(prepared)
    _assert_one_shot_available(prepared.claim_path, prepared.receipt_path)
    claim = {
        "schema": CLAIM_SCHEMA,
        "created_at": utc_now(),
        "status": "claimed",
        "one_shot": True,
        "run_id": prepared.run_id,
        "competition_id": prepared.competition_id,
        "approval_id": prepared.approval["approval_id"],
        "approval_sha256": prepared.approval_sha256,
        "candidate_manifest_sha256": prepared.candidate_manifest_sha256,
        "source_result_sha256": prepared.source_result_sha256,
        "submission_sha256": prepared.submission_sha256,
        "regrade_dir": str(regrade_dir),
        "frozen_submission_path": str(frozen_submission_path),
    }
    try:
        _write_json_exclusive(prepared.claim_path, claim)
    except RuntimeError as exc:
        raise RuntimeError("Official private grade one-shot claim already exists") from exc
    claim_sha256 = _sha256_bytes(prepared.claim_path.read_bytes())

    grader_path = regrade_dir / "private_grader.json"
    grader: dict[str, Any] | None = None
    try:
        _, frozen_submission_sha256 = _read_bytes(frozen_submission_path)
        _require(
            frozen_submission_sha256 == prepared.submission_sha256,
            "Frozen submission snapshot changed before grader execution",
        )
        grader = grade_private_submission(
            frozen_submission_path,
            prepared.competition_id,
            data_root,
            official_source_root=official_source_root,
            seed=(prepared.source_result.get("budget") or {}).get("seed"),
            budget=prepared.source_result.get("budget") or {},
            code_paths=[Path(__file__), Path(sys.modules["research_os.mlebench_phase_a"].__file__)],
            output_path=grader_path,
        )
        grader_submission_sha256 = _normalized_sha256(
            (grader.get("provenance") or {}).get("submission_sha256"),
            label="Private grader submission SHA256",
        )
        _require(
            grader_submission_sha256 == prepared.submission_sha256,
            "Private grader provenance submission SHA256 mismatch",
        )
        passed = bool(
            validation.get("valid")
            and grader.get("status") == "passed"
            and grader.get("official_mlebench_grader_executed") is True
        )
        grade_result = {
            "schema": GRADE_RESULT_SCHEMA,
            "created_at": utc_now(),
            "run_id": prepared.run_id,
            "competition_id": prepared.competition_id,
            "status": "passed" if passed else "failed",
            "approval_id": prepared.approval["approval_id"],
            "approval_sha256": prepared.approval_sha256,
            "candidate_manifest_sha256": prepared.candidate_manifest_sha256,
            "source_result_sha256": prepared.source_result_sha256,
            "submission_sha256": prepared.submission_sha256,
            "frozen_submission_path": str(frozen_submission_path),
            "validation": validation,
            "private_grader": grader,
            "official_grader_executed": bool(
                grader.get("official_mlebench_grader_executed")
            ),
            "mle_private_grader_score": grader.get("score"),
            "training_reused": True,
            "training_executed": False,
            "kaggle_submission_executed": False,
            "frozen_candidate_artifacts_mutated": False,
            "candidate_task_directory_appended": True,
            "one_shot": True,
            "regrade_dir": str(regrade_dir),
        }
        grade_result_path = _write_json_exclusive(regrade_dir / "grade_result.json", grade_result)
        grade_result_sha256 = _sha256_bytes(grade_result_path.read_bytes())
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "created_at": utc_now(),
            "status": grade_result["status"],
            "one_shot": True,
            "run_id": prepared.run_id,
            "competition_id": prepared.competition_id,
            "approval_id": prepared.approval["approval_id"],
            "approval_sha256": prepared.approval_sha256,
            "candidate_manifest_sha256": prepared.candidate_manifest_sha256,
            "source_result_sha256": prepared.source_result_sha256,
            "submission_sha256": prepared.submission_sha256,
            "claim_sha256": claim_sha256,
            "regrade_dir": str(regrade_dir),
            "grade_result_sha256": grade_result_sha256,
            "official_grader_executed": grade_result["official_grader_executed"],
            "mle_private_grader_score": grade_result["mle_private_grader_score"],
            "frozen_candidate_artifacts_mutated": False,
            "candidate_task_directory_appended": True,
        }
        _write_json_exclusive(prepared.receipt_path, receipt)
        return grade_result
    except Exception as exc:
        failure_result = {
            "schema": GRADE_RESULT_SCHEMA,
            "created_at": utc_now(),
            "run_id": prepared.run_id,
            "competition_id": prepared.competition_id,
            "status": "grader_failed",
            "approval_id": prepared.approval["approval_id"],
            "approval_sha256": prepared.approval_sha256,
            "candidate_manifest_sha256": prepared.candidate_manifest_sha256,
            "source_result_sha256": prepared.source_result_sha256,
            "submission_sha256": prepared.submission_sha256,
            "training_executed": False,
            "kaggle_submission_executed": False,
            "frozen_candidate_artifacts_mutated": False,
            "candidate_task_directory_appended": True,
            "one_shot": True,
            "regrade_dir": str(regrade_dir),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        failure_result_path = _write_json_exclusive(
            regrade_dir / "grade_failure.json", failure_result
        )
        _write_failure_receipt(
            prepared=prepared,
            claim_sha256=claim_sha256,
            regrade_dir=regrade_dir,
            failure_result_path=failure_result_path,
            exc=exc,
            grader=grader,
        )
        raise


def regrade_one(
    *,
    competition_id: str,
    run_dir: Path,
    data_root: Path,
    official_source_root: Path,
    allowed_root: Path,
    approval_path: Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    prepared = prepare_candidate(
        run_id=run_id or run_dir.name,
        competition_id=competition_id,
        run_dir=run_dir,
        data_root=data_root,
        allowed_root=allowed_root,
        approval_path=approval_path,
    )
    return execute_prepared_regrade(
        prepared=prepared,
        data_root=data_root,
        official_source_root=official_source_root,
    )


def _index_approval_files(
    approval_paths: Iterable[Path], *, allowed_root: Path, run_id: str
) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for raw_path in approval_paths:
        path = wave0.ensure_within(raw_path, allowed_root)
        approval, _, _ = _read_json_evidence(path, label="human approval")
        _require(approval.get("schema") == APPROVAL_SCHEMA, "Human approval schema mismatch")
        _require(approval.get("run_id") == run_id, "Human approval run ID mismatch")
        competition_id = str(approval.get("competition_id") or "").strip()
        _require(bool(competition_id), "Human approval competition ID is missing")
        if competition_id in indexed:
            raise RuntimeError(f"Duplicate human approval for competition: {competition_id}")
        indexed[competition_id] = path
    return indexed


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    allowed_root = args.allowed_root.expanduser().resolve()
    output_root = wave0.ensure_within(args.output_root, allowed_root)
    data_root = wave0.ensure_within(args.data_root, allowed_root)
    official_source_root = wave0.ensure_within(args.official_source_root, allowed_root)
    run_dir = wave0.ensure_within(output_root / args.run_id, allowed_root)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")

    requested = load_requested(run_dir)
    approvals = _index_approval_files(
        args.approval_file, allowed_root=allowed_root, run_id=args.run_id
    )
    _require(
        set(approvals) == set(requested),
        "Approval files must match the requested competition set exactly",
    )

    report_dir = next_run_report_dir(run_dir)
    prepared: list[PreparedCandidate] = []
    try:
        for competition_id in requested:
            prepared.append(
                prepare_candidate(
                    run_id=args.run_id,
                    competition_id=competition_id,
                    run_dir=run_dir,
                    data_root=data_root,
                    allowed_root=allowed_root,
                    approval_path=approvals[competition_id],
                )
            )
    except Exception as exc:
        summary = {
            "schema": RUN_REPORT_SCHEMA,
            "created_at": utc_now(),
            "run_id": args.run_id,
            "status": "preflight_failed",
            "competition_count": len(requested),
            "official_private_grades_executed": 0,
            "training_executed": False,
            "kaggle_submission_executed": False,
            "frozen_candidate_artifacts_mutated": False,
            "candidate_task_directory_appended": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json_exclusive(report_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for candidate in prepared:
        try:
            results.append(
                execute_prepared_regrade(
                    prepared=candidate,
                    data_root=data_root,
                    official_source_root=official_source_root,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "competition_id": candidate.competition_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            break

    passed = sum(result.get("status") == "passed" for result in results)
    summary = {
        "schema": RUN_REPORT_SCHEMA,
        "created_at": utc_now(),
        "run_id": args.run_id,
        "status": "passed" if passed == len(requested) and not failures else "partial_failure",
        "competition_count": len(requested),
        "passed": passed,
        "failed": len(requested) - passed,
        "official_private_grades_executed": sum(
            bool(result.get("official_grader_executed")) for result in results
        ),
        "results": results,
        "failures": failures,
        "regrade_only": True,
        "training_reused": True,
        "training_executed": False,
        "kaggle_submission_executed": False,
        "frozen_candidate_artifacts_mutated": False,
        "candidate_task_directory_appended": bool(results or failures),
    }
    _write_json_exclusive(report_dir / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
