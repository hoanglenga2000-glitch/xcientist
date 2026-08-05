#!/usr/bin/env python3
"""Build the SIIM Claim Audit from frozen, reviewed, post-grader evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

REQUIRED_REVIEW_CHECKS = (
    "patient_group_overlap_zero",
    "content_group_overlap_zero",
    "oof_coverage_exactly_once",
    "submission_schema_and_order",
    "private_labels_unavailable_during_training",
    "private_grader_not_executed",
    "official_submission_not_executed",
)


class ClaimAuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClaimAuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClaimAuditError(f"invalid {label}") from exc
    require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def same_run(payload: Mapping[str, Any], run_id: str, label: str) -> None:
    require(str(payload.get("run_id") or "") == run_id, f"{label} run_id mismatch")


def passed(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("status") or "").lower() in {
        "passed",
        "completed",
        "verified",
        "review_passed",
        "terminal_execution_recorded",
    }


def terminal_grader_recorded(payload: Mapping[str, Any]) -> bool:
    """The grader contract allows a single fail-closed terminal record.

    A failed-closed grader is not a score and must not be converted into a
    leaderboard, medal, or tuning signal.  It is still evidence that the
    exactly-once post-freeze gate was consumed and recorded.
    """

    return passed(payload) or str(payload.get("status") or "").lower() == "failed_closed"


def safe_artifact(run_dir: Path, relative: str) -> Path:
    candidate = Path(relative)
    require(relative != "" and not candidate.is_absolute() and ".." not in candidate.parts, "unsafe frozen artifact path")
    resolved = (run_dir / candidate).resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ClaimAuditError("frozen artifact escaped the Run directory") from exc
    return resolved


def _stable_time(ledger: Mapping[str, Any], grader: Mapping[str, Any], freeze: Mapping[str, Any]) -> str:
    for value in (ledger.get("recorded_at"), grader.get("completed_at"), freeze.get("frozen_at")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ClaimAuditError("stable audit time is missing")


def _atomic_same_or_write(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file() and path.read_bytes() == encoded, "existing Claim Audit conflicts with verified evidence")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_claim_audit(run_dir: Path, output: Path | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    require(run_dir.is_dir() and not run_dir.is_symlink(), "SIIM Run directory is missing or unsafe")
    paths = {
        name: run_dir / name
        for name in (
            "run.json",
            "metrics.json",
            "hpc_runtime.json",
            "review.json",
            "candidate_freeze.json",
            "private_grader.json",
            "private_grader_ledger.json",
        )
    }
    for name, path in paths.items():
        require(path.is_file() and not path.is_symlink(), f"required evidence is missing: {name}")
    payloads = {name: read_json(path, name) for name, path in paths.items()}
    run = payloads["run.json"]
    run_id = str(run.get("run_id") or "")
    require(run_id != "" and run_dir.name == run_id, "Run identity is invalid")
    require(run.get("task_type") == "image_classification", "Run task type is not image classification")
    require(run.get("dataset") == "siim-isic-melanoma-classification", "Run dataset is not SIIM-ISIC")
    gates = run.get("gates") if isinstance(run.get("gates"), dict) else {}
    require(gates.get("official_submission") == "forbidden", "Run does not forbid official submission")

    for name, payload in payloads.items():
        if name != "run.json":
            same_run(payload, run_id, name)

    freeze = payloads["candidate_freeze.json"]
    require(freeze.get("status") == "frozen_before_private_grader", "candidate is not frozen before grader")
    require(freeze.get("tuning_closed") is True, "candidate tuning is not closed")
    require(freeze.get("official_submission") == "forbidden", "candidate freeze permits official submission")
    require(int(freeze.get("private_grader_execution_count_before_freeze") or 0) == 0, "private grader ran before freeze")
    frozen_records = freeze.get("artifacts")
    require(isinstance(frozen_records, list) and bool(frozen_records), "candidate freeze artifact list is empty")
    frozen_hashes: dict[str, str] = {}
    for record in frozen_records:
        require(isinstance(record, dict), "invalid candidate freeze record")
        relative = str(record.get("path") or "")
        expected = str(record.get("sha256") or "").lower()
        require(len(expected) == 64, f"invalid frozen hash: {relative}")
        artifact = safe_artifact(run_dir, relative)
        require(artifact.is_file() and sha256_file(artifact) == expected, f"frozen artifact changed: {relative}")
        require(relative not in frozen_hashes, f"duplicate frozen artifact: {relative}")
        frozen_hashes[relative] = expected

    review = payloads["review.json"]
    require(passed(review), "Independent Review did not pass")
    review_checks = review.get("checks") if isinstance(review.get("checks"), dict) else {}
    for check in REQUIRED_REVIEW_CHECKS:
        require(review_checks.get(check) is True, f"Independent Review check failed: {check}")
    review_hashes = review.get("artifact_hashes") if isinstance(review.get("artifact_hashes"), dict) else {}
    for relative, expected in review_hashes.items():
        artifact = safe_artifact(run_dir, str(relative))
        require(artifact.is_file() and sha256_file(artifact) == str(expected).lower(), f"reviewed artifact changed: {relative}")

    freeze_sha = sha256_file(paths["candidate_freeze.json"])
    grader = payloads["private_grader.json"]
    ledger = payloads["private_grader_ledger.json"]
    require(terminal_grader_recorded(grader), "terminal private grader was not recorded")
    require(int(grader.get("execution_index") or grader.get("execution_count") or 0) == 1, "private grader execution count is not one")
    require(int(ledger.get("execution_count") or 0) == 1, "private grader ledger count is not one")
    require(str(grader.get("candidate_freeze_sha256") or "").lower() == freeze_sha, "private grader freeze binding mismatch")
    require(str(ledger.get("candidate_freeze_sha256") or "").lower() == freeze_sha, "private grader ledger freeze binding mismatch")
    require(str(ledger.get("result_sha256") or "").lower() == sha256_file(paths["private_grader.json"]), "private grader result hash mismatch")
    require(grader.get("executed_after_freeze") is True, "private grader did not attest post-freeze execution")
    require(grader.get("feedback_used_for_tuning") is False and ledger.get("feedback_used_for_tuning") is False, "private grader feedback entered tuning")
    require(grader.get("official_submission_executed") is False, "private grader executed an official submission")
    grader_status = str(grader.get("status") or "").lower()
    score: float | None
    if grader_status == "failed_closed":
        require(grader.get("mle_private_grader_score", grader.get("score")) in (None, ""), "failed-closed grader must not contain a score")
        require(str(grader.get("error") or grader.get("failure_reason") or "").strip(), "failed-closed grader lacks failure evidence")
        require(str(ledger.get("outcome") or "").lower() == "failed_closed", "grader ledger outcome mismatch")
        require(ledger.get("score") in (None, ""), "failed-closed grader ledger must not contain a score")
        score = None
    else:
        score = float(grader.get("mle_private_grader_score", grader.get("score")))
        require(math.isfinite(score) and 0 <= score <= 1, "private grader score is invalid")
        require(abs(score - float(ledger.get("score"))) <= 1e-12, "private grader ledger score mismatch")

    metrics = payloads["metrics.json"]
    require(int(metrics.get("private_grader_execution_count") or 0) == 0, "training metrics contain private grader feedback")
    require(metrics.get("kaggle_submission_executed") is False, "training metrics report a Kaggle submission")
    require(metrics.get("clinical_diagnosis_claimed") is False, "training metrics contain a clinical claim")
    runtime = payloads["hpc_runtime.json"]
    require(runtime.get("other_processes_modified") is False, "HPC evidence reports another process was modified")
    require(int(runtime.get("signals_sent") or 0) == 0, "HPC evidence reports process signals")

    audit = {
        "schema": "evomind.siim.claim_audit.v1",
        "run_id": run_id,
        "status": "passed",
        "checks": {
            "no_public_leaderboard_claim": True,
            "no_official_medal_claim": True,
            "no_clinical_diagnosis_claim": True,
            "private_grader_not_used_for_tuning": True,
            "terminal_private_grader_recorded_once": True,
            "candidate_hashes_unchanged": True,
            "official_submission_not_executed": True,
        },
        "bindings": {
            "candidate_freeze_sha256": freeze_sha,
            "review_sha256": sha256_file(paths["review.json"]),
            "metrics_sha256": sha256_file(paths["metrics.json"]),
            "private_grader_sha256": sha256_file(paths["private_grader.json"]),
            "private_grader_ledger_sha256": sha256_file(paths["private_grader_ledger.json"]),
        },
        "evaluation_scope": "independent_offline_patient_content_grouped",
        "terminal_private_grader_status": grader_status,
        "terminal_private_grader_score": score,
        "terminal_private_grader_failure": (
            str(grader.get("error") or grader.get("failure_reason") or "")
            if grader_status == "failed_closed"
            else None
        ),
        "public_leaderboard_submission": "not_executed",
        "official_rank_or_medal": "not_claimed",
        "clinical_use": "not_claimed",
        "post_grader_tuning": "forbidden",
        "signals_sent": 0,
        "other_processes_modified": False,
        "audited_at": _stable_time(ledger, grader, freeze),
    }
    destination = Path(output).resolve() if output else run_dir / "ingress" / "claim_audit.json"
    if destination != run_dir / "ingress" / "claim_audit.json":
        require(destination.parent.is_dir() or destination.parent.parent.exists(), "Claim Audit output parent is invalid")
    _atomic_same_or_write(destination, audit)
    return {"status": "passed", "run_id": run_id, "claim_audit": str(destination), "sha256": sha256_file(destination)}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_claim_audit(args.run_dir, args.output)
    except (ClaimAuditError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed_closed", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
