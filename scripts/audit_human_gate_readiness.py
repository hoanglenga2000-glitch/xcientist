#!/usr/bin/env python3
"""Audit frozen Human Gate candidates without approving or grading them."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts import regrade_mlebench_lite_run as regrade  # noqa: E402
from scripts import stage_mlebench_human_gate_candidate as stage  # noqa: E402

DEFAULT_REQUEST = stage.ALLOWED_OUTPUT_ROOT / "approval_request_current.json"
DEFAULT_OUTPUT = stage.ALLOWED_OUTPUT_ROOT / "human_gate_readiness_audit_current.json"
AUDIT_SCHEMA = "evomind.mlebench_lite.human_gate_readiness_audit.v1"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{Path.cwd().name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def audit_candidate(
    candidate: dict[str, Any],
    *,
    staged_root: Path,
) -> dict[str, Any]:
    run_id = str(candidate.get("run_id") or "")
    competition_id = str(candidate.get("competition_id") or "")
    run_root = staged_root / run_id
    task_root = run_root / competition_id
    approval_template = Path(str(candidate["approval_template"]["path"]))
    manifest_path = Path(str(candidate["candidate_manifest"]["path"]))
    submission_path = Path(str(candidate["submission"]["path"]))
    item_errors: list[str] = []

    try:
        verified = stage.verify_staged_run(run_root)
        approval = read_json(approval_template)
        result = stage.read_json(task_root / "result.json", label="staged result")
        regrade._validate_source_result(result, competition_id)
    except Exception as exc:  # noqa: BLE001
        return {
            "run_id": run_id,
            "competition_id": competition_id,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }

    manifest_sha256 = sha256_file(manifest_path)
    submission_sha256 = sha256_file(submission_path)
    if verified.get("status") != "verified_human_approval_pending":
        item_errors.append("staged_run_not_pending")
    if verified.get("approved") is not False:
        item_errors.append("staged_run_not_unapproved")
    if approval.get("approved") is not False:
        item_errors.append("approval_template_already_approved")
    if approval.get("run_id") != run_id:
        item_errors.append("approval_run_id_mismatch")
    if approval.get("competition_id") != competition_id:
        item_errors.append("approval_competition_id_mismatch")
    if approval.get("candidate_manifest_sha256") != manifest_sha256:
        item_errors.append("manifest_hash_mismatch")
    if approval.get("submission_sha256") != submission_sha256:
        item_errors.append("submission_hash_mismatch")
    regrades_dir_exists = (task_root / "regrades").exists()
    if regrades_dir_exists:
        item_errors.append("regrades_directory_already_exists")

    return {
        "run_id": run_id,
        "competition_id": competition_id,
        "staged_status": verified.get("status"),
        "approved": verified.get("approved"),
        "approval_template_approved": approval.get("approved"),
        "candidate_manifest_sha256": manifest_sha256,
        "submission_sha256": submission_sha256,
        "source_result_contract": "passed",
        "regrades_dir_exists": regrades_dir_exists,
        "official_grader_executed": verified.get("official_grader_executed"),
        "kaggle_submission_executed": verified.get("kaggle_submission_executed"),
        "errors": item_errors,
    }


def build_audit(
    *,
    approval_request_path: Path = DEFAULT_REQUEST,
    staged_root: Path = stage.ALLOWED_OUTPUT_ROOT,
) -> dict[str, Any]:
    request = read_json(approval_request_path)
    checks = [
        audit_candidate(candidate, staged_root=staged_root)
        for candidate in request.get("candidates", [])
        if isinstance(candidate, dict)
    ]
    errors = [
        f"{item.get('run_id')}:{error}"
        for item in checks
        for error in item.get("errors", [])
    ]
    return {
        "schema": AUDIT_SCHEMA,
        "created_at": now_iso(),
        "approval_request": {
            "path": str(Path(approval_request_path).resolve()),
            "sha256": sha256_file(approval_request_path),
            "status": request.get("status"),
            "candidate_count": len(request.get("candidates") or []),
            "exact_approval_phrase": request.get("exact_approval_phrase"),
        },
        "checks": checks,
        "ready_for_explicit_human_approval": bool(checks) and not errors,
        "automatic_approval": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "errors": errors,
        "rule": (
            "This audit only verifies frozen staged packages and approval templates. "
            "It does not approve, grade, submit, train, or use private labels."
        ),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval-request", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--staged-root", type=Path, default=stage.ALLOWED_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_audit(
        approval_request_path=args.approval_request,
        staged_root=args.staged_root,
    )
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "ready_for_explicit_human_approval": payload[
                    "ready_for_explicit_human_approval"
                ],
                "candidate_count": len(payload["checks"]),
                "errors": payload["errors"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["ready_for_explicit_human_approval"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
