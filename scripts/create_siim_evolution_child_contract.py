#!/usr/bin/env python3
"""Reserve an append-only SIIM evolution child Run without mutating its parent.

This tool records the operator-approved policy supersession separately from the
completed parent Run.  It never connects to HPC, runs a grader, or creates a
Kaggle submission.  The resulting full-file manifest is rechecked before an
evolution campaign is allowed to launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}")
REMOTE_ROOT = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"
CONTROL_PARENT = Path("workspace") / "siim_evolution_control"
FORMAL_RUN_PARENT = Path("workspace") / "evomind_runs"


class EvolutionContractError(RuntimeError):
    """Raised when lineage or immutable-parent evidence is inconsistent."""


def utc8_now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def validate_run_id(value: str) -> str:
    selected = str(value or "").strip()
    if not SAFE_RUN_ID.fullmatch(selected):
        raise EvolutionContractError("invalid SIIM Run ID")
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvolutionContractError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise EvolutionContractError(f"{label} must be an object")
    return payload


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_completed_parent(parent: Path, parent_run_id: str) -> dict[str, bool]:
    payloads = {
        name: read_json(parent / name, name)
        for name in (
            "run.json",
            "metrics.json",
            "review.json",
            "claim_audit.json",
            "candidate_freeze.json",
            "private_grader.json",
            "private_grader_ledger.json",
        )
    }
    for label, payload in payloads.items():
        if str(payload.get("run_id") or "") != parent_run_id:
            raise EvolutionContractError(f"{label} belongs to a different Run")
    run = payloads["run.json"]
    raw_tasks = run.get("tasks") or {}
    tasks = list(raw_tasks.values()) if isinstance(raw_tasks, dict) else list(raw_tasks)
    checks = {
        "run_completed": run.get("status") == "completed",
        "nine_tasks_completed": len(tasks) == 9
        and all(isinstance(task, dict) and task.get("status") == "completed" for task in tasks),
        "review_passed": payloads["review.json"].get("status") == "review_passed",
        "claim_audit_passed": payloads["claim_audit.json"].get("status") == "passed",
        "candidate_frozen": (
            payloads["candidate_freeze.json"].get("status")
            == "frozen_before_private_grader"
            and payloads["candidate_freeze.json"].get("tuning_closed") is True
        ),
        "parent_official_submission_not_executed": (
            payloads["private_grader.json"].get("official_submission_executed") is False
        ),
        "parent_grader_execution_count_one": int(
            payloads["private_grader_ledger.json"].get("execution_count") or 0
        )
        == 1,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise EvolutionContractError(f"parent completion contract failed: {failed}")
    return checks


def build_parent_manifest(parent: Path, parent_run_id: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    files = sorted(
        (candidate for candidate in parent.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(parent).as_posix(),
    )
    for path in files:
        records.append(
            {
                "path": path.relative_to(parent).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    canonical = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": "evomind.siim.parent_immutable_manifest.v1",
        "created_at": utc8_now(),
        "parent_run_id": parent_run_id,
        "parent_run_path": str(parent),
        "file_count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "records_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": records,
        "mutation_allowed": False,
    }


def _records_by_path(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = manifest.get("files")
    if not isinstance(records, list):
        raise EvolutionContractError("parent manifest file list is missing")
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict) or not item.get("path"):
            raise EvolutionContractError("parent manifest contains an invalid record")
        relative = str(item["path"])
        if relative in indexed:
            raise EvolutionContractError("parent manifest repeats a path")
        indexed[relative] = item
    return indexed


def verify_parent_manifest(parent: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    parent_run_id = validate_run_id(str(manifest.get("parent_run_id") or ""))
    current = build_parent_manifest(parent, parent_run_id)
    expected = _records_by_path(manifest)
    observed = _records_by_path(current)
    changed = sorted(
        path
        for path in set(expected) | set(observed)
        if expected.get(path) != observed.get(path)
    )
    return {
        "passed": not changed,
        "parent_run_id": parent_run_id,
        "expected_file_count": int(manifest.get("file_count") or -1),
        "observed_file_count": current["file_count"],
        "expected_records_sha256": manifest.get("records_sha256"),
        "observed_records_sha256": current["records_sha256"],
        "changed_paths": changed,
    }


def reserve(
    project_root: Path,
    *,
    parent_run_id: str,
    child_run_id: str,
    purpose: str,
) -> dict[str, Any]:
    root = project_root.resolve()
    parent_id = validate_run_id(parent_run_id)
    child_id = validate_run_id(child_run_id)
    if parent_id == child_id:
        raise EvolutionContractError("child Run must differ from its parent")
    parent = root / "workspace" / "evomind_runs" / parent_id
    if not parent.is_dir():
        raise EvolutionContractError("parent Run directory is missing")
    child = root / "workspace" / "evomind_runs" / child_id
    if child.exists():
        raise EvolutionContractError("child Run directory already exists")
    control = root / CONTROL_PARENT / child_id
    if control.exists():
        raise EvolutionContractError("child evolution control directory already exists")

    checks = validate_completed_parent(parent, parent_id)
    manifest = build_parent_manifest(parent, parent_id)
    manifest_path = control / "parent_immutable_manifest.json"
    atomic_json(manifest_path, manifest)
    metrics = read_json(parent / "metrics.json", "metrics.json")
    review = read_json(parent / "review.json", "review.json")
    claim = read_json(parent / "claim_audit.json", "claim_audit.json")
    grader = read_json(parent / "private_grader.json", "private_grader.json")
    constraint = {
        "schema": "evomind.siim.constraint_supersession.v1",
        "created_at": utc8_now(),
        "status": "effective",
        "supersedes_policy": "single_run_only",
        "authorized_change": {
            "new_candidate_run_allowed": True,
            "new_run_id_reserved": child_id,
            "purpose": purpose,
        },
        "parent_run": {
            "run_id": parent_id,
            "status": "completed_immutable",
            "path": str(parent),
            "immutable_manifest_path": str(manifest_path),
            "immutable_manifest_sha256": sha256_file(manifest_path),
            "baseline": {
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "brier": metrics.get("brier_score", metrics.get("brier")),
                "review_status": review.get("status"),
                "claim_audit_status": claim.get("status"),
                "grader_status": grader.get("status"),
                "grader_score": grader.get("score"),
            },
        },
        "child_run_contract": {
            "run_id": child_id,
            "parent_run_id": parent_id,
            "job_id": 90353,
            "credential_profile": "job90353",
            "remote_root": REMOTE_ROOT,
            "official_submission": "forbidden",
            "private_grader": "once_after_candidate_freeze",
            "parent_grader_reuse": "forbidden",
            "post_grader_tuning": "forbidden",
            "private_labels_in_training": "forbidden",
            "signals_sent": 0,
            "other_processes_modified": False,
        },
        "remaining_restrictions": [
            "parent_run_mutation_forbidden",
            "parent_grader_rerun_forbidden",
            "official_kaggle_submission_forbidden",
            "child_grader_more_than_once_forbidden",
            "post_grader_tuning_forbidden",
            "other_process_modification_forbidden",
            "remote_writes_outside_dedicated_root_forbidden",
        ],
        "verification": checks,
    }
    constraint_path = control / "constraint_supersession.json"
    atomic_json(constraint_path, constraint)
    summary = {
        "schema": "evomind.siim.evolution_reservation_summary.v1",
        "child_run_id": child_id,
        "control_dir": str(control),
        "constraint_sha256": sha256_file(constraint_path),
        "manifest_sha256": sha256_file(manifest_path),
        "parent_file_count": manifest["file_count"],
        "parent_total_bytes": manifest["total_bytes"],
        "checks": checks,
    }
    atomic_json(control / "reservation_summary.json", summary)
    return summary


def verify(project_root: Path, *, child_run_id: str) -> dict[str, Any]:
    root = project_root.resolve()
    child_id = validate_run_id(child_run_id)
    control = root / CONTROL_PARENT / child_id
    rescission_path = control / "constraint_rescission.json"
    if rescission_path.is_file():
        rescission = read_json(rescission_path, "constraint rescission")
        if (
            rescission.get("status") == "effective"
            and rescission.get("child_run_id") == child_id
            and (rescission.get("active_policy") or {}).get(
                "new_candidate_run_allowed"
            )
            is False
        ):
            raise EvolutionContractError("child evolution reservation was rescinded")
        raise EvolutionContractError("child evolution rescission record is invalid")
    constraint_path = control / "constraint_supersession.json"
    manifest_path = control / "parent_immutable_manifest.json"
    constraint = read_json(constraint_path, "constraint supersession")
    manifest = read_json(manifest_path, "parent immutable manifest")
    child_contract = constraint.get("child_run_contract")
    if not isinstance(child_contract, dict) or child_contract.get("run_id") != child_id:
        raise EvolutionContractError("constraint is not bound to the child Run")
    if constraint.get("status") != "effective":
        raise EvolutionContractError("constraint supersession is not effective")
    if child_contract.get("official_submission") != "forbidden":
        raise EvolutionContractError("official submission boundary changed")
    if child_contract.get("private_grader") != "once_after_candidate_freeze":
        raise EvolutionContractError("child grader boundary changed")
    if constraint.get("parent_run", {}).get("immutable_manifest_sha256") != sha256_file(
        manifest_path
    ):
        raise EvolutionContractError("constraint does not bind the parent manifest")
    parent_id = validate_run_id(str(manifest.get("parent_run_id") or ""))
    parent = root / "workspace" / "evomind_runs" / parent_id
    result = verify_parent_manifest(parent, manifest)
    result.update(
        {
            "schema": "evomind.siim.evolution_contract_verification.v1",
            "child_run_id": child_id,
            "constraint_sha256": sha256_file(constraint_path),
            "manifest_sha256": sha256_file(manifest_path),
            "official_submission": child_contract["official_submission"],
            "private_grader": child_contract["private_grader"],
        }
    )
    if not result["passed"]:
        raise EvolutionContractError(
            f"parent Run changed after reservation: {result['changed_paths']}"
        )
    return result


def rescind(
    project_root: Path,
    *,
    child_run_id: str,
    reason: str,
    goal_id: str = "",
) -> dict[str, Any]:
    """Append a fail-closed cancellation without deleting reservation evidence."""

    root = project_root.resolve()
    child_id = validate_run_id(child_run_id)
    selected_reason = str(reason or "").strip()
    if not selected_reason:
        raise EvolutionContractError("rescission reason is required")
    control = root / CONTROL_PARENT / child_id
    constraint_path = control / "constraint_supersession.json"
    constraint = read_json(constraint_path, "constraint supersession")
    authorized = constraint.get("authorized_change")
    child_contract = constraint.get("child_run_contract")
    if not isinstance(authorized, dict) or not isinstance(child_contract, dict):
        raise EvolutionContractError("reservation constraint is incomplete")
    if (
        constraint.get("status") != "effective"
        or authorized.get("new_candidate_run_allowed") is not True
        or authorized.get("new_run_id_reserved") != child_id
        or child_contract.get("run_id") != child_id
    ):
        raise EvolutionContractError("reservation constraint is not active for this child")

    formal_child = root / FORMAL_RUN_PARENT / child_id
    if formal_child.exists():
        raise EvolutionContractError("formal child Run already exists")

    preflight_path = control / "hpc_readonly_preflight.json"
    preflight = (
        read_json(preflight_path, "HPC read-only preflight")
        if preflight_path.is_file()
        else {}
    )
    if preflight:
        if preflight.get("run_id") != child_id:
            raise EvolutionContractError("HPC preflight belongs to another child Run")
        if preflight.get("training_started") is not False:
            raise EvolutionContractError("child training has already started")
        if preflight.get("remote_write_executed") is not False:
            raise EvolutionContractError("child remote write was already executed")

    campaign = (
        root
        / "workspace"
        / "hpc"
        / "job90353_siim_campaign"
        / child_id
    )
    campaign_files = sorted(
        path.relative_to(campaign).as_posix()
        for path in campaign.rglob("*")
        if path.is_file()
    ) if campaign.is_dir() else []
    active_markers = {
        "campaign_state.json",
        "launch.json",
        "remote_supervisor.json",
        "aggregation.json",
    }
    if active_markers.intersection(campaign_files):
        raise EvolutionContractError("child campaign already has execution evidence")

    rescission_path = control / "constraint_rescission.json"
    constraint_sha256 = sha256_file(constraint_path)
    if rescission_path.exists():
        existing = read_json(rescission_path, "constraint rescission")
        if (
            existing.get("status") != "effective"
            or existing.get("child_run_id") != child_id
            or (existing.get("rescinds") or {}).get("sha256")
            != constraint_sha256
            or (existing.get("active_policy") or {}).get(
                "new_candidate_run_allowed"
            )
            is not False
        ):
            raise EvolutionContractError("existing rescission record is inconsistent")
        return {
            "schema": "evomind.siim.evolution_rescission_summary.v1",
            "status": "already_rescinded",
            "child_run_id": child_id,
            "rescission_path": str(rescission_path),
            "rescission_sha256": sha256_file(rescission_path),
            "formal_child_run_exists": False,
            "training_started": False,
        }

    payload = {
        "schema": "evomind.siim.constraint_rescission.v1",
        "created_at": utc8_now(),
        "status": "effective",
        "child_run_id": child_id,
        "goal_id": str(goal_id or "").strip() or None,
        "reason": selected_reason,
        "rescinds": {
            "path": str(constraint_path),
            "sha256": constraint_sha256,
        },
        "active_policy": {
            "policy": "single_run_only",
            "new_candidate_run_allowed": False,
            "reserved_child_status": "cancelled_before_formal_run_creation",
        },
        "verification": {
            "formal_child_run_exists": False,
            "training_started": False,
            "remote_write_executed": False,
            "campaign_files": campaign_files,
            "parent_run_mutated": False,
            "grader_reexecuted": False,
            "official_submission_executed": False,
            "signals_sent": 0,
            "other_processes_modified": False,
        },
        "preservation": {
            "reservation_files_deleted": False,
            "parent_run_files_deleted": False,
            "append_only": True,
        },
    }
    atomic_json(rescission_path, payload)
    return {
        "schema": "evomind.siim.evolution_rescission_summary.v1",
        "status": "rescinded",
        "child_run_id": child_id,
        "rescission_path": str(rescission_path),
        "rescission_sha256": sha256_file(rescission_path),
        "formal_child_run_exists": False,
        "training_started": False,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    reserve_parser = subparsers.add_parser("reserve")
    reserve_parser.add_argument("--parent-run-id", required=True)
    reserve_parser.add_argument("--child-run-id", required=True)
    reserve_parser.add_argument(
        "--purpose", default="medal-target evolution candidate"
    )
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--child-run-id", required=True)
    rescind_parser = subparsers.add_parser("rescind")
    rescind_parser.add_argument("--child-run-id", required=True)
    rescind_parser.add_argument("--reason", required=True)
    rescind_parser.add_argument("--goal-id", default="")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "reserve":
            payload = reserve(
                args.project_root,
                parent_run_id=args.parent_run_id,
                child_run_id=args.child_run_id,
                purpose=args.purpose,
            )
        elif args.command == "verify":
            payload = verify(args.project_root, child_run_id=args.child_run_id)
        else:
            payload = rescind(
                args.project_root,
                child_run_id=args.child_run_id,
                reason=args.reason,
                goal_id=args.goal_id,
            )
    except (EvolutionContractError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps({"ok": True, **payload}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
