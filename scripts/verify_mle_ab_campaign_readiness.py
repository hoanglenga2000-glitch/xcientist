#!/usr/bin/env python3
"""Fail-closed readiness gate for a preregistered MLE-Bench Lite A/B campaign.

The gate is read-only.  It verifies immutable campaign inputs, official
prepared-data MD5 contracts, SIIM inventory completeness, a namespace/lock
bound authenticated metadata probe, host identity, and same-hardware receipt
evidence.  It never starts a task-run, requests a completion, grades a
candidate, calls the SIIM private grader, or submits to Kaggle.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research_os.mle_ab_campaign import (  # noqa: E402
    ARMS,
    SCREEN_TASKS,
    UNIQUE_SIIM_RUN,
    canonical_bytes,
    load_campaign,
    sha256_bytes,
    sha256_file,
    write_json_exclusive,
)
from research_os.mlebench_phase_a import resolve_competition  # noqa: E402
from scripts.build_mle_prepared_contract import (  # noqa: E402
    DEFAULT_UPSTREAM_ROOT,
    PINNED_MLEBENCH_COMMIT,
    PINNED_MLEBENCH_VERSION,
    PINNED_SIIM_INVENTORY,
    PreparedContractError,
    verify_prepared_contract,
)

READINESS_SCHEMA = "evomind.mle_lite_ab.readiness.v2"
EXECUTION_ENVIRONMENT_SCHEMA = "evomind.mle_lite_ab.execution_environment.v2"
AUTH_NO_COMPLETION_PROBE_SCHEMA = "evomind.model_gateway.auth_no_completion_probe.v1"
HOST_FINGERPRINT_SCHEMA = "evomind.execution_host.fingerprint.v1"
SAME_HARDWARE_POLICY_SCHEMA = "evomind.mle_lite_ab.same_hardware_policy.v1"
SAME_HARDWARE_RECEIPT_SCHEMA = "evomind.mle_lite_ab.same_hardware_receipt.v1"
DEFAULT_DATA_ROOT = ROOT / "workspace" / "local_gpu" / "mlebench_official_data"
DEFAULT_CANONICAL_MANIFEST = ROOT / "workspace" / "mle_ab_campaigns" / "canonical_mlebench_lite22_v1.json"
_SAFE_PROVIDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,95}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_projection(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return "external_read_only_data_root"


def _canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def _validate_host_fingerprint(value: Any) -> tuple[dict[str, Any], str]:
    _require(isinstance(value, dict), "host fingerprint must be an object")
    fingerprint = dict(value)
    _require(fingerprint.get("schema") == HOST_FINGERPRINT_SCHEMA, "host fingerprint schema mismatch")
    _require(
        bool(_SHA256.fullmatch(str(fingerprint.get("machine_id_sha256") or "").lower())),
        "host machine ID hash is invalid",
    )
    _require(bool(str(fingerprint.get("os_family") or "").strip()), "host OS family is missing")
    _require(bool(str(fingerprint.get("cpu_architecture") or "").strip()), "host CPU architecture is missing")
    return fingerprint, _canonical_sha256(fingerprint)


def _read_execution_environment(
    path: Path,
    prereg: Mapping[str, Any],
    *,
    campaign_lock_sha256: str,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file() and not resolved.is_symlink(), "execution-environment evidence must be a regular file")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid execution-environment evidence") from exc
    _require(isinstance(value, dict), "execution-environment evidence must be an object")
    _require(value.get("schema") == EXECUTION_ENVIRONMENT_SCHEMA, "execution-environment evidence schema mismatch")
    _require(value.get("schema_version") == 2, "execution-environment schema version mismatch")

    provider = str(value.get("provider") or "")
    _require(bool(_SAFE_PROVIDER.fullmatch(provider)), "execution-environment provider is invalid")
    namespace_matches = value.get("namespace") == prereg.get("namespace")
    lock_matches = value.get("campaign_lock_sha256") == campaign_lock_sha256
    provider_matches = provider == prereg.get("provider")
    model_matches = value.get("model") == prereg.get("model")

    probe = value.get("authenticated_no_completion_probe")
    _require(isinstance(probe, dict), "authenticated no-completion probe is missing")
    probe_valid = (
        probe.get("schema") == AUTH_NO_COMPLETION_PROBE_SCHEMA
        and probe.get("request_kind") == "authenticated_metadata"
        and probe.get("authentication_succeeded") is True
        and probe.get("completion_requested") is False
        and isinstance(probe.get("http_status"), int)
        and 200 <= int(probe["http_status"]) < 300
        and bool(_SHA256.fullmatch(str(probe.get("response_sha256") or "").lower()))
        and bool(str(probe.get("observed_at") or "").strip())
    )

    _fingerprint, computed_fingerprint_sha256 = _validate_host_fingerprint(value.get("host_fingerprint"))
    claimed_fingerprint_sha256 = str(value.get("host_fingerprint_sha256") or "").lower()
    host_fingerprint_valid = claimed_fingerprint_sha256 == computed_fingerprint_sha256

    policy = value.get("same_hardware_policy")
    _require(isinstance(policy, dict), "same-hardware policy is missing")
    policy_valid = (
        policy.get("schema") == SAME_HARDWARE_POLICY_SCHEMA
        and policy.get("receipts_required_for_every_run") is True
        and policy.get("receipt_schema") == SAME_HARDWARE_RECEIPT_SCHEMA
        and policy.get("pair_key_fields") == ["task_id", "seed"]
        and policy.get("paired_arms") == list(ARMS)
        and policy.get("host_fingerprint_sha256") == claimed_fingerprint_sha256
    )
    environment_sha256 = sha256_file(resolved)
    valid = all(
        (
            namespace_matches,
            lock_matches,
            provider_matches,
            model_matches,
            probe_valid,
            host_fingerprint_valid,
            policy_valid,
        )
    )
    return {
        "present": True,
        "valid": valid,
        "sha256": environment_sha256,
        "namespace_matches": namespace_matches,
        "campaign_lock_matches": lock_matches,
        "provider_matches": provider_matches,
        "model_matches": model_matches,
        "authenticated_no_completion_probe_verified": probe_valid,
        "host_fingerprint_verified": host_fingerprint_valid,
        "host_fingerprint_sha256": computed_fingerprint_sha256,
        "same_hardware_policy_verified": policy_valid,
    }


def _missing_execution_environment(error: str | None = None) -> dict[str, Any]:
    return {
        "present": False,
        "valid": False,
        "sha256": None,
        "namespace_matches": False,
        "campaign_lock_matches": False,
        "provider_matches": False,
        "model_matches": False,
        "authenticated_no_completion_probe_verified": False,
        "host_fingerprint_verified": False,
        "host_fingerprint_sha256": None,
        "same_hardware_policy_verified": False,
        "error": error,
    }


def build_same_hardware_receipt(
    campaign_dir: str | Path,
    *,
    target_run_id: str,
    execution_environment: str | Path,
    observed_host_fingerprint: Mapping[str, Any],
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Build one pre-execution receipt without starting or grading the run."""

    campaign = Path(campaign_dir).expanduser().resolve()
    prereg, schedule = load_campaign(campaign)
    campaign_lock_sha256 = sha256_file(campaign / "campaign-lock.json")
    environment = _read_execution_environment(
        Path(execution_environment),
        prereg,
        campaign_lock_sha256=campaign_lock_sha256,
    )
    _require(environment["valid"], "execution environment is not fully bound")
    fingerprint, fingerprint_sha256 = _validate_host_fingerprint(observed_host_fingerprint)
    _require(
        fingerprint_sha256 == environment["host_fingerprint_sha256"],
        "observed host differs from the campaign execution host",
    )
    matches = [
        dict(item)
        for item in list(schedule.get("runs") or [])
        if isinstance(item, dict) and item.get("run_id") == target_run_id
    ]
    _require(len(matches) == 1, "run is not uniquely preregistered")
    record = matches[0]
    return {
        "schema": SAME_HARDWARE_RECEIPT_SCHEMA,
        "schema_version": 1,
        "namespace": prereg["namespace"],
        "campaign_lock_sha256": campaign_lock_sha256,
        "execution_environment_sha256": environment["sha256"],
        "run_id": target_run_id,
        "task_id": record["task_id"],
        "seed": record["seed"],
        "arm": record["arm"],
        "host_fingerprint": fingerprint,
        "host_fingerprint_sha256": fingerprint_sha256,
        "status": "host_bound_before_execution",
        "completion_started": False,
        "recorded_at": recorded_at or utc_now(),
    }


def write_same_hardware_receipt_exclusive(
    campaign_dir: str | Path,
    *,
    target_run_id: str,
    execution_environment: str | Path,
    observed_host_fingerprint: Mapping[str, Any],
    receipts_root: str | Path | None = None,
    recorded_at: str | None = None,
) -> Path:
    """Persist one immutable receipt outside the run directory."""

    campaign = Path(campaign_dir).expanduser().resolve()
    payload = build_same_hardware_receipt(
        campaign,
        target_run_id=target_run_id,
        execution_environment=execution_environment,
        observed_host_fingerprint=observed_host_fingerprint,
        recorded_at=recorded_at,
    )
    root = Path(receipts_root).expanduser().resolve() if receipts_root is not None else campaign / "same-hardware-receipts"
    return write_json_exclusive(root / f"{target_run_id}.json", payload)


def verify_same_hardware_receipts(
    schedule: Mapping[str, Any],
    *,
    namespace: str,
    campaign_lock_sha256: str,
    execution_environment: Mapping[str, Any],
    receipts_root: Path,
    launched_run_ids: Sequence[str] = (),
    require_all: bool = False,
) -> dict[str, Any]:
    scheduled = {
        str(item["run_id"]): dict(item)
        for item in list(schedule.get("runs") or [])
        if isinstance(item, dict) and item.get("run_id")
    }
    required = set(scheduled) if require_all else set(launched_run_ids)
    valid: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    root = receipts_root.expanduser().resolve()
    paths: list[Path] = []
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            errors.append("receipt_root_not_regular_directory")
        else:
            paths = sorted(root.iterdir(), key=lambda item: item.name)
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".json":
            errors.append(f"unexpected_receipt_entry:{path.name}")
            continue
        try:
            receipt = json.loads(path.read_text(encoding="utf-8-sig"))
            _require(isinstance(receipt, dict), "receipt must be an object")
            run_id = str(receipt.get("run_id") or "")
            _require(path.name == f"{run_id}.json", "receipt filename mismatch")
            _require(run_id in scheduled and run_id not in valid, "unknown or duplicate receipt run")
            record = scheduled[run_id]
            fingerprint, fingerprint_sha256 = _validate_host_fingerprint(receipt.get("host_fingerprint"))
            del fingerprint
            checks = (
                receipt.get("schema") == SAME_HARDWARE_RECEIPT_SCHEMA,
                receipt.get("schema_version") == 1,
                receipt.get("namespace") == namespace,
                receipt.get("campaign_lock_sha256") == campaign_lock_sha256,
                receipt.get("execution_environment_sha256") == execution_environment.get("sha256"),
                receipt.get("task_id") == record.get("task_id"),
                receipt.get("seed") == record.get("seed"),
                receipt.get("arm") == record.get("arm"),
                receipt.get("host_fingerprint_sha256") == fingerprint_sha256,
                fingerprint_sha256 == execution_environment.get("host_fingerprint_sha256"),
                receipt.get("status") == "host_bound_before_execution",
                receipt.get("completion_started") is False,
                bool(str(receipt.get("recorded_at") or "").strip()),
            )
            _require(all(checks), "receipt binding mismatch")
            valid[run_id] = receipt
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"invalid_receipt:{path.name}:{type(exc).__name__}")

    missing_required = sorted(required - set(valid))
    pair_hosts: dict[tuple[str, int], set[str]] = {}
    for receipt in valid.values():
        pair = (str(receipt["task_id"]), int(receipt["seed"]))
        pair_hosts.setdefault(pair, set()).add(str(receipt["host_fingerprint_sha256"]))
    pair_mismatches = sorted(f"{task}:s{seed}" for (task, seed), hosts in pair_hosts.items() if len(hosts) != 1)
    all_valid = not errors and not missing_required and not pair_mismatches
    return {
        "status": "verified" if all_valid else "failed_closed",
        "receipt_schema": SAME_HARDWARE_RECEIPT_SCHEMA,
        "expected_receipts": len(scheduled),
        "observed_valid_receipts": len(valid),
        "required_receipts_now": len(required),
        "missing_required_run_ids": missing_required,
        "invalid_receipts": errors,
        "pair_host_mismatches": pair_mismatches,
        "paired_arms_same_host": not pair_mismatches,
        "all_required_receipts_verified": not missing_required and not errors,
    }


def require_same_hardware_receipt_for_run(
    campaign_dir: str | Path,
    *,
    target_run_id: str,
    execution_environment: str | Path,
    receipts_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless a particular run has a valid pre-execution receipt."""

    campaign = Path(campaign_dir).expanduser().resolve()
    prereg, schedule = load_campaign(campaign)
    lock_sha256 = sha256_file(campaign / "campaign-lock.json")
    environment = _read_execution_environment(
        Path(execution_environment), prereg, campaign_lock_sha256=lock_sha256
    )
    _require(environment["valid"], "execution environment is not fully bound")
    root = Path(receipts_root).expanduser().resolve() if receipts_root is not None else campaign / "same-hardware-receipts"
    audit = verify_same_hardware_receipts(
        schedule,
        namespace=str(prereg["namespace"]),
        campaign_lock_sha256=lock_sha256,
        execution_environment=environment,
        receipts_root=root,
        launched_run_ids=[target_run_id],
    )
    _require(audit["status"] == "verified", "same-hardware receipt verification failed")
    return audit


def _evaluator_info() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("mlebench")
    except importlib.metadata.PackageNotFoundError:
        return {"available": False, "package": "mlebench", "version": None}
    return {"available": True, "package": "mlebench", "version": version}


def _task_readiness(
    task_id: str,
    data_root: Path,
    upstream_root: Path,
    *,
    expected_upstream_commit: str,
    expected_upstream_version: str,
    upstream_commit_override: str | None,
    upstream_version_override: str | None,
    siim_expected: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = resolve_competition(task_id, data_root, require_exists=False, require_private=True)
    required_paths = {
        "competition_root": resolved.competition_root,
        "public_dir": resolved.public_dir,
        "private_dir": resolved.private_dir,
        "sample_submission": resolved.sample_submission_path,
        "answers": resolved.answers_path,
        "prepared_contract": resolved.competition_root / "prepared-contract.json",
    }
    missing = [label for label, path in required_paths.items() if not path.exists()]
    if not resolved.train_paths:
        missing.append("train_inputs")
    if not resolved.test_paths:
        missing.append("test_inputs")
    if not resolved.sample_columns:
        missing.append("sample_submission_header")
    if not resolved.prediction_columns:
        missing.append("prediction_columns")
    missing = sorted(set(missing))
    if missing:
        return {
            "task_id": task_id,
            "status": "blocked",
            "reason": "prepared_contract_missing",
            "missing_contract_fields": missing,
        }
    try:
        verified = verify_prepared_contract(
            required_paths["prepared_contract"],
            resolved.competition_root,
            upstream_root,
            expected_upstream_commit=expected_upstream_commit,
            expected_upstream_version=expected_upstream_version,
            upstream_commit_override=upstream_commit_override,
            upstream_version_override=upstream_version_override,
            siim_expected=siim_expected,
        )
    except (PreparedContractError, OSError, subprocess.SubprocessError, ValueError) as exc:
        return {
            "task_id": task_id,
            "status": "blocked",
            "reason": "prepared_contract_verification_error",
            "missing_contract_fields": [],
            "error_type": type(exc).__name__,
        }
    recomputed = verified["recomputed"]
    if verified["status"] != "verified":
        return {
            "task_id": task_id,
            "status": "blocked",
            "reason": "official_md5_or_inventory_mismatch",
            "missing_contract_fields": [],
            "prepared_contract_sha256": verified["contract_sha256"],
            "contract_matches_recomputed": verified["contract_matches_recomputed"],
            "checks": recomputed["checks"],
            "siim_inventory": recomputed.get("siim_inventory"),
        }
    return {
        "task_id": task_id,
        "status": "ready",
        "reason": "official_upstream_md5_and_prepared_contract_verified",
        "missing_contract_fields": [],
        "metric": resolved.spec.metric,
        "direction": resolved.spec.direction,
        "train_input_count": len(resolved.train_paths),
        "test_input_count": len(resolved.test_paths),
        "prepared_contract_sha256": verified["contract_sha256"],
        "upstream_commit": recomputed["upstream"]["repository_commit"],
        "upstream_version": recomputed["upstream"]["package_version"],
        "official_zip_md5": recomputed["zip"]["actual_md5"],
        "official_public_md5": recomputed["public"]["actual_md5"],
        "official_private_md5": recomputed["private"]["actual_md5"],
        "public_recursive_file_count": recomputed["public"]["recursive_file_count"],
        "public_recursive_bytes": recomputed["public"]["recursive_bytes"],
        "private_recursive_file_count": recomputed["private"]["recursive_file_count"],
        "private_recursive_bytes": recomputed["private"]["recursive_bytes"],
        "siim_inventory": recomputed.get("siim_inventory"),
    }


def _launched_run_ids(campaign: Path, schedule: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    scheduled = {str(item["run_id"]) for item in list(schedule.get("runs") or []) if isinstance(item, dict)}
    run_root = campaign / "runs"
    if not run_root.exists():
        return [], []
    _require(run_root.is_dir() and not run_root.is_symlink(), "campaign runs root is unsafe")
    launched: list[str] = []
    unexpected: list[str] = []
    for path in sorted(run_root.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_dir():
            unexpected.append(path.name)
            continue
        if not any(child.is_file() for child in path.rglob("*")):
            continue
        if path.name in scheduled:
            launched.append(path.name)
        else:
            unexpected.append(path.name)
    return launched, unexpected


def evaluate_readiness(
    campaign_dir: str | Path,
    *,
    canonical_manifest: str | Path,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    upstream_root: str | Path = DEFAULT_UPSTREAM_ROOT,
    execution_environment: str | Path | None = None,
    same_hardware_receipts: str | Path | None = None,
    expected_upstream_commit: str = PINNED_MLEBENCH_COMMIT,
    expected_upstream_version: str = PINNED_MLEBENCH_VERSION,
    upstream_commit_override: str | None = None,
    upstream_version_override: str | None = None,
    siim_expected: Mapping[str, Any] = PINNED_SIIM_INVENTORY,
) -> dict[str, Any]:
    campaign = Path(campaign_dir).expanduser().resolve()
    prereg, schedule = load_campaign(campaign)
    if prereg.get("phase") != "screen" or tuple(prereg.get("tasks") or ()) != SCREEN_TASKS:
        raise ValueError("readiness gate currently requires the frozen six-task screen phase")
    manifest = Path(canonical_manifest).expanduser().resolve()
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("canonical manifest must be a regular file")
    manifest_matches = sha256_file(manifest) == prereg.get("canonical_manifest_sha256")
    root = Path(data_root).expanduser().resolve()
    upstream = Path(upstream_root).expanduser().resolve()
    task_rows = [
        _task_readiness(
            task_id,
            root,
            upstream,
            expected_upstream_commit=expected_upstream_commit,
            expected_upstream_version=expected_upstream_version,
            upstream_commit_override=upstream_commit_override,
            upstream_version_override=upstream_version_override,
            siim_expected=siim_expected,
        )
        for task_id in prereg["tasks"]
    ]
    prepared = sum(row["status"] == "ready" for row in task_rows)
    launched_ids, unexpected_run_entries = _launched_run_ids(campaign, schedule)
    evaluator = _evaluator_info()
    campaign_lock_sha256 = sha256_file(campaign / "campaign-lock.json")
    if execution_environment is None:
        environment = _missing_execution_environment()
    else:
        try:
            environment = _read_execution_environment(
                Path(execution_environment),
                prereg,
                campaign_lock_sha256=campaign_lock_sha256,
            )
        except (OSError, ValueError) as exc:
            environment = _missing_execution_environment(type(exc).__name__)
            environment["present"] = True
    receipt_root = (
        Path(same_hardware_receipts).expanduser().resolve()
        if same_hardware_receipts is not None
        else campaign / "same-hardware-receipts"
    )
    receipt_audit = verify_same_hardware_receipts(
        schedule,
        namespace=str(prereg["namespace"]),
        campaign_lock_sha256=campaign_lock_sha256,
        execution_environment=environment,
        receipts_root=receipt_root,
        launched_run_ids=launched_ids,
    )
    schedule_runs = list(schedule.get("runs") or [])
    checks = {
        "campaign_lock_and_source_hashes_verified": True,
        "canonical_manifest_hash_matches_preregistration": manifest_matches,
        "schedule_cardinality_verified": len(schedule_runs) == int(schedule.get("expected_task_runs") or -1) == 36,
        "six_screen_tasks_official_md5_verified": prepared == len(SCREEN_TASKS),
        "siim_inventory_complete_exact_files_and_bytes": any(
            row.get("task_id") == "siim-isic-melanoma-classification"
            and row.get("status") == "ready"
            and (row.get("siim_inventory") or {}).get("status") == "verified"
            for row in task_rows
        ),
        "upstream_mlebench_evaluator_available_and_pinned": evaluator["available"]
        and evaluator["version"] == expected_upstream_version,
        "execution_environment_namespace_and_lock_bound": environment["namespace_matches"]
        and environment["campaign_lock_matches"],
        "model_gateway_authenticated_no_completion_probe_verified": environment[
            "authenticated_no_completion_probe_verified"
        ],
        "execution_host_fingerprint_verified": environment["host_fingerprint_verified"],
        "same_hardware_receipt_policy_bound": environment["same_hardware_policy_verified"],
        "same_hardware_receipts_verified_for_launched_runs": receipt_audit["all_required_receipts_verified"],
        "paired_baseline_treatment_same_host": receipt_audit["paired_arms_same_host"],
        "provider_and_model_match_preregistration": environment["provider_matches"]
        and environment["model_matches"],
        "no_task_runs_launched_before_readiness": not launched_ids and not unexpected_run_entries,
        "no_private_siim_grader_scheduled": prereg.get("siim_private_grader") == "forbidden",
        "no_siim_identity_reuse": prereg.get("frozen_siim_run") == UNIQUE_SIIM_RUN
        and prereg.get("namespace") != UNIQUE_SIIM_RUN
        and "job90353" not in str(prereg.get("namespace") or "").lower(),
    }
    passed = all(checks.values())
    return {
        "schema": READINESS_SCHEMA,
        "schema_version": 2,
        "created_at": utc_now(),
        "namespace": prereg["namespace"],
        "phase": prereg["phase"],
        "status": "ready" if passed else "failed_closed",
        "expected_task_runs": int(schedule["expected_task_runs"]),
        "launched_task_runs": len(launched_ids),
        "launched_run_ids": launched_ids,
        "unexpected_run_entries": unexpected_run_entries,
        "prepared_tasks": prepared,
        "blocked_tasks": len(SCREEN_TASKS) - prepared,
        "data_root_projection": _safe_projection(root),
        "upstream_root_projection": _safe_projection(upstream),
        "canonical_manifest_sha256": sha256_file(manifest),
        "preregistration_sha256": sha256_file(campaign / "preregistration.json"),
        "schedule_sha256": sha256_file(campaign / "schedule.json"),
        "campaign_lock_sha256": campaign_lock_sha256,
        "checks": checks,
        "evaluator": evaluator,
        "execution_environment": environment,
        "same_hardware_receipts": receipt_audit,
        "tasks": task_rows,
        "boundaries": {
            "llm_calls": 0,
            "completion_requests": 0,
            "grader_calls": 0,
            "task_runs_launched": len(launched_ids),
            "private_siim_grader_calls": 0,
            "kaggle_submissions": 0,
        },
        "claim_boundary": "local preregistration/readiness only; no benchmark score and no official Kaggle result",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CANONICAL_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument("--execution-environment", type=Path)
    parser.add_argument("--same-hardware-receipts", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_readiness(
        args.campaign,
        canonical_manifest=args.canonical_manifest,
        data_root=args.data_root,
        upstream_root=args.upstream_root,
        execution_environment=args.execution_environment,
        same_hardware_receipts=args.same_hardware_receipts,
    )
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = args.campaign / f"readiness-{stamp}.json"
    write_json_exclusive(output.expanduser().resolve(), report)
    rendered = {**report, "output": str(output), "output_sha256": sha256_file(output)}
    print(json.dumps(rendered, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
