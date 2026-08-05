#!/usr/bin/env python3
"""Build hash-bound SIIM workflow ingress from one named HPC allocation.

The HPC campaign and the EvoMind nine-node workflow deliberately have separate
lifecycles.  This module is the narrow bridge between them.  It only derives
workflow inputs from already-collected, hash-verifiable evidence; it never
starts training, reads private labels, executes a grader, or invents metrics.

The recovery chain is intentionally split into four explicit phases:

``preflight``
    Bind the exact GO gate, launch record, isolated runtime, and dataset plan.
``campaign``
    Validate collection/aggregation evidence, stage data/ablation/training,
    resume the same Run, independently recompute checks, then freeze hashes.
``grader``
    Register an externally produced, post-freeze terminal grader result.
``delivery``
    Register the Claim Audit and four finished user deliverables.

Every phase is fail-closed and idempotent.  Existing ingress may be reused only
when its bytes are identical; changed evidence is never silently overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _entry in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from research_os.agent.aibuild_v1 import run_directory  # noqa: E402
from research_os.agent.siim_hpc_workflow import (  # noqa: E402
    ABLATION_SEEDS,
    DELIVERABLE_NAMES,
    FORMAL_SEEDS,
    FROZEN_FILES,
    resume_siim_hpc_research,
)
from research_os.siim_hpc_binding import binding_from_environment  # noqa: E402
from scripts.mlebench_medal_recovery_adapters import (  # noqa: E402
    MIN_SIIM_ABLATION_FOLDS,
    SIIM_LEAKAGE_GROUP_POLICY,
    SIIM_PERCEPTUAL_EDGE_POLICY,
)

BINDING = binding_from_environment()
HPC_JOB_ID = BINDING.job_id
CREDENTIAL_PROFILE = BINDING.credential_profile
JOB_TAG = BINDING.job_tag
TASK_ID = "siim-isic-melanoma-classification"
REMOTE_ROOT = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"
EXPECTED_DATASET_MANIFEST_SHA256 = "8fef392368fcc433f4532129312f3b2ef7118d76cad04dc70f694246c34eff13"
EXPECTED_DATASET = {
    "files": 33_129,
    "bytes": 25_765_345_055,
    "train_images": 28_984,
    "test_images": 4_142,
    "positive_rows": 513,
    "patients": 2_056,
}
EXPECTED_GATE_SCHEMA = "evomind.hpc_gpu_resource_gate.v2"
EXPECTED_PLAN_SCHEMA = "evomind.siim.hpc_campaign_plan.v1"
EXPECTED_RUNTIME_SCHEMA = "evomind.siim.isolated_runtime_verification.v1"
EXPECTED_FORMAL_STATES = {
    "passed",
    "promotion_gate_failed",
    "candidate_ready_confirmation_pending",
    "promotion_gate_passed_confirmation_pending",
}
PROFILE_MAP = {
    "raw_multiview_v1": "raw_multiview",
    "border_multiview_v1": "border_removal",
    "color_multiview_v1": "color_constancy",
    "hair_multiview_v1": "hair_suppression",
    "robust_multiview_v1": "robust_combined_pipeline",
}
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_CAMPAIGN_PARENT = PROJECT_ROOT / "workspace" / "hpc" / f"{JOB_TAG}_siim_campaign"
DEFAULT_RUNTIME_VERIFICATION = (
    PROJECT_ROOT / "workspace" / "hpc" / f"{JOB_TAG}_siim_runtime" / "verify_current.json"
)
DEFAULT_COLLECTED_ROOT = PROJECT_ROOT / "workspace" / "hpc" / "mlebench_remote_ops" / "collected"
DEFAULT_CANDIDATE_PARENT = PROJECT_ROOT / "workspace" / f"siim_{JOB_TAG}" / "candidates"
DEFAULT_SAMPLE_SUBMISSION = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "mlebench_official_data"
    / TASK_ID
    / "prepared"
    / "public"
    / "sample_submission.csv"
)


class SiimWorkflowIngressError(RuntimeError):
    """Raised before any unverified SIIM evidence can advance the workflow."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SiimWorkflowIngressError(message)


def validate_run_id(run_id: str) -> str:
    require(bool(SAFE_RUN_ID.fullmatch(str(run_id))), "invalid SIIM run_id")
    return str(run_id)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, label: str | None = None) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise SiimWorkflowIngressError(f"missing {label or path.name}: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SiimWorkflowIngressError(f"invalid {label or path.name}: {path}") from exc
    require(isinstance(payload, dict), f"{label or path.name} must contain a JSON object")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def is_within(path: Path, root: Path) -> bool:
    path = Path(path).resolve()
    root = Path(root).resolve()
    return path == root or root in path.parents


def require_file(path: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    require(resolved.is_file() and not resolved.is_symlink(), f"missing or unsafe {label}: {resolved}")
    require(resolved.stat().st_size > 0, f"empty {label}: {resolved}")
    return resolved


def file_record(path: Path, *, root: Path | None = None, label: str | None = None) -> dict[str, Any]:
    path = require_file(path, label or Path(path).name)
    if root is not None:
        require(is_within(path, root), f"{label or path.name} escaped its evidence root")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def same_run(payload: Mapping[str, Any], run_id: str, label: str) -> None:
    require(str(payload.get("run_id") or "") == run_id, f"{label} belongs to a different run")


def finite_metric(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SiimWorkflowIngressError(f"{label} is not numeric") from exc
    require(math.isfinite(number) and 0.0 <= number <= 1.0, f"{label} escaped [0,1]")
    return number


def _validate_declared_file(record: Mapping[str, Any], *, roots: Sequence[Path], label: str) -> Path:
    raw_path = str(record.get("local") or record.get("path") or "")
    require(bool(raw_path), f"{label} has no path")
    path = require_file(Path(raw_path), label)
    require(any(is_within(path, root) for root in roots), f"{label} escaped the collected evidence roots")
    require(path.stat().st_size == int(record.get("bytes") or -1), f"{label} byte count changed")
    declared_hash = str(record.get("sha256") or "").lower()
    require(bool(HEX_SHA256.fullmatch(declared_hash)), f"{label} has an invalid SHA-256")
    require(sha256_file(path) == declared_hash, f"{label} SHA-256 changed")
    return path


def _validate_remote_path(path: Any, label: str) -> None:
    candidate = PurePosixPath(str(path or ""))
    root = PurePosixPath(REMOTE_ROOT)
    require(candidate == root or root in candidate.parents, f"{label} escaped the dedicated HPC root")


def _generated_json(directory: Path, name: str, payload: Mapping[str, Any]) -> Path:
    path = directory / name
    write_json_atomic(path, payload)
    return path


def _ingress_manifest_path(run_dir: Path) -> Path:
    return run_dir / "ingress" / "workflow_ingress_manifest.json"


def _load_ingress_manifest(run_dir: Path, run_id: str) -> dict[str, Any]:
    path = _ingress_manifest_path(run_dir)
    if not path.is_file():
        return {
            "schema": "evomind.siim.workflow_ingress_manifest.v1",
            "run_id": run_id,
            "stages": {},
        }
    payload = read_json(path, label="workflow ingress manifest")
    same_run(payload, run_id, "workflow ingress manifest")
    require(payload.get("schema") == "evomind.siim.workflow_ingress_manifest.v1", "wrong ingress manifest schema")
    require(isinstance(payload.get("stages"), dict), "ingress manifest stages are invalid")
    return payload


def stage_verified_files(
    workspace_root: Path,
    run_id: str,
    *,
    stage: str,
    files: Mapping[str, Path],
    source_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Atomically stage a fully validated set without overwriting drifted bytes."""

    run_id = validate_run_id(run_id)
    root = Path(workspace_root).resolve()
    run_dir = run_directory(root, run_id)
    require((run_dir / "request.json").is_file(), "EvoMind Run request is missing")
    require(bool(stage) and "/" not in stage and "\\" not in stage, "unsafe ingress stage name")
    normalized: dict[str, Path] = {}
    staged_records: list[dict[str, Any]] = []
    for relative, raw_source in sorted(files.items()):
        target_rel = Path(relative)
        require(not target_rel.is_absolute() and ".." not in target_rel.parts, "unsafe SIIM ingress target")
        source = require_file(raw_source, f"ingress source {relative}")
        normalized[target_rel.as_posix()] = source
        staged_records.append(
            {
                "path": target_rel.as_posix(),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )

    manifest = _load_ingress_manifest(run_dir, run_id)
    stages = dict(manifest["stages"])
    proposed = {
        "status": "verified_and_staged",
        "source_artifacts": [dict(record) for record in source_artifacts],
        "staged_artifacts": staged_records,
    }
    existing_stage = stages.get(stage)
    if existing_stage is not None:
        require(existing_stage == proposed, f"ingress stage {stage} differs from its existing hash seal")

    ingress_root = run_dir / "ingress"
    ingress_root.mkdir(parents=True, exist_ok=True)
    for record in staged_records:
        destination = ingress_root / record["path"]
        if destination.exists():
            require(destination.is_file(), f"ingress destination is not a file: {record['path']}")
            require(destination.stat().st_size == record["bytes"], f"existing ingress bytes drifted: {record['path']}")
            require(sha256_file(destination) == record["sha256"], f"existing ingress hash drifted: {record['path']}")

    temporary_files: list[tuple[Path, Path]] = []
    try:
        for relative, source in normalized.items():
            destination = ingress_root / relative
            if destination.is_file():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.stage")
            shutil.copyfile(source, temporary)
            require(sha256_file(temporary) == sha256_file(source), f"staging copy changed: {relative}")
            temporary_files.append((temporary, destination))
        for temporary, destination in temporary_files:
            os.replace(temporary, destination)
    finally:
        for temporary, _destination in temporary_files:
            if temporary.exists():
                temporary.unlink()

    stages[stage] = proposed
    manifest["stages"] = stages
    manifest["status"] = "hash_bound"
    manifest["updated_at"] = utc_now()
    write_json_atomic(_ingress_manifest_path(run_dir), manifest)
    return proposed


def _validate_plan(plan: Mapping[str, Any], run_id: str) -> None:
    same_run(plan, run_id, "campaign plan")
    require(plan.get("schema") == EXPECTED_PLAN_SCHEMA, "wrong SIIM campaign plan schema")
    require(
        int(plan.get("job_id") or 0) == HPC_JOB_ID,
        f"campaign plan is not bound to {JOB_TAG}",
    )
    require(
        plan.get("credential_profile") == CREDENTIAL_PROFILE,
        "campaign plan uses the wrong credential profile",
    )
    require(plan.get("remote_root") == REMOTE_ROOT, "campaign plan uses the wrong remote root")
    require(tuple(int(value) for value in plan.get("ablation_seeds") or []) == ABLATION_SEEDS, "ablation seeds changed")
    require(tuple(int(value) for value in plan.get("formal_seeds") or []) == FORMAL_SEEDS, "formal seeds changed")
    require(set(ABLATION_SEEDS).isdisjoint(FORMAL_SEEDS), "formal and ablation seed sets overlap")
    require(plan.get("official_submission") == "forbidden", "official submission is not forbidden")
    require(plan.get("private_grader") == "once_after_candidate_freeze", "private grader policy changed")
    require(plan.get("post_grader_tuning") == "forbidden", "post-grader tuning is not forbidden")
    resource = plan.get("resource_policy") if isinstance(plan.get("resource_policy"), dict) else {}
    require(int(resource.get("memory_limit_mib") or 0) == 55 * 1024, "campaign memory limit changed")
    require(int(resource.get("effective_batch_size") or 0) == 384, "effective batch contract changed")
    require(int(resource.get("max_workers") or 0) <= 8, "DataLoader worker contract changed")
    require(resource.get("other_processes_modified") is False, "campaign planned to modify another process")
    require(int(resource.get("signals_sent") or 0) == 0, "campaign planned a process signal")
    dataset = plan.get("dataset") if isinstance(plan.get("dataset"), dict) else {}
    require(dataset.get("competition_id") == TASK_ID, "campaign dataset changed")
    observed = {
        "files": dataset.get("file_count"),
        "bytes": dataset.get("total_bytes"),
        "train_images": dataset.get("train_images"),
        "test_images": dataset.get("test_images"),
        "positive_rows": dataset.get("positive_rows"),
        "patients": dataset.get("patients"),
    }
    for key, expected in EXPECTED_DATASET.items():
        require(int(observed.get(key) or -1) == expected, f"campaign dataset {key} changed")
    require(
        str(dataset.get("manifest_sha256") or "").lower() == EXPECTED_DATASET_MANIFEST_SHA256,
        "campaign dataset manifest changed",
    )


def _validate_gate(gate: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(gate.get("schema") == EXPECTED_GATE_SCHEMA, "wrong GPU gate schema")
    require(
        gate.get("credential_profile") == CREDENTIAL_PROFILE,
        "GPU gate uses the wrong credential profile",
    )
    require(gate.get("remote_root") == REMOTE_ROOT, "GPU gate uses the wrong remote root")
    require(gate.get("passed") is True and gate.get("read_only_gate_passed") is True, "GPU gate is HOLD")
    require(gate.get("dedicated_root_writable") is True, "dedicated HPC root was not writable")
    require(gate.get("hold_reasons") == [], "GPU gate contains HOLD reasons")
    require(gate.get("other_processes_modified") is False, "GPU gate modified another process")
    require(int(gate.get("signals_sent") or 0) == 0, "GPU gate sent a process signal")
    identity = gate.get("identity") if isinstance(gate.get("identity"), dict) else {}
    require(identity.get("stable") is True, "GPU identity is unstable")
    require(identity.get("expected_host_uuid_bound") is True, "GPU gate host UUID is not bound")
    require(identity.get("expected_gpu_uuid_bound") is True, "GPU gate GPU UUID is not bound")
    require(bool(identity.get("host_uuid")) and bool(identity.get("gpu_uuids")), "GPU identity is incomplete")
    policy = gate.get("policy") if isinstance(gate.get("policy"), dict) else {}
    samples = gate.get("samples") if isinstance(gate.get("samples"), list) else []
    declared = int(policy.get("samples_required") or 0)
    require(declared >= 5 and len(samples) == declared, "GPU gate does not contain five declared samples")
    normalized: list[dict[str, Any]] = []
    first_gpu: dict[str, Any] = {}
    for index, sample in enumerate(samples):
        require(isinstance(sample, dict), f"GPU sample {index} is invalid")
        require(sample.get("eligible") is True, f"GPU sample {index} is not eligible")
        require(sample.get("probe_errors") == [] and sample.get("hold_reasons") == [], f"GPU sample {index} is dirty")
        require(sample.get("other_processes_modified") is False, f"GPU sample {index} modified another process")
        gpus = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
        require(len(gpus) == 1 and isinstance(gpus[0], dict), f"GPU sample {index} inventory changed")
        gpu = gpus[0]
        if not first_gpu:
            first_gpu = dict(gpu)
        require("A800" in str(gpu.get("name") or ""), f"GPU sample {index} is not A800")
        require(int(gpu.get("memory_total_mib") or 0) >= 80_000, f"GPU sample {index} memory inventory is incomplete")
        free_mib = int(gpu.get("memory_free_mib") or 0)
        require(free_mib >= 60 * 1024, f"GPU sample {index} free memory is below the launch gate")
        require(int(gpu.get("utilization_percent") or 0) <= 5, f"GPU sample {index} utilization exceeds the gate")
        apps = sample.get("compute_apps") if isinstance(sample.get("compute_apps"), list) else []
        other_mib = sum(int(item.get("used_memory_mib") or item.get("memory_mib") or 0) for item in apps if isinstance(item, dict))
        require(other_mib <= 8 * 1024, f"GPU sample {index} other-process memory exceeds the gate")
        normalized.append(
            {
                "captured_at": sample.get("captured_at") or sample.get("captured_at_epoch"),
                "free_memory_mb": free_mib,
                "memory_used_mb": int(gpu.get("memory_used_mib") or 0),
                "utilization_percent": int(gpu.get("utilization_percent") or 0),
                "other_process_memory_mb": other_mib,
                "eligible": True,
                "hold_reasons": [],
            }
        )
    return normalized, first_gpu


def stage_preflight_ingress(
    workspace_root: str | Path,
    run_id: str,
    *,
    campaign_dir: str | Path | None = None,
    runtime_verification_path: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Stage a launch-time GO gate and exact dataset profile for one Run."""

    root = Path(workspace_root).resolve()
    run_id = validate_run_id(run_id)
    campaign = Path(
        campaign_dir or (root / "workspace" / "hpc" / f"{JOB_TAG}_siim_campaign" / run_id)
    ).resolve()
    runtime_path = Path(
        runtime_verification_path
        or (root / "workspace" / "hpc" / f"{JOB_TAG}_siim_runtime" / "verify_current.json")
    ).resolve()
    require(campaign.is_dir(), "SIIM campaign directory is missing")
    plan_path = require_file(campaign / "campaign_plan.json", "campaign plan")
    gate_path = require_file(campaign / "gpu_gate.json", "GPU gate")
    deploy_path = require_file(campaign / "deployment.json", "deployment record")
    launch_path = require_file(campaign / "launch.json", "launch record")
    runtime_path = require_file(runtime_path, "isolated runtime verification")
    plan = read_json(plan_path, label="campaign plan")
    gate = read_json(gate_path, label="GPU gate")
    deployment = read_json(deploy_path, label="deployment record")
    launch = read_json(launch_path, label="launch record")
    runtime = read_json(runtime_path, label="isolated runtime verification")
    _validate_plan(plan, run_id)
    samples, first_gpu = _validate_gate(gate)

    same_run(deployment, run_id, "deployment")
    require(deployment.get("schema") == BINDING.schema("deployment"), "wrong deployment schema")
    require(deployment.get("status") == "deployed_not_started", "deployment record is incomplete")
    require(int(deployment.get("job_id") or 0) == HPC_JOB_ID, "deployment job binding changed")
    require(
        deployment.get("credential_profile") == CREDENTIAL_PROFILE,
        "deployment credential profile changed",
    )
    require(deployment.get("remote_root") == REMOTE_ROOT, "deployment remote root changed")
    require(deployment.get("bundle_sha256") == plan.get("bundle_sha256"), "deployment bundle differs from plan")
    require(deployment.get("dataset_uploaded") is False, "campaign redundantly uploaded the SIIM dataset")
    require(deployment.get("other_processes_modified") is False, "deployment modified another process")
    require(int(deployment.get("signals_sent") or 0) == 0, "deployment sent a process signal")
    require("import_smoke=passed" in list(deployment.get("import_smoke") or []), "bundle import smoke did not pass")
    for name, remote_path in (deployment.get("remote") or {}).items():
        _validate_remote_path(remote_path, f"deployment remote path {name}")

    same_run(launch, run_id, "launch")
    require(launch.get("schema") == BINDING.schema("launch"), "wrong launch schema")
    require(launch.get("status") == "started", "campaign was not launched")
    require(int(launch.get("job_id") or 0) == HPC_JOB_ID, "launch job binding changed")
    require(launch.get("credential_profile") == CREDENTIAL_PROFILE, "launch credential profile changed")
    require(launch.get("action") in {"new_supervisor_started", "existing_supervisor_reused"}, "invalid launch action")
    require(int(launch.get("supervisor_pid") or 0) > 0, "launch supervisor PID is missing")
    require(launch.get("bundle_sha256") == plan.get("bundle_sha256"), "launch bundle differs from plan")
    require(launch.get("other_processes_modified") is False, "launch modified another process")
    require(int(launch.get("signals_sent") or 0) == 0, "launch sent a process signal")
    launch_gate = launch.get("gate") if isinstance(launch.get("gate"), dict) else {}
    require(launch_gate.get("passed") is True, "launch did not record a passed freshness check")
    require(launch_gate.get("created_at") == gate.get("created_at"), "launch and current GPU gate are not the same sample set")
    require(float(launch_gate.get("age_seconds") or 0) <= 600, "launch used a stale GPU gate")
    require(launch_gate.get("host_uuid") == (gate.get("identity") or {}).get("host_uuid"), "launch host identity changed")
    require(list(launch_gate.get("gpu_uuids") or []) == list((gate.get("identity") or {}).get("gpu_uuids") or []), "launch GPU identity changed")

    require(runtime.get("schema") == EXPECTED_RUNTIME_SCHEMA, "wrong isolated runtime schema")
    require(runtime.get("status") == "passed", "isolated runtime verification did not pass")
    require(int(runtime.get("job_id") or 0) == HPC_JOB_ID, "isolated runtime job binding changed")
    require(
        runtime.get("credential_profile") == CREDENTIAL_PROFILE,
        "isolated runtime credential profile changed",
    )
    require(runtime.get("requirements_sha256") == plan.get("runtime_requirements_sha256"), "runtime requirements differ from plan")
    require((runtime.get("cuda") or {}).get("available") is True, "isolated runtime CUDA smoke failed")
    require("A800" in str((runtime.get("cuda") or {}).get("device") or ""), "runtime verified a different GPU")
    require((runtime.get("cuda") or {}).get("bf16") is True, "runtime BF16 smoke failed")
    require(runtime.get("catboost_smoke") is True, "runtime CatBoost smoke failed")
    require(runtime.get("other_processes_modified") is False, "runtime setup modified another process")
    require(int(runtime.get("signals_sent") or 0) == 0, "runtime setup sent a process signal")

    dataset = plan["dataset"]
    runtime_payload = {
        "schema": "evomind.siim.hpc_preflight.v1",
        "run_id": run_id,
        "status": "passed",
        "job_id": str(HPC_JOB_ID),
        "credential_profile": CREDENTIAL_PROFILE,
        "remote_root": REMOTE_ROOT,
        "gpu": {
            "name": first_gpu["name"],
            "uuid": first_gpu.get("uuid"),
            "memory_total_mb": int(first_gpu["memory_total_mib"]),
            "identity_bound": True,
        },
        "samples": samples,
        "launch_decision": "GO",
        "runtime_verification": {
            "status": "passed",
            "requirements_sha256": runtime["requirements_sha256"],
            "python": runtime.get("python"),
            "versions": runtime.get("versions"),
            "source_sha256": sha256_file(runtime_path),
        },
        "gate_source_sha256": sha256_file(gate_path),
        "launch_source_sha256": sha256_file(launch_path),
        "other_processes_modified": False,
        "signals_sent": 0,
    }
    dataset_payload = {
        "schema": "evomind.siim.dataset_profile.v1",
        "run_id": run_id,
        "status": "passed",
        "dataset": TASK_ID,
        "counts": {
            "files": int(dataset["file_count"]),
            "bytes": int(dataset["total_bytes"]),
            "train_images": int(dataset["train_images"]),
            "test_images": int(dataset["test_images"]),
            "positive_rows": int(dataset["positive_rows"]),
            "patients": int(dataset["patients"]),
        },
        "positive_rate": int(dataset["positive_rows"]) / int(dataset["train_images"]),
        "manifest_sha256": dataset["manifest_sha256"],
        "complete": True,
        "private_paths_accessed": False,
        "campaign_plan_sha256": sha256_file(plan_path),
    }
    source_records = [
        {**file_record(path, root=campaign), "kind": kind}
        for path, kind in (
            (plan_path, "campaign_plan"),
            (gate_path, "gpu_gate"),
            (deploy_path, "deployment"),
            (launch_path, "launch"),
        )
    ]
    source_records.append({**file_record(runtime_path), "kind": "isolated_runtime_verification"})
    with tempfile.TemporaryDirectory(prefix="siim-preflight-ingress-", dir=campaign) as temporary:
        build = Path(temporary)
        staged = stage_verified_files(
            root,
            run_id,
            stage="preflight",
            files={
                "hpc_preflight.json": _generated_json(build, "hpc_preflight.json", runtime_payload),
                "dataset_profile.json": _generated_json(build, "dataset_profile.json", dataset_payload),
            },
            source_artifacts=source_records,
        )
    # Pre-create the remote evidence layout used by the collection/ablation
    # ingress stage.  The preflight stage owns the campaign workspace shape but
    # does not invent any metrics or evidence files.
    (campaign / "remote_evidence" / "ablation" / "runs" / f"{run_id}_ablation").mkdir(
        parents=True,
        exist_ok=True,
    )
    run = resume_siim_hpc_research(root, run_id) if resume else None
    if run is not None:
        require(run.tasks["hpc_data_preflight"].status == "completed", "EvoMind preflight node did not complete")
    return {
        "schema": "evomind.siim.workflow_ingress_stage_result.v1",
        "run_id": run_id,
        "stage": "preflight",
        "status": "staged",
        "ingress": staged,
        "workflow_status": getattr(run, "status", None),
    }


def _validate_collection_manifest(run_root: Path, expected_run_id: str) -> tuple[dict[str, Any], dict[str, Path]]:
    run_root = Path(run_root).resolve()
    manifest = read_json(run_root / "collection_manifest.json", label=f"collection manifest {expected_run_id}")
    same_run(manifest, expected_run_id, "collection manifest")
    require(manifest.get("schema") == "evomind.mlebench_remote_ops.collection.v1", "wrong collection schema")
    require(manifest.get("passed") is True, f"collection {expected_run_id} did not pass")
    records = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    require(records and int(manifest.get("file_count") or -1) == len(records), f"collection {expected_run_id} is incomplete")
    indexed: dict[str, Path] = {}
    for index, record in enumerate(records):
        require(isinstance(record, dict), f"collection record {index} is invalid")
        path = _validate_declared_file(record, roots=[run_root], label=f"collection {expected_run_id} record {index}")
        require(str(path) not in indexed, f"collection {expected_run_id} repeats a path")
        indexed[str(path)] = path
    return manifest, indexed


def _unique_indexed_file(root: Path, name: str, indexed: Mapping[str, Path], *, prefer_top_level: bool = False) -> Path:
    matches = [path.resolve() for path in Path(root).rglob(name) if path.is_file() and not path.is_symlink()]
    if prefer_top_level:
        top = [path for path in matches if "attempts" not in path.parts]
        if len(top) == 1:
            matches = top
    require(len(matches) == 1, f"expected one {name} below {root}, found {len(matches)}")
    require(str(matches[0]) in indexed, f"{name} was not hash-collected")
    return matches[0]


def validate_candidate_manifest(candidate_root: str | Path, run_id: str) -> dict[str, Path]:
    """Validate the aggregation artifact manifest and return its indexed files."""

    run_id = validate_run_id(run_id)
    root = Path(candidate_root).resolve()
    require(root.is_dir() and not root.is_symlink(), "frozen candidate directory is missing or unsafe")
    manifest = read_json(root / "artifact_manifest.json", label="candidate artifact manifest")
    same_run(manifest, run_id, "candidate artifact manifest")
    require(manifest.get("schema") == "evomind.siim.frozen_artifact_manifest.v1", "wrong candidate manifest schema")
    require(manifest.get("all_sha256_bound") is True, "candidate manifest is not hash bound")
    records = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
    require(records and int(manifest.get("artifact_count") or -1) == len(records), "candidate artifact count mismatch")
    indexed: dict[str, Path] = {}
    for index, record in enumerate(records):
        require(isinstance(record, dict), f"candidate artifact record {index} is invalid")
        path = _validate_declared_file(record, roots=[root], label=f"candidate artifact {index}")
        require(str(path) not in indexed, "candidate manifest repeats an artifact")
        indexed[str(path)] = path
    for required_name in (
        "candidate_submission_withheld.csv",
        "ensemble_oof_predictions.csv",
        "fold_metrics.csv",
        "metrics.json",
        "frozen_plan.json",
        "candidate_freeze.json",
        "independent_verification.json",
    ):
        path = (root / required_name).resolve()
        require(str(path) in indexed, f"candidate manifest is missing {required_name}")
    return indexed


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise SiimWorkflowIngressError(f"invalid {label}: {path}") from exc
    require(len(frame.columns) > 0, f"{label} has no columns")
    return frame


def _verify_aggregate_predictions(
    candidate_root: Path,
    sample_submission: Path,
    metrics: Mapping[str, Any],
    *,
    oof_name: str = "ensemble_oof_predictions.csv",
    fold_metrics_name: str = "fold_metrics.csv",
    submission_name: str = "candidate_submission_withheld.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    oof = _read_csv(candidate_root / oof_name, "ensemble OOF predictions")
    required_oof = {"image_name", "patient_id", "leakage_group", "target", "fold", "probability"}
    require(required_oof.issubset(oof.columns), "ensemble OOF schema is incomplete")
    require(len(oof) == EXPECTED_DATASET["train_images"], "ensemble OOF row count changed")
    require(oof["image_name"].astype(str).nunique() == len(oof), "ensemble OOF identifiers are duplicated")
    truth = pd.to_numeric(oof["target"], errors="coerce").to_numpy(dtype=np.float64)
    probability = pd.to_numeric(oof["probability"], errors="coerce").to_numpy(dtype=np.float64)
    folds = pd.to_numeric(oof["fold"], errors="coerce").to_numpy(dtype=np.float64)
    require(np.isfinite(truth).all() and set(np.unique(truth).tolist()) == {0.0, 1.0}, "OOF targets are invalid")
    require(int(truth.sum()) == EXPECTED_DATASET["positive_rows"], "OOF positive count changed")
    require(np.isfinite(probability).all() and np.logical_and(probability >= 0, probability <= 1).all(), "OOF probabilities are invalid")
    require(np.isfinite(folds).all() and set(folds.astype(int).tolist()) == set(range(5)), "OOF fold coverage changed")
    require(np.array_equal(folds, folds.astype(int)), "OOF folds are not integral")
    patient = oof["patient_id"].fillna("").astype(str).str.strip()
    non_empty_patients = patient[patient.ne("")]
    require(non_empty_patients.nunique() == EXPECTED_DATASET["patients"], "OOF patient count changed")
    patient_fold_counts = pd.DataFrame({"patient": patient, "fold": folds.astype(int)})
    patient_fold_counts = patient_fold_counts.loc[patient_fold_counts["patient"].ne("")]
    patient_overlap = int((patient_fold_counts.groupby("patient")["fold"].nunique() > 1).sum())
    groups = oof["leakage_group"].fillna("").astype(str).str.strip()
    require(groups.ne("").all(), "OOF leakage groups are missing")
    content_overlap = int((pd.DataFrame({"group": groups, "fold": folds.astype(int)}).groupby("group")["fold"].nunique() > 1).sum())
    require(patient_overlap == 0, "patient groups overlap across outer folds")
    require(content_overlap == 0, "content groups overlap across outer folds")

    recomputed = {
        "roc_auc": float(roc_auc_score(truth, probability)),
        "pr_auc": float(average_precision_score(truth, probability)),
        "brier": float(brier_score_loss(truth, probability)),
    }
    for name, value in recomputed.items():
        require(abs(value - finite_metric(metrics.get(name), f"candidate {name}")) <= 1e-10, f"candidate {name} does not match OOF")

    fold_metrics = _read_csv(candidate_root / fold_metrics_name, "fold metrics")
    require({"fold", "roc_auc", "pr_auc", "brier"}.issubset(fold_metrics.columns), "fold metrics schema is incomplete")
    require(set(pd.to_numeric(fold_metrics["fold"]).astype(int).tolist()) == set(range(5)), "fold metrics do not cover 0..4")
    for fold in range(5):
        mask = folds.astype(int) == fold
        row = fold_metrics.loc[pd.to_numeric(fold_metrics["fold"]).astype(int) == fold].iloc[0]
        expected = {
            "roc_auc": float(roc_auc_score(truth[mask], probability[mask])),
            "pr_auc": float(average_precision_score(truth[mask], probability[mask])),
            "brier": float(brier_score_loss(truth[mask], probability[mask])),
        }
        for name, value in expected.items():
            require(abs(float(row[name]) - value) <= 1e-10, f"fold {fold} {name} does not match OOF")

    submission = _read_csv(candidate_root / submission_name, "withheld candidate submission")
    sample = _read_csv(sample_submission, "sample submission")
    require(list(submission.columns) == ["image_name", "target"], "candidate submission schema changed")
    require(list(sample.columns) == ["image_name", "target"], "sample submission schema changed")
    require(len(submission) == EXPECTED_DATASET["test_images"], "candidate submission row count changed")
    require(len(sample) == len(submission), "sample submission row count differs")
    require(submission["image_name"].astype(str).is_unique, "candidate submission identifiers are duplicated")
    require(submission["image_name"].astype(str).tolist() == sample["image_name"].astype(str).tolist(), "candidate submission order differs from sample")
    test_probability = pd.to_numeric(submission["target"], errors="coerce").to_numpy(dtype=np.float64)
    require(np.isfinite(test_probability).all() and np.logical_and(test_probability >= 0, test_probability <= 1).all(), "test probabilities are invalid")
    return oof, fold_metrics, submission, {**recomputed, "patient_group_overlap": patient_overlap, "content_group_overlap": content_overlap}


def _validate_ablation(ablation: Mapping[str, Any], manifest_path: Path) -> tuple[dict[str, Any], str]:
    require(ablation.get("schema") == "evomind.siim_preprocessing_ablation.v1", "wrong preprocessing ablation schema")
    require(ablation.get("passed") is True, "preprocessing ablation did not pass")
    raw_names = [str(value) for value in ablation.get("profile_order") or []]
    require(raw_names == list(PROFILE_MAP), "preprocessing ablation profile order changed")
    seeds = tuple(int(value) for value in ablation.get("evaluation_seeds") or [])
    require(seeds == ABLATION_SEEDS, "preprocessing ablation seeds changed")
    require(int(ablation.get("evaluation_seed_count") or 0) == len(seeds), "ablation seed count changed")
    reported_fold_count = int(ablation.get("fold_count") or 0)
    require(
        reported_fold_count == MIN_SIIM_ABLATION_FOLDS,
        "preprocessing ablation fold count changed",
    )
    require(
        ablation.get("leakage_group_policy") == SIIM_LEAKAGE_GROUP_POLICY,
        "preprocessing ablation leakage-group policy changed",
    )
    require(
        ablation.get("perceptual_edge_policy") == SIIM_PERCEPTUAL_EDGE_POLICY,
        "preprocessing ablation perceptual-edge policy changed",
    )
    require(
        int(ablation.get("score_count_per_profile") or 0)
        == reported_fold_count * len(seeds),
        "preprocessing ablation score count changed",
    )
    require(ablation.get("private_labels_used_for_training") is False, "private labels entered ablation")
    require(ablation.get("official_grader_executed") is False, "grader executed during ablation")
    require(ablation.get("kaggle_submission_executed") is False, "Kaggle submission executed during ablation")
    require(sha256_file(manifest_path) == ablation.get("image_content_manifest_sha256"), "ablation image manifest hash changed")
    raw_records = ablation.get("profiles") if isinstance(ablation.get("profiles"), list) else []
    require(len(raw_records) == len(PROFILE_MAP), "ablation profile records are incomplete")
    records: list[dict[str, Any]] = []
    by_raw: dict[str, Mapping[str, Any]] = {}
    rejected: list[str] = []
    for raw_record in raw_records:
        require(isinstance(raw_record, dict), "ablation profile record is invalid")
        raw_name = str(raw_record.get("profile") or "")
        require(raw_name in PROFILE_MAP and raw_name not in by_raw, "ablation profile identity changed")
        by_raw[raw_name] = raw_record
        seed_fold_auc = raw_record.get("seed_fold_auc") if isinstance(raw_record.get("seed_fold_auc"), dict) else {}
        require(set(seed_fold_auc) == {str(seed) for seed in ABLATION_SEEDS}, f"ablation {raw_name} seed evidence is incomplete")
        for values in seed_fold_auc.values():
            try:
                array = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise SiimWorkflowIngressError(
                    f"ablation {raw_name} fold evidence is invalid"
                ) from exc
            require(
                array.ndim == 1
                and len(array) == reported_fold_count
                and np.isfinite(array).all(),
                f"ablation {raw_name} fold evidence is invalid",
            )
            require(np.logical_and(array >= 0, array <= 1).all(), f"ablation {raw_name} AUC escaped [0,1]")
        canonical = PROFILE_MAP[raw_name]
        normalized = dict(raw_record)
        normalized.update({"source_profile": raw_name, "profile": canonical})
        records.append(normalized)
        if raw_name != "raw_multiview_v1" and raw_record.get("eligible_for_selection") is not True:
            rejected.append(canonical)
    selected_raw = str(ablation.get("selected_profile") or "")
    require(selected_raw in PROFILE_MAP, "ablation selected profile changed")
    selected_record = by_raw[selected_raw]
    selected = PROFILE_MAP[selected_raw]
    mean_gain = 0.0 if selected_raw == "raw_multiview_v1" else float(selected_record.get("mean_gain"))
    worst_delta = 0.0 if selected_raw == "raw_multiview_v1" else float(selected_record.get("worst_fold_delta"))
    seed_passes = int(selected_record.get("seed_stability_pass_count") or len(ABLATION_SEEDS))
    if selected_raw != "raw_multiview_v1":
        require(mean_gain >= 0.0005, "selected preprocessing missed the mean gain gate")
        require(worst_delta >= -0.002, "selected preprocessing missed the worst-fold gate")
        require(seed_passes >= 2, "selected preprocessing missed the seed stability gate")
    return (
        {
            "schema": "evomind.siim.preprocessing_ablation_ingress.v1",
            "status": "passed",
            "seeds": list(ABLATION_SEEDS),
            "fold_count": reported_fold_count,
            "score_count_per_profile": reported_fold_count * len(ABLATION_SEEDS),
            "profiles": records,
            "patient_group_overlap": 0,
            "content_group_overlap": 0,
            "selected_profile": selected,
            "selected_source_profile": selected_raw,
            "decision": {
                "mean_gain": mean_gain,
                "worst_fold_delta": worst_delta,
                "seed_passes": seed_passes,
            },
            "rejected_profiles": rejected,
            "source_image_manifest_sha256": sha256_file(manifest_path),
        },
        selected_raw,
    )


def _validate_formal_seed(
    collected_root: Path,
    run_id: str,
    seed: int,
    selected_profile: str,
    selected_batch_size: int,
) -> dict[str, Any]:
    formal_id = f"{run_id}_s{seed}"
    formal_root = collected_root / formal_id
    collection, indexed = _validate_collection_manifest(formal_root, formal_id)
    competition_root = formal_root / TASK_ID
    result_path = (competition_root / "result.json").resolve()
    require(str(result_path) in indexed, f"formal seed {seed} top-level result was not collected")
    result = read_json(result_path, label=f"formal seed {seed} result")
    require(result.get("competition_id") == TASK_ID, f"formal seed {seed} dataset changed")
    require(result.get("metric") == "roc_auc" and result.get("direction") == "maximize", f"formal seed {seed} metric changed")
    require(result.get("status") in EXPECTED_FORMAL_STATES, f"formal seed {seed} is not complete")
    require(result.get("valid_submission") is True, f"formal seed {seed} submission is invalid")
    require(int((result.get("budget") or {}).get("seed", -1)) == seed, f"formal seed {seed} identity changed")
    require(result.get("official_grader_executed") is False, f"formal seed {seed} executed the grader")
    require(result.get("mle_private_grader_score") is None, f"formal seed {seed} contains a private score")
    require(result.get("kaggle_public_score") is None and result.get("kaggle_private_score") is None, f"formal seed {seed} contains Kaggle scores")
    finite_metric(result.get("cv_score"), f"formal seed {seed} CV score")
    history_path = _unique_indexed_file(competition_root, "siim_training_history.json", indexed)
    nested_path = _unique_indexed_file(competition_root, "siim_nested_patient_folds.json", indexed)
    history = read_json(history_path, label=f"formal seed {seed} training history")
    nested = read_json(nested_path, label=f"formal seed {seed} nested folds")
    require(nested.get("schema") == "evomind.siim_nested_patient_folds.v2", f"formal seed {seed} nested schema changed")
    require(int(nested.get("outer_fold_count") or 0) == 5, f"formal seed {seed} outer folds changed")
    require(int(nested.get("requested_inner_fold_count") or 0) == 3, f"formal seed {seed} inner folds changed")
    require(nested.get("all_inner_folds_aggregated") is True, f"formal seed {seed} did not aggregate inner folds")
    require(nested.get("outer_validation_role") == "final_oof_only", f"formal seed {seed} outer validation role changed")
    performance = history.get("performance") if isinstance(history.get("performance"), dict) else {}
    require(int(performance.get("physical_batch_size") or 0) == selected_batch_size, f"formal seed {seed} batch size changed")
    require(int(performance.get("effective_batch_size") or 0) == 384, f"formal seed {seed} effective batch changed")
    require(int(performance.get("dataloader_workers") or 0) <= 8, f"formal seed {seed} worker count changed")
    require(performance.get("channels_last") is True, f"formal seed {seed} did not use channels_last")
    require(str(performance.get("amp_dtype") or "").lower() in {"bfloat16", "bf16"}, f"formal seed {seed} did not use BF16")
    require(performance.get("preprocessing_profile") == selected_profile, f"formal seed {seed} used a different preprocessing profile")
    preprocessing_gate = performance.get("preprocessing_ablation") if isinstance(performance.get("preprocessing_ablation"), dict) else {}
    require(preprocessing_gate.get("validated") is True, f"formal seed {seed} preprocessing report was not validated")
    return {
        "seed": seed,
        "run_id": formal_id,
        "result": result,
        "result_path": result_path,
        "history": history,
        "history_path": history_path,
        "nested_path": nested_path,
        "collection_path": formal_root / "collection_manifest.json",
        "collection": collection,
    }


def _validate_candidate_freeze(candidate_root: Path, run_id: str, roots: Sequence[Path]) -> dict[str, Any]:
    freeze = read_json(candidate_root / "candidate_freeze.json", label="aggregated candidate freeze")
    same_run(freeze, run_id, "aggregated candidate freeze")
    require(freeze.get("schema") == "evomind.siim.candidate_freeze.v1", "wrong aggregated freeze schema")
    require(freeze.get("status") == "frozen_before_private_grader", "aggregated candidate was not frozen")
    require(tuple(int(value) for value in freeze.get("formal_seeds") or []) == FORMAL_SEEDS, "freeze formal seeds changed")
    require(tuple(int(value) for value in freeze.get("ablation_seeds") or []) == ABLATION_SEEDS, "freeze ablation seeds changed")
    require(freeze.get("seed_sets_disjoint") is True, "freeze seed sets overlap")
    require(freeze.get("candidate_hash_bound") is True, "aggregated candidate is not hash bound")
    require(int(freeze.get("private_grader_execution_count_before_freeze") or 0) == 0, "grader ran before aggregate freeze")
    require(freeze.get("official_submission_executed") is False, "official submission ran before aggregate freeze")
    for name in ("candidate_submission", "ensemble_oof_predictions", "metrics"):
        record = freeze.get(name)
        require(isinstance(record, dict), f"freeze is missing {name}")
        _validate_declared_file(record, roots=roots, label=f"freeze {name}")
    seed_records = freeze.get("seed_records") if isinstance(freeze.get("seed_records"), list) else []
    require([int(item.get("model_seed") or -1) for item in seed_records] == list(FORMAL_SEEDS), "freeze seed records changed")
    for item in seed_records:
        require(item.get("run_id") == f"{run_id}_s{int(item['model_seed'])}", "freeze seed run_id changed")
        for name in ("result", "prediction_bundle", "oof_predictions", "submission"):
            require(isinstance(item.get(name), dict), f"freeze seed record is missing {name}")
            _validate_declared_file(item[name], roots=roots, label=f"freeze seed {item['model_seed']} {name}")
    return freeze


def _campaign_sources(
    root: Path,
    run_id: str,
    campaign_dir: Path,
    candidate_root: Path,
    collected_root: Path,
    sample_submission: Path,
) -> dict[str, Any]:
    plan_path = require_file(campaign_dir / "campaign_plan.json", "campaign plan")
    collection_path = require_file(campaign_dir / "collection.json", "campaign collection")
    plan = read_json(plan_path, label="campaign plan")
    collection = read_json(collection_path, label="campaign collection")
    _validate_plan(plan, run_id)
    same_run(collection, run_id, "campaign collection")
    require(collection.get("schema") == BINDING.schema("collection"), "wrong campaign collection schema")
    require(int(collection.get("job_id") or 0) == HPC_JOB_ID, "campaign collection job binding changed")
    require(
        collection.get("credential_profile") == CREDENTIAL_PROFILE,
        "campaign collection credential profile changed",
    )
    require(collection.get("status") == "collected", "campaign has not been collected")
    require(collection.get("other_processes_modified") is False, "campaign collection modified another process")
    require(int(collection.get("signals_sent") or 0) == 0, "campaign collection sent a process signal")

    evidence_root = Path(collection.get("remote_evidence_root") or (campaign_dir / "remote_evidence")).resolve()
    require(is_within(evidence_root, campaign_dir) and evidence_root.is_dir(), "remote evidence root escaped the campaign")
    remote_plan_path = require_file(evidence_root / "campaign_plan.json", "collected remote campaign plan")
    require(sha256_file(remote_plan_path) == sha256_file(plan_path), "remote and local campaign plans differ")
    state_path = require_file(evidence_root / "campaign_state.json", "campaign state")
    state = read_json(state_path, label="campaign state")
    same_run(state, run_id, "campaign state")
    require(state.get("schema") == BINDING.schema("campaign_state"), "wrong campaign state schema")
    require(int(state.get("job_id") or 0) == HPC_JOB_ID, "campaign state job binding changed")
    require(state.get("credential_profile") == CREDENTIAL_PROFILE, "campaign state credential profile changed")
    require(state.get("status") == "awaiting_collection_and_freeze", "campaign training is not complete")
    require(tuple(int(value) for value in state.get("completed_seeds") or []) == FORMAL_SEEDS, "campaign formal seeds are incomplete")
    require(state.get("other_processes_modified") is False, "campaign modified another process")
    require(int(state.get("signals_sent") or 0) == 0, "campaign sent a process signal")
    selected_profile = str(state.get("selected_profile") or "")
    require(selected_profile in PROFILE_MAP, "campaign selected preprocessing profile changed")
    selected_batch = int(state.get("selected_batch_size") or 0)
    require(selected_batch in (128, 96, 64, 48, 32), "campaign selected batch size is invalid")

    telemetry_path = require_file(evidence_root / "hpc_telemetry.jsonl", "HPC telemetry")
    telemetry_count = 0
    for line_number, line in enumerate(telemetry_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SiimWorkflowIngressError(f"HPC telemetry line {line_number} is invalid") from exc
        require(isinstance(event, dict), f"HPC telemetry line {line_number} is not an object")
        same_run(event, run_id, f"HPC telemetry line {line_number}")
        require(int(event.get("job_id") or 0) == HPC_JOB_ID, f"HPC telemetry line {line_number} job changed")
        require(
            event.get("credential_profile") == CREDENTIAL_PROFILE,
            f"HPC telemetry line {line_number} credential profile changed",
        )
        require(event.get("other_processes_modified") is False, f"HPC telemetry line {line_number} modified another process")
        require(int(event.get("signals_sent") or 0) == 0, f"HPC telemetry line {line_number} sent a signal")
        telemetry_count += 1
    require(telemetry_count > 0, "HPC telemetry is empty")

    ablation_path = require_file(
        evidence_root / "ablation" / "runs" / f"{run_id}_ablation" / "siim_preprocessing_ablation.json",
        "preprocessing ablation report",
    )
    ablation_manifest_path = require_file(
        evidence_root / "ablation" / "runs" / f"{run_id}_ablation" / "siim_image_content_manifest.csv",
        "ablation image content manifest",
    )
    duplicate_path = require_file(
        evidence_root / "ablation" / "runs" / f"{run_id}_ablation" / "siim_duplicate_connected_groups.json",
        "ablation duplicate groups",
    )
    ablation_raw = read_json(ablation_path, label="preprocessing ablation report")
    ablation_payload, ablation_selected_profile = _validate_ablation(ablation_raw, ablation_manifest_path)
    require(ablation_selected_profile == selected_profile, "campaign state and ablation selection differ")
    ablation_payload["run_id"] = run_id
    duplicate = read_json(duplicate_path, label="ablation duplicate groups")
    require(duplicate.get("private_labels_used") is False, "private labels entered duplicate grouping")

    validate_candidate_manifest(candidate_root, run_id)
    metrics_path = require_file(candidate_root / "metrics.json", "candidate metrics")
    metrics = read_json(metrics_path, label="candidate metrics")
    same_run(metrics, run_id, "candidate metrics")
    require(int(metrics.get("private_grader_execution_count") or 0) == 0, "candidate metrics contain a private grader execution")
    require(metrics.get("kaggle_submission_executed") is False, "candidate metrics contain a Kaggle submission")
    require(metrics.get("clinical_diagnosis_claimed") is False, "candidate metrics contain a clinical claim")
    oof, _folds, _submission, recomputed = _verify_aggregate_predictions(candidate_root, sample_submission, metrics)
    independent_path = require_file(candidate_root / "independent_verification.json", "aggregate independent verification")
    independent = read_json(independent_path, label="aggregate independent verification")
    same_run(independent, run_id, "aggregate independent verification")
    require(independent.get("status") == "passed" and independent.get("passed") is True, "aggregate verification did not pass")
    require(int(independent.get("train_rows") or -1) == EXPECTED_DATASET["train_images"], "aggregate verification train count changed")
    require(int(independent.get("test_rows") or -1) == EXPECTED_DATASET["test_images"], "aggregate verification test count changed")
    require(int(independent.get("positive_rows") or -1) == EXPECTED_DATASET["positive_rows"], "aggregate verification positives changed")
    for name in (
        "oof_coverage_exactly_once",
        "patient_content_grouping_bound",
        "train_id_order_verified",
        "test_id_order_verified",
        "submission_schema_verified",
        "formal_ablation_seed_separation_verified",
    ):
        require(independent.get(name) is True, f"aggregate verification check failed: {name}")
    require(independent.get("private_labels_used") is False, "aggregate verification used private labels")
    require(independent.get("official_grader_executed") is False, "aggregate verification executed the grader")
    require(independent.get("kaggle_submission_executed") is False, "aggregate verification executed Kaggle submission")
    require(independent.get("other_processes_modified") is False, "aggregate verification modified another process")
    require(int(independent.get("signals_sent") or 0) == 0, "aggregate verification sent a process signal")
    independently_recomputed = (
        independent.get("recomputed_metrics")
        if isinstance(independent.get("recomputed_metrics"), dict)
        else {}
    )
    for name in ("roc_auc", "pr_auc", "brier"):
        require(
            abs(finite_metric(independently_recomputed.get(name), f"independent {name}") - recomputed[name])
            <= 1e-10,
            f"independent verification {name} differs from OOF",
        )
    require(
        independent.get("candidate_freeze_sha256") == sha256_file(candidate_root / "candidate_freeze.json"),
        "independent verification is bound to a different aggregate freeze",
    )
    _validate_candidate_freeze(candidate_root, run_id, [candidate_root, collected_root])

    formal = [
        _validate_formal_seed(collected_root, run_id, seed, selected_profile, selected_batch)
        for seed in FORMAL_SEEDS
    ]
    declared_seed_reports = collection.get("seed_collections") if isinstance(collection.get("seed_collections"), list) else []
    require([str(item.get("run_id") or "") for item in declared_seed_reports] == [item["run_id"] for item in formal], "campaign collection seed order changed")
    for declared, item in zip(declared_seed_reports, formal, strict=True):
        require(declared == item["collection"], f"campaign collection record differs for seed {item['seed']}")

    patient_values = oof["patient_id"].fillna("").astype(str).str.strip()
    data_audit = {
        "schema": "evomind.siim.data_audit_ingress.v1",
        "run_id": run_id,
        "status": "passed",
        "train_rows": len(oof),
        "test_rows": EXPECTED_DATASET["test_images"],
        "positive_rows": int(pd.to_numeric(oof["target"]).sum()),
        "patients": int(patient_values[patient_values.ne("")].nunique()),
        "patient_group_overlap": int(recomputed["patient_group_overlap"]),
        "content_group_overlap": int(recomputed["content_group_overlap"]),
        "target_in_test_features": False,
        "private_label_access_count": 0,
        "split_policy": "patient_and_content_grouped",
        "checks": {
            "oof_identifiers_unique": True,
            "oof_coverage_exactly_once": True,
            "submission_schema_and_order": True,
            "formal_and_ablation_seeds_disjoint": True,
        },
        "evidence": {
            "ensemble_oof_sha256": sha256_file(candidate_root / "ensemble_oof_predictions.csv"),
            "independent_verification_sha256": sha256_file(independent_path),
            "duplicate_groups_sha256": sha256_file(duplicate_path),
        },
    }
    training_metrics = dict(metrics)
    training_metrics.update(
        {
            "schema": "evomind.siim.training_metrics_ingress.v1",
            "run_id": run_id,
            "mle_private_grader_score": None,
            "source_metrics_sha256": sha256_file(metrics_path),
        }
    )
    training_history = {
        "schema": "evomind.siim.multiseed_training_history.v1",
        "run_id": run_id,
        "status": "completed",
        "formal_seeds": list(FORMAL_SEEDS),
        "selected_preprocessing_profile": PROFILE_MAP[selected_profile],
        "selected_source_preprocessing_profile": selected_profile,
        "selected_batch_size": selected_batch,
        "source_runs": [
            {
                "seed": item["seed"],
                "run_id": item["run_id"],
                "result_sha256": sha256_file(item["result_path"]),
                "training_history_sha256": sha256_file(item["history_path"]),
                "nested_folds_sha256": sha256_file(item["nested_path"]),
                "history": item["history"],
            }
            for item in formal
        ],
        "private_grader_execution_count": 0,
        "official_submission_executed": False,
    }
    training_result = {
        "schema": "evomind.siim.training_result_ingress.v1",
        "run_id": run_id,
        "status": "completed",
        "formal_seeds": list(FORMAL_SEEDS),
        "selected_batch_size": selected_batch,
        "selected_preprocessing_profile": PROFILE_MAP[selected_profile],
        "official_submission_executed": False,
        "private_grader_execution_count": 0,
        "private_label_access_count": 0,
        "candidate_manifest_sha256": sha256_file(candidate_root / "artifact_manifest.json"),
        "aggregate_independent_verification_sha256": sha256_file(independent_path),
    }
    source_records: list[dict[str, Any]] = []
    for path, kind in (
        (plan_path, "campaign_plan"),
        (collection_path, "campaign_collection"),
        (state_path, "campaign_state"),
        (telemetry_path, "hpc_telemetry"),
        (ablation_path, "preprocessing_ablation"),
        (ablation_manifest_path, "ablation_image_manifest"),
        (duplicate_path, "ablation_duplicate_groups"),
        (candidate_root / "artifact_manifest.json", "candidate_artifact_manifest"),
        (candidate_root / "candidate_freeze.json", "aggregate_candidate_freeze"),
        (independent_path, "aggregate_independent_verification"),
        (sample_submission, "sample_submission"),
    ):
        source_records.append({**file_record(path), "kind": kind})
    for item in formal:
        for key, kind in (
            ("collection_path", "formal_collection_manifest"),
            ("result_path", "formal_result"),
            ("history_path", "formal_training_history"),
            ("nested_path", "formal_nested_folds"),
        ):
            source_records.append({**file_record(item[key]), "kind": kind, "formal_seed": item["seed"]})
    return {
        "data_audit": data_audit,
        "ablation": ablation_payload,
        "metrics": training_metrics,
        "training_history": training_history,
        "training_result": training_result,
        "telemetry_path": telemetry_path,
        "candidate_root": candidate_root,
        "sample_submission": sample_submission,
        "source_records": source_records,
    }


def _validate_run_freeze_inputs(run_dir: Path, run_id: str) -> dict[str, Any]:
    """Recompute the assertions represented by Independent Review."""

    dataset = read_json(run_dir / "dataset_profile.json", label="workflow dataset profile")
    same_run(dataset, run_id, "workflow dataset profile")
    require(str(dataset.get("status") or "").lower() == "passed", "workflow dataset profile did not pass")
    counts = dataset.get("counts") if isinstance(dataset.get("counts"), dict) else {}
    for name, expected in EXPECTED_DATASET.items():
        require(int(counts.get(name) or -1) == expected, f"workflow dataset {name} changed")
    require(dataset.get("manifest_sha256") == EXPECTED_DATASET_MANIFEST_SHA256, "workflow dataset manifest changed")
    require(dataset.get("complete") is True, "workflow dataset profile is incomplete")

    audit = read_json(run_dir / "data_audit.json", label="workflow data audit")
    same_run(audit, run_id, "workflow data audit")
    require(str(audit.get("status") or "").lower() == "passed", "workflow data audit did not pass")
    require(int(audit.get("train_rows") or -1) == EXPECTED_DATASET["train_images"], "workflow audit train count changed")
    require(int(audit.get("test_rows") or -1) == EXPECTED_DATASET["test_images"], "workflow audit test count changed")
    require(int(audit.get("positive_rows") or -1) == EXPECTED_DATASET["positive_rows"], "workflow audit positive count changed")
    require(int(audit.get("patients") or -1) == EXPECTED_DATASET["patients"], "workflow audit patient count changed")
    require("patient_group_overlap" in audit and int(audit["patient_group_overlap"]) == 0, "workflow audit reports patient overlap")
    require("content_group_overlap" in audit and int(audit["content_group_overlap"]) == 0, "workflow audit reports content overlap")
    require(audit.get("target_in_test_features") is False, "workflow audit reports target leakage")
    require(int(audit.get("private_label_access_count") or 0) == 0, "workflow audit reports private-label access")

    design = read_json(run_dir / "research_design.json", label="workflow research design")
    same_run(design, run_id, "workflow research design")
    require(design.get("status") == "frozen_before_experiment", "workflow research design is not frozen")
    require(design.get("primary_metric") == "roc_auc", "workflow primary metric changed")
    validation = design.get("validation") if isinstance(design.get("validation"), dict) else {}
    require(int(validation.get("outer_folds") or 0) == 5, "workflow outer-fold design changed")
    require(int(validation.get("inner_folds") or 0) == 3, "workflow inner-fold design changed")
    require(tuple(int(value) for value in design.get("ablation_seeds") or []) == ABLATION_SEEDS, "workflow ablation seeds changed")
    require(tuple(int(value) for value in design.get("formal_seeds") or []) == FORMAL_SEEDS, "workflow formal seeds changed")
    require(design.get("official_submission") == "forbidden", "workflow design permits official submission")

    ablation = read_json(run_dir / "preprocessing_ablation.json", label="workflow preprocessing ablation")
    same_run(ablation, run_id, "workflow preprocessing ablation")
    require(str(ablation.get("status") or "").lower() == "passed", "workflow preprocessing ablation did not pass")
    require(tuple(int(value) for value in ablation.get("seeds") or []) == ABLATION_SEEDS, "workflow ablation evidence seeds changed")
    require(
        "patient_group_overlap" in ablation and int(ablation["patient_group_overlap"]) == 0,
        "workflow ablation reports patient overlap",
    )
    require(
        "content_group_overlap" in ablation and int(ablation["content_group_overlap"]) == 0,
        "workflow ablation reports content overlap",
    )

    metrics = read_json(run_dir / "metrics.json", label="workflow metrics")
    same_run(metrics, run_id, "workflow metrics")
    require(metrics.get("mle_private_grader_score") is None, "workflow metrics contain a pre-freeze private score")
    verified = _verify_aggregate_predictions(
        run_dir,
        run_dir / "sample_submission.csv",
        metrics,
        oof_name="oof_predictions.csv",
        fold_metrics_name="fold_metrics.csv",
        submission_name="submission.csv",
    )[3]
    result = read_json(run_dir / "training_result.json", label="workflow training result")
    same_run(result, run_id, "workflow training result")
    require(str(result.get("status") or "").lower() in {"passed", "completed"}, "workflow training result is incomplete")
    require(tuple(int(value) for value in result.get("formal_seeds") or []) == FORMAL_SEEDS, "workflow formal seed evidence changed")
    require(int(result.get("private_grader_execution_count") or 0) == 0, "workflow training ran the private grader")
    require(int(result.get("private_label_access_count") or 0) == 0, "workflow training accessed private labels")
    require(result.get("official_submission_executed") is False, "workflow training executed official submission")
    return verified


def stage_review_freeze_ingress(
    workspace_root: str | Path,
    run_id: str,
    *,
    source_artifacts: Sequence[Mapping[str, Any]] = (),
    resume: bool = True,
) -> dict[str, Any]:
    """Hash the already-ingested public candidate and stage Independent Review."""

    root = Path(workspace_root).resolve()
    run_id = validate_run_id(run_id)
    run_dir = run_directory(root, run_id)
    verified = _validate_run_freeze_inputs(run_dir, run_id)
    hashes: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for name in FROZEN_FILES:
        path = require_file(run_dir / name, f"review freeze input {name}")
        digest = sha256_file(path)
        hashes[name] = digest
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": digest, "kind": "freeze_input"})
    require(not (run_dir / "private_grader_ledger.json").exists(), "private grader ledger exists before review freeze")
    require(not (run_dir / "private_grader.json").exists(), "private grader result exists before review freeze")
    review = {
        "schema": "evomind.siim.independent_review_ingress.v1",
        "run_id": run_id,
        "status": "review_passed",
        "checks": {
            "patient_group_overlap_zero": True,
            "content_group_overlap_zero": True,
            "oof_coverage_exactly_once": True,
            "submission_schema_and_order": True,
            "private_labels_unavailable_during_training": True,
            "private_grader_not_executed": True,
            "official_submission_not_executed": True,
        },
        "artifact_hashes": hashes,
        "review_scope": "public_train_grouped_oof_and_withheld_test_predictions",
        "source_evidence": records,
        "recomputed_metrics": {
            "roc_auc": verified["roc_auc"],
            "pr_auc": verified["pr_auc"],
            "brier": verified["brier"],
        },
    }
    source_records = [dict(record) for record in source_artifacts] + records
    campaign_parent = root / "workspace" / "hpc" / f"{JOB_TAG}_siim_campaign" / run_id
    campaign_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="siim-review-ingress-", dir=campaign_parent) as temporary:
        path = _generated_json(Path(temporary), "review.json", review)
        staged = stage_verified_files(
            root,
            run_id,
            stage="independent_review",
            files={"review.json": path},
            source_artifacts=source_records,
        )
    run = resume_siim_hpc_research(root, run_id) if resume else None
    if run is not None:
        require(run.tasks["independent_review_freeze"].status == "completed", "Independent Review node did not complete")
        freeze = require_file(run_dir / "candidate_freeze.json", "workflow candidate freeze")
        freeze_payload = read_json(freeze, label="workflow candidate freeze")
        same_run(freeze_payload, run_id, "workflow candidate freeze")
        require(freeze_payload.get("status") == "frozen_before_private_grader", "workflow candidate is not frozen")
        require(freeze_payload.get("tuning_closed") is True, "workflow tuning is not closed")
    return {
        "schema": "evomind.siim.workflow_ingress_stage_result.v1",
        "run_id": run_id,
        "stage": "independent_review",
        "status": "staged",
        "ingress": staged,
        "candidate_freeze_sha256": sha256_file(run_dir / "candidate_freeze.json") if (run_dir / "candidate_freeze.json").is_file() else None,
        "workflow_status": getattr(run, "status", None),
    }


def stage_campaign_ingress(
    workspace_root: str | Path,
    run_id: str,
    *,
    campaign_dir: str | Path | None = None,
    candidate_root: str | Path | None = None,
    collected_root: str | Path | None = None,
    sample_submission: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Stage collected campaign evidence, then independently review and freeze."""

    root = Path(workspace_root).resolve()
    run_id = validate_run_id(run_id)
    campaign = Path(
        campaign_dir or (root / "workspace" / "hpc" / f"{JOB_TAG}_siim_campaign" / run_id)
    ).resolve()
    candidate = Path(
        candidate_root or (root / "workspace" / f"siim_{JOB_TAG}" / "candidates" / run_id)
    ).resolve()
    collected = Path(collected_root or (root / "workspace" / "hpc" / "mlebench_remote_ops" / "collected")).resolve()
    sample = require_file(
        Path(sample_submission or (root / "workspace" / "local_gpu" / "mlebench_official_data" / TASK_ID / "prepared" / "public" / "sample_submission.csv")),
        "sample submission",
    )
    require(campaign.is_dir() and candidate.is_dir() and collected.is_dir(), "campaign evidence roots are incomplete")
    evidence = _campaign_sources(root, run_id, campaign, candidate, collected, sample)
    with tempfile.TemporaryDirectory(prefix="siim-campaign-ingress-", dir=campaign) as temporary:
        build = Path(temporary)
        files = {
            "data_audit.json": _generated_json(build, "data_audit.json", evidence["data_audit"]),
            "preprocessing_ablation.json": _generated_json(build, "preprocessing_ablation.json", evidence["ablation"]),
            "training/metrics.json": _generated_json(build, "metrics.json", evidence["metrics"]),
            "training/fold_metrics.csv": candidate / "fold_metrics.csv",
            "training/oof_predictions.csv": candidate / "ensemble_oof_predictions.csv",
            "training/submission.csv": candidate / "candidate_submission_withheld.csv",
            "training/sample_submission.csv": sample,
            "training/training_history.json": _generated_json(build, "training_history.json", evidence["training_history"]),
            "training/hpc_telemetry.jsonl": evidence["telemetry_path"],
            "training/training_result.json": _generated_json(build, "training_result.json", evidence["training_result"]),
        }
        staged = stage_verified_files(
            root,
            run_id,
            stage="campaign",
            files=files,
            source_artifacts=evidence["source_records"],
        )
    run = resume_siim_hpc_research(root, run_id) if resume else None
    review_result = None
    if run is not None:
        for task_id in ("data_audit", "research_design", "preprocessing_ablation", "full_training"):
            require(run.tasks[task_id].status == "completed", f"EvoMind {task_id} node did not complete")
        require(run.tasks["independent_review_freeze"].status == "failed", "workflow did not stop at Independent Review ingress")
        review_result = stage_review_freeze_ingress(
            root,
            run_id,
            source_artifacts=evidence["source_records"],
            resume=True,
        )
    return {
        "schema": "evomind.siim.workflow_ingress_stage_result.v1",
        "run_id": run_id,
        "stage": "campaign",
        "status": "staged",
        "ingress": staged,
        "review_freeze": review_result,
        "workflow_status": getattr(run, "status", None),
    }


def stage_private_grader_ingress(
    workspace_root: str | Path,
    run_id: str,
    result_path: str | Path,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Bind, but never execute, one externally produced terminal grader result."""

    root = Path(workspace_root).resolve()
    run_id = validate_run_id(run_id)
    run_dir = run_directory(root, run_id)
    freeze_path = require_file(run_dir / "candidate_freeze.json", "workflow candidate freeze")
    freeze = read_json(freeze_path, label="workflow candidate freeze")
    same_run(freeze, run_id, "workflow candidate freeze")
    require(freeze.get("status") == "frozen_before_private_grader", "workflow candidate is not frozen")
    require(freeze.get("tuning_closed") is True, "workflow candidate tuning is not closed")
    frozen_records = freeze.get("artifacts") if isinstance(freeze.get("artifacts"), list) else []
    require(bool(frozen_records), "workflow candidate freeze has no artifacts")
    for index, record in enumerate(frozen_records):
        require(isinstance(record, dict), f"workflow freeze record {index} is invalid")
        relative = Path(str(record.get("path") or ""))
        require(not relative.is_absolute() and ".." not in relative.parts, "workflow freeze path escaped the Run")
        frozen_path = require_file(run_dir / relative, f"workflow freeze artifact {index}")
        require(frozen_path.stat().st_size == int(record.get("bytes") or -1), f"workflow freeze bytes changed: {relative}")
        require(sha256_file(frozen_path) == str(record.get("sha256") or "").lower(), f"workflow freeze hash changed: {relative}")
    result_path = require_file(Path(result_path), "private grader result")
    payload = read_json(result_path, label="private grader result")
    same_run(payload, run_id, "private grader result")
    grader_status = str(payload.get("status") or "").lower()
    require(grader_status in {"passed", "completed", "verified", "failed_closed"}, "private grader is incomplete")
    require(int(payload.get("execution_index") or payload.get("execution_count") or 0) == 1, "private grader execution index is not one")
    require(payload.get("candidate_freeze_sha256") == sha256_file(freeze_path), "private grader is bound to a different freeze")
    require(payload.get("executed_after_freeze") is True, "private grader did not execute after freeze")
    require(payload.get("feedback_used_for_tuning") is False, "private grader feedback entered tuning")
    require(payload.get("official_submission_executed") is False, "private grader executed an official submission")
    if grader_status == "failed_closed":
        require(payload.get("mle_private_grader_score", payload.get("score")) in (None, ""), "failed-closed grader must not contain a score")
        require(str(payload.get("error") or payload.get("failure_reason") or "").strip(), "failed-closed grader lacks failure evidence")
    else:
        finite_metric(payload.get("mle_private_grader_score", payload.get("score")), "private grader score")
    staged = stage_verified_files(
        root,
        run_id,
        stage="private_grader",
        files={"private_grader.json": result_path},
        source_artifacts=[{**file_record(result_path), "kind": "terminal_private_grader_result"}],
    )
    run = resume_siim_hpc_research(root, run_id) if resume else None
    if run is not None:
        require(run.tasks["terminal_private_grader"].status == "completed", "terminal private grader node did not complete")
        ledger = read_json(run_dir / "private_grader_ledger.json", label="private grader ledger")
        require(int(ledger.get("execution_count") or 0) == 1, "private grader ledger count changed")
    return {
        "schema": "evomind.siim.workflow_ingress_stage_result.v1",
        "run_id": run_id,
        "stage": "private_grader",
        "status": "staged",
        "ingress": staged,
        "workflow_status": getattr(run, "status", None),
    }


def _validate_delivery_files(run_dir: Path, deliverables_dir: Path) -> dict[str, Path]:
    html = require_file(deliverables_dir / "research_report.html", "HTML report")
    files = {"deliverables/research_report.html": html}
    submission = _read_csv(run_dir / "submission.csv", "workflow submission")
    for name in DELIVERABLE_NAMES:
        path = require_file(deliverables_dir / name, name)
        if name.endswith(".pdf"):
            require(path.read_bytes()[:5] == b"%PDF-", "report deliverable is not a PDF")
        elif name.endswith("results.csv"):
            result = _read_csv(path, "results CSV")
            require(list(result.columns)[:2] == ["image_name", "target"], "results CSV schema is invalid")
            require(len(result) == EXPECTED_DATASET["test_images"], "results CSV row count changed")
            require(result["image_name"].astype(str).tolist() == submission["image_name"].astype(str).tolist(), "results CSV order differs from frozen submission")
        elif name.endswith(".zip"):
            require(zipfile.is_zipfile(path), f"invalid ZIP deliverable: {name}")
            with zipfile.ZipFile(path) as archive:
                members = [member for member in archive.namelist() if not member.endswith("/")]
                require(bool(members), f"empty ZIP deliverable: {name}")
                if name.endswith("code.zip"):
                    require(any(member.lower().endswith(".py") for member in members), "code ZIP contains no Python source")
                if name.endswith("evidence.zip"):
                    required = {"metrics.json", "review.json", "claim_audit.json"}
                    require(required.issubset({Path(member).name for member in members}), "evidence ZIP is incomplete")
        files[f"deliverables/{name}"] = path
    return files


def stage_delivery_ingress(
    workspace_root: str | Path,
    run_id: str,
    claim_audit_path: str | Path,
    deliverables_dir: str | Path,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Stage a passed Claim Audit plus four already-rendered deliverables."""

    root = Path(workspace_root).resolve()
    run_id = validate_run_id(run_id)
    run_dir = run_directory(root, run_id)
    freeze_path = require_file(run_dir / "candidate_freeze.json", "workflow candidate freeze")
    ledger_path = require_file(run_dir / "private_grader_ledger.json", "private grader ledger")
    ledger = read_json(ledger_path, label="private grader ledger")
    same_run(ledger, run_id, "private grader ledger")
    require(int(ledger.get("execution_count") or 0) == 1, "delivery requires exactly one private grader execution")
    require(int(ledger.get("job_id") or 0) == HPC_JOB_ID, "private grader ledger job binding changed")
    require(
        ledger.get("credential_profile") == CREDENTIAL_PROFILE,
        "private grader ledger credential profile changed",
    )
    claim_source = require_file(Path(claim_audit_path), "Claim Audit")
    raw_claim = read_json(claim_source, label="Claim Audit")
    same_run(raw_claim, run_id, "Claim Audit")
    require(str(raw_claim.get("status") or "").lower() in {"passed", "completed", "verified", "review_passed"}, "Claim Audit did not pass")
    checks = raw_claim.get("checks") if isinstance(raw_claim.get("checks"), dict) else {}
    required_checks = (
        "no_public_leaderboard_claim",
        "no_official_medal_claim",
        "no_clinical_diagnosis_claim",
        "private_grader_not_used_for_tuning",
        "candidate_hashes_unchanged",
        "official_submission_not_executed",
    )
    for name in required_checks:
        require(checks.get(name) is True, f"Claim Audit check failed: {name}")
    delivery_root = Path(deliverables_dir).resolve()
    require(delivery_root.is_dir() and not delivery_root.is_symlink(), "deliverables directory is missing or unsafe")
    files = _validate_delivery_files(run_dir, delivery_root)
    normalized_claim = dict(raw_claim)
    normalized_claim.update(
        {
            "schema": "evomind.siim.claim_audit_ingress.v1",
            "run_id": run_id,
            "status": "passed",
            "candidate_freeze_sha256": sha256_file(freeze_path),
            "private_grader_ledger_sha256": sha256_file(ledger_path),
            "source_claim_audit_sha256": sha256_file(claim_source),
        }
    )
    campaign_parent = root / "workspace" / "hpc" / f"{JOB_TAG}_siim_campaign" / run_id
    campaign_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="siim-delivery-ingress-", dir=campaign_parent) as temporary:
        claim = _generated_json(Path(temporary), "claim_audit.json", normalized_claim)
        staged = stage_verified_files(
            root,
            run_id,
            stage="delivery",
            files={"claim_audit.json": claim, **files},
            source_artifacts=[
                {**file_record(claim_source), "kind": "claim_audit"},
                *[{**file_record(path), "kind": relative} for relative, path in sorted(files.items())],
            ],
        )
    run = resume_siim_hpc_research(root, run_id) if resume else None
    if run is not None:
        require(run.status == "completed", "SIIM workflow did not complete delivery")
        manifest = read_json(run_dir / "artifact_manifest.json", label="final artifact manifest")
        require(manifest.get("status") == "verified", "final artifact manifest is not verified")
    return {
        "schema": "evomind.siim.workflow_ingress_stage_result.v1",
        "run_id": run_id,
        "stage": "delivery",
        "status": "staged",
        "ingress": staged,
        "workflow_status": getattr(run, "status", None),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-id", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="stage GO gate, runtime, and dataset evidence")
    preflight.add_argument("--campaign-dir", type=Path)
    preflight.add_argument("--runtime-verification", type=Path)
    preflight.add_argument("--no-resume", action="store_true")

    campaign = subparsers.add_parser("campaign", help="stage collected training evidence and freeze the candidate")
    campaign.add_argument("--campaign-dir", type=Path)
    campaign.add_argument("--candidate-root", type=Path)
    campaign.add_argument("--collected-root", type=Path)
    campaign.add_argument("--sample-submission", type=Path)
    campaign.add_argument("--no-resume", action="store_true")

    grader = subparsers.add_parser("grader", help="stage one externally executed post-freeze grader result")
    grader.add_argument("--result", type=Path, required=True)
    grader.add_argument("--no-resume", action="store_true")

    delivery = subparsers.add_parser("delivery", help="stage Claim Audit and four completed downloads")
    delivery.add_argument("--claim-audit", type=Path, required=True)
    delivery.add_argument("--deliverables-dir", type=Path, required=True)
    delivery.add_argument("--no-resume", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "preflight":
        result = stage_preflight_ingress(
            args.workspace_root,
            args.run_id,
            campaign_dir=args.campaign_dir,
            runtime_verification_path=args.runtime_verification,
            resume=not args.no_resume,
        )
    elif args.command == "campaign":
        result = stage_campaign_ingress(
            args.workspace_root,
            args.run_id,
            campaign_dir=args.campaign_dir,
            candidate_root=args.candidate_root,
            collected_root=args.collected_root,
            sample_submission=args.sample_submission,
            resume=not args.no_resume,
        )
    elif args.command == "grader":
        result = stage_private_grader_ingress(
            args.workspace_root,
            args.run_id,
            args.result,
            resume=not args.no_resume,
        )
    else:
        result = stage_delivery_ingress(
            args.workspace_root,
            args.run_id,
            args.claim_audit,
            args.deliverables_dir,
            resume=not args.no_resume,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
