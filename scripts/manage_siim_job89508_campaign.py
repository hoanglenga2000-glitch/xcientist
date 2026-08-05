#!/usr/bin/env python3
"""Deploy, gate, launch, inspect, collect, and freeze the SIIM HPC campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shlex
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from paramiko import SSHException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for entry in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from research_agent_workstation.server.core.gpu_credentials import (  # noqa: E402
    ALLOWED_GPU_REMOTE_ROOT,
    connect_ssh,
    load_gpu_ssh_config,
    verify_job_container_identity,
)
from research_os.siim_hpc_binding import binding_from_environment  # noqa: E402
from scripts import aggregate_siim_multiseed_candidate as aggregator  # noqa: E402
from scripts import mlebench_remote_ops as remote_ops  # noqa: E402
from scripts.prepare_siim_job89508_runtime import (  # noqa: E402
    DEFAULT_REQUIREMENTS,
    runtime_paths,
)

BINDING = binding_from_environment()
HPC_JOB_ID = BINDING.job_id
CREDENTIAL_PROFILE = BINDING.credential_profile
JOB_TAG = BINDING.job_tag
BASELINE_FORMAL_SEEDS = (43, 44, 45)
R2_FORMAL_SEEDS = (46, 47, 48)
DEFAULT_RUN_ID = os.environ.get(
    "EVOMIND_SIIM_RUN_ID",
    (
        "evomind_siim_isic_a800_20260729_223613"
        if JOB_TAG == "job89508"
        else f"evomind_siim_isic_a800_{JOB_TAG}"
    ),
)
COMPETITION = "siim-isic-melanoma-classification"
REMOTE_DATA_ROOT = f"{ALLOWED_GPU_REMOTE_ROOT}/mlebench_official_data"
REMOTE_PUBLIC_ROOT = f"{REMOTE_DATA_ROOT}/{COMPETITION}/prepared/public"
REMOTE_OFFICIAL_SOURCE = f"{ALLOWED_GPU_REMOTE_ROOT}/mle-bench"
REMOTE_TORCH_HOME = f"{ALLOWED_GPU_REMOTE_ROOT}/mlebench_model_cache/torch"
REMOTE_CAMPAIGN_PARENT = f"{ALLOWED_GPU_REMOTE_ROOT}/siim_{JOB_TAG}/campaigns"
REMOTE_BUNDLE_PARENT = f"{ALLOWED_GPU_REMOTE_ROOT}/siim_{JOB_TAG}/bundles"
REMOTE_RUNTIME_PARENT = f"{ALLOWED_GPU_REMOTE_ROOT}/siim_{JOB_TAG}/runtime"
REMOTE_PRIVATE_GRADER_PARENT = f"{ALLOWED_GPU_REMOTE_ROOT}/siim_{JOB_TAG}/private_grader"
REMOTE_SHORT_TEMP_PARENT = f"{ALLOWED_GPU_REMOTE_ROOT}/.t"
LOCAL_ROOT = PROJECT_ROOT / "workspace" / "hpc" / f"{JOB_TAG}_siim_campaign"
LOCAL_COLLECTED_ROOT = remote_ops.LOCAL_CONTROL_ROOT / "collected"
LOCAL_SAMPLE_SUBMISSION = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "mlebench_official_data"
    / COMPETITION
    / "prepared"
    / "public"
    / "sample_submission.csv"
)
SOURCE_BINDINGS = (
    (PROJECT_ROOT / "scripts" / "run_siim_job89508_campaign.py", "scripts/run_siim_job89508_campaign.py"),
    (PROJECT_ROOT / "scripts" / "run_siim_preprocessing_ablation.py", "scripts/run_siim_preprocessing_ablation.py"),
    (PROJECT_ROOT / "scripts" / "run_mlebench_lite_full.py", "scripts/run_mlebench_lite_full.py"),
    (PROJECT_ROOT / "scripts" / "run_mlebench_lite_wave0.py", "scripts/run_mlebench_lite_wave0.py"),
    (PROJECT_ROOT / "scripts" / "mlebench_medal_recovery_adapters.py", "scripts/mlebench_medal_recovery_adapters.py"),
    (PROJECT_ROOT / "scripts" / "mlebench_wave2_adapters.py", "scripts/mlebench_wave2_adapters.py"),
    (PROJECT_ROOT / "scripts" / "russian_transliteration.py", "scripts/russian_transliteration.py"),
    (PROJECT_ROOT / "scripts" / "probe_siim_candidate_batch.py", "scripts/probe_siim_candidate_batch.py"),
    (PROJECT_ROOT / "src" / "research_os" / "mlebench_phase_a.py", "src/research_os/mlebench_phase_a.py"),
    (PROJECT_ROOT / "src" / "research_os" / "siim_hpc_binding.py", "src/research_os/siim_hpc_binding.py"),
)
COLLECT_NAMES = {
    "manifest.json",
    "phase_a_audit.json",
    "environment.json",
    "checkpoint.json",
    "results_current.json",
    "summary.json",
    "result.json",
    "submission.csv",
    "submission_validation.json",
    "run.log",
    "promotion_gate.json",
    "siim_fold_ensemble.npz",
    "siim_oof_predictions.csv",
    "siim_test_components.csv",
    "siim_training_history.json",
    "siim_nested_patient_folds.json",
    "siim_duplicate_connected_groups.json",
    "siim_image_content_manifest.csv",
    "siim_artifact_manifest.json",
    "siim_preprocessing_ablation_gate.json",
    "siim_harmonization_boundary_hotfix.json",
    "siim_resume_contract.json",
    "resume_contract.json",
    "runtime_budget.json",
    "supersession_record.json",
}
COLLECTION_IO_TIMEOUT_SECONDS = 60


class CampaignManagerError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_json_new_or_same(path: Path, payload: Mapping[str, Any], *, label: str) -> None:
    """Create immutable Run-scoped control JSON or verify identical bytes."""

    encoded = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != encoded:
            raise CampaignManagerError(f"existing {label} differs; refusing overwrite")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise CampaignManagerError(f"{label} appeared during creation") from exc


def configure_collection_timeout(sftp: Any) -> None:
    """Bound stalled SFTP reads so the same Run can reconnect and retry."""

    get_channel = getattr(sftp, "get_channel", None)
    if not callable(get_channel):
        return
    channel = get_channel()
    settimeout = getattr(channel, "settimeout", None)
    if callable(settimeout):
        settimeout(COLLECTION_IO_TIMEOUT_SECONDS)


def ensure_remote(path: str) -> str:
    candidate = PurePosixPath(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise CampaignManagerError("remote path escaped the dedicated root")
    normalized = posixpath.normpath(candidate.as_posix())
    root = posixpath.normpath(ALLOWED_GPU_REMOTE_ROOT)
    if normalized != root and not normalized.startswith(root + "/"):
        raise CampaignManagerError("remote path escaped the dedicated root")
    return normalized


def validate_run_id(run_id: str) -> str:
    return remote_ops.validate_run_id(run_id)


def validate_sha256(value: object, *, label: str) -> str:
    digest = str(value or "").strip()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise CampaignManagerError(f"{label} is not a lowercase SHA-256")
    return digest


def build_remote_storage_policy(run_id: str) -> dict[str, Any]:
    """Bind every campaign-controlled remote write to the operator's root."""

    selected_run = validate_run_id(run_id)
    return {
        "schema": "evomind.siim.dedicated_remote_storage.v1",
        "mode": "dedicated_remote_root_only",
        "allowed_root": ALLOWED_GPU_REMOTE_ROOT,
        "dataset_root": REMOTE_DATA_ROOT,
        "official_source_root": REMOTE_OFFICIAL_SOURCE,
        "torch_home": REMOTE_TORCH_HOME,
        "campaign_root": f"{REMOTE_CAMPAIGN_PARENT}/{selected_run}",
        "bundle_parent": REMOTE_BUNDLE_PARENT,
        "runtime_parent": REMOTE_RUNTIME_PARENT,
        "private_grader_parent": REMOTE_PRIVATE_GRADER_PARENT,
        "short_temp_parent": REMOTE_SHORT_TEMP_PARENT,
        "home_cache_temp_isolated": True,
        "external_remote_writes": "forbidden",
    }


def validate_campaign_plan_identity(
    plan: Mapping[str, Any], run_id: str
) -> dict[str, Any]:
    """Validate immutable identity/boundary fields for legacy or corrected plans."""

    selected_run = validate_run_id(run_id)
    if plan.get("schema") != "evomind.siim.hpc_campaign_plan.v1":
        raise CampaignManagerError("campaign plan schema changed")
    if str(plan.get("run_id") or "") != selected_run:
        raise CampaignManagerError("campaign plan belongs to a different run")
    if int(plan.get("job_id") or 0) != HPC_JOB_ID:
        raise CampaignManagerError("campaign plan job binding changed")
    if plan.get("credential_profile") != CREDENTIAL_PROFILE:
        raise CampaignManagerError("campaign plan credential profile changed")
    if plan.get("remote_root") != ALLOWED_GPU_REMOTE_ROOT:
        raise CampaignManagerError("campaign plan remote root changed")
    validate_sha256(plan.get("bundle_sha256"), label="campaign bundle")
    validate_sha256(
        plan.get("runtime_requirements_sha256"),
        label="campaign runtime requirements",
    )
    formal_runs = plan.get("formal_runs")
    evolution_protocol = plan.get("evolution_protocol")
    formal_seeds = (
        R2_FORMAL_SEEDS if isinstance(evolution_protocol, dict) else BASELINE_FORMAL_SEEDS
    )
    expected_formal = [
        {"seed": seed, "run_id": f"{selected_run}_s{seed}"}
        for seed in formal_seeds
    ]
    if formal_runs != expected_formal:
        raise CampaignManagerError("campaign formal Run bindings changed")
    if plan.get("official_submission") != "forbidden":
        raise CampaignManagerError("campaign official-submission boundary changed")
    if evolution_protocol is not None:
        validate_evolution_protocol(plan, selected_run)
    budget = plan.get("budget_hours") if isinstance(plan.get("budget_hours"), dict) else {}
    legacy_budget = budget == {
        "ablation": 4,
        "formal_training": 18,
        "delivery": 2,
        "total": 24,
    }
    corrected_budget = budget == {
        "ablation": 4,
        "formal_training": 72,
        "delivery": 2,
        "total": 78,
    }
    if not (legacy_budget or corrected_budget):
        raise CampaignManagerError("campaign budget is neither legacy nor corrected")
    return dict(plan)


def validate_evolution_protocol(plan: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    protocol = plan.get("evolution_protocol")
    lineage_basis = plan.get("lineage_basis")
    if not isinstance(protocol, dict) or not isinstance(lineage_basis, dict):
        raise CampaignManagerError("evolution protocol or lineage basis is missing")
    if protocol.get("candidate_id") != "r2_foldwise_rank_channel_consensus_v1":
        raise CampaignManagerError("unsupported SIIM evolution protocol")
    frozen = protocol.get("frozen_inputs") if isinstance(protocol.get("frozen_inputs"), dict) else {}
    if tuple(int(value) for value in frozen.get("fresh_formal_seeds") or ()) != R2_FORMAL_SEEDS:
        raise CampaignManagerError("R2 fresh formal seeds changed")
    if frozen.get("preprocessing_profile") != "raw_multiview":
        raise CampaignManagerError("R2 preprocessing profile changed")
    weights = (
        (protocol.get("score_definition") or {}).get("weights")
        if isinstance(protocol.get("score_definition"), dict)
        else None
    )
    if not isinstance(weights, dict) or {
        "seed_final_foldrank": float(weights.get("seed_final_foldrank", -1)),
        "pure_image": float(weights.get("pure_image", -1)),
        "image_metadata_fusion": float(weights.get("image_metadata_fusion", -1)),
        "lesion_focus": float(weights.get("lesion_focus", -1)),
        "metadata_catboost": float(weights.get("metadata_catboost", -1)),
    } != {
        "seed_final_foldrank": 0.5,
        "pure_image": 0.3,
        "image_metadata_fusion": 0.2,
        "lesion_focus": 0.0,
        "metadata_catboost": 0.0,
    }:
        raise CampaignManagerError("R2 fixed fusion weights changed")
    if tuple(int(value) for value in plan.get("formal_seeds") or ()) != R2_FORMAL_SEEDS:
        raise CampaignManagerError("R2 campaign formal seeds changed")
    if lineage_basis.get("child_run_id") != run_id:
        raise CampaignManagerError("R2 lineage basis belongs to another child Run")
    parent_id = str(lineage_basis.get("parent_run_id") or "")
    validate_run_id(parent_id)
    if parent_id == run_id:
        raise CampaignManagerError("R2 child Run equals its parent")
    for name in (
        "constraint_supersession_sha256",
        "parent_immutable_manifest_sha256",
        "requested_change_sha256",
        "evolution_protocol_sha256",
    ):
        validate_sha256(lineage_basis.get(name), label=f"R2 {name}")
    preprocessing = plan.get("preprocessing_evidence")
    if not isinstance(preprocessing, dict) or preprocessing.get("mode") != "parent_frozen_reuse":
        raise CampaignManagerError("R2 parent preprocessing evidence is missing")
    if preprocessing.get("parent_run_id") != parent_id:
        raise CampaignManagerError("R2 preprocessing parent changed")
    for name in (
        "report_sha256",
        "image_content_manifest_sha256",
        "duplicate_groups_sha256",
    ):
        validate_sha256(preprocessing.get(name), label=f"R2 preprocessing {name}")
    training = plan.get("candidate_training")
    expected_training = {
        "image_size": 384,
        "epochs": 8,
        "full_backbone": "convnext_small",
        "lesion_backbone": "efficientnet_v2_s",
        "learning_rate": 0.0003,
        "metadata_iterations": 700,
        "effective_batch_size": 384,
        "max_workers": 8,
        "memory_limit_mib": 55 * 1024,
        "batch_probe": [128, 96, 64, 48, 32],
    }
    if training != expected_training:
        raise CampaignManagerError("R2 base-training configuration changed")
    return dict(protocol)


def validate_campaign_plan(plan: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    """Require the user-selected corrected contract for every mutating stage."""

    validated = validate_campaign_plan_identity(plan, run_id)
    if plan.get("budget_policy") != "user_selected_A_corrected_full_closure":
        raise CampaignManagerError("campaign corrected budget policy changed")
    if plan.get("budget_hours") != {
        "ablation": 4,
        "formal_training": 72,
        "delivery": 2,
        "total": 78,
    }:
        raise CampaignManagerError("campaign corrected formal budget changed")
    if plan.get("corrected_protocol_version") != "siim_exact_duplicate_grouping_v2":
        raise CampaignManagerError("campaign corrected scientific protocol changed")
    if plan.get("remote_storage_policy") != build_remote_storage_policy(run_id):
        raise CampaignManagerError("campaign dedicated remote-storage policy changed")
    return validated


def validate_gate_binding(gate: Mapping[str, Any]) -> dict[str, Any]:
    if gate.get("schema") != remote_ops.EXPECTED_GATE_SCHEMA:
        raise CampaignManagerError("GPU gate schema changed")
    if gate.get("credential_profile") != CREDENTIAL_PROFILE:
        raise CampaignManagerError("GPU gate credential profile changed")
    if gate.get("remote_root") != ALLOWED_GPU_REMOTE_ROOT:
        raise CampaignManagerError("GPU gate remote root changed")
    return dict(gate)


def file_record(local: Path, relative: str) -> dict[str, Any]:
    local = local.resolve()
    if not local.is_file():
        raise FileNotFoundError(local)
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise CampaignManagerError("unsafe bundle relative path")
    return {
        "local": str(local),
        "path": relative_path.as_posix(),
        "bytes": local.stat().st_size,
        "sha256": sha256_file(local),
    }


def source_records() -> list[dict[str, Any]]:
    return [file_record(local, relative) for local, relative in SOURCE_BINDINGS]


def bundle_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    public = [
        {"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]}
        for item in records
    ]
    digest = hashlib.sha256(
        json.dumps(public, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": BINDING.schema("bundle"),
        "created_at": utc_now(),
        "bundle_sha256": digest,
        "files": public,
        "official_submission_enabled": False,
        "private_grader_in_training_enabled": False,
    }


def build_campaign_plan(
    run_id: str,
    manifest: Mapping[str, Any],
    requirements_sha256: str,
    *,
    evolution_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = validate_run_id(run_id)
    if evolution_spec is not None:
        if evolution_spec.get("schema") != "evomind.siim.evolution_campaign_spec.v1":
            raise CampaignManagerError("evolution campaign spec schema changed")
        if evolution_spec.get("status") != "frozen":
            raise CampaignManagerError("evolution campaign spec is not frozen")
        if evolution_spec.get("run_id") != run_id:
            raise CampaignManagerError("evolution campaign spec belongs to another Run")
        formal_seeds = R2_FORMAL_SEEDS
    else:
        formal_seeds = BASELINE_FORMAL_SEEDS
    formal_runs = [{"seed": seed, "run_id": f"{run_id}_s{seed}"} for seed in formal_seeds]
    plan = {
        "schema": "evomind.siim.hpc_campaign_plan.v1",
        "created_at": utc_now(),
        "run_id": run_id,
        "job_id": HPC_JOB_ID,
        "credential_profile": CREDENTIAL_PROFILE,
        "remote_root": ALLOWED_GPU_REMOTE_ROOT,
        "remote_storage_policy": build_remote_storage_policy(run_id),
        "bundle_sha256": manifest["bundle_sha256"],
        "runtime_requirements_sha256": requirements_sha256,
        "dataset": {
            "competition_id": COMPETITION,
            "manifest_sha256": "8fef392368fcc433f4532129312f3b2ef7118d76cad04dc70f694246c34eff13",
            "file_count": 33_129,
            "total_bytes": 25_765_345_055,
            "train_images": 28_984,
            "test_images": 4_142,
            "positive_rows": 513,
            "patients": 2_056,
        },
        "ablation_seeds": [40, 41, 42],
        "formal_seeds": list(formal_seeds),
        "formal_runs": formal_runs,
        "batch_probe": [128, 96, 64, 48, 32],
        "resource_policy": {
            "memory_limit_mib": 55 * 1024,
            "effective_batch_size": 384,
            "max_workers": 8,
            "minimum_free_prelaunch_mib": 60 * 1024,
            "pause_free_memory_mib": 12 * 1024,
            "other_process_memory_limit_mib": 8 * 1024,
            "signals_sent": 0,
            "other_processes_modified": False,
        },
        "budget_hours": {
            "ablation": 4,
            "formal_training": 72,
            "delivery": 2,
            "total": 78,
        },
        "budget_policy": "user_selected_A_corrected_full_closure",
        "original_budget_hours": {
            "ablation": 4,
            "formal_training": 18,
            "delivery": 2,
            "total": 24,
        },
        "corrected_protocol_version": "siim_exact_duplicate_grouping_v2",
        "evaluation": {"primary_metric": "roc_auc", "bootstrap_samples": 2_000},
        "historical_thresholds": {
            "historical_private_score": 0.92165,
            "bronze": 0.9370,
            "silver": 0.9401,
            "gold": 0.9455,
            "reference_only": True,
        },
        "official_submission": "forbidden",
        "private_grader": "once_after_candidate_freeze",
        "post_grader_tuning": "forbidden",
        "clinical_claim": "medical_imaging_research_benchmark_only",
    }
    if evolution_spec is not None:
        plan.update(
            {
                "evolution_protocol": dict(evolution_spec["evolution_protocol"]),
                "lineage_basis": dict(evolution_spec["lineage_basis"]),
                "preprocessing_evidence": dict(
                    evolution_spec["preprocessing_evidence"]
                ),
                "candidate_training": dict(evolution_spec["candidate_training"]),
                "evolution_spec_sha256": str(
                    evolution_spec.get("evolution_spec_sha256") or ""
                ),
            }
        )
        validate_evolution_protocol(plan, run_id)
    return plan


def local_paths(run_id: str) -> dict[str, Path]:
    root = LOCAL_ROOT / validate_run_id(run_id)
    return {
        "root": root,
        "plan": root / "campaign_plan.json",
        "manifest": root / "bundle_manifest.json",
        "deploy": root / "deployment.json",
        "gate": root / "gpu_gate.json",
        "launch": root / "launch.json",
        "status": root / "status.json",
        "collect": root / "collection.json",
        "aggregate": root / "aggregation.json",
    }


def remote_paths(run_id: str, bundle_sha256: str) -> dict[str, str]:
    campaign = ensure_remote(f"{REMOTE_CAMPAIGN_PARENT}/{validate_run_id(run_id)}")
    bundle = ensure_remote(
        f"{REMOTE_BUNDLE_PARENT}/{validate_sha256(bundle_sha256, label='bundle')}"
    )
    requirements_sha256 = sha256_file(DEFAULT_REQUIREMENTS.resolve())
    runtime = runtime_paths(requirements_sha256)
    environment_root = ensure_remote(f"{campaign}/runtime_environment")
    short_temp = ensure_remote(
        f"{REMOTE_SHORT_TEMP_PARENT}/{hashlib.sha256(run_id.encode('utf-8')).hexdigest()[:12]}"
    )
    return {
        "campaign": campaign,
        "bundle": bundle,
        "plan": ensure_remote(f"{campaign}/campaign_plan.json"),
        "gate": ensure_remote(f"{campaign}/gpu_gate.json"),
        "state": ensure_remote(f"{campaign}/campaign_state.json"),
        "supervisor_log": ensure_remote(f"{campaign}/logs/supervisor.log"),
        "runtime_python": ensure_remote(f"{runtime['venv']}/bin/python"),
        "runtime_verification": ensure_remote(runtime["verification"]),
        "environment_root": environment_root,
        "environment_home": ensure_remote(f"{environment_root}/home"),
        "environment_cache": ensure_remote(f"{environment_root}/cache"),
        "environment_tmp": short_temp,
    }


def remote_environment(remote: Mapping[str, str]) -> dict[str, str]:
    """Return a complete remote environment with no implicit home/cache writes."""

    cache = ensure_remote(str(remote["environment_cache"]))
    temporary = ensure_remote(str(remote["environment_tmp"]))
    return {
        "HOME": ensure_remote(str(remote["environment_home"])),
        "TMPDIR": temporary,
        "TEMP": temporary,
        "TMP": temporary,
        "XDG_CACHE_HOME": ensure_remote(f"{cache}/xdg"),
        "HF_HOME": ensure_remote(f"{cache}/huggingface"),
        "TORCH_HOME": REMOTE_TORCH_HOME,
        "PIP_CACHE_DIR": ensure_remote(f"{cache}/pip"),
        "MPLCONFIGDIR": ensure_remote(f"{cache}/matplotlib"),
        "NUMBA_CACHE_DIR": ensure_remote(f"{cache}/numba"),
        "CUDA_CACHE_PATH": ensure_remote(f"{cache}/cuda"),
        "TRITON_CACHE_DIR": ensure_remote(f"{cache}/triton"),
        "CUPY_CACHE_DIR": ensure_remote(f"{cache}/cupy"),
        "JOBLIB_TEMP_FOLDER": temporary,
        "PYTHONPYCACHEPREFIX": ensure_remote(f"{cache}/python"),
        "KAGGLE_CONFIG_DIR": ensure_remote(f"{remote['environment_root']}/kaggle_disabled"),
    }


def connect() -> tuple[Any, Any]:
    config = load_gpu_ssh_config(strict_named_profile=True)
    if config.credential_profile != CREDENTIAL_PROFILE:
        raise CampaignManagerError(
            f"SIIM campaign requires the named {CREDENTIAL_PROFILE} profile"
        )
    client = connect_ssh(config, timeout=30)
    try:
        identity = verify_job_container_identity(
            client,
            config,
            expected_job_id=HPC_JOB_ID,
            expected_root=ALLOWED_GPU_REMOTE_ROOT,
            expected_gpu_name_fragment="A800",
            minimum_gpu_memory_mib=80_000,
        )
    except Exception:
        client.close()
        raise
    setattr(client, "_evomind_container_identity", identity)
    return client, config


def run_remote(client: Any, command: str, *, timeout: int = 120) -> tuple[int, str, str]:
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode("utf-8", "replace")
    error = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), output, error


def verify_remote_path_chain(
    sftp: Any,
    path: str,
    *,
    require_leaf_regular: bool = False,
) -> str:
    """Reject symlinked or non-directory parents before any SFTP operation."""

    normalized = ensure_remote(path)
    root = PurePosixPath(ALLOWED_GPU_REMOTE_ROOT)
    candidate = PurePosixPath(normalized)
    relative = candidate.relative_to(root)
    current = root
    deepest = root
    for index, part in enumerate(relative.parts):
        current /= part
        try:
            metadata = sftp.lstat(current.as_posix())
        except OSError:
            break
        deepest = current
        if stat.S_ISLNK(metadata.st_mode):
            raise CampaignManagerError("remote path contains a symbolic link")
        is_leaf = index == len(relative.parts) - 1
        if not is_leaf and not stat.S_ISDIR(metadata.st_mode):
            raise CampaignManagerError("remote path parent is not a directory")
        if is_leaf and require_leaf_regular and not stat.S_ISREG(metadata.st_mode):
            raise CampaignManagerError("remote artifact is not a regular file")
    resolved = ensure_remote(str(sftp.normalize(deepest.as_posix())))
    if resolved != deepest.as_posix():
        raise CampaignManagerError("remote path realpath changed inside the dedicated root")
    return normalized


def remote_sha256(client: Any, path: str) -> str:
    path = ensure_remote(path)
    with client.open_sftp() as sftp:
        verify_remote_path_chain(sftp, posixpath.dirname(path))
        try:
            metadata = sftp.lstat(path)
        except OSError:
            return ""
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise CampaignManagerError("remote hash target is not a regular non-link file")
        verify_remote_path_chain(sftp, path, require_leaf_regular=True)
    code, output, _error = run_remote(
        client,
        f"if [ -f {shlex.quote(path)} ]; then sha256sum -- {shlex.quote(path)} | cut -d' ' -f1; fi",
    )
    return output.strip().splitlines()[-1] if code == 0 and output.strip() else ""


def upload_atomic(client: Any, local: Path, remote: str) -> str:
    expected = sha256_file(local)
    if remote_sha256(client, remote) == expected:
        return "reused"
    parent = posixpath.dirname(remote)
    with client.open_sftp() as sftp:
        verify_remote_path_chain(sftp, parent)
        verify_remote_path_chain(sftp, remote)
    code, output, error = run_remote(
        client,
        f"umask 077; mkdir -p -m 700 -- {shlex.quote(parent)}",
    )
    if code:
        raise CampaignManagerError(f"remote directory creation failed: {error[-200:] or output[-200:]}")
    partial = f"{remote}.part.{os.getpid()}"
    with client.open_sftp() as sftp:
        verify_remote_path_chain(sftp, parent)
        verify_remote_path_chain(sftp, remote)
        sftp.put(str(local), partial)
        sftp.chmod(partial, 0o600)
        try:
            sftp.remove(remote)
        except OSError:
            pass
        sftp.rename(partial, remote)
    if remote_sha256(client, remote) != expected:
        raise CampaignManagerError("remote file hash differs after upload")
    return "uploaded"


def _load_evolution_spec(path: str | Path | None, run_id: str) -> dict[str, Any] | None:
    if path is None:
        return None
    child_id = validate_run_id(run_id)
    rescission_path = (
        PROJECT_ROOT
        / "workspace"
        / "siim_evolution_control"
        / child_id
        / "constraint_rescission.json"
    )
    if rescission_path.is_file():
        rescission = json.loads(rescission_path.read_text(encoding="utf-8"))
        if (
            not isinstance(rescission, dict)
            or rescission.get("status") != "effective"
            or rescission.get("child_run_id") != child_id
            or (rescission.get("active_policy") or {}).get(
                "new_candidate_run_allowed"
            )
            is not False
        ):
            raise CampaignManagerError("evolution reservation rescission is invalid")
        raise CampaignManagerError("evolution reservation was rescinded")
    selected = Path(path).expanduser().resolve()
    project = PROJECT_ROOT.resolve()
    try:
        selected.relative_to(project)
    except ValueError as exc:
        raise CampaignManagerError("evolution spec escaped the project workspace") from exc
    if not selected.is_file():
        raise CampaignManagerError("evolution spec is missing")
    payload = json.loads(selected.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampaignManagerError("evolution spec must contain an object")
    if payload.get("run_id") != validate_run_id(run_id):
        raise CampaignManagerError("evolution spec belongs to another Run")
    declared = str(payload.get("evolution_spec_sha256") or "")
    unsigned = dict(payload)
    unsigned.pop("evolution_spec_sha256", None)
    computed = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if declared != computed:
        raise CampaignManagerError("evolution spec self-hash changed")
    return payload


def _prepare_once(
    run_id: str,
    *,
    evolution_spec_path: str | Path | None = None,
) -> dict[str, Any]:
    records = source_records()
    manifest = bundle_manifest(records)
    requirements_sha256 = sha256_file(DEFAULT_REQUIREMENTS.resolve())
    evolution_spec = _load_evolution_spec(evolution_spec_path, run_id)
    plan = build_campaign_plan(
        run_id,
        manifest,
        requirements_sha256,
        evolution_spec=evolution_spec,
    )
    local = local_paths(run_id)
    manifest_exists = local["manifest"].is_file()
    plan_exists = local["plan"].is_file()
    if manifest_exists != plan_exists:
        raise CampaignManagerError("partial local campaign reservation exists")
    if manifest_exists:
        persisted_manifest = json.loads(local["manifest"].read_text(encoding="utf-8"))
        persisted_plan = json.loads(local["plan"].read_text(encoding="utf-8"))
        expected_files = [
            {key: record[key] for key in ("path", "bytes", "sha256")}
            for record in records
        ]
        if (
            persisted_manifest.get("bundle_sha256") != manifest.get("bundle_sha256")
            or persisted_manifest.get("files") != expected_files
        ):
            raise CampaignManagerError("existing bundle manifest differs; refusing overwrite")
        comparable_persisted = dict(persisted_plan)
        comparable_expected = dict(plan)
        comparable_persisted.pop("created_at", None)
        comparable_expected.pop("created_at", None)
        if comparable_persisted != comparable_expected:
            raise CampaignManagerError("existing campaign plan differs; refusing overwrite")
        manifest = persisted_manifest
        plan = persisted_plan
    else:
        write_json_new_or_same(local["manifest"], manifest, label="bundle manifest")
        write_json_new_or_same(local["plan"], plan, label="campaign plan")
    remote = remote_paths(run_id, str(manifest["bundle_sha256"]))
    isolated_environment = remote_environment(remote)
    client, config = connect()
    container_identity = dict(
        getattr(client, "_evomind_container_identity", {}) or {}
    )
    try:
        uploads = {}
        for record in records:
            target = ensure_remote(f"{remote['bundle']}/{record['path']}")
            uploads[record["path"]] = upload_atomic(client, Path(record["local"]), target)
        uploads["bundle_manifest.json"] = upload_atomic(
            client, local["manifest"], ensure_remote(f"{remote['bundle']}/bundle_manifest.json")
        )
        remote_plan_hash = remote_sha256(client, remote["plan"])
        local_plan_hash = sha256_file(local["plan"])
        if remote_plan_hash and remote_plan_hash != local_plan_hash:
            raise CampaignManagerError("existing remote campaign plan differs; refusing overwrite")
        uploads["campaign_plan.json"] = upload_atomic(client, local["plan"], remote["plan"])
        preparation_directories = (
            remote["campaign"],
            f"{remote['campaign']}/logs",
            f"{remote['campaign']}/formal_runs",
            *isolated_environment.values(),
        )
        with client.open_sftp() as sftp:
            for directory in preparation_directories:
                verify_remote_path_chain(sftp, directory)
        code, output, error = run_remote(
            client,
            "umask 077; mkdir -p -m 700 -- "
            + " ".join(
                shlex.quote(value)
                for value in preparation_directories
            ),
        )
        if code:
            raise CampaignManagerError(f"remote campaign preparation failed: {error[-200:] or output[-200:]}")
        environment_prefix = " ".join(
            f"{name}={shlex.quote(value)}"
            for name, value in isolated_environment.items()
        )
        import_command = (
            f"cd -- {shlex.quote(remote['bundle'])} && umask 077 && "
            "env -u KAGGLE_USERNAME -u KAGGLE_KEY "
            f"{environment_prefix} "
            f"EVOMIND_SIIM_HPC_JOB_ID={HPC_JOB_ID} "
            f"EVOMIND_HPC_CREDENTIAL_PROFILE={shlex.quote(CREDENTIAL_PROFILE)} "
            f"EVOMIND_SIIM_RUN_ID={shlex.quote(run_id)} "
            f"PYTHONNOUSERSITE=1 PYTHONPATH={shlex.quote(remote['bundle'] + ':' + remote['bundle'] + '/scripts:' + remote['bundle'] + '/src')} "
            f"CUDA_VISIBLE_DEVICES='' {shlex.quote(remote['runtime_python'])} -c "
            + shlex.quote(
                "import scripts.run_siim_job89508_campaign;"
                "import scripts.run_siim_preprocessing_ablation;"
                "import scripts.run_mlebench_lite_full;"
                "print('import_smoke=passed')"
            )
        )
        code, output, error = run_remote(client, import_command, timeout=180)
        if code:
            raise CampaignManagerError(f"content-addressed bundle import failed: {error[-500:] or output[-500:]}")
    finally:
        client.close()
    payload = {
        "schema": BINDING.schema("deployment"),
        "created_at": utc_now(),
        "status": "deployed_not_started",
        "run_id": run_id,
        "job_id": HPC_JOB_ID,
        "credential_profile": config.credential_profile,
        "remote_root": ALLOWED_GPU_REMOTE_ROOT,
        "bundle_sha256": manifest["bundle_sha256"],
        "remote": remote,
        "remote_environment": isolated_environment,
        "container_identity_gate": container_identity,
        "source_uploads": uploads,
        "import_smoke": output.strip().splitlines(),
        "dataset_uploaded": False,
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    write_json(local["deploy"], payload)
    return payload


def prepare(
    run_id: str,
    *,
    attempts: int = 4,
    evolution_spec_path: str | Path | None = None,
) -> dict[str, Any]:
    if attempts < 1:
        raise ValueError("prepare attempts must be positive")
    last_error: Exception | None = None
    for index in range(attempts):
        try:
            payload = (
                _prepare_once(run_id, evolution_spec_path=evolution_spec_path)
                if evolution_spec_path is not None
                else _prepare_once(run_id)
            )
            payload["connection_attempt"] = index + 1
            write_json(local_paths(run_id)["deploy"], payload)
            return payload
        except (EOFError, OSError, SSHException) as exc:
            last_error = exc
            if index + 1 < attempts:
                time.sleep(5 * (index + 1))
    assert last_error is not None
    raise CampaignManagerError(
        f"content-addressed deployment connection failed after {attempts} attempts"
    ) from last_error


def _local_campaign_record(
    run_id: str,
    path: str | Path | None,
    *,
    default: Path,
    label: str,
) -> Path:
    if path is None:
        return default
    root = local_paths(run_id)["root"].resolve()
    selected = Path(path).expanduser().resolve()
    if selected.parent != root:
        raise CampaignManagerError(f"{label} must be a direct child of the campaign directory")
    return selected


def gate(
    run_id: str,
    *,
    interval_seconds: int = 15,
    output_path: str | Path | None = None,
    publish_default: bool = True,
) -> dict[str, Any]:
    client, config = connect()
    try:
        payload = remote_ops.sample_gpu_idle_gate(
            samples_required=5,
            interval_seconds=interval_seconds,
            min_free_memory_mib=60 * 1024,
            max_other_process_memory_mib=8 * 1024,
            max_memory_growth_mib=256,
            max_utilization_percent=5,
            config=config,
            client=client,
        )
    finally:
        client.close()
    validate_gate_binding(payload)
    local = local_paths(run_id)
    target = _local_campaign_record(
        run_id,
        output_path,
        default=local["gate"],
        label="GPU gate record",
    )
    write_json(target, payload)
    if publish_default:
        write_json(remote_ops.DEFAULT_GATE_REPORT, payload)
    return payload


def publish_status(run_id: str) -> dict[str, Any]:
    local = local_paths(run_id)
    if not local["gate"].is_file():
        raise CampaignManagerError("GPU gate evidence is missing")
    gate_payload = json.loads(local["gate"].read_text(encoding="utf-8"))
    validate_gate_binding(gate_payload)
    run_dir = PROJECT_ROOT / "workspace" / "evomind_runs" / validate_run_id(run_id)
    if not run_dir.is_dir():
        raise CampaignManagerError("EvoMind Run directory is missing")
    samples = []
    for sample in gate_payload.get("samples") or []:
        gpus = sample.get("gpus") if isinstance(sample, dict) else []
        gpu = gpus[0] if isinstance(gpus, list) and gpus else {}
        compute_apps = sample.get("compute_apps") if isinstance(sample, dict) else []
        other_memory = sum(
            int(item.get("used_memory_mib") or 0)
            for item in compute_apps or []
            if isinstance(item, dict)
        )
        samples.append(
            {
                "captured_at": sample.get("captured_at"),
                "free_memory_mb": int(gpu.get("memory_free_mib") or 0),
                "memory_used_mb": int(gpu.get("memory_used_mib") or 0),
                "utilization_percent": int(gpu.get("utilization_percent") or 0),
                "other_process_memory_mb": other_memory,
                "eligible": sample.get("eligible") is True,
                "hold_reasons": list(sample.get("hold_reasons") or []),
            }
        )
    first_gpu = {}
    raw_samples = gate_payload.get("samples") or []
    if raw_samples and isinstance(raw_samples[0], dict):
        raw_gpus = raw_samples[0].get("gpus") or []
        if raw_gpus and isinstance(raw_gpus[0], dict):
            first_gpu = raw_gpus[0]
    passed = gate_payload.get("passed") is True
    runtime = {
        "schema": "evomind.siim.hpc_preflight.v1",
        "run_id": run_id,
        "status": "passed" if passed else "hold",
        "job_id": HPC_JOB_ID,
        "credential_profile": CREDENTIAL_PROFILE,
        "remote_root": ALLOWED_GPU_REMOTE_ROOT,
        "gpu": {
            "name": first_gpu.get("name", "NVIDIA A800-SXM4-80GB"),
            "memory_total_mb": int(first_gpu.get("memory_total_mib") or 81_920),
            "uuid_bound": bool((gate_payload.get("identity") or {}).get("gpu_uuids")),
        },
        "samples": samples,
        "launch_decision": "GO" if passed else "HOLD",
        "hold_reasons": list(gate_payload.get("hold_reasons") or []),
        "runtime_verification": "passed",
        "other_processes_modified": False,
        "signals_sent": 0,
        "updated_at": utc_now(),
    }
    dataset = {
        "schema": "evomind.siim.dataset_profile.v1",
        "run_id": run_id,
        "status": "passed",
        "dataset": COMPETITION,
        "counts": {
            "files": 33_129,
            "bytes": 25_765_345_055,
            "train_images": 28_984,
            "test_images": 4_142,
            "positive_rows": 513,
            "patients": 2_056,
        },
        "positive_rate": 513 / 28_984,
        "manifest_sha256": "8fef392368fcc433f4532129312f3b2ef7118d76cad04dc70f694246c34eff13",
        "complete": True,
        "private_paths_accessed": False,
        "updated_at": utc_now(),
    }
    write_json(run_dir / "hpc_runtime.json", runtime)
    write_json(run_dir / "dataset_profile.json", dataset)
    payload = {
        "schema": "evomind.siim.run_status_publication.v1",
        "run_id": run_id,
        "status": "published",
        "launch_decision": runtime["launch_decision"],
        "run_hpc_runtime": str((run_dir / "hpc_runtime.json").resolve()),
        "run_dataset_profile": str((run_dir / "dataset_profile.json").resolve()),
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    write_json(local["root"] / "published_status.json", payload)
    return payload


def launch(
    run_id: str,
    *,
    max_gate_age: int = 600,
    gate_path: str | Path | None = None,
    launch_record_path: str | Path | None = None,
) -> dict[str, Any]:
    local = local_paths(run_id)
    if not local["plan"].is_file() or not local["manifest"].is_file():
        raise CampaignManagerError("prepare must complete before launch")
    plan = json.loads(local["plan"].read_text(encoding="utf-8"))
    validate_campaign_plan(plan, run_id)
    manifest = json.loads(local["manifest"].read_text(encoding="utf-8"))
    if manifest.get("schema") != BINDING.schema("bundle"):
        raise CampaignManagerError("bundle manifest allocation binding changed")
    if manifest.get("bundle_sha256") != plan.get("bundle_sha256"):
        raise CampaignManagerError("campaign plan and bundle manifest differ")
    selected_gate = _local_campaign_record(
        run_id,
        gate_path,
        default=local["gate"],
        label="GPU gate record",
    )
    selected_launch = _local_campaign_record(
        run_id,
        launch_record_path,
        default=local["launch"],
        label="launch record",
    )
    selected_gate_payload = json.loads(selected_gate.read_text(encoding="utf-8"))
    validate_gate_binding(selected_gate_payload)
    gate_validation = remote_ops.validate_gate_report(selected_gate, max_age_seconds=max_gate_age)
    remote = remote_paths(run_id, str(manifest["bundle_sha256"]))
    isolated_environment = remote_environment(remote)
    client, _config = connect()
    container_identity = dict(
        getattr(client, "_evomind_container_identity", {}) or {}
    )
    try:
        with client.open_sftp() as sftp:
            try:
                with sftp.open(remote["runtime_verification"], "r") as handle:
                    verification = json.loads(handle.read().decode("utf-8"))
            except OSError as exc:
                raise CampaignManagerError("isolated runtime verification is missing") from exc
        if verification.get("status") != "passed":
            raise CampaignManagerError("isolated runtime verification has not passed")
        if verification.get("schema") != "evomind.siim.isolated_runtime_verification.v1":
            raise CampaignManagerError("isolated runtime verification schema changed")
        if int(verification.get("job_id") or 0) != HPC_JOB_ID:
            raise CampaignManagerError("isolated runtime job binding changed")
        if verification.get("credential_profile") != CREDENTIAL_PROFILE:
            raise CampaignManagerError("isolated runtime credential profile changed")
        if verification.get("requirements_sha256") != plan["runtime_requirements_sha256"]:
            raise CampaignManagerError("isolated runtime and campaign requirements differ")
        upload_atomic(client, selected_gate, remote["gate"])
        state = {}
        with client.open_sftp() as sftp:
            try:
                with sftp.open(remote["state"], "r") as handle:
                    state = json.loads(handle.read().decode("utf-8"))
            except OSError:
                pass
        prior_pid = int(state.get("supervisor_pid") or 0)
        active = False
        if prior_pid > 0:
            code, output, _error = run_remote(
                client,
                f"if [ -r /proc/{prior_pid}/cmdline ] && tr '\\0' ' ' </proc/{prior_pid}/cmdline | grep -F -- "
                f"{shlex.quote('run_siim_job89508_campaign.py')} >/dev/null "
                f"&& tr '\\0' ' ' </proc/{prior_pid}/cmdline | grep -F -- "
                f"{shlex.quote(remote['campaign'])} >/dev/null; then echo yes; fi",
            )
            active = code == 0 and output.strip() == "yes"
        if active:
            action = "existing_supervisor_reused"
            pid = prior_pid
        else:
            command = [
                "env",
                "-u", "KAGGLE_USERNAME",
                "-u", "KAGGLE_KEY",
                *(f"{name}={value}" for name, value in isolated_environment.items()),
                "PYTHONNOUSERSITE=1",
                (
                    "PYTHONPATH="
                    f"{remote['bundle']}:{remote['bundle']}/scripts:{remote['bundle']}/src"
                ),
                f"EVOMIND_SIIM_HPC_JOB_ID={HPC_JOB_ID}",
                f"EVOMIND_HPC_CREDENTIAL_PROFILE={CREDENTIAL_PROFILE}",
                f"EVOMIND_SIIM_RUN_ID={run_id}",
                remote["runtime_python"],
                f"{remote['bundle']}/scripts/run_siim_job89508_campaign.py",
                "--plan", remote["plan"],
                "--bundle-root", remote["bundle"],
                "--campaign-root", remote["campaign"],
                "--runtime-python", remote["runtime_python"],
                "--data-root", REMOTE_DATA_ROOT,
                "--official-source-root", REMOTE_OFFICIAL_SOURCE,
                "--torch-home", REMOTE_TORCH_HOME,
            ]
            shell_command = " ".join(shlex.quote(value) for value in command)
            launch_command = (
                f"cd -- {shlex.quote(remote['campaign'])} && umask 077 && "
                f"nohup nice -n 10 {shell_command} >{shlex.quote(remote['supervisor_log'])} "
                f"2>&1 </dev/null & echo $!"
            )
            code, output, error = run_remote(client, launch_command)
            if code or not output.strip():
                raise CampaignManagerError(f"campaign launch failed: {error[-300:] or output[-300:]}")
            pid = int(output.strip().splitlines()[-1])
            action = "new_supervisor_started"
            time.sleep(3)
            code, output, _error = run_remote(
                client,
                f"if [ -r /proc/{pid}/cmdline ] && tr '\\0' ' ' </proc/{pid}/cmdline | grep -F -- "
                f"{shlex.quote('run_siim_job89508_campaign.py')} >/dev/null "
                f"&& tr '\\0' ' ' </proc/{pid}/cmdline | grep -F -- "
                f"{shlex.quote(remote['campaign'])} >/dev/null; then echo yes; fi",
            )
            if code or output.strip() != "yes":
                raise CampaignManagerError("campaign supervisor exited before publishing state")
    finally:
        client.close()
    payload = {
        "schema": BINDING.schema("launch"),
        "created_at": utc_now(),
        "run_id": run_id,
        "job_id": HPC_JOB_ID,
        "credential_profile": CREDENTIAL_PROFILE,
        "status": "started",
        "action": action,
        "supervisor_pid": pid,
        "gate": gate_validation,
        "bundle_sha256": manifest["bundle_sha256"],
        "remote_environment": isolated_environment,
        "container_identity_gate": container_identity,
        "local_gate_path": str(selected_gate),
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    write_json(selected_launch, payload)
    return payload


def status(run_id: str) -> dict[str, Any]:
    local = local_paths(run_id)
    plan = json.loads(local["plan"].read_text(encoding="utf-8"))
    validate_campaign_plan_identity(plan, run_id)
    remote = remote_paths(run_id, str(plan["bundle_sha256"]))
    source = (
        "import json,pathlib;"
        f"p=pathlib.Path({remote['state']!r});"
        f"l=pathlib.Path({remote['supervisor_log']!r});"
        "s=json.loads(p.read_text()) if p.is_file() else {'status':'missing'};"
        "pid=int(s.get('supervisor_pid') or -1);"
        "q=pathlib.Path('/proc')/str(pid)/'status';"
        "tail=(l.read_bytes()[-12000:].decode('utf-8','replace') if l.is_file() else '');"
        "print(json.dumps({'state':s,'process_exists':q.is_file(),'log_tail':tail,"
        "'signals_sent':0,'other_processes_modified':False}))"
    )
    client, _config = connect()
    try:
        code, output, error = run_remote(
            client,
            f"python3 -c {shlex.quote(source)}",
        )
    finally:
        client.close()
    if code:
        raise CampaignManagerError(f"campaign status failed: {error[-300:] or output[-300:]}")
    payload = json.loads(output.strip().splitlines()[-1])
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    if state.get("status") != "missing":
        if state.get("schema") != BINDING.schema("campaign_state"):
            raise CampaignManagerError("remote campaign state schema changed")
        if state.get("run_id") != run_id:
            raise CampaignManagerError("remote campaign state belongs to a different run")
        if int(state.get("job_id") or 0) != HPC_JOB_ID:
            raise CampaignManagerError("remote campaign state job binding changed")
        if state.get("credential_profile") != CREDENTIAL_PROFILE:
            raise CampaignManagerError("remote campaign state credential profile changed")
    payload.update(
        {
            "schema": BINDING.schema("status"),
            "run_id": run_id,
            "job_id": HPC_JOB_ID,
            "credential_profile": CREDENTIAL_PROFILE,
            "created_at": utc_now(),
        }
    )
    write_json(local["status"], payload)
    return payload


def collect_remote_run(client: Any, remote_run: str, run_id: str) -> dict[str, Any]:
    local_root = LOCAL_COLLECTED_ROOT / run_id
    local_root.mkdir(parents=True, exist_ok=True)
    downloaded = []
    with client.open_sftp() as sftp:
        configure_collection_timeout(sftp)
        stack = [(ensure_remote(remote_run), local_root)]
        while stack:
            remote_dir, local_dir = stack.pop()
            verify_remote_path_chain(sftp, remote_dir)
            directory_metadata = sftp.lstat(remote_dir)
            if stat.S_ISLNK(directory_metadata.st_mode) or not stat.S_ISDIR(
                directory_metadata.st_mode
            ):
                raise CampaignManagerError("remote collection directory is not a real directory")
            local_dir.mkdir(parents=True, exist_ok=True)
            for item in sftp.listdir_attr(remote_dir):
                remote_path = ensure_remote(posixpath.join(remote_dir, item.filename))
                local_path = local_dir / item.filename
                if stat.S_ISLNK(item.st_mode):
                    if item.filename in COLLECT_NAMES:
                        raise CampaignManagerError("remote collection artifact is a symbolic link")
                    continue
                if stat.S_ISDIR(item.st_mode):
                    stack.append((remote_path, local_path))
                    continue
                if item.filename not in COLLECT_NAMES:
                    continue
                if not stat.S_ISREG(item.st_mode):
                    raise CampaignManagerError("remote collection artifact is not a regular file")
                verify_remote_path_chain(
                    sftp,
                    remote_path,
                    require_leaf_regular=True,
                )
                temporary = local_path.with_name(
                    f".{local_path.name}.{os.getpid()}.download"
                )
                try:
                    sftp.get(remote_path, str(temporary))
                    os.replace(temporary, local_path)
                finally:
                    temporary.unlink(missing_ok=True)
                downloaded.append(
                    {
                        "remote": remote_path,
                        "local": str(local_path.resolve()),
                        "bytes": local_path.stat().st_size,
                        "sha256": sha256_file(local_path),
                    }
                )
    report = {
        "schema": "evomind.mlebench_remote_ops.collection.v1",
        "created_at": utc_now(),
        "run_id": run_id,
        "remote_run": remote_run,
        "local_root": str(local_root.resolve()),
        "include_checkpoints": False,
        "file_count": len(downloaded),
        "files": downloaded,
        "passed": bool(downloaded),
    }
    write_json(local_root / "collection_manifest.json", report)
    return report


def collect(run_id: str) -> dict[str, Any]:
    local = local_paths(run_id)
    plan = json.loads(local["plan"].read_text(encoding="utf-8"))
    validate_campaign_plan(plan, run_id)
    remote = remote_paths(run_id, str(plan["bundle_sha256"]))
    current = status(run_id)
    if current["state"].get("status") != "awaiting_collection_and_freeze":
        raise CampaignManagerError("campaign is not ready for collection")
    client, _config = connect()
    try:
        reports = []
        for record in plan["formal_runs"]:
            formal_id = str(record["run_id"])
            reports.append(
                collect_remote_run(
                    client,
                    f"{remote['campaign']}/formal_runs/{formal_id}",
                    formal_id,
                )
            )
        evidence_root = local["root"] / "remote_evidence"
        evidence_root.mkdir(parents=True, exist_ok=True)
        optional_evidence = {
            "ablation_supersession.json",
            "harmonization_boundary_campaign_recovery.json",
            *{
                f"formal_s{seed}_resume_supersession.json"
                for seed in (int(value) for value in plan["formal_seeds"])
            },
        }
        with client.open_sftp() as sftp:
            configure_collection_timeout(sftp)
            fixed_evidence = [
                "campaign_state.json",
                "campaign_plan.json",
                "hpc_telemetry.jsonl",
                "ablation_supersession.json",
                "harmonization_boundary_campaign_recovery.json",
                f"ablation/runs/{run_id}_ablation/siim_preprocessing_ablation.json",
                f"ablation/runs/{run_id}_ablation/siim_duplicate_connected_groups.json",
                f"ablation/runs/{run_id}_ablation/siim_image_content_manifest.csv",
            ]
            fixed_evidence.extend(
                f"formal_s{int(seed)}_resume_supersession.json"
                for seed in plan["formal_seeds"]
            )
            for relative in fixed_evidence:
                remote_file = ensure_remote(f"{remote['campaign']}/{relative}")
                local_file = evidence_root / relative
                local_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    metadata = sftp.lstat(remote_file)
                    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(
                        metadata.st_mode
                    ):
                        raise CampaignManagerError(
                            "remote evidence is not a regular non-link file"
                        )
                    verify_remote_path_chain(
                        sftp,
                        remote_file,
                        require_leaf_regular=True,
                    )
                    temporary = local_file.with_name(
                        f".{local_file.name}.{os.getpid()}.download"
                    )
                    try:
                        sftp.get(remote_file, str(temporary))
                        os.replace(temporary, local_file)
                    finally:
                        temporary.unlink(missing_ok=True)
                except OSError:
                    if relative in optional_evidence:
                        continue
                    raise
    finally:
        client.close()
    payload = {
        "schema": BINDING.schema("collection"),
        "created_at": utc_now(),
        "run_id": run_id,
        "job_id": HPC_JOB_ID,
        "credential_profile": CREDENTIAL_PROFILE,
        "status": "collected",
        "seed_collections": reports,
        "remote_evidence_root": str((local["root"] / "remote_evidence").resolve()),
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    write_json(local["collect"], payload)
    return payload


def aggregate(run_id: str) -> dict[str, Any]:
    local = local_paths(run_id)
    plan = json.loads(local["plan"].read_text(encoding="utf-8"))
    validate_campaign_plan(plan, run_id)
    output = PROJECT_ROOT / "workspace" / f"siim_{JOB_TAG}" / "candidates" / run_id
    if output.exists():
        raise CampaignManagerError("candidate output already exists; verify it instead of overwriting")
    if not LOCAL_SAMPLE_SUBMISSION.is_file():
        raise CampaignManagerError("local sample submission is missing")
    result = aggregator.aggregate(
        plan_path=local["plan"],
        collected_root=LOCAL_COLLECTED_ROOT,
        sample_path=LOCAL_SAMPLE_SUBMISSION,
        output_dir=output,
    )
    payload = {
        "schema": BINDING.schema("aggregation"),
        "created_at": utc_now(),
        "run_id": run_id,
        "job_id": HPC_JOB_ID,
        "credential_profile": CREDENTIAL_PROFILE,
        "status": "frozen_before_private_grader",
        "result": result,
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    write_json(local["aggregate"], payload)
    return payload


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "gate", "publish-status", "launch", "status", "collect", "aggregate"),
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--interval-seconds", type=int, default=15)
    parser.add_argument("--max-gate-age", type=int, default=600)
    parser.add_argument("--evolution-spec", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = (
            prepare(args.run_id, evolution_spec_path=args.evolution_spec)
            if args.command == "prepare"
            else gate(args.run_id, interval_seconds=args.interval_seconds)
            if args.command == "gate"
            else publish_status(args.run_id)
            if args.command == "publish-status"
            else launch(args.run_id, max_gate_age=args.max_gate_age)
            if args.command == "launch"
            else status(args.run_id)
            if args.command == "status"
            else collect(args.run_id)
            if args.command == "collect"
            else aggregate(args.run_id)
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if args.command == "gate" and payload.get("passed") is not True:
            return 4
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        SSHException,
        CampaignManagerError,
        remote_ops.RemoteOpsError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema": BINDING.schema("manager_failure"),
                    "created_at": utc_now(),
                    "command": args.command,
                    "run_id": args.run_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "signals_sent": 0,
                    "other_processes_modified": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
