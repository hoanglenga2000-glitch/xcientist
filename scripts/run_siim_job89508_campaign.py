#!/usr/bin/env python3
"""Run the governed SIIM campaign inside one content-addressed HPC bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from research_os.siim_hpc_binding import binding_from_environment

ALLOWED_ROOT = pathlib.Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
COMPETITION = "siim-isic-melanoma-classification"
BINDING = binding_from_environment()
TERMINAL_CANDIDATE_STATES = {
    "candidate_ready_confirmation_pending",
    "promotion_gate_passed_confirmation_pending",
}
KNOWN_TERMINAL_RESULT_STATES = {*TERMINAL_CANDIDATE_STATES, "passed"}
CORRECTED_PROTOCOL_VERSION = "siim_exact_duplicate_grouping_v2"
CORRECTED_FORMAL_BUDGET_SECONDS = 72 * 3600
FORMAL_OUTER_FOLDS = 5
FORMAL_INNER_FOLDS = 3
FORMAL_RESUME_SCHEMA = "evomind.siim.formal_resume_contract.v1"
FORMAL_GROUP_SCHEMA = "evomind.siim_patient_content_connected_groups.v3"
FORMAL_GROUP_POLICY = "patient_exact_file_decoded_pixel_v2"
FORMAL_PERCEPTUAL_POLICY = "audit_only_no_cross_patient_union_v1"
BASELINE_FORMAL_SEEDS = (43, 44, 45)
R2_FORMAL_SEEDS = (46, 47, 48)
DEDICATED_ENVIRONMENT_KEYS = (
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "TORCH_HOME",
    "PIP_CACHE_DIR",
    "MPLCONFIGDIR",
    "NUMBA_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "TRITON_CACHE_DIR",
    "CUPY_CACHE_DIR",
    "JOBLIB_TEMP_FOLDER",
    "PYTHONPYCACHEPREFIX",
    "KAGGLE_CONFIG_DIR",
)


class CampaignError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampaignError(f"expected a JSON object: {path.name}")
    return payload


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_immutable_file(source: pathlib.Path, destination: pathlib.Path) -> str:
    source = ensure_within(source)
    destination = ensure_within(destination)
    if not source.is_file() or source.is_symlink():
        raise CampaignError(f"immutable source is missing or linked: {source}")
    expected = sha256_file(source)
    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise CampaignError("immutable destination is not a regular file")
        if sha256_file(destination) != expected:
            raise CampaignError("immutable destination differs; refusing overwrite")
        return "reused"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    if sha256_file(destination) != expected:
        raise CampaignError("immutable copy hash changed")
    return "copied"


def validate_r2_plan(plan: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    protocol = plan.get("evolution_protocol")
    lineage = plan.get("lineage_basis")
    preprocessing = plan.get("preprocessing_evidence")
    training = plan.get("candidate_training")
    if not all(isinstance(value, dict) for value in (protocol, lineage, preprocessing, training)):
        raise CampaignError("R2 plan sections are incomplete")
    assert isinstance(protocol, dict)
    assert isinstance(lineage, dict)
    assert isinstance(preprocessing, dict)
    assert isinstance(training, dict)
    if protocol.get("candidate_id") != "r2_foldwise_rank_channel_consensus_v1":
        raise CampaignError("R2 candidate protocol changed")
    if lineage.get("child_run_id") != run_id:
        raise CampaignError("R2 lineage child changed")
    parent_id = str(lineage.get("parent_run_id") or "")
    if not parent_id or parent_id == run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in parent_id
    ):
        raise CampaignError("R2 parent Run ID is invalid")
    if preprocessing.get("mode") != "parent_frozen_reuse":
        raise CampaignError("R2 preprocessing reuse mode changed")
    if preprocessing.get("parent_run_id") != parent_id:
        raise CampaignError("R2 preprocessing parent changed")
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
        raise CampaignError("R2 base-training configuration changed")
    if tuple(int(value) for value in plan.get("formal_seeds") or ()) != R2_FORMAL_SEEDS:
        raise CampaignError("R2 fresh formal seeds changed")
    return {
        "parent_run_id": parent_id,
        "protocol": protocol,
        "preprocessing": preprocessing,
        "training": training,
    }


def ensure_within(path: pathlib.Path, root: pathlib.Path = ALLOWED_ROOT) -> pathlib.Path:
    candidate = path.resolve(strict=False)
    boundary = root.resolve(strict=False)
    if candidate != boundary and boundary not in candidate.parents:
        raise CampaignError(f"path escaped the dedicated root: {candidate}")
    return candidate


def ensure_venv_python(path: pathlib.Path, root: pathlib.Path = ALLOWED_ROOT) -> pathlib.Path:
    """Validate a venv interpreter without resolving away its activation symlink.

    ``venv/bin/python`` normally points at the shared base interpreter.  Returning
    ``Path.resolve()`` here would execute that base interpreter directly and drop
    the venv's site-packages.  Both the lexical path and its symlink target must
    stay inside the dedicated remote root, while the returned path remains the
    lexical venv entry point.
    """

    lexical = pathlib.Path(os.path.abspath(os.fspath(path)))
    lexical_root = pathlib.Path(os.path.abspath(os.fspath(root)))
    if lexical != lexical_root and lexical_root not in lexical.parents:
        raise CampaignError(f"venv Python path escaped the dedicated root: {lexical}")
    resolved = lexical.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise CampaignError(f"venv Python target escaped the dedicated root: {resolved}")
    return lexical


def expected_remote_storage_policy(run_id: str) -> dict[str, Any]:
    job_root = ALLOWED_ROOT / f"siim_{BINDING.job_tag}"
    return {
        "schema": "evomind.siim.dedicated_remote_storage.v1",
        "mode": "dedicated_remote_root_only",
        "allowed_root": str(ALLOWED_ROOT),
        "dataset_root": str(ALLOWED_ROOT / "mlebench_official_data"),
        "official_source_root": str(ALLOWED_ROOT / "mle-bench"),
        "torch_home": str(ALLOWED_ROOT / "mlebench_model_cache" / "torch"),
        "campaign_root": str(job_root / "campaigns" / run_id),
        "bundle_parent": str(job_root / "bundles"),
        "runtime_parent": str(job_root / "runtime"),
        "private_grader_parent": str(job_root / "private_grader"),
        "short_temp_parent": str(ALLOWED_ROOT / ".t"),
        "home_cache_temp_isolated": True,
        "external_remote_writes": "forbidden",
    }


def verify_bundle(bundle_root: pathlib.Path) -> dict[str, Any]:
    manifest_path = bundle_root / "bundle_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") != BINDING.schema("bundle"):
        raise CampaignError("bundle manifest schema changed")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise CampaignError("bundle manifest is empty")
    for record in records:
        relative = pathlib.PurePosixPath(str(record.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise CampaignError("unsafe bundle manifest path")
        path = ensure_within(bundle_root / pathlib.Path(*relative.parts), bundle_root)
        if not path.is_file():
            raise CampaignError(f"bundle file missing: {relative}")
        if path.stat().st_size != int(record.get("bytes") or -1):
            raise CampaignError(f"bundle size changed: {relative}")
        if sha256_file(path) != str(record.get("sha256") or ""):
            raise CampaignError(f"bundle hash changed: {relative}")
    return manifest


def query_gpu() -> dict[str, Any]:
    gpu_command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    app_command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    gpu = subprocess.run(gpu_command, check=False, capture_output=True, text=True, timeout=20)
    apps = subprocess.run(app_command, check=False, capture_output=True, text=True, timeout=20)
    if gpu.returncode:
        raise CampaignError("nvidia-smi GPU query failed")
    rows = list(csv.reader(gpu.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 6:
        raise CampaignError("unexpected GPU inventory")
    row = [value.strip() for value in rows[0]]
    processes = []
    if apps.returncode == 0:
        for item in csv.reader(apps.stdout.splitlines()):
            if len(item) >= 2 and item[0].strip():
                processes.append({"pid": int(item[0]), "memory_mib": int(float(item[1]))})
    return {
        "name": row[0],
        "uuid": row[1],
        "memory_total_mib": int(float(row[2])),
        "memory_used_mib": int(float(row[3])),
        "memory_free_mib": int(float(row[4])),
        "utilization_percent": int(float(row[5])),
        "compute_apps": processes,
    }


def descendant_pids(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for proc in pathlib.Path("/proc").glob("[0-9]*"):
        try:
            fields = (proc / "stat").read_text(encoding="utf-8", errors="replace").split()
            parents[int(proc.name)] = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
    result = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in result and pid not in result:
                result.add(pid)
                changed = True
    return result


class ResourceMonitor:
    def __init__(self, *, run_id: str, root_pid: int, telemetry: pathlib.Path, control: pathlib.Path) -> None:
        self.run_id = run_id
        self.root_pid = int(root_pid)
        self.telemetry = telemetry
        self.control = control
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="siim-resource-monitor", daemon=True)
        initial = query_gpu()
        own = descendant_pids(self.root_pid)
        self.initial_other_mib = sum(
            item["memory_mib"] for item in initial["compute_apps"] if item["pid"] not in own
        )
        atomic_json(
            self.control,
            {
                "status": "continue",
                "run_id": self.run_id,
                "job_id": BINDING.job_id,
                "credential_profile": BINDING.credential_profile,
                "created_at": utc_now(),
            },
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=30)

    def _append(self, payload: dict[str, Any]) -> None:
        self.telemetry.parent.mkdir(parents=True, exist_ok=True)
        with self.telemetry.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                snapshot = query_gpu()
                own = descendant_pids(self.root_pid)
                other_mib = sum(
                    item["memory_mib"]
                    for item in snapshot["compute_apps"]
                    if item["pid"] not in own
                )
                reasons = []
                if other_mib > 8 * 1024:
                    reasons.append("other_process_memory_above_8gib")
                if other_mib > self.initial_other_mib + 256:
                    reasons.append("other_process_memory_growth")
                if snapshot["memory_free_mib"] < 12 * 1024:
                    reasons.append("gpu_free_memory_below_12gib")
                event = {
                    "schema": "evomind.siim.hpc_telemetry.v1",
                    "run_id": self.run_id,
                    "job_id": BINDING.job_id,
                    "credential_profile": BINDING.credential_profile,
                    "captured_at": utc_now(),
                    "gpu": snapshot,
                    "other_process_memory_mib": other_mib,
                    "hold_reasons": reasons,
                    "signals_sent": 0,
                    "other_processes_modified": False,
                }
                self._append(event)
                if reasons:
                    atomic_json(
                        self.control,
                        {
                            "status": "pause_after_epoch",
                            "run_id": self.run_id,
                            "job_id": BINDING.job_id,
                            "credential_profile": BINDING.credential_profile,
                            "reason": ",".join(reasons),
                            "created_at": utc_now(),
                            "signals_sent": 0,
                            "other_processes_modified": False,
                        },
                    )
            except Exception as exc:
                self._append(
                    {
                        "schema": "evomind.siim.hpc_telemetry.v1",
                        "run_id": self.run_id,
                        "job_id": BINDING.job_id,
                        "credential_profile": BINDING.credential_profile,
                        "captured_at": utc_now(),
                        "monitor_error_type": type(exc).__name__,
                        "hold_reasons": ["resource_monitor_error"],
                        "signals_sent": 0,
                        "other_processes_modified": False,
                    }
                )
                atomic_json(
                    self.control,
                    {
                        "status": "pause_after_epoch",
                        "run_id": self.run_id,
                        "job_id": BINDING.job_id,
                        "credential_profile": BINDING.credential_profile,
                        "reason": "resource_monitor_error",
                        "created_at": utc_now(),
                    },
                )
            self.stop_event.wait(15)


def command_environment(
    bundle_root: pathlib.Path,
    torch_home: pathlib.Path,
    campaign_root: pathlib.Path,
) -> dict[str, str]:
    """Isolate home, caches, bytecode, and temporary files below ALLOWED_ROOT."""

    environment_root = ensure_within(
        campaign_root / "runtime_environment", ALLOWED_ROOT
    )
    cache_root = ensure_within(environment_root / "cache", ALLOWED_ROOT)
    temporary_root = ensure_within(
        ALLOWED_ROOT
        / ".t"
        / hashlib.sha256(campaign_root.name.encode("utf-8")).hexdigest()[:12],
        ALLOWED_ROOT,
    )
    paths = {
        "HOME": ensure_within(environment_root / "home", ALLOWED_ROOT),
        "TMPDIR": temporary_root,
        "TEMP": temporary_root,
        "TMP": temporary_root,
        "XDG_CACHE_HOME": ensure_within(cache_root / "xdg", ALLOWED_ROOT),
        "HF_HOME": ensure_within(cache_root / "huggingface", ALLOWED_ROOT),
        "TORCH_HOME": ensure_within(torch_home, ALLOWED_ROOT),
        "PIP_CACHE_DIR": ensure_within(cache_root / "pip", ALLOWED_ROOT),
        "MPLCONFIGDIR": ensure_within(cache_root / "matplotlib", ALLOWED_ROOT),
        "NUMBA_CACHE_DIR": ensure_within(cache_root / "numba", ALLOWED_ROOT),
        "CUDA_CACHE_PATH": ensure_within(cache_root / "cuda", ALLOWED_ROOT),
        "TRITON_CACHE_DIR": ensure_within(cache_root / "triton", ALLOWED_ROOT),
        "CUPY_CACHE_DIR": ensure_within(cache_root / "cupy", ALLOWED_ROOT),
        "JOBLIB_TEMP_FOLDER": temporary_root,
        "PYTHONPYCACHEPREFIX": ensure_within(cache_root / "python", ALLOWED_ROOT),
        "KAGGLE_CONFIG_DIR": ensure_within(
            environment_root / "kaggle_disabled", ALLOWED_ROOT
        ),
    }
    for path in dict.fromkeys(paths.values()):
        path.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("KAGGLE_USERNAME", None)
    environment.pop("KAGGLE_KEY", None)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": ":".join(
                [str(bundle_root), str(bundle_root / "scripts"), str(bundle_root / "src")]
            ),
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": (
                "backend:native,expandable_segments:True,garbage_collection_threshold:0.9"
            ),
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "EVOMIND_SIIM_HPC_JOB_ID": str(BINDING.job_id),
            "EVOMIND_HPC_CREDENTIAL_PROFILE": BINDING.credential_profile,
            **{name: str(path) for name, path in paths.items()},
        }
    )
    if set(DEDICATED_ENVIRONMENT_KEYS) - environment.keys():
        raise CampaignError("dedicated runtime environment is incomplete")
    return environment


def run_phase(
    command: list[str],
    *,
    phase: str,
    log_root: pathlib.Path,
    environment: dict[str, str],
    monitor: ResourceMonitor,
) -> int:
    stdout_path = log_root / f"{phase}.stdout.log"
    stderr_path = log_root / f"{phase}.stderr.log"
    log_root.mkdir(parents=True, exist_ok=True)
    monitor.start()
    try:
        with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open("ab", buffering=0) as stderr:
            completed = subprocess.run(
                command,
                env=environment,
                cwd=log_root.parent,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        return int(completed.returncode)
    finally:
        monitor.stop()


def candidate_result(run_root: pathlib.Path) -> dict[str, Any] | None:
    path = run_root / COMPETITION / "result.json"
    return read_json(path) if path.is_file() else None


def siim_resume_root(run_root: pathlib.Path) -> pathlib.Path:
    return ensure_within(
        run_root / COMPETITION / "attempts" / "siim_resume_state"
    )


def resolve_formal_runtime_budget(
    run_root: pathlib.Path,
    *,
    model_seed: int,
    remaining_seconds: float,
) -> float:
    """Reuse the immutable per-seed budget or validate a new corrected budget."""

    remaining = float(remaining_seconds)
    if not (remaining > 0 and remaining < float("inf")):
        raise CampaignError("corrected formal runtime budget is invalid")
    budget_state = siim_resume_root(run_root) / "runtime_budget.json"
    if not budget_state.is_file():
        return remaining
    persisted = read_json(budget_state)
    try:
        started = float(persisted["started_unix"])
        configured = float(persisted["runtime_budget_seconds"])
        persisted_seed = int(persisted["model_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignError("persisted formal runtime budget is malformed") from exc
    if persisted.get("schema") != "evomind.siim.runtime_budget.v1":
        raise CampaignError("persisted formal runtime budget schema changed")
    if persisted_seed != int(model_seed):
        raise CampaignError("persisted formal runtime budget seed changed")
    if not (started > 0 and started < float("inf")):
        raise CampaignError("persisted formal runtime budget start is invalid")
    if not (configured > 0 and configured < float("inf")):
        raise CampaignError("persisted formal runtime budget value is invalid")
    return configured


def ablation_contract_checks(run_dir: pathlib.Path, *, expected_folds: int) -> dict[str, bool]:
    report_path = run_dir / "siim_preprocessing_ablation.json"
    split_path = run_dir / "siim_preprocessing_ablation_split_contract.json"
    groups_path = run_dir / "siim_duplicate_connected_groups.json"
    if not all(path.is_file() for path in (report_path, split_path, groups_path)):
        return {"material_files_exist": False}
    report = read_json(report_path)
    split = read_json(split_path)
    groups = read_json(groups_path)
    seed_contracts = split.get("seed_contracts")
    return {
        "material_files_exist": True,
        "report_passed": report.get("passed") is True,
        "report_fold_count": int(report.get("fold_count") or 0) == int(expected_folds),
        "report_group_policy": report.get("leakage_group_policy")
        == "patient_exact_file_decoded_pixel_v2",
        "report_perceptual_policy": report.get("perceptual_edge_policy")
        == "audit_only_no_cross_patient_union_v1",
        "split_group_policy": split.get("leakage_group_policy")
        == "patient_exact_file_decoded_pixel_v2",
        "split_perceptual_policy": split.get("perceptual_edge_policy")
        == "audit_only_no_cross_patient_union_v1",
        "all_requested_folds_present": bool(seed_contracts)
        and all(
            int(record.get("requested_folds") or 0) == int(expected_folds)
            and int(record.get("actual_folds") or 0) == int(expected_folds)
            for record in seed_contracts
        ),
        "group_schema": groups.get("schema")
        == "evomind.siim_patient_content_connected_groups.v3",
        "group_policy": groups.get("leakage_group_policy")
        == "patient_exact_file_decoded_pixel_v2",
        "perceptual_audit_only": groups.get("perceptual_edge_policy")
        == "audit_only_no_cross_patient_union_v1"
        and int(groups.get("perceptual_edges_applied_to_groups") or 0) == 0,
    }


def supersede_directory(
    path: pathlib.Path,
    *,
    reason: str,
    protocol_version: str = CORRECTED_PROTOCOL_VERSION,
) -> tuple[pathlib.Path, dict[str, Any]]:
    """Atomically preserve an incompatible same-Run directory without deletion."""

    source = ensure_within(path)
    if not source.is_dir():
        raise CampaignError(f"supersession source is not a directory: {source}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(
        f"{source}:{reason}:{protocol_version}".encode("utf-8")
    ).hexdigest()[:12]
    target = ensure_within(
        source.with_name(f"{source.name}.superseded_{stamp}_{digest}")
    )
    if target.exists():
        raise CampaignError(f"supersession target already exists: {target}")
    records = []
    for file_path in sorted(source.rglob("*")):
        if not file_path.is_file():
            continue
        records.append({
            "path": file_path.relative_to(source).as_posix(),
            "bytes": int(file_path.stat().st_size),
            "sha256": sha256_file(file_path),
        })
    os.replace(source, target)
    payload = {
        "schema": "evomind.siim.same_run_supersession.v1",
        "created_at": utc_now(),
        "reason": str(reason),
        "protocol_version": str(protocol_version),
        "source": str(source),
        "preserved_at": str(target),
        "files": records,
        "files_deleted": 0,
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    atomic_json(target / "supersession_record.json", payload)
    return target, payload


def _contract_sha256(payload: Mapping[str, Any]) -> str:
    base = {key: value for key, value in payload.items() if key != "contract_sha256"}
    return hashlib.sha256(
        json.dumps(
            base,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _int_equals(value: Any, expected: int) -> bool:
    try:
        return int(value) == int(expected)
    except (TypeError, ValueError):
        return False


def formal_resume_contract_checks(
    resume_contract_path: pathlib.Path,
    *,
    expected_seed: int,
    expected_manifest_sha256: str,
    bundle_root: pathlib.Path,
) -> dict[str, bool]:
    """Validate every scientific binding needed to resume or reuse a SIIM seed."""

    path = pathlib.Path(resume_contract_path)
    if not path.is_file():
        return {"resume_contract_exists": False}
    try:
        contract = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, CampaignError):
        return {"resume_contract_exists": True, "resume_contract_readable": False}
    model = contract.get("model") if isinstance(contract.get("model"), dict) else {}
    adapter = pathlib.Path(bundle_root) / "scripts" / "mlebench_medal_recovery_adapters.py"
    wave2 = pathlib.Path(bundle_root) / "scripts" / "mlebench_wave2_adapters.py"
    return {
        "resume_contract_exists": True,
        "resume_contract_readable": True,
        "resume_contract_schema": contract.get("schema") == FORMAL_RESUME_SCHEMA,
        "resume_contract_competition": contract.get("competition_id") == COMPETITION,
        "resume_contract_seed": _int_equals(contract.get("model_seed"), expected_seed),
        "resume_contract_outer": _int_equals(contract.get("outer_folds"), FORMAL_OUTER_FOLDS),
        "resume_contract_inner": _int_equals(contract.get("inner_folds"), FORMAL_INNER_FOLDS),
        "resume_contract_group_policy": contract.get("leakage_group_policy") == FORMAL_GROUP_POLICY,
        "resume_contract_perceptual_policy": contract.get("perceptual_edge_policy")
        == FORMAL_PERCEPTUAL_POLICY,
        "resume_contract_manifest": contract.get("image_content_manifest_sha256")
        == expected_manifest_sha256,
        "resume_contract_self_hash": contract.get("contract_sha256")
        == _contract_sha256(contract),
        "resume_contract_adapter_source": adapter.is_file()
        and contract.get("adapter_source_sha256") == sha256_file(adapter),
        "resume_contract_wave2_source": wave2.is_file()
        and contract.get("wave2_source_sha256") == sha256_file(wave2),
        "resume_contract_workers": _int_equals(model.get("workers"), 8),
        "resume_contract_fast_kernels": model.get("fast_kernel_mode") is True,
    }


def corrected_formal_result_checks(
    run_root: pathlib.Path,
    result: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_manifest_sha256: str,
    bundle_root: pathlib.Path,
) -> dict[str, bool]:
    """Reject terminal-looking results that predate the corrected 5x3 contract."""

    budget = result.get("budget") if isinstance(result.get("budget"), dict) else {}
    groups = (
        budget.get("duplicate_group_report")
        if isinstance(budget.get("duplicate_group_report"), dict)
        else {}
    )
    contract_checks = formal_resume_contract_checks(
        siim_resume_root(pathlib.Path(run_root)) / "resume_contract.json",
        expected_seed=expected_seed,
        expected_manifest_sha256=expected_manifest_sha256,
        bundle_root=bundle_root,
    )
    contract = {}
    contract_path = siim_resume_root(pathlib.Path(run_root)) / "resume_contract.json"
    if contract_path.is_file() and contract_checks.get("resume_contract_readable") is True:
        contract = read_json(contract_path)
    return {
        "terminal_status": result.get("status") in TERMINAL_CANDIDATE_STATES,
        "competition_id": result.get("competition_id") == COMPETITION,
        "candidate_only": result.get("candidate_only") is True,
        "valid_submission": result.get("valid_submission") is True,
        "official_grader_excluded": result.get("official_grader_executed") is False,
        "result_seed": _int_equals(budget.get("seed"), expected_seed),
        "outer_folds_exact": _int_equals(budget.get("folds"), FORMAL_OUTER_FOLDS),
        "result_manifest": budget.get("image_content_manifest_sha256")
        == expected_manifest_sha256,
        "group_schema": groups.get("schema") == FORMAL_GROUP_SCHEMA,
        "group_policy": groups.get("leakage_group_policy") == FORMAL_GROUP_POLICY,
        "perceptual_audit_only": groups.get("perceptual_edge_policy")
        == FORMAL_PERCEPTUAL_POLICY,
        "perceptual_edges_not_applied": _int_equals(
            groups.get("perceptual_edges_applied_to_groups"), 0
        ),
        "result_contract_matches": bool(contract)
        and budget.get("resume_contract_sha256") == contract.get("contract_sha256"),
        **contract_checks,
    }


def all_checks_pass(checks: Mapping[str, bool]) -> bool:
    return bool(checks) and all(value is True for value in checks.values())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=pathlib.Path, required=True)
    parser.add_argument("--bundle-root", type=pathlib.Path, required=True)
    parser.add_argument("--campaign-root", type=pathlib.Path, required=True)
    parser.add_argument("--runtime-python", type=pathlib.Path, required=True)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument("--official-source-root", type=pathlib.Path, required=True)
    parser.add_argument("--torch-home", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    plan_path = ensure_within(args.plan)
    bundle_root = ensure_within(args.bundle_root)
    campaign_root = ensure_within(args.campaign_root)
    runtime_python = ensure_venv_python(args.runtime_python)
    data_root = ensure_within(args.data_root)
    official_source_root = ensure_within(args.official_source_root)
    torch_home = ensure_within(args.torch_home)
    plan = read_json(plan_path)
    if plan.get("schema") != "evomind.siim.hpc_campaign_plan.v1":
        raise CampaignError("campaign plan schema changed")
    if int(plan.get("job_id") or 0) != BINDING.job_id:
        raise CampaignError("campaign plan job binding changed")
    if plan.get("credential_profile") != BINDING.credential_profile:
        raise CampaignError("campaign plan credential profile changed")
    run_id = str(plan.get("run_id") or "")
    if not run_id or plan.get("official_submission") != "forbidden":
        raise CampaignError("campaign boundary is invalid")
    if plan.get("budget_policy") != "user_selected_A_corrected_full_closure":
        raise CampaignError("campaign corrected budget policy changed")
    if plan.get("budget_hours") != {
        "ablation": 4,
        "formal_training": 72,
        "delivery": 2,
        "total": 78,
    }:
        raise CampaignError("campaign corrected budget changed")
    if plan.get("original_budget_hours") != {
        "ablation": 4,
        "formal_training": 18,
        "delivery": 2,
        "total": 24,
    }:
        raise CampaignError("campaign original budget audit changed")
    if plan.get("corrected_protocol_version") != CORRECTED_PROTOCOL_VERSION:
        raise CampaignError("campaign corrected scientific protocol changed")
    if os.environ.get("EVOMIND_SIIM_HPC_JOB_ID", "").strip() != str(BINDING.job_id):
        raise CampaignError("campaign job environment binding is missing or changed")
    if os.environ.get("EVOMIND_HPC_CREDENTIAL_PROFILE", "").strip() != BINDING.credential_profile:
        raise CampaignError("campaign credential environment binding is missing or changed")
    if os.environ.get("EVOMIND_SIIM_RUN_ID", "").strip() != run_id:
        raise CampaignError("campaign Run environment binding is missing or changed")
    if tuple(plan.get("ablation_seeds") or ()) != (40, 41, 42):
        raise CampaignError("ablation seeds changed")
    evolution = (
        validate_r2_plan(plan, run_id)
        if isinstance(plan.get("evolution_protocol"), dict)
        else None
    )
    formal_seeds = R2_FORMAL_SEEDS if evolution is not None else BASELINE_FORMAL_SEEDS
    if tuple(plan.get("formal_seeds") or ()) != formal_seeds:
        raise CampaignError("formal seeds changed")
    expected_formal_runs = [
        {"seed": seed, "run_id": f"{run_id}_s{seed}"}
        for seed in formal_seeds
    ]
    if plan.get("formal_runs") != expected_formal_runs:
        raise CampaignError("formal Run bindings changed")
    if plan.get("remote_root") != str(ALLOWED_ROOT):
        raise CampaignError("campaign remote root changed")
    if plan.get("remote_storage_policy") != expected_remote_storage_policy(run_id):
        raise CampaignError("campaign dedicated remote-storage policy changed")
    expected_campaign_root = ensure_within(
        ALLOWED_ROOT / f"siim_{BINDING.job_tag}" / "campaigns" / run_id
    )
    if campaign_root != expected_campaign_root or plan_path != campaign_root / "campaign_plan.json":
        raise CampaignError("campaign paths are not bound to the Run")
    expected_bundle_root = ensure_within(
        ALLOWED_ROOT / f"siim_{BINDING.job_tag}" / "bundles" / str(plan.get("bundle_sha256") or "")
    )
    if bundle_root != expected_bundle_root:
        raise CampaignError("bundle path is not bound to this allocation")
    expected_runtime_python = ensure_venv_python(
        ALLOWED_ROOT
        / f"siim_{BINDING.job_tag}"
        / "runtime"
        / str(plan.get("runtime_requirements_sha256") or "")
        / "venv"
        / "bin"
        / "python"
    )
    if runtime_python != expected_runtime_python:
        raise CampaignError("runtime Python is not bound to this allocation")
    if data_root != ensure_within(ALLOWED_ROOT / "mlebench_official_data"):
        raise CampaignError("dataset root changed")
    if official_source_root != ensure_within(ALLOWED_ROOT / "mle-bench"):
        raise CampaignError("official source root changed")
    if torch_home != ensure_within(ALLOWED_ROOT / "mlebench_model_cache" / "torch"):
        raise CampaignError("Torch model cache root changed")
    manifest = verify_bundle(bundle_root)
    if manifest.get("bundle_sha256") != plan.get("bundle_sha256"):
        raise CampaignError("campaign plan and bundle differ")
    if not runtime_python.is_file():
        raise CampaignError("isolated runtime Python is missing")

    state_path = campaign_root / "campaign_state.json"
    telemetry_path = campaign_root / "hpc_telemetry.jsonl"
    control_path = campaign_root / "resource_control.json"
    log_root = campaign_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    existing = read_json(state_path) if state_path.is_file() else {}
    if existing:
        if existing.get("schema") != BINDING.schema("campaign_state"):
            raise CampaignError("existing campaign state schema changed")
        if existing.get("run_id") != run_id:
            raise CampaignError("existing campaign state belongs to a different run")
        if int(existing.get("job_id") or 0) != BINDING.job_id:
            raise CampaignError("existing campaign state job binding changed")
        if existing.get("credential_profile") != BINDING.credential_profile:
            raise CampaignError("existing campaign state credential profile changed")
    acquired = float(existing.get("gpu_acquired_at_epoch") or time.time())
    corrected_protocol_started = (
        float(existing.get("corrected_protocol_started_at_epoch"))
        if existing.get("corrected_protocol_version") == CORRECTED_PROTOCOL_VERSION
        and existing.get("corrected_protocol_started_at_epoch") is not None
        else time.time()
    )
    corrected_formal_budget_started = (
        float(existing.get("corrected_formal_budget_started_at_epoch"))
        if existing.get("corrected_protocol_version") == CORRECTED_PROTOCOL_VERSION
        and existing.get("corrected_formal_budget_started_at_epoch") is not None
        else None
    )

    def write_state(status: str, **extra: Any) -> None:
        atomic_json(
            state_path,
            {
                "schema": BINDING.schema("campaign_state"),
                "run_id": run_id,
                "job_id": BINDING.job_id,
                "credential_profile": BINDING.credential_profile,
                "status": status,
                "supervisor_pid": os.getpid(),
                "gpu_acquired_at_epoch": acquired,
                "corrected_protocol_version": CORRECTED_PROTOCOL_VERSION,
                "corrected_protocol_started_at_epoch": corrected_protocol_started,
                "corrected_formal_budget_started_at_epoch": corrected_formal_budget_started,
                "corrected_formal_budget_seconds": CORRECTED_FORMAL_BUDGET_SECONDS,
                "updated_at": utc_now(),
                "signals_sent": 0,
                "other_processes_modified": False,
                **extra,
            },
        )

    environment = command_environment(bundle_root, torch_home, campaign_root)
    environment["EVOMIND_SIIM_RUN_ID"] = run_id
    write_state("preflight")
    gpu = query_gpu()
    if "A800" not in gpu["name"] or gpu["memory_total_mib"] < 80_000:
        raise CampaignError("campaign is not running on the expected A800")

    candidate_training = (
        dict(evolution["training"])
        if evolution is not None
        else {
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
    )
    ablation_root = campaign_root / "ablation"
    ablation_id = f"{run_id}_ablation"
    ablation_run_dir = ensure_within(ablation_root / "runs" / ablation_id)
    ablation_report = ablation_run_dir / "siim_preprocessing_ablation.json"
    precomputed_ablation_manifest: pathlib.Path | None = None
    if evolution is not None and not ablation_report.is_file():
        preprocessing = dict(evolution["preprocessing"])
        parent_id = str(evolution["parent_run_id"])
        parent_ablation_id = f"{parent_id}_ablation"
        parent_ablation = ensure_within(
            ALLOWED_ROOT
            / f"siim_{BINDING.job_tag}"
            / "campaigns"
            / parent_id
            / "ablation"
            / "runs"
            / parent_ablation_id
        )
        sources = {
            "siim_preprocessing_ablation.json": (
                parent_ablation / "siim_preprocessing_ablation.json",
                str(preprocessing.get("report_sha256") or ""),
            ),
            "siim_image_content_manifest.csv": (
                parent_ablation / "siim_image_content_manifest.csv",
                str(preprocessing.get("image_content_manifest_sha256") or ""),
            ),
            "siim_duplicate_connected_groups.json": (
                parent_ablation / "siim_duplicate_connected_groups.json",
                str(preprocessing.get("duplicate_groups_sha256") or ""),
            ),
        }
        actions: dict[str, str] = {}
        for name, (source, expected_sha256) in sources.items():
            if sha256_file(source) != expected_sha256:
                raise CampaignError(f"parent preprocessing evidence hash changed: {name}")
            actions[name] = copy_immutable_file(source, ablation_run_dir / name)
        atomic_json(
            campaign_root / "ablation_parent_reuse.json",
            {
                "schema": "evomind.siim.parent_ablation_reuse.v1",
                "created_at": utc_now(),
                "run_id": run_id,
                "parent_run_id": parent_id,
                "parent_ablation_run_id": parent_ablation_id,
                "actions": actions,
                "hashes": {name: expected for name, (_source, expected) in sources.items()},
                "private_labels_used": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
                "signals_sent": 0,
                "other_processes_modified": False,
            },
        )
    if ablation_report.is_file():
        prior_ablation_checks = ablation_contract_checks(
            ablation_run_dir, expected_folds=3
        )
        if not prior_ablation_checks or not all(prior_ablation_checks.values()):
            prior_group_path = ablation_run_dir / "siim_duplicate_connected_groups.json"
            prior_group_report = (
                read_json(prior_group_path) if prior_group_path.is_file() else {}
            )
            preserved_ablation, supersession = supersede_directory(
                ablation_run_dir,
                reason=(
                    "perceptual_transitive_closure_reduced_requested_folds__"
                    "rerun_with_exact_duplicate_grouping"
                ),
            )
            precomputed_ablation_manifest = (
                preserved_ablation / "siim_image_content_manifest.csv"
            )
            if not precomputed_ablation_manifest.is_file():
                raise CampaignError(
                    "superseded ablation is missing its immutable image manifest"
                )
            atomic_json(
                campaign_root / "ablation_supersession.json",
                {
                    **supersession,
                    "run_id": run_id,
                    "old_contract_checks": prior_ablation_checks,
                    "perceptual_counterfactual": {
                        "largest_component_rows": int(
                            prior_group_report.get("largest_component_rows") or 0
                        ),
                        "train_rows": int(prior_group_report.get("rows") or 0),
                        "fraction": (
                            float(prior_group_report.get("largest_component_rows") or 0)
                            / max(1, int(prior_group_report.get("rows") or 0))
                        ),
                        "near_edges": int(
                            prior_group_report.get("perceptual_near_edges") or 0
                        ),
                        "edges_applied_to_corrected_groups": 0,
                    },
                    "precomputed_manifest": str(precomputed_ablation_manifest),
                    "precomputed_manifest_sha256": sha256_file(
                        precomputed_ablation_manifest
                    ),
                },
            )
    if not ablation_report.is_file():
        if time.time() - corrected_protocol_started >= 4 * 3600:
            write_state("needs_continuation", reason="ablation_budget_exhausted")
            return 75
        write_state("ablation_training")
        command = [
            str(runtime_python),
            str(bundle_root / "scripts" / "run_siim_preprocessing_ablation.py"),
            "--data-root", str(data_root),
            "--output-root", str(ablation_root),
            "--run-id", ablation_id,
            "--seed", "42",
            "--evaluation-seeds", "40,41,42",
            "--folds", "3",
            "--image-size", "224",
            "--batch-size", "128",
            "--memory-limit-mib", str(55 * 1024),
            "--workers", "8",
            "--manifest-workers", "8",
            "--full-backbone", "convnext_small",
            "--lesion-backbone", "efficientnet_v2_s",
            "--minimum-mean-gain", "0.0005",
            "--maximum-worst-fold-regression", "0.002",
            "--maximum-seed-mean-regression", "0.001",
            "--minimum-seed-pass-fraction", "0.6666666666666666",
            "--torch-home", str(torch_home),
        ]
        if precomputed_ablation_manifest is not None:
            command.extend(
                ["--image-content-manifest", str(precomputed_ablation_manifest)]
            )
        if (ablation_root / "runs" / ablation_id).exists():
            command.append("--resume")
        monitor = ResourceMonitor(
            run_id=run_id, root_pid=os.getpid(), telemetry=telemetry_path, control=control_path
        )
        code = run_phase(
            command,
            phase="ablation",
            log_root=log_root,
            environment=environment,
            monitor=monitor,
        )
        if code or not ablation_report.is_file():
            write_state("failed", phase="ablation", exit_code=code)
            return code or 2
        control = read_json(control_path)
        if control.get("status") == "pause_after_epoch":
            write_state("needs_continuation", phase="ablation", reason=control.get("reason"))
            return 75
    ablation = read_json(ablation_report)
    if ablation.get("passed") is not True:
        raise CampaignError("preprocessing ablation did not pass its contract")
    corrected_ablation_checks = ablation_contract_checks(
        ablation_run_dir, expected_folds=3
    )
    if not corrected_ablation_checks or not all(corrected_ablation_checks.values()):
        raise CampaignError("corrected preprocessing ablation contract is incomplete")
    selected_profile = str(ablation.get("selected_profile") or "")
    if corrected_formal_budget_started is None:
        corrected_formal_budget_started = time.time()

    probe_root = campaign_root / "batch_probes"
    selected_batch = int(existing.get("selected_batch_size") or 0)
    batch_probe = tuple(int(value) for value in candidate_training["batch_probe"])
    if selected_batch not in batch_probe:
        write_state("batch_probing", selected_profile=selected_profile)
        probe_root.mkdir(parents=True, exist_ok=True)
        for batch in batch_probe:
            report = probe_root / f"batch_{batch}.json"
            command = [
                str(runtime_python),
                str(bundle_root / "scripts" / "probe_siim_candidate_batch.py"),
                "--report", str(report),
                "--batch-size", str(batch),
                "--image-size", str(candidate_training["image_size"]),
                "--metadata-width", "16",
                "--full-backbone", str(candidate_training["full_backbone"]),
                "--lesion-backbone", str(candidate_training["lesion_backbone"]),
                "--torch-home", str(torch_home),
                "--expected-gpu-name", "A800",
                "--memory-limit-mib", str(candidate_training["memory_limit_mib"]),
                "--effective-batch-size", str(candidate_training["effective_batch_size"]),
            ]
            with (log_root / f"probe_{batch}.stdout.log").open("ab") as stdout, (
                log_root / f"probe_{batch}.stderr.log"
            ).open("ab") as stderr:
                code = subprocess.run(
                    command,
                    env=environment,
                    cwd=campaign_root,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                ).returncode
            payload = read_json(report) if report.is_file() else {}
            if code == 0 and payload.get("status") == "passed":
                selected_batch = batch
                break
        if not selected_batch:
            write_state("failed", phase="batch_probe", reason="no_passing_batch")
            return 3

    formal_root = campaign_root / "formal_runs"
    formal_root.mkdir(parents=True, exist_ok=True)
    formal_manifest_sha256 = sha256_file(
        ablation_run_dir / "siim_image_content_manifest.csv"
    )
    completed_seeds = []
    for record in plan.get("formal_runs") or []:
        seed = int(record["seed"])
        formal_id = str(record["run_id"])
        run_root = formal_root / formal_id
        result = candidate_result(run_root)
        result_checks = (
            corrected_formal_result_checks(
                run_root,
                result,
                expected_seed=seed,
                expected_manifest_sha256=formal_manifest_sha256,
                bundle_root=bundle_root,
            )
            if result
            else {}
        )
        if result and all_checks_pass(result_checks):
            completed_seeds.append(seed)
            continue
        if result and result.get("status") in KNOWN_TERMINAL_RESULT_STATES:
            preserved_run, supersession = supersede_directory(
                run_root,
                reason="terminal_result_failed_corrected_formal_contract",
            )
            atomic_json(
                campaign_root / f"formal_s{seed}_result_supersession.json",
                {
                    **supersession,
                    "run_id": run_id,
                    "formal_seed": seed,
                    "formal_run_id": formal_id,
                    "preserved_formal_run_root": str(preserved_run),
                    "superseded_result_status": result.get("status"),
                    "superseded_result_sha256": (
                        sha256_file(preserved_run / COMPETITION / "result.json")
                        if (preserved_run / COMPETITION / "result.json").is_file()
                        else None
                    ),
                    "checks": result_checks,
                },
            )
            result = None
        resume_root = siim_resume_root(run_root)
        existing_stage_checkpoints = (
            list(resume_root.glob("stage_*.pt")) if resume_root.is_dir() else []
        )
        resume_contract_path = resume_root / "resume_contract.json"
        resume_checks = (
            formal_resume_contract_checks(
                resume_contract_path,
                expected_seed=seed,
                expected_manifest_sha256=formal_manifest_sha256,
                bundle_root=bundle_root,
            )
            if existing_stage_checkpoints
            else {}
        )
        if existing_stage_checkpoints and not all_checks_pass(resume_checks):
            preserved_resume, supersession = supersede_directory(
                resume_root,
                reason=(
                    "checkpoint_chain_failed_corrected_formal_resume_contract"
                ),
            )
            atomic_json(
                campaign_root / f"formal_s{seed}_resume_supersession.json",
                {
                    **supersession,
                    "run_id": run_id,
                    "formal_seed": seed,
                    "formal_run_id": formal_id,
                    "preserved_resume_root": str(preserved_resume),
                    "checks": resume_checks,
                },
            )
        remaining = (
            corrected_formal_budget_started
            + CORRECTED_FORMAL_BUDGET_SECONDS
            - time.time()
        )
        if remaining <= 0:
            write_state(
                "needs_continuation",
                phase="formal_training",
                reason="formal_budget_exhausted",
                completed_seeds=completed_seeds,
                selected_profile=selected_profile,
                selected_batch_size=selected_batch,
            )
            return 75
        run_budget_seconds = resolve_formal_runtime_budget(
            run_root,
            model_seed=seed,
            remaining_seconds=remaining,
        )
        write_state(
            "formal_training",
            formal_seed=seed,
            completed_seeds=completed_seeds,
            selected_profile=selected_profile,
            selected_batch_size=selected_batch,
        )
        command = [
            str(runtime_python),
            str(bundle_root / "scripts" / "run_mlebench_lite_full.py"),
            "--data-root", str(data_root),
            "--output-root", str(formal_root),
            "--allowed-root", str(ALLOWED_ROOT),
            "--official-source-root", str(official_source_root),
            "--waves", "Wave0",
            "--competitions", COMPETITION,
            "--run-id", formal_id,
            "--seed", str(seed),
            "--phase-a-scope", "requested",
            "--candidate-only",
            "--hold-cuda-lease",
            "--siim-preprocessing-profile", selected_profile,
            "--siim-preprocessing-ablation-report", str(ablation_report),
            "--siim-image-content-manifest",
            str(ablation_run_dir / "siim_image_content_manifest.csv"),
            "--siim-backbone", str(candidate_training["full_backbone"]),
            "--siim-secondary-backbone", str(candidate_training["lesion_backbone"]),
            "--siim-epochs", str(candidate_training["epochs"]),
            "--siim-folds", "5",
            "--siim-inner-folds", "3",
            "--siim-batch-size", str(selected_batch),
            "--siim-image-size", str(candidate_training["image_size"]),
            "--siim-learning-rate", str(candidate_training["learning_rate"]),
            "--siim-metadata-iterations", str(candidate_training["metadata_iterations"]),
            "--siim-catboost-task-type", "GPU",
            "--siim-runtime-budget-seconds", str(run_budget_seconds),
            "--siim-control-file", str(control_path),
            "--wave2-fast-kernels",
        ]
        if evolution is None:
            command.extend([
                "--siim-effective-batch-size", "384",
                "--siim-memory-limit-mib", str(55 * 1024),
                "--siim-workers", "8",
            ])
        else:
            command.extend([
                "--siim-effective-batch-size", str(candidate_training["effective_batch_size"]),
                "--siim-memory-limit-mib", str(candidate_training["memory_limit_mib"]),
                "--siim-workers", str(candidate_training["max_workers"]),
            ])
        if run_root.exists():
            command.append("--resume")
        monitor = ResourceMonitor(
            run_id=run_id, root_pid=os.getpid(), telemetry=telemetry_path, control=control_path
        )
        code = run_phase(
            command,
            phase=f"formal_s{seed}",
            log_root=log_root,
            environment=environment,
            monitor=monitor,
        )
        result = candidate_result(run_root)
        if code == 3 and result and result.get("status") == "paused_resource_guard":
            write_state(
                "needs_continuation",
                phase="formal_training",
                formal_seed=seed,
                reason=result.get("pause_reason"),
                completed_seeds=completed_seeds,
                selected_profile=selected_profile,
                selected_batch_size=selected_batch,
            )
            return 75
        post_run_checks = (
            corrected_formal_result_checks(
                run_root,
                result,
                expected_seed=seed,
                expected_manifest_sha256=formal_manifest_sha256,
                bundle_root=bundle_root,
            )
            if result
            else {}
        )
        atomic_json(
            campaign_root / f"formal_s{seed}_result_validation.json",
            {
                "schema": "evomind.siim.corrected_formal_result_validation.v1",
                "created_at": utc_now(),
                "run_id": run_id,
                "formal_seed": seed,
                "formal_run_id": formal_id,
                "passed": all_checks_pass(post_run_checks),
                "checks": post_run_checks,
                "signals_sent": 0,
                "other_processes_modified": False,
            },
        )
        if code or not result or not all_checks_pass(post_run_checks):
            write_state("failed", phase="formal_training", formal_seed=seed, exit_code=code)
            return code or 4
        completed_seeds.append(seed)

    write_state(
        "awaiting_collection_and_freeze",
        completed_seeds=completed_seeds,
        selected_profile=selected_profile,
        selected_batch_size=selected_batch,
        ablation_report=str(ablation_report),
        telemetry_path=str(telemetry_path),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
