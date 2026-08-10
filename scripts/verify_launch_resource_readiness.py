from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

HPC_GATEWAY_HOST = "100.85.169.63"
HPC_GATEWAY_PORT = 1235
HPC_SOCKS_HOST = "127.0.0.1"
HPC_SOCKS_PORT = 7890
ALLOWED_HPC_REMOTE_ROOT = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"
DPAPI_PROFILE_SCHEMA_V2 = "evomind.hpc.dpapi_profile.v2"
SMOKE_POINTER_SCHEMA = "evomind.hpc.bounded_smoke_pointer.v1"


TASKS = {
    "house_prices": {
        "config": ROOT / "configs" / "house_prices.yaml",
        "experiment_root": ROOT / "experiments" / "house_prices",
        "required_inputs": [
            ROOT / "tasks" / "house_prices" / "overview.txt",
            ROOT / "tasks" / "house_prices" / "data" / "train.csv",
            ROOT / "tasks" / "house_prices" / "data" / "test.csv",
            ROOT / "tasks" / "house_prices" / "data" / "sample_submission.csv",
        ],
        "metric_checks": [
            ("cv_rmsle_mean", "<=", 0.18),
            ("holdout_rmsle", "<=", 0.20),
            ("submission_rows", "==", 1459),
        ],
    },
    "titanic": {
        "config": ROOT / "configs" / "titanic.yaml",
        "experiment_root": ROOT / "experiments" / "titanic",
        "required_inputs": [
            ROOT / "tasks" / "titanic" / "overview.txt",
            ROOT / "tasks" / "titanic" / "data" / "train.csv",
            ROOT / "tasks" / "titanic" / "data" / "test.csv",
            ROOT / "tasks" / "titanic" / "data" / "sample_submission.csv",
        ],
        "metric_checks": [
            ("cv_accuracy_mean", ">=", 0.78),
        ],
    },
    "telco_churn": {
        "config": ROOT / "configs" / "telco_churn.yaml",
        "experiment_root": ROOT / "experiments" / "telco_churn",
        "required_inputs": [
            ROOT / "tasks" / "telco_churn" / "overview.txt",
            ROOT / "tasks" / "telco_churn" / "data" / "train.csv",
            ROOT / "tasks" / "telco_churn" / "data" / "test.csv",
            ROOT / "tasks" / "telco_churn" / "data" / "sample_submission.csv",
        ],
        "metric_checks": [
            ("cv_accuracy_mean", ">=", 0.78),
        ],
    },
}


CODE_AGENT_ENV = ["DEEPSEEK_API_KEY or ANTHROPIC_API_KEY"]
GPU_ENV = ["GPU_SSH_HOST", "GPU_SSH_USER", "GPU_SSH_PASSWORD or GPU_SSH_KEY_PATH", "GPU_REMOTE_WORKSPACE"]
KAGGLE_ENV = ["KAGGLE_USERNAME", "KAGGLE_KEY"]
HPC_PROBE = ROOT / "workspace" / "hpc" / "web_terminal_probe.txt"
LOCAL_READY_ARTIFACTS = [
    "validation_gate.json",
    "experiment_log.json",
    "workflow_stage_audit.json",
    "submission.csv",
    "model_results.json",
]


def read_file_if_present(file_path: str | None) -> str:
    if not file_path:
        return ""
    try:
        return Path(file_path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def secret_dir_file(names: list[str]) -> Path | None:
    secret_dir = os.environ.get("WORKSTATION_SECRET_DIR")
    if not secret_dir:
        return None
    for name in names:
        candidate = Path(secret_dir) / name
        if candidate.exists():
            return candidate
    return None


def secret_value(key: str, aliases: list[str] | None = None) -> str:
    keys = [key, *(aliases or [])]
    for candidate in keys:
        direct = os.environ.get(candidate)
        if direct:
            return direct
        file_value = read_file_if_present(os.environ.get(f"{candidate}_FILE"))
        if file_value:
            return file_value
    dir_file = secret_dir_file(keys)
    return read_file_if_present(str(dir_file) if dir_file else None)


def secret_path(key: str, names: list[str] | None = None) -> str:
    direct = os.environ.get(key) or os.environ.get(f"{key}_FILE")
    if direct:
        return direct
    dir_file = secret_dir_file([key, *(names or [])])
    return str(dir_file) if dir_file else ""


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_metadata_candidates() -> list[Path]:
    appdata = Path(os.environ.get("APPDATA", "")).expanduser()
    profiles_root = appdata / "ResearchAgentWorkstation" / "profiles"
    requested = os.environ.get("EVOMIND_HPC_CREDENTIAL_PROFILE", "").strip()
    if requested:
        return [profiles_root / requested / "hpc_ssh_metadata.json"]
    return sorted(
        profiles_root.glob("job*/hpc_ssh_metadata.json"),
        key=lambda item: item.stat().st_mtime_ns if item.exists() else 0,
        reverse=True,
    )


def _safe_metadata_summary(metadata_path: Path | None, metadata: dict[str, Any]) -> dict[str, Any]:
    profile = str(metadata.get("credential_profile") or (metadata_path.parent.name if metadata_path else ""))
    gpu_uuid = str(metadata.get("expected_gpu_uuid") or metadata.get("gpu_uuid") or "")
    return {
        "profile": profile,
        "job_id": int(metadata.get("job_id") or 0),
        "profile_state": str(metadata.get("profile_state") or "unknown"),
        "schema": str(metadata.get("schema") or ""),
        "gateway_host": str(metadata.get("host") or ""),
        "gateway_port": int(metadata.get("port") or 0),
        "socks_host": str(metadata.get("socks_host") or ""),
        "socks_port": int(metadata.get("socks_port") or 0),
        "remote_workspace": str(metadata.get("remote_workspace") or ""),
        "host_uuid_present": bool(str(metadata.get("expected_host_uuid") or "").strip()),
        "gpu_uuid_suffix": gpu_uuid[-12:] if gpu_uuid else "",
        "container_binding_sha256_present": bool(str(metadata.get("container_binding_sha256") or "").strip()),
        "metadata_path": str(metadata_path) if metadata_path else "",
    }


def _load_active_hpc_profile(root: Path) -> tuple[Path | None, dict[str, Any], list[str]]:
    failures: list[str] = []
    for metadata_path in _profile_metadata_candidates():
        if not metadata_path.is_file():
            failures.append("metadata_file_missing")
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append("metadata_parse_failed")
            continue
        profile = str(metadata.get("credential_profile") or metadata_path.parent.name)
        job_id = int(metadata.get("job_id") or 0)
        expected_profile = f"job{job_id}" if job_id > 0 else ""
        local_failures: list[str] = []
        if metadata.get("schema") != DPAPI_PROFILE_SCHEMA_V2:
            local_failures.append("schema_v2_required")
        if metadata.get("profile_state") != "active":
            local_failures.append("profile_state_not_active")
        if profile != expected_profile:
            local_failures.append("profile_job_binding_mismatch")
        if str(metadata.get("host") or "") != HPC_GATEWAY_HOST or int(metadata.get("port") or 0) != HPC_GATEWAY_PORT:
            local_failures.append("gateway_binding_mismatch")
        if str(metadata.get("socks_host") or "") != HPC_SOCKS_HOST or int(metadata.get("socks_port") or 0) != HPC_SOCKS_PORT:
            local_failures.append("designated_socks_binding_mismatch")
        if str(metadata.get("remote_workspace") or "").rstrip("/") != ALLOWED_HPC_REMOTE_ROOT:
            local_failures.append("remote_root_mismatch")
        if not str(metadata.get("expected_host_uuid") or "").strip():
            local_failures.append("expected_host_uuid_missing")
        if not str(metadata.get("expected_gpu_uuid") or "").strip():
            local_failures.append("expected_gpu_uuid_missing")
        if not str(metadata.get("container_binding_sha256") or "").strip():
            local_failures.append("container_binding_sha256_missing")
        if (metadata_path.parent / "hpc_profile_frozen.tombstone.json").exists():
            local_failures.append("profile_frozen_tombstone_present")
        if (metadata_path.parent / "hpc_profile_retired.tombstone.json").exists():
            local_failures.append("profile_retired_tombstone_present")
        inner_endpoint = str(metadata.get("allocation_inner_endpoint") or metadata.get("inner_endpoint") or "")
        if inner_endpoint.startswith("10.120."):
            # The allocation-page private IP may exist as metadata only.  It must
            # never replace the gateway/proxy binding above.
            pass
        if local_failures:
            failures.extend(local_failures)
            continue
        return metadata_path, metadata, []
    return None, {}, failures or ["no_active_named_profile"]


def _validate_profile_readiness(root: Path, profile: str, job_id: int) -> dict[str, Any]:
    path = root / "workspace" / "hpc" / f"{profile}_profile_readiness_current.json"
    payload = read_json(path)
    failures: list[str] = []
    if not payload:
        failures.append("profile_readiness_missing_or_invalid")
    else:
        details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
        if payload.get("status") != "ready":
            failures.append("profile_readiness_not_ready")
        if payload.get("profile") != profile:
            failures.append("profile_readiness_profile_mismatch")
        if int(details.get("job_id") or 0) != int(job_id):
            failures.append("profile_readiness_job_mismatch")
        if payload.get("failed_checks"):
            failures.append("profile_readiness_failed_checks_present")
        if details.get("gateway_host") != HPC_GATEWAY_HOST or int(details.get("gateway_port") or 0) != HPC_GATEWAY_PORT:
            failures.append("profile_readiness_gateway_mismatch")
        if details.get("socks_host") != HPC_SOCKS_HOST or int(details.get("socks_port") or 0) != HPC_SOCKS_PORT:
            failures.append("profile_readiness_socks_mismatch")
        if str(details.get("remote_workspace") or "").rstrip("/") != ALLOWED_HPC_REMOTE_ROOT:
            failures.append("profile_readiness_remote_root_mismatch")
    return {
        "ok": not failures,
        "path": str(path.relative_to(root)),
        "status": payload.get("status") if payload else "missing",
        "failed_checks": payload.get("failed_checks", []) if payload else [],
        "failures": failures,
    }


def _smoke_pointer_candidates(root: Path, profile: str) -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get("EVOMIND_HPC_BOUNDED_SMOKE_CURRENT", "").strip()
    if override:
        candidates.append(Path(override))
    candidates.extend(
        [
            root / "workspace" / "hpc" / f"{profile}_bounded_smoke_current.json",
            root / "workspace" / "hpc" / f"{profile}_bounded_smoke_collection_current.json",
            root / "workspace" / "hpc" / "bounded_smoke_current.json",
        ]
    )
    return candidates


def _read_smoke_pointer(root: Path, profile: str) -> tuple[Path | None, dict[str, Any] | None, list[str]]:
    failures: list[str] = []
    for path in _smoke_pointer_candidates(root, profile):
        if not path.is_file():
            failures.append(f"missing_pointer:{path.name}")
            continue
        payload = read_json(path)
        if not payload:
            failures.append("smoke_pointer_invalid_json")
            continue
        if payload.get("schema") != SMOKE_POINTER_SCHEMA:
            failures.append("smoke_pointer_schema_mismatch")
            continue
        if payload.get("profile") != profile:
            failures.append("smoke_pointer_profile_mismatch")
            continue
        return path, payload, []
    return None, None, failures or ["bounded_smoke_pointer_missing"]


def _validate_bounded_smoke(root: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    profile = str(metadata.get("credential_profile") or "")
    job_id = int(metadata.get("job_id") or 0)
    pointer_path, pointer, pointer_failures = _read_smoke_pointer(root, profile)
    failures: list[str] = list(pointer_failures)
    collection: dict[str, Any] | None = None
    if pointer:
        collection_path = Path(str(pointer.get("collection_path") or ""))
        if not collection_path.is_file():
            failures.append("bounded_smoke_collection_missing")
        elif str(pointer.get("collection_sha256") or "").lower() != _hash_file(collection_path).lower():
            failures.append("bounded_smoke_collection_hash_mismatch")
        else:
            collection = read_json(collection_path)
            if not collection:
                failures.append("bounded_smoke_collection_invalid_json")
        required_files = pointer.get("required_files") if isinstance(pointer.get("required_files"), dict) else {}
        evidence_root = Path(str(pointer.get("evidence_root") or collection_path.parent if pointer else ""))
        for name, expected_hash in required_files.items():
            file_path = evidence_root / str(name)
            if not file_path.is_file():
                failures.append(f"bounded_smoke_required_file_missing:{name}")
            elif str(expected_hash).lower() != _hash_file(file_path).lower():
                failures.append(f"bounded_smoke_required_file_hash_mismatch:{name}")

    if collection:
        identity = collection.get("identity_gate") if isinstance(collection.get("identity_gate"), dict) else {}
        smoke = collection.get("gpu_smoke") if isinstance(collection.get("gpu_smoke"), dict) else {}
        expected_gpu = str(metadata.get("expected_gpu_uuid") or "").strip().lower()
        observed_gpu = str(smoke.get("gpu_uuid") or "").strip().lower()
        collection_gpu_values = {str(value).strip().lower() for value in identity.get("gpu_uuids", []) if str(value).strip()}
        if collection.get("profile") != profile or int(collection.get("job_id") or 0) != job_id:
            failures.append("bounded_smoke_profile_job_mismatch")
        if identity.get("job_container_verified") is not True:
            failures.append("bounded_smoke_identity_not_verified")
        if identity.get("designated_proxy_path_verified") is not True:
            failures.append("bounded_smoke_proxy_path_not_verified")
        if identity.get("pinned_host_key_verified") is not True:
            failures.append("bounded_smoke_pinned_host_key_not_verified")
        if str(identity.get("remote_root") or "").rstrip("/") != ALLOWED_HPC_REMOTE_ROOT:
            failures.append("bounded_smoke_remote_root_mismatch")
        if str(identity.get("host_uuid") or "").strip().lower() != str(metadata.get("expected_host_uuid") or "").strip().lower():
            failures.append("bounded_smoke_host_uuid_mismatch")
        if expected_gpu not in collection_gpu_values or observed_gpu != expected_gpu:
            failures.append("bounded_smoke_gpu_uuid_mismatch")
        if "A800" not in str(identity.get("gpu_name") or smoke.get("device_name") or ""):
            failures.append("bounded_smoke_gpu_model_mismatch")
        if int(identity.get("gpu_memory_total_mib") or 0) < 80_000:
            failures.append("bounded_smoke_gpu_memory_mismatch")
        if smoke.get("status") != "passed":
            failures.append("bounded_smoke_status_not_passed")
        if smoke.get("remote_write_boundary_ok") is not True:
            failures.append("bounded_smoke_remote_write_boundary_failed")
        for key, expected in {
            "residual_running_processes": 0,
            "training_started": False,
            "signals_sent": 0,
            "other_processes_modified": False,
        }.items():
            if smoke.get(key) != expected:
                failures.append(f"bounded_smoke_{key}_violated")
        if collection.get("signals_sent") != 0 or collection.get("other_processes_modified") is not False:
            failures.append("bounded_smoke_collection_process_boundary_violated")
        remote_run_dir = str(collection.get("remote_run_dir") or smoke.get("run_dir") or "")
        allowed_prefix = ALLOWED_HPC_REMOTE_ROOT.rstrip("/") + f"/evomind_{profile}_smoke/"
        if not remote_run_dir.startswith(allowed_prefix):
            failures.append("bounded_smoke_remote_run_dir_outside_allowed_job_scope")

    return {
        "ok": not failures,
        "pointer_path": str(pointer_path.relative_to(root)) if pointer_path and pointer_path.is_relative_to(root) else str(pointer_path or ""),
        "collection_path": str(pointer.get("collection_path") if pointer else ""),
        "status": "passed" if collection and not failures else "blocked",
        "run_id": collection.get("run_id") if collection else "",
        "failures": failures,
        "remote_run_dir": collection.get("remote_run_dir") if collection else "",
    }


def _live_probe(profile: str, job_id: int, sample_count: int = 5) -> dict[str, Any]:
    try:
        from xsci.terminal_tools import _live_hpc_connection_probe
    except Exception as exc:
        return {
            "ok": False,
            "status": "live_probe_import_failed",
            "job_container_verified": False,
            "samples_requested": sample_count,
            "samples_passed": 0,
            "error_type": type(exc).__name__,
        }
    return _live_hpc_connection_probe(profile, job_id, sample_count=sample_count)


def strict_hpc_runtime_status(
    root: Path = ROOT,
    *,
    live_probe: Callable[[str, int, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metadata_path, metadata, profile_failures = _load_active_hpc_profile(root)
    profile_summary = _safe_metadata_summary(metadata_path, metadata)
    if not metadata:
        return {
            "configured": False,
            "state": "no_active_named_dpapi_profile",
            "missing_keys": profile_failures,
            "profile": profile_summary,
            "claim_boundary": "named DPAPI profile required; legacy evidence is not release proof",
        }
    profile = str(metadata.get("credential_profile") or "")
    job_id = int(metadata.get("job_id") or 0)
    readiness = _validate_profile_readiness(root, profile, job_id)
    probe_func = live_probe or (lambda p, j, n: _live_probe(p, j, sample_count=n))
    live = probe_func(profile, job_id, 5)
    smoke = _validate_bounded_smoke(root, metadata)
    failures = list(profile_failures) + readiness["failures"] + smoke["failures"]
    if live.get("ok") is not True or live.get("job_container_verified") is not True or int(live.get("samples_passed") or 0) != 5:
        failures.append(str(live.get("status") or "live_hpc_probe_failed"))
    if live.get("signals_sent") != 0 or live.get("other_processes_modified") is not False:
        failures.append("live_hpc_probe_process_boundary_violated")
    configured = not failures
    return {
        "configured": configured,
        "state": "strict_hpc_runtime_verified" if configured else "blocked_hpc_runtime_verification_required",
        "profile": profile_summary,
        "profile_readiness": readiness,
        "live_probe": live,
        "bounded_smoke": smoke,
        "missing_keys": failures,
        "required_keys": [
            "active named DPAPI profile",
            "profile readiness current report",
            "5/5 live job-container identity samples",
            "bounded GPU smoke collection with hashes",
        ],
        "claim_boundary": "strict release proof requires current profile, live probe, and bounded smoke; legacy Web Terminal text is ignored",
        "secrets_returned": False,
    }


def hpc_probe_status() -> dict[str, Any]:
    if not HPC_PROBE.is_file():
        return {
            "configured": False,
            "path": str(HPC_PROBE.relative_to(ROOT)),
            "state": "missing_web_terminal_probe",
            "required_keys": [],
            "missing_keys": ["workspace/hpc/web_terminal_probe.txt"],
            "detail": "No Web Terminal nvidia-smi evidence file is present.",
        }
    text = HPC_PROBE.read_text(encoding="utf-8", errors="replace")
    required = ["whoami", "hostname", "pwd", "Python", "NVIDIA-SMI", "Filesystem", "Mem:"]
    missing = [term for term in required if term not in text]
    a800_hits = text.count("NVIDIA A800") + text.count("NVIDIAA800") + text.count("A800-SXM4")
    return {
        "configured": not missing and a800_hits >= 4,
        "path": str(HPC_PROBE.relative_to(ROOT)),
        "state": "gpu_verified_via_login_node_web_terminal" if not missing and a800_hits >= 4 else "incomplete_web_terminal_probe",
        "required_keys": [],
        "missing_keys": missing,
        "missing_terms": missing,
        "a800_text_hits": a800_hits,
        "detail": "4 x NVIDIA A800 is proven by Web Terminal evidence." if not missing and a800_hits >= 4 else "GPU evidence is present but incomplete.",
    }


def latest_experiment(root: Path) -> Path | None:
    if not root.exists():
        return None
    runs = sorted(path for path in root.iterdir() if path.is_dir())
    for run_dir in reversed(runs):
        if all((run_dir / name).exists() and (run_dir / name).stat().st_size > 0 for name in LOCAL_READY_ARTIFACTS):
            return run_dir
    return runs[-1] if runs else None


def metric_pass(value: Any, operator: str, threshold: float) -> bool:
    if not isinstance(value, (int, float)):
        return False
    if operator == "<=":
        return value <= threshold
    if operator == ">=":
        return value >= threshold
    if operator == "==":
        return value == threshold
    raise ValueError(f"unsupported operator: {operator}")


def env_status(keys: list[str], aliases: dict[str, list[str]] | None = None, path_keys: dict[str, list[str]] | None = None) -> dict[str, Any]:
    aliases = aliases or {}
    path_keys = path_keys or {}
    missing = []
    for key in keys:
        if key in path_keys:
            configured = bool(secret_path(key, path_keys[key]))
        else:
            configured = bool(secret_value(key, aliases.get(key, [])))
        if not configured:
            missing.append(key)
    return {
        "configured": not missing,
        "required_keys": keys,
        "missing_keys": missing,
        "accepted_secret_sources": ["direct env", "*_FILE", "WORKSTATION_SECRET_DIR"],
    }


def code_agent_status() -> dict[str, Any]:
    configured = bool(secret_value("DEEPSEEK_API_KEY") or secret_value("ANTHROPIC_API_KEY", ["CLAUDE_API_KEY"]))
    return {
        "configured": configured,
        "required_keys": CODE_AGENT_ENV,
        "missing_keys": [] if configured else CODE_AGENT_ENV,
        "accepted_secret_sources": ["direct env", "*_FILE", "WORKSTATION_SECRET_DIR"],
    }


def gpu_gateway_status() -> dict[str, Any]:
    missing = []
    for key in ["GPU_SSH_HOST", "GPU_SSH_USER", "GPU_REMOTE_WORKSPACE"]:
        if not secret_value(key):
            missing.append(key)
    if not (secret_value("GPU_SSH_PASSWORD", ["HPC_SSH_PASSWORD"]) or secret_path("GPU_SSH_KEY_PATH", ["GPU_SSH_PRIVATE_KEY", "gpu_ssh_private_key", "id_rsa"])):
        missing.append("GPU_SSH_PASSWORD or GPU_SSH_KEY_PATH")
    return {
        "configured": not missing,
        "required_keys": GPU_ENV,
        "missing_keys": missing,
        "accepted_secret_sources": ["direct env", "*_FILE", "WORKSTATION_SECRET_DIR"],
    }


def inspect_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = latest_experiment(spec["experiment_root"])
    validation = read_json(run_dir / "validation_gate.json") if run_dir else None
    experiment_log = read_json(run_dir / "experiment_log.json") if run_dir else None

    metrics = {}
    if validation and isinstance(validation.get("metrics"), dict):
        metrics.update(validation["metrics"])
    if validation:
        for key in ("cv_rmsle_mean", "holdout_rmsle", "submission_rows", "cv_accuracy_mean", "holdout_accuracy"):
            if key in validation:
                metrics[key] = validation[key]
    if experiment_log and isinstance(experiment_log.get("best_metrics"), dict):
        metrics.update(experiment_log["best_metrics"])

    metric_checks = []
    for metric_name, operator, threshold in spec["metric_checks"]:
        value = metrics.get(metric_name)
        metric_checks.append(
            {
                "metric": metric_name,
                "operator": operator,
                "threshold": threshold,
                "value": value,
                "passed": metric_pass(value, operator, threshold),
            }
        )

    required_artifacts = []
    if run_dir:
        for name in [
            "validation_gate.json",
            "experiment_log.json",
            "workflow_stage_audit.json",
            "submission.csv",
            "model_results.json",
        ]:
            required_artifacts.append({"path": str((run_dir / name).relative_to(ROOT)), "exists": (run_dir / name).exists()})

    missing_inputs = [str(path.relative_to(ROOT)) for path in spec["required_inputs"] if not path.exists()]
    missing_artifacts = [item["path"] for item in required_artifacts if not item["exists"]]
    validation_passed = bool(validation and validation.get("status") == "passed")

    return {
        "task_id": task_id,
        "config_exists": spec["config"].exists(),
        "latest_experiment": str(run_dir.relative_to(ROOT)) if run_dir else None,
        "data_ready": not missing_inputs,
        "missing_inputs": missing_inputs,
        "validation_gate_passed": validation_passed,
        "metrics": metrics,
        "metric_checks": metric_checks,
        "required_artifacts": required_artifacts,
        "artifact_ready": not missing_artifacts,
        "missing_artifacts": missing_artifacts,
        "ready_for_local_training": bool(not missing_inputs and validation_passed and not missing_artifacts and all(item["passed"] for item in metric_checks)),
    }


def write_markdown(report: dict[str, Any], target: Path) -> None:
    lines = [
        "# 科研 Agent 工作站上线资源就绪审计",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 总体状态：{report['overall_status']}",
        f"- 本地 Kaggle 风格训练闭环：{report['local_training_status']}",
        f"- 严格 HPC/GPU 运行时：{report['strict_hpc_runtime_status']}",
        "",
        "## 结论",
        "",
        report["conclusion"],
        "",
        "## 外部资源状态",
        "",
    ]
    for key, item in report["external_resources"].items():
        lines.append(f"- {key}: {'已配置' if item['configured'] else '未配置'}")
        if item["missing_keys"]:
            lines.append(f"  缺少：{', '.join(item['missing_keys'])}")
    lines.extend(
        [
            "",
            "## Kaggle 数据训练任务",
            "",
        ]
    )
    for task in report["tasks"]:
        lines.append(f"### {task['task_id']}")
        lines.append(f"- 最新实验：{task['latest_experiment']}")
        lines.append(f"- 数据就绪：{task['data_ready']}")
        lines.append(f"- Validation Gate：{task['validation_gate_passed']}")
        lines.append(f"- 本地训练就绪：{task['ready_for_local_training']}")
        for check in task["metric_checks"]:
            lines.append(
                f"- 指标 {check['metric']} {check['operator']} {check['threshold']}："
                f" 当前 {check['value']}，{'通过' if check['passed'] else '未通过'}"
            )
        if task["missing_inputs"]:
            lines.append(f"- 缺失输入：{', '.join(task['missing_inputs'])}")
        if task["missing_artifacts"]:
            lines.append(f"- 缺失产物：{', '.join(task['missing_artifacts'])}")
        lines.append("")

    lines.extend(
        [
            "## 配置后直接执行",
            "",
            "1. 配置 Claude：`ANTHROPIC_API_KEY`。",
            "2. 配置 GPU SSH：`GPU_SSH_HOST`、`GPU_SSH_USER`、`GPU_SSH_PASSWORD` 或 `GPU_SSH_KEY_PATH`、`GPU_REMOTE_WORKSPACE`。",
            "3. 可选配置 Kaggle 官方下载/提交：`KAGGLE_USERNAME`、`KAGGLE_KEY`。",
            "4. 重新运行：`python scripts\\verify_launch_resource_readiness.py --write-report`。",
            "5. 浏览器打开 `http://127.0.0.1:8088`，在 Code Runner 中启动 Claude Session 或 GPU Job。",
            "",
            "说明：Kaggle token 只影响官方 API 下载/leaderboard 提交；当前本地训练使用已验证的 Kaggle 风格输入文件，因此不把 Kaggle token 计入本轮最后两项阻塞资源。",
            "",
        ]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify launch readiness for Kaggle-style training and external resources.")
    parser.add_argument("--write-report", action="store_true", help="Write JSON and Markdown readiness reports under docs/.")
    args = parser.parse_args()

    tasks = [inspect_task(task_id, spec) for task_id, spec in TASKS.items()]
    local_ready = all(task["ready_for_local_training"] for task in tasks)
    legacy_hpc_probe = hpc_probe_status()
    strict_hpc = strict_hpc_runtime_status(ROOT)
    external_resources = {
        "code_agent": code_agent_status(),
        "gpu_ssh_gateway_legacy_env": gpu_gateway_status(),
        "hpc_gpu_legacy_web_terminal": legacy_hpc_probe,
        "hpc_gpu_strict_runtime": strict_hpc,
        "kaggle_official_api_optional": env_status(KAGGLE_ENV),
    }
    required_external_ready = external_resources["code_agent"]["configured"] and strict_hpc["configured"]
    missing_required_external = (
        external_resources["code_agent"]["missing_keys"]
        + strict_hpc["missing_keys"]
    )
    conclusion = (
        "本地 Kaggle 风格数据训练、指标阈值、submission 和审计产物已经就绪；"
        "严格 HPC/GPU 运行时证据已通过；正式进入外部增强训练/自动代码优化前，还缺代码 Agent 密钥。"
        if local_ready and not required_external_ready and strict_hpc["configured"]
        else "本地 Kaggle 风格数据训练、指标阈值、submission 和审计产物已经就绪；"
        "正式进入外部增强训练/自动代码优化前，仍缺代码 Agent 密钥或严格 HPC/GPU 运行时证据。"
        if local_ready and not required_external_ready
        else "本地训练或外部资源仍有未满足项，请查看下方明细。"
    )
    if local_ready and required_external_ready:
        conclusion = "本地训练、代码 Agent 与严格 HPC/GPU 运行时均已就绪，可以启动受控增强训练流程。"

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall_status": "ready_for_external_resources" if local_ready and not required_external_ready else "fully_ready" if local_ready and required_external_ready else "not_ready",
        "local_training_status": "ready" if local_ready else "not_ready",
        "required_external_status": "ready" if required_external_ready else "missing",
        "strict_hpc_runtime_status": "ready" if strict_hpc["configured"] else "blocked",
        "missing_required_external_keys": missing_required_external,
        "legacy_local_evidence": "not_release_readiness",
        "external_resources": external_resources,
        "tasks": tasks,
        "conclusion": conclusion,
    }

    if args.write_report:
        json_path = ROOT / "docs" / "launch_resource_readiness.json"
        md_path = ROOT / "docs" / "上线资源接入与Kaggle训练就绪审计.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(report, md_path)
        report["report_paths"] = {
            "json": str(json_path.relative_to(ROOT)),
            "markdown": str(md_path.relative_to(ROOT)),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not local_ready or not required_external_ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
