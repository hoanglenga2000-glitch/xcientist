from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from research_agent_workstation.server.core import gpu_credentials


def _load_verifier():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_hpc_profile_readiness.py"
    spec = importlib.util.spec_from_file_location("verify_hpc_profile_readiness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _profile_dir(appdata: Path, profile: str = "job90353") -> Path:
    path = appdata / "ResearchAgentWorkstation" / "profiles" / profile
    path.mkdir(parents=True)
    (path / gpu_credentials.PROFILE_CREDENTIAL_FILENAME).write_text("<credential />", encoding="utf-8")
    (path / gpu_credentials.PROFILE_KNOWN_HOSTS_FILENAME).write_text("gateway key\n", encoding="utf-8")
    return path


def _active_metadata(profile: str = "job90353") -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": gpu_credentials.PROFILE_METADATA_SCHEMA,
        "credential_profile": profile,
        "job_id": 90353,
        "profile_state": gpu_credentials.PROFILE_STATE_ACTIVE,
        "allocation_binding_id": "alloc-job90353-a",
        "allocation_generation": 1,
        "profile_instance_id": "12345678-1234-5678-9234-567812345678",
        "lifecycle_revision": 1,
        "host": gpu_credentials.HPC_SSH_GATEWAY_HOST,
        "port": gpu_credentials.HPC_SSH_GATEWAY_PORT,
        "socks_host": "127.0.0.1",
        "socks_port": 7890,
        "known_hosts_path": gpu_credentials.PROFILE_KNOWN_HOSTS_FILENAME,
        "expected_host_uuid": "host-job90353",
        "expected_gpu_uuid": "gpu-job90353",
        "remote_workspace": gpu_credentials.ALLOWED_GPU_REMOTE_ROOT,
    }
    payload["container_binding_sha256"] = gpu_credentials.compute_container_binding_sha256(payload)
    return payload


def test_hpc_profile_readiness_accepts_active_v2_profile(tmp_path):
    verifier = _load_verifier()
    appdata = tmp_path / "AppData"
    profile = _profile_dir(appdata)
    (profile / gpu_credentials.PROFILE_METADATA_FILENAME).write_text(
        json.dumps(_active_metadata(), indent=2),
        encoding="utf-8",
    )

    report = verifier.evaluate_profile("job90353", appdata=appdata)

    assert report["status"] == "ready"
    assert report["failed_checks"] == []
    assert report["boundaries"] == {
        "dpapi_decrypted": False,
        "ssh_connections": 0,
        "remote_commands": 0,
        "training_started": False,
        "grader_calls": 0,
        "kaggle_submissions": 0,
    }


def test_hpc_profile_readiness_fails_closed_on_legacy_metadata(tmp_path):
    verifier = _load_verifier()
    appdata = tmp_path / "AppData"
    profile = _profile_dir(appdata)
    (profile / gpu_credentials.PROFILE_METADATA_FILENAME).write_text(
        json.dumps({"credential_profile": "job90353", "job_id": 90353}),
        encoding="utf-8",
    )

    report = verifier.evaluate_profile("job90353", appdata=appdata)

    assert report["status"] == "failed_closed"
    assert "schema_v2" in report["failed_checks"]
    assert report["boundaries"]["ssh_connections"] == 0


def test_hpc_profile_readiness_fails_closed_on_tombstone(tmp_path):
    verifier = _load_verifier()
    appdata = tmp_path / "AppData"
    profile = _profile_dir(appdata)
    (profile / gpu_credentials.PROFILE_METADATA_FILENAME).write_text(
        json.dumps(_active_metadata(), indent=2),
        encoding="utf-8",
    )
    (profile / gpu_credentials.PROFILE_FROZEN_TOMBSTONE_FILENAME).write_text("{}", encoding="utf-8")

    report = verifier.evaluate_profile("job90353", appdata=appdata)

    assert report["status"] == "failed_closed"
    assert "no_frozen_tombstone" in report["failed_checks"]
