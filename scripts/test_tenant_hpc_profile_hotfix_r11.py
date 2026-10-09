"""Offline contract fixture for the two-file tenant HPC profile hotfix."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path


TENANT_ID = "tenant_0123456789abcdef01234567"
PROFILE = f"{TENANT_ID}_job91699_g15"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module spec unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _metadata(gpu) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": gpu.PROFILE_METADATA_SCHEMA,
        "credential_profile": PROFILE,
        "tenant_id": TENANT_ID,
        "job_id": 91699,
        "profile_state": gpu.PROFILE_STATE_ACTIVE,
        "allocation_binding_id": "aimslab-job91699-20260817T092437",
        "allocation_generation": 15,
        "profile_instance_id": "8d06ef65-129f-4845-a327-79580f3bc6ef",
        "lifecycle_revision": 2,
        "host": gpu.HPC_SSH_GATEWAY_HOST,
        "port": gpu.HPC_SSH_GATEWAY_PORT,
        "socks_host": "127.0.0.1",
        "socks_port": 7890,
        "socks_credential_path": gpu.PROFILE_SOCKS_CREDENTIAL_FILENAME,
        "known_hosts_path": gpu.PROFILE_KNOWN_HOSTS_FILENAME,
        "expected_host_uuid": "host-fixture",
        "expected_gpu_uuid": "GPU-fixture",
        "remote_workspace": gpu.ALLOWED_GPU_REMOTE_ROOT,
    }
    payload["container_binding_sha256"] = gpu.compute_container_binding_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-credentials", type=Path, required=True)
    parser.add_argument("--terminal-tools", type=Path, required=True)
    args = parser.parse_args()

    gpu = _load_module(
        "research_agent_workstation.server.core.gpu_credentials",
        args.gpu_credentials.resolve(strict=True),
    )
    terminal = _load_module(
        "xsci.terminal_tools_r11_candidate",
        args.terminal_tools.resolve(strict=True),
    )

    legacy = gpu.parse_named_hpc_profile_identity("job91699")
    tenant = gpu.parse_named_hpc_profile_identity(PROFILE)
    assert (legacy.job_id, legacy.tenant_id, legacy.allocation_generation) == (
        91699,
        None,
        None,
    )
    assert (tenant.job_id, tenant.tenant_id, tenant.allocation_generation) == (
        91699,
        TENANT_ID,
        15,
    )
    for invalid in (
        "job0",
        f"{TENANT_ID}_job0_g15",
        f"{TENANT_ID}_job91699_g0",
        "tenant_aaaaaaaaaaaaaaaaaaaaaaa_job91699_g15",
        f"{TENANT_ID}_job91699_g15_extra",
    ):
        try:
            gpu.parse_named_hpc_profile_identity(invalid)
        except gpu.CredentialError:
            pass
        else:
            raise AssertionError(f"malformed profile accepted: {invalid}")

    with tempfile.TemporaryDirectory(prefix="ev11-") as directory:
        root = Path(directory)
        profile_dir = root / PROFILE
        profile_dir.mkdir()
        metadata = _metadata(gpu)
        lifecycle = gpu._validate_named_profile_lifecycle(
            metadata,
            profile=PROFILE,
            profile_dir=profile_dir,
            allowed_states=frozenset({gpu.PROFILE_STATE_ACTIVE}),
        )
        assert lifecycle["allocation_generation"] == 15

        for field, value, expected in (
            ("tenant_id", "tenant_aaaaaaaaaaaaaaaaaaaaaaaa", "tenant binding changed"),
            ("job_id", 91493, "job binding changed"),
            ("allocation_generation", 14, "allocation generation changed"),
        ):
            drift = dict(metadata)
            drift[field] = value
            try:
                gpu._validate_named_profile_lifecycle(
                    drift,
                    profile=PROFILE,
                    profile_dir=profile_dir,
                    allowed_states=frozenset({gpu.PROFILE_STATE_ACTIVE}),
                )
            except gpu.CredentialError as exc:
                assert expected in str(exc)
            else:
                raise AssertionError(f"identity drift accepted: {field}")

        tombstone = profile_dir / gpu.PROFILE_FROZEN_TOMBSTONE_FILENAME
        tombstone.write_text("{}", encoding="utf-8")
        try:
            gpu._validate_named_profile_lifecycle(
                metadata,
                profile=PROFILE,
                profile_dir=profile_dir,
                allowed_states=frozenset({gpu.PROFILE_STATE_ACTIVE}),
            )
        except gpu.CredentialError as exc:
            assert "frozen or retired" in str(exc)
        else:
            raise AssertionError("tombstoned profile accepted")
        tombstone.unlink()

        appdata = root / "AppData"
        discovered_dir = appdata / "ResearchAgentWorkstation" / "profiles" / PROFILE
        discovered_dir.mkdir(parents=True)
        metadata_path = discovered_dir / gpu.PROFILE_METADATA_FILENAME
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        previous_appdata = os.environ.get("APPDATA")
        os.environ["APPDATA"] = str(appdata)
        try:
            selected, selected_path, selected_metadata, readiness = (
                terminal._latest_hpc_profile(root)
            )
        finally:
            if previous_appdata is None:
                os.environ.pop("APPDATA", None)
            else:
                os.environ["APPDATA"] = previous_appdata
        assert selected == PROFILE
        assert selected_path == metadata_path
        assert selected_metadata["allocation_generation"] == 15
        assert readiness == {}

    observed: dict[str, object] = {}

    class FakeClient:
        def close(self) -> None:
            observed["closed"] = True

    def fake_load(*, strict_named_profile: bool = False):
        observed["strict"] = strict_named_profile
        observed["profile"] = os.environ.get("EVOMIND_HPC_CREDENTIAL_PROFILE")
        return object()

    verify_identity = gpu.verify_job_container_identity
    gpu.load_gpu_ssh_config = fake_load
    gpu.connect_ssh = lambda config, timeout=30: FakeClient()
    gpu.verify_job_container_identity = lambda client, config, expected_job_id: {
        "job_id": expected_job_id,
        "credential_profile": PROFILE,
        "host_uuid": "fixture-host-uuid",
        "gpu_uuids": ["GPU-fixture-uuid"],
        "job_container_verified": True,
        "gpu_name": "NVIDIA A800-SXM4-80GB",
        "gpu_memory_total_mib": 81920,
        "remote_root": gpu.ALLOWED_GPU_REMOTE_ROOT,
        "designated_proxy_path_verified": True,
        "pinned_gateway_host_key_verified": True,
        "allocation_role_authenticated": True,
        "expected_host_uuid_match": True,
        "expected_gpu_uuid_match": True,
        "expected_gpu_model_and_memory_match": True,
        "allowed_remote_root_match": True,
        "read_only": True,
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    probe = terminal._live_hpc_connection_probe(PROFILE, 91699, sample_count=5)
    assert probe["ok"] is True
    assert probe["samples_passed"] == 5
    assert probe["signals_sent"] == 0
    assert probe["other_processes_modified"] is False
    assert observed == {"strict": True, "profile": PROFILE, "closed": True}

    mismatched = gpu.GpuSshConfig(
        host=gpu.HPC_SSH_GATEWAY_HOST,
        port=gpu.HPC_SSH_GATEWAY_PORT,
        username="fixture",
        password="test-only",
        socks=gpu.SocksConfig("127.0.0.1", 7890, "fixture", "test-only"),
        credential_profile=PROFILE,
        job_id=91699,
        strict_named_profile=True,
        allocation_generation=14,
    )

    class NoRemoteClient:
        def exec_command(self, *_args, **_kwargs):
            raise AssertionError("generation drift reached remote execution")

    try:
        verify_identity(
            NoRemoteClient(), mismatched, expected_job_id=91699
        )
    except gpu.CredentialError as exc:
        assert "allocation generation binding changed" in str(exc)
    else:
        raise AssertionError("generation drift accepted by container verifier")

    print(
        json.dumps(
            {
                "schema": "evomind.tenant_hpc_profile_hotfix_fixture.v1",
                "status": "passed",
                "legacy_profile": True,
                "tenant_profile": True,
                "identity_drift_rejected": 3,
                "tombstone_rejected": True,
                "tenant_profile_discovered": True,
                "offline_probe_samples": probe["samples_passed"],
                "remote_commands": 0,
                "training_started": False,
                "kaggle_submissions": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
