from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from research_agent_workstation.server.core import gpu_credentials
from scripts import manage_hpc_profile_lifecycle as lifecycle


def _profile(
    tmp_path: Path,
    *,
    state: str = gpu_credentials.PROFILE_STATE_ACTIVE,
) -> tuple[Path, dict[str, object]]:
    profile = "job90353"
    directory = tmp_path / "ResearchAgentWorkstation" / "profiles" / profile
    directory.mkdir(parents=True)
    metadata: dict[str, object] = {
        "schema": gpu_credentials.PROFILE_METADATA_SCHEMA,
        "credential_profile": profile,
        "job_id": 90353,
        "profile_state": state,
        "profile_state_reason": "fixture",
        "allocation_binding_id": "allocation-fixture-90353",
        "allocation_generation": 3,
        "profile_instance_id": str(uuid.uuid5(uuid.NAMESPACE_URL, profile)),
        "lifecycle_revision": 2 if state == gpu_credentials.PROFILE_STATE_ACTIVE else 1,
        "host": gpu_credentials.HPC_SSH_GATEWAY_HOST,
        "port": gpu_credentials.HPC_SSH_GATEWAY_PORT,
        "socks_host": "127.0.0.1",
        "socks_port": 17897,
        "known_hosts_path": gpu_credentials.PROFILE_KNOWN_HOSTS_FILENAME,
        "expected_host_uuid": "host-job90353",
        "expected_gpu_uuid": "gpu-job90353",
        "remote_workspace": gpu_credentials.ALLOWED_GPU_REMOTE_ROOT,
    }
    if state == gpu_credentials.PROFILE_STATE_ACTIVE:
        metadata["container_binding_sha256"] = (
            gpu_credentials.compute_container_binding_sha256(metadata)
        )
    (directory / gpu_credentials.PROFILE_METADATA_FILENAME).write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return directory, metadata


def test_freeze_writes_irreversible_tombstone_before_disabling_metadata(tmp_path: Path) -> None:
    directory, original = _profile(tmp_path)
    result = lifecycle.transition_profile(
        profile="job90353",
        target_state=gpu_credentials.PROFILE_STATE_FROZEN,
        reason="allocation_expired",
        now=datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc),
        appdata=tmp_path,
    )
    metadata = json.loads(
        (directory / gpu_credentials.PROFILE_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    tombstone_path = directory / gpu_credentials.PROFILE_FROZEN_TOMBSTONE_FILENAME
    tombstone = json.loads(tombstone_path.read_text(encoding="utf-8"))

    assert result["profile_state"] == gpu_credentials.PROFILE_STATE_FROZEN
    assert result["credential_decrypted"] is False
    assert result["network_accessed"] is False
    assert metadata["profile_state"] == gpu_credentials.PROFILE_STATE_FROZEN
    assert metadata["lifecycle_revision"] == int(original["lifecycle_revision"]) + 1
    assert tombstone["previous_state"] == gpu_credentials.PROFILE_STATE_ACTIVE
    assert tombstone["profile_state"] == gpu_credentials.PROFILE_STATE_FROZEN
    assert tombstone["allocation_generation"] == 3
    assert tombstone["credential_decrypted"] is False
    assert metadata["tombstone_sha256"][tombstone_path.name] == result["tombstone_sha256"]

    with pytest.raises(lifecycle.LifecycleError, match="cannot transition"):
        lifecycle.transition_profile(
            profile="job90353",
            target_state=gpu_credentials.PROFILE_STATE_FROZEN,
            reason="allocation_expired",
            appdata=tmp_path,
        )


def test_retire_preserves_freeze_tombstone_and_adds_retirement_tombstone(
    tmp_path: Path,
) -> None:
    directory, _metadata = _profile(tmp_path)
    lifecycle.transition_profile(
        profile="job90353",
        target_state=gpu_credentials.PROFILE_STATE_FROZEN,
        reason="role_account_reclaimed",
        appdata=tmp_path,
    )
    frozen_bytes = (
        directory / gpu_credentials.PROFILE_FROZEN_TOMBSTONE_FILENAME
    ).read_bytes()

    result = lifecycle.transition_profile(
        profile="job90353",
        target_state=gpu_credentials.PROFILE_STATE_RETIRED,
        reason="audit_retired",
        appdata=tmp_path,
    )
    metadata = json.loads(
        (directory / gpu_credentials.PROFILE_METADATA_FILENAME).read_text(encoding="utf-8")
    )

    assert result["previous_state"] == gpu_credentials.PROFILE_STATE_FROZEN
    assert metadata["profile_state"] == gpu_credentials.PROFILE_STATE_RETIRED
    assert (
        directory / gpu_credentials.PROFILE_FROZEN_TOMBSTONE_FILENAME
    ).read_bytes() == frozen_bytes
    assert (directory / gpu_credentials.PROFILE_RETIRED_TOMBSTONE_FILENAME).is_file()
    assert set(metadata["tombstone_sha256"]) == {
        gpu_credentials.PROFILE_FROZEN_TOMBSTONE_FILENAME,
        gpu_credentials.PROFILE_RETIRED_TOMBSTONE_FILENAME,
    }


def test_legacy_metadata_is_fail_closed_and_left_unchanged(tmp_path: Path) -> None:
    directory, _metadata = _profile(tmp_path)
    metadata_path = directory / gpu_credentials.PROFILE_METADATA_FILENAME
    legacy = {"credential_profile": "job90353", "job_id": 90353}
    original = json.dumps(legacy).encode("utf-8")
    metadata_path.write_bytes(original)

    with pytest.raises(lifecycle.LifecycleError, match="secure re-enrollment"):
        lifecycle.transition_profile(
            profile="job90353",
            target_state=gpu_credentials.PROFILE_STATE_FROZEN,
            reason="legacy_quarantine",
            appdata=tmp_path,
        )

    assert metadata_path.read_bytes() == original
    assert not (directory / gpu_credentials.PROFILE_FROZEN_TOMBSTONE_FILENAME).exists()


def test_provisioning_profile_can_be_retired_without_credential_access(tmp_path: Path) -> None:
    directory, _metadata = _profile(
        tmp_path,
        state=gpu_credentials.PROFILE_STATE_PROVISIONING,
    )

    result = lifecycle.transition_profile(
        profile="job90353",
        target_state=gpu_credentials.PROFILE_STATE_RETIRED,
        reason="enrollment_abandoned",
        appdata=tmp_path,
    )

    assert result["previous_state"] == gpu_credentials.PROFILE_STATE_PROVISIONING
    assert result["credential_decrypted"] is False
    assert (directory / gpu_credentials.PROFILE_RETIRED_TOMBSTONE_FILENAME).is_file()


def test_tombstone_is_durable_before_metadata_state_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, _metadata = _profile(tmp_path)
    real_replace = os.replace
    observations: list[bool] = []

    def observed_replace(source: Path, target: Path) -> None:
        observations.append(
            (directory / gpu_credentials.PROFILE_FROZEN_TOMBSTONE_FILENAME).is_file()
        )
        real_replace(source, target)

    monkeypatch.setattr(lifecycle.os, "replace", observed_replace)

    lifecycle.transition_profile(
        profile="job90353",
        target_state=gpu_credentials.PROFILE_STATE_FROZEN,
        reason="allocation_reclaimed",
        appdata=tmp_path,
    )

    assert observations == [True]
