from __future__ import annotations

import base64
import inspect
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_agent_workstation.server.core.gpu_credentials import (
    ALLOWED_GPU_REMOTE_ROOT,
    GpuSshConfig,
)
from scripts import bootstrap_hpc_profile_identity as bootstrap


def _profile_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: str = "job90353",
) -> tuple[Path, GpuSshConfig]:
    # The bootstrap contract intentionally rejects ambient endpoint/auth
    # overrides.  Keep this fixture hermetic so developer-shell credentials or
    # an earlier test's exported configuration cannot change its outcome.
    for name in bootstrap.DIRECT_OVERRIDE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", profile)
    profile_dir = tmp_path / "ResearchAgentWorkstation" / "profiles" / profile
    profile_dir.mkdir(parents=True)
    (profile_dir / "hpc_ssh_credential.xml").write_text("dpapi-fixture", encoding="utf-8")
    profile_instance_id = str(uuid.uuid5(uuid.NAMESPACE_URL, profile))
    (profile_dir / "hpc_ssh_metadata.json").write_text(
        json.dumps(
            {
                "schema": bootstrap.PROFILE_METADATA_SCHEMA,
                "credential_profile": profile,
                "job_id": int(profile.removeprefix("job")),
                "profile_state": bootstrap.PROFILE_STATE_PROVISIONING,
                "profile_state_reason": "awaiting_identity_bootstrap",
                "allocation_binding_id": f"allocation-fixture-{profile.removeprefix('job')}",
                "allocation_generation": 1,
                "profile_instance_id": profile_instance_id,
                "lifecycle_revision": 1,
                "host": "10.20.30.40",
                "port": 6988,
                "remote_workspace": ALLOWED_GPU_REMOTE_ROOT,
                "known_hosts_path": "known_hosts",
            }
        ),
        encoding="utf-8",
    )
    config = GpuSshConfig(
        host="10.20.30.40",
        port=6988,
        username="fixture-user",
        password="test-never-emit-this-secret",
        known_hosts_path=str((profile_dir / "known_hosts").resolve()),
        credential_profile=profile,
        profile_state=bootstrap.PROFILE_STATE_PROVISIONING,
        allocation_binding_id=f"allocation-fixture-{profile.removeprefix('job')}",
        allocation_generation=1,
        profile_instance_id=profile_instance_id,
    )
    monkeypatch.setattr(
        bootstrap,
        "load_gpu_ssh_config_for_identity_bootstrap",
        lambda: config,
    )
    return profile_dir, config


def _valid_probe_output(*, model: str = "NVIDIA A800 80GB PCIe", memory: int = 81920) -> bytes:
    return (
        "EVOMIND_HOST_UUID_SOURCE=product_uuid\n"
        "EVOMIND_HOST_UUID=01234567-89ab-cdef-0123-456789abcdef\n"
        "EVOMIND_REMOTE_ROOT_EXISTS=1\n"
        "EVOMIND_REMOTE_ROOT_WRITABLE=1\n"
        "EVOMIND_GPU_CSV_BEGIN\n"
        f"0, GPU-01234567-89ab-cdef-0123-456789abcdef, {model}, {memory}\n"
        "EVOMIND_GPU_CSV_END\n"
    ).encode()


def _host_key() -> bootstrap.HostKeyEvidence:
    key_bytes = b"fixture-ed25519-public-key"
    digest = base64.b64encode(__import__("hashlib").sha256(key_bytes).digest()).decode().rstrip("=")
    return bootstrap.HostKeyEvidence(
        algorithm="ssh-ed25519",
        base64_value=base64.b64encode(key_bytes).decode(),
        fingerprint_sha256=f"SHA256:{digest}",
    )


def test_load_named_profile_uses_only_metadata_endpoint_and_named_dpapi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir, config = _profile_fixture(tmp_path, monkeypatch)

    context = bootstrap.load_named_profile_context()

    assert context.profile == "job90353"
    assert context.config is config
    assert context.config.host == "10.20.30.40"
    assert context.config.port == 6988
    assert context.known_hosts_path == (profile_dir / "known_hosts").resolve()


@pytest.mark.parametrize(
    "env_name",
    bootstrap.DIRECT_OVERRIDE_ENV_VARS,
)
def test_load_named_profile_rejects_direct_endpoint_auth_or_identity_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
) -> None:
    _profile_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv(env_name, "fixture-override")

    with pytest.raises(bootstrap.BootstrapError, match="overrides must be cleared"):
        bootstrap.load_named_profile_context()


def test_load_named_profile_rejects_empty_direct_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _profile_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("GPU_SSH_HOST", "")

    with pytest.raises(bootstrap.BootstrapError, match="overrides must be cleared"):
        bootstrap.load_named_profile_context()


def test_nonstandard_port_known_hosts_format_is_paramiko_compatible() -> None:
    config = GpuSshConfig(host="10.20.30.40", port=6988, username="fixture")
    rendered = bootstrap.render_known_hosts(config, _host_key()).decode()

    assert rendered.startswith("[10.20.30.40]:6988 ssh-ed25519 ")
    assert rendered.endswith("\n")


def test_collect_host_key_uses_transport_without_auth_or_auto_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_bytes = b"transport-fixture-key"
    fake_key = SimpleNamespace(
        get_name=lambda: "ssh-ed25519",
        get_base64=lambda: base64.b64encode(key_bytes).decode(),
        asbytes=lambda: key_bytes,
    )
    calls: list[tuple[str, object]] = []

    class FakeSocket:
        def settimeout(self, timeout):
            calls.append(("socket_timeout", timeout))

        def close(self):
            calls.append(("socket_close", True))

    class FakeTransport:
        def __init__(self, sock):
            calls.append(("transport_socket", sock))

        def start_client(self, timeout):
            calls.append(("start_client", timeout))

        def get_remote_server_key(self):
            calls.append(("get_remote_server_key", True))
            return fake_key

        def close(self):
            calls.append(("transport_close", True))

    fake_socket = FakeSocket()
    monkeypatch.setattr(
        bootstrap.socket,
        "create_connection",
        lambda endpoint, timeout: calls.append(("endpoint", endpoint)) or fake_socket,
    )
    monkeypatch.setitem(sys.modules, "paramiko", SimpleNamespace(Transport=FakeTransport))
    config = GpuSshConfig(
        host="10.20.30.40",
        port=6988,
        username="fixture-user",
        password="fixture-secret",
    )

    evidence = bootstrap.collect_server_host_key(config, timeout=17)

    assert ("endpoint", ("10.20.30.40", 6988)) in calls
    assert ("start_client", 17) in calls
    assert ("get_remote_server_key", True) in calls
    assert evidence.algorithm == "ssh-ed25519"
    assert "fixture-secret" not in repr(calls)
    assert "AutoAddPolicy" not in inspect.getsource(bootstrap.collect_server_host_key)


def test_read_only_probe_uses_staged_pin_and_parses_a800(
    tmp_path: Path,
) -> None:
    staged_pin = tmp_path / "known_hosts.staged"
    staged_pin.write_text("[10.20.30.40]:6988 ssh-ed25519 fixture\n", encoding="utf-8")
    connector_calls: list[tuple[GpuSshConfig, int]] = []
    command_calls: list[str] = []

    class Stream:
        def __init__(self, payload: bytes, status: int = 0):
            self.payload = payload
            self.channel = SimpleNamespace(recv_exit_status=lambda: status)

        def read(self, _size: int = -1) -> bytes:
            return self.payload

    class Client:
        def exec_command(self, command: str, timeout: int):
            command_calls.append(command)
            assert timeout == 13
            return None, Stream(_valid_probe_output()), Stream(b"")

        def close(self):
            command_calls.append("closed")

    def connector(config: GpuSshConfig, *, timeout: int):
        connector_calls.append((config, timeout))
        return Client()

    config = GpuSshConfig(
        host="10.20.30.40",
        port=6988,
        username="fixture-user",
        password="fixture-secret",
    )

    identity = bootstrap.inspect_remote_identity(
        config,
        staged_known_hosts_path=staged_pin,
        timeout=13,
        connector=connector,
    )

    assert connector_calls[0][0].known_hosts_path == str(staged_pin)
    assert connector_calls[0][1] == 13
    assert identity["host_uuid_source"] == "product_uuid"
    assert identity["gpu"]["name"] == "NVIDIA A800 80GB PCIe"
    assert identity["gpu"]["memory_total_mib"] == 81920
    assert identity["remote_workspace"]["write_probe_performed"] is False
    assert "touch " not in command_calls[0]
    assert "mkdir " not in command_calls[0]
    assert "rm " not in command_calls[0]
    assert "nvidia-smi" in command_calls[0]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_valid_probe_output(model="NVIDIA H800 80GB HBM3"), "not an A800"),
        (_valid_probe_output(memory=49140), "approximately 80 GB"),
        (
            _valid_probe_output()
            .replace(b"EVOMIND_GPU_CSV_END", b"1, GPU-second, NVIDIA A800, 81920\nEVOMIND_GPU_CSV_END"),
            "exactly one GPU",
        ),
    ],
)
def test_probe_fails_closed_for_wrong_gpu_contract(payload: bytes, message: str) -> None:
    with pytest.raises(bootstrap.BootstrapError, match=message):
        bootstrap.parse_remote_identity_output(payload)


def test_probe_accepts_machine_id_fallback_and_hyphenated_a800_name() -> None:
    payload = _valid_probe_output(model="NVIDIA A800-SXM4-80GB").replace(
        b"EVOMIND_HOST_UUID_SOURCE=product_uuid",
        b"EVOMIND_HOST_UUID_SOURCE=machine_id",
    )

    identity = bootstrap.parse_remote_identity_output(payload)

    assert identity["host_uuid_source"] == "machine_id"
    assert identity["gpu"]["name"] == "NVIDIA A800-SXM4-80GB"


def test_success_commits_identity_metadata_and_never_emits_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir, _config = _profile_fixture(tmp_path, monkeypatch)
    context = bootstrap.load_named_profile_context()
    observed_staged_pin: list[bytes] = []

    def inspect(_config, *, staged_known_hosts_path: Path, timeout: int):
        assert timeout == 11
        observed_staged_pin.append(staged_known_hosts_path.read_bytes())
        return bootstrap.parse_remote_identity_output(_valid_probe_output())

    result = bootstrap.bootstrap_profile_identity(
        timeout=11,
        now=datetime(2026, 7, 30, 1, 30, tzinfo=timezone.utc),
        context_loader=lambda: context,
        host_key_collector=lambda _config, *, timeout: _host_key(),
        identity_inspector=inspect,
    )
    metadata_text = (profile_dir / "hpc_ssh_metadata.json").read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    known_hosts = (profile_dir / "known_hosts").read_text(encoding="ascii")
    serialized_result = json.dumps(result)

    assert observed_staged_pin == [known_hosts.encode("ascii")]
    assert known_hosts.startswith("[10.20.30.40]:6988 ssh-ed25519 ")
    assert metadata["expected_host_uuid"] == "01234567-89ab-cdef-0123-456789abcdef"
    assert metadata["expected_gpu_uuid"] == "GPU-01234567-89ab-cdef-0123-456789abcdef"
    assert metadata["identity_bootstrapped_at"] == "2026-07-30T01:30:00Z"
    assert metadata["profile_state"] == bootstrap.PROFILE_STATE_ACTIVE
    assert metadata["profile_state_reason"] == "identity_binding_verified"
    assert metadata["lifecycle_revision"] == 2
    assert metadata["container_binding_sha256"] == result["container_binding_sha256"]
    evidence = metadata["identity_bootstrap_evidence"]
    assert evidence["mode"] == "tofu_once_then_reject_policy"
    assert evidence["reject_policy_verified"] is True
    assert evidence["remote_workspace"]["write_probe_performed"] is False
    assert result["secrets_emitted"] is False
    assert result["profile_state"] == bootstrap.PROFILE_STATE_ACTIVE
    assert result["allocation_generation"] == 1
    assert "never-emit-this-secret" not in metadata_text
    assert "never-emit-this-secret" not in known_hosts
    assert "never-emit-this-secret" not in serialized_result


def test_failure_before_commit_leaves_profile_unbound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir, _config = _profile_fixture(tmp_path, monkeypatch)
    original_metadata = (profile_dir / "hpc_ssh_metadata.json").read_bytes()
    context = bootstrap.load_named_profile_context()

    def fail_inspection(*_args, **_kwargs):
        raise bootstrap.BootstrapError("fixture probe rejected")

    with pytest.raises(bootstrap.BootstrapError, match="fixture probe rejected"):
        bootstrap.bootstrap_profile_identity(
            context_loader=lambda: context,
            host_key_collector=lambda _config, *, timeout: _host_key(),
            identity_inspector=fail_inspection,
        )

    assert (profile_dir / "hpc_ssh_metadata.json").read_bytes() == original_metadata
    assert not (profile_dir / "known_hosts").exists()
    assert not list(profile_dir.glob(".*.tmp"))


def test_failed_second_replace_rolls_back_known_hosts_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir, _config = _profile_fixture(tmp_path, monkeypatch)
    context = bootstrap.load_named_profile_context()
    metadata_before = context.metadata_path.read_bytes()
    replace_calls = 0

    def fail_second_replace(source: Path, target: Path):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("fixture metadata replace failure")
        os.replace(source, target)

    with pytest.raises(bootstrap.BootstrapError, match="original state was restored"):
        bootstrap.commit_profile_binding(
            context,
            known_hosts_payload=b"[10.20.30.40]:6988 ssh-ed25519 fixture\n",
            metadata_payload=b'{"bound":true}\n',
            replace_file=fail_second_replace,
        )

    assert context.metadata_path.read_bytes() == metadata_before
    assert not context.known_hosts_path.exists()
    assert not list(profile_dir.glob(".*.tmp"))


def test_activation_commit_refuses_profile_frozen_during_remote_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir, _config = _profile_fixture(tmp_path, monkeypatch)
    context = bootstrap.load_named_profile_context()
    original_metadata = context.metadata_path.read_bytes()
    (profile_dir / bootstrap.PROFILE_FROZEN_TOMBSTONE_FILENAME).write_text(
        "{}", encoding="utf-8"
    )

    with pytest.raises(bootstrap.BootstrapError, match="frozen or retired"):
        bootstrap.commit_profile_binding(
            context,
            known_hosts_payload=b"fixture-host-key\n",
            metadata_payload=b'{"profile_state":"active"}\n',
        )

    assert context.metadata_path.read_bytes() == original_metadata
    assert not context.known_hosts_path.exists()


def test_activation_commit_refuses_metadata_changed_during_remote_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _profile_dir, _config = _profile_fixture(tmp_path, monkeypatch)
    context = bootstrap.load_named_profile_context()
    metadata = json.loads(context.metadata_path.read_text(encoding="utf-8"))
    metadata["profile_state_reason"] = "concurrent-local-change"
    context.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(bootstrap.BootstrapError, match="metadata changed"):
        bootstrap.commit_profile_binding(
            context,
            known_hosts_payload=b"fixture-host-key\n",
            metadata_payload=b'{"profile_state":"active"}\n',
        )

    assert not context.known_hosts_path.exists()


def test_bootstrap_is_one_shot_and_refuses_existing_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir, _config = _profile_fixture(tmp_path, monkeypatch)
    metadata_path = profile_dir / "hpc_ssh_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["expected_host_uuid"] = "already-bound-host"
    metadata["expected_gpu_uuid"] = "already-bound-gpu"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(bootstrap.BootstrapError, match="already bound"):
        bootstrap.load_named_profile_context()


def test_bootstrap_rejects_legacy_metadata_before_loading_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir, _config = _profile_fixture(tmp_path, monkeypatch)
    metadata_path = profile_dir / "hpc_ssh_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("schema")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    calls: list[bool] = []
    monkeypatch.setattr(
        bootstrap,
        "load_gpu_ssh_config_for_identity_bootstrap",
        lambda: calls.append(True),
    )

    with pytest.raises(bootstrap.BootstrapError, match="secure re-enrollment"):
        bootstrap.load_named_profile_context()

    assert calls == []


def test_bootstrap_rejects_tombstoned_profile_before_loading_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir, _config = _profile_fixture(tmp_path, monkeypatch)
    (profile_dir / bootstrap.PROFILE_FROZEN_TOMBSTONE_FILENAME).write_text(
        "{}", encoding="utf-8"
    )
    calls: list[bool] = []
    monkeypatch.setattr(
        bootstrap,
        "load_gpu_ssh_config_for_identity_bootstrap",
        lambda: calls.append(True),
    )

    with pytest.raises(bootstrap.BootstrapError, match="cannot be bootstrapped"):
        bootstrap.load_named_profile_context()

    assert calls == []
