"""Tests for secure GPU/SSH credential resolution.

Most important guarantee: a password must never appear in repr()/logs, and
missing credentials must fail loudly rather than silently using a default.
"""
from __future__ import annotations

import inspect
import json
import sys as python_sys
import uuid
from types import SimpleNamespace

import pytest

from research_agent_workstation.server.core import gpu_credentials
from research_agent_workstation.server.core.gpu_credentials import (
    CredentialError,
    GpuSshConfig,
    RetryableTransportError,
    connect_ssh,
    load_gpu_ssh_config,
    load_socks_config,
)

_ENV_KEYS = [
    "GPU_SSH_HOST", "GPU_SSH_HOST_FILE", "GPU_SSH_PORT", "GPU_SSH_USER", "GPU_SSH_USER_FILE",
    "GPU_SSH_PASSWORD", "GPU_SSH_PASSWORD_FILE",
    "GPU_SSH_KEY_PATH", "GPU_SSH_KEY_PATH_FILE",
    "GPU_SSH_SOCKS_HOST", "GPU_SSH_SOCKS_HOST_FILE", "GPU_SSH_SOCKS_PORT", "GPU_SSH_SOCKS_USER",
    "GPU_SSH_SOCKS_USER_FILE", "GPU_SSH_SOCKS_PASSWORD", "GPU_SSH_SOCKS_PASSWORD_FILE",
    "GPU_SSH_JUMP_HOST", "GPU_SSH_JUMP_HOST_FILE", "GPU_SSH_JUMP_PORT", "GPU_SSH_JUMP_USER",
    "GPU_SSH_JUMP_USER_FILE",
    "GPU_SSH_KNOWN_HOSTS_PATH", "GPU_SSH_KNOWN_HOSTS_PATH_FILE",
    "GPU_REMOTE_WORKSPACE", "GPU_REMOTE_WORKSPACE_FILE",
    "EVOMIND_HPC_CREDENTIAL_PROFILE", "EVOMIND_HPC_EXPECTED_HOST_UUID",
    "EVOMIND_HPC_EXPECTED_HOST_UUID_FILE", "EVOMIND_HPC_EXPECTED_GPU_UUID",
    "EVOMIND_HPC_EXPECTED_GPU_UUID_FILE",
]


def _named_profile_metadata(
    profile: str,
    *,
    host: str,
    port: int,
    socks_host: str = "127.0.0.1",
    socks_port: int = 17897,
    state: str = gpu_credentials.PROFILE_STATE_ACTIVE,
) -> dict[str, object]:
    job_id = int(profile.removeprefix("job"))
    metadata: dict[str, object] = {
        "schema": gpu_credentials.PROFILE_METADATA_SCHEMA,
        "credential_profile": profile,
        "job_id": job_id,
        "profile_state": state,
        "allocation_binding_id": f"allocation-fixture-{job_id}",
        "allocation_generation": 1,
        "profile_instance_id": str(uuid.uuid5(uuid.NAMESPACE_URL, profile)),
        "lifecycle_revision": 2 if state == gpu_credentials.PROFILE_STATE_ACTIVE else 1,
        "host": host,
        "port": port,
        "socks_host": socks_host,
        "socks_port": socks_port,
        "known_hosts_path": "known_hosts",
        "remote_workspace": gpu_credentials.ALLOWED_GPU_REMOTE_ROOT,
    }
    if state == gpu_credentials.PROFILE_STATE_ACTIVE:
        metadata["expected_host_uuid"] = f"host-{profile}"
        metadata["expected_gpu_uuid"] = f"gpu-{profile}"
        metadata["container_binding_sha256"] = (
            gpu_credentials.compute_container_binding_sha256(metadata)
        )
    return metadata


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.setattr(gpu_credentials, "sys", SimpleNamespace(platform="linux"))
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_missing_host_raises():
    with pytest.raises(CredentialError, match="GPU_SSH_HOST"):
        load_gpu_ssh_config()


def test_missing_auth_raises(monkeypatch):
    monkeypatch.setenv("GPU_SSH_HOST", "10.0.0.1")
    monkeypatch.setenv("GPU_SSH_USER", "researcher")
    with pytest.raises(CredentialError, match="authentication"):
        load_gpu_ssh_config()


def test_windows_dpapi_auth_is_used_when_plain_host_metadata_was_injected(monkeypatch):
    monkeypatch.setattr(gpu_credentials, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("GPU_SSH_HOST", "stale-configured-host")
    monkeypatch.setenv("GPU_SSH_USER", "stale-configured-user")
    expected = GpuSshConfig(
        host="dpapi-current-host",
        port=2200,
        username="dpapi-current-user",
        password="test-in-memory-only",
    )
    calls = []

    def _load_dpapi():
        calls.append(True)
        return expected

    monkeypatch.setattr(gpu_credentials, "_load_windows_dpapi_config", _load_dpapi)

    config = load_gpu_ssh_config()

    assert calls == [True]
    assert config is expected
    assert config.host == "dpapi-current-host"
    assert "in-memory-only" not in repr(config)


def test_windows_explicit_environment_auth_keeps_environment_precedence(monkeypatch):
    monkeypatch.setattr(gpu_credentials, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("GPU_SSH_HOST", "explicit-host")
    monkeypatch.setenv("GPU_SSH_USER", "explicit-user")
    monkeypatch.setenv("GPU_SSH_PASSWORD", "explicit-secret")

    def _tripwire():
        raise AssertionError("explicit environment authentication must not read DPAPI")

    monkeypatch.setattr(gpu_credentials, "_load_windows_dpapi_config", _tripwire)

    config = load_gpu_ssh_config()

    assert config.host == "explicit-host"
    assert config.username == "explicit-user"
    assert config.password == "explicit-secret"


def test_password_auth_resolves(monkeypatch):
    monkeypatch.setenv("GPU_SSH_HOST", "10.0.0.1")
    monkeypatch.setenv("GPU_SSH_USER", "researcher")
    monkeypatch.setenv("GPU_SSH_PASSWORD", "s3cr3t-value")
    cfg = load_gpu_ssh_config()
    assert cfg.host == "10.0.0.1"
    assert cfg.username == "researcher"
    assert cfg.password == "s3cr3t-value"
    assert cfg.has_auth() is True


def test_password_never_appears_in_repr(monkeypatch):
    monkeypatch.setenv("GPU_SSH_HOST", "10.0.0.1")
    monkeypatch.setenv("GPU_SSH_USER", "researcher")
    monkeypatch.setenv("GPU_SSH_PASSWORD", "TOP-SECRET-PW")
    cfg = load_gpu_ssh_config()
    assert "TOP-SECRET-PW" not in repr(cfg)
    assert "TOP-SECRET-PW" not in str(cfg)
    assert "password" in repr(cfg).lower() or "auth=" in repr(cfg)


def test_password_file_indirection(monkeypatch, tmp_path):
    secret_file = tmp_path / "pw.txt"
    secret_file.write_text("from-file-secret\n", encoding="utf-8")
    monkeypatch.setenv("GPU_SSH_HOST", "10.0.0.1")
    monkeypatch.setenv("GPU_SSH_USER", "researcher")
    monkeypatch.setenv("GPU_SSH_PASSWORD_FILE", str(secret_file))
    cfg = load_gpu_ssh_config()
    assert cfg.password == "from-file-secret"


def test_password_file_missing_raises(monkeypatch):
    monkeypatch.setenv("GPU_SSH_HOST", "10.0.0.1")
    monkeypatch.setenv("GPU_SSH_USER", "researcher")
    monkeypatch.setenv("GPU_SSH_PASSWORD_FILE", "/nonexistent/path/pw.txt")
    with pytest.raises(CredentialError):
        load_gpu_ssh_config()


def test_key_auth_satisfies_require_auth(monkeypatch, tmp_path):
    key = tmp_path / "id_rsa"
    key.write_text("dummy", encoding="utf-8")
    monkeypatch.setenv("GPU_SSH_HOST", "10.0.0.1")
    monkeypatch.setenv("GPU_SSH_USER", "researcher")
    monkeypatch.setenv("GPU_SSH_KEY_PATH", str(key))
    cfg = load_gpu_ssh_config()
    assert cfg.has_auth() is True
    assert cfg.key_path == str(key)


def test_socks_config_none_when_unset():
    assert load_socks_config() is None


def test_socks_config_resolves_and_hides_password(monkeypatch):
    monkeypatch.setenv("GPU_SSH_SOCKS_HOST", "127.0.0.1")
    monkeypatch.setenv("GPU_SSH_SOCKS_PORT", "1080")
    monkeypatch.setenv("GPU_SSH_SOCKS_PASSWORD", "proxy-secret")
    socks = load_socks_config()
    assert socks is not None
    assert socks.host == "127.0.0.1"
    assert socks.port == 1080
    assert "proxy-secret" not in repr(socks)


def test_default_ports(monkeypatch):
    monkeypatch.setenv("GPU_SSH_HOST", "10.0.0.1")
    monkeypatch.setenv("GPU_SSH_USER", "researcher")
    monkeypatch.setenv("GPU_SSH_PASSWORD", "x")
    cfg = load_gpu_ssh_config()
    assert cfg.port == 22  # default when GPU_SSH_PORT unset


def test_jump_gateway_resolves_without_exposing_auth(monkeypatch):
    monkeypatch.setenv("GPU_SSH_HOST", "10.120.18.240")
    monkeypatch.setenv("GPU_SSH_PORT", "6988")
    monkeypatch.setenv("GPU_SSH_USER", "researcher")
    monkeypatch.setenv("GPU_SSH_PASSWORD", "jump-secret")
    monkeypatch.setenv("GPU_SSH_JUMP_HOST", "100.85.169.63")
    monkeypatch.setenv("GPU_SSH_JUMP_PORT", "1235")

    cfg = load_gpu_ssh_config()

    assert cfg.jump_host == "100.85.169.63"
    assert cfg.jump_port == 1235
    assert "jump-secret" not in repr(cfg)


def test_named_windows_dpapi_profile_is_isolated_and_identity_bound(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_credentials, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job89508")
    monkeypatch.setenv("UNRELATED_SECRET_SENTINEL", "must-not-cross-dpapi-boundary")
    profile_dir = tmp_path / "ResearchAgentWorkstation" / "profiles" / "job89508"
    profile_dir.mkdir(parents=True)
    (profile_dir / "hpc_ssh_credential.xml").write_text("dpapi-fixture", encoding="utf-8")
    (profile_dir / "known_hosts").write_text(
        "[fixture-host]:2200 ssh-ed25519 fixture-key\n",
        encoding="utf-8",
    )
    (profile_dir / "hpc_ssh_metadata.json").write_text(
        json.dumps(
            _named_profile_metadata(
                "job89508",
                host="fixture-host",
                port=2200,
            )
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_run(*_args, **kwargs):
        calls.append(dict(kwargs["env"]))
        return SimpleNamespace(stdout=json.dumps({
            "user": "fixture-user",
            "password": "fixture-secret",
        }))

    monkeypatch.setattr(gpu_credentials.subprocess, "run", fake_run)

    config = load_gpu_ssh_config()

    assert config.credential_profile == "job89508"
    assert config.known_hosts_path == str((profile_dir / "known_hosts").resolve())
    assert config.expected_host_uuid == "host-job89508"
    assert config.expected_gpu_uuid == "gpu-job89508"
    assert calls[0]["EVOMIND_HPC_CREDENTIAL_PATH"] == str(
        profile_dir / "hpc_ssh_credential.xml"
    )
    assert "UNRELATED_SECRET_SENTINEL" not in calls[0]
    assert "fixture-secret" not in repr(config)


@pytest.mark.parametrize(
    "override_name",
    gpu_credentials.STRICT_NAMED_PROFILE_OVERRIDE_KEYS,
)
def test_strict_named_profile_rejects_environment_override(monkeypatch, override_name):
    monkeypatch.setattr(gpu_credentials, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job90353")
    monkeypatch.setenv(override_name, "injected-value")

    with pytest.raises(CredentialError, match="rejects ordinary environment overrides"):
        load_gpu_ssh_config(strict_named_profile=True)


def test_strict_named_profile_rejects_even_empty_environment_override(monkeypatch):
    monkeypatch.setattr(gpu_credentials, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job90353")
    monkeypatch.setenv("GPU_SSH_HOST", "")

    with pytest.raises(CredentialError, match="rejects ordinary environment overrides"):
        load_gpu_ssh_config(strict_named_profile=True)


def test_dpapi_child_environment_does_not_inherit_parent_secrets(monkeypatch, tmp_path):
    credential_path = tmp_path / "hpc_ssh_credential.xml"
    monkeypatch.setenv("Path", "fixture-runtime-path")
    monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "competition-secret")
    monkeypatch.setenv("GPU_SSH_PASSWORD", "legacy-ssh-secret")
    monkeypatch.setenv("EVOMIND_SIIM_RUN_ID", "run-private-context")

    environment = gpu_credentials._dpapi_subprocess_environment(credential_path)

    assert environment["PATH"] == "fixture-runtime-path"
    assert environment["SYSTEMROOT"] == r"C:\Windows"
    assert environment["EVOMIND_HPC_CREDENTIAL_PATH"] == str(credential_path)
    assert set(environment) <= {
        *gpu_credentials.DPAPI_SUBPROCESS_ENV_ALLOWLIST,
        "EVOMIND_HPC_CREDENTIAL_PATH",
    }
    assert "OPENAI_API_KEY" not in environment
    assert "KAGGLE_API_TOKEN" not in environment
    assert "GPU_SSH_PASSWORD" not in environment
    assert "EVOMIND_SIIM_RUN_ID" not in environment


@pytest.mark.parametrize("appdata", [None, "relative/appdata"])
def test_dpapi_profile_state_directory_requires_absolute_appdata(monkeypatch, appdata):
    if appdata is None:
        monkeypatch.delenv("APPDATA", raising=False)
    else:
        monkeypatch.setenv("APPDATA", appdata)

    with pytest.raises(CredentialError, match="APPDATA"):
        gpu_credentials._credential_state_dir("job90353")


def test_strict_named_profile_requires_job_proxy_and_identity_metadata(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(gpu_credentials, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job90353")
    profile_dir = (
        tmp_path / "ResearchAgentWorkstation" / "profiles" / "job90353"
    )
    profile_dir.mkdir(parents=True)
    (profile_dir / "hpc_ssh_credential.xml").write_text(
        "dpapi-fixture", encoding="utf-8"
    )
    (profile_dir / "known_hosts").write_text(
        "[100.85.169.63]:1235 ssh-ed25519 fixture-key\n", encoding="utf-8"
    )
    (profile_dir / "hpc_ssh_metadata.json").write_text(
        json.dumps(
            _named_profile_metadata(
                "job90353",
                host=gpu_credentials.HPC_SSH_GATEWAY_HOST,
                port=gpu_credentials.HPC_SSH_GATEWAY_PORT,
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        gpu_credentials.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=json.dumps(
                {"user": "fixture-user", "password": "fixture-secret"}
            )
        ),
    )

    config = load_gpu_ssh_config(strict_named_profile=True)

    assert config.strict_named_profile is True
    assert config.job_id == 90353
    assert config.credential_profile == "job90353"
    assert config.socks is not None
    assert config.remote_workspace == gpu_credentials.ALLOWED_GPU_REMOTE_ROOT
    assert config.profile_state == gpu_credentials.PROFILE_STATE_ACTIVE
    assert config.allocation_binding_id == "allocation-fixture-90353"
    assert config.allocation_generation == 1


@pytest.mark.parametrize(
    "state",
    [
        gpu_credentials.PROFILE_STATE_PROVISIONING,
        gpu_credentials.PROFILE_STATE_FROZEN,
        gpu_credentials.PROFILE_STATE_RETIRED,
    ],
)
def test_named_profile_rejects_non_active_state_before_dpapi_decryption(
    monkeypatch, tmp_path, state
):
    monkeypatch.setattr(gpu_credentials, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job90353")
    profile_dir = tmp_path / "ResearchAgentWorkstation" / "profiles" / "job90353"
    profile_dir.mkdir(parents=True)
    (profile_dir / gpu_credentials.PROFILE_CREDENTIAL_FILENAME).write_text(
        "dpapi-fixture", encoding="utf-8"
    )
    metadata = _named_profile_metadata(
        "job90353",
        host=gpu_credentials.HPC_SSH_GATEWAY_HOST,
        port=gpu_credentials.HPC_SSH_GATEWAY_PORT,
        state=state,
    )
    (profile_dir / gpu_credentials.PROFILE_METADATA_FILENAME).write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    calls = []
    monkeypatch.setattr(
        gpu_credentials.subprocess,
        "run",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(CredentialError, match="not active"):
        load_gpu_ssh_config()

    assert calls == []


def test_identity_bootstrap_loader_accepts_only_provisioning_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_credentials, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job90353")
    profile_dir = tmp_path / "ResearchAgentWorkstation" / "profiles" / "job90353"
    profile_dir.mkdir(parents=True)
    (profile_dir / gpu_credentials.PROFILE_CREDENTIAL_FILENAME).write_text(
        "dpapi-fixture", encoding="utf-8"
    )
    metadata = _named_profile_metadata(
        "job90353",
        host=gpu_credentials.HPC_SSH_GATEWAY_HOST,
        port=gpu_credentials.HPC_SSH_GATEWAY_PORT,
        state=gpu_credentials.PROFILE_STATE_PROVISIONING,
    )
    (profile_dir / gpu_credentials.PROFILE_METADATA_FILENAME).write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    monkeypatch.setattr(
        gpu_credentials.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=json.dumps({"user": "fixture-user", "password": "fixture-secret"})
        ),
    )

    config = gpu_credentials.load_gpu_ssh_config_for_identity_bootstrap()

    assert config.profile_state == gpu_credentials.PROFILE_STATE_PROVISIONING
    assert config.allocation_binding_id == "allocation-fixture-90353"
    assert config.strict_named_profile is False


def test_identity_bootstrap_loader_rejects_active_profile_before_decryption(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(gpu_credentials, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job90353")
    profile_dir = tmp_path / "ResearchAgentWorkstation" / "profiles" / "job90353"
    profile_dir.mkdir(parents=True)
    (profile_dir / gpu_credentials.PROFILE_CREDENTIAL_FILENAME).write_text(
        "dpapi-fixture", encoding="utf-8"
    )
    (profile_dir / gpu_credentials.PROFILE_METADATA_FILENAME).write_text(
        json.dumps(
            _named_profile_metadata(
                "job90353",
                host=gpu_credentials.HPC_SSH_GATEWAY_HOST,
                port=gpu_credentials.HPC_SSH_GATEWAY_PORT,
            )
        ),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        gpu_credentials.subprocess,
        "run",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(CredentialError, match="not permitted"):
        gpu_credentials.load_gpu_ssh_config_for_identity_bootstrap()

    assert calls == []


def test_legacy_named_profile_fails_closed_before_dpapi_decryption(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_credentials, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job90353")
    profile_dir = tmp_path / "ResearchAgentWorkstation" / "profiles" / "job90353"
    profile_dir.mkdir(parents=True)
    (profile_dir / gpu_credentials.PROFILE_CREDENTIAL_FILENAME).write_text(
        "dpapi-fixture", encoding="utf-8"
    )
    (profile_dir / gpu_credentials.PROFILE_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "credential_profile": "job90353",
                "job_id": 90353,
                "host": gpu_credentials.HPC_SSH_GATEWAY_HOST,
                "port": gpu_credentials.HPC_SSH_GATEWAY_PORT,
                "remote_workspace": gpu_credentials.ALLOWED_GPU_REMOTE_ROOT,
            }
        ),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        gpu_credentials.subprocess,
        "run",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(CredentialError, match="secure re-enrollment"):
        load_gpu_ssh_config()

    assert calls == []


def test_active_named_profile_tombstone_blocks_decryption(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_credentials, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job90353")
    profile_dir = tmp_path / "ResearchAgentWorkstation" / "profiles" / "job90353"
    profile_dir.mkdir(parents=True)
    (profile_dir / gpu_credentials.PROFILE_CREDENTIAL_FILENAME).write_text(
        "dpapi-fixture", encoding="utf-8"
    )
    (profile_dir / gpu_credentials.PROFILE_METADATA_FILENAME).write_text(
        json.dumps(
            _named_profile_metadata(
                "job90353",
                host=gpu_credentials.HPC_SSH_GATEWAY_HOST,
                port=gpu_credentials.HPC_SSH_GATEWAY_PORT,
            )
        ),
        encoding="utf-8",
    )
    (profile_dir / gpu_credentials.PROFILE_FROZEN_TOMBSTONE_FILENAME).write_text(
        "{}", encoding="utf-8"
    )
    calls = []
    monkeypatch.setattr(
        gpu_credentials.subprocess,
        "run",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(CredentialError, match="tombstone"):
        load_gpu_ssh_config()

    assert calls == []


def test_loaded_strict_profile_is_revalidated_before_network_use(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_credentials, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job90353")
    profile_dir = tmp_path / "ResearchAgentWorkstation" / "profiles" / "job90353"
    profile_dir.mkdir(parents=True)
    (profile_dir / gpu_credentials.PROFILE_CREDENTIAL_FILENAME).write_text(
        "dpapi-fixture", encoding="utf-8"
    )
    (profile_dir / gpu_credentials.PROFILE_KNOWN_HOSTS_FILENAME).write_text(
        "[100.85.169.63]:1235 ssh-ed25519 fixture\n", encoding="utf-8"
    )
    (profile_dir / gpu_credentials.PROFILE_METADATA_FILENAME).write_text(
        json.dumps(
            _named_profile_metadata(
                "job90353",
                host=gpu_credentials.HPC_SSH_GATEWAY_HOST,
                port=gpu_credentials.HPC_SSH_GATEWAY_PORT,
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        gpu_credentials.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=json.dumps({"user": "fixture-user", "password": "fixture-secret"})
        ),
    )
    config = load_gpu_ssh_config(strict_named_profile=True)
    (profile_dir / gpu_credentials.PROFILE_FROZEN_TOMBSTONE_FILENAME).write_text(
        "{}", encoding="utf-8"
    )

    with pytest.raises(CredentialError, match="tombstone"):
        gpu_credentials._assert_named_profile_runtime_active(config)


def test_same_connection_job_container_identity_gate_is_read_only():
    payload = {
        "host_uuid": "host-job90353",
        "gpus": [
            {
                "name": "NVIDIA A800-SXM4-80GB",
                "uuid": "GPU-job90353",
                "memory_total_mib": 81920,
            }
        ],
        "root_requested": gpu_credentials.ALLOWED_GPU_REMOTE_ROOT,
        "root_realpath": gpu_credentials.ALLOWED_GPU_REMOTE_ROOT,
        "root_exists": True,
        "root_writable": True,
        "read_only": True,
        "signals_sent": 0,
        "other_processes_modified": False,
    }

    class Channel:
        @staticmethod
        def recv_exit_status():
            return 0

    class Stream:
        def __init__(self, value):
            self.value = value
            self.channel = Channel()

        def read(self):
            return self.value

    class Client:
        def exec_command(self, command, timeout):
            assert "python3 -c" in command
            assert timeout == 30
            return None, Stream(json.dumps(payload).encode()), Stream(b"")

    config = GpuSshConfig(
        host=gpu_credentials.HPC_SSH_GATEWAY_HOST,
        port=gpu_credentials.HPC_SSH_GATEWAY_PORT,
        username="fixture",
        password="test-secret",
        socks=gpu_credentials.SocksConfig("127.0.0.1", 17897),
        known_hosts_path="fixture",
        credential_profile="job90353",
        expected_host_uuid="host-job90353",
        expected_gpu_uuid="GPU-job90353",
        job_id=90353,
        remote_workspace=gpu_credentials.ALLOWED_GPU_REMOTE_ROOT,
        strict_named_profile=True,
        profile_state=gpu_credentials.PROFILE_STATE_ACTIVE,
        allocation_binding_id="allocation-fixture-90353",
        allocation_generation=1,
        profile_instance_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "job90353")),
    )

    evidence = gpu_credentials.verify_job_container_identity(
        Client(), config, expected_job_id=90353
    )

    assert evidence["job_container_verified"] is True
    assert evidence["designated_proxy_path_verified"] is True
    assert evidence["signals_sent"] == 0
    assert evidence["other_processes_modified"] is False


@pytest.mark.parametrize("profile", ["../job89508", "job/89508", "job 89508", "..", "路径"])
def test_credential_profile_rejects_unsafe_path_characters(monkeypatch, profile):
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", profile)

    with pytest.raises(CredentialError, match="safe profile characters"):
        load_gpu_ssh_config(require_auth=False)


def test_default_windows_dpapi_profile_preserves_legacy_root_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_credentials, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    state_dir = tmp_path / "ResearchAgentWorkstation"
    state_dir.mkdir()
    (state_dir / "hpc_ssh_credential.xml").write_text("dpapi-fixture", encoding="utf-8")
    (state_dir / "known_hosts").write_text("fixture-host ssh-ed25519 fixture-key\n", encoding="utf-8")
    (state_dir / "hpc_ssh_metadata.json").write_text(json.dumps({
        "host": "fixture-host",
        "port": 22,
        "remote_workspace": gpu_credentials.ALLOWED_GPU_REMOTE_ROOT,
    }), encoding="utf-8")
    monkeypatch.setattr(
        gpu_credentials.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=json.dumps({
            "user": "fixture-user",
            "password": "fixture-secret",
        })),
    )

    config = load_gpu_ssh_config()

    assert config.credential_profile == "default"
    assert config.known_hosts_path == str((state_dir / "known_hosts").resolve())


def test_connect_ssh_consumes_only_pinned_known_hosts_and_reject_policy(
    monkeypatch,
    tmp_path,
):
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("fixture-host ssh-ed25519 fixture-key\n", encoding="utf-8")
    clients = []

    class RejectPolicy:
        pass

    class FakeSSHClient:
        def __init__(self):
            self.loaded = []
            self.policy = None
            self.connect_kwargs = None
            self.closed = False
            clients.append(self)

        def load_host_keys(self, path):
            self.loaded.append(path)

        def set_missing_host_key_policy(self, policy):
            self.policy = policy

        def connect(self, **kwargs):
            self.connect_kwargs = kwargs

        def close(self):
            self.closed = True

    fake_paramiko = SimpleNamespace(SSHClient=FakeSSHClient, RejectPolicy=RejectPolicy)
    monkeypatch.setitem(python_sys.modules, "paramiko", fake_paramiko)
    config = GpuSshConfig(
        host="fixture-host",
        port=22,
        username="fixture-user",
        password="fixture-secret",
        known_hosts_path=str(known_hosts),
    )

    client = connect_ssh(config)

    assert client is clients[0]
    assert clients[0].loaded == [str(known_hosts)]
    assert isinstance(clients[0].policy, RejectPolicy)
    assert clients[0].connect_kwargs["hostname"] == "fixture-host"
    assert "AutoAddPolicy" not in inspect.getsource(gpu_credentials.connect_ssh)


def test_connect_ssh_fails_closed_without_nonempty_known_hosts(monkeypatch, tmp_path):
    fake_paramiko = SimpleNamespace(SSHClient=object, RejectPolicy=object)
    monkeypatch.setitem(python_sys.modules, "paramiko", fake_paramiko)
    config = GpuSshConfig(
        host="fixture-host",
        port=22,
        username="fixture-user",
        password="fixture-secret",
    )
    with pytest.raises(CredentialError, match="known-hosts path is required"):
        connect_ssh(config)

    empty = tmp_path / "known_hosts"
    empty.write_text("", encoding="utf-8")
    config.known_hosts_path = str(empty)
    with pytest.raises(CredentialError, match="missing or empty"):
        connect_ssh(config)


class _SocksSocketFixture:
    def __init__(self, responses: list[bytes]) -> None:
        self._responses = iter(responses)
        self.connected_to = None
        self.closed = False
        self.sent: list[bytes] = []

    def settimeout(self, _timeout: int) -> None:
        pass

    def connect(self, address: tuple[str, int]) -> None:
        self.connected_to = address

    def send(self, payload: bytes) -> int:
        self.sent.append(payload)
        return len(payload)

    def recv(self, _size: int) -> bytes:
        return next(self._responses, b"")

    def close(self) -> None:
        self.closed = True


def test_socks_handshake_close_is_explicitly_retryable(monkeypatch):
    fixture = _SocksSocketFixture([])
    monkeypatch.setattr(
        __import__("socket"),
        "socket",
        lambda *_args: fixture,
    )

    with pytest.raises(RetryableTransportError, match="closed during handshake"):
        gpu_credentials.open_socks_channel(
            gpu_credentials.SocksConfig(host="proxy.fixture", port=1080),
            "target.fixture",
            22,
        )

    assert fixture.closed is True


def test_socks_target_connect_failure_is_explicitly_retryable(monkeypatch):
    fixture = _SocksSocketFixture([b"\x05\x00", b"\x05\x05\x00\x01"])
    monkeypatch.setattr(
        __import__("socket"),
        "socket",
        lambda *_args: fixture,
    )

    with pytest.raises(RetryableTransportError, match="could not connect to target"):
        gpu_credentials.open_socks_channel(
            gpu_credentials.SocksConfig(host="proxy.fixture", port=1080),
            "target.fixture",
            22,
        )

    assert fixture.closed is True


def test_socks_authentication_failure_remains_nonretryable(monkeypatch):
    fixture = _SocksSocketFixture([b"\x05\x02", b"\x01\x01"])
    monkeypatch.setattr(
        __import__("socket"),
        "socket",
        lambda *_args: fixture,
    )

    with pytest.raises(CredentialError, match="authentication failed") as captured:
        gpu_credentials.open_socks_channel(
            gpu_credentials.SocksConfig(
                host="proxy.fixture",
                port=1080,
                username="fixture-user",
                password="fixture-password",
            ),
            "target.fixture",
            22,
        )

    assert type(captured.value) is CredentialError
    assert fixture.closed is True
