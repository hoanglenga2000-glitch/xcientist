"""Tests for secure GPU/SSH credential resolution.

Most important guarantee: a password must never appear in repr()/logs, and
missing credentials must fail loudly rather than silently using a default.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from research_agent_workstation.server.core.gpu_credentials import (
    CredentialError,
    GpuSshConfig,
    connect_ssh,
    load_gpu_ssh_config,
    load_socks_config,
)

_ENV_KEYS = [
    "GPU_SSH_HOST", "GPU_SSH_PORT", "GPU_SSH_USER", "GPU_SSH_PASSWORD", "GPU_SSH_PASSWORD_FILE",
    "GPU_SSH_KEY_PATH", "GPU_SSH_KEY_PATH_FILE",
    "GPU_SSH_KNOWN_HOSTS_PATH", "GPU_SSH_KNOWN_HOSTS_PATH_FILE",
    "GPU_SSH_SOCKS_HOST", "GPU_SSH_SOCKS_PORT", "GPU_SSH_SOCKS_USER", "GPU_SSH_SOCKS_PASSWORD",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
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


def test_connect_ssh_rejects_unknown_hosts_and_loads_configured_keys(monkeypatch, tmp_path):
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("example.invalid ssh-ed25519 AAAA\n", encoding="utf-8")
    calls = []
    policy = object()

    class FakeClient:
        def load_system_host_keys(self):
            calls.append(("system",))

        def load_host_keys(self, path):
            calls.append(("custom", path))

        def set_missing_host_key_policy(self, value):
            calls.append(("policy", value))

        def connect(self, **kwargs):
            calls.append(("connect", kwargs))

    client = FakeClient()
    monkeypatch.setitem(
        sys.modules,
        "paramiko",
        SimpleNamespace(SSHClient=lambda: client, RejectPolicy=lambda: policy),
    )
    config = GpuSshConfig(
        host="example.invalid",
        port=22,
        username="researcher",
        password="fixture-secret",
        known_hosts_path=str(known_hosts),
    )

    assert connect_ssh(config) is client
    assert calls[:3] == [("system",), ("custom", str(known_hosts)), ("policy", policy)]
    assert calls[3][0] == "connect"


def test_connect_ssh_rejects_missing_known_hosts_file(monkeypatch, tmp_path):
    class FakeClient:
        def load_system_host_keys(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "paramiko",
        SimpleNamespace(SSHClient=FakeClient, RejectPolicy=object),
    )
    config = GpuSshConfig(
        host="example.invalid",
        port=22,
        username="researcher",
        password="fixture-secret",
        known_hosts_path=str(tmp_path / "missing"),
    )

    with pytest.raises(CredentialError, match="KNOWN_HOSTS"):
        connect_ssh(config)
