"""Tests for secure GPU/SSH credential resolution.

Most important guarantee: a password must never appear in repr()/logs, and
missing credentials must fail loudly rather than silently using a default.
"""
from __future__ import annotations

import pytest

from research_agent_workstation.server.core.gpu_credentials import (
    CredentialError,
    GpuSshConfig,
    SocksConfig,
    load_gpu_ssh_config,
    load_socks_config,
)

_ENV_KEYS = [
    "GPU_SSH_HOST", "GPU_SSH_PORT", "GPU_SSH_USER", "GPU_SSH_PASSWORD", "GPU_SSH_PASSWORD_FILE",
    "GPU_SSH_KEY_PATH", "GPU_SSH_KEY_PATH_FILE",
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
