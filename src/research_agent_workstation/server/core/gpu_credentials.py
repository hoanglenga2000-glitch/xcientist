"""Secure resolution of GPU/SSH credentials from the environment.

Centralizes credential loading so no script needs to hardcode a password. Values
come from environment variables (see .env.example), with a ``*_FILE`` indirection
so secrets can live in a file (e.g. a Docker/K8s secret mount) instead of the
environment directly.

Design rules:
  * Never log, print, or repr a secret value.
  * Missing required credentials raise a clear, actionable error.
  * No side effects on import; nothing is read until you call a resolver.

Env vars (all optional at import time, validated on use):
  GPU_SSH_HOST / GPU_SSH_PORT / GPU_SSH_USER / GPU_SSH_PASSWORD[_FILE]
  GPU_SSH_KEY_PATH[_FILE]
  GPU_SSH_SOCKS_HOST / GPU_SSH_SOCKS_PORT / GPU_SSH_SOCKS_USER[_FILE] / GPU_SSH_SOCKS_PASSWORD[_FILE]
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class CredentialError(RuntimeError):
    """Raised when a required credential is missing or unreadable."""


def _read_value(name: str) -> Optional[str]:
    """Return env[name], or the contents of the file named by env[name + '_FILE']."""
    direct = os.environ.get(name)
    if direct:
        return direct
    file_var = os.environ.get(f"{name}_FILE")
    if file_var:
        path = Path(file_var)
        if not path.exists():
            raise CredentialError(f"{name}_FILE points to a missing file (path hidden)")
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    return None


def _require(name: str) -> str:
    value = _read_value(name)
    if not value:
        raise CredentialError(
            f"Missing required credential {name!r}. Set it in your environment or a .env file "
            f"(see .env.example). Never hardcode it in source."
        )
    return value


@dataclass
class SocksConfig:
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None  # never logged

    def __repr__(self) -> str:  # avoid leaking password in logs/tracebacks
        return f"SocksConfig(host={self.host!r}, port={self.port}, username={'set' if self.username else None})"


@dataclass
class GpuSshConfig:
    host: str
    port: int
    username: str
    password: Optional[str] = None  # never logged
    key_path: Optional[str] = None
    socks: Optional[SocksConfig] = None

    def __repr__(self) -> str:  # avoid leaking password in logs/tracebacks
        return (
            f"GpuSshConfig(host={self.host!r}, port={self.port}, username={self.username!r}, "
            f"auth={'key' if self.key_path else ('password' if self.password else 'none')}, "
            f"socks={self.socks!r})"
        )

    def has_auth(self) -> bool:
        return bool(self.password or self.key_path)


def load_socks_config() -> Optional[SocksConfig]:
    """Return SOCKS proxy config if GPU_SSH_SOCKS_HOST is set, else None."""
    host = _read_value("GPU_SSH_SOCKS_HOST")
    if not host:
        return None
    port = int(os.environ.get("GPU_SSH_SOCKS_PORT", "1080"))
    return SocksConfig(
        host=host,
        port=port,
        username=_read_value("GPU_SSH_SOCKS_USER"),
        password=_read_value("GPU_SSH_SOCKS_PASSWORD"),
    )


def load_gpu_ssh_config(*, require_auth: bool = True) -> GpuSshConfig:
    """Resolve the full GPU SSH config from the environment.

    Raises CredentialError if host/user are missing, or (when require_auth) if
    neither a password nor a key path is available.
    """
    config = GpuSshConfig(
        host=_require("GPU_SSH_HOST"),
        port=int(os.environ.get("GPU_SSH_PORT", "22")),
        username=_require("GPU_SSH_USER"),
        password=_read_value("GPU_SSH_PASSWORD"),
        key_path=_read_value("GPU_SSH_KEY_PATH"),
        socks=load_socks_config(),
    )
    if require_auth and not config.has_auth():
        raise CredentialError(
            "No GPU SSH authentication configured. Set GPU_SSH_PASSWORD[_FILE] or "
            "GPU_SSH_KEY_PATH[_FILE] (see .env.example)."
        )
    return config


def open_socks_channel(socks: "SocksConfig", dest_host: str, dest_port: int, timeout: int = 20):
    """Open a raw socket to dest via a SOCKS5 proxy (no external deps).

    Returns a connected socket suitable for paramiko's ``sock=`` argument.
    Supports optional username/password (RFC 1929) auth.
    """
    import socket
    import struct

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((socks.host, socks.port))
    if socks.username and socks.password:
        sock.send(b"\x05\x01\x02")  # one method: username/password
        sock.recv(2)
        user = socks.username.encode()
        pw = socks.password.encode()
        sock.send(b"\x01" + bytes([len(user)]) + user + bytes([len(pw)]) + pw)
        sock.recv(2)
    else:
        sock.send(b"\x05\x01\x00")  # no auth
        sock.recv(2)
    host_bytes = dest_host.encode()
    sock.send(b"\x05\x01\x00\x03" + bytes([len(host_bytes)]) + host_bytes + struct.pack("!H", dest_port))
    sock.recv(10)  # CONNECT response
    return sock


def connect_ssh(config: Optional[GpuSshConfig] = None, *, timeout: int = 20):
    """Open a paramiko SSH connection using env-resolved credentials.

    This is the single secure entry point that replaces hardcoded connection
    helpers. paramiko is imported lazily so this module stays import-safe when
    paramiko is not installed. Never logs the password.
    """
    try:
        import paramiko
    except ImportError as exc:  # pragma: no cover
        raise CredentialError("paramiko is required for connect_ssh (pip install -r requirements.txt)") from exc

    config = config or load_gpu_ssh_config()
    sock = None
    if config.socks is not None:
        sock = open_socks_channel(config.socks, config.host, config.port, timeout=timeout)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs = dict(
        hostname=config.host,
        port=config.port,
        username=config.username,
        sock=sock,
        timeout=timeout,
        banner_timeout=max(timeout, 30),
        allow_agent=False,
        look_for_keys=False,
    )
    if config.key_path:
        connect_kwargs["key_filename"] = config.key_path
    if config.password:
        connect_kwargs["password"] = config.password
    client.connect(**connect_kwargs)
    return client
