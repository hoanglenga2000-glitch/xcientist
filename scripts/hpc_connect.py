"""Create Paramiko clients that reject unknown SSH host keys."""

from __future__ import annotations

import os
from pathlib import Path

import paramiko


def configure_host_key_verification(ssh: paramiko.SSHClient) -> None:
    """Load trusted host keys and reject every unknown server identity."""

    ssh.load_system_host_keys()
    known_hosts_value = os.environ.get("GPU_SSH_KNOWN_HOSTS_PATH")
    if known_hosts_value:
        known_hosts = Path(known_hosts_value).expanduser()
        if not known_hosts.is_file():
            raise RuntimeError(
                "GPU_SSH_KNOWN_HOSTS_PATH points to a missing file (path hidden)"
            )
        ssh.load_host_keys(str(known_hosts))
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())


def secure_ssh_client() -> paramiko.SSHClient:
    """Create an SSH client with the repository-wide host-key policy."""

    ssh = paramiko.SSHClient()
    try:
        configure_host_key_verification(ssh)
        return ssh
    except BaseException:
        ssh.close()
        raise
