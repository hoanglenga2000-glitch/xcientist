"""Pinned-host-key SSH client helper for the managed HPC gateway.

Allocation routing and credentials are resolved by the named DPAPI profile and
the verified workstation gateway.  This module deliberately contains no job,
endpoint, account, or password registry.
"""
from __future__ import annotations

import os
from pathlib import Path

import paramiko


def secure_ssh_client() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    try:
        client.load_system_host_keys()
        configured = os.environ.get("GPU_SSH_KNOWN_HOSTS_PATH", "").strip()
        if not configured:
            raise RuntimeError("Pinned known-hosts path is not configured (path hidden).")
        known_hosts = Path(configured).expanduser()
        if not known_hosts.is_file() or known_hosts.is_symlink():
            raise RuntimeError("Pinned known-hosts file is missing or unsafe (path hidden).")
        client.load_host_keys(str(known_hosts))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        return client
    except Exception:
        client.close()
        raise


def hpc_connect(*_args, **_kwargs):
    raise RuntimeError("Use the named DPAPI job profile through the verified workstation HPC gateway.")


def hpc_exec(*_args, **_kwargs):
    raise RuntimeError("Use the verified workstation HPC command route after job-container identity passes.")
