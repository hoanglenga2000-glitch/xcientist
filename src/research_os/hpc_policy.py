from __future__ import annotations

import os
import re
from pathlib import PurePosixPath
from typing import Mapping


class HPCPolicyError(RuntimeError):
    pass


_SAFE_REMOTE_SEGMENT = re.compile(r"[A-Za-z0-9._+-]+\Z")


def _validate_remote_segments(path: str, *, field: str) -> None:
    segments = path.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise HPCPolicyError(f"{field} must not contain empty, '.' or '..' segments")
    if any(_SAFE_REMOTE_SEGMENT.fullmatch(segment) is None for segment in segments):
        raise HPCPolicyError(
            f"{field} contains whitespace, shell metacharacters, or unsupported path characters"
        )


def validate_remote_workspace(value: str) -> str:
    workspace = str(value or "")
    if not workspace:
        raise HPCPolicyError("EVOMIND_HPC_REMOTE_WORKSPACE must be configured explicitly")
    if workspace != workspace.strip():
        raise HPCPolicyError("remote workspace must not contain leading or trailing whitespace")
    if any(ord(char) < 32 or ord(char) == 127 for char in workspace):
        raise HPCPolicyError("remote workspace contains control characters")
    if workspace.startswith("~/"):
        relative_part = workspace[2:]
    elif workspace.startswith("/") and not workspace.startswith("//"):
        relative_part = workspace[1:]
    else:
        raise HPCPolicyError("remote workspace must be an absolute POSIX path or start with ~/")
    _validate_remote_segments(relative_part, field="remote workspace")
    if workspace in {"/", "/home", "/root", "/tmp", "/usr", "/var", "/opt", "~"}:
        raise HPCPolicyError("remote workspace must be a dedicated project directory, not a shared root")
    return workspace


def validate_remote_relative_path(value: str, *, field: str = "remote path") -> str:
    """Validate a shell-safe relative POSIX path supplied by a task or config."""

    path = str(value or "")
    if not path:
        raise HPCPolicyError(f"{field} must not be empty")
    if path != path.strip():
        raise HPCPolicyError(f"{field} must not contain leading or trailing whitespace")
    if path.startswith(("/", "~")) or PurePosixPath(path).is_absolute():
        raise HPCPolicyError(f"{field} must be relative to EVOMIND_HPC_REMOTE_WORKSPACE")
    _validate_remote_segments(path, field=field)
    return path


def join_remote_workspace(workspace: str, *relative_paths: str) -> str:
    """Join validated relative paths and prove lexical containment in ``workspace``."""

    root_text = validate_remote_workspace(workspace)
    if not relative_paths:
        raise HPCPolicyError("at least one remote relative path is required")
    safe_paths = [
        validate_remote_relative_path(path, field=f"remote path {index}")
        for index, path in enumerate(relative_paths, start=1)
    ]
    root = PurePosixPath(root_text)
    candidate = root.joinpath(*(PurePosixPath(path) for path in safe_paths))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:  # pragma: no cover - validation is defense in depth
        raise HPCPolicyError("remote path escapes EVOMIND_HPC_REMOTE_WORKSPACE") from exc
    if relative == PurePosixPath("."):
        raise HPCPolicyError("remote path must resolve below EVOMIND_HPC_REMOTE_WORKSPACE")
    return candidate.as_posix()


def require_remote_workspace(env: Mapping[str, str] | None = None) -> str:
    source = os.environ if env is None else env
    return validate_remote_workspace(source.get("EVOMIND_HPC_REMOTE_WORKSPACE", ""))


def require_hpc_compute(compute: str) -> None:
    if str(compute or "").strip().lower() != "gpu":
        raise HPCPolicyError(
            "Local training is disabled by release policy. Configure the gated HPC/GPU runtime or report Blocked."
        )
