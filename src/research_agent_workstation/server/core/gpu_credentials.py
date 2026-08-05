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
  GPU_SSH_JUMP_HOST / GPU_SSH_JUMP_PORT / GPU_SSH_JUMP_USER
"""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class CredentialError(RuntimeError):
    """Raised when a required credential is missing or unreadable."""


class RetryableTransportError(CredentialError):
    """A temporary SOCKS transport failure that callers may retry."""


ALLOWED_GPU_REMOTE_ROOT = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"
DEFAULT_CREDENTIAL_PROFILE = "default"
SAFE_CREDENTIAL_PROFILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PROFILE_CREDENTIAL_FILENAME = "hpc_ssh_credential.xml"
PROFILE_METADATA_FILENAME = "hpc_ssh_metadata.json"
PROFILE_KNOWN_HOSTS_FILENAME = "known_hosts"
PROFILE_FROZEN_TOMBSTONE_FILENAME = "hpc_profile_frozen.tombstone.json"
PROFILE_RETIRED_TOMBSTONE_FILENAME = "hpc_profile_retired.tombstone.json"
PROFILE_LIFECYCLE_LOCK_FILENAME = ".hpc_profile_lifecycle.lock"
PROFILE_LIFECYCLE_LOCK_TIMEOUT_SECONDS = 300.0
PROFILE_METADATA_SCHEMA = "evomind.hpc.dpapi_profile.v2"
PROFILE_STATE_PROVISIONING = "provisioning"
PROFILE_STATE_ACTIVE = "active"
PROFILE_STATE_FROZEN = "frozen"
PROFILE_STATE_RETIRED = "retired"
PROFILE_STATES = frozenset(
    {
        PROFILE_STATE_PROVISIONING,
        PROFILE_STATE_ACTIVE,
        PROFILE_STATE_FROZEN,
        PROFILE_STATE_RETIRED,
    }
)
SAFE_ALLOCATION_BINDING_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
HPC_SSH_GATEWAY_HOST = "100.85.169.63"
HPC_SSH_GATEWAY_PORT = 1235
_STRICT_NAMED_PROFILE_VALUE_KEYS = (
    "GPU_SSH_HOST",
    "GPU_SSH_USER",
    "GPU_SSH_PASSWORD",
    "GPU_SSH_KEY_PATH",
    "GPU_SSH_SOCKS_HOST",
    "GPU_SSH_SOCKS_USER",
    "GPU_SSH_SOCKS_PASSWORD",
    "GPU_SSH_JUMP_HOST",
    "GPU_SSH_JUMP_USER",
    "GPU_SSH_KNOWN_HOSTS_PATH",
    "GPU_REMOTE_WORKSPACE",
    "EVOMIND_HPC_EXPECTED_HOST_UUID",
    "EVOMIND_HPC_EXPECTED_GPU_UUID",
)
STRICT_NAMED_PROFILE_OVERRIDE_KEYS = tuple(
    dict.fromkeys(
        name
        for base_name in _STRICT_NAMED_PROFILE_VALUE_KEYS
        for name in (base_name, f"{base_name}_FILE")
    )
) + (
    "GPU_SSH_PORT",
    "GPU_SSH_SOCKS_PORT",
    "GPU_SSH_JUMP_PORT",
)

# Import-Clixml only needs the Windows process/runtime environment plus the
# one credential path supplied by this module.  Passing ``os.environ`` through
# wholesale would unnecessarily expose unrelated API keys, legacy SSH
# passwords, and experiment tokens to the PowerShell child process.
DPAPI_SUBPROCESS_ENV_ALLOWLIST = (
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATH",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "PSMODULEPATH",
)


def _credential_profile() -> str:
    """Return a validated, single-path-segment credential profile name."""

    profile = os.environ.get("EVOMIND_HPC_CREDENTIAL_PROFILE", "").strip()
    if not profile:
        return DEFAULT_CREDENTIAL_PROFILE
    if profile in {".", ".."} or not SAFE_CREDENTIAL_PROFILE.fullmatch(profile):
        raise CredentialError(
            "EVOMIND_HPC_CREDENTIAL_PROFILE must contain only safe profile characters"
        )
    return profile


def _credential_state_dir(profile: Optional[str] = None) -> Path:
    """Resolve the legacy default directory or an isolated named profile."""

    selected = profile or _credential_profile()
    appdata = str(os.environ.get("APPDATA") or "").strip()
    if not appdata:
        raise CredentialError("APPDATA is required for Windows DPAPI HPC profiles")
    appdata_path = Path(appdata).expanduser()
    if not appdata_path.is_absolute():
        raise CredentialError("APPDATA must be absolute for Windows DPAPI HPC profiles")
    state_root = appdata_path / "ResearchAgentWorkstation"
    if selected == DEFAULT_CREDENTIAL_PROFILE:
        return state_root
    return state_root / "profiles" / selected


def _profile_artifact_path(profile_dir: Path, value: object, default_name: str) -> Path:
    """Resolve a metadata-owned artifact without allowing profile-directory escape."""

    raw = str(value or default_name)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = profile_dir / candidate
    try:
        resolved = candidate.resolve(strict=False)
        profile_root = profile_dir.resolve(strict=False)
        resolved.relative_to(profile_root)
    except (OSError, ValueError) as exc:
        raise CredentialError("Windows DPAPI HPC profile artifact is outside its profile") from exc
    return resolved


def _dpapi_subprocess_environment(credential_path: Path) -> dict[str, str]:
    """Build the minimum environment needed by the DPAPI PowerShell child.

    Windows environment names are case-insensitive, while a monkeypatched test
    mapping may not be.  Canonicalizing the lookup lets the production path and
    red tests enforce the same explicit allowlist without copying secret-bearing
    variables from the parent process.
    """

    available = {str(name).upper(): str(value) for name, value in os.environ.items()}
    environment = {
        name: available[name]
        for name in DPAPI_SUBPROCESS_ENV_ALLOWLIST
        if name in available
    }
    environment["EVOMIND_HPC_CREDENTIAL_PATH"] = str(credential_path)
    return environment


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
    jump_host: Optional[str] = None
    jump_port: int = 22
    jump_username: Optional[str] = None
    known_hosts_path: Optional[str] = None
    credential_profile: str = DEFAULT_CREDENTIAL_PROFILE
    expected_host_uuid: Optional[str] = None
    expected_gpu_uuid: Optional[str] = None
    job_id: Optional[int] = None
    remote_workspace: Optional[str] = None
    strict_named_profile: bool = False
    profile_state: Optional[str] = None
    allocation_binding_id: Optional[str] = None
    allocation_generation: Optional[int] = None
    profile_instance_id: Optional[str] = None

    def __repr__(self) -> str:  # avoid leaking password in logs/tracebacks
        return (
            f"GpuSshConfig(host={self.host!r}, port={self.port}, username={self.username!r}, "
            f"auth={'key' if self.key_path else ('password' if self.password else 'none')}, "
            f"socks={self.socks!r}, jump_host={self.jump_host!r}, jump_port={self.jump_port}, "
            f"credential_profile={self.credential_profile!r}, host_key_pinned="
            f"{bool(self.known_hosts_path)})"
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


def load_gpu_ssh_config(
    *,
    require_auth: bool = True,
    strict_named_profile: bool = False,
) -> GpuSshConfig:
    """Resolve the full GPU SSH config from the environment.

    Raises CredentialError if host/user are missing, or (when require_auth) if
    neither a password nor a key path is available.
    """
    profile = _credential_profile()
    if profile != DEFAULT_CREDENTIAL_PROFILE:
        if sys.platform != "win32":
            raise CredentialError("named HPC profiles require Windows DPAPI")
        injected = [
            name for name in STRICT_NAMED_PROFILE_OVERRIDE_KEYS if name in os.environ
        ]
        if injected:
            raise CredentialError(
                "named HPC profile rejects ordinary environment overrides"
            )
        config = _load_windows_dpapi_config(strict_named_profile=strict_named_profile)
        if require_auth and not config.has_auth():
            raise CredentialError("named HPC profile has no authentication")
        return config
    if strict_named_profile:
        raise CredentialError("strict HPC access requires a named job profile")
    host = _read_value("GPU_SSH_HOST")
    username = _read_value("GPU_SSH_USER")
    password = _read_value("GPU_SSH_PASSWORD")
    key_path = _read_value("GPU_SSH_KEY_PATH")
    use_windows_dpapi = sys.platform == "win32" and (
        not host
        or not username
        or (require_auth and not password and not key_path)
    )
    if use_windows_dpapi:
        config = _load_windows_dpapi_config()
    else:
        config = GpuSshConfig(
            host=host or _require("GPU_SSH_HOST"),
            port=int(os.environ.get("GPU_SSH_PORT", "22")),
            username=username or _require("GPU_SSH_USER"),
            password=password,
            key_path=key_path,
            socks=load_socks_config(),
            jump_host=_read_value("GPU_SSH_JUMP_HOST"),
            jump_port=int(os.environ.get("GPU_SSH_JUMP_PORT", "22")),
            jump_username=_read_value("GPU_SSH_JUMP_USER"),
            known_hosts_path=_read_value("GPU_SSH_KNOWN_HOSTS_PATH"),
            credential_profile=profile,
            expected_host_uuid=_read_value("EVOMIND_HPC_EXPECTED_HOST_UUID"),
            expected_gpu_uuid=_read_value("EVOMIND_HPC_EXPECTED_GPU_UUID"),
        )
    if require_auth and not config.has_auth():
        raise CredentialError(
            "No GPU SSH authentication configured. Set GPU_SSH_PASSWORD[_FILE] or "
            "GPU_SSH_KEY_PATH[_FILE] (see .env.example)."
        )
    return config


def _named_job_id(profile: str) -> int:
    match = re.fullmatch(r"job([0-9]+)", profile)
    if not match:
        raise CredentialError("named HPC profile must use job<job_id>")
    job_id = int(match.group(1))
    if job_id <= 0:
        raise CredentialError("named HPC profile job binding is invalid")
    return job_id


def _profile_tombstone_paths(profile_dir: Path) -> tuple[Path, Path]:
    return (
        profile_dir / PROFILE_FROZEN_TOMBSTONE_FILENAME,
        profile_dir / PROFILE_RETIRED_TOMBSTONE_FILENAME,
    )


def _profile_marker_present(path: Path) -> bool:
    """Treat even a broken symlink/reparse marker as fail-closed state."""

    try:
        return path.exists() or path.is_symlink()
    except OSError:
        return True


@contextmanager
def profile_lifecycle_lock(profile_dir: Path):
    """Serialize profile decryption, bootstrap commit, freeze, and retirement."""

    if not profile_dir.is_dir():
        raise CredentialError("named HPC profile directory is not installed")
    lock_path = _profile_artifact_path(
        profile_dir,
        PROFILE_LIFECYCLE_LOCK_FILENAME,
        PROFILE_LIFECYCLE_LOCK_FILENAME,
    )
    try:
        handle = lock_path.open("a+b")
    except OSError as exc:
        raise CredentialError("named HPC profile lifecycle lock is unavailable") from exc
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + PROFILE_LIFECYCLE_LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
        else:  # pragma: no cover - exercised by Linux CI
            import fcntl

            deadline = time.monotonic() + PROFILE_LIFECYCLE_LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - exercised by Linux CI
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise CredentialError("named HPC profile lifecycle lock failed") from exc
    finally:
        handle.close()


def _validate_profile_instance_id(value: object) -> str:
    raw = str(value or "").strip()
    try:
        parsed = uuid.UUID(raw)
    except (ValueError, AttributeError) as exc:
        raise CredentialError("named HPC profile instance identity is invalid") from exc
    if str(parsed) != raw.lower():
        raise CredentialError("named HPC profile instance identity is not canonical")
    return str(parsed)


def _positive_int(value: object, message: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CredentialError(message) from exc
    if parsed <= 0:
        raise CredentialError(message)
    return parsed


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def compute_container_binding_sha256(metadata: dict[str, object]) -> str:
    """Return the deterministic active-container binding identity.

    The digest is not an authentication signature.  It prevents accidental
    mixing of allocation generations, profile instances, host/GPU identities,
    endpoints, and remote roots across profile metadata updates.
    """

    payload = {
        "schema": "evomind.hpc.container_binding.v1",
        "credential_profile": str(metadata.get("credential_profile") or ""),
        "job_id": int(metadata.get("job_id") or 0),
        "allocation_binding_id": str(metadata.get("allocation_binding_id") or ""),
        "allocation_generation": int(metadata.get("allocation_generation") or 0),
        "profile_instance_id": str(metadata.get("profile_instance_id") or ""),
        "host": str(metadata.get("host") or ""),
        "port": int(metadata.get("port") or 0),
        "socks_host": str(metadata.get("socks_host") or ""),
        "socks_port": int(metadata.get("socks_port") or 0),
        "known_hosts_path": str(
            metadata.get("known_hosts_path") or PROFILE_KNOWN_HOSTS_FILENAME
        ),
        "expected_host_uuid": str(metadata.get("expected_host_uuid") or ""),
        "expected_gpu_uuid": str(metadata.get("expected_gpu_uuid") or ""),
        "remote_workspace": str(metadata.get("remote_workspace") or ""),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _validate_named_profile_lifecycle(
    metadata: dict[str, object],
    *,
    profile: str,
    profile_dir: Path,
    allowed_states: frozenset[str],
) -> dict[str, object]:
    """Validate lifecycle metadata before any DPAPI credential is decrypted."""

    expected_job_id = _named_job_id(profile)
    tombstones = [
        path for path in _profile_tombstone_paths(profile_dir) if _profile_marker_present(path)
    ]
    if tombstones:
        raise CredentialError("named HPC profile is frozen or retired by tombstone")
    if metadata.get("schema") != PROFILE_METADATA_SCHEMA:
        raise CredentialError(
            "legacy named HPC profile metadata requires secure re-enrollment"
        )
    if metadata.get("credential_profile") != profile:
        raise CredentialError("named HPC profile metadata binding changed")
    try:
        metadata_job_id = int(metadata.get("job_id") or 0)
    except (TypeError, ValueError) as exc:
        raise CredentialError("named HPC profile job binding is invalid") from exc
    if metadata_job_id != expected_job_id:
        raise CredentialError("named HPC profile job binding changed")
    state = str(metadata.get("profile_state") or "").strip()
    if state not in PROFILE_STATES:
        raise CredentialError("named HPC profile lifecycle state is invalid")
    if state not in allowed_states:
        if allowed_states == frozenset({PROFILE_STATE_ACTIVE}):
            raise CredentialError(f"named HPC profile is not {PROFILE_STATE_ACTIVE}")
        raise CredentialError("named HPC profile state is not permitted for this operation")
    binding_id = str(metadata.get("allocation_binding_id") or "").strip()
    if not SAFE_ALLOCATION_BINDING_ID.fullmatch(binding_id):
        raise CredentialError("named HPC profile allocation binding identity is invalid")
    generation = _positive_int(
        metadata.get("allocation_generation"),
        "named HPC profile allocation generation is invalid",
    )
    instance_id = _validate_profile_instance_id(metadata.get("profile_instance_id"))
    revision = _positive_int(
        metadata.get("lifecycle_revision"),
        "named HPC profile lifecycle revision is invalid",
    )
    if state == PROFILE_STATE_ACTIVE:
        if not str(metadata.get("expected_host_uuid") or "").strip() or not str(
            metadata.get("expected_gpu_uuid") or ""
        ).strip():
            raise CredentialError("active named HPC profile has no container identity binding")
        observed_digest = str(metadata.get("container_binding_sha256") or "").strip()
        try:
            expected_digest = compute_container_binding_sha256(metadata)
        except (TypeError, ValueError) as exc:
            raise CredentialError(
                "active named HPC profile container binding fields are invalid"
            ) from exc
        if not re.fullmatch(r"[0-9a-f]{64}", observed_digest) or observed_digest != expected_digest:
            raise CredentialError("active named HPC profile container binding identity changed")
    return {
        "profile_state": state,
        "allocation_binding_id": binding_id,
        "allocation_generation": generation,
        "profile_instance_id": instance_id,
        "lifecycle_revision": revision,
    }


def load_gpu_ssh_config_for_identity_bootstrap() -> GpuSshConfig:
    """Load one provisioning profile exclusively for one-shot identity bootstrap.

    Runtime callers must use :func:`load_gpu_ssh_config`, which accepts only an
    active named profile.  Keeping this entry point separate makes the sole
    provisioning-time credential use explicit and testable.
    """

    profile = _credential_profile()
    if profile == DEFAULT_CREDENTIAL_PROFILE:
        raise CredentialError("identity bootstrap requires a named job profile")
    if sys.platform != "win32":
        raise CredentialError("identity bootstrap requires Windows DPAPI")
    injected = [name for name in STRICT_NAMED_PROFILE_OVERRIDE_KEYS if name in os.environ]
    if injected:
        raise CredentialError("identity bootstrap rejects ordinary environment overrides")
    return _load_windows_dpapi_config(
        allowed_named_profile_states=frozenset({PROFILE_STATE_PROVISIONING})
    )


def _load_windows_dpapi_config(
    *,
    strict_named_profile: bool = False,
    allowed_named_profile_states: frozenset[str] | None = None,
) -> GpuSshConfig:
    profile = _credential_profile()
    state_dir = _credential_state_dir(profile)
    if profile == DEFAULT_CREDENTIAL_PROFILE:
        return _load_windows_dpapi_config_locked(
            profile=profile,
            state_dir=state_dir,
            strict_named_profile=strict_named_profile,
            allowed_named_profile_states=allowed_named_profile_states,
        )
    with profile_lifecycle_lock(state_dir):
        return _load_windows_dpapi_config_locked(
            profile=profile,
            state_dir=state_dir,
            strict_named_profile=strict_named_profile,
            allowed_named_profile_states=allowed_named_profile_states,
        )


def _load_windows_dpapi_config_locked(
    *,
    profile: str,
    state_dir: Path,
    strict_named_profile: bool,
    allowed_named_profile_states: frozenset[str] | None,
) -> GpuSshConfig:
    """Load the user-bound CLIXML credential without persisting plaintext.

    The child PowerShell process emits one JSON object to this process. The
    secret is held only in memory and is never included in logs or exceptions.
    """
    credential_path = _profile_artifact_path(
        state_dir,
        PROFILE_CREDENTIAL_FILENAME,
        PROFILE_CREDENTIAL_FILENAME,
    )
    metadata_path = _profile_artifact_path(
        state_dir,
        PROFILE_METADATA_FILENAME,
        PROFILE_METADATA_FILENAME,
    )
    if not credential_path.is_file() or not metadata_path.is_file():
        raise CredentialError("Windows DPAPI HPC credential and metadata are not installed")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CredentialError("Windows DPAPI HPC metadata is unreadable") from exc
    if not isinstance(metadata, dict):
        raise CredentialError("Windows DPAPI HPC metadata must be a JSON object")
    lifecycle: dict[str, object] | None = None
    if profile != DEFAULT_CREDENTIAL_PROFILE:
        lifecycle = _validate_named_profile_lifecycle(
            metadata,
            profile=profile,
            profile_dir=state_dir,
            allowed_states=(
                allowed_named_profile_states
                if allowed_named_profile_states is not None
                else frozenset({PROFILE_STATE_ACTIVE})
            ),
        )
    remote_root = str(metadata.get("remote_workspace") or "")
    if remote_root.rstrip("/") != ALLOWED_GPU_REMOTE_ROOT:
        raise CredentialError("Windows DPAPI HPC metadata remote workspace is outside the allowed root")
    if strict_named_profile:
        expected_job_id = _named_job_id(profile)
        try:
            metadata_job_id = int(metadata.get("job_id") or 0)
        except (TypeError, ValueError) as exc:
            raise CredentialError("Windows DPAPI HPC metadata job binding is invalid") from exc
        if metadata_job_id != expected_job_id:
            raise CredentialError("Windows DPAPI HPC metadata job binding changed")
        if (
            str(metadata.get("host") or "") != HPC_SSH_GATEWAY_HOST
            or int(metadata.get("port") or 0) != HPC_SSH_GATEWAY_PORT
        ):
            raise CredentialError("strict named HPC profile gateway binding changed")
        try:
            host_address = ipaddress.ip_address(str(metadata.get("host") or ""))
        except ValueError as exc:
            raise CredentialError("strict named HPC profile host is invalid") from exc
        if host_address in ipaddress.ip_network("10.120.0.0/16"):
            raise CredentialError("allocation private IP is forbidden for strict HPC access")
        if not str(metadata.get("socks_host") or "").strip() or int(
            metadata.get("socks_port") or 0
        ) <= 0:
            raise CredentialError("strict named HPC profile has no designated proxy binding")
    explicit_known_hosts = (
        None
        if strict_named_profile
        else _read_value("GPU_SSH_KNOWN_HOSTS_PATH")
    )
    if explicit_known_hosts:
        known_hosts_path = Path(explicit_known_hosts).expanduser().resolve(strict=False)
    else:
        known_hosts_path = _profile_artifact_path(
            state_dir,
            metadata.get("known_hosts_path"),
            PROFILE_KNOWN_HOSTS_FILENAME,
        )
    script = (
        "$ErrorActionPreference='Stop';"
        "$c=Import-Clixml -LiteralPath $env:EVOMIND_HPC_CREDENTIAL_PATH;"
        "@{user=$c.UserName;password=$c.GetNetworkCredential().Password}|ConvertTo-Json -Compress"
    )
    env = _dpapi_subprocess_environment(credential_path)
    if profile != DEFAULT_CREDENTIAL_PROFILE and any(
        _profile_marker_present(path) for path in _profile_tombstone_paths(state_dir)
    ):
        raise CredentialError("named HPC profile is frozen or retired by tombstone")
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=15,
        )
        secret_payload = json.loads(completed.stdout.lstrip("\ufeff").strip())
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise CredentialError("Windows DPAPI HPC credential could not be loaded") from exc
    if profile != DEFAULT_CREDENTIAL_PROFILE and any(
        _profile_marker_present(path) for path in _profile_tombstone_paths(state_dir)
    ):
        secret_payload.clear()
        raise CredentialError("named HPC profile changed lifecycle state during credential load")
    socks_host = str(metadata.get("socks_host") or "")
    socks = None
    if socks_host:
        socks = SocksConfig(host=socks_host, port=int(metadata.get("socks_port") or 7890))
    config = GpuSshConfig(
        host=str(metadata.get("host") or ""),
        port=int(metadata.get("port") or 0),
        username=str(secret_payload.get("user") or ""),
        password=str(secret_payload.get("password") or ""),
        socks=socks,
        jump_host=str(metadata.get("jump_host") or "") or None,
        jump_port=int(metadata.get("jump_port") or 22),
        jump_username=str(metadata.get("jump_user") or "") or None,
        known_hosts_path=str(known_hosts_path),
        credential_profile=profile,
        expected_host_uuid=(
            (None if strict_named_profile else _read_value("EVOMIND_HPC_EXPECTED_HOST_UUID"))
            or str(metadata.get("expected_host_uuid") or metadata.get("host_uuid") or "")
            or None
        ),
        expected_gpu_uuid=(
            (None if strict_named_profile else _read_value("EVOMIND_HPC_EXPECTED_GPU_UUID"))
            or str(metadata.get("expected_gpu_uuid") or metadata.get("gpu_uuid") or "")
            or None
        ),
        job_id=(int(metadata.get("job_id")) if metadata.get("job_id") is not None else None),
        remote_workspace=remote_root,
        strict_named_profile=strict_named_profile,
        profile_state=(str(lifecycle["profile_state"]) if lifecycle else None),
        allocation_binding_id=(
            str(lifecycle["allocation_binding_id"]) if lifecycle else None
        ),
        allocation_generation=(
            int(lifecycle["allocation_generation"]) if lifecycle else None
        ),
        profile_instance_id=(str(lifecycle["profile_instance_id"]) if lifecycle else None),
    )
    if not config.host or config.port <= 0 or not config.username or not config.password:
        raise CredentialError("Windows DPAPI HPC credential metadata is incomplete")
    return config


def _assert_named_profile_runtime_active(
    config: GpuSshConfig,
    *,
    lifecycle_lock_held: bool = False,
) -> None:
    """Revalidate a strict profile immediately before opening a network socket."""

    if not config.strict_named_profile:
        return
    profile = config.credential_profile
    profile_dir = _credential_state_dir(profile)
    if not lifecycle_lock_held:
        with profile_lifecycle_lock(profile_dir):
            _assert_named_profile_runtime_active(
                config,
                lifecycle_lock_held=True,
            )
        return
    metadata_path = _profile_artifact_path(
        profile_dir,
        PROFILE_METADATA_FILENAME,
        PROFILE_METADATA_FILENAME,
    )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CredentialError("named HPC profile runtime metadata is unreadable") from exc
    if not isinstance(metadata, dict):
        raise CredentialError("named HPC profile runtime metadata must be a JSON object")
    lifecycle = _validate_named_profile_lifecycle(
        metadata,
        profile=profile,
        profile_dir=profile_dir,
        allowed_states=frozenset({PROFILE_STATE_ACTIVE}),
    )
    expected = {
        "host": config.host,
        "port": config.port,
        "remote_workspace": config.remote_workspace,
        "expected_host_uuid": config.expected_host_uuid,
        "expected_gpu_uuid": config.expected_gpu_uuid,
        "allocation_binding_id": config.allocation_binding_id,
        "allocation_generation": config.allocation_generation,
        "profile_instance_id": config.profile_instance_id,
    }
    try:
        observed = {
            "host": str(metadata.get("host") or ""),
            "port": int(metadata.get("port") or 0),
            "remote_workspace": str(metadata.get("remote_workspace") or ""),
            "expected_host_uuid": str(metadata.get("expected_host_uuid") or "") or None,
            "expected_gpu_uuid": str(metadata.get("expected_gpu_uuid") or "") or None,
            "allocation_binding_id": str(lifecycle["allocation_binding_id"]),
            "allocation_generation": int(lifecycle["allocation_generation"]),
            "profile_instance_id": str(lifecycle["profile_instance_id"]),
        }
    except (TypeError, ValueError) as exc:
        raise CredentialError("named HPC profile runtime binding is invalid") from exc
    if observed != expected:
        raise CredentialError("named HPC profile runtime binding changed after credential load")


def open_socks_channel(socks: "SocksConfig", dest_host: str, dest_port: int, timeout: int = 20):
    """Open a raw socket to dest via a SOCKS5 proxy (no external deps).

    Returns a connected socket suitable for paramiko's ``sock=`` argument.
    Supports optional username/password (RFC 1929) auth.
    """
    import socket
    import struct

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((socks.host, socks.port))
    except Exception:
        sock.close()
        raise
    def recv_exact(size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                raise RetryableTransportError("SOCKS5 connection closed during handshake")
            data.extend(chunk)
        return bytes(data)

    try:
        if socks.username and socks.password:
            sock.send(b"\x05\x01\x02")  # one method: username/password
            if recv_exact(2) != b"\x05\x02":
                raise CredentialError("SOCKS5 proxy rejected username/password authentication")
            user = socks.username.encode()
            pw = socks.password.encode()
            sock.send(b"\x01" + bytes([len(user)]) + user + bytes([len(pw)]) + pw)
            if recv_exact(2) != b"\x01\x00":
                raise CredentialError("SOCKS5 username/password authentication failed")
        else:
            sock.send(b"\x05\x01\x00")  # no auth
            if recv_exact(2) != b"\x05\x00":
                raise CredentialError("SOCKS5 proxy rejected no-auth mode")
        host_bytes = dest_host.encode()
        if len(host_bytes) > 255:
            raise CredentialError("SOCKS5 destination host is too long")
        sock.send(b"\x05\x01\x00\x03" + bytes([len(host_bytes)]) + host_bytes + struct.pack("!H", dest_port))
        head = recv_exact(4)
        if head[0] != 5:
            raise CredentialError("SOCKS5 proxy returned an invalid CONNECT response")
        if head[1] != 0:
            raise RetryableTransportError(
                f"SOCKS5 proxy could not connect to target (code={head[1]})"
            )
        address_type = head[3]
        if address_type == 1:
            recv_exact(4)
        elif address_type == 3:
            recv_exact(recv_exact(1)[0])
        elif address_type == 4:
            recv_exact(16)
        else:
            raise CredentialError("SOCKS5 proxy returned an unsupported address type")
        recv_exact(2)
        return sock
    except Exception:
        sock.close()
        raise


def _connect_ssh_unlocked(config: GpuSshConfig, *, timeout: int = 20):
    """Open a paramiko SSH connection using env-resolved credentials.

    This is the single secure entry point that replaces hardcoded connection
    helpers. paramiko is imported lazily so this module stays import-safe when
    paramiko is not installed. Never logs the password.
    """
    try:
        import paramiko
    except ImportError as exc:  # pragma: no cover
        raise CredentialError("paramiko is required for connect_ssh (pip install -r requirements.txt)") from exc

    if not config.known_hosts_path:
        raise CredentialError(
            "Pinned SSH known-hosts path is required; set GPU_SSH_KNOWN_HOSTS_PATH"
        )
    known_hosts_path = Path(config.known_hosts_path).expanduser()
    try:
        known_hosts_ready = known_hosts_path.is_file() and known_hosts_path.stat().st_size > 0
    except OSError:
        known_hosts_ready = False
    if not known_hosts_ready:
        raise CredentialError("Pinned SSH known-hosts file is missing or empty")

    class CascadingSshClient(paramiko.SSHClient):
        def __init__(self, upstream=None):
            super().__init__()
            self._evomind_upstream = upstream

        def close(self):
            try:
                super().close()
            finally:
                if self._evomind_upstream is not None:
                    self._evomind_upstream.close()
                    self._evomind_upstream = None

    def apply_pinned_host_keys(client) -> None:
        try:
            client.load_host_keys(str(known_hosts_path))
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
        except Exception as exc:
            raise CredentialError("Pinned SSH known-hosts file could not be loaded") from exc

    def auth_kwargs(*, host: str, port: int, username: str, sock=None) -> dict:
        kwargs = {
            "hostname": host,
            "port": port,
            "username": username,
            "sock": sock,
            "timeout": timeout,
            "banner_timeout": max(timeout, 30),
            "allow_agent": False,
            "look_for_keys": False,
        }
        if config.key_path:
            kwargs["key_filename"] = config.key_path
        if config.password:
            kwargs["password"] = config.password
        return kwargs

    jump_client = None
    entry_host = config.jump_host or config.host
    entry_port = config.jump_port if config.jump_host else config.port
    entry_socket = None
    if config.socks is not None:
        entry_socket = open_socks_channel(config.socks, entry_host, entry_port, timeout=timeout)
    try:
        target_socket = entry_socket
        if config.jump_host:
            jump_client = paramiko.SSHClient()
            apply_pinned_host_keys(jump_client)
            jump_client.connect(**auth_kwargs(
                host=config.jump_host,
                port=config.jump_port,
                username=config.jump_username or config.username,
                sock=entry_socket,
            ))
            transport = jump_client.get_transport()
            if transport is None or not transport.is_active():
                raise CredentialError("HPC SSH jump transport is not active")
            target_socket = transport.open_channel(
                "direct-tcpip",
                (config.host, config.port),
                ("127.0.0.1", 0),
            )

        client = CascadingSshClient(upstream=jump_client)
        apply_pinned_host_keys(client)
        client.connect(**auth_kwargs(
            host=config.host,
            port=config.port,
            username=config.username,
            sock=target_socket,
        ))
        return client
    except Exception:
        if jump_client is not None:
            jump_client.close()
        elif entry_socket is not None:
            entry_socket.close()
        raise


def connect_ssh(config: Optional[GpuSshConfig] = None, *, timeout: int = 20):
    """Connect using one lifecycle-linearized strict named-profile snapshot."""

    resolved = config or load_gpu_ssh_config()
    if not resolved.strict_named_profile:
        return _connect_ssh_unlocked(resolved, timeout=timeout)
    profile_dir = _credential_state_dir(resolved.credential_profile)
    with profile_lifecycle_lock(profile_dir):
        _assert_named_profile_runtime_active(
            resolved,
            lifecycle_lock_held=True,
        )
        return _connect_ssh_unlocked(resolved, timeout=timeout)


def verify_job_container_identity(
    client,
    config: GpuSshConfig,
    *,
    expected_job_id: int,
    expected_root: str = ALLOWED_GPU_REMOTE_ROOT,
    expected_gpu_name_fragment: str = "A800",
    minimum_gpu_memory_mib: int = 80_000,
) -> dict:
    """Verify the routed job container on the same SSH connection, read-only."""

    expected_profile = f"job{int(expected_job_id)}"
    if not config.strict_named_profile:
        raise CredentialError("job-container verification requires a strict named profile")
    if config.credential_profile != expected_profile or config.job_id != int(expected_job_id):
        raise CredentialError("job-container credential binding changed")
    if config.host != HPC_SSH_GATEWAY_HOST or config.port != HPC_SSH_GATEWAY_PORT:
        raise CredentialError("job-container gateway binding changed")
    if config.socks is None:
        raise CredentialError("job-container connection did not use the designated proxy")
    if (config.remote_workspace or "").rstrip("/") != expected_root.rstrip("/"):
        raise CredentialError("job-container remote workspace binding changed")
    if not config.expected_host_uuid or not config.expected_gpu_uuid:
        raise CredentialError("job-container expected identity is incomplete")
    source = r'''from __future__ import annotations
import csv, io, json, os, pathlib, subprocess

root = pathlib.Path(__EXPECTED_ROOT__)
try:
    resolved = root.resolve(strict=True)
except OSError:
    resolved = pathlib.Path("")
host_uuid = ""
for candidate in ("/sys/class/dmi/id/product_uuid", "/etc/machine-id"):
    try:
        host_uuid = pathlib.Path(candidate).read_text(encoding="utf-8").strip()
    except OSError:
        continue
    if host_uuid:
        break
completed = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader,nounits"],
    check=False, capture_output=True, text=True, encoding="utf-8", timeout=20,
)
gpus = []
if completed.returncode == 0:
    for row in csv.reader(io.StringIO(completed.stdout)):
        if len(row) >= 3:
            try:
                gpus.append({"name":row[0].strip(),"uuid":row[1].strip(),
                             "memory_total_mib":int(float(row[2].strip()))})
            except ValueError:
                pass
print(json.dumps({
    "host_uuid": host_uuid,
    "gpus": gpus,
    "root_requested": str(root),
    "root_realpath": str(resolved),
    "root_exists": root.is_dir(),
    "root_writable": os.access(root, os.W_OK),
    "read_only": True,
    "signals_sent": 0,
    "other_processes_modified": False,
}, sort_keys=True))
'''.replace("__EXPECTED_ROOT__", repr(expected_root))
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    command = "python3 -c " + shlex.quote(
        f"import base64;exec(base64.b64decode({encoded!r}))"
    )
    _stdin, stdout, stderr = client.exec_command(command, timeout=30)
    output = stdout.read()
    error = stderr.read()
    if isinstance(output, bytes):
        output = output.decode("utf-8", "replace")
    if isinstance(error, bytes):
        error = error.decode("utf-8", "replace")
    exit_code = stdout.channel.recv_exit_status()
    if exit_code or not str(output).strip():
        raise CredentialError("read-only job-container identity probe failed")
    try:
        payload = json.loads(str(output).strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise CredentialError("job-container identity probe returned invalid evidence") from exc
    observed_host = str(payload.get("host_uuid") or "").strip().lower()
    expected_host = str(config.expected_host_uuid).strip().lower()
    observed_gpus = payload.get("gpus") if isinstance(payload.get("gpus"), list) else []
    expected_gpu_uuids = {
        value.strip().lower()
        for value in str(config.expected_gpu_uuid).split(",")
        if value.strip()
    }
    observed_gpu_uuids = {
        str(item.get("uuid") or "").strip().lower()
        for item in observed_gpus
        if isinstance(item, dict) and str(item.get("uuid") or "").strip()
    }
    if observed_host != expected_host:
        raise CredentialError("job-container Host UUID mismatch")
    if observed_gpu_uuids != expected_gpu_uuids or len(observed_gpus) != 1:
        raise CredentialError("job-container GPU UUID mismatch")
    gpu = observed_gpus[0]
    if expected_gpu_name_fragment not in str(gpu.get("name") or ""):
        raise CredentialError("job-container GPU model mismatch")
    if int(gpu.get("memory_total_mib") or 0) < int(minimum_gpu_memory_mib):
        raise CredentialError("job-container GPU memory mismatch")
    if (
        payload.get("root_exists") is not True
        or payload.get("root_writable") is not True
        or str(payload.get("root_realpath") or "").rstrip("/")
        != expected_root.rstrip("/")
    ):
        raise CredentialError("job-container dedicated root mismatch")
    if payload.get("signals_sent") != 0 or payload.get("other_processes_modified") is not False:
        raise CredentialError("job-container identity probe violated process boundaries")
    return {
        "job_id": int(expected_job_id),
        "credential_profile": config.credential_profile,
        "host_uuid": observed_host,
        "gpu_uuids": sorted(observed_gpu_uuids),
        "gpu_name": str(gpu.get("name") or ""),
        "gpu_memory_total_mib": int(gpu.get("memory_total_mib") or 0),
        "remote_root": expected_root,
        "designated_proxy_path_verified": True,
        "pinned_host_key_verified": True,
        "job_container_verified": True,
        "read_only": True,
        "signals_sent": 0,
        "other_processes_modified": False,
    }
