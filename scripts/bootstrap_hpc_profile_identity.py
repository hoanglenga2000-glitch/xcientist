#!/usr/bin/env python3
"""Bootstrap one named HPC profile with a pinned SSH key and host/GPU identity.

This command is intentionally a one-shot trust-on-first-use (TOFU) operation.
It accepts no endpoint or authentication arguments: both are loaded from the
named Windows DPAPI profile selected by ``EVOMIND_HPC_CREDENTIAL_PROFILE``.
The observed host key is first written to a private staging file and all remote
checks then reconnect through the regular RejectPolicy-only SSH path.  Profile
metadata is committed only after every check passes.

The remote probe is read-only.  It reads the host UUID, queries the visible GPU,
and checks the dedicated EvoMind root with ``test -d``/``test -w``.  It never
prints authentication material or raw remote command output.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import socket
import sys
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_agent_workstation.server.core.gpu_credentials import (  # noqa: E402
    ALLOWED_GPU_REMOTE_ROOT,
    DEFAULT_CREDENTIAL_PROFILE,
    PROFILE_CREDENTIAL_FILENAME,
    PROFILE_FROZEN_TOMBSTONE_FILENAME,
    PROFILE_KNOWN_HOSTS_FILENAME,
    PROFILE_METADATA_FILENAME,
    PROFILE_METADATA_SCHEMA,
    PROFILE_RETIRED_TOMBSTONE_FILENAME,
    PROFILE_STATE_ACTIVE,
    PROFILE_STATE_PROVISIONING,
    SAFE_CREDENTIAL_PROFILE,
    STRICT_NAMED_PROFILE_OVERRIDE_KEYS,
    CredentialError,
    GpuSshConfig,
    compute_container_binding_sha256,
    connect_ssh,
    load_gpu_ssh_config_for_identity_bootstrap,
    open_socks_channel,
    profile_lifecycle_lock,
)

PROFILE_ENV = "EVOMIND_HPC_CREDENTIAL_PROFILE"
DIRECT_OVERRIDE_ENV_VARS = STRICT_NAMED_PROFILE_OVERRIDE_KEYS
SAFE_ENDPOINT_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,252}$")
SAFE_IDENTITY_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
MIN_A800_MEMORY_MIB = 76_000
MAX_A800_MEMORY_MIB = 86_000
MAX_REMOTE_OUTPUT_BYTES = 64 * 1024

REMOTE_IDENTITY_COMMAND = f"""set -eu
host_uuid=''
host_uuid_source=''
if [ -r /sys/class/dmi/id/product_uuid ]; then
  host_uuid=$(cat /sys/class/dmi/id/product_uuid)
  host_uuid_source='product_uuid'
fi
if [ -z "$host_uuid" ] && [ -r /etc/machine-id ]; then
  host_uuid=$(cat /etc/machine-id)
  host_uuid_source='machine_id'
fi
if [ -z "$host_uuid" ]; then
  exit 51
fi
printf 'EVOMIND_HOST_UUID_SOURCE=%s\n' "$host_uuid_source"
printf 'EVOMIND_HOST_UUID=%s\n' "$host_uuid"
if [ -d '{ALLOWED_GPU_REMOTE_ROOT}' ]; then
  printf 'EVOMIND_REMOTE_ROOT_EXISTS=1\n'
else
  printf 'EVOMIND_REMOTE_ROOT_EXISTS=0\n'
fi
if [ -w '{ALLOWED_GPU_REMOTE_ROOT}' ]; then
  printf 'EVOMIND_REMOTE_ROOT_WRITABLE=1\n'
else
  printf 'EVOMIND_REMOTE_ROOT_WRITABLE=0\n'
fi
printf 'EVOMIND_GPU_CSV_BEGIN\n'
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits
printf 'EVOMIND_GPU_CSV_END\n'
"""


class BootstrapError(RuntimeError):
    """Raised when a profile cannot be safely identity-bound."""


@dataclass(frozen=True)
class ProfileContext:
    profile: str
    profile_dir: Path
    credential_path: Path
    metadata_path: Path
    known_hosts_path: Path
    metadata: dict[str, Any]
    metadata_sha256: str
    config: GpuSshConfig


@dataclass(frozen=True)
class HostKeyEvidence:
    algorithm: str
    base64_value: str
    fingerprint_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BootstrapError(message)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _marker_present(path: Path) -> bool:
    try:
        return path.exists() or path.is_symlink()
    except OSError:
        return True


def _artifact_path(profile_dir: Path, raw_value: object, default_name: str) -> Path:
    raw = str(raw_value or default_name)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = profile_dir / candidate
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(profile_dir.resolve(strict=False))
    except (OSError, ValueError) as exc:
        raise BootstrapError("profile artifact path escapes the named profile directory") from exc
    return resolved


def _validate_endpoint(host: str, port: int) -> None:
    _require(bool(SAFE_ENDPOINT_HOST.fullmatch(host)), "profile endpoint host is invalid")
    _require("[" not in host and "]" not in host, "profile endpoint host must not be bracketed")
    _require(1 <= port <= 65535, "profile endpoint port is invalid")


def _validate_identity(value: str, label: str) -> str:
    normalized = value.strip()
    _require(bool(SAFE_IDENTITY_VALUE.fullmatch(normalized)), f"observed {label} is invalid")
    return normalized


def load_named_profile_context() -> ProfileContext:
    """Load exactly one named Windows DPAPI profile and its explicit endpoint."""

    profile = os.environ.get(PROFILE_ENV, "").strip()
    _require(bool(profile), f"{PROFILE_ENV} must select a named DPAPI profile")
    _require(
        profile != DEFAULT_CREDENTIAL_PROFILE,
        "identity bootstrap requires an isolated named DPAPI profile",
    )
    _require(
        profile not in {".", ".."} and bool(SAFE_CREDENTIAL_PROFILE.fullmatch(profile)),
        f"{PROFILE_ENV} contains unsafe profile characters",
    )
    active_overrides = [name for name in DIRECT_OVERRIDE_ENV_VARS if name in os.environ]
    _require(
        not active_overrides,
        "direct SSH or identity overrides must be cleared when bootstrapping a DPAPI profile",
    )

    appdata = os.environ.get("APPDATA", "").strip()
    _require(bool(appdata), "APPDATA is required to resolve the named DPAPI profile")
    appdata_path = Path(appdata).expanduser()
    _require(appdata_path.is_absolute(), "APPDATA must be absolute for named DPAPI profiles")
    profile_dir = (
        appdata_path / "ResearchAgentWorkstation" / "profiles" / profile
    ).resolve(strict=False)
    credential_path = _artifact_path(
        profile_dir,
        PROFILE_CREDENTIAL_FILENAME,
        PROFILE_CREDENTIAL_FILENAME,
    )
    metadata_path = _artifact_path(
        profile_dir,
        PROFILE_METADATA_FILENAME,
        PROFILE_METADATA_FILENAME,
    )
    _require(credential_path.is_file(), "named DPAPI credential is not installed")
    _require(metadata_path.is_file(), "named DPAPI metadata is not installed")
    try:
        metadata_bytes = metadata_path.read_bytes()
        metadata_payload = json.loads(metadata_bytes.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError("named DPAPI metadata is unreadable") from exc
    _require(isinstance(metadata_payload, dict), "named DPAPI metadata must be a JSON object")
    metadata: dict[str, Any] = dict(metadata_payload)
    _require(
        metadata.get("schema") == PROFILE_METADATA_SCHEMA,
        "legacy DPAPI metadata requires secure re-enrollment",
    )
    _require(
        metadata.get("profile_state") == PROFILE_STATE_PROVISIONING,
        "identity bootstrap requires a provisioning profile",
    )
    _require(
        not _marker_present(profile_dir / PROFILE_FROZEN_TOMBSTONE_FILENAME)
        and not _marker_present(profile_dir / PROFILE_RETIRED_TOMBSTONE_FILENAME),
        "frozen or retired profile cannot be bootstrapped",
    )
    _require(
        metadata.get("credential_profile") == profile,
        "DPAPI metadata is bound to a different credential profile",
    )
    job_match = re.fullmatch(r"job([0-9]+)", profile)
    if job_match:
        _require(
            int(metadata.get("job_id") or 0) == int(job_match.group(1)),
            "DPAPI metadata is bound to a different HPC job",
        )
    _require(bool(job_match), "identity bootstrap requires job<job_id> profile naming")
    allocation_binding_id = str(metadata.get("allocation_binding_id") or "").strip()
    _require(
        bool(allocation_binding_id),
        "DPAPI metadata has no allocation binding identity",
    )
    try:
        allocation_generation = int(metadata.get("allocation_generation") or 0)
        lifecycle_revision = int(metadata.get("lifecycle_revision") or 0)
        uuid.UUID(str(metadata.get("profile_instance_id") or ""))
    except (TypeError, ValueError, AttributeError) as exc:
        raise BootstrapError("DPAPI allocation lifecycle metadata is invalid") from exc
    _require(
        allocation_generation > 0 and lifecycle_revision > 0,
        "DPAPI allocation lifecycle metadata is invalid",
    )
    _require(
        str(metadata.get("remote_workspace") or "").rstrip("/")
        == ALLOWED_GPU_REMOTE_ROOT,
        "DPAPI metadata remote workspace differs from the dedicated EvoMind root",
    )
    host = str(metadata.get("host") or "").strip()
    try:
        port = int(metadata.get("port") or 0)
    except (TypeError, ValueError) as exc:
        raise BootstrapError("profile endpoint port is invalid") from exc
    _validate_endpoint(host, port)
    known_hosts_path = _artifact_path(
        profile_dir,
        metadata.get("known_hosts_path"),
        PROFILE_KNOWN_HOSTS_FILENAME,
    )
    _require(
        not metadata.get("expected_host_uuid") and not metadata.get("expected_gpu_uuid"),
        "profile identity is already bound",
    )
    try:
        known_hosts_nonempty = known_hosts_path.is_file() and known_hosts_path.stat().st_size > 0
    except OSError as exc:
        raise BootstrapError("profile known-hosts state is unreadable") from exc
    _require(not known_hosts_nonempty, "profile SSH host key is already pinned")

    try:
        config = load_gpu_ssh_config_for_identity_bootstrap()
    except CredentialError as exc:
        raise BootstrapError("named DPAPI profile could not be loaded") from exc
    _require(config.credential_profile == profile, "credential loader selected a different profile")
    _require(
        config.host == host and config.port == port,
        "credential loader endpoint differs from the DPAPI metadata endpoint",
    )
    _require(config.has_auth(), "named DPAPI profile has no usable authentication material")
    _require(
        config.profile_state == PROFILE_STATE_PROVISIONING,
        "credential loader did not select the provisioning profile state",
    )
    _require(
        config.allocation_binding_id == allocation_binding_id
        and config.allocation_generation == allocation_generation
        and config.profile_instance_id == str(metadata.get("profile_instance_id")),
        "credential loader allocation binding differs from metadata",
    )
    _require(not config.jump_host, "identity bootstrap only accepts the profile's direct endpoint")
    _require(
        bool(config.known_hosts_path)
        and Path(str(config.known_hosts_path)).resolve(strict=False) == known_hosts_path,
        "credential loader known-hosts path differs from the profile-owned path",
    )
    _require(
        not config.expected_host_uuid and not config.expected_gpu_uuid,
        "credential loader returned an already-bound identity",
    )

    return ProfileContext(
        profile=profile,
        profile_dir=profile_dir,
        credential_path=credential_path,
        metadata_path=metadata_path,
        known_hosts_path=known_hosts_path,
        metadata=metadata,
        metadata_sha256=_sha256_hex(metadata_bytes),
        config=config,
    )


def collect_server_host_key(config: GpuSshConfig, *, timeout: int) -> HostKeyEvidence:
    """Collect one endpoint key without authentication for the one-shot TOFU step."""

    try:
        import paramiko
    except ImportError as exc:  # pragma: no cover - project dependency
        raise BootstrapError("paramiko is required for HPC identity bootstrap") from exc

    raw_socket = None
    transport = None
    try:
        if config.socks is None:
            raw_socket = socket.create_connection((config.host, config.port), timeout=timeout)
            raw_socket.settimeout(timeout)
        else:
            raw_socket = open_socks_channel(
                config.socks,
                config.host,
                config.port,
                timeout=timeout,
            )
        # This one-shot bootstrap only reads the endpoint key before any
        # authentication. It never attaches the unauthenticated transport to a
        # client or executes a remote command.
        transport_factory = paramiko.Transport
        transport = transport_factory(raw_socket)
        transport.start_client(timeout=timeout)
        key = transport.get_remote_server_key()
        algorithm = str(key.get_name() or "").strip()
        base64_value = str(key.get_base64() or "").strip()
        key_bytes = bytes(key.asbytes())
    except Exception as exc:
        raise BootstrapError("SSH endpoint host key could not be collected") from exc
    finally:
        if transport is not None:
            transport.close()
        elif raw_socket is not None:
            raw_socket.close()

    _require(bool(algorithm) and not any(ch.isspace() for ch in algorithm), "SSH host key algorithm is invalid")
    _require(bool(base64_value) and not any(ch.isspace() for ch in base64_value), "SSH host key payload is invalid")
    try:
        decoded = base64.b64decode(base64_value, validate=True)
    except (ValueError, TypeError) as exc:
        raise BootstrapError("SSH host key payload is invalid") from exc
    _require(decoded == key_bytes, "SSH host key encoding is inconsistent")
    fingerprint = base64.b64encode(hashlib.sha256(key_bytes).digest()).decode("ascii").rstrip("=")
    return HostKeyEvidence(
        algorithm=algorithm,
        base64_value=base64_value,
        fingerprint_sha256=f"SHA256:{fingerprint}",
    )


def known_hosts_alias(host: str, port: int) -> str:
    """Return the host token Paramiko/OpenSSH use for this endpoint."""

    _validate_endpoint(host, port)
    return host if port == 22 else f"[{host}]:{port}"


def render_known_hosts(config: GpuSshConfig, key: HostKeyEvidence) -> bytes:
    alias = known_hosts_alias(config.host, config.port)
    return f"{alias} {key.algorithm} {key.base64_value}\n".encode("ascii")


def _read_stream_limited(stream: Any) -> bytes:
    payload = stream.read(MAX_REMOTE_OUTPUT_BYTES + 1)
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    _require(isinstance(payload, bytes), "remote identity probe returned an invalid stream")
    _require(len(payload) <= MAX_REMOTE_OUTPUT_BYTES, "remote identity probe output exceeded its limit")
    return payload


def parse_remote_identity_output(raw: bytes) -> dict[str, Any]:
    """Parse and validate the fixed, read-only remote identity probe output."""

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BootstrapError("remote identity probe output is not UTF-8") from exc
    lines = text.splitlines()
    _require(lines.count("EVOMIND_GPU_CSV_BEGIN") == 1, "remote GPU evidence start marker is invalid")
    _require(lines.count("EVOMIND_GPU_CSV_END") == 1, "remote GPU evidence end marker is invalid")
    begin = lines.index("EVOMIND_GPU_CSV_BEGIN")
    end = lines.index("EVOMIND_GPU_CSV_END")
    _require(begin < end, "remote GPU evidence markers are out of order")

    scalar_values: dict[str, str] = {}
    for line in lines[:begin]:
        key, separator, value = line.partition("=")
        if separator and key.startswith("EVOMIND_"):
            _require(key not in scalar_values, "remote identity probe repeated a scalar field")
            scalar_values[key] = value.strip()
    host_uuid_source = scalar_values.get("EVOMIND_HOST_UUID_SOURCE", "")
    _require(
        host_uuid_source in {"product_uuid", "machine_id"},
        "remote host UUID source is invalid",
    )
    host_uuid = _validate_identity(
        scalar_values.get("EVOMIND_HOST_UUID", ""),
        "host UUID",
    )
    _require(
        scalar_values.get("EVOMIND_REMOTE_ROOT_EXISTS") == "1",
        "dedicated EvoMind remote root does not exist",
    )
    _require(
        scalar_values.get("EVOMIND_REMOTE_ROOT_WRITABLE") == "1",
        "dedicated EvoMind remote root is not writable by the profile user",
    )

    gpu_rows = [row for row in csv.reader(io.StringIO("\n".join(lines[begin + 1 : end]))) if row]
    _require(len(gpu_rows) == 1, "HPC allocation must expose exactly one GPU")
    row = gpu_rows[0]
    _require(len(row) == 4, "remote GPU identity row is invalid")
    try:
        index = int(row[0].strip())
        memory_total_mib = int(round(float(row[3].strip())))
    except ValueError as exc:
        raise BootstrapError("remote GPU numeric evidence is invalid") from exc
    gpu_uuid = _validate_identity(row[1].strip(), "GPU UUID")
    gpu_name = row[2].strip()
    _require(
        bool(re.search(r"(?<![A-Za-z0-9])A800(?![A-Za-z0-9])", gpu_name, flags=re.IGNORECASE)),
        "visible GPU is not an A800",
    )
    _require(
        MIN_A800_MEMORY_MIB <= memory_total_mib <= MAX_A800_MEMORY_MIB,
        "visible A800 does not have approximately 80 GB of memory",
    )
    return {
        "host_uuid": host_uuid,
        "host_uuid_source": host_uuid_source,
        "gpu": {
            "index": index,
            "uuid": gpu_uuid,
            "name": gpu_name,
            "memory_total_mib": memory_total_mib,
        },
        "remote_workspace": {
            "path": ALLOWED_GPU_REMOTE_ROOT,
            "exists": True,
            "writable": True,
            "write_probe_performed": False,
        },
        "read_only": True,
    }


def inspect_remote_identity(
    config: GpuSshConfig,
    *,
    staged_known_hosts_path: Path,
    timeout: int,
    connector: Callable[..., Any] = connect_ssh,
) -> dict[str, Any]:
    """Reconnect with RejectPolicy and execute the fixed read-only probe."""

    staged_config = replace(config, known_hosts_path=str(staged_known_hosts_path))
    client = None
    try:
        client = connector(staged_config, timeout=timeout)
        _stdin, stdout, stderr = client.exec_command(REMOTE_IDENTITY_COMMAND, timeout=timeout)
        output = _read_stream_limited(stdout)
        _read_stream_limited(stderr)
        exit_status = int(stdout.channel.recv_exit_status())
    except BootstrapError:
        raise
    except Exception as exc:
        raise BootstrapError("pinned SSH identity verification failed") from exc
    finally:
        if client is not None:
            client.close()
    _require(exit_status == 0, "remote identity probe failed")
    return parse_remote_identity_output(output)


def _stage_bytes(target: Path, payload: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _commit_profile_binding_locked(
    context: ProfileContext,
    *,
    known_hosts_payload: bytes,
    metadata_payload: bytes,
    replace_file: Callable[[Path, Path], Any] = os.replace,
) -> None:
    """Commit both artifacts and restore the original state on a failed replace."""

    known_hosts_existed = context.known_hosts_path.exists()
    try:
        previous_known_hosts = context.known_hosts_path.read_bytes() if known_hosts_existed else b""
        previous_metadata = context.metadata_path.read_bytes()
    except OSError as exc:
        raise BootstrapError("profile state could not be snapshotted before commit") from exc
    staged_known_hosts = _stage_bytes(context.known_hosts_path, known_hosts_payload)
    staged_metadata = _stage_bytes(context.metadata_path, metadata_payload)
    known_hosts_replaced = False
    try:
        replace_file(staged_known_hosts, context.known_hosts_path)
        known_hosts_replaced = True
        replace_file(staged_metadata, context.metadata_path)
    except Exception as exc:
        rollback_errors: list[Exception] = []
        if known_hosts_replaced:
            try:
                if known_hosts_existed:
                    rollback_known_hosts = _stage_bytes(
                        context.known_hosts_path,
                        previous_known_hosts,
                    )
                    replace_file(rollback_known_hosts, context.known_hosts_path)
                else:
                    context.known_hosts_path.unlink(missing_ok=True)
            except Exception as rollback_exc:  # pragma: no cover - catastrophic filesystem failure
                rollback_errors.append(rollback_exc)
        try:
            if context.metadata_path.read_bytes() != previous_metadata:
                rollback_metadata = _stage_bytes(context.metadata_path, previous_metadata)
                replace_file(rollback_metadata, context.metadata_path)
        except Exception as rollback_exc:  # pragma: no cover - catastrophic filesystem failure
            rollback_errors.append(rollback_exc)
        staged_known_hosts.unlink(missing_ok=True)
        staged_metadata.unlink(missing_ok=True)
        if rollback_errors:
            raise BootstrapError("profile binding commit and rollback both failed") from exc
        raise BootstrapError("profile binding commit failed; original state was restored") from exc
    finally:
        staged_known_hosts.unlink(missing_ok=True)
        staged_metadata.unlink(missing_ok=True)


def _require_activation_context_unchanged_locked(context: ProfileContext) -> None:
    if (
        _marker_present(context.profile_dir / PROFILE_FROZEN_TOMBSTONE_FILENAME)
        or _marker_present(context.profile_dir / PROFILE_RETIRED_TOMBSTONE_FILENAME)
    ):
        raise BootstrapError("profile was frozen or retired during bootstrap")
    try:
        current_metadata = context.metadata_path.read_bytes()
    except OSError as exc:
        raise BootstrapError("profile metadata changed during bootstrap") from exc
    if _sha256_hex(current_metadata) != context.metadata_sha256:
        raise BootstrapError("profile metadata changed during bootstrap")
    try:
        current_payload = json.loads(current_metadata.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError("profile metadata changed during bootstrap") from exc
    if (
        not isinstance(current_payload, dict)
        or current_payload.get("schema") != PROFILE_METADATA_SCHEMA
        or current_payload.get("profile_state") != PROFILE_STATE_PROVISIONING
    ):
        raise BootstrapError("profile is no longer eligible for activation")


def commit_profile_binding(
    context: ProfileContext,
    *,
    known_hosts_payload: bytes,
    metadata_payload: bytes,
    replace_file: Callable[[Path, Path], Any] = os.replace,
) -> None:
    """Serialize activation against freeze/retire and commit both artifacts."""

    try:
        with profile_lifecycle_lock(context.profile_dir):
            _require_activation_context_unchanged_locked(context)
            _commit_profile_binding_locked(
                context,
                known_hosts_payload=known_hosts_payload,
                metadata_payload=metadata_payload,
                replace_file=replace_file,
            )
    except CredentialError as exc:
        raise BootstrapError("profile lifecycle lock failed during bootstrap") from exc


def _bootstrap_profile_identity_locked(
    *,
    context: ProfileContext,
    timeout: int,
    now: datetime | None = None,
    host_key_collector: Callable[..., HostKeyEvidence] = collect_server_host_key,
    identity_inspector: Callable[..., dict[str, Any]] = inspect_remote_identity,
) -> dict[str, Any]:
    """Perform bootstrap while the caller holds the lifecycle lock."""

    key = host_key_collector(context.config, timeout=timeout)
    known_hosts_payload = render_known_hosts(context.config, key)
    staged_known_hosts = _stage_bytes(context.known_hosts_path, known_hosts_payload)
    try:
        identity = identity_inspector(
            context.config,
            staged_known_hosts_path=staged_known_hosts,
            timeout=timeout,
        )
    finally:
        staged_known_hosts.unlink(missing_ok=True)

    captured_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    captured_at_text = captured_at.isoformat(timespec="seconds").replace("+00:00", "Z")
    evidence_digest = _sha256_hex(_canonical_json_bytes(identity))
    evidence_summary = {
        "schema": "evomind.hpc.identity_bootstrap_evidence.v1",
        "mode": "tofu_once_then_reject_policy",
        "endpoint": {
            "host": context.config.host,
            "port": context.config.port,
        },
        "ssh_host_key": {
            "algorithm": key.algorithm,
            "fingerprint_sha256": key.fingerprint_sha256,
            "known_hosts_alias": known_hosts_alias(context.config.host, context.config.port),
        },
        "host": {
            "uuid_source": identity["host_uuid_source"],
        },
        "gpu": {
            "count": 1,
            "name": identity["gpu"]["name"],
            "memory_total_mib": identity["gpu"]["memory_total_mib"],
        },
        "remote_workspace": identity["remote_workspace"],
        "probe_read_only": True,
        "reject_policy_verified": True,
        "evidence_sha256": evidence_digest,
    }
    updated_metadata = dict(context.metadata)
    updated_metadata.update(
        {
            "expected_host_uuid": identity["host_uuid"],
            "expected_gpu_uuid": identity["gpu"]["uuid"],
            "identity_bootstrapped_at": captured_at_text,
            "identity_bootstrap_evidence": evidence_summary,
            "profile_state": PROFILE_STATE_ACTIVE,
            "profile_state_reason": "identity_binding_verified",
            "state_changed_at": captured_at_text,
            "activated_at": captured_at_text,
            "lifecycle_revision": int(context.metadata["lifecycle_revision"]) + 1,
            "updated_at": captured_at_text,
        }
    )
    updated_metadata["container_binding_sha256"] = compute_container_binding_sha256(
        updated_metadata
    )
    metadata_payload = json.dumps(
        updated_metadata,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    _commit_profile_binding_locked(
        context,
        known_hosts_payload=known_hosts_payload,
        metadata_payload=metadata_payload,
    )
    return {
        "status": "bound",
        "profile_state": PROFILE_STATE_ACTIVE,
        "credential_profile": context.profile,
        "allocation_binding_id": context.metadata["allocation_binding_id"],
        "allocation_generation": context.metadata["allocation_generation"],
        "profile_instance_id": context.metadata["profile_instance_id"],
        "container_binding_sha256": updated_metadata["container_binding_sha256"],
        "endpoint": {
            "host": context.config.host,
            "port": context.config.port,
        },
        "known_hosts_path": str(context.known_hosts_path),
        "ssh_host_key_fingerprint": key.fingerprint_sha256,
        "expected_host_uuid": identity["host_uuid"],
        "expected_gpu_uuid": identity["gpu"]["uuid"],
        "gpu_name": identity["gpu"]["name"],
        "gpu_memory_total_mib": identity["gpu"]["memory_total_mib"],
        "remote_workspace": ALLOWED_GPU_REMOTE_ROOT,
        "remote_workspace_writable": True,
        "reject_policy_verified": True,
        "probe_read_only": True,
        "identity_bootstrapped_at": captured_at_text,
        "evidence_sha256": evidence_digest,
        "secrets_emitted": False,
    }


def bootstrap_profile_identity(
    *,
    timeout: int = 20,
    now: datetime | None = None,
    context_loader: Callable[[], ProfileContext] = load_named_profile_context,
    host_key_collector: Callable[..., HostKeyEvidence] = collect_server_host_key,
    identity_inspector: Callable[..., dict[str, Any]] = inspect_remote_identity,
) -> dict[str, Any]:
    """Perform a lifecycle-linearized one-shot TOFU identity bootstrap."""

    _require(1 <= timeout <= 120, "SSH timeout must be between 1 and 120 seconds")
    context = context_loader()
    try:
        with profile_lifecycle_lock(context.profile_dir):
            _require_activation_context_unchanged_locked(context)
            return _bootstrap_profile_identity_locked(
                context=context,
                timeout=timeout,
                now=now,
                host_key_collector=host_key_collector,
                identity_inspector=identity_inspector,
            )
    except CredentialError as exc:
        raise BootstrapError("profile lifecycle lock failed during bootstrap") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = bootstrap_profile_identity(timeout=args.timeout)
    except (BootstrapError, CredentialError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 2
    except Exception:
        print(
            json.dumps(
                {"status": "error", "message": "unexpected identity bootstrap failure"},
                ensure_ascii=False,
            )
        )
        return 3
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
