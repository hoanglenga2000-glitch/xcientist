"""Capability contracts shared by the EvoMind super-agent runtime.

The objects in this module are deliberately transport friendly.  Opaque
connector roots and read payloads are never included in serialized receipts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any, Iterable
from uuid import uuid4


DIRECTORY_OPERATIONS = frozenset(
    {"health", "list", "stat", "read", "hash", "write", "mkdir", "copy", "sync", "delete", "execute"}
)


class CapabilityError(RuntimeError):
    """Base class for capability contract failures."""


class CapabilityNotFoundError(CapabilityError):
    """A requested connector or directory capability is not registered."""


class CapabilityPermissionError(CapabilityError):
    """An operation is outside a capability's granted operation set."""


class CapabilityExpiredError(CapabilityPermissionError):
    """A time-bounded capability is no longer valid."""


class DirectoryBoundaryError(CapabilityPermissionError):
    """A requested path would cross the mounted directory boundary."""


class DirectoryQuotaError(CapabilityPermissionError):
    """A requested mutation would exceed a mounted directory quota."""


class ApprovalRequiredError(CapabilityPermissionError):
    """A destructive operation lacks its explicit approval bit."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _non_empty(value: str, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{label} must be non-empty")
    return normalized


def _operations(values: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).strip().lower() for value in values if str(value).strip()))
    unknown = sorted(set(normalized) - DIRECTORY_OPERATIONS)
    if unknown:
        raise ValueError(f"unsupported directory operations: {', '.join(unknown)}")
    if not normalized:
        raise ValueError("operations must contain at least one operation")
    return normalized


def _parse_utc(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("expires_at must include an explicit timezone")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class CapabilityDescriptor:
    """A discoverable tool or connector capability."""

    capability_id: str
    provider_id: str
    version: str
    operations: tuple[str, ...]
    input_schema: dict[str, Any] = field(default_factory=dict)
    risk_class: str = "bounded"
    idempotency: str = "idempotent"
    health_evidence: dict[str, Any] = field(default_factory=dict)
    tool_package_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability_id", _non_empty(self.capability_id, "capability_id"))
        object.__setattr__(self, "provider_id", _non_empty(self.provider_id, "provider_id"))
        object.__setattr__(self, "version", _non_empty(self.version, "version"))
        object.__setattr__(self, "operations", tuple(dict.fromkeys(str(value) for value in self.operations)))
        if not self.operations:
            raise ValueError("operations must contain at least one operation")
        if self.tool_package_sha256 and (
            len(self.tool_package_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in self.tool_package_sha256)
        ):
            raise ValueError("tool_package_sha256 must be a hexadecimal SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["operations"] = list(self.operations)
        return value

    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DirectoryCapability:
    """A revocable mount grant addressed by ``directory_id`` only.

    ``opaque_root_ref`` is connector-private.  It must never be surfaced in an
    operation receipt or model-visible path.
    """

    directory_id: str
    connector_id: str
    opaque_root_ref: str
    operations: tuple[str, ...]
    tenant_id: str = "default"
    project_id: str = "default"
    run_id: str = ""
    connector_version: str = ""
    max_files: int = 1_000_000
    max_bytes: int = 1 << 40
    symlink_or_reparse_policy: str = "deny"
    expires_at: str = ""
    credential_lease_ref: str = ""
    approval_fingerprint: str = ""
    health_receipt: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory_id", _non_empty(self.directory_id, "directory_id"))
        object.__setattr__(self, "connector_id", _non_empty(self.connector_id, "connector_id"))
        object.__setattr__(self, "opaque_root_ref", _non_empty(self.opaque_root_ref, "opaque_root_ref"))
        object.__setattr__(self, "operations", _operations(self.operations))
        if self.max_files < 0 or self.max_bytes < 0:
            raise ValueError("directory quotas must be non-negative")
        if self.symlink_or_reparse_policy != "deny":
            raise ValueError("V1 supports only fail-closed symlink_or_reparse_policy='deny'")
        if self.expires_at:
            _parse_utc(self.expires_at)

    def allows(self, operation: str) -> bool:
        requested = str(operation).strip().lower()
        return requested in self.operations or requested == "health"

    def is_expired(self, *, now: datetime | None = None) -> bool:
        if not self.expires_at:
            return False
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return current.astimezone(timezone.utc) >= _parse_utc(self.expires_at)

    def to_dict(self, *, include_opaque_root: bool = False) -> dict[str, Any]:
        value = asdict(self)
        value["operations"] = list(self.operations)
        if not include_opaque_root:
            value.pop("opaque_root_ref", None)
        return value


@dataclass
class OperationReceipt:
    """Structured evidence emitted for every connector operation.

    Read bytes are available to the in-process caller through ``data`` but are
    intentionally omitted from :meth:`to_dict`.
    """

    operation: str
    connector_id: str
    directory_id: str
    relative_path: str = ""
    ok: bool = True
    status: str = ""
    exit_code: int | None = None
    receipt_id: str = field(default_factory=lambda: f"op_{uuid4().hex}")
    started_at: str = field(default_factory=utc_now)
    completed_at: str = field(default_factory=utc_now)
    source_directory_id: str = ""
    source_relative_path: str = ""
    destination_directory_id: str = ""
    destination_relative_path: str = ""
    bytes_read: int = 0
    bytes_written: int = 0
    sha256: str = ""
    entries: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    error_class: str = ""
    error_message: str = ""
    data: bytes | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.status:
            self.status = "completed" if self.ok else "failed"
        if self.exit_code is None and self.ok:
            self.exit_code = 0

    @property
    def content(self) -> bytes | None:
        """Compatibility alias for callers that describe read bytes as content."""

        return self.data

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("data", None)
        value["data_present"] = self.data is not None
        value["data_bytes"] = len(self.data) if self.data is not None else 0
        return value
