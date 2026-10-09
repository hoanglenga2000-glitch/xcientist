from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _expired(value: str, now: datetime | None = None) -> bool:
    current = now or datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed <= current


class CredentialLeaseState(str, Enum):
    PENDING = "pending"
    LEASED = "leased"
    CONNECTOR_PREFLIGHTED = "connector_preflighted"
    DECRYPTED = "decrypted"
    CONSUMED = "consumed"
    REVOKED = "revoked"
    EXPIRED = "expired"


TERMINAL_STATES = {
    CredentialLeaseState.CONSUMED,
    CredentialLeaseState.REVOKED,
    CredentialLeaseState.EXPIRED,
}

ALLOWED_TRANSITIONS = {
    CredentialLeaseState.PENDING: {
        CredentialLeaseState.LEASED,
        CredentialLeaseState.REVOKED,
        CredentialLeaseState.EXPIRED,
    },
    CredentialLeaseState.LEASED: {
        CredentialLeaseState.CONNECTOR_PREFLIGHTED,
        CredentialLeaseState.REVOKED,
        CredentialLeaseState.EXPIRED,
    },
    CredentialLeaseState.CONNECTOR_PREFLIGHTED: {
        CredentialLeaseState.DECRYPTED,
        CredentialLeaseState.REVOKED,
        CredentialLeaseState.EXPIRED,
    },
    CredentialLeaseState.DECRYPTED: {
        CredentialLeaseState.CONSUMED,
        CredentialLeaseState.REVOKED,
    },
    CredentialLeaseState.CONSUMED: set(),
    CredentialLeaseState.REVOKED: set(),
    CredentialLeaseState.EXPIRED: set(),
}


@dataclass(frozen=True)
class PreflightEvidence:
    connector_version_sha256: str
    adapter_sha256: str
    destination_capability_id: str
    destination_write_verified: bool
    runtime_dependencies_verified: bool
    parser_verified: bool
    time_contract_verified: bool
    evidence_sha256: str

    @property
    def complete(self) -> bool:
        return all(
            (
                self.destination_write_verified,
                self.runtime_dependencies_verified,
                self.parser_verified,
                self.time_contract_verified,
            )
        ) and all(
            len(value) == 64
            for value in (
                self.connector_version_sha256,
                self.adapter_sha256,
                self.evidence_sha256,
            )
        )


@dataclass
class CredentialLease:
    lease_id: str
    purpose: str
    tenant_id: str
    project_id: str
    credential_ref_sha256: str
    expires_at_utc: str
    state: str = CredentialLeaseState.PENDING.value
    plaintext_exposed: bool = False
    preflight: dict[str, Any] = field(default_factory=dict)
    failure_signature: str = ""
    created_at_utc: str = field(default_factory=_now)
    updated_at_utc: str = field(default_factory=_now)

    @property
    def state_enum(self) -> CredentialLeaseState:
        return CredentialLeaseState(self.state)

    @property
    def safe_to_retry(self) -> bool:
        return (
            not self.plaintext_exposed
            and self.state_enum
            in {
                CredentialLeaseState.PENDING,
                CredentialLeaseState.LEASED,
                CredentialLeaseState.CONNECTOR_PREFLIGHTED,
            }
            and not _expired(self.expires_at_utc)
        )

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("credential_ref_sha256", None)
        value["safe_to_retry"] = self.safe_to_retry
        return value


class LeaseBackend(Protocol):
    def put(self, lease: CredentialLease) -> None: ...

    def get(self, lease_id: str) -> CredentialLease | None: ...


class JsonLeaseBackend:
    """Small atomic backend; it persists lease metadata and never secret values."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, lease_id: str) -> Path:
        if not lease_id.startswith("lease_") or not lease_id[6:].isalnum():
            raise ValueError("invalid lease id")
        return self.root / f"{lease_id}.json"

    def put(self, lease: CredentialLease) -> None:
        target = self._path(lease.lease_id)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(lease), sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(target)

    def get(self, lease_id: str) -> CredentialLease | None:
        target = self._path(lease_id)
        if not target.is_file() or target.is_symlink():
            return None
        return CredentialLease(**json.loads(target.read_text(encoding="utf-8")))


class CredentialLeaseManager:
    def __init__(self, backend: LeaseBackend) -> None:
        self.backend = backend
        self._lock = threading.RLock()

    def create(
        self,
        *,
        purpose: str,
        tenant_id: str,
        project_id: str,
        credential_ref: str,
        expires_at_utc: str,
    ) -> CredentialLease:
        if not all((purpose, tenant_id, project_id, credential_ref)):
            raise ValueError("credential lease identity is incomplete")
        if _expired(expires_at_utc):
            raise ValueError("credential lease is already expired")
        lease = CredentialLease(
            lease_id=f"lease_{uuid4().hex}",
            purpose=purpose,
            tenant_id=tenant_id,
            project_id=project_id,
            credential_ref_sha256=hashlib.sha256(credential_ref.encode("utf-8")).hexdigest(),
            expires_at_utc=expires_at_utc,
        )
        self.backend.put(lease)
        return lease

    def transition(
        self,
        lease_id: str,
        target: CredentialLeaseState,
        *,
        preflight: PreflightEvidence | None = None,
        failure_signature: str = "",
    ) -> CredentialLease:
        with self._lock:
            lease = self.backend.get(lease_id)
            if lease is None:
                raise KeyError(lease_id)
            current = lease.state_enum
            if current in TERMINAL_STATES or target not in ALLOWED_TRANSITIONS[current]:
                raise ValueError(f"invalid credential lease transition: {current.value}->{target.value}")
            if _expired(lease.expires_at_utc) and target is not CredentialLeaseState.EXPIRED:
                raise ValueError("expired credential lease must enter expired state")
            if target is CredentialLeaseState.CONNECTOR_PREFLIGHTED:
                if preflight is None or not preflight.complete:
                    raise ValueError("connector preflight evidence is incomplete")
                lease.preflight = asdict(preflight)
            if target is CredentialLeaseState.DECRYPTED:
                if current is not CredentialLeaseState.CONNECTOR_PREFLIGHTED or not lease.preflight:
                    raise ValueError("decryption requires verified connector preflight")
                lease.plaintext_exposed = True
            lease.state = target.value
            lease.failure_signature = str(failure_signature)[:256]
            lease.updated_at_utc = _now()
            self.backend.put(lease)
            return lease

    def fail(self, lease_id: str, *, failure_signature: str) -> CredentialLease:
        """Keep a proven pre-decryption lease retryable; revoke anything ambiguous."""
        with self._lock:
            lease = self.backend.get(lease_id)
            if lease is None:
                raise KeyError(lease_id)
            if lease.safe_to_retry:
                lease.failure_signature = str(failure_signature)[:256]
                lease.updated_at_utc = _now()
                self.backend.put(lease)
                return lease
            if lease.state_enum not in TERMINAL_STATES:
                return self.transition(
                    lease_id,
                    CredentialLeaseState.REVOKED,
                    failure_signature=failure_signature,
                )
            return lease


__all__ = [
    "CredentialLease",
    "CredentialLeaseManager",
    "CredentialLeaseState",
    "JsonLeaseBackend",
    "PreflightEvidence",
]
