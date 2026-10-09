from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


class EvidenceStatus(str, Enum):
    OBSERVED = "observed"
    PROVEN = "proven"
    REJECTED = "rejected"


class EvidenceScope(str, Enum):
    TENANT_PROJECT = "tenant_project"
    ANONYMOUS_GLOBAL = "anonymous_global"


_SENSITIVE_KEY = re.compile(
    r"(?:password|passwd|pwd|token|secret|credential|api[_-]?key|authorization|bearer|cookie|密码|口令|凭据|密钥)",
    re.IGNORECASE,
)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
_ABSOLUTE_PATH = re.compile(r"(?:\b[A-Za-z]:[\\/]|/(?:home|Users|var|srv|mnt|opt)/)")
_INLINE_SECRET = re.compile(
    r"(?:password|passwd|pwd|token|secret|api[_ -]?key|authorization|bearer|密码|口令|凭据|密钥)\s*[:=]\s*\S+",
    re.IGNORECASE,
)


class SensitiveEvidenceError(ValueError):
    pass


def _sanitize(value: Any, *, global_scope: bool = False) -> Any:
    """Return a persistable value while failing closed on raw secret material."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if _SENSITIVE_KEY.search(key):
                result[key] = "[REDACTED]"
            else:
                result[key] = _sanitize(raw_value, global_scope=global_scope)
        return result
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(item, global_scope=global_scope) for item in value]
    if isinstance(value, str):
        if _JWT.search(value) or _PRIVATE_KEY.search(value) or _INLINE_SECRET.search(value):
            raise SensitiveEvidenceError("evidence contains forbidden secret material")
        if global_scope and (_ABSOLUTE_PATH.search(value) or value.startswith("[REDACTED_PATH:")):
            return "[ANONYMIZED_PATH]"
        if _ABSOLUTE_PATH.search(value):
            return f"[REDACTED_PATH:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}]"
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


@dataclass(frozen=True)
class EvidenceRef:
    sha256: str
    kind: str = "artifact"
    opaque_ref: str = ""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-fA-F0-9]{64}", self.sha256):
            raise ValueError("evidence reference requires a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceRef":
        return cls(**dict(payload))


@dataclass(frozen=True)
class EvidenceRecord:
    claim: str
    status: str
    scope: str
    payload: dict[str, Any]
    source_evidence: tuple[EvidenceRef, ...]
    tenant_id: str = ""
    project_id: str = ""
    category: str = "general"
    failure_signature: str = ""
    anonymous: bool = False
    id: str = field(default_factory=lambda: f"memory_{uuid4().hex}")
    created_at: str = field(default_factory=_utc_now)
    content_sha256: str = ""

    def __post_init__(self) -> None:
        status = EvidenceStatus(self.status)
        scope = EvidenceScope(self.scope)
        global_scope = scope is EvidenceScope.ANONYMOUS_GLOBAL
        sanitized_claim = _sanitize(self.claim, global_scope=global_scope)
        sanitized_payload = _sanitize(self.payload, global_scope=global_scope)
        sanitized_refs = tuple(
            EvidenceRef(
                item.sha256,
                kind=item.kind,
                opaque_ref=_sanitize(item.opaque_ref, global_scope=global_scope),
            )
            for item in self.source_evidence
        )
        object.__setattr__(self, "claim", sanitized_claim)
        object.__setattr__(self, "payload", sanitized_payload)
        object.__setattr__(self, "source_evidence", sanitized_refs)
        if not sanitized_claim.strip():
            raise ValueError("evidence record requires a claim")
        if scope is EvidenceScope.TENANT_PROJECT and (not self.tenant_id or not self.project_id):
            raise ValueError("tenant/project evidence requires tenant_id and project_id")
        if scope is EvidenceScope.ANONYMOUS_GLOBAL:
            if self.tenant_id or self.project_id or not self.anonymous:
                raise ValueError("global evidence must be anonymous and tenant independent")
            if status is not EvidenceStatus.PROVEN:
                raise ValueError("only proven evidence may enter anonymous global memory")
            if not self.source_evidence:
                raise ValueError("global evidence requires hash-bound source evidence")
        expected = _sha256(
            {
                "claim": sanitized_claim,
                "status": self.status,
                "scope": self.scope,
                "payload": sanitized_payload,
                "source_evidence": [item.to_dict() for item in sanitized_refs],
                "category": self.category,
                "failure_signature": self.failure_signature,
            }
        )
        if self.content_sha256 and self.content_sha256 != expected:
            raise ValueError("evidence content hash does not match record")
        object.__setattr__(self, "content_sha256", expected)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_evidence"] = [item.to_dict() for item in self.source_evidence]
        return value

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceRecord":
        value = dict(payload)
        value["source_evidence"] = tuple(EvidenceRef.from_dict(item) for item in value.get("source_evidence", []))
        return cls(**value)


class EvidenceBackend(Protocol):
    def put(self, record: EvidenceRecord) -> EvidenceRecord: ...

    def list(self) -> list[EvidenceRecord]: ...


class MemoryEvidenceBackend:
    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}
        self._lock = threading.RLock()

    def put(self, record: EvidenceRecord) -> EvidenceRecord:
        with self._lock:
            current = self._records.get(record.id)
            if current is not None and current.content_sha256 != record.content_sha256:
                raise ValueError("evidence record id is immutable")
            self._records[record.id] = record
        return record

    def list(self) -> list[EvidenceRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda item: (item.created_at, item.id))


class JsonEvidenceBackend:
    """Append-by-identity JSON backend with atomic whole-index replacement."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _read(self) -> list[EvidenceRecord]:
        if not self.path.exists():
            return []
        if not self.path.is_file() or self.path.is_symlink():
            raise ValueError("evidence backend must be a regular file")
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema") != "evomind.evidence_memory.v1":
            raise ValueError("unsupported evidence memory schema")
        rows = value.get("records")
        if not isinstance(rows, list):
            raise ValueError("evidence memory records must be a list")
        return [EvidenceRecord.from_dict(item) for item in rows]

    def _write(self, records: list[EvidenceRecord]) -> None:
        payload = {
            "schema": "evomind.evidence_memory.v1",
            "records": [item.to_dict() for item in records],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def put(self, record: EvidenceRecord) -> EvidenceRecord:
        with self._lock:
            records = self._read()
            current = next((item for item in records if item.id == record.id), None)
            if current is not None:
                if current.content_sha256 != record.content_sha256:
                    raise ValueError("evidence record id is immutable")
                return current
            records.append(record)
            self._write(records)
        return record

    def list(self) -> list[EvidenceRecord]:
        with self._lock:
            return sorted(self._read(), key=lambda item: (item.created_at, item.id))


class EvidenceMemory:
    def __init__(self, backend: EvidenceBackend | None = None) -> None:
        self.backend = backend or MemoryEvidenceBackend()

    def remember(
        self,
        *,
        claim: str,
        status: str,
        payload: Mapping[str, Any] | None = None,
        source_evidence: list[EvidenceRef | Mapping[str, Any]] | None = None,
        tenant_id: str = "",
        project_id: str = "",
        category: str = "general",
        failure_signature: str = "",
        scope: str = EvidenceScope.TENANT_PROJECT.value,
        record_id: str = "",
    ) -> EvidenceRecord:
        global_scope = EvidenceScope(scope) is EvidenceScope.ANONYMOUS_GLOBAL
        sanitized_claim = _sanitize(str(claim), global_scope=global_scope)
        sanitized_payload = _sanitize(dict(payload or {}), global_scope=global_scope)
        refs = tuple(
            EvidenceRef(
                item.sha256,
                kind=item.kind,
                opaque_ref=_sanitize(item.opaque_ref, global_scope=global_scope),
            )
            if isinstance(item, EvidenceRef)
            else EvidenceRef.from_dict(
                {
                    **dict(item),
                    "opaque_ref": _sanitize(str(item.get("opaque_ref") or ""), global_scope=global_scope),
                }
            )
            for item in (source_evidence or [])
        )
        record = EvidenceRecord(
            id=record_id or f"memory_{uuid4().hex}",
            claim=sanitized_claim,
            status=EvidenceStatus(status).value,
            scope=EvidenceScope(scope).value,
            payload=sanitized_payload,
            source_evidence=refs,
            tenant_id="" if global_scope else tenant_id,
            project_id="" if global_scope else project_id,
            category=category,
            failure_signature=failure_signature,
            anonymous=global_scope,
        )
        return self.backend.put(record)

    def query(
        self,
        *,
        tenant_id: str = "",
        project_id: str = "",
        category: str = "",
        status: str = "",
        failure_signature: str = "",
        include_global: bool = True,
    ) -> list[EvidenceRecord]:
        if status:
            EvidenceStatus(status)
        result: list[EvidenceRecord] = []
        for item in self.backend.list():
            if item.scope == EvidenceScope.TENANT_PROJECT.value:
                if item.tenant_id != tenant_id or item.project_id != project_id:
                    continue
            elif not include_global:
                continue
            if category and item.category != category:
                continue
            if status and item.status != status:
                continue
            if failure_signature and item.failure_signature != failure_signature:
                continue
            result.append(item)
        return result

    def promote_anonymous_global(self, record_id: str) -> EvidenceRecord:
        source = next((item for item in self.backend.list() if item.id == record_id), None)
        if source is None:
            raise KeyError(record_id)
        if source.status != EvidenceStatus.PROVEN.value:
            raise ValueError("only proven project evidence may be promoted")
        return self.remember(
            claim=source.claim,
            status=source.status,
            payload=source.payload,
            source_evidence=list(source.source_evidence),
            category=source.category,
            failure_signature=source.failure_signature,
            scope=EvidenceScope.ANONYMOUS_GLOBAL.value,
        )
