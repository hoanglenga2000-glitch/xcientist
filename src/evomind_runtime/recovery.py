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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def canonical_json(value: Any) -> bytes:
    """Return the stable representation used for signatures and receipts."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def precondition_fingerprint(preconditions: Mapping[str, Any] | None) -> str:
    """Bind a retry decision to the facts that were true before execution."""

    return canonical_sha256(dict(preconditions or {}))


def failure_signature(*, stage: str, error_class: str, sanitized_error: str = "") -> str:
    """Create a stable failure signature without storing raw exception output."""

    normalized_error = " ".join(str(sanitized_error).strip().split())[:512]
    return canonical_sha256(
        {
            "stage": str(stage).strip().lower(),
            "error_class": str(error_class).strip(),
            "sanitized_error": normalized_error,
        }
    )


class RecoveryDecision(str, Enum):
    RETRY = "retry"
    REPAIR = "repair"
    EXACT_GATE = "exact_gate"


class RetrySuppressed(RuntimeError):
    """Raised when an unchanged failure would be blindly replayed."""

    def __init__(self, signature: str, preconditions: str) -> None:
        super().__init__("same failure signature and unchanged preconditions cannot be retried")
        self.signature = signature
        self.preconditions = preconditions


@dataclass(frozen=True)
class ExactGate:
    run_id: str
    node_id: str
    reason: str
    required_action: str
    resume_point: str
    failure_signature: str = ""
    precondition_fingerprint: str = ""
    id: str = field(default_factory=lambda: f"gate_{uuid4().hex}")
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("exact gate requires one concrete reason")
        if not self.required_action.strip():
            raise ValueError("exact gate requires one concrete human action")
        if "\n" in self.required_action.strip():
            raise ValueError("exact gate required_action must be a single action")
        if not self.resume_point.strip():
            raise ValueError("exact gate requires a durable resume point")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExactGate":
        return cls(**dict(payload))


@dataclass(frozen=True)
class FailureObservation:
    node_id: str
    signature: str
    precondition_fingerprint: str
    attempt_id: str
    observed_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FailureObservation":
        return cls(**dict(payload))


class FailureLedger:
    """Tracks proven failures and enforces changed-precondition retries."""

    def __init__(self, observations: list[FailureObservation] | None = None) -> None:
        self._observations = list(observations or [])

    @property
    def observations(self) -> tuple[FailureObservation, ...]:
        return tuple(self._observations)

    def record(
        self,
        *,
        node_id: str,
        signature: str,
        preconditions: str,
        attempt_id: str,
    ) -> FailureObservation:
        item = FailureObservation(node_id, signature, preconditions, attempt_id)
        self._observations.append(item)
        return item

    def has_seen(self, *, node_id: str, signature: str, preconditions: str) -> bool:
        return any(
            item.node_id == node_id
            and item.signature == signature
            and item.precondition_fingerprint == preconditions
            for item in self._observations
        )

    def assert_retry_allowed(self, *, node_id: str, signature: str, preconditions: str) -> None:
        if self.has_seen(node_id=node_id, signature=signature, preconditions=preconditions):
            raise RetrySuppressed(signature, preconditions)

    def to_list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._observations]

    @classmethod
    def from_list(cls, rows: list[Mapping[str, Any]]) -> "FailureLedger":
        return cls([FailureObservation.from_dict(row) for row in rows])


class RecoveryPolicy:
    """Make one evidence-bound recovery decision; never guess at a retry."""

    def decide(
        self,
        *,
        ledger: FailureLedger,
        node_id: str,
        signature: str,
        preconditions: str,
        retryable: bool,
        repair_available: bool,
    ) -> RecoveryDecision:
        if not retryable:
            return RecoveryDecision.EXACT_GATE
        if ledger.has_seen(node_id=node_id, signature=signature, preconditions=preconditions):
            return RecoveryDecision.REPAIR if repair_available else RecoveryDecision.EXACT_GATE
        return RecoveryDecision.REPAIR if repair_available else RecoveryDecision.RETRY


class RunStateRepository(Protocol):
    """Persistence boundary used by RunSupervisor.

    Backends store sanitized snapshots only. They must replace one run snapshot
    atomically and must not reinterpret or execute snapshot data.
    """

    def save(self, run_id: str, snapshot: Mapping[str, Any]) -> None: ...

    def load(self, run_id: str) -> dict[str, Any] | None: ...

    def list_run_ids(self) -> list[str]: ...


class MemoryRunStateRepository:
    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def save(self, run_id: str, snapshot: Mapping[str, Any]) -> None:
        with self._lock:
            self._rows[run_id] = json.loads(canonical_json(dict(snapshot)).decode("utf-8"))

    def load(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._rows.get(run_id)
            return json.loads(canonical_json(row).decode("utf-8")) if row is not None else None

    def list_run_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._rows)


class JsonRunStateRepository:
    """Small, dependency-free atomic JSON backend for shadow-mode V2 runs."""

    _SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, run_id: str) -> Path:
        if not self._SAFE_ID.fullmatch(str(run_id)):
            raise ValueError("run_id is not safe for persistent storage")
        return self.root / f"{run_id}.json"

    def save(self, run_id: str, snapshot: Mapping[str, Any]) -> None:
        target = self._path(run_id)
        payload = json.dumps(dict(snapshot), ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n"
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        with self._lock:
            try:
                with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(target)
            finally:
                if temporary.exists():
                    temporary.unlink()

    def load(self, run_id: str) -> dict[str, Any] | None:
        target = self._path(run_id)
        with self._lock:
            if not target.is_file() or target.is_symlink():
                return None
            value = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("run snapshot must be a JSON object")
        return value

    def list_run_ids(self) -> list[str]:
        with self._lock:
            return sorted(
                path.stem
                for path in self.root.glob("*.json")
                if path.is_file() and not path.is_symlink() and self._SAFE_ID.fullmatch(path.stem)
            )
