"""Run-scoped, metadata-only lifecycle service for synthesized tool packages.

The service composes :class:`ecosystem.ToolPackage` with
:class:`super_agent_store.SuperAgentStore`.  It intentionally does not install,
import, execute, or probe package-provided code.  Promotion means only that the
package's declared capability metadata is eligible for the owning run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sqlite3
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .ecosystem import (
    ExternalExecutionDisabled,
    ManifestError,
    PackageCheckStatus,
    PackagePromotionStatus,
    ToolPackage,
)
from .super_agent_store import SuperAgentStore


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_LOW_RISK_CLASSES = frozenset({"observe", "read", "low", "bounded", "write", "medium"})
_APPROVAL_OPERATION_FRAGMENTS = (
    "delete",
    "remove",
    "purge",
    "destroy",
    "unlink",
    "truncate",
    "drop",
    "elevat",
    "privileg",
    "sudo",
    "submit",
    "payment",
    "purchase",
    "accept-terms",
    "accept_terms",
    "join-competition",
    "join_competition",
)
_TERMINAL_STATES = frozenset(
    {PackagePromotionStatus.PROMOTED.value, PackagePromotionStatus.REJECTED.value}
)


class ToolPackageStateError(ManifestError):
    """Raised when a lifecycle transition would violate persisted state."""


class ToolPackageApprovalRequired(ToolPackageStateError):
    """Raised when promotion needs an exact, externally obtained approval."""


def _valid_sha256(value: Any) -> str:
    digest = str(value or "").strip().casefold()
    if not _SHA256_RE.fullmatch(digest):
        raise ToolPackageStateError("evidence must include a valid SHA-256 digest")
    return digest


def _storage_key(run_id: str, package_id: str) -> str:
    raw = json.dumps([run_id, package_id], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "runpkg_" + hashlib.sha256(raw).hexdigest()


def _evidence_digest(evidence: Mapping[str, Any], explicit: str | None) -> str:
    if explicit is not None:
        return _valid_sha256(explicit)
    candidates = {
        str(value).casefold()
        for key, value in evidence.items()
        if str(key).casefold().endswith("sha256") and _SHA256_RE.fullmatch(str(value or ""))
    }
    if len(candidates) != 1:
        raise ToolPackageStateError("exactly one evidence SHA-256 is required")
    return candidates.pop()


def _package_reversible(package: ToolPackage) -> bool:
    rollback = package.manifest.get("rollback")
    permissions = package.manifest.get("permissions")
    if not isinstance(rollback, Mapping) or not rollback:
        return False
    if rollback.get("reversible") is False or rollback.get("irreversible") is True:
        return False
    if isinstance(permissions, Mapping) and permissions.get("irreversible") is True:
        return False
    meaningful = [
        value
        for key, value in rollback.items()
        if key not in {"reversible", "irreversible"} and value not in (None, "", False, [], {})
    ]
    if not meaningful:
        return False
    action = str(rollback.get("action") or "").strip().casefold()
    if action in {"none", "noop", "no-op", "not-applicable", "irreversible"}:
        return False
    return True


def _package_requires_approval(package: ToolPackage) -> tuple[bool, tuple[str, ...]]:
    reasons: set[str] = set()
    if not _package_reversible(package):
        reasons.add("irreversible")
    for descriptor in package.descriptors:
        risk = descriptor.risk_class.casefold()
        if risk not in _LOW_RISK_CLASSES:
            reasons.add(f"risk:{risk}")
        for operation in descriptor.operations:
            normalized = operation.casefold()
            if any(fragment in normalized for fragment in _APPROVAL_OPERATION_FRAGMENTS):
                reasons.add("sensitive-operation")
    return bool(reasons), tuple(sorted(reasons))


class ToolPackageRuntime:
    """Persist and gate one run's declarative tool-package lifecycle.

    A runtime instance has exactly one ``run_id``.  Package rows use a hashed
    composite storage key because the additive V1 store predates run-scoped
    package identities and has a single-column primary key.
    """

    def __init__(self, store: SuperAgentStore, *, run_id: str) -> None:
        normalized_run_id = str(run_id or "").strip()
        if not normalized_run_id or len(normalized_run_id) > 128:
            raise ValueError("run_id must contain between 1 and 128 characters")
        self.store = store
        self.run_id = normalized_run_id
        self._lock = threading.RLock()
        self._packages: dict[str, ToolPackage] = {}
        self._records: dict[str, dict[str, Any]] = {}

    def stage(
        self,
        manifest: ToolPackage | Mapping[str, Any] | str | bytes | Path,
    ) -> dict[str, Any]:
        """Validate and persist a declarative package without executing it."""

        source = manifest.manifest if isinstance(manifest, ToolPackage) else manifest
        package = ToolPackage.from_manifest(source, run_id=self.run_id)
        with self._lock:
            existing = self._load(package.package_id)
            if existing is not None:
                if existing.manifest_sha256 != package.manifest_sha256:
                    raise ToolPackageStateError(
                        "package_id is already staged for this run with a different manifest"
                    )
                return self._snapshot(existing)
            self._packages[package.package_id] = package
            self._persist(package)
            return self._snapshot(package)

    def get(self, package_id: str) -> dict[str, Any] | None:
        with self._lock:
            package = self._load(package_id)
            return None if package is None else self._snapshot(package)

    def record_tests(
        self,
        package_id: str,
        *,
        passed: bool,
        evidence: Mapping[str, Any] | None = None,
        evidence_sha256: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            package = self._require_mutable(package_id)
            normalized_evidence = self._evidence(evidence, evidence_sha256)
            package.record_tests(passed=bool(passed), evidence=normalized_evidence)
            self._persist(package)
            return self._snapshot(package)

    def record_canary(
        self,
        package_id: str,
        *,
        passed: bool,
        evidence: Mapping[str, Any] | None = None,
        evidence_sha256: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            package = self._require_mutable(package_id)
            normalized_evidence = self._evidence(evidence, evidence_sha256)
            package.record_canary(passed=bool(passed), evidence=normalized_evidence)
            self._persist(package)
            return self._snapshot(package)

    def promote(
        self,
        package_id: str,
        *,
        approval_evidence_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Promote metadata after verification and any required exact approval."""

        with self._lock:
            package = self._require_mutable(package_id)
            if package.test_status != PackageCheckStatus.PASSED.value:
                package.promotion_status = PackagePromotionStatus.BLOCKED.value
                self._persist(package)
                raise ToolPackageStateError("promotion requires passing declared tests")
            if package.canary_status != PackageCheckStatus.PASSED.value:
                package.promotion_status = PackagePromotionStatus.BLOCKED.value
                self._persist(package)
                raise ToolPackageStateError("promotion requires a passing canary")

            approval_required, reasons = _package_requires_approval(package)
            approval_digest = ""
            if approval_required:
                if approval_evidence_sha256 is None:
                    self._persist(
                        package,
                        gate={
                            "status": "waiting_exact_gate",
                            "approval_required": True,
                            "reasons": list(reasons),
                        },
                    )
                    raise ToolPackageApprovalRequired(
                        "exact approval evidence is required for this package risk boundary"
                    )
                approval_digest = _valid_sha256(approval_evidence_sha256)

            package.promote()
            self._persist(
                package,
                gate={
                    "status": "satisfied" if approval_required else "not_required",
                    "approval_required": approval_required,
                    "reasons": list(reasons),
                    "approval_evidence_sha256": approval_digest,
                },
            )
            return self._snapshot(package)

    def reject(self, package_id: str) -> dict[str, Any]:
        with self._lock:
            package = self._require_mutable(package_id)
            package.reject()
            self._persist(package)
            return self._snapshot(package)

    def install(self, *_args: Any, **_kwargs: Any) -> None:
        raise ExternalExecutionDisabled("tool package installation is disabled")

    def execute(self, *_args: Any, **_kwargs: Any) -> None:
        raise ExternalExecutionDisabled("tool package execution is disabled")

    def _require_mutable(self, package_id: str) -> ToolPackage:
        package = self._load(package_id)
        if package is None:
            raise ToolPackageStateError("unknown tool package for this run")
        if package.promotion_status in _TERMINAL_STATES:
            raise ToolPackageStateError(
                f"{package.promotion_status} tool package state is immutable"
            )
        return package

    @staticmethod
    def _evidence(
        evidence: Mapping[str, Any] | None,
        explicit_digest: str | None,
    ) -> dict[str, Any]:
        if evidence is not None and not isinstance(evidence, Mapping):
            raise ToolPackageStateError("verification evidence must be a mapping")
        source = evidence or {}
        digest = _evidence_digest(source, explicit_digest)
        normalized = copy.deepcopy(dict(source))
        normalized["evidence_sha256"] = digest
        return normalized

    def _persist(
        self,
        package: ToolPackage,
        *,
        gate: Mapping[str, Any] | None = None,
    ) -> None:
        previous = self._records.get(package.package_id, {})
        approval_gate = copy.deepcopy(dict(gate)) if gate is not None else copy.deepcopy(
            previous.get("approval_gate", {"status": "not_evaluated", "approval_required": False})
        )
        approval_required, reasons = _package_requires_approval(package)
        record = {
            "schema": "evomind.tool_package_runtime.v1",
            # SuperAgentStore's physical key is composite and opaque; the logical
            # identity remains separately available inside the payload.
            "package_id": _storage_key(self.run_id, package.package_id),
            "logical_package_id": package.package_id,
            "run_id": self.run_id,
            "version": package.version,
            "status": package.promotion_status,
            "package_sha256": package.manifest_sha256,
            "package": package.to_dict(include_manifest=True),
            "reversible": _package_reversible(package),
            "approval_required": approval_required,
            "approval_reasons": list(reasons),
            "approval_gate": approval_gate,
            "external_install_enabled": False,
            "external_execution_enabled": False,
        }
        self.store.put_tool_package(record)
        self._records[package.package_id] = record

    def _load(self, package_id: str) -> ToolPackage | None:
        logical_id = str(package_id or "")
        if logical_id in self._packages:
            return self._packages[logical_id]
        record = self._read_record(_storage_key(self.run_id, logical_id))
        if record is None:
            return None
        if record.get("run_id") != self.run_id or record.get("logical_package_id") != logical_id:
            raise ToolPackageStateError("persisted tool package scope mismatch")
        package_payload = record.get("package")
        if not isinstance(package_payload, Mapping) or not isinstance(package_payload.get("manifest"), Mapping):
            raise ToolPackageStateError("persisted tool package payload is invalid")
        package = ToolPackage.from_manifest(package_payload["manifest"], run_id=self.run_id)
        if package.manifest_sha256 != record.get("package_sha256"):
            raise ToolPackageStateError("persisted tool package manifest hash mismatch")

        test_status = str(package_payload.get("test_status") or PackageCheckStatus.PENDING.value)
        canary_status = str(package_payload.get("canary_status") or PackageCheckStatus.PENDING.value)
        promotion_status = str(package_payload.get("promotion_status") or PackagePromotionStatus.DRAFT.value)
        PackageCheckStatus(test_status)
        PackageCheckStatus(canary_status)
        PackagePromotionStatus(promotion_status)
        package.test_status = test_status
        package.canary_status = canary_status
        package.promotion_status = promotion_status
        package.test_evidence = copy.deepcopy(dict(package_payload.get("test_evidence") or {}))
        package.canary_evidence = copy.deepcopy(dict(package_payload.get("canary_evidence") or {}))
        self._packages[logical_id] = package
        self._records[logical_id] = copy.deepcopy(record)
        return package

    def _read_record(self, storage_id: str) -> dict[str, Any] | None:
        # The additive store currently exposes writes but no package getter.  A
        # separate read-only SQLite connection preserves its encapsulated writer
        # transaction and can be removed once a public getter exists.
        connection = sqlite3.connect(str(self.store.path), timeout=30)
        try:
            row = connection.execute(
                "SELECT payload_json FROM tool_packages_v2 WHERE package_id=?",
                (storage_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        try:
            payload = json.loads(row[0])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ToolPackageStateError("persisted tool package JSON is invalid") from exc
        if not isinstance(payload, dict):
            raise ToolPackageStateError("persisted tool package root is invalid")
        return payload

    def _snapshot(self, package: ToolPackage) -> dict[str, Any]:
        state = package.to_dict(include_manifest=False)
        record = self._records.get(package.package_id, {})
        state.update(
            {
                "reversible": bool(record.get("reversible", _package_reversible(package))),
                "approval_required": bool(record.get("approval_required", False)),
                "approval_reasons": copy.deepcopy(record.get("approval_reasons", [])),
                "approval_gate": copy.deepcopy(record.get("approval_gate", {})),
                "external_install_enabled": False,
                "external_execution_enabled": False,
            }
        )
        return state


__all__ = [
    "RunScopedToolPackageService",
    "ToolPackageApprovalRequired",
    "ToolPackageRuntime",
    "ToolPackageStateError",
]


RunScopedToolPackageService = ToolPackageRuntime
