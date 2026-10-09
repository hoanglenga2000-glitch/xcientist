"""Capability-mediated directory broker.

The broker is the only model-facing path router: callers select an authorized
``directory_id`` and a relative path, never a connector root.
"""

from __future__ import annotations

from hashlib import sha256
import threading
from typing import Iterable

from .capabilities import (
    ApprovalRequiredError,
    CapabilityDescriptor,
    CapabilityExpiredError,
    CapabilityNotFoundError,
    CapabilityPermissionError,
    DirectoryCapability,
    OperationReceipt,
)
from .connectors import ConnectorContract


class DirectoryBroker:
    """Register connectors, mount bounded capabilities, and route operations."""

    def __init__(self, connectors: Iterable[ConnectorContract] | None = None) -> None:
        self._connectors: dict[str, ConnectorContract] = {}
        self._directories: dict[str, DirectoryCapability] = {}
        self._lock = threading.RLock()
        for connector in connectors or ():
            self.register_connector(connector)

    def register_connector(self, connector: ConnectorContract, *, replace: bool = False) -> None:
        if not isinstance(connector, ConnectorContract):
            raise TypeError("connector must implement ConnectorContract")
        with self._lock:
            if connector.connector_id in self._connectors and not replace:
                raise ValueError(f"connector already registered: {connector.connector_id}")
            self._connectors[connector.connector_id] = connector

    def mount(self, capability: DirectoryCapability, *, replace: bool = False, require_healthy: bool = True) -> OperationReceipt:
        connector = self._connectors.get(capability.connector_id)
        if connector is None:
            raise CapabilityNotFoundError(f"unknown connector: {capability.connector_id}")
        unsupported = sorted(set(capability.operations) - set(connector.supported_operations))
        if unsupported:
            raise ValueError(f"connector does not support: {', '.join(unsupported)}")
        if capability.is_expired():
            raise CapabilityExpiredError(f"directory capability is expired: {capability.directory_id}")
        if capability.connector_version and capability.connector_version != connector.version:
            raise ValueError("directory capability connector version mismatch")
        receipt = connector.health(capability)
        if require_healthy and not receipt.ok:
            raise CapabilityNotFoundError(
                f"connector health failed for {capability.directory_id}: {receipt.error_class or 'unavailable'}"
            )
        with self._lock:
            if capability.directory_id in self._directories and not replace:
                raise ValueError(f"directory capability already mounted: {capability.directory_id}")
            self._directories[capability.directory_id] = capability
        receipt.operation = "mount"
        return receipt

    register_capability = mount
    mount_directory = mount
    register_directory = mount

    def unmount(self, directory_id: str) -> None:
        with self._lock:
            if self._directories.pop(directory_id, None) is None:
                raise CapabilityNotFoundError(f"unknown directory capability: {directory_id}")

    def capability(self, directory_id: str) -> DirectoryCapability:
        capability = self._directories.get(directory_id)
        if capability is None:
            raise CapabilityNotFoundError(f"unknown directory capability: {directory_id}")
        if capability.is_expired():
            raise CapabilityExpiredError(f"directory capability is expired: {directory_id}")
        return capability

    get_capability = capability

    def transfer_target(self, directory_id: str) -> ConnectorContract:
        """Return the connector backing an authorized streaming destination."""

        capability = self.capability(directory_id)
        connector = self._connectors.get(capability.connector_id)
        if connector is None:
            raise CapabilityNotFoundError(f"unknown connector: {capability.connector_id}")
        missing = [name for name in ("current_size", "usage", "write_stream") if not callable(getattr(connector, name, None))]
        if missing:
            raise CapabilityNotFoundError("directory connector does not implement the streaming target contract")
        return connector

    def discover(self, *, operation: str = "", tenant_id: str = "", project_id: str = "") -> list[CapabilityDescriptor]:
        descriptors: list[CapabilityDescriptor] = []
        for capability in self._directories.values():
            if capability.is_expired():
                continue
            if operation and not capability.allows(operation):
                continue
            if tenant_id and capability.tenant_id != tenant_id:
                continue
            if project_id and capability.project_id != project_id:
                continue
            connector = self._connectors[capability.connector_id]
            health = connector.health(capability)
            descriptors.append(
                CapabilityDescriptor(
                    capability_id=capability.directory_id,
                    provider_id=capability.connector_id,
                    version=connector.version,
                    operations=capability.operations,
                    input_schema={
                        "type": "object",
                        "properties": {
                            "directory_id": {"const": capability.directory_id},
                            "relative_path": {"type": "string"},
                        },
                        "required": ["directory_id", "relative_path"],
                        "additionalProperties": False,
                    },
                    risk_class="destructive" if "delete" in capability.operations else "bounded",
                    health_evidence={"ok": health.ok, "receipt_id": health.receipt_id},
                )
            )
        return sorted(descriptors, key=lambda item: item.capability_id)

    capability_discover = discover

    def _route(self, directory_id: str, operation: str) -> tuple[DirectoryCapability, ConnectorContract]:
        capability = self.capability(directory_id)
        if not capability.allows(operation):
            raise CapabilityPermissionError(f"operation '{operation}' is not granted for directory '{directory_id}'")
        return capability, self._connectors[capability.connector_id]

    def connector_health(self, directory_id: str) -> OperationReceipt:
        capability = self.capability(directory_id)
        return self._connectors[capability.connector_id].health(capability)

    def list(self, directory_id: str, relative_path: str = "", *, recursive: bool = False, limit: int = 5_000) -> OperationReceipt:
        capability, connector = self._route(directory_id, "list")
        return connector.list(capability, relative_path, recursive=recursive, limit=limit)

    directory_list = list

    def stat(self, directory_id: str, relative_path: str = "") -> OperationReceipt:
        capability = self.capability(directory_id)
        if not capability.allows("stat") and not any(capability.allows(value) for value in ("list", "read", "hash")):
            raise CapabilityPermissionError(f"operation 'stat' is not granted for directory '{directory_id}'")
        return self._connectors[capability.connector_id].stat(capability, relative_path)

    directory_stat = stat

    def read(self, directory_id: str, relative_path: str, *, max_bytes: int | None = None) -> OperationReceipt:
        capability, connector = self._route(directory_id, "read")
        return connector.read(capability, relative_path, max_bytes=max_bytes)

    directory_read = read

    def hash(self, directory_id: str, relative_path: str) -> OperationReceipt:
        capability, connector = self._route(directory_id, "hash")
        return connector.hash(capability, relative_path)

    directory_hash = hash

    def mkdir(
        self, directory_id: str, relative_path: str, *, parents: bool = True, exist_ok: bool = True
    ) -> OperationReceipt:
        capability, connector = self._route(directory_id, "mkdir")
        return connector.mkdir(capability, relative_path, parents=parents, exist_ok=exist_ok)

    directory_mkdir = mkdir

    def write_atomic(
        self,
        directory_id: str,
        relative_path: str,
        data: bytes | str,
        *,
        create_parents: bool = True,
        expected_sha256: str = "",
        encoding: str = "utf-8",
    ) -> OperationReceipt:
        capability, connector = self._route(directory_id, "write")
        payload = data.encode(encoding) if isinstance(data, str) else data
        if not isinstance(payload, bytes):
            raise TypeError("data must be bytes or str")
        return connector.write_atomic(
            capability,
            relative_path,
            payload,
            create_parents=create_parents,
            expected_sha256=expected_sha256,
        )

    directory_write_atomic = write_atomic

    def copy(
        self,
        source_directory_id: str,
        source_relative_path: str,
        destination_directory_id: str,
        destination_relative_path: str,
    ) -> OperationReceipt:
        source_capability, source_connector = self._route(source_directory_id, "read")
        destination_capability, destination_connector = self._route(destination_directory_id, "copy")
        if source_directory_id == destination_directory_id:
            return source_connector.copy(source_capability, source_relative_path, destination_relative_path)
        source = source_connector.read(source_capability, source_relative_path)
        destination = destination_connector.write_atomic(
            destination_capability, destination_relative_path, source.data or b"", expected_sha256=source.sha256
        )
        return OperationReceipt(
            "copy",
            destination_connector.connector_id,
            destination_directory_id,
            destination.relative_path,
            source_directory_id=source_directory_id,
            source_relative_path=source.relative_path,
            destination_directory_id=destination_directory_id,
            destination_relative_path=destination.relative_path,
            bytes_read=source.bytes_read,
            bytes_written=destination.bytes_written,
            sha256=destination.sha256,
        )

    directory_copy = copy

    def sync(
        self,
        source_directory_id: str,
        source_relative_path: str,
        destination_directory_id: str,
        destination_relative_path: str,
        *,
        delete_extraneous: bool = False,
        approved_delete: bool = False,
    ) -> OperationReceipt:
        source_capability, source_connector = self._route(source_directory_id, "list")
        if not source_capability.allows("read"):
            raise CapabilityPermissionError(f"operation 'read' is not granted for directory '{source_directory_id}'")
        destination_capability, destination_connector = self._route(destination_directory_id, "sync")
        if delete_extraneous and not approved_delete:
            raise ApprovalRequiredError("sync deletion requires explicit approved_delete=True")
        if delete_extraneous and not destination_capability.allows("delete"):
            raise CapabilityPermissionError(
                f"operation 'delete' is not granted for directory '{destination_directory_id}'"
            )
        if source_directory_id == destination_directory_id:
            return source_connector.sync(
                source_capability,
                source_relative_path,
                destination_relative_path,
                delete_extraneous=delete_extraneous,
            )
        source_rows = source_connector.list(source_capability, source_relative_path, recursive=True, limit=100_000)
        source_prefix = source_rows.relative_path
        copied: set[str] = set()
        bytes_written = 0
        for row in source_rows.entries:
            relative = row["path"]
            if source_prefix:
                relative = relative[len(source_prefix) :].lstrip("/")
            destination_path = "/".join(part for part in (destination_relative_path.strip("/"), relative) if part)
            if row["kind"] == "directory":
                destination_connector.mkdir(destination_capability, destination_path)
                continue
            source_read = source_connector.read(source_capability, row["path"])
            written = destination_connector.write_atomic(
                destination_capability, destination_path, source_read.data or b"", expected_sha256=source_read.sha256
            )
            copied.add(relative)
            bytes_written += written.bytes_written
        deleted = 0
        if delete_extraneous:
            destination_rows = destination_connector.list(
                destination_capability, destination_relative_path, recursive=True, limit=100_000
            )
            destination_prefix = destination_rows.relative_path
            for row in reversed(destination_rows.entries):
                relative = row["path"]
                if destination_prefix:
                    relative = relative[len(destination_prefix) :].lstrip("/")
                if row["kind"] == "file" and relative not in copied:
                    destination_connector.delete(destination_capability, row["path"])
                    deleted += 1
        return OperationReceipt(
            "sync",
            destination_connector.connector_id,
            destination_directory_id,
            destination_relative_path.strip("/"),
            source_directory_id=source_directory_id,
            source_relative_path=source_relative_path.strip("/"),
            destination_directory_id=destination_directory_id,
            destination_relative_path=destination_relative_path.strip("/"),
            bytes_written=bytes_written,
            metadata={"files_copied": len(copied), "files_deleted": deleted},
        )

    directory_sync = sync

    def delete(
        self, directory_id: str, relative_path: str, *, recursive: bool = False, approved: bool = False
    ) -> OperationReceipt:
        if not approved:
            raise ApprovalRequiredError("directory deletion requires explicit approved=True")
        capability, connector = self._route(directory_id, "delete")
        return connector.delete(capability, relative_path, recursive=recursive)

    directory_delete = delete

    def verify_roundtrip(self, directory_id: str, relative_path: str, data: bytes) -> OperationReceipt:
        """Write, read, and hash a bounded canary without deleting it."""

        expected = sha256(data).hexdigest()
        written = self.write_atomic(directory_id, relative_path, data, expected_sha256=expected)
        read = self.read(directory_id, relative_path, max_bytes=len(data))
        hashed = self.hash(directory_id, relative_path)
        ok = written.sha256 == read.sha256 == hashed.sha256 == expected and read.data == data
        return OperationReceipt(
            "verify_roundtrip",
            written.connector_id,
            directory_id,
            written.relative_path,
            ok=ok,
            bytes_read=read.bytes_read,
            bytes_written=written.bytes_written,
            sha256=expected if ok else "",
            metadata={"write_read_hash_match": ok},
            error_class="" if ok else "RoundtripMismatch",
            error_message="" if ok else "write/read/hash roundtrip did not match",
        )


__all__ = ["DirectoryBroker"]
