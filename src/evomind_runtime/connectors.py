"""Directory connector contracts and safe local/in-memory implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat as stat_module
import threading
import time
from typing import Any

from .capabilities import (
    DIRECTORY_OPERATIONS,
    DirectoryBoundaryError,
    DirectoryCapability,
    DirectoryQuotaError,
    OperationReceipt,
    utc_now,
)


_REPARSE_POINT_ATTRIBUTE = 0x400


def normalize_relative_path(value: str | os.PathLike[str], *, allow_root: bool = True) -> str:
    """Return a portable normalized relative path or fail closed."""

    raw = os.fspath(value)
    if not isinstance(raw, str):
        raw = os.fsdecode(raw)
    if "\x00" in raw:
        raise DirectoryBoundaryError("relative_path contains a null byte")
    if not raw or raw == ".":
        if allow_root:
            return ""
        raise DirectoryBoundaryError("relative_path must identify an entry below the directory root")
    windows = PureWindowsPath(raw)
    posix = PurePosixPath(raw.replace("\\", "/"))
    if windows.is_absolute() or windows.drive or posix.is_absolute() or raw.startswith(("\\\\", "//")):
        raise DirectoryBoundaryError("absolute paths are not allowed; use directory_id + relative_path")
    parts = tuple(part for part in posix.parts if part not in {"", "."})
    if any(part == ".." for part in parts):
        raise DirectoryBoundaryError("relative_path cannot contain '..'")
    reserved_windows_names = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
    }
    for part in parts:
        if ":" in part:
            raise DirectoryBoundaryError("relative_path cannot address an alternate data stream")
        if part != part.rstrip(" ."):
            raise DirectoryBoundaryError("relative_path cannot contain Windows-ambiguous trailing characters")
        if part.split(".", 1)[0].upper() in reserved_windows_names:
            raise DirectoryBoundaryError("relative_path contains a reserved device name")
    if not parts:
        if allow_root:
            return ""
        raise DirectoryBoundaryError("relative_path must identify an entry below the directory root")
    return "/".join(parts)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat_module.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE)


class ConnectorContract(ABC):
    """Uniform connector interface consumed by :class:`DirectoryBroker`."""

    connector_id: str
    version: str
    supported_operations: frozenset[str] = DIRECTORY_OPERATIONS

    @abstractmethod
    def health(self, capability: DirectoryCapability) -> OperationReceipt: ...

    @abstractmethod
    def list(
        self, capability: DirectoryCapability, relative_path: str = "", *, recursive: bool = False, limit: int = 5_000
    ) -> OperationReceipt: ...

    @abstractmethod
    def stat(self, capability: DirectoryCapability, relative_path: str = "") -> OperationReceipt: ...

    @abstractmethod
    def read(
        self, capability: DirectoryCapability, relative_path: str, *, max_bytes: int | None = None
    ) -> OperationReceipt: ...

    @abstractmethod
    def hash(self, capability: DirectoryCapability, relative_path: str) -> OperationReceipt: ...

    @abstractmethod
    def mkdir(
        self, capability: DirectoryCapability, relative_path: str, *, parents: bool = True, exist_ok: bool = True
    ) -> OperationReceipt: ...

    @abstractmethod
    def write_atomic(
        self,
        capability: DirectoryCapability,
        relative_path: str,
        data: bytes,
        *,
        create_parents: bool = True,
        expected_sha256: str = "",
    ) -> OperationReceipt: ...

    @abstractmethod
    def copy(
        self, capability: DirectoryCapability, source_relative_path: str, destination_relative_path: str
    ) -> OperationReceipt: ...

    @abstractmethod
    def sync(
        self,
        capability: DirectoryCapability,
        source_relative_path: str,
        destination_relative_path: str,
        *,
        delete_extraneous: bool = False,
    ) -> OperationReceipt: ...

    @abstractmethod
    def delete(
        self, capability: DirectoryCapability, relative_path: str, *, recursive: bool = False
    ) -> OperationReceipt: ...

    @abstractmethod
    def usage(self, capability: DirectoryCapability) -> tuple[int, int]: ...


class LocalDirectoryConnector(ConnectorContract):
    """Fail-closed connector for server-side local directory mounts."""

    def __init__(
        self,
        connector_id: str = "local",
        *,
        version: str = "1.0.0",
        allowed_roots: Iterable[str | os.PathLike[str]] | None = None,
    ) -> None:
        self.connector_id = str(connector_id).strip()
        self.version = str(version).strip()
        if not self.connector_id or not self.version:
            raise ValueError("connector_id and version must be non-empty")
        trusted_roots: list[Path] = []
        for value in allowed_roots or ():
            requested = Path(value).expanduser()
            if _is_link_or_reparse(requested):
                raise DirectoryBoundaryError("an allowlisted connector root cannot be a link or reparse point")
            trusted_roots.append(requested.resolve(strict=False))
        self._allowed_roots = tuple(trusted_roots)
        self._lock = threading.RLock()

    def _root(self, capability: DirectoryCapability) -> Path:
        if capability.connector_id != self.connector_id:
            raise DirectoryBoundaryError("directory capability is bound to a different connector")
        requested = Path(capability.opaque_root_ref).expanduser()
        if not requested.is_absolute():
            raise DirectoryBoundaryError("local opaque_root_ref must be an absolute server-side path")
        if _is_link_or_reparse(requested):
            raise DirectoryBoundaryError("directory root cannot be a symbolic link or reparse point")
        root = requested.resolve(strict=False)
        if self._allowed_roots and not any(root == allowed or root.is_relative_to(allowed) for allowed in self._allowed_roots):
            raise DirectoryBoundaryError("directory root is not inside an allowlisted connector root")
        if not root.exists() or not root.is_dir() or _is_link_or_reparse(root):
            raise DirectoryBoundaryError("directory root must be an existing regular non-reparse directory")
        return root

    def _resolve(self, capability: DirectoryCapability, relative_path: str, *, allow_root: bool = True) -> tuple[Path, str]:
        root = self._root(capability)
        normalized = normalize_relative_path(relative_path, allow_root=allow_root)
        candidate = root.joinpath(*normalized.split("/")) if normalized else root
        current = root
        for part in normalized.split("/") if normalized else ():
            current = current / part
            if current.exists() and _is_link_or_reparse(current):
                raise DirectoryBoundaryError("relative_path crosses a symbolic link or reparse point")
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise DirectoryBoundaryError("relative_path escapes the mounted directory") from exc
        return candidate, normalized

    def _receipt(self, capability: DirectoryCapability, operation: str, relative_path: str = "", **values: Any) -> OperationReceipt:
        return OperationReceipt(operation, self.connector_id, capability.directory_id, relative_path, **values)

    def _tree_rows(self, root: Path, target: Path, *, recursive: bool, limit: int) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 100_000))
        rows: list[dict[str, Any]] = []
        pending = [target]
        while pending:
            parent = pending.pop()
            children = sorted(parent.iterdir(), key=lambda item: item.name.casefold())
            for child in children:
                if _is_link_or_reparse(child):
                    raise DirectoryBoundaryError("directory tree contains a symbolic link or reparse point")
                info = child.stat()
                relative = child.relative_to(root).as_posix()
                rows.append(
                    {
                        "path": relative,
                        "kind": "directory" if child.is_dir() else "file",
                        "bytes": info.st_size if child.is_file() else 0,
                        "modified_ns": info.st_mtime_ns,
                    }
                )
                if len(rows) >= bounded_limit:
                    return rows
                if recursive and child.is_dir():
                    pending.append(child)
        return rows

    def _check_quota(self, capability: DirectoryCapability, target: Path, new_size: int) -> None:
        file_count, byte_count = self.usage(capability)
        old_size = 0
        old_file = False
        if target.exists():
            if _is_link_or_reparse(target) or not target.is_file():
                raise DirectoryBoundaryError("write destination must be a regular file or a new path")
            old_file = True
            old_size = target.stat().st_size
        projected_files = file_count + (0 if old_file else 1)
        projected_bytes = byte_count - old_size + new_size
        if projected_files > capability.max_files:
            raise DirectoryQuotaError("write would exceed the directory file quota")
        if projected_bytes > capability.max_bytes:
            raise DirectoryQuotaError("write would exceed the directory byte quota")

    def health(self, capability: DirectoryCapability) -> OperationReceipt:
        started = utc_now()
        try:
            self._root(capability)
        except Exception as exc:
            return self._receipt(
                capability,
                "health",
                ok=False,
                started_at=started,
                error_class=type(exc).__name__,
                error_message=str(exc),
            )
        return self._receipt(capability, "health", started_at=started, metadata={"ready": True, "version": self.version})

    def list(
        self, capability: DirectoryCapability, relative_path: str = "", *, recursive: bool = False, limit: int = 5_000
    ) -> OperationReceipt:
        target, normalized = self._resolve(capability, relative_path)
        if not target.is_dir() or _is_link_or_reparse(target):
            raise NotADirectoryError(normalized or ".")
        rows = self._tree_rows(self._root(capability), target, recursive=recursive, limit=limit)
        return self._receipt(capability, "list", normalized, entries=rows, metadata={"recursive": recursive})

    def stat(self, capability: DirectoryCapability, relative_path: str = "") -> OperationReceipt:
        target, normalized = self._resolve(capability, relative_path)
        if not target.exists():
            raise FileNotFoundError(normalized)
        if _is_link_or_reparse(target):
            raise DirectoryBoundaryError("stat target is a symbolic link or reparse point")
        info = target.stat()
        metadata = {
            "kind": "directory" if target.is_dir() else "file",
            "bytes": info.st_size if target.is_file() else 0,
            "modified_ns": info.st_mtime_ns,
        }
        return self._receipt(capability, "stat", normalized, metadata=metadata)

    def read(
        self, capability: DirectoryCapability, relative_path: str, *, max_bytes: int | None = None
    ) -> OperationReceipt:
        target, normalized = self._resolve(capability, relative_path, allow_root=False)
        if not target.is_file() or _is_link_or_reparse(target):
            raise FileNotFoundError(normalized)
        size = target.stat().st_size
        if max_bytes is not None and size > max(0, int(max_bytes)):
            raise DirectoryQuotaError("read exceeds the requested maximum byte count")
        data = target.read_bytes()
        return self._receipt(
            capability, "read", normalized, bytes_read=len(data), sha256=sha256(data).hexdigest(), data=data
        )

    def hash(self, capability: DirectoryCapability, relative_path: str) -> OperationReceipt:
        read_receipt = self.read(capability, relative_path)
        return self._receipt(
            capability,
            "hash",
            read_receipt.relative_path,
            bytes_read=read_receipt.bytes_read,
            sha256=read_receipt.sha256,
        )

    def mkdir(
        self, capability: DirectoryCapability, relative_path: str, *, parents: bool = True, exist_ok: bool = True
    ) -> OperationReceipt:
        target, normalized = self._resolve(capability, relative_path, allow_root=False)
        target.mkdir(parents=parents, exist_ok=exist_ok)
        self._resolve(capability, normalized, allow_root=False)
        return self._receipt(capability, "mkdir", normalized, metadata={"created": True})

    def write_atomic(
        self,
        capability: DirectoryCapability,
        relative_path: str,
        data: bytes,
        *,
        create_parents: bool = True,
        expected_sha256: str = "",
    ) -> OperationReceipt:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        digest = sha256(data).hexdigest()
        if expected_sha256 and digest.lower() != expected_sha256.lower():
            raise ValueError("write payload does not match expected_sha256")
        target, normalized = self._resolve(capability, relative_path, allow_root=False)
        with self._lock:
            self._check_quota(capability, target, len(data))
            if create_parents:
                target.parent.mkdir(parents=True, exist_ok=True)
            self._resolve(capability, normalized, allow_root=False)
            if not target.parent.is_dir() or _is_link_or_reparse(target.parent):
                raise DirectoryBoundaryError("write parent is not a regular directory")
            temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{time.time_ns()}")
            try:
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                if _is_link_or_reparse(target):
                    raise DirectoryBoundaryError("write destination became a link or reparse point")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return self._receipt(capability, "write", normalized, bytes_written=len(data), sha256=digest)

    def copy(
        self, capability: DirectoryCapability, source_relative_path: str, destination_relative_path: str
    ) -> OperationReceipt:
        source = self.read(capability, source_relative_path)
        destination = self.write_atomic(capability, destination_relative_path, source.data or b"")
        return self._receipt(
            capability,
            "copy",
            destination.relative_path,
            source_relative_path=source.relative_path,
            destination_relative_path=destination.relative_path,
            bytes_read=source.bytes_read,
            bytes_written=destination.bytes_written,
            sha256=destination.sha256,
        )

    def sync(
        self,
        capability: DirectoryCapability,
        source_relative_path: str,
        destination_relative_path: str,
        *,
        delete_extraneous: bool = False,
    ) -> OperationReceipt:
        source, source_normalized = self._resolve(capability, source_relative_path)
        destination, destination_normalized = self._resolve(capability, destination_relative_path, allow_root=False)
        if not source.is_dir():
            return self.copy(capability, source_normalized, destination_normalized)
        source_resolved = source.resolve(strict=True)
        destination_resolved = destination.resolve(strict=False)
        if destination_resolved == source_resolved or destination_resolved.is_relative_to(source_resolved):
            raise DirectoryBoundaryError("sync destination cannot be the source or a child of the source")
        rows = self._tree_rows(self._root(capability), source, recursive=True, limit=100_000)
        copied: set[str] = set()
        bytes_written = 0
        for row in rows:
            relative = Path(row["path"]).relative_to(source.relative_to(self._root(capability))).as_posix()
            target_relative = "/".join(part for part in (destination_normalized, relative) if part)
            if row["kind"] == "directory":
                self.mkdir(capability, target_relative)
            else:
                receipt = self.copy(capability, row["path"], target_relative)
                copied.add(relative)
                bytes_written += receipt.bytes_written
        deleted = 0
        if delete_extraneous and destination.exists():
            destination_rows = self._tree_rows(self._root(capability), destination, recursive=True, limit=100_000)
            for row in reversed(destination_rows):
                relative = Path(row["path"]).relative_to(destination.relative_to(self._root(capability))).as_posix()
                if row["kind"] == "file" and relative not in copied:
                    self.delete(capability, row["path"])
                    deleted += 1
        return self._receipt(
            capability,
            "sync",
            destination_normalized,
            source_relative_path=source_normalized,
            destination_relative_path=destination_normalized,
            bytes_written=bytes_written,
            metadata={"files_copied": len(copied), "files_deleted": deleted},
        )

    def delete(
        self, capability: DirectoryCapability, relative_path: str, *, recursive: bool = False
    ) -> OperationReceipt:
        target, normalized = self._resolve(capability, relative_path, allow_root=False)
        if not target.exists():
            raise FileNotFoundError(normalized)
        if _is_link_or_reparse(target):
            raise DirectoryBoundaryError("delete target is a symbolic link or reparse point")
        bytes_removed = 0
        if target.is_file():
            bytes_removed = target.stat().st_size
            target.unlink()
        elif recursive:
            rows = self._tree_rows(self._root(capability), target, recursive=True, limit=100_000)
            bytes_removed = sum(int(row["bytes"]) for row in rows if row["kind"] == "file")
            shutil.rmtree(target)
        else:
            target.rmdir()
        return self._receipt(
            capability, "delete", normalized, metadata={"deleted": True, "bytes_removed": bytes_removed}
        )

    def current_size(self, capability: DirectoryCapability, relative_path: str) -> int | None:
        """Return the current regular-file size for a resumable transfer target."""

        target, normalized = self._resolve(capability, relative_path, allow_root=False)
        if not target.exists():
            return None
        if _is_link_or_reparse(target) or not target.is_file():
            raise DirectoryBoundaryError(f"transfer target is not a regular file: {normalized}")
        return int(target.stat().st_size)

    def write_stream(
        self,
        capability: DirectoryCapability,
        relative_path: str,
        chunks: Iterable[bytes],
        *,
        offset: int,
        expected_total_bytes: int | None,
        expected_sha256: str,
    ) -> OperationReceipt:
        """Stage a bounded stream and atomically replace the destination on success."""

        requested_offset = int(offset)
        if requested_offset < 0:
            raise ValueError("offset must be non-negative")
        expected_total = None if expected_total_bytes is None else int(expected_total_bytes)
        if expected_total is not None and expected_total < requested_offset:
            raise ValueError("expected_total_bytes cannot be smaller than offset")
        target, normalized = self._resolve(capability, relative_path, allow_root=False)
        with self._lock:
            existing_size = self.current_size(capability, normalized)
            if requested_offset and existing_size != requested_offset:
                raise OSError("resume offset does not match the current target size")
            previous_size = int(existing_size or 0)
            files, byte_count = self.usage(capability)
            if files + (0 if existing_size is not None else 1) > capability.max_files:
                raise DirectoryQuotaError("stream write would exceed the directory file quota")
            if expected_total is not None and byte_count - previous_size + expected_total > capability.max_bytes:
                raise DirectoryQuotaError("stream write would exceed the directory byte quota")

            target.parent.mkdir(parents=True, exist_ok=True)
            target, normalized = self._resolve(capability, normalized, allow_root=False)
            if not target.parent.is_dir() or _is_link_or_reparse(target.parent):
                raise DirectoryBoundaryError("stream write parent is not a regular directory")
            temporary = target.with_name(f".{target.name}.stream-{os.getpid()}-{time.time_ns()}")
            digest = sha256()
            streamed = 0
            final_size = 0
            try:
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as handle:
                    if requested_offset:
                        with target.open("rb") as existing:
                            while True:
                                block = existing.read(1 << 20)
                                if not block:
                                    break
                                handle.write(block)
                                digest.update(block)
                                final_size += len(block)
                        if final_size != requested_offset:
                            raise OSError("target changed while preparing a resumed stream")
                    for chunk in chunks:
                        if not isinstance(chunk, bytes):
                            raise TypeError("stream chunks must be bytes")
                        if not chunk:
                            continue
                        streamed += len(chunk)
                        final_size += len(chunk)
                        if expected_total is not None and final_size > expected_total:
                            raise OSError("stream exceeds expected_total_bytes")
                        if byte_count - previous_size + final_size > capability.max_bytes:
                            raise DirectoryQuotaError("stream write would exceed the directory byte quota")
                        handle.write(chunk)
                        digest.update(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                if expected_total is not None and final_size != expected_total:
                    raise OSError("stream length does not match expected_total_bytes")
                actual_sha256 = digest.hexdigest()
                if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
                    raise ValueError("stream does not match expected_sha256")
                if _is_link_or_reparse(target):
                    raise DirectoryBoundaryError("stream destination became a link or reparse point")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return self._receipt(
            capability,
            "write_stream",
            normalized,
            bytes_written=streamed,
            sha256=actual_sha256,
            metadata={"offset": requested_offset, "final_bytes": final_size},
        )

    def usage(self, capability: DirectoryCapability) -> tuple[int, int]:
        root = self._root(capability)
        files = 0
        byte_count = 0
        pending = [root]
        while pending:
            parent = pending.pop()
            for child in parent.iterdir():
                if _is_link_or_reparse(child):
                    raise DirectoryBoundaryError("directory tree contains a symbolic link or reparse point")
                if child.is_dir():
                    pending.append(child)
                elif child.is_file():
                    files += 1
                    byte_count += child.stat().st_size
                else:
                    raise DirectoryBoundaryError("directory tree contains a non-regular entry")
        return files, byte_count


class MemoryDirectoryConnector(ConnectorContract):
    """Thread-safe in-memory connector used for tests, shadow plans, and canaries."""

    def __init__(self, connector_id: str = "memory", *, version: str = "1.0.0") -> None:
        self.connector_id = str(connector_id).strip()
        self.version = str(version).strip()
        if not self.connector_id or not self.version:
            raise ValueError("connector_id and version must be non-empty")
        self._roots: dict[str, tuple[dict[str, bytes], set[str], dict[str, int]]] = {}
        self._lock = threading.RLock()

    def _state(self, capability: DirectoryCapability) -> tuple[dict[str, bytes], set[str], dict[str, int]]:
        if capability.connector_id != self.connector_id:
            raise DirectoryBoundaryError("directory capability is bound to a different connector")
        with self._lock:
            return self._roots.setdefault(capability.opaque_root_ref, ({}, {""}, {"": time.time_ns()}))

    def _path(self, capability: DirectoryCapability, value: str, *, allow_root: bool = True) -> str:
        if capability.connector_id != self.connector_id:
            raise DirectoryBoundaryError("directory capability is bound to a different connector")
        return normalize_relative_path(value, allow_root=allow_root)

    @staticmethod
    def _parents(path: str) -> list[str]:
        parts = path.split("/") if path else []
        return ["/".join(parts[:index]) for index in range(1, len(parts))]

    def _receipt(self, capability: DirectoryCapability, operation: str, relative_path: str = "", **values: Any) -> OperationReceipt:
        return OperationReceipt(operation, self.connector_id, capability.directory_id, relative_path, **values)

    def _ensure_quota(
        self, capability: DirectoryCapability, files: dict[str, bytes], path: str, data: bytes
    ) -> None:
        projected_files = len(files) + (0 if path in files else 1)
        projected_bytes = sum(len(value) for value in files.values()) - len(files.get(path, b"")) + len(data)
        if projected_files > capability.max_files:
            raise DirectoryQuotaError("write would exceed the directory file quota")
        if projected_bytes > capability.max_bytes:
            raise DirectoryQuotaError("write would exceed the directory byte quota")

    def health(self, capability: DirectoryCapability) -> OperationReceipt:
        self._path(capability, "")
        self._state(capability)
        return self._receipt(capability, "health", metadata={"ready": True, "version": self.version})

    def list(
        self, capability: DirectoryCapability, relative_path: str = "", *, recursive: bool = False, limit: int = 5_000
    ) -> OperationReceipt:
        root = self._path(capability, relative_path)
        files, directories, modified_ns = self._state(capability)
        if root not in directories:
            raise NotADirectoryError(root or ".")
        prefix = f"{root}/" if root else ""
        rows: list[dict[str, Any]] = []
        candidates = sorted((directories - {root}) | set(files))
        for candidate in candidates:
            if not candidate.startswith(prefix):
                continue
            remainder = candidate[len(prefix) :]
            if not remainder or (not recursive and "/" in remainder):
                continue
            is_file = candidate in files
            rows.append(
                {
                    "path": candidate,
                    "kind": "file" if is_file else "directory",
                    "bytes": len(files[candidate]) if is_file else 0,
                    "modified_ns": modified_ns.get(candidate, 0),
                }
            )
            if len(rows) >= max(1, min(int(limit), 100_000)):
                break
        return self._receipt(capability, "list", root, entries=rows, metadata={"recursive": recursive})

    def stat(self, capability: DirectoryCapability, relative_path: str = "") -> OperationReceipt:
        path = self._path(capability, relative_path)
        files, directories, modified_ns = self._state(capability)
        if path not in files and path not in directories:
            raise FileNotFoundError(path)
        is_file = path in files
        return self._receipt(
            capability,
            "stat",
            path,
            metadata={
                "kind": "file" if is_file else "directory",
                "bytes": len(files[path]) if is_file else 0,
                "modified_ns": modified_ns.get(path, 0),
            },
        )

    def read(
        self, capability: DirectoryCapability, relative_path: str, *, max_bytes: int | None = None
    ) -> OperationReceipt:
        path = self._path(capability, relative_path, allow_root=False)
        files, _, _ = self._state(capability)
        if path not in files:
            raise FileNotFoundError(path)
        data = files[path]
        if max_bytes is not None and len(data) > max(0, int(max_bytes)):
            raise DirectoryQuotaError("read exceeds the requested maximum byte count")
        return self._receipt(capability, "read", path, bytes_read=len(data), sha256=sha256(data).hexdigest(), data=data)

    def hash(self, capability: DirectoryCapability, relative_path: str) -> OperationReceipt:
        read_receipt = self.read(capability, relative_path)
        return self._receipt(
            capability, "hash", read_receipt.relative_path, bytes_read=read_receipt.bytes_read, sha256=read_receipt.sha256
        )

    def mkdir(
        self, capability: DirectoryCapability, relative_path: str, *, parents: bool = True, exist_ok: bool = True
    ) -> OperationReceipt:
        path = self._path(capability, relative_path, allow_root=False)
        files, directories, modified_ns = self._state(capability)
        with self._lock:
            if path in files:
                raise FileExistsError(path)
            parent = path.rpartition("/")[0]
            if not parents and parent not in directories:
                raise FileNotFoundError(parent)
            if not exist_ok and path in directories:
                raise FileExistsError(path)
            if parents:
                directories.update(self._parents(path))
            directories.add(path)
            modified_ns[path] = time.time_ns()
        return self._receipt(capability, "mkdir", path, metadata={"created": True})

    def write_atomic(
        self,
        capability: DirectoryCapability,
        relative_path: str,
        data: bytes,
        *,
        create_parents: bool = True,
        expected_sha256: str = "",
    ) -> OperationReceipt:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        path = self._path(capability, relative_path, allow_root=False)
        digest = sha256(data).hexdigest()
        if expected_sha256 and digest.lower() != expected_sha256.lower():
            raise ValueError("write payload does not match expected_sha256")
        parent = path.rpartition("/")[0]
        files, directories, modified_ns = self._state(capability)
        with self._lock:
            if parent not in directories and not create_parents:
                raise FileNotFoundError(parent)
            self._ensure_quota(capability, files, path, data)
            if create_parents:
                directories.update(self._parents(path))
            files[path] = bytes(data)
            modified_ns[path] = time.time_ns()
        return self._receipt(capability, "write", path, bytes_written=len(data), sha256=digest)

    def copy(
        self, capability: DirectoryCapability, source_relative_path: str, destination_relative_path: str
    ) -> OperationReceipt:
        source = self.read(capability, source_relative_path)
        destination = self.write_atomic(capability, destination_relative_path, source.data or b"")
        return self._receipt(
            capability,
            "copy",
            destination.relative_path,
            source_relative_path=source.relative_path,
            destination_relative_path=destination.relative_path,
            bytes_read=source.bytes_read,
            bytes_written=destination.bytes_written,
            sha256=destination.sha256,
        )

    def sync(
        self,
        capability: DirectoryCapability,
        source_relative_path: str,
        destination_relative_path: str,
        *,
        delete_extraneous: bool = False,
    ) -> OperationReceipt:
        source = self._path(capability, source_relative_path)
        destination = self._path(capability, destination_relative_path, allow_root=False)
        files, directories, modified_ns = self._state(capability)
        if source in files:
            return self.copy(capability, source, destination)
        if source not in directories:
            raise FileNotFoundError(source)
        if destination == source or destination.startswith(f"{source}/"):
            raise DirectoryBoundaryError("sync destination cannot be the source or a child of the source")
        source_prefix = f"{source}/" if source else ""
        copied: set[str] = set()
        total = 0
        for path, data in list(files.items()):
            if not path.startswith(source_prefix):
                continue
            relative = path[len(source_prefix) :]
            target = f"{destination}/{relative}" if relative else destination
            receipt = self.write_atomic(capability, target, data)
            copied.add(relative)
            total += receipt.bytes_written
        deleted = 0
        if delete_extraneous:
            destination_prefix = f"{destination}/"
            for path in list(files):
                if path.startswith(destination_prefix) and path[len(destination_prefix) :] not in copied:
                    del files[path]
                    modified_ns.pop(path, None)
                    deleted += 1
        return self._receipt(
            capability,
            "sync",
            destination,
            source_relative_path=source,
            destination_relative_path=destination,
            bytes_written=total,
            metadata={"files_copied": len(copied), "files_deleted": deleted},
        )

    def delete(
        self, capability: DirectoryCapability, relative_path: str, *, recursive: bool = False
    ) -> OperationReceipt:
        path = self._path(capability, relative_path, allow_root=False)
        files, directories, modified_ns = self._state(capability)
        with self._lock:
            if path in files:
                removed = len(files.pop(path))
                modified_ns.pop(path, None)
            elif path in directories:
                prefix = f"{path}/"
                children = [item for item in files if item.startswith(prefix)]
                child_dirs = [item for item in directories if item.startswith(prefix)]
                if (children or child_dirs) and not recursive:
                    raise OSError("directory is not empty")
                removed = sum(len(files[item]) for item in children)
                for item in children:
                    del files[item]
                    modified_ns.pop(item, None)
                directories.difference_update(child_dirs)
                directories.discard(path)
                modified_ns.pop(path, None)
            else:
                raise FileNotFoundError(path)
        return self._receipt(capability, "delete", path, metadata={"deleted": True, "bytes_removed": removed})

    def usage(self, capability: DirectoryCapability) -> tuple[int, int]:
        self._path(capability, "")
        files, _, _ = self._state(capability)
        with self._lock:
            return len(files), sum(len(value) for value in files.values())


FakeDirectoryConnector = MemoryDirectoryConnector
FakeConnector = MemoryDirectoryConnector
MemoryConnector = MemoryDirectoryConnector


__all__ = [
    "ConnectorContract",
    "FakeConnector",
    "FakeDirectoryConnector",
    "LocalDirectoryConnector",
    "MemoryConnector",
    "MemoryDirectoryConnector",
    "normalize_relative_path",
]
