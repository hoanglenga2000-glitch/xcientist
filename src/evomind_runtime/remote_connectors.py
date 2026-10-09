"""Fail-closed remote connector building blocks for the super-agent runtime.

This module never opens a real network connection by itself.  Network clients,
HTTP openers, transfer targets, identity verifiers, and job executors are all
injected by the server-side integration layer.  Model-facing callers address a
mount with ``directory_id`` plus a normalized relative path; opaque roots and
credentials never appear in operation receipts.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import posixpath
from pathlib import PurePosixPath
import re
import stat as stat_module
import threading
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from .capabilities import (
    ApprovalRequiredError,
    CapabilityDescriptor,
    CapabilityPermissionError,
    DirectoryBoundaryError,
    DirectoryCapability,
    DirectoryQuotaError,
    OperationReceipt,
    utc_now,
)
from .connectors import ConnectorContract, normalize_relative_path


_SFTP_OPERATIONS = frozenset({"health", "list", "stat", "read", "hash", "write", "mkdir", "copy", "sync", "delete"})
_CONTENT_RANGE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$", re.IGNORECASE)


def _normalized_allowed_hosts(values: Iterable[str]) -> frozenset[str]:
    hosts: set[str] = set()
    for raw in values:
        value = str(raw).strip().lower().rstrip(".")
        if not value or "/" in value or "\\" in value or ":" in value or ".." in value:
            raise ValueError("allowed source hosts must be exact DNS host names")
        try:
            encoded = value.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ValueError("allowed source host is not a valid DNS name") from exc
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", encoded):
            raise ValueError("allowed source host is not a valid DNS name")
        hosts.add(encoded)
    if not hosts:
        raise ValueError("at least one exact source host must be allowlisted")
    return frozenset(hosts)


def _validate_https_source(value: str, allowed_hosts: frozenset[str]) -> str:
    url = str(value).strip()
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError("source_url must be an absolute HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("source_url must not embed credentials")
    if parsed.fragment:
        raise ValueError("source_url must not contain a fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("source_url contains an invalid port") from exc
    if port not in {None, 443}:
        raise ValueError("source_url must use the default HTTPS port")
    host = (parsed.hostname or "").lower().rstrip(".")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("source_url host is invalid") from exc
    if host not in allowed_hosts:
        raise ValueError("source_url host is not allowlisted")
    return url


class _AllowlistedHttpsRedirectHandler(HTTPRedirectHandler):
    def __init__(self, allowed_hosts: frozenset[str]) -> None:
        super().__init__()
        self._allowed_hosts = allowed_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validated = _validate_https_source(newurl, self._allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, validated)


class RestrictedHttpOpener:
    """urllib adapter that rejects off-allowlist and downgraded redirects."""

    def __init__(self, *, allowed_hosts: Iterable[str], timeout_seconds: float = 30.0) -> None:
        self.allowed_hosts = _normalized_allowed_hosts(allowed_hosts)
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._opener = build_opener(_AllowlistedHttpsRedirectHandler(self.allowed_hosts))

    def __call__(self, url: str, *, headers: Mapping[str, str] | None = None):
        validated = _validate_https_source(url, self.allowed_hosts)
        request = Request(validated, headers=dict(headers or {}), method="GET")
        response = self._opener.open(request, timeout=self.timeout_seconds)
        final_url = response.geturl() if callable(getattr(response, "geturl", None)) else validated
        _validate_https_source(final_url, self.allowed_hosts)
        return response


def _remote_root(capability: DirectoryCapability, connector_id: str) -> str:
    if capability.connector_id != connector_id:
        raise DirectoryBoundaryError("directory capability is bound to a different connector")
    raw = str(capability.opaque_root_ref)
    if not raw or "\x00" in raw or "\\" in raw:
        raise DirectoryBoundaryError("remote opaque root must be a canonical POSIX absolute path")
    root = PurePosixPath(raw)
    if not root.is_absolute() or any(part in {".", ".."} for part in root.parts):
        raise DirectoryBoundaryError("remote opaque root must be a canonical POSIX absolute path")
    canonical = str(root)
    if raw != canonical and raw.rstrip("/") != canonical:
        raise DirectoryBoundaryError("remote opaque root must not contain ambiguous path syntax")
    return canonical


def _remote_path(
    capability: DirectoryCapability,
    connector_id: str,
    relative_path: str,
    *,
    allow_root: bool = True,
) -> tuple[str, str, str]:
    root = _remote_root(capability, connector_id)
    normalized = normalize_relative_path(relative_path, allow_root=allow_root)
    candidate = posixpath.join(root, normalized) if normalized else root
    canonical = posixpath.normpath(candidate)
    if canonical != root and not canonical.startswith(f"{root.rstrip('/')}/"):
        raise DirectoryBoundaryError("relative_path escapes the mounted directory")
    return root, canonical, normalized


def _mode_kind(mode: int) -> str:
    if stat_module.S_ISLNK(mode):
        return "symlink"
    if stat_module.S_ISDIR(mode):
        return "directory"
    if stat_module.S_ISREG(mode):
        return "file"
    return "other"


def _modified_ns(attributes: Any) -> int:
    return int(float(getattr(attributes, "st_mtime", 0) or 0) * 1_000_000_000)


@contextmanager
def _managed(value: Any) -> Iterator[Any]:
    if hasattr(value, "__enter__") and hasattr(value, "__exit__"):
        with value as entered:
            yield entered
        return
    try:
        yield value
    finally:
        close = getattr(value, "close", None)
        if callable(close):
            close()


class SftpDirectoryConnector(ConnectorContract):
    """Capability-bounded SFTP connector using an injected client factory.

    The factory receives the server-side :class:`DirectoryCapability` and must
    return an SFTP-like client.  No host, account, or credential is accepted by
    an operation method.  Required client methods intentionally match the
    portable subset of Paramiko's SFTP client.
    """

    supported_operations = _SFTP_OPERATIONS

    def __init__(
        self,
        client_factory: Callable[[DirectoryCapability], Any],
        connector_id: str = "sftp",
        *,
        version: str = "1.0.0",
        chunk_size: int = 1 << 20,
    ) -> None:
        if not callable(client_factory):
            raise TypeError("client_factory must be callable")
        self.connector_id = str(connector_id).strip()
        self.version = str(version).strip()
        if not self.connector_id or not self.version:
            raise ValueError("connector_id and version must be non-empty")
        self._client_factory = client_factory
        self._chunk_size = max(4096, int(chunk_size))
        self._lock = threading.RLock()

    @contextmanager
    def _client(self, capability: DirectoryCapability) -> Iterator[Any]:
        _remote_root(capability, self.connector_id)
        with _managed(self._client_factory(capability)) as client:
            yield client

    def _receipt(
        self,
        capability: DirectoryCapability,
        operation: str,
        relative_path: str = "",
        **values: Any,
    ) -> OperationReceipt:
        return OperationReceipt(operation, self.connector_id, capability.directory_id, relative_path, **values)

    @staticmethod
    def _lstat_or_none(client: Any, path: str) -> Any | None:
        try:
            return client.lstat(path)
        except FileNotFoundError:
            return None

    def _assert_safe_path(
        self,
        client: Any,
        root: str,
        normalized: str,
        *,
        missing_leaf_ok: bool = False,
        missing_parents_ok: bool = False,
    ) -> Any | None:
        root_attributes = client.lstat(root)
        if _mode_kind(int(root_attributes.st_mode)) != "directory":
            raise DirectoryBoundaryError("remote mount root must be a regular directory")
        current = root
        leaf_attributes: Any | None = root_attributes
        parts = normalized.split("/") if normalized else []
        for index, part in enumerate(parts):
            current = posixpath.join(current, part)
            attributes = self._lstat_or_none(client, current)
            is_leaf = index == len(parts) - 1
            if attributes is None:
                if (is_leaf and missing_leaf_ok) or missing_parents_ok:
                    leaf_attributes = None
                    continue
                raise FileNotFoundError(normalized)
            kind = _mode_kind(int(attributes.st_mode))
            if kind == "symlink":
                raise DirectoryBoundaryError("relative_path crosses a remote symbolic link")
            if not is_leaf and kind != "directory":
                raise DirectoryBoundaryError("relative_path crosses a non-directory entry")
            if kind == "other":
                raise DirectoryBoundaryError("relative_path addresses a non-regular remote entry")
            leaf_attributes = attributes
        return leaf_attributes

    def _tree_rows(
        self,
        client: Any,
        root: str,
        target: str,
        *,
        recursive: bool,
        limit: int,
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 100_000))
        rows: list[dict[str, Any]] = []
        pending = [target]
        while pending:
            parent = pending.pop()
            children = sorted(client.listdir_attr(parent), key=lambda item: str(item.filename).casefold())
            for listed in children:
                name = str(listed.filename)
                if name in {"", ".", ".."} or "/" in name or "\\" in name:
                    raise DirectoryBoundaryError("remote listing contains an invalid entry name")
                path = posixpath.join(parent, name)
                attributes = client.lstat(path)
                kind = _mode_kind(int(attributes.st_mode))
                if kind in {"symlink", "other"}:
                    raise DirectoryBoundaryError("remote directory tree contains an unsafe entry")
                relative = posixpath.relpath(path, root)
                if relative == ".." or relative.startswith("../"):
                    raise DirectoryBoundaryError("remote listing escaped the mounted directory")
                rows.append(
                    {
                        "path": relative,
                        "kind": kind,
                        "bytes": int(getattr(attributes, "st_size", 0) or 0) if kind == "file" else 0,
                        "modified_ns": _modified_ns(attributes),
                    }
                )
                if len(rows) >= bounded_limit:
                    return rows
                if recursive and kind == "directory":
                    pending.append(path)
        return rows

    def _usage_with_client(self, client: Any, capability: DirectoryCapability, root: str) -> tuple[int, int]:
        file_count = 0
        byte_count = 0
        pending = [root]
        while pending:
            parent = pending.pop()
            for listed in client.listdir_attr(parent):
                name = str(listed.filename)
                if name in {"", ".", ".."} or "/" in name or "\\" in name:
                    raise DirectoryBoundaryError("remote listing contains an invalid entry name")
                path = posixpath.join(parent, name)
                attributes = client.lstat(path)
                kind = _mode_kind(int(attributes.st_mode))
                if kind == "directory":
                    pending.append(path)
                elif kind == "file":
                    file_count += 1
                    byte_count += int(getattr(attributes, "st_size", 0) or 0)
                else:
                    raise DirectoryBoundaryError("remote directory tree contains an unsafe entry")
        return file_count, byte_count

    def _check_quota(
        self,
        client: Any,
        capability: DirectoryCapability,
        root: str,
        target_attributes: Any | None,
        new_size: int,
    ) -> None:
        file_count, byte_count = self._usage_with_client(client, capability, root)
        old_size = 0
        existing_file = False
        if target_attributes is not None:
            if _mode_kind(int(target_attributes.st_mode)) != "file":
                raise DirectoryBoundaryError("write destination must be a regular file or a new path")
            existing_file = True
            old_size = int(getattr(target_attributes, "st_size", 0) or 0)
        if file_count + (0 if existing_file else 1) > capability.max_files:
            raise DirectoryQuotaError("write would exceed the directory file quota")
        if byte_count - old_size + int(new_size) > capability.max_bytes:
            raise DirectoryQuotaError("write would exceed the directory byte quota")

    def _mkdir_with_client(
        self,
        client: Any,
        capability: DirectoryCapability,
        root: str,
        normalized: str,
        *,
        parents: bool,
        exist_ok: bool,
    ) -> bool:
        current = root
        parts = normalized.split("/") if normalized else []
        created = False
        for index, part in enumerate(parts):
            current = posixpath.join(current, part)
            attributes = self._lstat_or_none(client, current)
            leaf = index == len(parts) - 1
            if attributes is not None:
                if _mode_kind(int(attributes.st_mode)) != "directory":
                    raise DirectoryBoundaryError("mkdir path crosses a non-directory or symbolic link")
                if leaf and not exist_ok:
                    raise FileExistsError(normalized)
                continue
            if not parents and not leaf:
                raise FileNotFoundError(posixpath.dirname(normalized))
            client.mkdir(current, mode=0o700)
            created = True
        self._assert_safe_path(client, root, normalized)
        return created

    def health(self, capability: DirectoryCapability) -> OperationReceipt:
        started = utc_now()
        try:
            root, _, _ = _remote_path(capability, self.connector_id, "")
            with self._client(capability) as client:
                self._assert_safe_path(client, root, "")
        except Exception as exc:
            return self._receipt(
                capability,
                "health",
                ok=False,
                started_at=started,
                error_class=type(exc).__name__,
                error_message="remote connector health check failed",
                metadata={"ready": False, "root_verified": False},
            )
        return self._receipt(
            capability,
            "health",
            started_at=started,
            metadata={"ready": True, "root_verified": True, "symlink_policy": "deny", "version": self.version},
        )

    def list(
        self,
        capability: DirectoryCapability,
        relative_path: str = "",
        *,
        recursive: bool = False,
        limit: int = 5_000,
    ) -> OperationReceipt:
        root, target, normalized = _remote_path(capability, self.connector_id, relative_path)
        with self._client(capability) as client:
            attributes = self._assert_safe_path(client, root, normalized)
            if attributes is None or _mode_kind(int(attributes.st_mode)) != "directory":
                raise NotADirectoryError(normalized or ".")
            rows = self._tree_rows(client, root, target, recursive=recursive, limit=limit)
        return self._receipt(capability, "list", normalized, entries=rows, metadata={"recursive": bool(recursive)})

    def stat(self, capability: DirectoryCapability, relative_path: str = "") -> OperationReceipt:
        root, _, normalized = _remote_path(capability, self.connector_id, relative_path)
        with self._client(capability) as client:
            attributes = self._assert_safe_path(client, root, normalized)
        if attributes is None:
            raise FileNotFoundError(normalized)
        kind = _mode_kind(int(attributes.st_mode))
        return self._receipt(
            capability,
            "stat",
            normalized,
            metadata={
                "kind": kind,
                "bytes": int(getattr(attributes, "st_size", 0) or 0) if kind == "file" else 0,
                "modified_ns": _modified_ns(attributes),
            },
        )

    def read(
        self,
        capability: DirectoryCapability,
        relative_path: str,
        *,
        max_bytes: int | None = None,
    ) -> OperationReceipt:
        root, target, normalized = _remote_path(
            capability, self.connector_id, relative_path, allow_root=False
        )
        with self._client(capability) as client:
            attributes = self._assert_safe_path(client, root, normalized)
            if attributes is None or _mode_kind(int(attributes.st_mode)) != "file":
                raise FileNotFoundError(normalized)
            size = int(getattr(attributes, "st_size", 0) or 0)
            bounded = capability.max_bytes if max_bytes is None else min(capability.max_bytes, max(0, int(max_bytes)))
            if size > bounded:
                raise DirectoryQuotaError("read exceeds the requested maximum byte count")
            chunks: list[bytes] = []
            count = 0
            with _managed(client.open(target, "rb")) as handle:
                while True:
                    chunk = handle.read(self._chunk_size)
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise TypeError("SFTP reader must return bytes")
                    count += len(chunk)
                    if count > bounded:
                        raise DirectoryQuotaError("read exceeds the requested maximum byte count")
                    chunks.append(chunk)
        data = b"".join(chunks)
        return self._receipt(
            capability, "read", normalized, bytes_read=len(data), sha256=sha256(data).hexdigest(), data=data
        )

    def hash(self, capability: DirectoryCapability, relative_path: str) -> OperationReceipt:
        root, target, normalized = _remote_path(
            capability, self.connector_id, relative_path, allow_root=False
        )
        digest = sha256()
        count = 0
        with self._client(capability) as client:
            attributes = self._assert_safe_path(client, root, normalized)
            if attributes is None or _mode_kind(int(attributes.st_mode)) != "file":
                raise FileNotFoundError(normalized)
            with _managed(client.open(target, "rb")) as handle:
                while True:
                    chunk = handle.read(self._chunk_size)
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise TypeError("SFTP reader must return bytes")
                    count += len(chunk)
                    digest.update(chunk)
        return self._receipt(capability, "hash", normalized, bytes_read=count, sha256=digest.hexdigest())

    def mkdir(
        self,
        capability: DirectoryCapability,
        relative_path: str,
        *,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> OperationReceipt:
        root, _, normalized = _remote_path(capability, self.connector_id, relative_path, allow_root=False)
        with self._lock, self._client(capability) as client:
            created = self._mkdir_with_client(
                client, capability, root, normalized, parents=parents, exist_ok=exist_ok
            )
        return self._receipt(capability, "mkdir", normalized, metadata={"created": created})

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
        root, target, normalized = _remote_path(
            capability, self.connector_id, relative_path, allow_root=False
        )
        parent_normalized = normalized.rpartition("/")[0]
        temporary = posixpath.join(posixpath.dirname(target), f".{posixpath.basename(target)}.tmp-{uuid4().hex}")
        with self._lock, self._client(capability) as client:
            target_attributes = self._assert_safe_path(
                client, root, normalized, missing_leaf_ok=True, missing_parents_ok=create_parents
            )
            self._check_quota(client, capability, root, target_attributes, len(data))
            if parent_normalized:
                self._mkdir_with_client(
                    client,
                    capability,
                    root,
                    parent_normalized,
                    parents=create_parents,
                    exist_ok=True,
                )
            elif not create_parents:
                self._assert_safe_path(client, root, "")
            if not callable(getattr(client, "posix_rename", None)):
                raise RuntimeError("atomic POSIX rename is unavailable")
            try:
                with _managed(client.open(temporary, "wb")) as handle:
                    written = handle.write(data)
                    if written is not None and int(written) != len(data):
                        raise OSError("short remote write")
                    flush = getattr(handle, "flush", None)
                    if callable(flush):
                        flush()
                temporary_attributes = client.lstat(temporary)
                if _mode_kind(int(temporary_attributes.st_mode)) != "file" or int(temporary_attributes.st_size) != len(data):
                    raise OSError("remote temporary file verification failed")
                current_target = self._lstat_or_none(client, target)
                if current_target is not None and _mode_kind(int(current_target.st_mode)) != "file":
                    raise DirectoryBoundaryError("write destination became an unsafe entry")
                client.posix_rename(temporary, target)
                final_attributes = client.lstat(target)
                if _mode_kind(int(final_attributes.st_mode)) != "file" or int(final_attributes.st_size) != len(data):
                    raise OSError("remote atomic write verification failed")
            finally:
                if self._lstat_or_none(client, temporary) is not None:
                    client.remove(temporary)
        return self._receipt(capability, "write", normalized, bytes_written=len(data), sha256=digest)

    def copy(
        self,
        capability: DirectoryCapability,
        source_relative_path: str,
        destination_relative_path: str,
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
        source = normalize_relative_path(source_relative_path)
        destination = normalize_relative_path(destination_relative_path, allow_root=False)
        source_stat = self.stat(capability, source)
        if source_stat.metadata["kind"] == "file":
            return self.copy(capability, source, destination)
        if destination == source or (source and destination.startswith(f"{source}/")):
            raise DirectoryBoundaryError("sync destination cannot be the source or a child of the source")
        rows = self.list(capability, source, recursive=True, limit=100_000)
        source_prefix = f"{source}/" if source else ""
        copied: set[str] = set()
        bytes_written = 0
        for row in rows.entries:
            relative = str(row["path"])[len(source_prefix) :] if source_prefix else str(row["path"])
            target = "/".join(part for part in (destination, relative) if part)
            if row["kind"] == "directory":
                self.mkdir(capability, target)
            else:
                result = self.copy(capability, str(row["path"]), target)
                copied.add(relative)
                bytes_written += result.bytes_written
        deleted = 0
        if delete_extraneous:
            try:
                destination_rows = self.list(capability, destination, recursive=True, limit=100_000)
            except FileNotFoundError:
                destination_rows = self._receipt(capability, "list", destination, entries=[])
            for row in reversed(destination_rows.entries):
                relative = str(row["path"])[len(destination) :].lstrip("/")
                if row["kind"] == "file" and relative not in copied:
                    self.delete(capability, str(row["path"]))
                    deleted += 1
        return self._receipt(
            capability,
            "sync",
            destination,
            source_relative_path=source,
            destination_relative_path=destination,
            bytes_written=bytes_written,
            metadata={"files_copied": len(copied), "files_deleted": deleted},
        )

    def delete(
        self,
        capability: DirectoryCapability,
        relative_path: str,
        *,
        recursive: bool = False,
    ) -> OperationReceipt:
        root, target, normalized = _remote_path(
            capability, self.connector_id, relative_path, allow_root=False
        )
        with self._lock, self._client(capability) as client:
            attributes = self._assert_safe_path(client, root, normalized)
            if attributes is None:
                raise FileNotFoundError(normalized)
            kind = _mode_kind(int(attributes.st_mode))
            bytes_removed = 0
            if kind == "file":
                bytes_removed = int(getattr(attributes, "st_size", 0) or 0)
                client.remove(target)
            elif kind == "directory":
                rows = self._tree_rows(client, root, target, recursive=True, limit=100_000)
                if rows and not recursive:
                    raise OSError("directory is not empty")
                for row in reversed(rows):
                    child = posixpath.join(root, str(row["path"]))
                    if row["kind"] == "file":
                        bytes_removed += int(row["bytes"])
                        client.remove(child)
                    else:
                        client.rmdir(child)
                client.rmdir(target)
            else:
                raise DirectoryBoundaryError("delete target is not a regular file or directory")
        return self._receipt(
            capability,
            "delete",
            normalized,
            metadata={"deleted": True, "bytes_removed": bytes_removed},
        )

    def usage(self, capability: DirectoryCapability) -> tuple[int, int]:
        root = _remote_root(capability, self.connector_id)
        with self._client(capability) as client:
            self._assert_safe_path(client, root, "")
            return self._usage_with_client(client, capability, root)


class TransferTarget(Protocol):
    """Streaming destination contract consumed by :class:`HttpSourceConnector`."""

    def current_size(self, capability: DirectoryCapability, relative_path: str) -> int | None: ...

    def usage(self, capability: DirectoryCapability) -> tuple[int, int]: ...

    def write_stream(
        self,
        capability: DirectoryCapability,
        relative_path: str,
        chunks: Iterable[bytes],
        *,
        offset: int,
        expected_total_bytes: int | None,
        expected_sha256: str,
    ) -> OperationReceipt: ...


class HttpSourceConnector:
    """HTTP/HTTPS range downloader with injected opener and stream adapter.

    ``opener`` has the signature ``opener(url, headers={...})``.  ``target`` is
    a capability-aware streaming writer; it is responsible for staging partial
    bytes and atomically committing a completed object.
    """

    supported_operations = frozenset({"fetch"})

    def __init__(
        self,
        opener: Callable[..., Any],
        target: TransferTarget,
        *,
        stream: Callable[[Any, int], Iterable[bytes]] | None = None,
        connector_id: str = "http-source",
        version: str = "1.0.0",
        chunk_size: int = 1 << 20,
        allowed_hosts: Iterable[str],
    ) -> None:
        if not callable(opener):
            raise TypeError("opener must be callable")
        if target is None:
            raise TypeError("target must implement the streaming target contract")
        self.connector_id = str(connector_id).strip()
        self.version = str(version).strip()
        if not self.connector_id or not self.version:
            raise ValueError("connector_id and version must be non-empty")
        self._opener = opener
        self._target = target
        self._stream = stream or self._default_stream
        self._chunk_size = max(4096, int(chunk_size))
        self._allowed_hosts = _normalized_allowed_hosts(allowed_hosts)

    @staticmethod
    def _default_stream(response: Any, chunk_size: int) -> Iterable[bytes]:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                return
            yield chunk

    @staticmethod
    def _header(headers: Mapping[str, Any] | Any, name: str) -> str:
        if headers is None:
            return ""
        for key, value in dict(headers).items():
            if str(key).lower() == name.lower():
                return str(value)
        return ""

    @staticmethod
    def _status(response: Any) -> int:
        value = getattr(response, "status", None)
        if value is None:
            value = getattr(response, "status_code", None)
        if value is None and callable(getattr(response, "getcode", None)):
            value = response.getcode()
        return int(value)

    def _validate_source(self, source_url: str) -> str:
        return _validate_https_source(source_url, self._allowed_hosts)

    def _target_for(self, destination: DirectoryCapability) -> TransferTarget:
        target = self._target(destination) if callable(self._target) else self._target
        missing = [name for name in ("current_size", "usage", "write_stream") if not callable(getattr(target, name, None))]
        if missing:
            raise TypeError("target does not implement the streaming target contract")
        return target

    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            capability_id=self.connector_id,
            provider_id="http",
            version=self.version,
            operations=("fetch", "transfer_fetch"),
            input_schema={
                "type": "object",
                "properties": {
                    "source_url": {"type": "string", "format": "uri"},
                    "directory_id": {"type": "string"},
                    "relative_path": {"type": "string"},
                },
                "required": ["source_url", "directory_id", "relative_path"],
                "additionalProperties": False,
            },
            risk_class="bounded",
            idempotency="resumable",
        )

    def fetch(
        self,
        source_url: str,
        destination: DirectoryCapability,
        relative_path: str,
        *,
        resume: bool = True,
        expected_sha256: str = "",
    ) -> OperationReceipt:
        url = self._validate_source(source_url)
        normalized = normalize_relative_path(relative_path, allow_root=False)
        target = self._target_for(destination)
        current = target.current_size(destination, normalized)
        if current is not None and int(current) < 0:
            raise ValueError("target current_size must be non-negative or None")
        existing = int(current or 0)
        offset = existing if resume else 0
        files, byte_count = target.usage(destination)
        if files + (0 if current is not None else 1) > destination.max_files:
            raise DirectoryQuotaError("fetch would exceed the directory file quota")
        if byte_count < existing:
            raise ValueError("target usage is inconsistent with current_size")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        started = utc_now()
        with _managed(self._opener(url, headers=headers)) as response:
            final_url = response.geturl() if callable(getattr(response, "geturl", None)) else url
            final_url = final_url or url
            final_url = self._validate_source(final_url)
            status = self._status(response)
            response_headers = getattr(response, "headers", {})
            if status not in {200, 206}:
                raise OSError(f"HTTP source returned unsupported status {status}")
            resumed = bool(offset and status == 206)
            if offset and status == 200:
                offset = 0
            expected_total: int | None = None
            expected_response_bytes: int | None = None
            if status == 206:
                value = self._header(response_headers, "Content-Range")
                match = _CONTENT_RANGE.fullmatch(value.strip())
                if not match or int(match.group(1)) != offset:
                    raise OSError("HTTP Content-Range does not match the requested resume offset")
                response_end = int(match.group(2))
                if response_end < offset:
                    raise OSError("HTTP Content-Range has an invalid end offset")
                expected_response_bytes = response_end - offset + 1
                if match.group(3) != "*":
                    expected_total = int(match.group(3))
                    if response_end >= expected_total:
                        raise OSError("HTTP Content-Range exceeds the advertised total")
            else:
                content_length = self._header(response_headers, "Content-Length")
                if content_length:
                    expected_total = int(content_length)
                    expected_response_bytes = expected_total
            if expected_total is not None:
                projected = byte_count - existing + expected_total
                if projected > destination.max_bytes:
                    raise DirectoryQuotaError("fetch would exceed the directory byte quota")

            streamed = 0

            def bounded_chunks() -> Iterator[bytes]:
                nonlocal streamed
                for chunk in self._stream(response, self._chunk_size):
                    if not isinstance(chunk, bytes):
                        raise TypeError("HTTP stream must yield bytes")
                    if not chunk:
                        continue
                    if expected_response_bytes is not None and streamed + len(chunk) > expected_response_bytes:
                        raise OSError("HTTP stream exceeds the advertised response range")
                    streamed += len(chunk)
                    projected_size = offset + streamed
                    projected_usage = byte_count - existing + projected_size
                    if projected_usage > destination.max_bytes:
                        raise DirectoryQuotaError("fetch would exceed the directory byte quota")
                    yield chunk

            target_receipt = target.write_stream(
                destination,
                normalized,
                bounded_chunks(),
                offset=offset,
                expected_total_bytes=expected_total,
                expected_sha256=expected_sha256,
            )
        if not target_receipt.ok:
            return OperationReceipt(
                "fetch",
                self.connector_id,
                destination.directory_id,
                normalized,
                ok=False,
                started_at=started,
                exit_code=target_receipt.exit_code,
                error_class=target_receipt.error_class or "TransferTargetFailure",
                error_message="transfer target rejected the HTTP stream",
                metadata={"resumed": resumed, "range_requested": bool(headers)},
            )
        if expected_response_bytes is not None and streamed != expected_response_bytes:
            raise OSError("HTTP stream length does not match the advertised response range")
        if expected_total is not None and offset + streamed != expected_total:
            raise OSError("HTTP stream length does not match the advertised total")
        if expected_sha256 and target_receipt.sha256.lower() != expected_sha256.lower():
            raise ValueError("downloaded object does not match expected_sha256")
        return OperationReceipt(
            "fetch",
            self.connector_id,
            destination.directory_id,
            normalized,
            started_at=started,
            bytes_read=streamed,
            bytes_written=target_receipt.bytes_written,
            sha256=target_receipt.sha256,
            evidence_refs=list(target_receipt.evidence_refs),
            metadata={
                "source_scheme": urlsplit(url).scheme.lower(),
                "redirected": final_url != url,
                "resumed": resumed,
                "range_requested": bool(headers),
                "response_status": status,
                "resume_offset": offset,
                "expected_total_bytes": expected_total,
                "target_receipt_id": target_receipt.receipt_id,
            },
        )


_HPC_IDENTITY_FLAGS = (
    "job_container_verified",
    "designated_proxy_path_verified",
    "pinned_gateway_host_key_verified",
    "allocation_role_authenticated",
    "expected_host_uuid_match",
    "expected_gpu_uuid_match",
    "expected_gpu_model_and_memory_match",
    "allowed_remote_root_match",
    "identity_consistent",
    "job_profile_match",
)


class HpcExecutionConnector:
    """Fail-closed execution adapter requiring fresh complete 5/5 identity proof."""

    supported_operations = frozenset({"health", "execute", "status", "cancel"})

    def __init__(
        self,
        identity_verifier: Callable[[DirectoryCapability], Mapping[str, Any] | OperationReceipt],
        executor: Callable[..., Mapping[str, Any] | OperationReceipt],
        connector_id: str = "hpc-execution",
        *,
        version: str = "1.0.0",
    ) -> None:
        if not callable(identity_verifier) or not callable(executor):
            raise TypeError("identity_verifier and executor must be callable")
        self.connector_id = str(connector_id).strip()
        self.version = str(version).strip()
        if not self.connector_id or not self.version:
            raise ValueError("connector_id and version must be non-empty")
        self._identity_verifier = identity_verifier
        self._executor = executor

    def _check_capability(self, capability: DirectoryCapability, operation: str) -> None:
        _remote_root(capability, self.connector_id)
        if operation == "execute" and not capability.allows("execute"):
            raise CapabilityPermissionError(
                f"operation 'execute' is not granted for directory '{capability.directory_id}'"
            )

    @staticmethod
    def _identity_payload(value: Mapping[str, Any] | OperationReceipt) -> dict[str, Any]:
        if isinstance(value, OperationReceipt):
            return {"ok": value.ok, **dict(value.metadata)}
        if not isinstance(value, Mapping):
            raise TypeError("identity verifier must return a mapping or OperationReceipt")
        return dict(value)

    @staticmethod
    def _samples_complete(payload: Mapping[str, Any]) -> bool:
        samples = payload.get("samples")
        return (
            isinstance(samples, (list, tuple))
            and len(samples) == 5
            and all(
                (item is True) or (isinstance(item, Mapping) and item.get("complete") is True)
                for item in samples
            )
        )

    def health(self, capability: DirectoryCapability) -> OperationReceipt:
        started = utc_now()
        try:
            self._check_capability(capability, "health")
            payload = self._identity_payload(self._identity_verifier(capability))
        except Exception as exc:
            return OperationReceipt(
                "health",
                self.connector_id,
                capability.directory_id,
                ok=False,
                started_at=started,
                error_class=type(exc).__name__,
                error_message="HPC identity verifier failed before execution",
                metadata={"identity_complete": False, "samples_requested": 5, "samples_passed": 0},
            )
        flag_values = {name: payload.get(name) is True for name in _HPC_IDENTITY_FLAGS}
        samples_requested = payload.get("samples_requested")
        samples_passed = payload.get("samples_passed")
        sample_counts_valid = (
            isinstance(samples_requested, int)
            and not isinstance(samples_requested, bool)
            and samples_requested == 5
            and isinstance(samples_passed, int)
            and not isinstance(samples_passed, bool)
            and samples_passed == 5
        )
        signals_sent = payload.get("signals_sent")
        signals_valid = (
            isinstance(signals_sent, int) and not isinstance(signals_sent, bool) and signals_sent == 0
        )
        complete = (
            payload.get("ok") is True
            and all(flag_values.values())
            and sample_counts_valid
            and self._samples_complete(payload)
            and payload.get("read_only") is True
            and signals_valid
            and payload.get("other_processes_modified") is False
        )
        missing_or_false = sorted(name for name, value in flag_values.items() if not value)
        if not sample_counts_valid or not self._samples_complete(payload):
            missing_or_false.append("samples_5_of_5_complete")
        if payload.get("read_only") is not True:
            missing_or_false.append("read_only")
        if not signals_valid:
            missing_or_false.append("signals_sent_zero")
        if payload.get("other_processes_modified") is not False:
            missing_or_false.append("other_processes_modified_false")
        metadata = {
            **flag_values,
            "identity_complete": complete,
            "samples_requested": samples_requested,
            "samples_passed": samples_passed,
            "samples_complete": self._samples_complete(payload),
            "read_only": payload.get("read_only") is True,
            "signals_sent": signals_sent
            if isinstance(signals_sent, int) and not isinstance(signals_sent, bool)
            else None,
            "other_processes_modified": payload.get("other_processes_modified")
            if isinstance(payload.get("other_processes_modified"), bool)
            else None,
            "missing_or_false": missing_or_false,
        }
        return OperationReceipt(
            "health",
            self.connector_id,
            capability.directory_id,
            ok=complete,
            started_at=started,
            exit_code=0 if complete else None,
            error_class="" if complete else "HpcIdentityEvidenceIncomplete",
            error_message="" if complete else "complete current 5/5 HPC identity evidence is required",
            metadata=metadata,
        )

    @staticmethod
    def _safe_result_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "job_state",
            "signals_sent",
            "other_processes_modified",
            "artifacts_count",
            "worker_alive",
            "cancelled",
        }
        return {key: payload[key] for key in allowed if key in payload}

    def _execute_action(
        self,
        action: str,
        capability: DirectoryCapability,
        relative_path: str,
        payload: Mapping[str, Any],
    ) -> OperationReceipt:
        self._check_capability(capability, "execute" if action == "execute" else "health")
        normalized = normalize_relative_path(relative_path, allow_root=True)
        identity = self.health(capability)
        if not identity.ok:
            return OperationReceipt(
                action,
                self.connector_id,
                capability.directory_id,
                normalized,
                ok=False,
                exit_code=None,
                error_class=identity.error_class,
                error_message=identity.error_message,
                metadata={
                    "identity_receipt_id": identity.receipt_id,
                    "identity_complete": False,
                    "samples_passed": identity.metadata.get("samples_passed"),
                },
            )
        result = self._executor(
            operation=action,
            directory_id=capability.directory_id,
            relative_path=normalized,
            payload=dict(payload),
            identity_receipt=identity.to_dict(),
        )
        if isinstance(result, OperationReceipt):
            raw: dict[str, Any] = result.to_dict()
            raw_metadata = dict(result.metadata)
        elif isinstance(result, Mapping):
            raw = dict(result)
            raw_metadata = dict(raw.get("metadata") or {})
        else:
            raise TypeError("executor must return a mapping or OperationReceipt")
        ok = raw.get("ok") is True
        digest = str(raw.get("sha256") or "")
        if digest and (len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest)):
            raise ValueError("executor returned an invalid SHA-256 digest")
        return OperationReceipt(
            action,
            self.connector_id,
            capability.directory_id,
            normalized,
            ok=ok,
            exit_code=int(raw["exit_code"]) if raw.get("exit_code") is not None else (0 if ok else None),
            bytes_read=int(raw.get("bytes_read") or 0),
            bytes_written=int(raw.get("bytes_written") or 0),
            sha256=digest,
            evidence_refs=[str(value) for value in raw.get("evidence_refs") or ()],
            error_class="" if ok else str(raw.get("error_class") or "HpcExecutionFailed"),
            error_message="" if ok else "managed HPC operation failed",
            metadata={
                "identity_receipt_id": identity.receipt_id,
                "identity_complete": True,
                "samples_passed": 5,
                **self._safe_result_metadata({**raw, **raw_metadata}),
            },
        )

    def execute(
        self,
        capability: DirectoryCapability,
        relative_path: str,
        job_spec: Mapping[str, Any],
    ) -> OperationReceipt:
        if not isinstance(job_spec, Mapping):
            raise TypeError("job_spec must be a structured mapping")
        return self._execute_action("execute", capability, relative_path, job_spec)

    job_execute = execute

    def status(self, capability: DirectoryCapability, job_ref: str) -> OperationReceipt:
        value = str(job_ref).strip()
        if not value:
            raise ValueError("job_ref must be non-empty")
        return self._execute_action("status", capability, "", {"job_ref": value})

    job_status = status

    def cancel(self, capability: DirectoryCapability, job_ref: str, *, approved: bool = False) -> OperationReceipt:
        if not approved:
            raise ApprovalRequiredError("job cancellation requires explicit approved=True")
        value = str(job_ref).strip()
        if not value:
            raise ValueError("job_ref must be non-empty")
        return self._execute_action("cancel", capability, "", {"job_ref": value})

    job_cancel = cancel


class DockerSftpFixtureConnector(SftpDirectoryConnector):
    """SFTP-contract fixture for isolated Docker canaries.

    It deliberately contains no Docker lifecycle or network code.  Tests or a
    deployment harness inject the already-scoped fixture client factory.
    """

    def __init__(
        self,
        client_factory: Callable[[DirectoryCapability], Any],
        connector_id: str = "docker-sftp-fixture",
        *,
        version: str = "1.0.0",
        chunk_size: int = 1 << 20,
    ) -> None:
        super().__init__(client_factory, connector_id, version=version, chunk_size=chunk_size)

    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            capability_id=self.connector_id,
            provider_id="sftp",
            version=self.version,
            operations=tuple(sorted(self.supported_operations)),
            input_schema={
                "type": "object",
                "properties": {
                    "directory_id": {"type": "string"},
                    "relative_path": {"type": "string"},
                },
                "required": ["directory_id", "relative_path"],
                "additionalProperties": False,
            },
            risk_class="fixture",
            idempotency="operation-dependent",
            health_evidence={
                "protocol": "sftp",
                "contract": "SftpDirectoryConnector",
                "docker_lifecycle_managed": False,
                "network_client_injected": True,
            },
        )


__all__ = [
    "DockerSftpFixtureConnector",
    "HpcExecutionConnector",
    "HttpSourceConnector",
    "SftpDirectoryConnector",
    "TransferTarget",
]
