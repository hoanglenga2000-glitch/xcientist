from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import posixpath
import stat
from typing import Any, Iterable

import pytest

from evomind_runtime.capabilities import (
    ApprovalRequiredError,
    DirectoryBoundaryError,
    DirectoryCapability,
    DirectoryQuotaError,
    OperationReceipt,
)
from evomind_runtime.directory_broker import DirectoryBroker
from evomind_runtime.connectors import LocalDirectoryConnector
from evomind_runtime.remote_connectors import (
    DockerSftpFixtureConnector,
    HpcExecutionConnector,
    HttpSourceConnector,
    SftpDirectoryConnector,
)


ALL_DIRECTORY_OPERATIONS = ("list", "stat", "read", "hash", "write", "mkdir", "copy", "sync", "delete")


def directory_capability(
    directory_id: str,
    connector_id: str,
    root: str,
    *,
    operations: tuple[str, ...] = ALL_DIRECTORY_OPERATIONS,
    max_files: int = 100,
    max_bytes: int = 1_000_000,
) -> DirectoryCapability:
    return DirectoryCapability(
        directory_id=directory_id,
        connector_id=connector_id,
        opaque_root_ref=root,
        operations=operations,
        max_files=max_files,
        max_bytes=max_bytes,
    )


@dataclass
class FakeSftpAttributes:
    filename: str = ""
    st_mode: int = stat.S_IFREG | 0o600
    st_size: int = 0
    st_mtime: int = 1


class _CommittedBytesIO(io.BytesIO):
    def __init__(self, commit: Any) -> None:
        super().__init__()
        self._commit = commit

    def close(self) -> None:
        if not self.closed:
            self._commit(self.getvalue())
        super().close()


class FakeSftpClient:
    def __init__(self, root: str = "/srv/data") -> None:
        self.root = root
        self.directories = {root}
        self.files: dict[str, bytes] = {}
        self.symlinks: set[str] = set()
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def lstat(self, path: str) -> FakeSftpAttributes:
        if path in self.symlinks:
            return FakeSftpAttributes(posixpath.basename(path), stat.S_IFLNK | 0o777)
        if path in self.directories:
            return FakeSftpAttributes(posixpath.basename(path), stat.S_IFDIR | 0o700)
        if path in self.files:
            return FakeSftpAttributes(posixpath.basename(path), stat.S_IFREG | 0o600, len(self.files[path]))
        raise FileNotFoundError(path)

    def listdir_attr(self, parent: str) -> list[FakeSftpAttributes]:
        if parent not in self.directories:
            raise NotADirectoryError(parent)
        prefix = f"{parent.rstrip('/')}/"
        names: set[str] = set()
        for path in self.directories | set(self.files) | self.symlinks:
            if path == parent or not path.startswith(prefix):
                continue
            remainder = path[len(prefix) :]
            if remainder and "/" not in remainder:
                names.add(remainder)
        return [self.lstat(posixpath.join(parent, name)) for name in sorted(names)]

    def open(self, path: str, mode: str) -> io.BytesIO:
        if mode == "rb":
            if path not in self.files:
                raise FileNotFoundError(path)
            return io.BytesIO(self.files[path])
        if mode == "wb":
            parent = posixpath.dirname(path)
            if parent not in self.directories:
                raise FileNotFoundError(parent)
            return _CommittedBytesIO(lambda value: self.files.__setitem__(path, value))
        raise ValueError(mode)

    def mkdir(self, path: str, mode: int = 0o700) -> None:
        del mode
        parent = posixpath.dirname(path)
        if parent not in self.directories:
            raise FileNotFoundError(parent)
        if path in self.files or path in self.symlinks:
            raise FileExistsError(path)
        self.directories.add(path)

    def posix_rename(self, source: str, destination: str) -> None:
        if source not in self.files:
            raise FileNotFoundError(source)
        if destination in self.directories or destination in self.symlinks:
            raise OSError("unsafe destination")
        self.files[destination] = self.files.pop(source)

    def remove(self, path: str) -> None:
        del self.files[path]

    def rmdir(self, path: str) -> None:
        prefix = f"{path.rstrip('/')}/"
        if any(item.startswith(prefix) for item in self.directories | set(self.files)):
            raise OSError("directory is not empty")
        self.directories.remove(path)


def test_sftp_connector_roundtrip_quota_boundary_and_symlink_fail_closed() -> None:
    client = FakeSftpClient()
    connector = SftpDirectoryConnector(lambda capability: client)
    capability = directory_capability("remote-data", "sftp", "/srv/data", max_files=2, max_bytes=8)
    broker = DirectoryBroker([connector])
    broker.mount(capability)

    written = broker.write_atomic("remote-data", "nested/a.bin", b"abc")
    assert written.ok is True
    assert written.sha256 == sha256(b"abc").hexdigest()
    assert broker.read("remote-data", "nested/a.bin").data == b"abc"
    assert broker.hash("remote-data", "nested/a.bin").sha256 == written.sha256
    assert broker.stat("remote-data", "nested/a.bin").metadata["bytes"] == 3

    broker.copy("remote-data", "nested/a.bin", "remote-data", "nested/b.bin")
    assert connector.usage(capability) == (2, 6)
    with pytest.raises(DirectoryQuotaError):
        broker.write_atomic("remote-data", "nested/c.bin", b"x")
    with pytest.raises(DirectoryQuotaError):
        broker.write_atomic("remote-data", "nested/b.bin", b"123456")

    client.symlinks.add("/srv/data/link")
    with pytest.raises(DirectoryBoundaryError):
        broker.read("remote-data", "link/escape.bin")
    with pytest.raises(DirectoryBoundaryError):
        broker.write_atomic("remote-data", "../escape.bin", b"x")
    with pytest.raises(DirectoryBoundaryError):
        broker.write_atomic("remote-data", "/absolute.bin", b"x")

    serialized = written.to_dict()
    assert "/srv/data" not in repr(serialized)
    assert serialized["directory_id"] == "remote-data"
    assert serialized["relative_path"] == "nested/a.bin"


def test_sftp_health_and_listing_reject_unsafe_remote_entries() -> None:
    client = FakeSftpClient()
    connector = SftpDirectoryConnector(lambda capability: client)
    capability = directory_capability("remote", "sftp", "/srv/data")
    assert connector.health(capability).metadata["root_verified"] is True

    client.directories.add("/srv/data/regular")
    client.files["/srv/data/regular/data.txt"] = b"data"
    rows = connector.list(capability, "", recursive=True).entries
    assert [(row["path"], row["kind"]) for row in rows] == [
        ("regular", "directory"),
        ("regular/data.txt", "file"),
    ]

    client.symlinks.add("/srv/data/unsafe")
    with pytest.raises(DirectoryBoundaryError):
        connector.list(capability, recursive=True)

    bad_root = directory_capability("bad", "sftp", "relative/root")
    receipt = connector.health(bad_root)
    assert receipt.ok is False
    assert receipt.metadata["root_verified"] is False
    assert "relative/root" not in receipt.error_message


class FakeHttpResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int, headers: dict[str, str], final_url: str = "") -> None:
        super().__init__(body)
        self.status = status
        self.headers = headers
        self._final_url = final_url

    def geturl(self) -> str:
        return self._final_url


class FakeTransferTarget:
    def __init__(self) -> None:
        self.files: dict[tuple[str, str], bytes] = {}
        self.calls: list[dict[str, Any]] = []

    def current_size(self, capability: DirectoryCapability, relative_path: str) -> int | None:
        value = self.files.get((capability.directory_id, relative_path))
        return None if value is None else len(value)

    def usage(self, capability: DirectoryCapability) -> tuple[int, int]:
        values = [value for (directory_id, _), value in self.files.items() if directory_id == capability.directory_id]
        return len(values), sum(len(value) for value in values)

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
        key = (capability.directory_id, relative_path)
        previous = self.files.get(key, b"")
        if offset > len(previous):
            raise OSError("invalid resume offset")
        appended = b"".join(chunks)
        final = previous[:offset] + appended
        if expected_total_bytes is not None and len(final) != expected_total_bytes:
            raise OSError("unexpected final size")
        digest = sha256(final).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise ValueError("hash mismatch")
        self.files[key] = final
        self.calls.append({"directory_id": capability.directory_id, "path": relative_path, "offset": offset})
        return OperationReceipt(
            "write_stream",
            "fake-target",
            capability.directory_id,
            relative_path,
            bytes_written=len(appended),
            sha256=digest,
        )


def test_http_connector_resumes_with_range_and_streams_to_directory_target() -> None:
    target = FakeTransferTarget()
    destination = directory_capability("dataset", "fake-target", "opaque", max_files=1, max_bytes=20)
    target.files[("dataset", "archive.bin")] = b"abcd"
    opener_calls: list[dict[str, Any]] = []

    def opener(url: str, *, headers: dict[str, str]) -> FakeHttpResponse:
        opener_calls.append({"url": url, "headers": dict(headers)})
        return FakeHttpResponse(b"efghij", 206, {"Content-Range": "bytes 4-9/10"})

    stream_calls: list[int] = []

    def stream(response: FakeHttpResponse, chunk_size: int) -> Iterable[bytes]:
        stream_calls.append(chunk_size)
        while True:
            chunk = response.read(2)
            if not chunk:
                break
            yield chunk

    connector = HttpSourceConnector(
        opener, target, stream=stream, chunk_size=4096, allowed_hosts=("example.invalid",)
    )
    expected = sha256(b"abcdefghij").hexdigest()
    receipt = connector.fetch(
        "https://example.invalid/data/archive.bin?opaque=1",
        destination,
        "archive.bin",
        expected_sha256=expected,
    )

    assert opener_calls[0]["headers"] == {"Range": "bytes=4-"}
    assert stream_calls == [4096]
    assert target.files[("dataset", "archive.bin")] == b"abcdefghij"
    assert target.calls == [{"directory_id": "dataset", "path": "archive.bin", "offset": 4}]
    assert receipt.ok is True and receipt.sha256 == expected
    assert receipt.metadata["resumed"] is True
    assert receipt.metadata["response_status"] == 206
    assert "example.invalid" not in repr(receipt.to_dict())
    assert "opaque" not in repr(receipt.to_dict())


def test_http_connector_restarts_when_range_is_ignored_and_enforces_quota() -> None:
    target = FakeTransferTarget()
    destination = directory_capability("dataset", "fake-target", "opaque", max_files=1, max_bytes=10)
    target.files[("dataset", "data.bin")] = b"old"

    def opener(url: str, *, headers: dict[str, str]) -> FakeHttpResponse:
        del url
        assert headers == {"Range": "bytes=3-"}
        return FakeHttpResponse(b"new-data", 200, {"Content-Length": "8"})

    receipt = HttpSourceConnector(opener, target, allowed_hosts=("example.invalid",)).fetch(
        "https://example.invalid/data", destination, "data.bin"
    )
    assert target.files[("dataset", "data.bin")] == b"new-data"
    assert target.calls[-1]["offset"] == 0
    assert receipt.metadata["resumed"] is False

    too_small = directory_capability("small", "fake-target", "opaque", max_files=1, max_bytes=5)
    with pytest.raises(DirectoryQuotaError):
        HttpSourceConnector(
            lambda url, *, headers: FakeHttpResponse(b"123456", 200, {"Content-Length": "6"}),
            target,
            allowed_hosts=("example.invalid",),
        ).fetch("https://example.invalid/large", too_small, "large.bin")
    assert ("small", "large.bin") not in target.files


def test_http_connector_rejects_inconsistent_content_range_before_success_receipt() -> None:
    target = FakeTransferTarget()
    destination = directory_capability("dataset", "fake-target", "opaque", max_files=1, max_bytes=20)
    target.files[("dataset", "data.bin")] = b"abcd"

    connector = HttpSourceConnector(
        lambda url, *, headers: FakeHttpResponse(b"efghij", 206, {"Content-Range": "bytes 4-8/10"}),
        target,
        allowed_hosts=("example.invalid",),
    )
    with pytest.raises(OSError, match="response range"):
        connector.fetch("https://example.invalid/data", destination, "data.bin")
    assert target.files[("dataset", "data.bin")] == b"abcd"


def test_http_connector_rejects_downgrades_off_allowlist_hosts_and_redirects() -> None:
    target = FakeTransferTarget()
    destination = directory_capability("dataset", "fake-target", "opaque", max_files=1, max_bytes=20)
    opener_calls: list[str] = []

    def opener(url: str, *, headers: dict[str, str]) -> FakeHttpResponse:
        del headers
        opener_calls.append(url)
        return FakeHttpResponse(
            b"ok",
            200,
            {"Content-Length": "2"},
            final_url="https://example.invalid/redirected",
        )

    connector = HttpSourceConnector(opener, target, allowed_hosts=("raw.githubusercontent.com",))
    with pytest.raises(ValueError, match="absolute HTTPS"):
        connector.fetch("http://raw.githubusercontent.com/data", destination, "data.bin")
    with pytest.raises(ValueError, match="not allowlisted"):
        connector.fetch("https://example.invalid/data", destination, "data.bin")
    with pytest.raises(ValueError, match="not allowlisted"):
        connector.fetch("https://raw.githubusercontent.com/data", destination, "data.bin")
    assert opener_calls == ["https://raw.githubusercontent.com/data"]
    assert target.files == {}


def test_local_directory_connector_is_an_atomic_resumable_transfer_target(tmp_path) -> None:
    connector = LocalDirectoryConnector("local-target", allowed_roots=(tmp_path,))
    destination = directory_capability(
        "dataset", "local-target", str(tmp_path), max_files=2, max_bytes=20
    )
    first = connector.write_stream(
        destination,
        "data.bin",
        [b"ab", b"cd"],
        offset=0,
        expected_total_bytes=4,
        expected_sha256=sha256(b"abcd").hexdigest(),
    )
    assert first.bytes_written == 4
    assert connector.current_size(destination, "data.bin") == 4
    resumed = connector.write_stream(
        destination,
        "data.bin",
        [b"ef"],
        offset=4,
        expected_total_bytes=6,
        expected_sha256=sha256(b"abcdef").hexdigest(),
    )
    assert resumed.bytes_written == 2
    assert (tmp_path / "data.bin").read_bytes() == b"abcdef"
    with pytest.raises(ValueError, match="expected_sha256"):
        connector.write_stream(
            destination,
            "data.bin",
            [b"bad"],
            offset=0,
            expected_total_bytes=3,
            expected_sha256="0" * 64,
        )
    assert (tmp_path / "data.bin").read_bytes() == b"abcdef"


def complete_identity() -> dict[str, Any]:
    return {
        "ok": True,
        "job_container_verified": True,
        "designated_proxy_path_verified": True,
        "pinned_gateway_host_key_verified": True,
        "allocation_role_authenticated": True,
        "expected_host_uuid_match": True,
        "expected_gpu_uuid_match": True,
        "expected_gpu_model_and_memory_match": True,
        "allowed_remote_root_match": True,
        "identity_consistent": True,
        "job_profile_match": True,
        "samples_requested": 5,
        "samples_passed": 5,
        "samples": [{"complete": True} for _ in range(5)],
        "read_only": True,
        "signals_sent": 0,
        "other_processes_modified": False,
    }


def test_hpc_connector_is_fail_closed_until_complete_current_5_of_5() -> None:
    verifier_calls: list[str] = []
    executor_calls: list[dict[str, Any]] = []

    def verifier(capability: DirectoryCapability) -> dict[str, Any]:
        verifier_calls.append(capability.directory_id)
        value = complete_identity()
        value["samples_passed"] = 4
        value["samples"][-1] = {"complete": False}
        return value

    def executor(**values: Any) -> dict[str, Any]:
        executor_calls.append(values)
        return {"ok": True, "exit_code": 0}

    connector = HpcExecutionConnector(verifier, executor)
    capability = directory_capability(
        "hpc-data", "hpc-execution", "/managed/root", operations=("execute",)
    )
    receipt = connector.execute(capability, "jobs/smoke", {"entrypoint": "train"})

    assert verifier_calls == ["hpc-data"]
    assert executor_calls == []
    assert receipt.ok is False
    assert receipt.error_class == "HpcIdentityEvidenceIncomplete"
    assert receipt.metadata["identity_complete"] is False
    assert "/managed/root" not in repr(receipt.to_dict())

    boolean_signal = complete_identity()
    boolean_signal["signals_sent"] = False
    boolean_connector = HpcExecutionConnector(lambda capability: boolean_signal, executor)
    assert boolean_connector.health(capability).ok is False
    assert executor_calls == []


def test_hpc_connector_executes_only_structured_relative_jobs_after_5_of_5() -> None:
    executor_calls: list[dict[str, Any]] = []

    def executor(**values: Any) -> dict[str, Any]:
        executor_calls.append(values)
        return {
            "ok": True,
            "exit_code": 0,
            "bytes_written": 7,
            "sha256": sha256(b"receipt").hexdigest(),
            "metadata": {
                "job_state": "completed",
                "signals_sent": 0,
                "other_processes_modified": False,
                "unsafe_extra": "/managed/root",
            },
        }

    connector = HpcExecutionConnector(lambda capability: complete_identity(), executor)
    capability = directory_capability(
        "hpc-data", "hpc-execution", "/managed/root", operations=("execute",)
    )
    receipt = connector.execute(capability, "jobs/./smoke", {"entrypoint": "train", "args": ["--smoke"]})

    assert receipt.ok is True
    assert receipt.relative_path == "jobs/smoke"
    assert receipt.metadata["identity_complete"] is True
    assert receipt.metadata["samples_passed"] == 5
    assert receipt.metadata["job_state"] == "completed"
    assert "unsafe_extra" not in receipt.metadata
    assert len(executor_calls) == 1
    assert executor_calls[0]["directory_id"] == "hpc-data"
    assert executor_calls[0]["relative_path"] == "jobs/smoke"
    assert "/managed/root" not in repr(executor_calls[0])

    before = len(executor_calls)
    with pytest.raises(DirectoryBoundaryError):
        connector.execute(capability, "/outside", {"entrypoint": "no"})
    assert len(executor_calls) == before
    with pytest.raises(ApprovalRequiredError):
        connector.cancel(capability, "job-ref")


def test_docker_sftp_fixture_is_protocol_compatible_without_docker_lifecycle() -> None:
    fixture = DockerSftpFixtureConnector(lambda capability: FakeSftpClient())
    descriptor = fixture.descriptor()
    assert isinstance(fixture, SftpDirectoryConnector)
    assert descriptor.provider_id == "sftp"
    assert descriptor.health_evidence == {
        "protocol": "sftp",
        "contract": "SftpDirectoryConnector",
        "docker_lifecycle_managed": False,
        "network_client_injected": True,
    }
