from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import pytest

from evomind_runtime.capabilities import (
    ApprovalRequiredError,
    CapabilityDescriptor,
    CapabilityExpiredError,
    CapabilityPermissionError,
    DirectoryBoundaryError,
    DirectoryCapability,
    DirectoryQuotaError,
)
from evomind_runtime import connectors as connector_module
from evomind_runtime.connectors import LocalDirectoryConnector, MemoryDirectoryConnector
from evomind_runtime.directory_broker import DirectoryBroker


ALL_OPERATIONS = ("list", "stat", "read", "hash", "write", "mkdir", "copy", "sync", "delete")


def capability(directory_id: str, connector_id: str, root: str, **values: object) -> DirectoryCapability:
    return DirectoryCapability(
        directory_id=directory_id,
        connector_id=connector_id,
        opaque_root_ref=root,
        operations=values.pop("operations", ALL_OPERATIONS),
        **values,
    )


def test_capability_descriptor_is_stable_and_directory_root_is_opaque() -> None:
    descriptor = CapabilityDescriptor("workspace", "local", "1.0", ("read", "write"))
    assert descriptor.fingerprint() == descriptor.fingerprint()
    mount = capability("workspace", "local", "C:/secret-root")
    assert "opaque_root_ref" not in mount.to_dict()
    assert mount.to_dict(include_opaque_root=True)["opaque_root_ref"] == "C:/secret-root"


def test_mount_rejects_connector_version_drift() -> None:
    broker = DirectoryBroker([MemoryDirectoryConnector(version="2.0.0")])
    with pytest.raises(ValueError, match="version mismatch"):
        broker.mount(
            capability(
                "versioned",
                "memory",
                "opaque",
                connector_version="1.0.0",
            )
        )


def test_memory_connector_roundtrip_permissions_quota_and_delete_approval() -> None:
    broker = DirectoryBroker([MemoryDirectoryConnector()])
    broker.mount(capability("memory-data", "memory", "opaque-memory", max_files=2, max_bytes=8))

    written = broker.write_atomic("memory-data", "nested/a.txt", b"abc")
    assert written.sha256 == sha256(b"abc").hexdigest()
    assert broker.read("memory-data", "nested/a.txt").data == b"abc"
    assert broker.hash("memory-data", "nested/a.txt").sha256 == written.sha256
    assert broker.stat("memory-data", "nested/a.txt").metadata["bytes"] == 3
    assert [row["path"] for row in broker.list("memory-data", "", recursive=True).entries] == [
        "nested",
        "nested/a.txt",
    ]

    broker.copy("memory-data", "nested/a.txt", "memory-data", "nested/b.txt")
    with pytest.raises(DirectoryQuotaError):
        broker.write_atomic("memory-data", "nested/c.txt", b"x")
    with pytest.raises(DirectoryQuotaError):
        broker.write_atomic("memory-data", "nested/b.txt", b"123456")
    with pytest.raises(ApprovalRequiredError):
        broker.delete("memory-data", "nested/a.txt")
    assert broker.delete("memory-data", "nested/a.txt", approved=True).metadata["deleted"] is True

    read_only = capability("read-only", "memory", "other", operations=("read",))
    broker.mount(read_only)
    with pytest.raises(FileNotFoundError):
        broker.read("read-only", "nested/b.txt")
    with pytest.raises(CapabilityPermissionError):
        broker.write_atomic("read-only", "x", b"x")


def test_local_connector_uses_directory_id_relative_paths_and_atomic_operations(tmp_path: Path) -> None:
    root = tmp_path / "mount"
    root.mkdir()
    broker = DirectoryBroker([LocalDirectoryConnector(allowed_roots=[tmp_path])])
    broker.mount(capability("workspace", "local", str(root), max_files=10, max_bytes=100))

    receipt = broker.verify_roundtrip("workspace", "canary/probe.bin", b"probe")
    assert receipt.ok is True
    assert receipt.status == "completed" and receipt.exit_code == 0
    assert receipt.to_dict()["data_present"] is False
    assert not any(path.name.startswith(".probe.bin.tmp-") for path in (root / "canary").iterdir())

    broker.mkdir("workspace", "source")
    broker.write_atomic("workspace", "source/a.txt", "alpha")
    broker.sync("workspace", "source", "workspace", "destination")
    assert (root / "destination" / "a.txt").read_text(encoding="utf-8") == "alpha"

    with pytest.raises(DirectoryBoundaryError):
        broker.read("workspace", str(tmp_path / "outside.txt"))
    with pytest.raises(DirectoryBoundaryError):
        broker.write_atomic("workspace", "../escape.txt", b"no")
    with pytest.raises(DirectoryBoundaryError):
        broker.write_atomic("workspace", "C:\\escape.txt", b"no")
    with pytest.raises(DirectoryBoundaryError):
        broker.write_atomic("workspace", "data.txt:alternate", b"no")


def test_local_connector_rejects_symlink_or_reparse_escape(tmp_path: Path) -> None:
    root = tmp_path / "mount"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic link creation is unavailable on this host")

    broker = DirectoryBroker([LocalDirectoryConnector()])
    broker.mount(capability("workspace", "local", str(root)))
    with pytest.raises(DirectoryBoundaryError):
        broker.write_atomic("workspace", "link/escape.txt", b"no")
    with pytest.raises(DirectoryBoundaryError):
        broker.list("workspace", recursive=True)


def test_local_connector_rejects_reparse_point_without_following_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "mount"
    simulated_reparse = root / "junction"
    simulated_reparse.mkdir(parents=True)
    real_check = connector_module._is_link_or_reparse
    monkeypatch.setattr(
        connector_module,
        "_is_link_or_reparse",
        lambda path: Path(path) == simulated_reparse or real_check(Path(path)),
    )

    broker = DirectoryBroker([LocalDirectoryConnector()])
    broker.mount(capability("workspace", "local", str(root)))
    with pytest.raises(DirectoryBoundaryError):
        broker.write_atomic("workspace", "junction/escape.txt", b"no")


def test_expired_mount_fails_closed_and_discovery_reports_health() -> None:
    broker = DirectoryBroker([MemoryDirectoryConnector()])
    active = capability("active", "memory", "active-root", operations=("read", "list"))
    broker.mount(active)
    descriptors = broker.discover(operation="read")
    assert [item.capability_id for item in descriptors] == ["active"]
    assert descriptors[0].health_evidence["ok"] is True

    expired_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    expired = capability("expired", "memory", "expired-root", expires_at=expired_at)
    with pytest.raises(CapabilityExpiredError):
        broker.mount(expired)


def test_cross_connector_copy_preserves_hash_without_serializing_payload(tmp_path: Path) -> None:
    root = tmp_path / "mount"
    root.mkdir()
    broker = DirectoryBroker([MemoryDirectoryConnector(), LocalDirectoryConnector()])
    broker.mount(capability("source", "memory", "source-root"))
    broker.mount(capability("destination", "local", str(root)))
    broker.write_atomic("source", "data.bin", b"cross-connector")

    receipt = broker.copy("source", "data.bin", "destination", "copied/data.bin")
    assert receipt.sha256 == sha256(b"cross-connector").hexdigest()
    assert (root / "copied" / "data.bin").read_bytes() == b"cross-connector"
    assert "data" not in receipt.to_dict()
