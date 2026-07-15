from __future__ import annotations

import hashlib
import stat
import zipfile
from pathlib import Path

import pytest

from scripts.extract_capability_evidence_bundle import BundleError, extract_bundle


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_extracts_hash_pinned_regular_evidence_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "evidence.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.json", "{}")
        archive.writestr("artifacts/raw-results.json", "{}")
    destination = tmp_path / "out"
    result = extract_bundle(bundle, destination, expected_sha256=digest(bundle))
    assert result["files"] == 2
    assert (destination / "report.json").read_text() == "{}"


@pytest.mark.parametrize("member", ["../escape.json", "/absolute.json", "C:/drive.json", "dir//file.json"])
def test_rejects_unsafe_member_paths(tmp_path: Path, member: str) -> None:
    bundle = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("report.json", "{}")
        archive.writestr(member, "bad")
    with pytest.raises(BundleError, match="unsafe bundle member"):
        extract_bundle(bundle, tmp_path / "out", expected_sha256=digest(bundle))


def test_rejects_symlink_and_duplicate_members(tmp_path: Path) -> None:
    symlink_bundle = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink_bundle, "w") as archive:
        archive.writestr("report.json", "{}")
        link = zipfile.ZipInfo("link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "report.json")
    with pytest.raises(BundleError, match="symbolic link"):
        extract_bundle(symlink_bundle, tmp_path / "symlink-out", expected_sha256=digest(symlink_bundle))

    duplicate_bundle = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate_bundle, "w") as archive:
        archive.writestr("report.json", "{}")
        archive.writestr("REPORT.JSON", "{}")
    with pytest.raises(BundleError, match="duplicate"):
        extract_bundle(duplicate_bundle, tmp_path / "duplicate-out", expected_sha256=digest(duplicate_bundle))


def test_rejects_wrong_bundle_digest_and_missing_report(tmp_path: Path) -> None:
    bundle = tmp_path / "evidence.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("artifact.json", "{}")
    with pytest.raises(BundleError, match="SHA-256 mismatch"):
        extract_bundle(bundle, tmp_path / "wrong-hash", expected_sha256="0" * 64)
    with pytest.raises(BundleError, match="report.json"):
        extract_bundle(bundle, tmp_path / "missing-report", expected_sha256=digest(bundle))


def test_validation_failure_preserves_existing_empty_destination(tmp_path: Path) -> None:
    bundle = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("report.json", "{}")
        archive.writestr("../escape.json", "bad")
    destination = tmp_path / "existing-empty"
    destination.mkdir()

    with pytest.raises(BundleError, match="unsafe bundle member"):
        extract_bundle(bundle, destination, expected_sha256=digest(bundle))

    assert destination.is_dir()
    assert not any(destination.iterdir())
