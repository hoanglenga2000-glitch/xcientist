from __future__ import annotations

import io
import stat
import tarfile
import zipfile

import pytest

from scripts.safe_archive import UnsafeArchiveError, safe_extract_tar, safe_extract_zip


def _tar_with_member(name: str, payload: bytes = b"content", *, member_type: bytes | None = None) -> io.BytesIO:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        if member_type is not None:
            member.type = member_type
            member.linkname = "target"
            member.size = 0
        archive.addfile(member, None if member_type is not None else io.BytesIO(payload))
    buffer.seek(0)
    return buffer


def test_safe_extract_tar_extracts_regular_files(tmp_path):
    with tarfile.open(fileobj=_tar_with_member("nested/data.csv"), mode="r:") as archive:
        safe_extract_tar(archive, tmp_path)

    assert (tmp_path / "nested" / "data.csv").read_bytes() == b"content"


@pytest.mark.parametrize("name", ["../outside.txt", "/absolute.txt", "C:\\outside.txt"])
def test_safe_extract_tar_rejects_escaping_paths(tmp_path, name):
    with tarfile.open(fileobj=_tar_with_member(name), mode="r:") as archive:
        with pytest.raises(UnsafeArchiveError):
            safe_extract_tar(archive, tmp_path)


def test_safe_extract_tar_rejects_links(tmp_path):
    with tarfile.open(fileobj=_tar_with_member("link", member_type=tarfile.SYMTYPE), mode="r:") as archive:
        with pytest.raises(UnsafeArchiveError):
            safe_extract_tar(archive, tmp_path)


def test_safe_extract_zip_extracts_regular_files(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr("nested/data.csv", b"content")
    buffer.seek(0)

    with zipfile.ZipFile(buffer, mode="r") as archive:
        safe_extract_zip(archive, tmp_path)

    assert (tmp_path / "nested" / "data.csv").read_bytes() == b"content"


def test_safe_extract_zip_rejects_links(tmp_path):
    buffer = io.BytesIO()
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr(link, "target")
    buffer.seek(0)

    with zipfile.ZipFile(buffer, mode="r") as archive:
        with pytest.raises(UnsafeArchiveError):
            safe_extract_zip(archive, tmp_path)


def test_safe_extract_zip_enforces_uncompressed_size_limit(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr("large.bin", b"12345")
    buffer.seek(0)

    with zipfile.ZipFile(buffer, mode="r") as archive:
        with pytest.raises(UnsafeArchiveError):
            safe_extract_zip(archive, tmp_path, max_bytes=4)
