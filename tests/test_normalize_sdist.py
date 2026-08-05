from __future__ import annotations

import gzip
import io
import tarfile

import pytest

from scripts.normalize_sdist import normalize_sdist


def _sdist(path, *, member_mtime, gzip_mtime):
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        directory = tarfile.TarInfo("package-1.0/")
        directory.type = tarfile.DIRTYPE
        directory.mtime = member_mtime
        archive.addfile(directory)
        payload = b"content\n"
        member = tarfile.TarInfo("package-1.0/data.txt")
        member.size = len(payload)
        member.mtime = member_mtime
        member.uid = 123
        member.gid = 456
        member.uname = "builder"
        member.gname = "builder"
        archive.addfile(member, io.BytesIO(payload))
    with path.open("wb") as output, gzip.GzipFile(fileobj=output, mode="wb", mtime=gzip_mtime) as compressed:
        compressed.write(tar_buffer.getvalue())


def test_normalize_sdist_is_byte_reproducible(tmp_path):
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _sdist(first, member_mtime=100, gzip_mtime=200)
    _sdist(second, member_mtime=300, gzip_mtime=400)

    normalize_sdist(first, 42)
    normalize_sdist(second, 42)

    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, mode="r:gz") as archive:
        assert {member.mtime for member in archive.getmembers()} == {42}
        assert {member.uid for member in archive.getmembers()} == {0}
        assert {member.gid for member in archive.getmembers()} == {0}


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_normalize_sdist_rejects_links(tmp_path, member_type):
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        member = tarfile.TarInfo("xcientist-0.2.2/link")
        member.type = member_type
        member.linkname = "../../outside"
        archive.addfile(member)

    with pytest.raises(ValueError, match="unsupported sdist member type"):
        normalize_sdist(archive_path, 42)
