"""Archive extraction helpers with traversal, link, and resource limits."""

from __future__ import annotations

import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

DEFAULT_MAX_MEMBERS = 100_000
DEFAULT_MAX_BYTES = 100 * 1024**3


class UnsafeArchiveError(ValueError):
    """Raised when an archive member cannot be extracted safely."""


def _member_target(root: Path, name: str) -> Path:
    if not name or "\x00" in name:
        raise UnsafeArchiveError("archive member has an empty or invalid name")

    normalized = name.replace("\\", "/")
    member_path = PurePosixPath(normalized)
    if member_path.is_absolute() or PureWindowsPath(name).drive:
        raise UnsafeArchiveError(f"archive member uses an absolute path: {name!r}")
    if any(part == ".." for part in member_path.parts):
        raise UnsafeArchiveError(f"archive member escapes the destination: {name!r}")

    relative_parts = [part for part in member_path.parts if part not in ("", ".")]
    target = (root / Path(*relative_parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise UnsafeArchiveError(f"archive member escapes the destination: {name!r}") from exc
    return target


def _check_budget(
    member_count: int,
    total_bytes: int,
    member_size: int,
    *,
    max_members: int,
    max_bytes: int,
) -> int:
    if member_count > max_members:
        raise UnsafeArchiveError(f"archive contains more than {max_members} members")
    if member_size < 0:
        raise UnsafeArchiveError("archive member has a negative size")
    total_bytes += member_size
    if total_bytes > max_bytes:
        raise UnsafeArchiveError(f"archive expands beyond {max_bytes} bytes")
    return total_bytes


def safe_extract_tar(
    archive: tarfile.TarFile,
    destination: str | Path,
    *,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> None:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    total_bytes = 0

    for member_count, member in enumerate(archive.getmembers(), start=1):
        target = _member_target(root, member.name)
        total_bytes = _check_budget(
            member_count,
            total_bytes,
            member.size,
            max_members=max_members,
            max_bytes=max_bytes,
        )
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not member.isfile():
            raise UnsafeArchiveError(f"unsupported tar member type: {member.name!r}")

        source = archive.extractfile(member)
        if source is None:
            raise UnsafeArchiveError(f"cannot read tar member: {member.name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with source, target.open("wb") as output:
            shutil.copyfileobj(source, output)


def safe_extract_zip(
    archive: zipfile.ZipFile,
    destination: str | Path,
    *,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> None:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    total_bytes = 0

    for member_count, member in enumerate(archive.infolist(), start=1):
        target = _member_target(root, member.filename)
        total_bytes = _check_budget(
            member_count,
            total_bytes,
            member.file_size,
            max_members=max_members,
            max_bytes=max_bytes,
        )
        unix_mode = member.external_attr >> 16
        if stat.S_ISLNK(unix_mode):
            raise UnsafeArchiveError(f"zip symbolic links are not supported: {member.filename!r}")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member, "r") as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
