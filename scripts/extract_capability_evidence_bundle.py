#!/usr/bin/env python3
"""Safely extract an externally hash-pinned capability evidence bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_FILES = 5000
MAX_TOTAL_SIZE = 4 * 1024 * 1024 * 1024
MAX_MEMBER_SIZE = 2 * 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 250


class BundleError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename
    if not name or "\\" in name or "//" in name or "\x00" in name or ":" in name:
        raise BundleError(f"unsafe bundle member: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BundleError(f"unsafe bundle member: {name!r}")
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    if file_type == stat.S_IFLNK:
        raise BundleError(f"symbolic link bundle member rejected: {name!r}")
    if info.flag_bits & 0x1:
        raise BundleError(f"encrypted bundle member rejected: {name!r}")
    if info.file_size > MAX_MEMBER_SIZE:
        raise BundleError(f"bundle member exceeds size limit: {name!r}")
    if info.file_size and info.compress_size == 0:
        raise BundleError(f"invalid compressed size: {name!r}")
    if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
        raise BundleError(f"bundle member compression ratio is unsafe: {name!r}")
    return path


def extract_bundle(bundle: Path, destination: Path, *, expected_sha256: str) -> dict[str, object]:
    expected = str(expected_sha256 or "").strip().lower()
    if SHA256_RE.fullmatch(expected) is None:
        raise BundleError("expected bundle digest must be a lowercase SHA-256")
    actual = sha256_file(bundle)
    if actual != expected:
        raise BundleError("capability evidence bundle SHA-256 mismatch")
    if destination.exists() and any(destination.iterdir()):
        raise BundleError("destination must be absent or empty")
    destination.mkdir(parents=True, exist_ok=True)
    resolved_root = destination.resolve()
    seen: set[str] = set()
    total_size = 0
    file_count = 0
    extraction_started = False
    try:
        with zipfile.ZipFile(bundle) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES:
                raise BundleError("capability evidence bundle contains too many members")
            validated: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
            for info in infos:
                path = _safe_member(info)
                key = path.as_posix().casefold().rstrip("/")
                if key in seen:
                    raise BundleError(f"duplicate capability bundle member: {path.as_posix()!r}")
                seen.add(key)
                if not info.is_dir():
                    file_count += 1
                    total_size += info.file_size
                if total_size > MAX_TOTAL_SIZE:
                    raise BundleError("capability evidence bundle exceeds total size limit")
                validated.append((info, path))
            for info, path in validated:
                extraction_started = True
                target = destination.joinpath(*path.parts)
                resolved = target.resolve()
                try:
                    resolved.relative_to(resolved_root)
                except ValueError as exc:
                    raise BundleError(f"bundle member escapes destination: {path.as_posix()!r}") from exc
                if info.is_dir():
                    resolved.mkdir(parents=True, exist_ok=True)
                    continue
                resolved.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, resolved.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    except (OSError, zipfile.BadZipFile) as exc:
        if extraction_started:
            shutil.rmtree(destination, ignore_errors=True)
        raise BundleError(f"capability evidence bundle is unreadable: {type(exc).__name__}") from exc
    report = destination / "report.json"
    if not report.is_file():
        shutil.rmtree(destination, ignore_errors=True)
        raise BundleError("capability evidence bundle must contain report.json at its root")
    return {
        "bundle_sha256": actual,
        "files": file_count,
        "uncompressed_bytes": total_size,
        "report_path": str(report),
        "destination": str(destination),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = extract_bundle(args.bundle, args.destination, expected_sha256=args.expected_sha256)
    except BundleError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps({"ok": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
