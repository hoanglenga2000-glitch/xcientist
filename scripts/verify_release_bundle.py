#!/usr/bin/env python3
"""End-to-end verification through the official signed EvoMind lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import json
import os
import re
import secrets
import shutil
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


MAX_ZIP_ENTRIES = 100_000
MAX_ZIP_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
MAX_ZIP_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 1_000.0
MAX_ZIP_PATH_CHARS = 1_024
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
ALLOWED_ZIP_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
OUTER_MANIFEST_SCHEMA = "evomind.release.manifest.v2"
INNER_MANIFEST_SCHEMA = "evomind.windows.bundle.v1"
INNER_SCHEMA_VERSION = 1
CLI_PACKAGE_NAME = "@evomind-ai/cli"
CLI_ENTRYPOINT = PurePosixPath("bin/evomind.mjs")
INTEGRITY_FILES = frozenset({
    "release-manifest.json",
    "release-manifest.sha256",
    "SHA256SUMS.json",
})
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * quantile))]


SESSION_OPENER: urllib.request.OpenerDirector | None = None
SESSION_CSRF = ""


def request(port: int, method: str, path: str, body: dict | None = None, expected: int = 200) -> tuple[bytes, float]:
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=payload, method=method,
        headers={
            "Content-Type": "application/json",
            **({"Origin": f"http://127.0.0.1:{port}", "x-evomind-csrf": SESSION_CSRF}
               if method not in {"GET", "HEAD", "OPTIONS"} and SESSION_CSRF else {}),
        },
    )
    started = time.perf_counter()
    try:
        opener = SESSION_OPENER.open if SESSION_OPENER is not None else urllib.request.urlopen
        with opener(req, timeout=30) as response:
            status, data = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, data = error.code, error.read()
    elapsed = (time.perf_counter() - started) * 1000
    if status != expected:
        raise AssertionError(f"{method} {path}: HTTP {status}, expected {expected}: {data[:1000]!r}")
    return data, elapsed


def bootstrap_session(port: int, token: str) -> None:
    global SESSION_OPENER, SESSION_CSRF
    jar = http.cookiejar.CookieJar()
    SESSION_OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    payload = json.dumps({"token": token}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/session/bootstrap", data=payload, method="POST",
        headers={"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}"},
    )
    with SESSION_OPENER.open(req, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    SESSION_CSRF = str(data.get("csrf_token") or "")
    if not SESSION_CSRF or not any(cookie.name for cookie in jar):
        raise RuntimeError("local session bootstrap did not establish both CSRF and HttpOnly session state")


def _zip_member_parts(member: zipfile.ZipInfo) -> tuple[tuple[str, ...], bool]:
    raw = member.filename.replace("\\", "/")
    if not raw or "\x00" in raw or len(raw) > MAX_ZIP_PATH_CHARS:
        raise RuntimeError(f"unsafe ZIP path: {member.filename!r}")
    is_directory = member.is_dir()
    if raw.startswith("/") or raw.startswith("//"):
        raise RuntimeError(f"unsafe ZIP path: {member.filename}")
    if is_directory:
        raw = raw[:-1]
    raw_parts = raw.split("/")
    if not raw or any(part in {"", ".", ".."} for part in raw_parts):
        raise RuntimeError(f"unsafe ZIP path: {member.filename}")
    for part in raw_parts:
        if len(part) > 255 or part.endswith((".", " ")):
            raise RuntimeError(f"unsafe Windows ZIP path component: {member.filename}")
        if any(ord(char) < 32 or char in '<>:"|?*' for char in part):
            raise RuntimeError(f"unsafe Windows ZIP path component: {member.filename}")
        device_stem = part.split(".", 1)[0].rstrip(" .").upper()
        if device_stem in WINDOWS_RESERVED_NAMES:
            raise RuntimeError(f"reserved Windows ZIP path component: {member.filename}")
    return tuple(raw_parts), is_directory


def _validate_zip_entry_type(member: zipfile.ZipInfo, is_directory: bool) -> None:
    if member.flag_bits & 0x1:
        raise RuntimeError(f"encrypted ZIP entry is forbidden: {member.filename}")
    if member.compress_type not in ALLOWED_ZIP_COMPRESSION:
        raise RuntimeError(f"unsupported ZIP compression method: {member.filename}")
    unix_mode = (member.external_attr >> 16) & 0xFFFF
    unix_type = stat.S_IFMT(unix_mode)
    if stat.S_ISLNK(unix_mode) or unix_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise RuntimeError(f"ZIP link or special-file entry is forbidden: {member.filename}")
    if unix_type == stat.S_IFDIR and not is_directory:
        raise RuntimeError(f"malformed ZIP directory entry: {member.filename}")
    dos_attributes = member.external_attr & 0xFFFF
    if dos_attributes & 0x0400:
        raise RuntimeError(f"ZIP reparse-point entry is forbidden: {member.filename}")
    if is_directory and member.file_size != 0:
        raise RuntimeError(f"ZIP directory has a payload: {member.filename}")


def safe_extract(
    archive: Path,
    destination: Path,
    *,
    max_entries: int = MAX_ZIP_ENTRIES,
    max_uncompressed_bytes: int = MAX_ZIP_UNCOMPRESSED_BYTES,
    max_member_bytes: int = MAX_ZIP_MEMBER_BYTES,
    max_compression_ratio: float = MAX_ZIP_COMPRESSION_RATIO,
) -> Path:
    """Extract a Windows release ZIP only after a complete metadata preflight."""
    if archive.is_symlink() or not archive.is_file():
        raise RuntimeError("release ZIP must be a regular file")
    if destination.exists():
        raise RuntimeError(f"ZIP destination already exists: {destination}")
    entries: list[tuple[zipfile.ZipInfo, tuple[str, ...], bool]] = []
    spellings: dict[tuple[str, ...], tuple[str, ...]] = {}
    kinds: dict[tuple[str, ...], str] = {}
    total_uncompressed = 0
    total_compressed = 0
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if not members or len(members) > max_entries:
            raise RuntimeError(f"ZIP entry-count limit exceeded: {len(members)} > {max_entries}")
        for member in members:
            parts, is_directory = _zip_member_parts(member)
            _validate_zip_entry_type(member, is_directory)
            folded_parts = tuple(unicodedata.normalize("NFC", part).casefold() for part in parts)
            for depth in range(1, len(parts) + 1):
                folded_prefix = folded_parts[:depth]
                original_prefix = parts[:depth]
                prior_spelling = spellings.get(folded_prefix)
                if prior_spelling is not None and prior_spelling != original_prefix:
                    raise RuntimeError(f"case-fold ZIP collision: {member.filename}")
                spellings[folded_prefix] = original_prefix
            if folded_parts in kinds:
                raise RuntimeError(f"duplicate ZIP entry: {member.filename}")
            for depth in range(1, len(folded_parts)):
                if kinds.get(folded_parts[:depth]) == "file":
                    raise RuntimeError(f"ZIP file/directory collision: {member.filename}")
            if not is_directory and any(
                existing[: len(folded_parts)] == folded_parts for existing in kinds if len(existing) > len(folded_parts)
            ):
                raise RuntimeError(f"ZIP file/directory collision: {member.filename}")
            kinds[folded_parts] = "directory" if is_directory else "file"
            if member.file_size < 0 or member.compress_size < 0 or member.file_size > max_member_bytes:
                raise RuntimeError(f"ZIP member-size limit exceeded: {member.filename}")
            total_uncompressed += member.file_size
            total_compressed += member.compress_size
            if total_uncompressed > max_uncompressed_bytes:
                raise RuntimeError(
                    f"ZIP uncompressed-size limit exceeded: {total_uncompressed} > {max_uncompressed_bytes}"
                )
            if member.file_size:
                member_ratio = member.file_size / max(member.compress_size, 1)
                if member_ratio > max_compression_ratio:
                    raise RuntimeError(f"ZIP compression-ratio limit exceeded: {member.filename}")
            entries.append((member, parts, is_directory))
        aggregate_ratio = total_uncompressed / max(total_compressed, 1)
        if total_uncompressed and aggregate_ratio > max_compression_ratio:
            raise RuntimeError(f"ZIP aggregate compression-ratio limit exceeded: {aggregate_ratio:.2f}")

        destination.mkdir(parents=True, exist_ok=False)
        destination_resolved = destination.resolve()
        extracted_bytes = 0
        for member, parts, is_directory in entries:
            target = destination.joinpath(*parts)
            target_resolved = target.resolve()
            if target_resolved != destination_resolved and destination_resolved not in target_resolved.parents:
                raise RuntimeError(f"unsafe ZIP path: {member.filename}")
            if is_directory:
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            for parent in (target.parent, *target.parent.parents):
                if parent == destination_resolved.parent:
                    break
                if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
                    raise RuntimeError(f"ZIP extraction traverses a reparse point: {member.filename}")
                if parent == destination_resolved:
                    break
            member_bytes = 0
            with zipped.open(member, "r") as source, target.open("xb") as output:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    member_bytes += len(block)
                    extracted_bytes += len(block)
                    if member_bytes > member.file_size or extracted_bytes > max_uncompressed_bytes:
                        raise RuntimeError(f"ZIP payload exceeded declared limits: {member.filename}")
                    output.write(block)
            if member_bytes != member.file_size:
                raise RuntimeError(f"ZIP member size changed during extraction: {member.filename}")
    roots = [candidate for candidate in destination.iterdir() if candidate.is_dir()]
    root_files = [candidate for candidate in destination.iterdir() if not candidate.is_dir()]
    if len(roots) != 1 or root_files:
        raise RuntimeError(f"expected exactly one bundle root, got {roots}")
    return roots[0]


def _regular_file(path: Path) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and not getattr(path, "is_junction", lambda: False)()
    )


def _read_json_object(path: Path, label: str) -> dict:
    if not _regular_file(path):
        raise RuntimeError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _manifest_path(value: object) -> tuple[str, tuple[str, ...]]:
    raw = value if isinstance(value, str) else ""
    pure = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or raw != pure.as_posix()
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or len(raw) > MAX_ZIP_PATH_CHARS
    ):
        raise RuntimeError(f"invalid manifest path: {value!r}")
    for part in pure.parts:
        if (
            len(part) > 255
            or part.endswith((".", " "))
            or any(ord(char) < 32 or char in '<>:"|?*' for char in part)
            or part.split(".", 1)[0].rstrip(" .").upper() in WINDOWS_RESERVED_NAMES
        ):
            raise RuntimeError(f"invalid Windows manifest path: {raw}")
    return raw, pure.parts


def _bundle_file_tree(bundle: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for candidate in bundle.rglob("*"):
        if candidate.is_symlink() or getattr(candidate, "is_junction", lambda: False)():
            raise RuntimeError(f"bundle tree contains a link or reparse point: {candidate}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise RuntimeError(f"bundle tree contains a non-regular entry: {candidate}")
        relative = candidate.relative_to(bundle).as_posix()
        files[relative] = candidate
    return files


def _require_dict(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def _verify_runtime_metadata(bundle: Path, manifest: dict) -> None:
    version = manifest["version"]
    compatibility = _require_dict(manifest.get("runtime_compatibility"), "runtime_compatibility")
    node_contract = _require_dict(compatibility.get("node"), "runtime_compatibility.node")
    python_contract = _require_dict(compatibility.get("python"), "runtime_compatibility.python")
    if (
        not isinstance(node_contract.get("bundled"), str)
        or not node_contract["bundled"].strip()
        or not SEMVER_PATTERN.fullmatch(str(node_contract.get("minimum") or ""))
    ):
        raise RuntimeError("bundled Node runtime contract is invalid")
    bundled_python = str(python_contract.get("bundled") or "")
    if (
        "/cp312/win_amd64/bundled" not in bundled_python
        or python_contract.get("minimum") != "3.12"
        or python_contract.get("maximum_exclusive") != "3.13"
    ):
        raise RuntimeError("bundled Python runtime contract is invalid")

    lock = _read_json_object(bundle / "metadata" / "package-lock.json", "npm package lock")
    packages = _require_dict(lock.get("packages"), "npm package lock packages")
    root_package = _require_dict(packages.get(""), "npm package lock root package")
    if (
        lock.get("lockfileVersion") not in {2, 3}
        or not isinstance(lock.get("name"), str)
        or lock.get("version") != version
        or root_package.get("name") != lock.get("name")
        or root_package.get("version") != version
    ):
        raise RuntimeError("npm package lock does not describe this release")

    sbom = _read_json_object(bundle / "metadata" / "node-sbom.cdx.json", "CycloneDX SBOM")
    sbom_metadata = _require_dict(sbom.get("metadata"), "CycloneDX metadata")
    sbom_component = _require_dict(sbom_metadata.get("component"), "CycloneDX root component")
    if (
        sbom.get("bomFormat") != "CycloneDX"
        or not re.fullmatch(r"1\.\d+", str(sbom.get("specVersion") or ""))
        or not isinstance(sbom.get("version"), int)
        or isinstance(sbom.get("version"), bool)
        or sbom["version"] <= 0
        or not isinstance(sbom.get("components"), list)
        or not sbom["components"]
        or sbom_component.get("name") != lock.get("name")
        or sbom_component.get("version") != version
    ):
        raise RuntimeError("CycloneDX SBOM does not describe the locked release")

    release_contract = _read_json_object(
        bundle / "metadata" / "release-contract.json", "release runtime contract"
    )
    layout = _require_dict(release_contract.get("layout"), "release runtime layout")
    entrypoints = _require_dict(release_contract.get("entrypoints"), "release entrypoints")
    web = _require_dict(release_contract.get("web"), "release web contract")
    targets = _require_dict(release_contract.get("release_targets"), "release targets")
    expected_layout = {
        "app": "app",
        "runtime": "runtime",
        "scripts": "scripts",
        "wheels": "runtime/wheels",
    }
    expected_entrypoints = {
        "install": "install.ps1",
        "start": "start.ps1",
        "stop": "stop.ps1",
        "status": "status.ps1",
    }
    if (
        release_contract.get("format_version") != 1
        or any(layout.get(key) != value for key, value in expected_layout.items())
        or any(entrypoints.get(key) != value for key, value in expected_entrypoints.items())
        or web.get("mode") != "next-standalone"
        or web.get("server") != "app/server.js"
        or web.get("bind") != "127.0.0.1"
        or targets.get("clean_install_required") is not True
        or targets.get("offline_when_wheels_present") is not True
        or targets.get("sha256_manifest") is not True
        or targets.get("no_user_data_in_bundle") is not True
    ):
        raise RuntimeError("release runtime contract is incomplete")

    wheelhouse = _read_json_object(
        bundle / "runtime" / "wheels" / "wheelhouse-manifest.json", "wheelhouse manifest"
    )
    wheels = wheelhouse.get("wheels")
    if (
        wheelhouse.get("platform") != "win_amd64"
        or wheelhouse.get("python") != ["3.12.x"]
        or wheelhouse.get("abi") != ["cp312"]
        or not isinstance(wheels, list)
        or not wheels
        or wheelhouse.get("wheel_count") != len(wheels)
        or not bundled_python.startswith(str(wheelhouse.get("bundled_python") or ""))
    ):
        raise RuntimeError("wheelhouse runtime contract is invalid")
    declared_wheels: set[str] = set()
    for item in wheels:
        if not isinstance(item, dict):
            raise RuntimeError("wheelhouse entry must be an object")
        name = item.get("name")
        size = item.get("size")
        digest = item.get("sha256")
        if (
            not isinstance(name, str)
            or PurePosixPath(name).name != name
            or not name.lower().endswith(".whl")
            or name.casefold() in declared_wheels
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not SHA256_PATTERN.fullmatch(str(digest or ""))
        ):
            raise RuntimeError("wheelhouse entry is invalid")
        declared_wheels.add(name.casefold())
        wheel = bundle / "runtime" / "wheels" / name
        if not _regular_file(wheel) or wheel.stat().st_size != size or sha256(wheel) != digest:
            raise RuntimeError(f"wheelhouse payload mismatch: {name}")
    actual_wheels = {
        path.name.casefold() for path in (bundle / "runtime" / "wheels").glob("*.whl") if _regular_file(path)
    }
    if actual_wheels != declared_wheels:
        raise RuntimeError("wheelhouse manifest does not exactly cover wheel payloads")

    try:
        pyproject = tomllib.loads((bundle / "runtime" / "python" / "pyproject.toml").read_text(encoding="utf-8"))
        uv_lock = tomllib.loads((bundle / "runtime" / "python" / "uv.lock").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError("bundled Python project/lock metadata is invalid") from error
    project = _require_dict(pyproject.get("project"), "bundled Python project")
    if (
        project.get("name") != "xcientist"
        or project.get("version") != version
        or not isinstance(project.get("requires-python"), str)
        or not isinstance(uv_lock.get("version"), int)
        or uv_lock["version"] <= 0
        or not isinstance(uv_lock.get("requires-python"), str)
        or not isinstance(uv_lock.get("package"), list)
        or not uv_lock["package"]
    ):
        raise RuntimeError("bundled Python project and lock do not describe this release")


def verify_inner_manifest(bundle: Path, outer_manifest: dict | None = None) -> dict:
    manifest_path = bundle / "release-manifest.json"
    manifest = _read_json_object(manifest_path, "inner bundle manifest")
    if manifest.get("schema") != INNER_MANIFEST_SCHEMA or manifest.get("schema_version") != INNER_SCHEMA_VERSION:
        raise RuntimeError("unsupported inner bundle manifest schema")
    if (
        not isinstance(manifest.get("version"), str)
        or not SEMVER_PATTERN.fullmatch(manifest["version"])
        or manifest.get("product") != "EvoMind"
        or manifest.get("platform") != "win-x64"
    ):
        raise RuntimeError("inner bundle manifest identity is invalid")
    if outer_manifest is not None:
        if not isinstance(outer_manifest, dict) or outer_manifest.get("schema") != OUTER_MANIFEST_SCHEMA:
            raise RuntimeError("outer release manifest schema is not supported")
        if manifest["version"] != outer_manifest.get("version"):
            raise RuntimeError("inner and outer release versions differ")

    items = manifest.get("files")
    if not isinstance(items, list) or not items:
        raise RuntimeError("inner bundle manifest files must be a non-empty array")
    seen: set[str] = set()
    paths: list[str] = []
    total_bytes = 0
    aggregate = hashlib.sha256()
    expected_sums: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError("inner bundle manifest file entry must be an object")
        relative, parts = _manifest_path(item.get("path"))
        folded = "/".join(unicodedata.normalize("NFC", part).casefold() for part in parts)
        size = item.get("bytes")
        digest = item.get("sha256")
        if (
            folded in seen
            or relative in INTEGRITY_FILES
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or item.get("size") != size
            or not SHA256_PATTERN.fullmatch(str(digest or ""))
        ):
            raise RuntimeError(f"invalid inner bundle manifest entry: {relative}")
        seen.add(folded)
        paths.append(relative)
        total_bytes += size
        aggregate.update(f"{relative}\0{digest}\0{size}\n".encode("utf-8"))
        expected_sums[relative] = digest
        target = bundle.joinpath(*parts)
        if not _regular_file(target) or target.stat().st_size != size or sha256(target) != digest:
            raise RuntimeError(f"inner bundle manifest mismatch: {relative}")
    if paths != sorted(paths):
        raise RuntimeError("inner bundle manifest paths are not in canonical order")
    if manifest.get("file_count") != len(items):
        raise RuntimeError("inner bundle manifest file_count mismatch")
    if manifest.get("payload_bytes") != total_bytes:
        raise RuntimeError("inner bundle manifest payload_bytes mismatch")
    if manifest.get("aggregate_sha256") != aggregate.hexdigest():
        raise RuntimeError("inner bundle manifest aggregate_sha256 mismatch")

    manifest_digest = sha256(manifest_path)
    manifest_sum_path = bundle / "release-manifest.sha256"
    try:
        manifest_sum_lines = manifest_sum_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise RuntimeError("release-manifest.sha256 is not valid ASCII") from error
    if (
        not _regular_file(manifest_sum_path)
        or manifest_sum_lines != [f"{manifest_digest}  release-manifest.json"]
    ):
        raise RuntimeError("release-manifest.sha256 does not authenticate release-manifest.json")
    sums = _read_json_object(bundle / "SHA256SUMS.json", "SHA256SUMS")
    expected_sums["release-manifest.json"] = manifest_digest
    if sums != expected_sums:
        raise RuntimeError("SHA256SUMS does not exactly cover the payload and inner manifest")

    actual_files = _bundle_file_tree(bundle)
    expected_files = set(paths) | set(INTEGRITY_FILES)
    if set(actual_files) != expected_files:
        missing = sorted(expected_files - set(actual_files))[:10]
        extra = sorted(set(actual_files) - expected_files)[:10]
        raise RuntimeError(f"inner manifest file-tree coverage mismatch: missing={missing}, extra={extra}")

    required = {
        "app/server.js", "app/.next/static", "app/public", "app/prisma/migrations",
        "runtime/node/node.exe", "runtime/python/python.exe", "runtime/python/src/xsci",
        "runtime/python/pyproject.toml", "runtime/python/uv.lock",
        "runtime/wheels/wheelhouse-manifest.json",
        "metadata/node-sbom.cdx.json", "metadata/package-lock.json", "metadata/release-contract.json",
        "scripts/release_db_migrate.py", "scripts/manage_workstation_dashboard.py",
        "install.ps1", "start.ps1", "stop.ps1", "status.ps1",
    }
    missing_required = sorted(relative for relative in required if not (bundle / relative).exists())
    if missing_required:
        raise RuntimeError(f"required payload missing: {missing_required}")
    forbidden = [
        path for path in bundle.rglob("*")
        if path.name in {".env", "workstation.db", "runtime.sqlite3", ".git"}
        or (path.relative_to(bundle).parts and path.relative_to(bundle).parts[0] == "user-data")
    ]
    if forbidden:
        raise RuntimeError(f"forbidden payload found: {forbidden[:5]}")
    _verify_runtime_metadata(bundle, manifest)
    return manifest


def parse_json_output(completed: subprocess.CompletedProcess[str], label: str) -> dict:
    if completed.returncode:
        raise RuntimeError(f"{label} failed ({completed.returncode}): {(completed.stderr or completed.stdout)[-6000:]}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} returned malformed JSON: {completed.stdout[-2000:]}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} did not return a JSON object")
    return value


def run_cli(cli: Path, env: dict[str, str], *arguments: str, allow_failure: bool = False) -> dict:
    if cli.suffix.lower() != ".mjs" or not _regular_file(cli):
        raise RuntimeError("EvoMind commands must execute the installed npm tarball CLI entrypoint")
    completed = subprocess.run(
        [str(env["EVOMIND_NODE"]), str(cli), *arguments, "--json"],
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    if allow_failure and completed.returncode:
        return {"ok": False, "returncode": completed.returncode, "stderr": completed.stderr[-2000:]}
    return parse_json_output(completed, f"evomind {' '.join(arguments)}")


def authenticate_release(cli: Path, env: dict[str, str], signed_manifest: Path, archive: Path) -> dict:
    """Use the bootstrap package's pinned trust root before opening the ZIP."""
    receipt = run_cli(
        cli,
        env,
        "verify-release",
        "--manifest",
        str(signed_manifest),
        "--bundle",
        str(archive),
    )
    if receipt.get("ok") is not True or receipt.get("status") != "verified":
        raise RuntimeError(f"package-pinned release authentication failed: {receipt}")
    if not re.fullmatch(r"[a-f0-9]{64}", str(receipt.get("sha256") or "")):
        raise RuntimeError("release authentication receipt omitted the bundle SHA-256")
    if not isinstance(receipt.get("bytes"), int) or receipt["bytes"] <= 0:
        raise RuntimeError("release authentication receipt omitted the bundle byte length")
    if not isinstance(receipt.get("version"), str) or not receipt["version"]:
        raise RuntimeError("release authentication receipt omitted the release version")
    if receipt.get("schema") != OUTER_MANIFEST_SCHEMA:
        raise RuntimeError("release authentication receipt used an unsupported manifest schema")
    return receipt


def stage_regular_file(source: Path, destination: Path, label: str) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"{label} must be a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def install_cli_from_tgz(
    cli_tgz: Path,
    prefix: Path,
    npm: Path,
    env: dict[str, str],
) -> tuple[Path, dict]:
    """Install one npm tarball into an empty, disposable prefix and return its packaged CLI."""
    if not _regular_file(cli_tgz) or cli_tgz.suffix.lower() != ".tgz":
        raise RuntimeError("--cli-tgz must name a regular npm .tgz artifact")
    if not _regular_file(npm):
        raise RuntimeError("npm executable is missing or not a regular file")
    if prefix.exists():
        raise RuntimeError("disposable npm prefix must not already exist")
    install_env = env.copy()
    install_env.update({
        "npm_config_audit": "false",
        "npm_config_fund": "false",
        "npm_config_ignore_scripts": "true",
        "npm_config_package_lock": "false",
        "npm_config_update_notifier": "false",
        "npm_config_cache": str(prefix.parent / "npm-cache"),
    })
    completed = subprocess.run(
        [
            str(npm),
            "install",
            "--offline",
            "--ignore-scripts",
            "--no-audit",
            "--no-fund",
            "--package-lock=false",
            "--prefix",
            str(prefix),
            str(cli_tgz),
        ],
        cwd=prefix.parent,
        env=install_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout)[-6000:]
        raise RuntimeError(f"npm CLI tarball installation failed ({completed.returncode}): {detail}")

    prefix_resolved = prefix.resolve()
    package_root = prefix / "node_modules" / "@evomind-ai" / "cli"
    package_resolved = package_root.resolve()
    if prefix_resolved not in package_resolved.parents:
        raise RuntimeError("installed CLI package escaped the disposable npm prefix")
    for candidate in (package_root, package_root.parent, prefix / "node_modules"):
        if candidate.is_symlink() or getattr(candidate, "is_junction", lambda: False)():
            raise RuntimeError("installed CLI package traverses a link or reparse point")
    package = _read_json_object(package_root / "package.json", "installed CLI package.json")
    package_bin = package.get("bin")
    if (
        package.get("name") != CLI_PACKAGE_NAME
        or not isinstance(package.get("version"), str)
        or not SEMVER_PATTERN.fullmatch(package["version"])
        or not isinstance(package_bin, dict)
        or package_bin.get("evomind") != CLI_ENTRYPOINT.as_posix()
    ):
        raise RuntimeError("npm tarball is not the expected EvoMind CLI package")
    cli = package_root.joinpath(*CLI_ENTRYPOINT.parts)
    cli_resolved = cli.resolve()
    if package_resolved not in cli_resolved.parents or not _regular_file(cli):
        raise RuntimeError("npm tarball did not install its declared EvoMind CLI entrypoint")
    trust_root = package_root / "keys" / "release-ed25519-public.pem"
    if not _regular_file(trust_root) or b"PUBLIC KEY" not in trust_root.read_bytes():
        raise RuntimeError("npm tarball did not contain its pinned release trust root")
    return cli, package


def cleanup_cli(cli: Path, env: dict[str, str], *arguments: str) -> dict:
    try:
        return run_cli(cli, env, *arguments, allow_failure=True)
    except BaseException as error:
        return {
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def read_bootstrap_token(log_root: Path, port: int) -> str:
    suffix = "" if port == 8088 else f".{port}"
    candidates = list(log_root.rglob(f"dashboard{suffix}.bootstrap.once"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one one-time bootstrap file, got {candidates}")
    path = candidates[0]
    url = path.read_text(encoding="utf-8").strip()
    path.unlink()
    fragment = urllib.parse.parse_qs(urllib.parse.urlsplit(url).fragment)
    token = fragment.get("bootstrap", [""])[0]
    if len(token) < 32:
        raise RuntimeError("bootstrap fragment token is missing")
    return token


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Ed25519-signed outer release envelope")
    parser.add_argument("--cli-tgz", type=Path, required=True, help="npm pack artifact containing the pinned CLI")
    parser.add_argument("--node", type=Path, default=Path(shutil.which("node") or "node"))
    parser.add_argument(
        "--npm",
        type=Path,
        default=Path(shutil.which("npm.cmd") or shutil.which("npm") or "npm"),
    )
    parser.add_argument("--port", type=int, default=18192)
    parser.add_argument("--runtime-port", type=int, default=18765)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--temp-root", type=Path)
    args = parser.parse_args()
    if args.port == args.runtime_port or port_open(args.port) or port_open(args.runtime_port):
        raise SystemExit("release verifier requires two distinct free loopback ports")

    archive = args.zip.expanduser().resolve()
    signed_manifest = args.manifest.expanduser().resolve()
    cli_tgz = args.cli_tgz.expanduser().resolve()
    node = args.node.expanduser().resolve()
    npm = args.npm.expanduser().resolve()

    temp_parent = args.temp_root.expanduser().resolve() if args.temp_root else None
    if temp_parent:
        temp_parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="evomind-release-smoke-", dir=temp_parent))
    local = temp / "local"
    roaming = temp / "roaming"
    layout_nonce = secrets.token_urlsafe(32)
    layout_capability = temp / "qa-layout-capability.json"
    with layout_capability.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump({
            "schema": "evomind.qa_layout_capability.v1",
            "nonce": layout_nonce,
            "local_base": str(local),
            "roaming_base": str(roaming),
        }, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(layout_capability, 0o600)
    env = os.environ.copy()
    env.update({
        "LOCALAPPDATA": str(local),
        "APPDATA": str(roaming),
        "WORKSTATION_PORT": str(args.port),
        "EVOMIND_RUNTIME_PORT": str(args.runtime_port),
        "EVOMIND_NODE": str(node),
        "EVOMIND_QA_LAYOUT_CAPABILITY": str(layout_capability),
        "EVOMIND_QA_LAYOUT_NONCE": layout_nonce,
        "NEXT_TELEMETRY_DISABLED": "1",
        "WORKSTATION_DISABLE_AGENT_EXECUTION": "1",
        "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "", "DEEPSEEK_API_KEY": "",
        "KAGGLE_USERNAME": "", "KAGGLE_KEY": "",
    })
    result: dict[str, object] = {
        "zip": str(archive),
        "cli_tgz": str(cli_tgz),
        "port": args.port,
        "runtime_port": args.runtime_port,
    }
    failure: BaseException | None = None
    cli: Path | None = None
    try:
        if not _regular_file(node):
            raise RuntimeError("Node.js executable is missing or not a regular file")
        staged_manifest = temp / "authenticated-input" / "latest.json"
        staged_archive = temp / "authenticated-input" / "EvoMind-win-x64.zip"
        staged_cli_tgz = temp / "bootstrap-input" / "evomind-cli.tgz"
        stage_regular_file(signed_manifest, staged_manifest, "signed release manifest")
        stage_regular_file(archive, staged_archive, "release ZIP")
        stage_regular_file(cli_tgz, staged_cli_tgz, "npm CLI tarball")

        cli, cli_package = install_cli_from_tgz(staged_cli_tgz, temp / "cli-prefix", npm, env)

        authentication = authenticate_release(cli, env, staged_manifest, staged_archive)
        outer = _read_json_object(staged_manifest, "outer release manifest")
        platform = outer.get("platforms", {}).get("win32-x64", {})
        if (
            outer.get("schema") != OUTER_MANIFEST_SCHEMA
            or authentication["schema"] != outer.get("schema")
            or authentication["version"] != outer.get("version")
            or cli_package["version"] != outer.get("version")
            or authentication["bytes"] != platform.get("bytes")
            or authentication["sha256"] != platform.get("sha256")
            or staged_archive.stat().st_size != authentication["bytes"]
            or sha256(staged_archive) != authentication["sha256"]
        ):
            raise RuntimeError("authenticated release receipt drifted before extraction")
        result.update({
            "version": outer["version"],
            "authentication": {
                "status": authentication["status"],
                "schema": authentication["schema"],
                "key_id": authentication.get("manifest_key_id"),
                "bundle_sha256": authentication["sha256"],
                "bundle_bytes": authentication["bytes"],
            },
            "cli_package": {
                "name": cli_package["name"],
                "version": cli_package["version"],
                "tgz_sha256": sha256(staged_cli_tgz),
                "tgz_bytes": staged_cli_tgz.stat().st_size,
                "source": "npm-tgz-disposable-prefix",
            },
        })

        # This is deliberately the first operation that opens the ZIP container.
        unpacked = safe_extract(staged_archive, temp / "unpacked")
        inner = verify_inner_manifest(unpacked, outer)
        cache = local / "EvoMind" / "cache" / "downloads"
        cache.mkdir(parents=True)
        cached_archive = cache / f"EvoMind-win-x64-{outer['version']}.zip"
        stage_regular_file(staged_archive, cached_archive, "authenticated cache bundle")
        if sha256(cached_archive) != authentication["sha256"]:
            raise RuntimeError("preloaded signed cache copy drifted")

        installation = run_cli(
            cli,
            env,
            "install",
            "--manifest",
            str(staged_manifest),
            "--version",
            str(outer["version"]),
        )
        if installation.get("ok") is not True or installation.get("strict_health", {}).get("passed") is not True:
            raise RuntimeError(f"official install did not pass strict health: {installation}")
        status = run_cli(cli, env, "status")
        doctor = run_cli(cli, env, "doctor")
        if status.get("ok") is not True or doctor.get("ok") is not True:
            raise RuntimeError("official status/doctor gate failed")
        strict = status.get("strict_health", {})
        dashboard = strict.get("dashboard", {})
        if dashboard.get("process_port_consistent") is not True or dashboard.get("runtime_process_consistent") is not True:
            raise RuntimeError("dashboard/runtime identity contract failed")

        token = read_bootstrap_token(local / "EvoMind", args.port)
        bootstrap_session(args.port, token)
        page, page_ms = request(args.port, "GET", "/")
        html = page.decode("utf-8", errors="replace")
        if "EvoMind" not in html:
            raise RuntimeError("production HTML is missing the EvoMind marker")
        assets = sorted(set(re.findall(r'["\'](/_next/static/[^"\']+)["\']', html)))
        if not assets:
            raise RuntimeError("production HTML did not reference hashed static assets")
        for asset in assets[:20]:
            request(args.port, "GET", asset)

        tasks, _ = request(args.port, "GET", "/api/tasks")
        summary, _ = request(args.port, "GET", "/api/workstation-summary")
        task_id = "release_bundle_smoke_task"
        created, _ = request(args.port, "POST", "/api/workstation-actions", {
            "action": "create_workstation_run", "task_id": task_id,
            "metadata": {"trigger": "release_bundle_smoke"},
        })
        if not json.loads(created).get("run_id"):
            raise RuntimeError("release smoke task creation failed")

        stopped = run_cli(cli, env, "stop")
        if stopped.get("ok") is not True or port_open(args.port) or port_open(args.runtime_port):
            raise RuntimeError("official stop did not release both managed ports")
        started = run_cli(cli, env, "start")
        if started.get("ok") is not True:
            raise RuntimeError("official restart failed")
        token = read_bootstrap_token(local / "EvoMind", args.port)
        bootstrap_session(args.port, token)
        after_restart, _ = request(args.port, "GET", "/api/tasks?include_archived=1")
        if task_id not in {item["id"] for item in json.loads(after_restart)["tasks"]}:
            raise RuntimeError("created task did not survive official restart")

        task_latencies: list[float] = []
        summary_latencies: list[float] = []
        latest_summary = b""
        for _ in range(20):
            _, elapsed = request(args.port, "GET", "/api/tasks")
            task_latencies.append(elapsed)
            latest_summary, elapsed = request(args.port, "GET", "/api/workstation-summary")
            summary_latencies.append(elapsed)
        request(args.port, "PATCH", f"/api/tasks/{task_id}", {"action": "archive"})
        request(args.port, "DELETE", f"/api/tasks/{task_id}", {"confirm_task_id": task_id})
        request(args.port, "GET", f"/api/tasks/{task_id}", expected=404)

        database = local / "EvoMind" / "data" / "prisma" / "workstation.db"
        with sqlite3.connect(database) as connection:
            quick = connection.execute("PRAGMA quick_check").fetchone()[0]
            residual = connection.execute("SELECT COUNT(*) FROM tasks WHERE id=?", (task_id,)).fetchone()[0]
        metrics = {
            "page_ms": round(page_ms, 3), "page_bytes": len(page),
            "tasks_p95_ms": round(percentile(task_latencies, 0.95), 3),
            "summary_p95_ms": round(percentile(summary_latencies, 0.95), 3),
            "summary_bytes": len(latest_summary),
        }
        if quick != "ok" or residual != 0:
            raise RuntimeError(f"SQLite cleanup gate failed: quick={quick}, residual={residual}")
        if metrics["tasks_p95_ms"] >= 200 or metrics["summary_p95_ms"] >= 150 or metrics["summary_bytes"] >= 200 * 1024:
            raise RuntimeError(f"performance budget failed: {metrics}")
        result.update({
            "ok": True,
            "inner_manifest": {
                "schema": inner.get("schema"),
                "version": inner.get("version"),
                "files": len(inner.get("files", [])),
                "payload_bytes": inner.get("payload_bytes"),
                "aggregate_sha256": inner.get("aggregate_sha256"),
            },
            "official_lifecycle": {"install": True, "status": True, "doctor": True, "restart": True},
            "identities": {"dashboard": True, "runtime": True},
            "static_assets_checked": min(20, len(assets)),
            "migration": {"quick_check": quick},
            "metrics": metrics,
            "persistence": True,
            "task_residual": residual,
            "initial_tasks_bytes": len(tasks),
            "initial_summary_bytes": len(summary),
        })
    except BaseException as error:
        failure = error
        primary_failure = {"error_type": type(error).__name__, "error": str(error)}
        result.update({"ok": False, **primary_failure, "primary_failure": primary_failure})
    finally:
        if cli is None:
            skipped = {"ok": True, "skipped": "npm CLI tarball was not installed"}
            cleanup_receipt: dict[str, object] = {
                "recover": skipped.copy(),
                "status": skipped.copy(),
                "stop": skipped.copy(),
                "uninstall": skipped.copy(),
            }
        else:
            cleanup_receipt = {
                "recover": cleanup_cli(cli, env, "recover"),
                "status": cleanup_cli(cli, env, "status"),
                "stop": cleanup_cli(cli, env, "stop"),
                "uninstall": cleanup_cli(cli, env, "uninstall", "--purge-data"),
            }
        cleanup_receipt.update({
            "recover_ok": cleanup_receipt["recover"].get("ok") is True,  # type: ignore[union-attr]
            "stop_ok": cleanup_receipt["stop"].get("ok") is True,  # type: ignore[union-attr]
            "uninstall_ok": cleanup_receipt["uninstall"].get("ok") is True,  # type: ignore[union-attr]
        })
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and (port_open(args.port) or port_open(args.runtime_port)):
            time.sleep(0.25)
        cleanup_receipt.update({
            "dashboard_port_released": not port_open(args.port),
            "runtime_port_released": not port_open(args.runtime_port),
        })
        result["cleanup"] = cleanup_receipt
        if cleanup_receipt["dashboard_port_released"] and cleanup_receipt["runtime_port_released"]:
            try:
                shutil.rmtree(temp, ignore_errors=False)
            except OSError as error:
                cleanup_receipt["temp_cleanup_error"] = f"{type(error).__name__}: {error}"
        else:
            cleanup_receipt["temp_preserved"] = str(temp)
        cleanup_receipt["temp_removed"] = not temp.exists()

    cleanup = result.get("cleanup", {})
    passed = result.get("ok") is True and all(
        cleanup.get(key) is True
        for key in (
            "recover_ok", "stop_ok", "uninstall_ok",
            "dashboard_port_released", "runtime_port_released", "temp_removed",
        )
    )
    if result.get("ok") is True and not passed:
        result.update({
            "ok": False,
            "error_type": "CleanupGateError",
            "error": "official release cleanup did not satisfy the release gate",
        })
    if args.json:
        args.json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.json.expanduser().resolve().write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failure and not passed:
        return 1
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
