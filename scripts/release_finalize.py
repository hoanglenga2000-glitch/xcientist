#!/usr/bin/env python3
"""Validate, receipt, and deterministically archive an EvoMind release bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tomllib
from typing import Any
from urllib.parse import unquote, urlparse
import uuid
import zipfile


FORBIDDEN_TOP_LEVEL = {
    ".git",
    "workspace",
    "tasks",
    "experiments",
    "mlebench_model_cache",
    "local-gpu-venv",
    "user-data",
}
FORBIDDEN_NAMES = {".env", "workstation.db", "runtime.sqlite3"}
CANONICAL_JSON_PATHS = {
    "app/.next/app-path-routes-manifest.json",
    "app/.next/server/app-paths-manifest.json",
}
BUILDER_METADATA_SCHEMA = "evomind.release.builder-metadata.v1"
BUILD_RECEIPT_SCHEMA = "evomind.release.build-receipt.v1"
WHEELHOUSE_MANIFEST_SCHEMA = "evomind.release.wheelhouse-manifest.v2"
REQUIRED_BUILDER_FIELDS = {
    "schema",
    "builder",
    "git_commit",
    "source_date_epoch",
    "source_digest",
    "build_id",
    "next_tree_sha256",
    "toolchain_contract_sha256",
    "release_contract_sha256",
    "package_lock_sha256",
    "uv_lock_sha256",
    "locked_requirements_sha256",
    "toolchain",
    "commands",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def tree_digest(root: Path) -> str:
    """Hash a tree by normalized relative path, content hash, and byte count."""

    assert_no_reparse_tree(root)
    aggregate = hashlib.sha256()
    files = sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix())
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        aggregate.update(f"{relative}\0{digest(path)}\0{size}\n".encode())
    return aggregate.hexdigest()


def normalized_datetime(epoch: int) -> tuple[int, int, int, int, int, int]:
    value = datetime.fromtimestamp(epoch, timezone.utc)
    year = min(2107, max(1980, value.year))
    return (year, value.month, value.day, value.hour, value.minute, value.second // 2 * 2)


def is_reparse_point(path: Path) -> bool:
    """Return True for symlinks, Windows junctions, and other reparse points."""

    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def assert_no_reparse_tree(root: Path) -> None:
    if not root.is_dir():
        raise RuntimeError(f"Release tree is not a directory: {root}")
    if is_reparse_point(root):
        raise RuntimeError(f"Release tree root is a reparse point: {root}")
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted([*directories, *files]):
            candidate = current_path / name
            if is_reparse_point(candidate):
                raise RuntimeError(f"Reparse points are forbidden in release payloads: {candidate.relative_to(root)}")


def _atomic_write_text(path: Path, text: str, *, encoding: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding=encoding, newline="\n") as handle:
            handle.write(text)
        os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _sha256_value(value: object, *, field: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeError(f"Builder metadata field {field} is not a SHA-256 digest")
    return text


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Required {label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Required {label} must contain a JSON object: {path}")
    return payload


def locked_wheel_hashes(uv_lock: Path) -> dict[str, str]:
    """Map every wheel filename in uv.lock to its locked SHA-256 digest."""

    with uv_lock.open("rb") as handle:
        payload = tomllib.load(handle)
    locked: dict[str, str] = {}
    for package in payload.get("package", []):
        for wheel in package.get("wheels", []):
            filename = unquote(PurePosixPath(urlparse(str(wheel["url"])).path).name)
            algorithm, separator, value = str(wheel["hash"]).partition(":")
            if separator != ":" or algorithm.lower() != "sha256":
                raise RuntimeError(f"uv.lock wheel does not use SHA-256: {filename}")
            value = _sha256_value(value, field=f"uv.lock:{filename}")
            previous = locked.setdefault(filename, value)
            if previous != value:
                raise RuntimeError(f"uv.lock contains conflicting hashes for {filename}")
    if not locked:
        raise RuntimeError("uv.lock does not contain any locked wheels")
    return locked


def selected_wheels_from_pip_report(report_path: Path) -> dict[str, dict[str, str]]:
    """Extract the exact wheels selected by pip's offline, hash-checked dry run."""

    report = _load_json_object(report_path, label="pip wheelhouse validation report")
    selected: dict[str, dict[str, str]] = {}
    for item in report.get("install", []):
        download = item.get("download_info", {})
        filename = unquote(PurePosixPath(urlparse(str(download.get("url", ""))).path).name)
        if not filename.lower().endswith(".whl"):
            raise RuntimeError(f"pip selected a non-wheel dependency: {filename or '<unknown>'}")
        archive = download.get("archive_info", {})
        hashes = archive.get("hashes", {})
        value = hashes.get("sha256")
        if value is None:
            legacy_hash = str(archive.get("hash", ""))
            prefix = "sha256="
            value = legacy_hash[len(prefix) :] if legacy_hash.startswith(prefix) else None
        value = _sha256_value(value, field=f"pip-report:{filename}")
        metadata = item.get("metadata", {})
        details = {
            "sha256": value,
            "package": str(metadata.get("name", "")),
            "version": str(metadata.get("version", "")),
        }
        previous = selected.setdefault(filename, details)
        if previous != details:
            raise RuntimeError(f"pip report contains conflicting selections for {filename}")
    if not selected:
        raise RuntimeError("pip wheelhouse validation report selected no dependencies")
    return selected


def validate_wheelhouse(
    wheelhouse: Path,
    *,
    uv_lock: Path,
    locked_requirements: Path,
    pip_report: Path,
    project_name: str,
    project_version: str,
) -> dict[str, Any]:
    """Reject missing, extra, or hash-drifted wheels and return a deterministic manifest."""

    assert_no_reparse_tree(wheelhouse)
    locked = locked_wheel_hashes(uv_lock)
    selected = selected_wheels_from_pip_report(pip_report)
    actual_paths = {path.name: path for path in wheelhouse.glob("*.whl") if path.is_file()}
    project_prefix = f"{project_name.replace('-', '_')}-{project_version}-".lower()
    project_wheels = sorted(name for name in actual_paths if name.lower().startswith(project_prefix))
    if len(project_wheels) != 1:
        raise RuntimeError(
            f"Wheelhouse must contain exactly one {project_name} {project_version} wheel; found {project_wheels}"
        )
    project_wheel = project_wheels[0]
    if project_wheel in selected:
        raise RuntimeError("The first-party project wheel must not be supplied by the dependency lock export")

    expected_names = set(selected) | {project_wheel}
    actual_names = set(actual_paths)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise RuntimeError(f"Wheelhouse exact-set mismatch; missing={missing}, extra={extra}")

    wheels: list[dict[str, Any]] = []
    for name in sorted(actual_paths):
        path = actual_paths[name]
        actual_hash = digest(path)
        if name == project_wheel:
            wheels.append(
                {
                    "name": name,
                    "size": path.stat().st_size,
                    "sha256": actual_hash,
                    "origin": "first-party-source-build",
                    "package": project_name,
                    "version": project_version,
                }
            )
            continue
        report_item = selected[name]
        if actual_hash != report_item["sha256"]:
            raise RuntimeError(f"Wheel hash drift from pip validation report: {name}")
        if locked.get(name) != actual_hash:
            raise RuntimeError(f"Wheel hash or filename is not bound to uv.lock: {name}")
        wheels.append(
            {
                "name": name,
                "size": path.stat().st_size,
                "sha256": actual_hash,
                "origin": "uv.lock",
                "package": report_item["package"],
                "version": report_item["version"],
            }
        )

    return {
        "schema": WHEELHOUSE_MANIFEST_SCHEMA,
        "platform": "win_amd64",
        "python": "3.12",
        "abi": "cp312",
        "uv_lock_sha256": digest(uv_lock),
        "locked_requirements_sha256": digest(locked_requirements),
        "wheel_count": len(wheels),
        "wheels": wheels,
    }


def validate_builder_metadata(
    bundle: Path,
    metadata: dict[str, Any],
    *,
    commit: str,
    source_date_epoch: int,
    uv_lock: Path,
    locked_requirements: Path,
) -> None:
    missing = sorted(REQUIRED_BUILDER_FIELDS - metadata.keys())
    if missing:
        raise RuntimeError(f"Builder metadata is missing required fields: {missing}")
    if metadata["schema"] != BUILDER_METADATA_SCHEMA:
        raise RuntimeError(f"Unsupported builder metadata schema: {metadata['schema']}")
    if metadata["git_commit"] != commit or int(metadata["source_date_epoch"]) != source_date_epoch:
        raise RuntimeError("Builder metadata is not bound to the requested commit/source epoch")
    for field in (
        "source_digest",
        "next_tree_sha256",
        "toolchain_contract_sha256",
        "release_contract_sha256",
        "package_lock_sha256",
        "uv_lock_sha256",
        "locked_requirements_sha256",
    ):
        _sha256_value(metadata[field], field=field)
    if metadata["uv_lock_sha256"] != digest(uv_lock):
        raise RuntimeError("Builder metadata uv.lock hash does not match the validated lock")
    if metadata["locked_requirements_sha256"] != digest(locked_requirements):
        raise RuntimeError("Builder metadata requirements hash does not match the validated export")

    build_id_path = bundle / "app/.next/BUILD_ID"
    if not build_id_path.is_file():
        raise RuntimeError("Required Next BUILD_ID is missing")
    build_id = build_id_path.read_text(encoding="utf-8").strip()
    if not build_id or metadata["build_id"] != build_id:
        raise RuntimeError("Builder metadata is not bound to the staged Next BUILD_ID")
    if metadata["next_tree_sha256"] != tree_digest(bundle / "app/.next"):
        raise RuntimeError("Builder metadata is not bound to the staged .next tree")
    bound_files = {
        "toolchain_contract_sha256": bundle / "metadata/toolchain.json",
        "release_contract_sha256": bundle / "metadata/release-contract.json",
        "package_lock_sha256": bundle / "metadata/package-lock.json",
        "uv_lock_sha256": bundle / "runtime/python/uv.lock",
        "locked_requirements_sha256": bundle / "metadata/python-locked-requirements.txt",
    }
    for field, path in bound_files.items():
        if not path.is_file() or digest(path) != metadata[field]:
            raise RuntimeError(f"Builder metadata binding does not match {path.relative_to(bundle)}")


def build_receipt(builder_metadata: dict[str, Any], wheelhouse_manifest_path: Path) -> dict[str, Any]:
    receipt = json.loads(json.dumps(builder_metadata))
    receipt["schema"] = BUILD_RECEIPT_SCHEMA
    receipt["wheelhouse_manifest_sha256"] = digest(wheelhouse_manifest_path)
    receipt["wheelhouse"] = _load_json_object(wheelhouse_manifest_path, label="wheelhouse manifest")
    if "published_utc" in receipt or "published_at" in receipt:
        raise RuntimeError("Published time must not be embedded in the reproducible build receipt")
    return receipt


def normalize_generated_json(
    bundle: Path,
    *,
    commit: str,
    source_date_epoch: int,
    receipt: dict[str, Any] | None = None,
) -> None:
    """Remove nondeterminism emitted by concurrent Next/npm build tooling."""

    for relative in sorted(CANONICAL_JSON_PATHS):
        path = bundle / relative
        if not path.is_file():
            raise RuntimeError(f"Required generated manifest is missing: {relative}")
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        path.write_text(_canonical_json(payload), encoding="utf-8")

    sbom_path = bundle / "metadata/node-sbom.cdx.json"
    if not sbom_path.is_file():
        raise RuntimeError("Required generated SBOM is missing: metadata/node-sbom.cdx.json")
    sbom = json.loads(sbom_path.read_text(encoding="utf-8-sig"))
    build_utc = datetime.fromtimestamp(source_date_epoch, timezone.utc).isoformat().replace("+00:00", "Z")
    serial_seed = commit if receipt is None else f"evomind:{commit}:{receipt['source_digest']}"
    sbom["serialNumber"] = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, serial_seed)}"
    sbom_metadata = sbom.setdefault("metadata", {})
    sbom_metadata["timestamp"] = build_utc
    if receipt is not None:
        property_values = {
            "evomind:build-id": receipt["build_id"],
            "evomind:build-receipt-sha256": digest(bundle / "metadata/build-receipt.json"),
            "evomind:git-commit": receipt["git_commit"],
            "evomind:release-contract-sha256": receipt["release_contract_sha256"],
            "evomind:source-date-epoch": str(receipt["source_date_epoch"]),
            "evomind:source-digest": receipt["source_digest"],
            "evomind:toolchain-contract-sha256": receipt["toolchain_contract_sha256"],
            "evomind:wheelhouse-manifest-sha256": receipt["wheelhouse_manifest_sha256"],
        }
        existing = [
            item
            for item in sbom_metadata.get("properties", [])
            if not str(item.get("name", "")).startswith("evomind:")
        ]
        existing.extend({"name": key, "value": str(value)} for key, value in sorted(property_values.items()))
        sbom_metadata["properties"] = sorted(existing, key=lambda item: (item["name"], str(item.get("value", ""))))
    sbom_path.write_text(_canonical_json(sbom), encoding="utf-8")


def _prepare_output_path(path: Path, *, label: str) -> Path:
    target = path.absolute()
    existing_parent = target.parent
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    if is_reparse_point(existing_parent):
        raise RuntimeError(f"{label} parent is a reparse point: {existing_parent}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"Refusing to overwrite existing {label}: {target}")
    return target


def finalize_release(args: argparse.Namespace) -> dict[str, Any]:
    bundle_argument = args.bundle.absolute()
    if is_reparse_point(bundle_argument):
        raise RuntimeError(f"Bundle root is a reparse point: {bundle_argument}")
    bundle = bundle_argument.resolve(strict=True)
    assert_no_reparse_tree(bundle)

    wheelhouse = args.wheelhouse.resolve(strict=True)
    expected_wheelhouse = (bundle / "runtime/wheels").resolve(strict=True)
    if wheelhouse != expected_wheelhouse:
        raise RuntimeError("Wheelhouse validation must target bundle/runtime/wheels")
    builder_metadata_path = args.builder_metadata.resolve(strict=True)
    expected_builder_metadata = (bundle / "metadata/builder-metadata.json").resolve(strict=True)
    if builder_metadata_path != expected_builder_metadata:
        raise RuntimeError("Builder metadata must be staged at metadata/builder-metadata.json")

    builder_metadata = _load_json_object(builder_metadata_path, label="builder metadata")
    validate_builder_metadata(
        bundle,
        builder_metadata,
        commit=args.commit,
        source_date_epoch=args.source_date_epoch,
        uv_lock=args.uv_lock,
        locked_requirements=args.locked_requirements,
    )
    wheelhouse_manifest = validate_wheelhouse(
        wheelhouse,
        uv_lock=args.uv_lock,
        locked_requirements=args.locked_requirements,
        pip_report=args.pip_report,
        project_name=args.project_name,
        project_version=args.version,
    )
    wheelhouse_manifest_path = wheelhouse / "wheelhouse-manifest.json"
    if wheelhouse_manifest_path.exists() or wheelhouse_manifest_path.is_symlink():
        raise RuntimeError("Refusing to overwrite a pre-existing wheelhouse manifest")
    wheelhouse_manifest_path.write_text(_canonical_json(wheelhouse_manifest), encoding="utf-8")

    receipt = build_receipt(builder_metadata, wheelhouse_manifest_path)
    receipt_path = bundle / "metadata/build-receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise RuntimeError("Refusing to overwrite a pre-existing build receipt")
    receipt_path.write_text(_canonical_json(receipt), encoding="utf-8")
    normalize_generated_json(
        bundle,
        commit=args.commit,
        source_date_epoch=args.source_date_epoch,
        receipt=receipt,
    )
    assert_no_reparse_tree(bundle)

    generated_paths = (
        bundle / "release-manifest.json",
        bundle / "release-manifest.sha256",
        bundle / "SHA256SUMS.json",
    )
    if any(path.exists() or path.is_symlink() for path in generated_paths):
        raise RuntimeError("Refusing to overwrite pre-existing release manifest output")

    files: list[dict[str, Any]] = []
    total = 0
    for path in sorted((item for item in bundle.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        relative = path.relative_to(bundle).as_posix()
        parts = PurePosixPath(relative).parts
        if (
            (parts and parts[0] in FORBIDDEN_TOP_LEVEL)
            or "__pycache__" in parts
            or path.name in FORBIDDEN_NAMES
            or path.name.startswith(".next-")
            or ("node_modules" in parts and parts[0] != "app")
        ):
            raise RuntimeError(f"Forbidden release payload: {relative}")
        size = path.stat().st_size
        total += size
        files.append({"path": relative, "size": size, "bytes": size, "sha256": digest(path)})

    limit = args.max_mib * 1024 * 1024
    if total > limit:
        raise RuntimeError(f"Bundle exceeds {args.max_mib} MiB: {total} bytes")
    aggregate = hashlib.sha256()
    for item in files:
        aggregate.update(f"{item['path']}\0{item['sha256']}\0{item['size']}\n".encode())
    build_utc = datetime.fromtimestamp(args.source_date_epoch, timezone.utc).isoformat().replace("+00:00", "Z")
    manifest = {
        "schema": "evomind.windows.bundle.v1",
        "schema_version": 1,
        "product": "EvoMind",
        "version": args.version,
        "platform": "win-x64",
        "runtime_compatibility": {
            "node": {"bundled": args.node_version, "minimum": "20.9.0"},
            "python": {"bundled": args.python_version, "minimum": "3.12", "maximum_exclusive": "3.13"},
        },
        "git_commit": args.commit,
        "source_digest": receipt["source_digest"],
        "source_date_epoch": args.source_date_epoch,
        "build_utc": build_utc,
        "build_id": receipt["build_id"],
        "build_receipt_sha256": digest(receipt_path),
        "toolchain_contract_sha256": receipt["toolchain_contract_sha256"],
        "release_contract_sha256": receipt["release_contract_sha256"],
        "wheelhouse_manifest_sha256": receipt["wheelhouse_manifest_sha256"],
        "payload_bytes": total,
        "payload_mib": round(total / 1024 / 1024, 3),
        "file_count": len(files),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": files,
    }
    manifest_path = bundle / "release-manifest.json"
    manifest_path.write_text(_canonical_json(manifest), encoding="utf-8")
    manifest_hash = digest(manifest_path)
    (bundle / "release-manifest.sha256").write_text(
        f"{manifest_hash}  release-manifest.json\n", encoding="ascii"
    )
    sums = {item["path"]: item["sha256"] for item in files}
    sums["release-manifest.json"] = manifest_hash
    (bundle / "SHA256SUMS.json").write_text(_canonical_json(sums), encoding="utf-8")

    zip_path = _prepare_output_path(args.zip, label="release ZIP")
    temporary_zip = zip_path.with_name(f".{zip_path.name}.tmp-{uuid.uuid4().hex}")
    fixed_time = normalized_datetime(args.source_date_epoch)
    try:
        with zipfile.ZipFile(temporary_zip, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted((item for item in bundle.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
                relative = f"{bundle.name}/{path.relative_to(bundle).as_posix()}"
                info = zipfile.ZipInfo(relative, fixed_time)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        os.rename(temporary_zip, zip_path)
    finally:
        temporary_zip.unlink(missing_ok=True)

    result = {
        "ok": True,
        "bundle": str(bundle),
        "zip": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": digest(zip_path),
        "manifest_sha256": manifest_hash,
        **{
            key: manifest[key]
            for key in (
                "version",
                "source_digest",
                "build_id",
                "build_receipt_sha256",
                "payload_bytes",
                "payload_mib",
                "file_count",
                "aggregate_sha256",
            )
        },
    }
    if args.result:
        result_path = _prepare_output_path(args.result, label="build result")
        _atomic_write_text(result_path, _canonical_json(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--project-name", default="xcientist")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    parser.add_argument("--node-version", required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--builder-metadata", type=Path, required=True)
    parser.add_argument("--uv-lock", type=Path, required=True)
    parser.add_argument("--locked-requirements", type=Path, required=True)
    parser.add_argument("--pip-report", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--max-mib", type=int, default=500)
    parser.add_argument("--result", type=Path)
    return parser


def main() -> int:
    finalize_release(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
