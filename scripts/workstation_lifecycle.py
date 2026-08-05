from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

MARKER_NAME = ".workstation-install.json"
WINDOWS_DEVICE = re.compile(r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)", re.IGNORECASE)


def lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.path.normpath(str(path.expanduser()))))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def safe_relative(value: str) -> Path:
    raw = str(value)
    normalized = raw.replace("\\", "/")
    if (
        not normalized
        or "\x00" in normalized
        or normalized.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", normalized)
    ):
        raise ValueError(f"unsafe package path: {value!r}")
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe package path: {value!r}")
    for part in parts:
        if ":" in part:
            raise ValueError(f"alternate data stream is forbidden: {value!r}")
        if part.endswith((".", " ")):
            raise ValueError(f"trailing dot or space is forbidden: {value!r}")
        if WINDOWS_DEVICE.match(part):
            raise ValueError(f"reserved Windows device name is forbidden: {value!r}")
    path = Path(*parts)
    if path.parts[0].casefold() == "user-data":
        raise ValueError("release manifest must not contain user-data")
    return path


def contained(root: Path, target: Path, *, allow_root: bool = False) -> Path:
    base = lexical_absolute(root)
    candidate = lexical_absolute(target)
    try:
        relative = candidate.relative_to(base)
    except ValueError as error:
        raise RuntimeError(f"path escapes managed root: {candidate}") from error
    if not allow_root and not relative.parts:
        raise RuntimeError(f"managed target must not be the root itself: {candidate}")
    return candidate


def reject_links(root: Path, target: Path, *, allow_missing: bool = True) -> Path:
    base = lexical_absolute(root)
    candidate = contained(base, target, allow_root=True)
    relative = candidate.relative_to(base)
    cursor = base
    for part in ((), *relative.parts):
        if part:
            cursor /= part
        if not cursor.exists():
            if allow_missing:
                continue
            raise RuntimeError(f"managed path does not exist: {cursor}")
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise RuntimeError(f"managed path traverses a symlink/reparse point: {cursor}")
    return candidate


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_release(package: Path) -> tuple[dict[str, Any], list[tuple[Path, str]]]:
    manifest_path = package / "release-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("release-manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    raw_files = manifest.get("files")
    entries: list[tuple[Path, str]] = []
    if isinstance(raw_files, dict):
        entries = [(safe_relative(path), str(checksum).lower()) for path, checksum in raw_files.items()]
    elif isinstance(raw_files, list):
        for item in raw_files:
            if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
                raise RuntimeError("invalid release manifest file entry")
            entries.append((safe_relative(str(item["path"])), str(item["sha256"]).lower()))
    else:
        raise RuntimeError("release manifest files must be a mapping or list")
    if not entries:
        raise RuntimeError("release manifest contains no files")
    # The manifest authenticates payload files; archive-level SHA-256 authenticates
    # the manifest itself. Treat it as a managed file so upgrades replace it.
    if not any(relative.as_posix().lower() == "release-manifest.json" for relative, _ in entries):
        entries.append((Path("release-manifest.json"), file_sha256(manifest_path)))
    seen: set[str] = set()
    for relative, expected in entries:
        normalized = relative.as_posix().casefold()
        if normalized in seen:
            raise RuntimeError(f"duplicate release path: {relative}")
        seen.add(normalized)
        source = reject_links(package, package / relative, allow_missing=False)
        if source.is_symlink() or getattr(source, "is_junction", lambda: False)() or not source.is_file():
            raise RuntimeError(f"release file missing or symbolic: {relative}")
        actual = file_sha256(source)
        if actual != expected:
            raise RuntimeError(f"release checksum mismatch: {relative}")
    return manifest, sorted(entries, key=lambda item: item[0].as_posix())


def load_marker(root: Path) -> dict[str, Any]:
    path = root / MARKER_NAME
    if not path.is_file():
        raise RuntimeError(f"{MARKER_NAME} is missing; refusing to alter an unmanaged directory")
    marker = json.loads(path.read_text(encoding="utf-8-sig"))
    if marker.get("format_version") != 1 or marker.get("product") != "research-workstation":
        raise RuntimeError("invalid installation marker")
    return marker


def backup_install(root: Path, marker: dict[str, Any], backups_root: Path) -> dict[str, Any]:
    backups_root = reject_links(backups_root, backups_root, allow_missing=True)
    release_root = backups_root / "release-upgrades"
    release_root.mkdir(parents=True, exist_ok=True)
    reject_links(backups_root, release_root, allow_missing=False)
    backup = release_root / f"upgrade-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    backup.mkdir(parents=True, exist_ok=False)
    backed_up: list[dict[str, Any]] = []
    for raw in marker.get("managed_files", []):
        relative = safe_relative(str(raw))
        source = reject_links(root, root / relative)
        if source.is_file():
            destination = backup / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            digest = file_sha256(destination)
            if digest != file_sha256(source):
                raise RuntimeError(f"release backup copy hash mismatch: {relative}")
            backed_up.append({"path": relative.as_posix(), "sha256": digest, "bytes": destination.stat().st_size})
    marker_path = root / MARKER_NAME
    shutil.copy2(marker_path, backup / MARKER_NAME)
    receipt = backup / "backup-manifest.json"
    atomic_json(receipt, {
        "format_version": 2,
        "root": str(root.resolve()),
        "install_id": marker.get("install_id"),
        "marker_sha256": file_sha256(backup / MARKER_NAME),
        "files": backed_up,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    return {"directory": backup, "receipt": receipt, "receipt_sha256": file_sha256(receipt)}


def remove_empty_parents(path: Path, root: Path) -> None:
    parent = path.parent
    while parent != root and parent != root.parent:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def restore_backup(
    root: Path,
    backup: Path,
    remove_current: bool = True,
    *,
    expected_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    receipt_path = backup / "backup-manifest.json"
    if expected_receipt_sha256 and file_sha256(receipt_path) != expected_receipt_sha256:
        raise RuntimeError("rollback backup receipt hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if receipt.get("format_version") != 2 or Path(str(receipt.get("root", ""))).resolve() != root.resolve():
        raise RuntimeError("rollback backup receipt target mismatch")
    previous = json.loads((backup / MARKER_NAME).read_text(encoding="utf-8-sig"))
    if file_sha256(backup / MARKER_NAME) != receipt.get("marker_sha256"):
        raise RuntimeError("rollback marker hash mismatch")
    records = receipt.get("files")
    if not isinstance(records, list):
        raise RuntimeError("rollback backup file ledger is invalid")
    record_map: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict):
            raise RuntimeError("rollback backup entry is invalid")
        relative = safe_relative(str(item.get("path", "")))
        source = reject_links(backup, backup / relative, allow_missing=False)
        if file_sha256(source) != item.get("sha256") or source.stat().st_size != item.get("bytes"):
            raise RuntimeError(f"rollback backup payload hash mismatch: {relative}")
        record_map[relative.as_posix()] = item
    current: dict[str, Any] = {}
    marker_path = root / MARKER_NAME
    if marker_path.is_file():
        current = json.loads(marker_path.read_text(encoding="utf-8-sig"))
    if remove_current:
        for raw in current.get("managed_files", []):
            relative = safe_relative(str(raw))
            target = root / relative
            if target.is_file() or target.is_symlink():
                target.unlink(missing_ok=True)
                remove_empty_parents(target, root)
    for raw in previous.get("managed_files", []):
        relative = safe_relative(str(raw))
        source = backup / relative
        if source.is_file():
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.restore")
            shutil.copy2(source, temporary)
            expected = record_map.get(relative.as_posix(), {}).get("sha256")
            if not expected or file_sha256(temporary) != expected:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"staged rollback payload hash mismatch: {relative}")
            temporary.replace(destination)
    shutil.copy2(backup / MARKER_NAME, marker_path)
    return previous


def initialize(root: Path, package: Path, data_root: Path, backups_root: Path) -> dict[str, Any]:
    root = reject_links(root, root, allow_missing=True)
    package = reject_links(package, package, allow_missing=False)
    manifest, files = load_release(package)
    if root.resolve() == package.resolve():
        managed = [relative.as_posix() for relative, _ in files]
    else:
        root.mkdir(parents=True, exist_ok=True)
        for relative, _ in files:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(package / relative, destination)
        managed = [relative.as_posix() for relative, _ in files]
    data_root.mkdir(parents=True, exist_ok=True)
    backups_root.mkdir(parents=True, exist_ok=True)
    reject_links(data_root, data_root, allow_missing=False)
    reject_links(backups_root, backups_root, allow_missing=False)
    install_id = str(uuid.uuid4())
    trusted_marker_path = data_root.parent / ".install-marker.json"
    if trusted_marker_path.is_file():
        trusted = json.loads(trusted_marker_path.read_text(encoding="utf-8-sig"))
        if (
            trusted.get("schema") != "evomind.install_marker.v1"
            or Path(str(trusted.get("paths", {}).get("data", ""))).resolve() != data_root.resolve()
            or Path(str(trusted.get("paths", {}).get("backups", ""))).resolve() != backups_root.resolve()
        ):
            raise RuntimeError("trusted install marker does not match lifecycle paths")
        install_id = str(trusted.get("install_id"))
    marker = {
        "format_version": 1,
        "product": "research-workstation",
        "version": manifest.get("version", "unknown"),
        "release_id": manifest.get("release_id"),
        "install_id": install_id,
        "managed_files": managed,
        "data_root": str(data_root),
        "backups_root": str(backups_root),
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(root / MARKER_NAME, marker)
    return marker


def upgrade(root: Path, package: Path, backups_root: Path) -> dict[str, Any]:
    marker = load_marker(root)
    manifest, files = load_release(package)
    if marker.get("release_id") and marker.get("release_id") == manifest.get("release_id"):
        unchanged = all((root / relative).is_file() and file_sha256(root / relative) == checksum for relative, checksum in files)
        if unchanged:
            return {
                "status": "already_current",
                "version": marker.get("version"),
                "release_id": marker.get("release_id"),
                "managed_files": len(files),
            }
    backup_record = backup_install(root, marker, backups_root)
    backup = Path(backup_record["directory"])
    staged: list[Path] = []
    old_names = {safe_relative(str(raw)).as_posix() for raw in marker.get("managed_files", [])}
    new_names = {relative.as_posix() for relative, _ in files}
    try:
        for relative, _ in files:
            source = package / relative
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_suffix(destination.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.upgrade")
            shutil.copy2(source, temp)
            staged.append(temp)
        for (relative, _), temp in zip(files, staged, strict=True):
            destination = root / relative
            temp.replace(destination)
        for raw in marker.get("managed_files", []):
            relative = safe_relative(str(raw))
            if relative.as_posix() not in new_names:
                target = root / relative
                if target.is_file() or target.is_symlink():
                    target.unlink(missing_ok=True)
                    remove_empty_parents(target, root)
        new_marker = {
            "format_version": 1,
            "product": "research-workstation",
            "version": manifest.get("version", "unknown"),
            "release_id": manifest.get("release_id"),
            "install_id": marker.get("install_id"),
            "managed_files": sorted(new_names),
            "data_root": marker.get("data_root"),
            "backups_root": str(backups_root),
            "installed_at": marker.get("installed_at"),
            "upgraded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "rollback_backup": str(backup),
            "rollback_receipt": str(backup_record["receipt"]),
            "rollback_receipt_sha256": backup_record["receipt_sha256"],
        }
        atomic_json(root / MARKER_NAME, new_marker)
        return {"status": "upgraded", "version": new_marker["version"], "backup": str(backup), "managed_files": len(new_names)}
    except BaseException:
        for temp in staged:
            temp.unlink(missing_ok=True)
        for raw in new_names - old_names:
            target = root / safe_relative(raw)
            if target.is_file() or target.is_symlink():
                target.unlink(missing_ok=True)
                remove_empty_parents(target, root)
        restore_backup(root, backup, expected_receipt_sha256=str(backup_record["receipt_sha256"]))
        raise


def rollback(root: Path, backup_arg: Path | None, backups_root: Path) -> dict[str, Any]:
    marker = load_marker(root)
    recorded_backup = Path(str(marker.get("rollback_backup", ""))).expanduser().resolve() if marker.get("rollback_backup") else None
    if backup_arg:
        backup = backup_arg.expanduser().resolve()
        if recorded_backup is None or backup != recorded_backup:
            raise RuntimeError("explicit rollback backup is not bound by the current installation marker")
    elif marker.get("rollback_backup"):
        backup = recorded_backup
    else:
        candidates = sorted((backups_root / "release-upgrades").glob("upgrade-*"), reverse=True)
        if not candidates:
            raise RuntimeError("no rollback backup is available")
        backup = candidates[0]
    if backup is None:
        raise RuntimeError("no rollback backup is available")
    release_root = (backups_root / "release-upgrades").resolve()
    contained(release_root, backup)
    reject_links(release_root, backup, allow_missing=False)
    if not (backup / MARKER_NAME).is_file() or not (backup / "backup-manifest.json").is_file():
        raise RuntimeError("invalid rollback backup")
    expected_receipt = marker.get("rollback_receipt_sha256")
    if not isinstance(expected_receipt, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_receipt):
        raise RuntimeError("rollback receipt is not hash-bound by the installation marker")
    restored = restore_backup(root, backup, expected_receipt_sha256=expected_receipt)
    return {"status": "rolled_back", "version": restored.get("version"), "backup": str(backup)}


def uninstall(root: Path, purge_user_data: bool, data_root: Path) -> dict[str, Any]:
    marker = load_marker(root)
    recorded_data = Path(str(marker.get("data_root", ""))).expanduser().resolve()
    if recorded_data != data_root.resolve():
        raise RuntimeError("uninstall data root does not match the installation marker")
    purge_target: Path | None = None
    if purge_user_data and data_root.is_dir():
        trusted_path = data_root.parent / ".install-marker.json"
        if not trusted_path.is_file():
            raise RuntimeError("trusted canonical install marker is required for data purge")
        trusted = json.loads(trusted_path.read_text(encoding="utf-8-sig"))
        resolved = reject_links(data_root.parent, data_root, allow_missing=False)
        if (
            trusted.get("schema") != "evomind.install_marker.v1"
            or trusted.get("install_id") != marker.get("install_id")
            or Path(str(trusted.get("paths", {}).get("data", ""))).resolve() != resolved
            or resolved != data_root.parent / "data"
            or resolved == Path(resolved.anchor)
            or len(resolved.parts) < 3
        ):
            raise RuntimeError("managed data-root verification failed")
        purge_target = resolved
    removed = 0
    for raw in marker.get("managed_files", []):
        relative = safe_relative(str(raw))
        target = reject_links(root, root / relative)
        if target.is_file() or target.is_symlink():
            target.unlink(missing_ok=True)
            remove_empty_parents(target, root)
            removed += 1
    (root / MARKER_NAME).unlink(missing_ok=True)
    generated_removed: list[str] = []
    generated_residual: list[str] = []
    for generated in (root / ".env", root / ".venv"):
        try:
            if generated.is_dir():
                shutil.rmtree(generated)
                generated_removed.append(generated.name)
            elif generated.exists():
                generated.unlink()
                generated_removed.append(generated.name)
        except OSError:
            generated_residual.append(str(generated))
    data = data_root
    if purge_target is not None:
        shutil.rmtree(purge_target)
    if not purge_user_data and data.exists():
        atomic_json(data / "install-state.json", {
            "format_version": 1,
            "status": "uninstalled",
            "user_data_preserved": True,
            "uninstalled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
    return {
        "status": "uninstalled" if not generated_residual else "uninstalled_cleanup_pending",
        "removed_files": removed,
        "generated_removed": generated_removed,
        "generated_residual": generated_residual,
        "user_data_preserved": not purge_user_data and data.exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Atomic workstation release install, upgrade, rollback and uninstall helper.")
    parser.add_argument("command", choices=["init", "upgrade", "rollback", "uninstall", "status"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--backups-root", type=Path)
    parser.add_argument("--purge-user-data", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    data_root = (args.data_root or root / "user-data").expanduser().resolve()
    backups_root = (args.backups_root or data_root / "backups").expanduser().resolve()
    if args.command in {"init", "upgrade"} and not args.package:
        raise SystemExit("LIFECYCLE_FAILED: --package is required")
    try:
        if args.command == "init":
            result = {"status": "initialized", **initialize(root, args.package.expanduser().resolve(), data_root, backups_root)}
        elif args.command == "upgrade":
            result = upgrade(root, args.package.expanduser().resolve(), backups_root)
        elif args.command == "rollback":
            result = rollback(root, args.backup, backups_root)
        elif args.command == "uninstall":
            result = uninstall(root, args.purge_user_data, data_root)
        else:
            marker = load_marker(root)
            marker_data = Path(str(marker.get("data_root") or data_root)).expanduser().resolve()
            result = {"status": "installed", **marker, "root": str(root), "user_data_exists": marker_data.is_dir()}
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "message": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
