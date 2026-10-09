"""Hash-first dependency lock, independent of untrusted generated shell code."""
from __future__ import annotations

import re
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any


INSTALL_RECEIPT_SCHEMA = "evomind.dependency_install.v1"


def validate_install_receipt(output: str, expected_target: str, pins: dict[str, str]) -> dict[str, Any]:
    """Reject ambiguous, unbound or non-JSON executor output without echoing it."""
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate_receipt_key")
            value[key] = item
        return value

    try:
        if not isinstance(output, str) or len(output.encode("utf-8")) > 64 * 1024:
            raise ValueError("receipt_size")
        receipt = json.loads(output, object_pairs_hook=unique_object)
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != INSTALL_RECEIPT_SCHEMA
            or receipt.get("status") != "ready"
            or receipt.get("target") != expected_target
            or receipt.get("requested_pins") != pins
            or receipt.get("hashes_before_install") is not True
            or type(receipt.get("reused")) is not bool
            or not isinstance(receipt.get("lock_sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", receipt["lock_sha256"])
        ):
            raise ValueError("receipt_binding")
        return receipt
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ValueError("managed_dependency_receipt_invalid") from None


def installation_receipt(target: Path, lock_path: Path, pins: dict[str, str], *, reused: bool) -> dict[str, Any]:
    return {
        "schema": INSTALL_RECEIPT_SCHEMA,
        "status": "ready",
        "target": str(target),
        "requested_pins": dict(pins),
        "hashes_before_install": True,
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "reused": reused,
    }


def lock_pip_report(report: dict[str, Any]) -> dict[str, Any]:
    rows = []
    names: set[str] = set()
    for item in report.get("install", []):
        metadata = item.get("metadata") or {}
        download = item.get("download_info") or {}
        url = urlsplit(str(download.get("url") or ""))
        digest = (download.get("archive_info", {}).get("hashes") or {}).get("sha256", "")
        name, version = str(metadata.get("name", "")), str(metadata.get("version", ""))
        if url.scheme != "https" or url.hostname not in {"files.pythonhosted.org", "download.pytorch.org"} or url.username or url.password or url.query or not url.path.endswith(".whl"):
            raise ValueError("dependency_source_not_official_wheel")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+!-]*", version) or not re.fullmatch("[a-f0-9]{64}", digest):
            raise ValueError("dependency_lock_missing_exact_version_or_hash")
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        if canonical in names:
            raise ValueError("duplicate_dependency")
        names.add(canonical)
        rows.append({"name": canonical, "version": version, "sha256": digest, "url": url.geturl()})
    if not rows:
        raise ValueError("empty_dependency_lock")
    return {"schema": "evomind.dependency_lock.v1", "hashes_before_install": True, "files": sorted(rows, key=lambda row: row["name"])}


def requirements_text(lock: dict[str, Any]) -> str:
    if lock.get("schema") != "evomind.dependency_lock.v1" or lock.get("hashes_before_install") is not True:
        raise ValueError("unverified_dependency_lock")
    # Revalidate the exact URLs/hash strings before constructing a pip input.
    validated = lock_pip_report({"install": [{"metadata": {"name": row["name"], "version": row["version"]}, "download_info": {"url": row["url"], "archive_info": {"hashes": {"sha256": row["sha256"]}}}} for row in lock["files"]]})
    return "\n".join(f"{row['name']}=={row['version']} --hash=sha256:{row['sha256']}" for row in validated["files"]) + "\n"


def prepare_target(root: Path, pins: dict[str, str]) -> dict[str, Any]:
    """Resolve first, lock official wheels, then install offline into a new target."""
    if not pins or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+!-]*", version) for name, version in pins.items()):
        raise ValueError("exact_dependency_pins_required")
    if root.is_symlink():
        raise ValueError("dependency_target_symlink")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("resolver-report.json", "dependency-lock.json", "requirements.lock", "installed-files.json", "wheels", "site-packages", "cache"):
        if (root / name).is_symlink():
            raise ValueError("dependency_state_symlink")
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith("PIP_"):
            environment.pop(key)
    environment.update({"PIP_CONFIG_FILE": os.devnull, "PIP_CACHE_DIR": str(root / "cache"), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    report_path = root / "resolver-report.json"
    lock_path = root / "dependency-lock.json"
    req_path = root / "requirements.lock"
    wheelhouse = root / "wheels"
    target = root / "site-packages"
    def command(*arguments: str) -> None:
        subprocess.run([sys.executable, "-m", "pip", "--isolated", "--cache-dir", str(root / "cache"), "--disable-pip-version-check", *arguments], env=environment, check=True, timeout=900, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not lock_path.exists():
        command("install", "--dry-run", "--ignore-installed", "--only-binary=:all:", "--index-url", "https://pypi.org/simple", "--report", str(report_path), *[f"{name}=={version}" for name, version in sorted(pins.items())])
        lock = lock_pip_report(json.loads(report_path.read_text(encoding="utf-8")))
        locked_versions = {item["name"]: item["version"] for item in lock["files"]}
        if any(locked_versions.get(re.sub(r"[-_.]+", "-", name).lower()) != version for name, version in pins.items()):
            raise ValueError("resolver_changed_requested_pins")
        lock_path.write_text(json.dumps(lock, sort_keys=True, indent=2), encoding="utf-8")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    locked_versions = {item["name"]: item["version"] for item in lock.get("files", [])}
    if any(locked_versions.get(re.sub(r"[-_.]+", "-", name).lower()) != version for name, version in pins.items()):
        raise ValueError("dependency_lock_pin_mismatch")
    req_path.write_text(requirements_text(lock), encoding="utf-8")
    manifest_path = root / "installed-files.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = []
        for path in target.rglob("*"):
            if path.is_symlink():
                raise ValueError("installed_dependency_symlink")
            if path.is_file():
                actual.append(path.relative_to(target).as_posix())
        if set(actual) != {row["name"] for row in manifest["files"]}:
            raise ValueError("installed_dependency_file_set_mismatch")
        for row in manifest["files"]:
            path = target / row["name"]
            path.resolve().relative_to(target.resolve())
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("installed_dependency_integrity_mismatch")
        return installation_receipt(target, lock_path, pins, reused=True)
    if target.exists() and any(target.iterdir()):
        raise ValueError("incomplete_environment_preserved_requires_reconciliation")
    wheelhouse.mkdir(exist_ok=True)
    command("download", "--require-hashes", "--only-binary=:all:", "--index-url", "https://pypi.org/simple", "--dest", str(wheelhouse), "-r", str(req_path))
    expected_hashes = {row["sha256"] for row in lock["files"]}
    actual_hashes = set()
    for wheel in wheelhouse.iterdir():
        if wheel.is_symlink() or not wheel.name.endswith(".whl"):
            raise ValueError("unexpected_wheelhouse_entry")
        actual_hashes.add(hashlib.sha256(wheel.read_bytes()).hexdigest())
    if expected_hashes != actual_hashes:
        raise ValueError("wheelhouse_hash_mismatch")
    command("install", "--no-index", "--find-links", str(wheelhouse), "--require-hashes", "--no-compile", "--target", str(target), "-r", str(req_path))
    files = []
    for path in sorted(target.rglob("*")):
        if path.is_symlink():
            raise ValueError("installed_dependency_symlink")
        if path.is_file():
            files.append({"name": path.relative_to(target).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest_path.write_text(json.dumps({"files": files}, sort_keys=True), encoding="utf-8")
    return installation_receipt(target, lock_path, pins, reused=False)


if __name__ == "__main__":
    managed_root = Path(sys.argv[1])
    managed_root.resolve().relative_to(Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"))
    print(json.dumps(prepare_target(managed_root, json.loads(sys.argv[2]))))
