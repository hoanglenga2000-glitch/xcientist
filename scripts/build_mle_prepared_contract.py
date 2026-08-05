#!/usr/bin/env python3
"""Build and verify immutable prepared-data contracts for MLE-Bench tasks.

The contract binds a prepared competition directory to the pinned upstream
MLE-Bench source, the official ZIP MD5 and every official public/private MD5.
It also records recursive file/byte totals.  SIIM additionally requires the
frozen public staging inventory to be complete by path and exact byte size.

No network, grader, campaign, training or submission operation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM_ROOT = ROOT / "external-projects" / "mle-bench"
PREPARED_CONTRACT_SCHEMA = "evomind.mle_bench.prepared_contract.v1"
SIIM_INVENTORY_SCHEMA = "evomind.siim.public_staging_inventory.v1"
SIIM_TASK = "siim-isic-melanoma-classification"
PINNED_MLEBENCH_COMMIT = "507f92e1138bb6e40dac5c6ee7a6758e6424bf97"
PINNED_MLEBENCH_VERSION = "1.0.0"
PINNED_SIIM_INVENTORY = {
    "manifest_sha256": "8fef392368fcc433f4532129312f3b2ef7118d76cad04dc70f694246c34eff13",
    "file_count": 33_129,
    "total_bytes": 25_765_345_055,
    "train_jpeg_count": 28_984,
    "test_jpeg_count": 4_142,
}

_MD5 = re.compile(r"^[a-f0-9]{32}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_TASK_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,95}$")


class PreparedContractError(RuntimeError):
    """Raised when prepared-data evidence is malformed or unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparedContractError(message)


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file() and not resolved.is_symlink(), f"{label} must be a regular file")
    return resolved


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    source = _regular_file(path, label=label)
    try:
        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparedContractError(f"invalid {label}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _safe_checksum_name(name: Any) -> str:
    normalized = str(name)
    candidate = PurePosixPath(normalized)
    _require(
        bool(normalized)
        and "\\" not in normalized
        and not candidate.is_absolute()
        and len(candidate.parts) == 1
        and candidate.parts[0] not in {"", ".", ".."},
        f"unsafe upstream checksum path: {normalized}",
    )
    return normalized


def load_upstream_checksums(path: Path) -> dict[str, Any]:
    source = _regular_file(path, label="upstream checksums.yaml")
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PreparedContractError("invalid upstream checksums.yaml") from exc
    _require(isinstance(value, dict), "upstream checksums must be an object")
    result: dict[str, Any] = {}
    zip_md5 = str(value.get("zip") or "").lower()
    _require(bool(_MD5.fullmatch(zip_md5)), "upstream ZIP MD5 is invalid")
    result["zip"] = zip_md5
    for visibility in ("public", "private"):
        raw = value.get(visibility)
        _require(isinstance(raw, dict) and bool(raw), f"upstream {visibility} MD5 map is missing")
        normalized: dict[str, str] = {}
        for raw_name, raw_digest in sorted(raw.items(), key=lambda item: str(item[0])):
            name = _safe_checksum_name(raw_name)
            digest = str(raw_digest or "").lower()
            _require(bool(_MD5.fullmatch(digest)), f"invalid upstream MD5: {visibility}/{name}")
            normalized[name] = digest
        result[visibility] = normalized
    return result


def recursive_tree_stats(root: Path) -> dict[str, int]:
    directory = root.expanduser().resolve()
    _require(directory.is_dir() and not directory.is_symlink(), f"prepared directory is missing: {directory}")
    file_count = 0
    total_bytes = 0
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        _require(not path.is_symlink(), f"prepared tree contains a symlink: {path}")
        if path.is_file():
            file_count += 1
            total_bytes += path.stat().st_size
        else:
            _require(path.is_dir(), f"prepared tree contains a non-file entry: {path}")
    return {"recursive_file_count": file_count, "recursive_bytes": total_bytes}


def _actual_md5_map(root: Path, expected: Mapping[str, str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    directory = root.expanduser().resolve()
    for name in sorted(expected):
        target = (directory / name).resolve()
        _require(target.parent == directory, f"checksum target escapes prepared directory: {name}")
        result[name] = md5_file(target) if target.is_file() and not target.is_symlink() else None
    return result


def _is_partial_name(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(".part") or ".part." in lowered


def _safe_siim_target(public_dir: Path, raw_name: Any) -> tuple[str, Path]:
    name = str(raw_name)
    relative = PurePosixPath(name)
    _require(
        bool(name) and "\\" not in name and not relative.is_absolute() and ".." not in relative.parts,
        f"unsafe SIIM inventory path: {name}",
    )
    parts = relative.parts
    allowed = name in {"train.csv", "test.csv", "sample_submission.csv"} or (
        len(parts) == 3
        and tuple(parts[:2]) in {("jpeg", "train"), ("jpeg", "test")}
        and parts[2].lower().endswith(".jpg")
    )
    _require(allowed, f"non-public SIIM inventory path: {name}")
    root = public_dir.expanduser().resolve()
    target = root.joinpath(*parts).resolve()
    _require(target == root or root in target.parents, f"SIIM inventory path escapes public root: {name}")
    return name, target


def scan_siim_inventory(
    inventory_path: Path,
    public_dir: Path,
    *,
    expected: Mapping[str, Any] = PINNED_SIIM_INVENTORY,
) -> dict[str, Any]:
    inventory = _read_json(inventory_path, label="SIIM public staging inventory")
    _require(inventory.get("schema") == SIIM_INVENTORY_SCHEMA, "SIIM inventory schema mismatch")
    _require(inventory.get("competition_id") == SIIM_TASK, "SIIM inventory competition mismatch")

    expected_manifest = str(expected.get("manifest_sha256") or "").lower()
    _require(bool(_SHA256.fullmatch(expected_manifest)), "pinned SIIM inventory manifest SHA-256 is invalid")
    expected_files = int(expected["file_count"])
    expected_bytes = int(expected["total_bytes"])
    expected_train = int(expected["train_jpeg_count"])
    expected_test = int(expected["test_jpeg_count"])
    entries = inventory.get("entries")
    _require(isinstance(entries, list), "SIIM inventory entries must be a list")

    seen: set[str] = set()
    parsed: list[tuple[str, Path, int]] = []
    entry_bytes = 0
    train_count = 0
    test_count = 0
    for item in entries:
        _require(isinstance(item, dict), "SIIM inventory entry must be an object")
        name, target = _safe_siim_target(public_dir, item.get("path"))
        _require(name not in seen, f"duplicate SIIM inventory path: {name}")
        seen.add(name)
        size = item.get("size")
        _require(isinstance(size, int) and not isinstance(size, bool) and size >= 0, f"invalid SIIM size: {name}")
        parsed.append((name, target, size))
        entry_bytes += size
        train_count += int(name.startswith("jpeg/train/"))
        test_count += int(name.startswith("jpeg/test/"))

    header_totals_match_entries = (
        int(inventory.get("file_count", -1)) == len(parsed)
        and int(inventory.get("total_bytes", -1)) == entry_bytes
        and int(inventory.get("train_jpeg_count", -1)) == train_count
        and int(inventory.get("test_jpeg_count", -1)) == test_count
    )
    pinned_totals_match = (
        inventory.get("manifest_sha256") == expected_manifest
        and len(parsed) == expected_files
        and entry_bytes == expected_bytes
        and train_count == expected_train
        and test_count == expected_test
    )

    missing: list[str] = []
    wrong_size: list[dict[str, Any]] = []
    completed_files = 0
    completed_bytes = 0
    for name, target, size in parsed:
        if target.is_symlink() or not target.is_file():
            missing.append(name)
            continue
        actual_size = target.stat().st_size
        if actual_size != size:
            wrong_size.append({"path": name, "expected_bytes": size, "actual_bytes": actual_size})
            continue
        completed_files += 1
        completed_bytes += actual_size

    root = public_dir.expanduser().resolve()
    part_files = []
    if root.is_dir() and not root.is_symlink():
        part_files = [
            path.relative_to(root).as_posix()
            for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
            if path.is_file() and _is_partial_name(path.name)
        ]

    status = (
        header_totals_match_entries
        and pinned_totals_match
        and completed_files == expected_files
        and completed_bytes == expected_bytes
        and not missing
        and not wrong_size
        and not part_files
    )
    return {
        "status": "verified" if status else "failed_closed",
        "inventory_sha256": sha256_file(_regular_file(inventory_path, label="SIIM inventory")),
        "inventory_manifest_sha256": inventory.get("manifest_sha256"),
        "expected_files": expected_files,
        "expected_bytes": expected_bytes,
        "expected_train_jpeg_count": expected_train,
        "expected_test_jpeg_count": expected_test,
        "inventory_files": len(parsed),
        "inventory_bytes": entry_bytes,
        "inventory_train_jpeg_count": train_count,
        "inventory_test_jpeg_count": test_count,
        "completed_files": completed_files,
        "completed_bytes": completed_bytes,
        "missing_files": len(missing),
        "wrong_size_files": len(wrong_size),
        "part_files": len(part_files),
        "missing_examples": missing[:20],
        "wrong_size_examples": wrong_size[:20],
        "part_examples": part_files[:20],
        "header_totals_match_entries": header_totals_match_entries,
        "pinned_totals_match": pinned_totals_match,
    }


def discover_upstream_commit(upstream_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(upstream_root.expanduser().resolve()), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    commit = completed.stdout.strip().lower()
    _require(bool(re.fullmatch(r"[a-f0-9]{40}", commit)), "upstream Git commit is invalid")
    return commit


def discover_upstream_version(upstream_root: Path) -> str:
    pyproject = _regular_file(upstream_root / "pyproject.toml", label="upstream pyproject.toml")
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r"(?m)^version\s*=\s*[\"']([^\"']+)[\"']\s*$", text)
    _require(match is not None, "upstream package version is missing")
    return str(match.group(1))


def build_prepared_contract(
    competition_root: Path,
    upstream_root: Path,
    *,
    upstream_commit: str | None = None,
    upstream_version: str | None = None,
    pinned_upstream_commit: str = PINNED_MLEBENCH_COMMIT,
    pinned_upstream_version: str = PINNED_MLEBENCH_VERSION,
    siim_inventory_path: Path | None = None,
    siim_expected: Mapping[str, Any] = PINNED_SIIM_INVENTORY,
) -> dict[str, Any]:
    competition = competition_root.expanduser().resolve()
    upstream = upstream_root.expanduser().resolve()
    task_id = competition.name
    _require(bool(_TASK_ID.fullmatch(task_id)), "competition directory name is not a safe task ID")
    _require(competition.is_dir() and not competition.is_symlink(), "competition root is missing")
    _require(upstream.is_dir() and not upstream.is_symlink(), "upstream root is missing")

    source_dir = (upstream / "mlebench" / "competitions" / task_id).resolve()
    _require(source_dir.parent == (upstream / "mlebench" / "competitions").resolve(), "upstream task path escaped")
    source_paths = {
        "prepare.py": _regular_file(source_dir / "prepare.py", label="upstream prepare.py"),
        "config.yaml": _regular_file(source_dir / "config.yaml", label="upstream config.yaml"),
        "checksums.yaml": _regular_file(source_dir / "checksums.yaml", label="upstream checksums.yaml"),
    }
    official = load_upstream_checksums(source_paths["checksums.yaml"])
    public_dir = competition / "prepared" / "public"
    private_dir = competition / "prepared" / "private"
    public_stats = recursive_tree_stats(public_dir)
    private_stats = recursive_tree_stats(private_dir)
    actual_public = _actual_md5_map(public_dir, official["public"])
    actual_private = _actual_md5_map(private_dir, official["private"])

    zip_files = sorted(
        path for path in competition.glob("*.zip") if path.is_file() and not path.is_symlink()
    )
    zip_path = zip_files[0] if len(zip_files) == 1 else None
    actual_zip = md5_file(zip_path) if zip_path is not None else None
    resolved_commit = (upstream_commit or discover_upstream_commit(upstream)).lower()
    resolved_version = upstream_version or discover_upstream_version(upstream)
    _require(bool(re.fullmatch(r"[a-f0-9]{40}", resolved_commit)), "upstream commit is invalid")
    _require(bool(str(resolved_version).strip()), "upstream version is invalid")

    public_matches = actual_public == official["public"]
    private_matches = actual_private == official["private"]
    zip_matches = len(zip_files) == 1 and actual_zip == official["zip"]
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    siim = None
    if task_id == SIIM_TASK:
        inventory = siim_inventory_path or (competition / "public_staging_inventory.json")
        siim = scan_siim_inventory(inventory, public_dir, expected=siim_expected)

    checks = {
        "upstream_commit_matches_pin": resolved_commit == pinned_upstream_commit,
        "upstream_version_matches_pin": resolved_version == pinned_upstream_version,
        "exactly_one_official_zip": len(zip_files) == 1,
        "official_zip_md5_matches": zip_matches,
        "official_public_md5_matches": public_matches,
        "official_private_md5_matches": private_matches,
        "siim_inventory_complete": siim is None or siim["status"] == "verified",
    }
    return {
        "schema": PREPARED_CONTRACT_SCHEMA,
        "schema_version": 1,
        "task_id": task_id,
        "status": "verified" if all(checks.values()) else "failed_closed",
        "upstream": {
            "repository_commit": resolved_commit,
            "package_version": resolved_version,
            "source_sha256": source_hashes,
        },
        "zip": {
            "file_name": zip_path.name if zip_path is not None else None,
            "bytes": zip_path.stat().st_size if zip_path is not None else None,
            "expected_md5": official["zip"],
            "actual_md5": actual_zip,
        },
        "public": {
            "expected_md5": official["public"],
            "actual_md5": actual_public,
            **public_stats,
        },
        "private": {
            "expected_md5": official["private"],
            "actual_md5": actual_private,
            **private_stats,
        },
        "siim_inventory": siim,
        "checks": checks,
    }


def verify_prepared_contract(
    contract_path: Path,
    competition_root: Path,
    upstream_root: Path,
    *,
    expected_upstream_commit: str = PINNED_MLEBENCH_COMMIT,
    expected_upstream_version: str = PINNED_MLEBENCH_VERSION,
    upstream_commit_override: str | None = None,
    upstream_version_override: str | None = None,
    siim_inventory_path: Path | None = None,
    siim_expected: Mapping[str, Any] = PINNED_SIIM_INVENTORY,
) -> dict[str, Any]:
    claimed = _read_json(contract_path, label="prepared-contract.json")
    actual_commit = upstream_commit_override or discover_upstream_commit(upstream_root)
    actual_version = upstream_version_override or discover_upstream_version(upstream_root)
    recomputed = build_prepared_contract(
        competition_root,
        upstream_root,
        upstream_commit=actual_commit,
        upstream_version=actual_version,
        pinned_upstream_commit=expected_upstream_commit,
        pinned_upstream_version=expected_upstream_version,
        siim_inventory_path=siim_inventory_path,
        siim_expected=siim_expected,
    )
    contract_matches = claimed == recomputed
    verified = claimed.get("schema") == PREPARED_CONTRACT_SCHEMA and contract_matches and recomputed["status"] == "verified"
    return {
        "status": "verified" if verified else "failed_closed",
        "contract_sha256": sha256_file(_regular_file(contract_path, label="prepared-contract.json")),
        "contract_matches_recomputed": contract_matches,
        "recomputed": recomputed,
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    try:
        descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PreparedContractError(f"prepared contract already exists: {target}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument("--upstream-commit")
    parser.add_argument("--upstream-version")
    parser.add_argument("--siim-inventory", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_prepared_contract(
        args.competition_root,
        args.upstream_root,
        upstream_commit=args.upstream_commit,
        upstream_version=args.upstream_version,
        siim_inventory_path=args.siim_inventory,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if payload["status"] != "verified":
        print(rendered)
        return 2
    output = args.output or (args.competition_root.expanduser().resolve() / "prepared-contract.json")
    _write_json_exclusive(output, payload)
    print(
        json.dumps(
            {**payload, "output": str(output.expanduser().resolve()), "output_sha256": sha256_file(output)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
