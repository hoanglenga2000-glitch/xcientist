"""Read-only verification that a runtime release contains the G21 Goal code."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
FIXED_ALLOCATION = "G21"
GOAL_RUNTIME_FILES = (
    "evomind_runtime/competition_goal.py",
    "evomind_runtime/goal_board.py",
)
REQUIRED_RUNTIME_FILES = (
    "evomind_runtime/__init__.py",
    "evomind_runtime/assistant_runs.py",
    "evomind_runtime/competition_goal.py",
    "evomind_runtime/goal_board.py",
    "evomind_runtime/http_server.py",
    "evomind_runtime/runtime.py",
    "evomind_runtime/store.py",
)
SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")


@dataclass(frozen=True)
class ReleaseScopeResult:
    valid: bool
    manifest_path: str = ""
    manifest_schema: str = ""
    manifest_file_count: int = 0
    missing: tuple[str, ...] = ()
    invalid_entries: tuple[str, ...] = ()
    duplicate_entries: tuple[str, ...] = ()
    hash_mismatches: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "manifest_path": self.manifest_path,
            "manifest_schema": self.manifest_schema,
            "manifest_file_count": self.manifest_file_count,
            "missing": list(self.missing),
            "invalid_entries": list(self.invalid_entries),
            "duplicate_entries": list(self.duplicate_entries),
            "hash_mismatches": list(self.hash_mismatches),
            "issues": list(self.issues),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _normalize_manifest_path(value: Any) -> str:
    path = str(value or "").replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if path.startswith("src/"):
        path = path[4:]
    if path.startswith("support/python-runtime/"):
        path = path[len("support/python-runtime/"):]
    return path


def verify_release_scope(
    manifest: Mapping[str, Any],
    *,
    source_root: str | Path | None = None,
    manifest_path: str = "",
) -> ReleaseScopeResult:
    """Verify the manifest's regular-file closure includes the Goal runtime."""

    schema = str(manifest.get("schema") or "")
    raw_files = manifest.get("files")
    missing: list[str] = []
    invalid: list[str] = []
    duplicates: list[str] = []
    mismatches: list[str] = []
    issues: list[str] = []
    if manifest.get("run_id") != FIXED_RUN_ID:
        issues.append("RELEASE_RUN_ID_MISMATCH")
    if str(manifest.get("allocation") or "").casefold() != FIXED_ALLOCATION.casefold():
        issues.append("RELEASE_ALLOCATION_MISMATCH")
    for field in ("goal_spec_sha256", "human_baseline_evidence_sha256"):
        if not SHA256_RE.fullmatch(str(manifest.get(field) or "")):
            issues.append(f"RELEASE_{field.upper()}_INVALID")
    entries: dict[str, Mapping[str, Any]] = {}
    if not isinstance(raw_files, list):
        issues.append("RELEASE_MANIFEST_FILES_MISSING")
        raw_files = []
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            invalid.append("<non-object>")
            continue
        relative = _normalize_manifest_path(raw.get("path"))
        relative_path = Path(relative)
        if (
            not relative
            or relative.endswith("/")
            or relative_path.is_absolute()
            or relative_path.drive
            or ".." in relative_path.parts
        ):
            invalid.append(relative or "<empty>")
            continue
        if relative in entries:
            duplicates.append(relative)
            continue
        sha = str(raw.get("sha256") or "")
        if not SHA256_RE.fullmatch(sha):
            invalid.append(relative)
        entries[relative] = raw
    for required in REQUIRED_RUNTIME_FILES:
        if required not in entries:
            missing.append(required)
    if source_root is not None:
        root = Path(source_root).expanduser().resolve()
        for relative, entry in entries.items():
            path = (root / "src" / relative).resolve()
            try:
                path.relative_to(root / "src")
            except ValueError:
                mismatches.append(relative)
                continue
            if path.is_symlink() or not path.is_file() or _sha256(path) != str(entry.get("sha256") or "").lower():
                mismatches.append(relative)
    if manifest.get("file_count") is not None:
        try:
            declared_count = int(manifest.get("file_count"))
        except (TypeError, ValueError):
            declared_count = -1
        if declared_count != len(raw_files):
            issues.append("RELEASE_MANIFEST_COUNT_MISMATCH")
    if missing:
        issues.append("GOAL_RUNTIME_RELEASE_SCOPE_MISSING")
    if invalid:
        issues.append("RELEASE_MANIFEST_ENTRY_INVALID")
    if duplicates:
        issues.append("RELEASE_MANIFEST_DUPLICATE_ENTRY")
    if mismatches:
        issues.append("RELEASE_SOURCE_HASH_MISMATCH")
    return ReleaseScopeResult(
        valid=not (missing or invalid or duplicates or mismatches or issues),
        manifest_path=str(manifest_path),
        manifest_schema=schema,
        manifest_file_count=len(raw_files),
        missing=tuple(sorted(set(missing))),
        invalid_entries=tuple(sorted(set(invalid))),
        duplicate_entries=tuple(sorted(set(duplicates))),
        hash_mismatches=tuple(sorted(set(mismatches))),
        issues=tuple(dict.fromkeys(issues)),
    )


def load_and_verify_release_scope(
    path: str | Path,
    *,
    source_root: str | Path | None = None,
) -> ReleaseScopeResult:
    manifest_path = Path(path).expanduser().resolve(strict=True)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return ReleaseScopeResult(False, str(manifest_path), issues=("RELEASE_MANIFEST_LOAD_FAILED",))
    if not isinstance(manifest, Mapping):
        return ReleaseScopeResult(False, str(manifest_path), issues=("RELEASE_MANIFEST_ROOT_INVALID",))
    return verify_release_scope(manifest, source_root=source_root, manifest_path=str(manifest_path))


__all__ = [
    "FIXED_ALLOCATION",
    "FIXED_RUN_ID",
    "GOAL_RUNTIME_FILES",
    "REQUIRED_RUNTIME_FILES",
    "ReleaseScopeResult",
    "load_and_verify_release_scope",
    "verify_release_scope",
]
