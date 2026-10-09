from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterable


SCHEMA = "evomind.g21_managed_asset_probe.v1"
PACKAGE_NAMES = ("textarena", "trueskill", "transformers", "torch")
MANIFEST_CANDIDATES = {
    "mindgames_formal": ("../.runtime/mindgames/formal-protocol.json",),
    "cure_encoder": (
        ".evomind/cure-bench/frozen-encoder.json",
        "managed_runtime/frozen-encoder.json",
    ),
    "primary_baseline": (
        ".evomind/primary-baseline.json",
        "managed_runtime/primary-baseline.json",
        "../.runtime/mindgames/primary-baseline.json",
    ),
    "holdout_ledger": (
        ".evomind/cure-bench/holdout-ledger.json",
        "managed_runtime/holdout-ledger.json",
        "../.runtime/mindgames/holdout-ledger.json",
    ),
}
MODEL_CANDIDATES = (
    "../.runtime/mindgames/models/Qwen3-8B",
    "../.runtime/mindgames/models/STARS",
    "../.runtime/mindgames/models/tungsten",
    "managed_runtime/models/Qwen3-8B",
    ".evomind/cure-bench/models",
)


class ProbeGate(ValueError):
    pass


def _is_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return path.is_symlink() or bool(flag and attributes & flag)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_bounded(path: Path, root: Path, *, must_exist: bool) -> Path:
    raw = path.expanduser().absolute()
    anchor = root.expanduser().absolute()
    if not _within(raw, anchor):
        raise ProbeGate("PATH_ESCAPE")
    current = anchor
    if _is_reparse(current):
        raise ProbeGate("REPARSE_ROOT_REJECTED")
    for part in raw.relative_to(anchor).parts:
        current = current / part
        if _is_reparse(current):
            raise ProbeGate("REPARSE_PATH_REJECTED")
    try:
        resolved = raw.resolve(strict=must_exist)
        resolved_root = anchor.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ProbeGate("REQUIRED_PATH_MISSING") from exc
    if not _within(resolved, resolved_root):
        raise ProbeGate("PATH_ESCAPE")
    return resolved


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _site_packages(runtime_root: Path, allowed_root: Path) -> tuple[Path, ...]:
    candidates = [runtime_root / "site-packages", runtime_root / "Lib/site-packages"]
    for library in (runtime_root / "lib", runtime_root / "lib64"):
        if library.is_dir() and not _is_reparse(library):
            for child in sorted(library.iterdir(), key=lambda value: value.name.casefold()):
                if child.is_dir() and child.name.casefold().startswith("python"):
                    candidates.append(child / "site-packages")
    result: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            result.append(_resolve_bounded(candidate, allowed_root, must_exist=True))
    return tuple(dict.fromkeys(result))


def _packages(site_packages: Iterable[Path], allowed_root: Path) -> list[dict[str, Any]]:
    sites = tuple(site_packages)
    rows: list[dict[str, Any]] = []
    for name in PACKAGE_NAMES:
        found: dict[str, Any] = {"name": name, "available": False, "version": None, "origin": None}
        for site in sites:
            spec = importlib.machinery.PathFinder.find_spec(name, [str(site)])
            if spec is None:
                continue
            origins = list(spec.submodule_search_locations or [])
            if spec.origin and spec.origin not in {"built-in", "frozen"}:
                origins.append(spec.origin)
            resolved_origins = [
                _resolve_bounded(Path(value), allowed_root, must_exist=True) for value in origins
            ]
            if not resolved_origins:
                raise ProbeGate("PACKAGE_ORIGIN_MISSING")
            version = None
            for distribution in importlib.metadata.distributions(path=[str(site)]):
                distribution_name = str(distribution.metadata.get("Name") or "").casefold().replace("-", "_")
                if distribution_name == name.casefold().replace("-", "_"):
                    version = distribution.version
                    break
            found = {
                "name": name,
                "available": True,
                "version": version,
                "origin": [_relative(value, allowed_root) for value in resolved_origins],
            }
            break
        rows.append(found)
    return rows


def _file_metadata(path: Path, root: Path) -> dict[str, Any]:
    metadata = path.stat()
    return {
        "path": _relative(path, root),
        "bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "regular_file": True,
        "content_read": False,
    }


def _model_closure(path: Path, allowed_root: Path) -> dict[str, Any]:
    root = _resolve_bounded(path, allowed_root, must_exist=True)
    if not root.is_dir():
        raise ProbeGate("MODEL_DIRECTORY_INVALID")
    files: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if _is_reparse(candidate):
            raise ProbeGate("REPARSE_PATH_REJECTED")
        if candidate.is_file():
            resolved = _resolve_bounded(candidate, allowed_root, must_exist=True)
            files.append(_file_metadata(resolved, root))
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "path": _relative(root, allowed_root),
        "file_count": len(files),
        "total_bytes": sum(int(row["bytes"]) for row in files),
        "metadata_closure_sha256": hashlib.sha256(canonical).hexdigest(),
        "content_sha256_computed": False,
        "files": files,
    }


def probe(data_dir: Path, out_dir: Path, allowed_remote_root: Path) -> dict[str, Any]:
    allowed = _resolve_bounded(allowed_remote_root, allowed_remote_root, must_exist=True)
    data = _resolve_bounded(data_dir, allowed, must_exist=True)
    if not data.is_dir():
        raise ProbeGate("DATA_DIRECTORY_INVALID")
    output = _resolve_bounded(out_dir, allowed, must_exist=False)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ProbeGate("OUTPUT_DIRECTORY_NOT_EMPTY")
    runtime_root = _resolve_bounded(data.parent / ".runtime/mindgames", allowed, must_exist=False)
    sites = _site_packages(runtime_root, allowed) if runtime_root.is_dir() else ()
    models: list[dict[str, Any]] = []
    for relative in MODEL_CANDIDATES:
        candidate = data / relative
        bounded = _resolve_bounded(candidate, allowed, must_exist=False)
        if bounded.is_dir():
            models.append(_model_closure(bounded, allowed))
    manifests: dict[str, list[dict[str, Any]]] = {}
    for role, relatives in MANIFEST_CANDIDATES.items():
        rows: list[dict[str, Any]] = []
        for relative in relatives:
            candidate = _resolve_bounded(data / relative, allowed, must_exist=False)
            if candidate.is_file():
                if _is_reparse(candidate):
                    raise ProbeGate("REPARSE_PATH_REJECTED")
                rows.append(_file_metadata(candidate, allowed))
        manifests[role] = rows
    return {
        "schema": SCHEMA,
        "status": "probe_completed",
        "allowed_remote_root": ".",
        "data_root": _relative(data, allowed),
        "packages": _packages(sites, allowed),
        "models": models,
        "manifest_candidates": manifests,
        "network_access": False,
        "test_labels_used": False,
        "file_contents_read": False,
        "signals_sent": 0,
        "other_processes_modified": False,
        "remote_writes": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only G21 managed asset metadata probe")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allowed-remote-root", default=os.environ.get("EVOMIND_ALLOWED_REMOTE_ROOT", ""))
    args = parser.parse_args()
    try:
        if not args.allowed_remote_root:
            raise ProbeGate("ALLOWED_REMOTE_ROOT_REQUIRED")
        output = Path(args.out_dir).expanduser().absolute()
        result = probe(Path(args.data_dir), output, Path(args.allowed_remote_root))
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "managed-asset-probe.json"
        with result_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
