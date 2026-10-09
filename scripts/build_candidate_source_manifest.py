#!/usr/bin/env python3
"""Build a deterministic SHA-256 inventory for an exported source candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256-output", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    checksum_output = args.sha256_output.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    for destination in (output, checksum_output):
        try:
            destination.relative_to(root)
        except ValueError:
            pass
        else:
            raise ValueError("manifest outputs must stay outside the candidate root")

    files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: path.relative_to(root).as_posix())
    casefolded: set[str] = set()
    records: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in files:
        if path.is_symlink():
            raise ValueError(f"candidate contains a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        folded = relative.casefold()
        if folded in casefolded:
            raise ValueError(f"candidate contains a case-insensitive path collision: {relative}")
        casefolded.add(folded)
        size = path.stat().st_size
        digest = sha256(path)
        records.append({"path": relative, "bytes": size, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
        total_bytes += size

    manifest = {
        "schema": "evomind.production_candidate.source_manifest.v1",
        "candidate_name": args.candidate_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "baseline_commit": args.baseline_commit,
        "source_index": str(args.source_index.expanduser().resolve()),
        "file_count": len(records),
        "total_bytes": total_bytes,
        "aggregate_sha256": aggregate.hexdigest(),
        "files": records,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    atomic_write(output, manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    atomic_write(checksum_output, f"{manifest_sha256}  {output.name}\n".encode("ascii"))
    print(json.dumps({
        "status": "passed",
        "manifest": str(output),
        "manifest_sha256": manifest_sha256,
        "sha256_file": str(checksum_output),
        "file_count": len(records),
        "total_bytes": total_bytes,
        "aggregate_sha256": manifest["aggregate_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
