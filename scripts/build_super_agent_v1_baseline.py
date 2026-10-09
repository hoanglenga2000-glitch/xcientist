from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SENSITIVE_PARTS = {
    ".env",
    ".ssh",
    "credential",
    "credentials",
    "password",
    "passwords",
    "secret",
    "secrets",
    "token",
    "tokens",
}

CORE_PATHS = (
    "src/evomind_runtime/models.py",
    "src/evomind_runtime/policy.py",
    "src/evomind_runtime/runtime.py",
    "src/evomind_runtime/store.py",
    "src/evomind_runtime/tools.py",
    "src/evomind_runtime/assistant_runs.py",
    "src/evomind_runtime/competition_data.py",
    "src/evomind_runtime/hpc_runtime_overlay.py",
)


def _run(root: Path, *argv: str) -> bytes:
    return subprocess.check_output(argv, cwd=root)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sensitive(path: str) -> bool:
    folded = path.replace("\\", "/").casefold()
    return any(part in SENSITIVE_PARTS for part in folded.split("/"))


def _status_rows(root: Path) -> list[dict[str, object]]:
    payload = _run(root, "git", "status", "--porcelain=v1", "-z", "--untracked-files=all")
    records = payload.decode("utf-8", "surrogateescape").split("\0")
    rows: list[dict[str, object]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        status, path = record[:2], record[3:]
        if status[0] in {"R", "C"} and index < len(records):
            path = records[index]
            index += 1
        rows.append(_file_row(root, path, status=status))
    return rows


def _file_row(root: Path, relative: str, *, status: str) -> dict[str, object]:
    normalized = relative.replace("\\", "/")
    if _sensitive(normalized):
        return {
            "status": status,
            "sensitive_path_redacted": True,
            "path_sha256": _sha256_bytes(normalized.encode("utf-8", "surrogateescape")),
        }
    target = (root / relative).resolve(strict=False)
    row: dict[str, object] = {"status": status, "path": normalized}
    try:
        stat = target.lstat()
    except OSError:
        row["exists"] = False
        return row
    row.update(exists=True, symlink=target.is_symlink())
    if target.is_file() and not target.is_symlink():
        data = target.read_bytes()
        row.update(bytes=stat.st_size, sha256=_sha256_bytes(data))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a content-free dirty-tree baseline for Super Agent V1.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve(strict=True)
    output = Path(args.output).resolve(strict=False)
    status = _status_rows(root)
    known = {str(row.get("path") or "") for row in status}
    core = [
        _file_row(root, path, status="core")
        for path in CORE_PATHS
        if path not in known
    ]
    manifest = {
        "schema": "evomind.super_agent_v1_worktree_baseline.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root_sha256": _sha256_bytes(os.fsencode(str(root))),
        "git_head": _run(root, "git", "rev-parse", "HEAD").decode().strip(),
        "git_branch": _run(root, "git", "branch", "--show-current").decode().strip(),
        "status_count": len(status),
        "sensitive_path_count": sum(bool(row.get("sensitive_path_redacted")) for row in status),
        "entries": sorted(status + core, key=lambda row: str(row.get("path") or row.get("path_sha256") or "")),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "status_count": manifest["status_count"],
        "sensitive_path_count": manifest["sensitive_path_count"],
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
