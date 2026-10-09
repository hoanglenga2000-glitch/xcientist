from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile


FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
FIXED_ALLOCATION = "G21"
REVIEW_SCHEMA = "evomind.g21_goal_patch_scope_review.v1"
DEPENDENT_FILES = (
    "evomind_runtime/__init__.py",
    "evomind_runtime/assistant_runs.py",
    "evomind_runtime/http_server.py",
    "evomind_runtime/runtime.py",
    "evomind_runtime/store.py",
)
GOAL_ONLY_FILES = (
    "evomind_runtime/competition_goal.py",
    "evomind_runtime/goal_board.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def audit_patch_scope(*, baseline_zip: Path, source_root: Path) -> dict[str, object]:
    baseline_zip = baseline_zip.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    baseline_sha = sha256_file(baseline_zip)
    files: list[dict[str, object]] = []
    missing: list[str] = []
    with tempfile.TemporaryDirectory(prefix="g21-goal-patch-scope-") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(baseline_zip) as archive:
            archive.extractall(extracted)
        for relative in (*DEPENDENT_FILES, *GOAL_ONLY_FILES):
            baseline = extracted / relative
            current = source_root / "src" / relative
            if not baseline.is_file() and relative in DEPENDENT_FILES:
                missing.append(f"baseline:{relative}")
                continue
            if not current.is_file() or current.is_symlink():
                missing.append(f"source:{relative}")
                continue
            old = baseline.read_text(encoding="utf-8", errors="replace").splitlines() if baseline.is_file() else []
            new = current.read_text(encoding="utf-8", errors="replace").splitlines()
            diff = list(difflib.unified_diff(old, new, fromfile=f"baseline/{relative}", tofile=f"source/{relative}", lineterm="\n"))
            added = [line for line in diff if line.startswith("+") and not line.startswith("+++")]
            deleted = [line for line in diff if line.startswith("-") and not line.startswith("---")]
            files.append(
                {
                    "path": relative,
                    "baseline_sha256": sha256_file(baseline) if baseline.is_file() else None,
                    "source_sha256": sha256_file(current),
                    "baseline_bytes": baseline.stat().st_size if baseline.is_file() else 0,
                    "source_bytes": current.stat().st_size,
                    "added_lines": len(added),
                    "deleted_lines": len(deleted),
                    "diff_sha256": sha256_bytes("\n".join(diff).encode("utf-8")),
                    "new_file": not baseline.is_file(),
                }
            )
    return {
        "schema": REVIEW_SCHEMA,
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "baseline_zip_sha256": baseline_sha,
        "dependent_files": list(DEPENDENT_FILES),
        "goal_only_files": list(GOAL_ONLY_FILES),
        "files": files,
        "missing": missing,
        "status": "BLOCKED" if missing else "REVIEW_REQUIRED",
        "production_deployable": False,
        "reviewed_goal_only": False,
        "reason": "dependent runtime files require frozen, human-reviewed Goal-only patches",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-zip", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_patch_scope(baseline_zip=args.baseline_zip, source_root=args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "production_deployable": False, "missing": result["missing"]}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
