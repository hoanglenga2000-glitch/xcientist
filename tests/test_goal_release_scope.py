from __future__ import annotations

import json
from pathlib import Path

from evomind_runtime.goal_release_scope import FIXED_ALLOCATION, FIXED_RUN_ID, REQUIRED_RUNTIME_FILES, verify_release_scope


def _manifest(root: Path) -> dict:
    files = []
    for relative in REQUIRED_RUNTIME_FILES:
        path = root / "src" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")
        import hashlib

        files.append({"path": "src/" + relative, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {
        "schema": "evomind.runtime_build.v1",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "goal_spec_sha256": "1" * 64,
        "human_baseline_evidence_sha256": "2" * 64,
        "file_count": len(files),
        "files": files,
    }


def test_release_scope_accepts_complete_goal_runtime_and_source_hashes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    result = verify_release_scope(manifest, source_root=tmp_path)
    assert result.valid is True
    assert result.missing == ()
    assert result.hash_mismatches == ()


def test_release_scope_reports_missing_goal_modules(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["files"] = [item for item in manifest["files"] if not item["path"].endswith(("competition_goal.py", "goal_board.py"))]
    manifest["file_count"] = len(manifest["files"])
    result = verify_release_scope(manifest, source_root=tmp_path)
    assert result.valid is False
    assert set(result.missing) == {"evomind_runtime/competition_goal.py", "evomind_runtime/goal_board.py"}
    assert "GOAL_RUNTIME_RELEASE_SCOPE_MISSING" in result.issues


def test_release_scope_rejects_duplicate_unsafe_and_hash_drift(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["files"].append(dict(manifest["files"][0]))
    manifest["files"].append({"path": "../escape.py", "sha256": "0" * 64})
    manifest["files"][0]["sha256"] = "0" * 64
    manifest["file_count"] = len(manifest["files"])
    result = verify_release_scope(manifest, source_root=tmp_path)
    assert result.valid is False
    assert "RELEASE_MANIFEST_DUPLICATE_ENTRY" in result.issues
    assert "RELEASE_MANIFEST_ENTRY_INVALID" in result.issues
    assert "RELEASE_SOURCE_HASH_MISMATCH" in result.issues


def test_release_scope_rejects_directory_and_drive_paths(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["files"].append({"path": "src/evomind_runtime/", "sha256": "0" * 64})
    manifest["files"].append({"path": "C:/outside.py", "sha256": "0" * 64})
    manifest["file_count"] = len(manifest["files"])
    result = verify_release_scope(manifest, source_root=tmp_path)
    assert result.valid is False
    assert "RELEASE_MANIFEST_ENTRY_INVALID" in result.issues


def test_current_r114_manifest_is_blocked_until_goal_runtime_is_included() -> None:
    path = Path("D:/AI-Outputs/EvoMind-Cloud-Deploy/artifacts/evomind-runtime-r114-weather-recursive-resume-r112-base-source-manifest-20260829.json")
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = verify_release_scope(manifest)
    assert result.valid is False
    assert "GOAL_RUNTIME_RELEASE_SCOPE_MISSING" in result.issues
