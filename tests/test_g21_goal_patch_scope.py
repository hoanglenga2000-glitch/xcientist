from __future__ import annotations

import json
from pathlib import Path
import zipfile

from scripts.audit_g21_goal_patch_scope import DEPENDENT_FILES, GOAL_ONLY_FILES, audit_patch_scope


def test_patch_scope_is_conservative_and_hash_bound(tmp_path: Path) -> None:
    source = tmp_path / "source"
    package = tmp_path / "package"
    for relative in (*DEPENDENT_FILES, *GOAL_ONLY_FILES):
        path = source / "src" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("goal baseline\n", encoding="utf-8")
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("goal baseline\n", encoding="utf-8")
    (source / "src" / "evomind_runtime" / "store.py").write_text("goal changed\n", encoding="utf-8")
    baseline_zip = tmp_path / "baseline.zip"
    with zipfile.ZipFile(baseline_zip, "w") as archive:
        for path in package.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(package).as_posix())
    result = audit_patch_scope(baseline_zip=baseline_zip, source_root=source)
    assert result["status"] == "REVIEW_REQUIRED"
    assert result["production_deployable"] is False
    assert result["missing"] == []
    assert len(result["files"]) == len(DEPENDENT_FILES) + len(GOAL_ONLY_FILES)
    changed = next(item for item in result["files"] if item["path"].endswith("store.py"))
    assert changed["added_lines"] == 1
    assert changed["diff_sha256"]


def test_patch_scope_reports_missing_baseline_dependent_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    package = tmp_path / "package"
    for relative in (*DEPENDENT_FILES, *GOAL_ONLY_FILES):
        path = source / "src" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("goal\n", encoding="utf-8")
        if relative.endswith("store.py"):
            continue
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("goal\n", encoding="utf-8")
    baseline_zip = tmp_path / "baseline.zip"
    with zipfile.ZipFile(baseline_zip, "w") as archive:
        for path in package.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(package).as_posix())
    result = audit_patch_scope(baseline_zip=baseline_zip, source_root=source)
    assert result["status"] == "BLOCKED"
    assert "baseline:evomind_runtime/store.py" in result["missing"]
