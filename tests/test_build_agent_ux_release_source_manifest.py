from __future__ import annotations

import subprocess
from pathlib import Path

from scripts import build_agent_ux_release_source_manifest as manifest


def _git_result(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["git"], 0, stdout, "")


def test_parse_git_status_handles_untracked_modified_and_rename() -> None:
    parsed = manifest.parse_git_status(
        "?? src/new_file.py\n"
        " M src/dirty_file.py\n"
        "R  src/old_name.py -> src/new_name.py\n"
    )

    assert parsed["src/new_file.py"] == ["??"]
    assert parsed["src/dirty_file.py"] == [" M"]
    assert parsed["src/new_name.py"] == ["R "]


def test_required_action_describes_release_review_state() -> None:
    assert manifest.required_action(exists=False, tracked=False, dirty=False) == "restore_or_remove_from_release_scope"
    assert manifest.required_action(exists=True, tracked=False, dirty=False) == "review_and_git_add_before_release_commit"
    assert manifest.required_action(exists=True, tracked=True, dirty=True) == "review_and_commit_or_revert_before_strict_gate"
    assert manifest.required_action(exists=True, tracked=True, dirty=False) == "already_tracked_clean"


def test_secret_scan_detects_high_confidence_literal_and_ignores_placeholder() -> None:
    synthetic_key = "sk-proj-" + "1234567890abcdefghijklmnop"
    findings = manifest.scan_text_for_secrets(
        f"OPENAI_API_KEY='{synthetic_key}'\n"
        "password='<redacted-password>'\n",
        rel_path="src/example.py",
    )

    assert [item["pattern_id"] for item in findings] == ["openai_or_anthropic_key"]
    assert findings[0]["path"] == "src/example.py"
    assert findings[0]["line"] == 1


def test_build_manifest_reports_untracked_and_dirty_tracked(monkeypatch, tmp_path: Path) -> None:
    untracked = tmp_path / "src" / "new_agent_ux.py"
    dirty = tmp_path / "src" / "dirty_agent_ux.py"
    untracked.parent.mkdir(parents=True)
    untracked.write_text("print('new')\n", encoding="utf-8")
    dirty.write_text("print('dirty')\n", encoding="utf-8")
    monkeypatch.setattr(manifest.gate, "RELEASE_RELEVANT_SOURCE_FILES", (untracked, dirty))

    def fake_run_git(args: list[str], *, root: Path = tmp_path) -> subprocess.CompletedProcess[str]:
        if args[:2] == ["ls-files", "--stage"]:
            return _git_result("100644 0123456789012345678901234567890123456789 0\tsrc/dirty_agent_ux.py\n")
        if args[:2] == ["status", "--porcelain=v1"]:
            return _git_result("?? src/new_agent_ux.py\n M src/dirty_agent_ux.py\n")
        raise AssertionError(args)

    monkeypatch.setattr(manifest, "run_git", fake_run_git)

    report = manifest.build_manifest(root=tmp_path)

    assert report["status"] == "needs_release_source_review"
    assert report["summary"] == {
        "file_count": 2,
        "missing_count": 0,
        "untracked_count": 1,
        "dirty_tracked_count": 1,
        "secret_finding_count": 0,
    }
    assert report["secret_scan"]["status"] == "passed"
    assert report["secret_scan"]["finding_count"] == 0
    assert report["untracked"] == ["src/new_agent_ux.py"]
    assert report["dirty_tracked"] == ["src/dirty_agent_ux.py"]
    actions = {item["path"]: item["required_action"] for item in report["files"]}
    assert actions["src/new_agent_ux.py"] == "review_and_git_add_before_release_commit"
    assert actions["src/dirty_agent_ux.py"] == "review_and_commit_or_revert_before_strict_gate"
