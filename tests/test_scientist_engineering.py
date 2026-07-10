from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from xsci.kaggle_session import SessionState
from xsci.scientist_engineering import run_scientist_engineering_loop


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "EvoMind Test")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "initial")
    return root


def _patch(root: Path, content: str, name: str = "candidate.diff") -> Path:
    target = root / "src" / "demo.py"
    original = target.read_text(encoding="utf-8")
    target.write_text(content, encoding="utf-8")
    diff = _git(root, "diff", "--binary", "--", "src/demo.py")
    target.write_text(original, encoding="utf-8")
    path = root / name
    path.write_text(diff, encoding="utf-8")
    return path


def _work_order(root: Path, *, external_only: bool = False) -> Path:
    path = root / ".xsci" / "test_work_order.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "work_order_id": "test_upgrade",
        "issue_id": "resource_gate_truthfulness" if external_only else "test_code_gap",
        "title": "Test isolated engineering loop",
        "files_to_edit": ["src/demo.py"],
        "acceptance_checks": [
            "python -m py_compile src/demo.py",
            "git diff --check",
        ],
        "rollback_condition": "discard temporary worktree on failure",
        "human_gate": "review_patch_before_merge",
        "self_evolution_context": {
            "execution_partition": {
                "code_agent_fixable_requirements": [] if external_only else [{"requirement_id": "test_code_gap"}],
                "external_resource_blockers": [{"requirement_id": "setup_gate_clearance"}] if external_only else [],
            }
        },
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_engineering_loop_validates_patch_without_modifying_main_worktree(tmp_path: Path):
    root = _repo(tmp_path)
    patch = _patch(root, "VALUE = 2\n")
    order = _work_order(root)
    before = (root / "src" / "demo.py").read_text(encoding="utf-8")
    head = _git(root, "rev-parse", "HEAD").strip()

    result = run_scientist_engineering_loop(
        SessionState(selected_task="test-task"),
        root,
        work_order_path=order,
        patch_path=patch,
    )

    assert result["ok"] is True
    assert result["status"] == "passed_review_candidate"
    assert result["merge_ready"] is True
    assert result["main_worktree_modified"] is False
    assert all(item["passed"] for item in result["acceptance_checks"])
    assert Path(result["candidate_diff_path"]).exists()
    assert (root / "src" / "demo.py").read_text(encoding="utf-8") == before
    assert _git(root, "rev-parse", "HEAD").strip() == head


def test_engineering_loop_rejects_patch_outside_work_order_scope(tmp_path: Path):
    root = _repo(tmp_path)
    patch = _patch(root, "VALUE = 3\n")
    order = root / ".xsci" / "wrong_scope.json"
    order.parent.mkdir(parents=True, exist_ok=True)
    order.write_text(
        json.dumps({
            "work_order_id": "wrong_scope",
            "issue_id": "test_code_gap",
            "files_to_edit": ["src/other.py"],
            "acceptance_checks": ["git diff --check"],
        }),
        encoding="utf-8",
    )

    result = run_scientist_engineering_loop(
        SessionState(selected_task="test-task"),
        root,
        work_order_path=order,
        patch_path=patch,
    )

    assert result["ok"] is False
    assert result["status"] == "blocked_patch_scope_violation"
    assert result["outside_work_order"] == ["src/demo.py"]
    assert (root / "src" / "demo.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_engineering_loop_blocks_external_resource_issue_as_non_code(tmp_path: Path):
    root = _repo(tmp_path)
    patch = _patch(root, "VALUE = 4\n")
    order = _work_order(root, external_only=True)

    result = run_scientist_engineering_loop(
        SessionState(selected_task="test-task"),
        root,
        work_order_path=order,
        patch_path=patch,
    )

    assert result["ok"] is False
    assert result["status"] == "blocked_external_gate_not_code"
    assert result["next_safe_command"] == "evomind ready"
    assert (root / "src" / "demo.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_engineering_loop_recounts_incorrect_llm_hunk_lengths(tmp_path: Path):
    root = _repo(tmp_path)
    patch = _patch(root, "VALUE = 5\n")
    malformed = re.sub(
        r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@",
        r"@@ -\1,9 +\2,17 @@",
        patch.read_text(encoding="utf-8"),
        count=1,
    )
    patch.write_text(malformed.rstrip("\n"), encoding="utf-8")
    order = _work_order(root)

    result = run_scientist_engineering_loop(
        SessionState(selected_task="test-task"),
        root,
        work_order_path=order,
        patch_path=patch,
    )

    assert result["ok"] is True
    assert result["status"] == "passed_review_candidate"
    assert result["main_worktree_modified"] is False
    assert result["merge_ready"] is True
