from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from research_os.agent.messaging import AssistantTurn, ToolCall
from xsci.workspace_agent import WorkspaceAgentLimits, _within_root, run_workspace_agent

PYTEST_COMMAND = "python -m pytest tests/test_demo.py -q"
MUTATING_PYTEST_COMMAND = "python -m pytest tests/test_mutating.py -q"


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


def _repo(tmp_path: Path, *, slow_test: bool = False) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n", encoding="utf-8")
    body = "from src.demo import VALUE\n\n"
    if slow_test:
        body += "import time\n\ndef test_value():\n    time.sleep(5)\n    assert VALUE == 2\n"
    else:
        body += "def test_value():\n    assert VALUE == 2\n"
    (root / "tests" / "test_demo.py").write_text(body, encoding="utf-8")
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "EvoMind Test")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "initial")
    return root


def test_within_root_canonicalizes_directory_alias_before_containment_check(tmp_path: Path):
    canonical_root = tmp_path / "canonical-root"
    target = canonical_root / "src" / "demo.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n", encoding="utf-8")
    alias_root = tmp_path / "alias-root"

    if os.name == "nt":
        created = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(alias_root), str(canonical_root)],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if created.returncode != 0:
            pytest.skip("directory junctions are unavailable")
    else:
        alias_root.symlink_to(canonical_root, target_is_directory=True)

    try:
        assert _within_root(alias_root, "src/demo.py", must_exist=True) == target.resolve()
        with pytest.raises(ValueError, match="path_traversal"):
            _within_root(alias_root, "../outside.py")
    finally:
        if alias_root.is_symlink():
            alias_root.unlink()
        elif alias_root.exists():
            alias_root.rmdir()


def _patch(old: int, new: int) -> str:
    return (
        "diff --git a/src/demo.py b/src/demo.py\n"
        "--- a/src/demo.py\n"
        "+++ b/src/demo.py\n"
        "@@ -1 +1 @@\n"
        f"-VALUE = {old}\n"
        f"+VALUE = {new}\n"
    )


def _replace_file_patch(path: str, old: str, new: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        f"-{old}\n"
        f"+{new}\n"
    )


def _new_file_patch(path: str, content: str) -> str:
    lines = content.rstrip("\n").splitlines()
    additions = "".join(f"+{line}\n" for line in lines)
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{additions}"
    )


class ScriptedPlanner:
    def __init__(self, actions: list[dict]) -> None:
        self.actions = list(actions)
        self.contexts: list[dict] = []

    def __call__(self, context: dict) -> dict:
        self.contexts.append(context)
        return self.actions.pop(0)


class ArtifactPathProbePlanner(ScriptedPlanner):
    def __init__(self, actions: list[dict], forbidden_prefix: Path) -> None:
        super().__init__(actions)
        self.forbidden_prefix = str(forbidden_prefix)
        self.audit_paths_hidden = False

    def __call__(self, context: dict) -> dict:
        if context.get("last_observation"):
            rendered = json.dumps(context["last_observation"])
            self.audit_paths_hidden = "[audit artifact recorded]" in rendered and self.forbidden_prefix not in rendered
        return super().__call__(context)


class PlannerMustNotRun:
    def __init__(self) -> None:
        self.called = False

    def __call__(self, _context: dict) -> dict:
        self.called = True
        raise AssertionError("planner must not run for a dirty source workspace")


def _success_actions() -> list[dict]:
    return [
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py", "start_line": 1, "end_line": 20},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "test", "command": "all"},
        {"action": "diff"},
        {
            "action": "finish",
            "summary": "The isolated candidate passes the configured checks.",
            "claims": ["review_candidate_ready", "main_worktree_unchanged"],
        },
    ]


def test_workspace_agent_completes_real_git_loop_without_touching_main(tmp_path: Path):
    root = _repo(tmp_path)
    planner = ScriptedPlanner(_success_actions())
    before = (root / "src" / "demo.py").read_text(encoding="utf-8")
    head = _git(root, "rev-parse", "HEAD").strip()

    result = run_workspace_agent(
        root,
        goal="Find VALUE, change it to 2, and prove the repository still passes.",
        planner=planner,
        acceptance_commands=[PYTEST_COMMAND, "git diff --check"],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-success",
    )

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert result["completed"] is True
    assert result["needs_continuation"] is False
    assert result["main_worktree_modified"] is False
    assert result["commit_created"] is False
    assert result["merged"] is False
    assert result["human_gate"] == "review_candidate_before_merge"
    assert [item["action"] for item in result["steps"]] == ["search", "read", "patch", "test", "diff", "finish"]
    assert planner.contexts[0]["workflow"]["phase"] == "discovery"
    assert planner.contexts[2]["workflow"]["phase"] == "edit"
    assert planner.contexts[2]["workflow"]["recommended_actions"][0] == "patch"
    assert result["steps"][1]["observation"]["ends_with_newline"] is True
    assert result["steps"][1]["observation"]["newline_style"] == "lf"
    assert all(item["passed"] for item in result["test_results"])
    assert "-VALUE = 1" in result["final_diff"]
    assert "+VALUE = 2" in result["final_diff"]
    assert "+VALUE = 2" in result["steps"][4]["observation"]["diff_preview"]
    assert result["steps"][4]["observation"]["diff_preview_truncated"] is False
    assert Path(result["candidate_diff_path"]).read_bytes() == result["final_diff"].encode("utf-8")
    assert Path(result["artifact_path"]).exists()
    assert result["command_logs"] and all(Path(path).exists() for path in result["command_logs"])
    assert result["unsupported_claims"] == []
    assert {item["claim"] for item in result["claims"]} >= {
        "workspace_searched",
        "workspace_file_read",
        "candidate_patch_applied",
        "acceptance_commands_passed",
        "final_diff_captured",
        "review_candidate_ready",
    }
    assert (root / "src" / "demo.py").read_text(encoding="utf-8") == before
    assert _git(root, "rev-parse", "HEAD").strip() == head


def test_workspace_agent_uses_bounded_python_search_without_ripgrep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = _repo(tmp_path)
    real_which = shutil.which

    def without_ripgrep(command: str, *args, **kwargs):
        if command == "rg":
            return None
        return real_which(command, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", without_ripgrep)
    result = run_workspace_agent(
        root,
        goal="Find VALUE without relying on an optional system search binary.",
        planner=ScriptedPlanner(_success_actions()),
        acceptance_commands=["git diff --check"],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-python-search",
    )

    assert result["ok"] is True, result
    search = result["steps"][0]["observation"]
    assert search["backend"] == "python"
    assert any("src/demo.py:1:1:VALUE = 1" in match for match in search["matches"])


def test_workspace_agent_normalizes_model_patch_without_terminal_newline(tmp_path: Path):
    root = _repo(tmp_path)
    actions = _success_actions()
    actions[2] = {"action": "patch", "unified_diff": _patch(1, 2).rstrip("\n")}

    result = run_workspace_agent(
        root,
        goal="Change VALUE to 2 even when the model omits the patch's terminal newline.",
        planner=ScriptedPlanner(actions),
        acceptance_commands=["git diff --check"],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-no-patch-newline",
    )

    assert result["ok"] is True
    assert result["budget"]["patch_attempts"] == 1
    patch_step = result["steps"][2]["observation"]
    assert patch_step["terminal_newline_added"] is True
    assert (tmp_path / "artifacts-no-patch-newline" / "patches" / "03.diff").read_bytes().endswith(b"\n")


def test_workspace_agent_default_artifacts_stay_outside_clean_repository(tmp_path: Path):
    root = _repo(tmp_path)

    result = run_workspace_agent(
        root,
        goal="Prepare a candidate without making the source repository dirty.",
        planner=ScriptedPlanner(_success_actions()),
        acceptance_commands=["git diff --check"],
        allowed_edit_paths=["src/demo.py"],
    )

    artifact_path = Path(result["artifact_path"])
    try:
        assert result["ok"] is True
        assert root not in artifact_path.parents
        assert _git(root, "status", "--porcelain=v1", "--untracked-files=all") == ""
    finally:
        shutil.rmtree(artifact_path.parent, ignore_errors=True)


def test_workspace_agent_includes_new_file_in_replayable_binary_diff(tmp_path: Path):
    root = _repo(tmp_path)
    content = "ANSWER = 42\n"
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _new_file_patch("src/answer.py", content)},
        {"action": "test", "command": "git diff --check"},
        {"action": "diff"},
        {"action": "finish", "summary": "New module is ready for review.", "claims": ["review_candidate_ready"]},
    ])

    result = run_workspace_agent(
        root,
        goal="Add a small answer module and produce a replayable candidate.",
        planner=planner,
        acceptance_commands=["git diff --check"],
        allowed_edit_paths=["src/answer.py"],
        artifact_dir=tmp_path / "artifacts-new-file",
    )

    assert result["ok"] is True
    assert result["evidence"]["intent_to_add_files"] == ["src/answer.py"]
    assert "new file mode 100644" in result["final_diff"]
    assert "+++ b/src/answer.py" in result["final_diff"]
    assert (root / "src" / "answer.py").exists() is False

    replay = tmp_path / "replay"
    subprocess.run(
        ["git", "clone", "--quiet", str(root), str(replay)],
        cwd=tmp_path,
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    candidate = Path(result["candidate_diff_path"])
    _git(replay, "apply", "--check", str(candidate))
    _git(replay, "apply", str(candidate))
    assert (replay / "src" / "answer.py").read_text(encoding="utf-8") == content


def test_workspace_agent_fails_closed_when_real_main_worktree_is_dirty(tmp_path: Path):
    root = _repo(tmp_path)
    planner = PlannerMustNotRun()
    (root / "src" / "demo.py").write_text("VALUE = 99\n", encoding="utf-8")
    head = _git(root, "rev-parse", "HEAD").strip()

    result = run_workspace_agent(
        root,
        goal="Do not silently ignore the user's uncommitted edit.",
        planner=planner,
        acceptance_commands=["git diff --check"],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-dirty",
    )

    assert result["ok"] is False
    assert result["status"] == "blocked_dirty_main_worktree"
    assert result["stop_reason"] == "dirty_main_worktree"
    assert result["main_dirty_before"] is True
    assert result["main_worktree_modified"] is False
    assert planner.called is False
    assert result["command_logs"] == []
    assert (root / "src" / "demo.py").read_text(encoding="utf-8") == "VALUE = 99\n"
    assert _git(root, "rev-parse", "HEAD").strip() == head


def _add_tracked_data_file(root: Path) -> None:
    (root / "data").mkdir()
    (root / "data" / "value.txt").write_text("old\n", encoding="utf-8")
    _git(root, "add", "data/value.txt")
    _git(root, "commit", "-m", "add data fixture")


@pytest.mark.parametrize("allowed_paths", [[], ["src"]])
def test_workspace_agent_reads_data_but_requires_explicit_data_edit_scope(
    tmp_path: Path,
    allowed_paths: list[str],
):
    root = _repo(tmp_path)
    _add_tracked_data_file(root)
    planner = ScriptedPlanner([
        {"action": "search", "query": "old", "path": "data"},
        {"action": "read", "path": "data/value.txt"},
        {"action": "patch", "unified_diff": _replace_file_patch("data/value.txt", "old", "new")},
        {"action": "finish", "summary": "The data edit was not authorized.", "claims": []},
    ])

    result = run_workspace_agent(
        root,
        goal="Inspect data, but do not edit it without an explicit scope.",
        planner=planner,
        acceptance_commands=["git diff --check"],
        allowed_edit_paths=allowed_paths,
        artifact_dir=tmp_path / ("artifacts-data-empty" if not allowed_paths else "artifacts-data-other"),
        limits=WorkspaceAgentLimits(max_steps=4),
    )

    assert result["ok"] is False
    assert result["steps"][1]["observation"]["ok"] is True
    assert "1: old" in result["steps"][1]["observation"]["content"]
    assert result["steps"][2]["observation"]["error"] == "patch_scope_violation"
    assert result["scope_violations"][0]["reason"] in {
        "explicit_allowed_edit_path_required",
        "outside_allowed_edit_paths",
    }
    assert (root / "data" / "value.txt").read_text(encoding="utf-8") == "old\n"


def test_workspace_agent_allows_data_edit_only_with_explicit_scope(tmp_path: Path):
    root = _repo(tmp_path)
    _add_tracked_data_file(root)
    planner = ScriptedPlanner([
        {"action": "search", "query": "old", "path": "data"},
        {"action": "read", "path": "data/value.txt"},
        {"action": "patch", "unified_diff": _replace_file_patch("data/value.txt", "old", "new")},
        {"action": "test", "command": "git diff --check"},
        {"action": "diff"},
        {"action": "finish", "summary": "Explicitly scoped data edit is ready.", "claims": ["review_candidate_ready"]},
    ])

    result = run_workspace_agent(
        root,
        goal="Update the explicitly authorized data file.",
        planner=planner,
        acceptance_commands=["git diff --check"],
        allowed_edit_paths=["data/value.txt"],
        artifact_dir=tmp_path / "artifacts-data-explicit",
    )

    assert result["ok"] is True
    assert "-old" in result["final_diff"]
    assert "+new" in result["final_diff"]
    assert (root / "data" / "value.txt").read_text(encoding="utf-8") == "old\n"


class RepairPlanner:
    def __init__(self) -> None:
        self.phase = 0
        self.failure_seen = False

    def __call__(self, context: dict) -> dict:
        self.phase += 1
        if self.phase == 1:
            return {"action": "search", "query": "VALUE", "path": "src"}
        if self.phase == 2:
            return {"action": "read", "path": "src/demo.py"}
        if self.phase == 3:
            return {"action": "patch", "unified_diff": _patch(1, 3)}
        if self.phase == 4:
            return {"action": "test", "command": PYTEST_COMMAND}
        if self.phase == 5:
            failure = context["last_observation"]
            assert failure["action"] == "test"
            assert failure["ok"] is False
            assert "assert 3 == 2" in json.dumps(failure)
            self.failure_seen = True
            return {"action": "patch", "unified_diff": _patch(3, 2), "rationale": "Repair the observed assertion."}
        if self.phase == 6:
            return {"action": "test", "command": PYTEST_COMMAND}
        if self.phase == 7:
            return {"action": "diff"}
        return {"action": "finish", "summary": "Failure repaired and revalidated.", "claims": ["review_candidate_ready"]}


def test_workspace_agent_replans_from_failed_test_and_repairs_candidate(tmp_path: Path):
    root = _repo(tmp_path)
    planner = RepairPlanner()

    result = run_workspace_agent(
        root,
        goal="Repair VALUE using the failing test as evidence.",
        planner=planner,
        acceptance_commands=[PYTEST_COMMAND],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-repair",
    )

    assert result["ok"] is True
    assert planner.failure_seen is True
    assert result["budget"]["patch_attempts"] == 2
    assert [item["passed"] for item in result["test_results"]] == [False, True]
    assert result["failure_observed"] is True
    assert result["repair_attempted"] is True
    assert result["replanned_after_failure"] is True
    assert result["evidence"]["patch_generation"] == 2
    assert "-VALUE = 1" in result["final_diff"]
    assert "+VALUE = 2" in result["final_diff"]
    assert "VALUE = 3" not in result["final_diff"]
    assert (root / "src" / "demo.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_workspace_agent_rejects_successful_acceptance_that_creates_untracked_file(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "tests" / "test_mutating.py").write_text(
        "from pathlib import Path\n\n"
        "def test_writes_artifact():\n"
        "    Path('generated-by-test.txt').write_text('generated\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    _git(root, "add", "tests/test_mutating.py")
    _git(root, "commit", "-m", "add mutating acceptance fixture")
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "test", "command": MUTATING_PYTEST_COMMAND},
        {"action": "diff"},
        {"action": "finish", "summary": "Exit code zero is insufficient after a workspace mutation.", "claims": []},
    ])

    result = run_workspace_agent(
        root,
        goal="Reject an acceptance command that leaves an undeclared artifact.",
        planner=planner,
        acceptance_commands=[MUTATING_PYTEST_COMMAND],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-mutating-test",
        limits=WorkspaceAgentLimits(max_steps=6),
    )

    test_result = result["test_results"][0]
    assert test_result["exit_code"] == 0
    assert test_result["passed"] is False
    assert test_result["candidate_mutated_by_test"] is True
    assert test_result["candidate_diff_mutated_by_test"] is False
    assert test_result["candidate_status_mutated_by_test"] is True
    assert test_result["status_sha256_before"] != test_result["status_sha256_after"]
    diff_observation = result["steps"][4]["observation"]
    assert diff_observation["error"] == "final_diff_scope_violation"
    assert diff_observation["untracked_files"] == ["generated-by-test.txt"]
    assert result["evidence"]["all_acceptance_passed"] is False
    assert result["ok"] is False
    assert (root / "generated-by-test.txt").exists() is False


def test_workspace_agent_blocks_scope_escape_and_filters_unsupported_claims(tmp_path: Path):
    root = _repo(tmp_path)
    outside = tmp_path / "outside.txt"
    escape_patch = (
        "diff --git a/../outside.txt b/../outside.txt\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/../outside.txt\n"
        "@@ -0,0 +1 @@\n"
        "+escaped\n"
    )
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": escape_patch},
        {
            "action": "finish",
            "summary": "Claim completion despite missing evidence.",
            "claims": ["review_candidate_ready", "deployed_to_production"],
        },
    ])

    result = run_workspace_agent(
        root,
        goal="Attempt an unsafe edit.",
        planner=planner,
        acceptance_commands=["git diff --check"],
        allowed_edit_paths=["src"],
        artifact_dir=tmp_path / "artifacts-scope",
        limits=WorkspaceAgentLimits(max_steps=6),
    )

    assert result["ok"] is False
    assert result["status"] == "needs_continuation"
    assert result["needs_continuation"] is True
    assert result["scope_violations"]
    assert result["scope_violations"][0]["reason"] == "path_traversal"
    assert {item["claim"] for item in result["unsupported_claims"]} == {
        "review_candidate_ready",
        "deployed_to_production",
    }
    assert result["final_diff"] == ""
    assert outside.exists() is False
    assert (root / "src" / "demo.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_workspace_agent_enforces_command_timeout_and_needs_continuation(tmp_path: Path):
    root = _repo(tmp_path, slow_test=True)
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "test", "command": PYTEST_COMMAND},
        {"action": "finish", "summary": "The timeout prevents completion.", "claims": ["acceptance_commands_passed"]},
    ])

    result = run_workspace_agent(
        root,
        goal="Prove the slow test within a one-second command budget.",
        planner=planner,
        acceptance_commands=[PYTEST_COMMAND],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-timeout",
        limits=WorkspaceAgentLimits(command_timeout_seconds=1, total_timeout_seconds=12, max_steps=6),
    )

    assert result["ok"] is False
    assert result["status"] == "needs_continuation"
    assert result["test_results"][0]["timed_out"] is True
    assert result["test_results"][0]["exit_code"] == 124
    assert result["test_results"][0]["passed"] is False
    assert result["unsupported_claims"] == [
        {"claim": "acceptance_commands_passed", "reason": "runtime_evidence_missing"}
    ]
    assert (root / "src" / "demo.py").read_text(encoding="utf-8") == "VALUE = 1\n"


class ScriptedClient:
    def __init__(self, turns: list[AssistantTurn]) -> None:
        self.turns = list(turns)
        self.messages_seen: list[list[dict]] = []
        self.systems_seen: list[str] = []

    def is_available(self) -> bool:
        return True

    def send(self, messages, **_kwargs):
        self.messages_seen.append(list(messages))
        self.systems_seen.append(str(_kwargs.get("system") or ""))
        return self.turns.pop(0)


def _tool_turn(index: int, name: str, payload: dict) -> AssistantTurn:
    call = ToolCall(id=f"call_{index}", name=name, input=payload)
    return AssistantTurn(
        text="",
        tool_calls=[call],
        stop_reason="tool_use",
        raw_content=[{"type": "tool_use", "id": call.id, "name": name, "input": payload}],
        provider="openai",
        model="scripted-model",
    )


def test_workspace_agent_model_client_drives_same_native_tool_loop(tmp_path: Path):
    root = _repo(tmp_path)
    client = ScriptedClient([
        _tool_turn(1, "workspace_search", {"query": "VALUE", "path": "src"}),
        _tool_turn(2, "workspace_read", {"path": "src/demo.py"}),
        _tool_turn(3, "workspace_patch", {"unified_diff": _patch(1, 2)}),
        _tool_turn(4, "workspace_test", {"command": "git diff --check"}),
        _tool_turn(5, "workspace_diff", {}),
        _tool_turn(6, "workspace_finish", {
            "summary": "Candidate ready for review.",
            "review": "The final diff changes only VALUE from 1 to 2, exactly matching the goal.",
            "claims": ["review_candidate_ready"],
        }),
    ])

    result = run_workspace_agent(
        root,
        goal="Use model-selected tools to prepare a tested candidate.",
        client=client,
        acceptance_commands=["git diff --check"],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-model",
    )

    assert result["ok"] is True
    assert result["provider"] == "openai"
    assert result["model"] == "scripted-model"
    assert len(client.messages_seen) == 6
    assert any(
        block.get("type") == "tool_result"
        for messages in client.messages_seen[1:]
        for message in messages
        for block in (message.get("content") if isinstance(message.get("content"), list) else [])
        if isinstance(block, dict)
    )
    assert (root / "src" / "demo.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_workspace_agent_model_prompt_marks_artifact_paths_as_read_only_evidence(tmp_path: Path):
    root = _repo(tmp_path)
    client = ScriptedClient([
        _tool_turn(1, "workspace_search", {"query": "VALUE", "path": "src"}),
        _tool_turn(2, "workspace_read", {"path": "src/demo.py"}),
        _tool_turn(3, "workspace_patch", {"unified_diff": _patch(1, 2)}),
        _tool_turn(4, "workspace_test", {"command": "git diff --check"}),
        _tool_turn(5, "workspace_diff", {}),
        _tool_turn(6, "workspace_finish", {
            "summary": "Candidate ready.",
            "review": "The final diff changes only the requested value and preserves the file scope.",
        }),
    ])

    result = run_workspace_agent(
        root,
        goal="Change VALUE and do not confuse artifact paths with repository paths.",
        client=client,
        acceptance_commands=["git diff --check"],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-prompt",
    )

    assert result["ok"] is True
    assert "artifact_path are read-only execution evidence" in client.systems_seen[0]
    assert "Read every file that the requested change depends on" in client.messages_seen[0][0]["content"]
    assert "workspace_finish.review" in client.messages_seen[0][0]["content"]


def test_workspace_agent_hides_absolute_audit_paths_from_planner_observations(tmp_path: Path):
    root = _repo(tmp_path)
    planner = ArtifactPathProbePlanner(_success_actions(), tmp_path)

    result = run_workspace_agent(
        root,
        goal="Prepare the candidate without following audit artifact paths.",
        planner=planner,
        acceptance_commands=[PYTEST_COMMAND, "git diff --check"],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-redacted",
    )

    assert result["ok"] is True
    assert planner.audit_paths_hidden is True


def test_workspace_agent_missing_repo_file_is_not_mislabeled_as_scope_escape(tmp_path: Path):
    root = _repo(tmp_path)
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/not-created-yet.py"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "test", "command": "all"},
        {"action": "diff"},
        {"action": "finish", "summary": "Candidate ready."},
    ])

    result = run_workspace_agent(
        root,
        goal="Recover after reading an allowed path that has not been created yet.",
        planner=planner,
        acceptance_commands=[PYTEST_COMMAND, "git diff --check"],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-missing-read",
    )

    assert result["ok"] is True
    assert result["steps"][1]["observation"]["ok"] is False
    assert result["scope_violations"] == []


def test_workspace_agent_requires_post_patch_read_before_finish_when_enabled(tmp_path: Path):
    root = _repo(tmp_path)
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "test", "command": "all"},
        {"action": "diff"},
        {"action": "finish", "summary": "Too early."},
        {"action": "read", "path": "src/demo.py"},
        {"action": "finish", "summary": "Verified candidate."},
    ])

    result = run_workspace_agent(
        root,
        goal="Change VALUE and verify the final file before completion.",
        planner=planner,
        acceptance_commands=[PYTEST_COMMAND, "git diff --check"],
        allowed_edit_paths=["src/demo.py"],
        required_edit_paths=["src/demo.py"],
        require_post_patch_read=True,
        artifact_dir=tmp_path / "artifacts-post-read",
    )

    assert result["ok"] is True
    assert result["steps"][5]["observation"]["error"] == "completion_contract_not_met"
    assert "post_patch_read" in result["steps"][5]["observation"]["missing_gates"]
    assert result["required_edit_paths"] == ["src/demo.py"]
    assert result["require_post_patch_read"] is True
