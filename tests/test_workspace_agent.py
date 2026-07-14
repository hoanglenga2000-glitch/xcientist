from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

import xsci.workspace_agent as workspace_agent_module
from research_os.agent.messaging import AssistantTurn, ToolCall
from research_os.llm_client import LLMError
from xsci.workspace_agent import (
    WorkspaceAgentLimits,
    _acceptance_evidence_kind,
    _junit_counts,
    _test_command_rejection_reason,
    _within_root,
    sanitize_workspace_result,
    write_workspace_result,
)
from xsci.workspace_agent import (
    run_workspace_agent as _run_workspace_agent,
)

PYTEST_COMMAND = "python -m pytest tests/test_demo.py -q"
ANSWER_PYTEST_COMMAND = "python -m pytest tests/test_answer.py -q"
DATA_PYTEST_COMMAND = "python -m pytest tests/test_data.py -q"
MUTATING_PYTEST_COMMAND = "python -m pytest tests/test_mutating.py -q"


def test_junit_counts_rejects_dtd_oversize_and_symlink_inputs(tmp_path: Path):
    valid = tmp_path / "valid.xml"
    valid.write_text(
        '<testsuite tests="4" failures="1" errors="0" skipped="1"/>',
        encoding="utf-8",
    )
    assert _junit_counts(valid) == {
        "tests": 4,
        "failures": 1,
        "errors": 0,
        "skipped": 1,
        "passed": 2,
        "executed": 3,
    }

    malicious = tmp_path / "malicious.xml"
    malicious.write_text(
        '<!DOCTYPE testsuite [<!ENTITY x "expanded">]>'
        '<testsuite tests="1" failures="0" errors="0" skipped="0">&x;</testsuite>',
        encoding="utf-8",
    )
    assert _junit_counts(malicious)["tests"] == 0

    oversized = tmp_path / "oversized.xml"
    oversized.write_bytes(b" " * (workspace_agent_module._JUNIT_XML_MAX_BYTES + 1))
    assert _junit_counts(oversized)["tests"] == 0

    linked = tmp_path / "linked.xml"
    try:
        linked.symlink_to(valid)
    except OSError:
        return
    assert _junit_counts(linked)["tests"] == 0


def test_workspace_result_sanitizer_redacts_nested_and_inline_secrets():
    marker = "virtual-sensitive-marker-8f139cc9"
    private_key_label = "PRIVATE" + " KEY"
    private_key = (
        f"-----BEGIN {private_key_label}-----\n"
        f"{marker}\n"
        f"-----END {private_key_label}-----"
    )
    payload = {
        "api_key": marker,
        "nested": {"password": marker, "authorization": f"Bearer {marker}"},
        "text": (
            f'Authorization: Bearer {marker}\n'
            f'{{"client_secret": "{marker}"}}\n'
            f"https://user:{marker}@example.invalid/path?token={marker}\n"
            + private_key
        ),
        "usage": {"input_tokens": 12, "output_tokens": 4},
        "code": 'api_key = os.environ.get("DEEPSEEK_API_KEY")',
        "release_note": "current_release_token=release-20260714",
    }

    sanitized = sanitize_workspace_result(payload)
    rendered = json.dumps(sanitized, ensure_ascii=False)

    assert marker not in rendered
    assert sanitized["api_key"] == "[redacted]"
    assert sanitized["nested"]["password"] == "[redacted]"
    assert sanitized["usage"] == {"input_tokens": 12, "output_tokens": 4}
    assert sanitized["code"] == payload["code"]
    assert sanitized["release_note"] == payload["release_note"]


def test_workspace_result_writer_is_atomic_private_and_rejects_symlink(tmp_path: Path):
    target = tmp_path / "result.json"
    sensitive_key = "pass" + "word"
    write_workspace_result(target, {"ok": True, sensitive_key: "virtual-sensitive-marker"})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "ok": True,
        sensitive_key: "[redacted]",
    }
    if os.name != "nt":
        assert target.stat().st_mode & 0o777 == 0o600

    victim = tmp_path / "victim.txt"
    victim.write_text("preserved", encoding="utf-8")
    target.unlink()
    try:
        target.symlink_to(victim)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(OSError, match="symlinked"):
        write_workspace_result(target, {"ok": False})
    assert victim.read_text(encoding="utf-8") == "preserved"


def test_workspace_result_writer_handles_concurrent_atomic_replacements(tmp_path: Path):
    target = tmp_path / "result.json"
    barrier = threading.Barrier(3)
    failures: list[BaseException] = []

    def write(marker: str) -> None:
        try:
            barrier.wait(timeout=5)
            for sequence in range(20):
                write_workspace_result(target, {"marker": marker, "sequence": sequence})
        except BaseException as exc:
            failures.append(exc)

    writers = [
        threading.Thread(target=write, args=(marker,), daemon=True)
        for marker in ("first", "second")
    ]
    for writer in writers:
        writer.start()
    barrier.wait(timeout=5)
    for writer in writers:
        writer.join(timeout=10)

    assert all(not writer.is_alive() for writer in writers)
    assert failures == []
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["marker"] in {"first", "second"}
    assert isinstance(payload["sequence"], int)
    assert list(tmp_path.glob(".*.tmp")) == []


def test_sensitive_candidate_diff_detector_ignores_environment_lookup_code():
    sensitive_assignment = "+pass" + "word = 'live-" + "secret-value'"
    assert workspace_agent_module._contains_sensitive_text(sensitive_assignment) is True
    assert (
        workspace_agent_module._contains_sensitive_text(
            '+api_key = os.environ.get("DEEPSEEK_API_KEY")'
        )
        is False
    )


def run_workspace_agent(*args, **kwargs):
    kwargs.setdefault(
        "behavioral_oracle",
        lambda evidence: bool(evidence.get("host_differential_validated")),
    )
    return _run_workspace_agent(*args, **kwargs)


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
    (root / "tests" / "test_answer.py").write_text(
        "def test_answer_module():\n"
        "    from src.answer import ANSWER\n\n"
        "    assert ANSWER == 42\n",
        encoding="utf-8",
    )
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "EvoMind Test")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "initial")
    return root


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git diff --check", "structural"),
        ("python -m py_compile src/demo.py", "structural"),
        ("npm run build", "structural"),
        ("cargo check", "structural"),
        ("python -m pytest tests/test_demo.py -q", "behavioral"),
        ("python.exe -m unittest tests.test_demo", "host_smoke"),
        ("pytest.exe tests/test_demo.py -q", "behavioral"),
        ("pnpm run test", "host_smoke"),
        ("dotnet test", "host_smoke"),
        ("cargo test", "host_smoke"),
        ("go test ./...", "host_smoke"),
        ("python -c 'print(1)'", "unsupported"),
    ],
)
def test_acceptance_evidence_kind_distinguishes_behavioral_from_structural(
    command: str,
    expected: str,
):
    assert _acceptance_evidence_kind(command) == expected


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        ("python -m pytest tests/test_demo.py -q; whoami", "shell_operator_not_allowed"),
        (r"python -m pytest C:\outside\test_demo.py -q", "absolute_path"),
        ("python -m pytest ../outside/test_demo.py -q", "path_traversal"),
        ("python -m pytest --help", "non_executing_test_mode"),
        ("python -m pytest --version", "non_executing_test_mode"),
        ("python -m pytest --collect-only", "non_executing_test_mode"),
        ("python -m pytest -p no:cacheprovider tests/test_demo.py", "test_configuration_override_not_allowed"),
        ("python -c print(1)", "python_module_mode_required"),
        ("python -m py_compile src/demo.py", "dynamic_runner_not_allowed"),
        ("npm run build", "dynamic_runner_not_allowed"),
        ("python -m unittest tests.test_demo", "dynamic_runner_not_allowed"),
        ("pytest tests/test_demo.py -q", "dynamic_runner_not_allowed"),
        ("npm test -- --runInBand", "dynamic_runner_not_allowed"),
        ("go test ./...", "dynamic_runner_not_allowed"),
        ("python -m pytest scripts/check.py -q", "dynamic_pytest_target_required"),
        ("python -m pytest tests/test_demo.py\x00-q", "control_character_not_allowed"),
    ],
)
def test_dynamic_behavioral_command_policy_rejects_unsafe_or_structural_commands(
    command: str,
    reason: str,
):
    assert _test_command_rejection_reason(command, require_behavioral=True) == reason


def test_dynamic_behavioral_command_policy_accepts_explicit_targeted_pytest():
    command = "python -m pytest tests/test_demo.py::test_value -q --maxfail=1 --tb=short"
    assert _test_command_rejection_reason(command, require_behavioral=True) == ""


@pytest.mark.parametrize(
    "command",
    [
        "python -m unittest tests.test_demo",
        "pytest tests/test_demo.py -q",
        "npm test -- --runInBand",
        "pnpm run test:unit",
        "dotnet test tests/Project.Tests.csproj",
        "cargo test test_value",
        "go test ./...",
    ],
)
def test_caller_configured_command_policy_retains_bounded_test_entrypoints(command: str):
    assert _test_command_rejection_reason(command) == ""


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
    assert result["evidence"]["behavioral_acceptance_passed"] is True
    assert result["evidence"]["behavioral_acceptance_patch_generation"] == 1
    assert result["evidence"]["acceptance_command_kinds"] == {
        PYTEST_COMMAND: "behavioral",
        "git diff --check": "structural",
    }
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


def test_workspace_agent_structural_only_candidate_is_not_completed(tmp_path: Path):
    root = _repo(tmp_path)
    result = run_workspace_agent(
        root,
        goal="Change VALUE to 2 and prove behavioral correctness.",
        planner=ScriptedPlanner(_success_actions()),
        acceptance_commands=["git diff --check"],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-structural-only",
    )

    assert result["ok"] is False
    assert result["completed"] is False
    assert result["status"] == "format_validated_only"
    assert result["stop_reason"] == "behavioral_acceptance_missing"
    assert result["epistemic_status"] == "format_validated_only_not_behaviorally_validated"
    assert result["evidence"]["all_acceptance_passed"] is True
    assert result["evidence"]["behavioral_acceptance_passed"] is False
    assert "review_candidate_ready" not in {item["claim"] for item in result["claims"]}
    assert {item["claim"] for item in result["unsupported_claims"]} == {"review_candidate_ready"}


def test_workspace_agent_host_pytest_requires_external_behavioral_oracle(tmp_path: Path):
    root = _repo(tmp_path)
    result = _run_workspace_agent(
        root,
        goal="Do not promote shared-host pytest observations to strong behavioral evidence.",
        planner=ScriptedPlanner(_success_actions()),
        acceptance_commands=[PYTEST_COMMAND, "git diff --check"],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-no-external-oracle",
        limits=WorkspaceAgentLimits(max_steps=6),
    )

    pytest_result = result["test_results"][0]
    assert pytest_result["host_differential_validated"] is True
    assert pytest_result["behavioral_oracle_configured"] is False
    assert pytest_result["behavioral_oracle_passed"] is False
    assert result["evidence"]["all_acceptance_passed"] is True
    assert result["evidence"]["behavioral_acceptance_passed"] is False
    assert result["status"] == "format_validated_only"
    assert result["ok"] is False


def test_workspace_agent_generic_test_runner_is_host_smoke_only(tmp_path: Path):
    root = _repo(tmp_path)
    oracle_calls: list[dict] = []
    command = "python -m unittest discover -s tests"

    result = _run_workspace_agent(
        root,
        goal="Do not treat an unstructured zero-test runner as strong behavioral evidence.",
        planner=ScriptedPlanner(_success_actions()),
        acceptance_commands=[command],
        behavioral_oracle=lambda evidence: oracle_calls.append(evidence) or True,
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-host-smoke-only",
        limits=WorkspaceAgentLimits(max_steps=6),
    )

    smoke_result = result["test_results"][0]
    assert smoke_result["evidence_kind"] == "host_smoke"
    assert smoke_result["exit_code"] in {0, 5}
    assert smoke_result["passed"] is (smoke_result["exit_code"] == 0)
    assert smoke_result["tests_executed"] is None
    assert smoke_result["host_differential_validated"] is False
    assert smoke_result["behavioral_oracle_passed"] is False
    assert oracle_calls == []
    assert result["evidence"]["host_smoke_commands"] == [command]
    assert result["evidence"]["behavioral_acceptance_passed"] is False
    assert result["status"] != "completed"
    assert result["ok"] is False


def test_workspace_agent_defaults_to_rejecting_model_proposed_dynamic_tests(tmp_path: Path):
    root = _repo(tmp_path)
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "test", "command": "git diff --check"},
        {"action": "read", "path": "tests/test_demo.py"},
        {"action": "test", "command": PYTEST_COMMAND},
        {"action": "finish", "summary": "Host dynamic execution must be disabled.", "claims": []},
    ])

    result = run_workspace_agent(
        root,
        goal="Keep proposed test execution disabled without an OS sandbox.",
        planner=planner,
        acceptance_commands=["git diff --check"],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-default-dynamic-disabled",
        limits=WorkspaceAgentLimits(max_steps=7),
    )

    rejected = result["steps"][5]["observation"]
    assert result["allow_dynamic_behavioral_tests"] is False
    assert rejected["error"] == "unsafe_dynamic_test_command"
    assert rejected["rejection_reason"] == "dynamic_behavioral_tests_disabled"
    assert rejected["policy_rejection"] is True
    assert result["budget"]["test_runs"] == 1
    assert result["scope_violations"] == []
    assert planner.contexts[4]["workflow"] == {
        "phase": "diff_review",
        "next_gate": "capture_diff",
        "recommended_actions": ["diff"],
        "unread_required_paths": [],
        "post_patch_unread_paths": [],
    }


def test_workspace_agent_model_tool_contract_matches_dynamic_test_policy():
    disabled = workspace_agent_module._tool_specs(
        ("git diff --check",),
        allow_dynamic_behavioral_tests=False,
    )
    enabled = workspace_agent_module._tool_specs(
        ("git diff --check",),
        allow_dynamic_behavioral_tests=True,
    )

    disabled_description = next(spec.description for spec in disabled if spec.name == "workspace_test")
    enabled_description = next(spec.description for spec in enabled if spec.name == "workspace_test")
    assert "Dynamic commands are disabled" in disabled_description
    assert "do not propose" in disabled_description
    assert "propose one bounded behavioral test" in enabled_description


def test_workspace_agent_retries_invalid_patch_before_touching_candidate(tmp_path: Path):
    root = _repo(tmp_path)
    invalid_patch = (
        "diff --git a/src/demo.py b/src/demo.py\n"
        "--- a/src/demo.py\n"
        "+++ b/src/demo.py\n"
        "@@ -1 +1 @@\n"
        "-MISSING = 1\n"
        "+VALUE = 2\n"
    )
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": invalid_patch},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "read", "path": "src/demo.py"},
        {"action": "test", "command": PYTEST_COMMAND},
        {"action": "diff"},
        {
            "action": "finish",
            "summary": "The corrected patch is ready for review.",
            "review": "The final diff changes only VALUE from 1 to 2.",
            "claims": ["review_candidate_ready"],
        },
    ])

    result = run_workspace_agent(
        root,
        goal="Recover from a malformed candidate patch and change VALUE to 2.",
        planner=planner,
        acceptance_commands=[PYTEST_COMMAND],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-invalid-patch-retry",
        limits=WorkspaceAgentLimits(max_steps=8, max_patch_attempts=2),
    )

    rejected = result["steps"][2]["observation"]
    assert rejected["step"] == 3
    assert rejected["action"] == "patch"
    assert rejected["ok"] is False
    assert rejected["error"] == "git_apply_check_failed"
    assert rejected["exit_code"] != 0
    assert rejected["log_path"]
    assert result["ok"] is True
    assert result["budget"]["patch_attempts"] == 2
    assert result["evidence"]["patch_generation"] == 1
    assert "-VALUE = 1" in result["final_diff"]
    assert "+VALUE = 2" in result["final_diff"]


def test_workspace_agent_does_not_advertise_missing_candidate_diff(tmp_path: Path):
    root = _repo(tmp_path)
    invalid_patch = (
        "diff --git a/src/demo.py b/src/demo.py\n"
        "--- a/src/demo.py\n"
        "+++ b/src/demo.py\n"
        "@@ -1 +1 @@\n"
        "-MISSING = 1\n"
        "+VALUE = 2\n"
    )
    result = run_workspace_agent(
        root,
        goal="Do not advertise a diff when the candidate patch is invalid.",
        planner=ScriptedPlanner([
            {"action": "search", "query": "VALUE", "path": "src"},
            {"action": "read", "path": "src/demo.py"},
            {"action": "patch", "unified_diff": invalid_patch},
        ]),
        acceptance_commands=[PYTEST_COMMAND],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-no-candidate-diff",
        limits=WorkspaceAgentLimits(max_steps=3, max_patch_attempts=1),
    )

    assert result["ok"] is False
    assert result["final_diff"] == ""
    assert result["candidate_diff_path"] == ""


def test_workspace_agent_runs_model_proposed_behavioral_test_after_structural_gate(tmp_path: Path):
    root = _repo(tmp_path)
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "test", "command": "git diff --check"},
        {"action": "search", "query": "test_value", "path": "tests"},
        {"action": "read", "path": "tests/test_demo.py"},
        {"action": "test", "command": PYTEST_COMMAND},
        {"action": "diff"},
        {
            "action": "finish",
            "summary": "The targeted behavioral test validates the final patch.",
            "claims": ["review_candidate_ready"],
        },
    ])

    result = run_workspace_agent(
        root,
        goal="Change VALUE to 2 and discover a targeted behavioral test.",
        planner=planner,
        acceptance_commands=["git diff --check"],
        allow_dynamic_behavioral_tests=True,
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-dynamic-behavioral-test",
        limits=WorkspaceAgentLimits(max_steps=9),
    )

    assert result["ok"] is True
    assert planner.contexts[4]["workflow"] == {
        "phase": "behavioral_validation",
        "next_gate": "targeted_behavioral_test",
        "recommended_actions": ["search", "read", "test", "patch"],
        "unread_required_paths": [],
        "post_patch_unread_paths": [],
    }
    assert result["evidence"]["dynamic_behavioral_commands"] == [PYTEST_COMMAND]
    assert result["evidence"]["current_test_generation"] == {
        "git diff --check": 1,
        PYTEST_COMMAND: 1,
    }
    assert result["evidence"]["behavioral_acceptance_patch_generation"] == 1
    assert result["steps"][6]["observation"]["dynamic_command_added"] is True
    assert result["test_results"][-1]["tests_executed"] == 1
    assert result["test_results"][-1]["disposable_test_worktree"] is True


def test_workspace_agent_rejects_unsafe_dynamic_command_without_spending_test_budget(tmp_path: Path):
    root = _repo(tmp_path)
    unsafe = "python -m pytest ../outside/test_demo.py -q"
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "test", "command": "git diff --check"},
        {"action": "test", "command": unsafe},
        {"action": "finish", "summary": "Unsafe command must not run.", "claims": []},
    ])

    result = run_workspace_agent(
        root,
        goal="Reject tests that escape the candidate worktree.",
        planner=planner,
        acceptance_commands=["git diff --check"],
        allow_dynamic_behavioral_tests=True,
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-rejected-dynamic-test",
        limits=WorkspaceAgentLimits(max_steps=6),
    )

    rejected = result["steps"][4]["observation"]
    assert rejected["error"] == "unsafe_dynamic_test_command"
    assert rejected["rejection_reason"] == "path_traversal"
    assert result["budget"]["test_runs"] == 1
    assert result["evidence"]["dynamic_behavioral_commands"] == []
    assert result["scope_violations"][-1]["reason"] == "path_traversal"


def test_workspace_agent_requires_dynamic_test_target_read_for_current_patch(tmp_path: Path):
    root = _repo(tmp_path)
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "test", "command": "git diff --check"},
        {"action": "test", "command": PYTEST_COMMAND},
        {"action": "finish", "summary": "Unread tests must not become evidence.", "claims": []},
    ])

    result = run_workspace_agent(
        root,
        goal="Require inspection of the exact targeted test before execution.",
        planner=planner,
        acceptance_commands=["git diff --check"],
        allow_dynamic_behavioral_tests=True,
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-unread-dynamic-test",
        limits=WorkspaceAgentLimits(max_steps=6),
    )

    rejected = result["steps"][4]["observation"]
    assert rejected["error"] == "unsafe_dynamic_test_command"
    assert rejected["rejection_reason"] == "dynamic_test_target_not_read_for_current_patch"
    assert result["budget"]["test_runs"] == 1
    assert result["evidence"]["dynamic_behavioral_commands"] == []


def test_workspace_agent_invalidates_dynamic_behavioral_pass_after_new_patch(tmp_path: Path):
    root = _repo(tmp_path)
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "test", "command": "git diff --check"},
        {"action": "read", "path": "tests/test_demo.py"},
        {"action": "test", "command": PYTEST_COMMAND},
        {"action": "patch", "unified_diff": _patch(2, 3)},
        {"action": "diff"},
        {"action": "finish", "summary": "The stale test must not validate generation two.", "claims": []},
    ])

    result = run_workspace_agent(
        root,
        goal="Bind a dynamically proposed test to the exact patch generation.",
        planner=planner,
        acceptance_commands=["git diff --check"],
        allow_dynamic_behavioral_tests=True,
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-stale-dynamic-test",
        limits=WorkspaceAgentLimits(max_steps=9),
    )

    assert result["ok"] is False
    assert result["evidence"]["patch_generation"] == 2
    assert result["evidence"]["current_test_status"] == {}
    assert result["evidence"]["current_test_generation"] == {}
    assert result["evidence"]["dynamic_behavioral_commands"] == [PYTEST_COMMAND]
    assert result["evidence"]["behavioral_acceptance_passed"] is False
    assert result["evidence"]["behavioral_acceptance_patch_generation"] is None


def test_workspace_agent_repairs_after_failed_dynamic_behavioral_test(tmp_path: Path):
    root = _repo(tmp_path)
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 3)},
        {"action": "test", "command": "git diff --check"},
        {"action": "read", "path": "tests/test_demo.py"},
        {"action": "test", "command": PYTEST_COMMAND},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(3, 2)},
        {"action": "read", "path": "tests/test_demo.py"},
        {"action": "test", "command": "all"},
        {"action": "test", "command": PYTEST_COMMAND},
        {"action": "diff"},
        {
            "action": "finish",
            "summary": "The failed targeted test drove a repair that now passes.",
            "claims": ["review_candidate_ready"],
        },
    ])

    result = run_workspace_agent(
        root,
        goal="Use a targeted behavioral failure to repair VALUE to 2.",
        planner=planner,
        acceptance_commands=["git diff --check"],
        allow_dynamic_behavioral_tests=True,
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-dynamic-test-repair",
        limits=WorkspaceAgentLimits(max_steps=13),
    )

    assert result["ok"] is True
    assert result["failure_observed"] is True
    assert result["repair_attempted"] is True
    assert result["replanned_after_failure"] is True
    assert [item["passed"] for item in result["test_results"]] == [True, False, True, True]
    assert result["evidence"]["behavioral_acceptance_patch_generation"] == 2


def test_workspace_agent_latches_behavioral_failure_until_next_patch_generation(tmp_path: Path):
    root = _repo(tmp_path)
    failing_command = "python -m pytest tests/test_always_fails.py -q"
    (root / "tests" / "test_always_fails.py").write_text(
        "def test_failure():\n    assert False\n",
        encoding="utf-8",
    )
    _git(root, "add", "tests/test_always_fails.py")
    _git(root, "commit", "-m", "add failing behavior fixture")
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "test", "command": "git diff --check"},
        {"action": "read", "path": "tests/test_always_fails.py"},
        {"action": "test", "command": failing_command},
        {"action": "read", "path": "tests/test_demo.py"},
        {"action": "test", "command": PYTEST_COMMAND},
        {"action": "diff"},
        {"action": "finish", "summary": "A later pass must not erase the earlier failure.", "claims": []},
    ])

    result = run_workspace_agent(
        root,
        goal="Keep a failed behavioral test latched until the candidate changes.",
        planner=planner,
        acceptance_commands=["git diff --check"],
        allow_dynamic_behavioral_tests=True,
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-latched-dynamic-failure",
        limits=WorkspaceAgentLimits(max_steps=10),
    )

    assert result["ok"] is False
    assert result["evidence"]["behavioral_acceptance_passed"] is True
    assert result["evidence"]["all_acceptance_passed"] is False
    assert result["evidence"]["failed_tests_this_generation"] == [failing_command]
    assert result["evidence"]["current_test_status"] == {
        "git diff --check": True,
        failing_command: False,
        PYTEST_COMMAND: True,
    }


def test_candidate_content_digest_tracks_declared_path_when_git_hides_it(tmp_path: Path):
    root = _repo(tmp_path)
    target = root / "src" / "demo.py"
    _git(root, "update-index", "--assume-unchanged", "src/demo.py")
    try:
        before, before_ok = workspace_agent_module._candidate_content_digest(
            root,
            timeout=30,
            required_paths=["src/demo.py"],
        )
        target.write_text("VALUE = 99\n", encoding="utf-8")
        after, after_ok = workspace_agent_module._candidate_content_digest(
            root,
            timeout=30,
            required_paths=["src/demo.py"],
        )
        assert _git(root, "diff", "--name-only", "--") == ""
        assert before_ok is True
        assert after_ok is True
        assert after != before
    finally:
        _git(root, "update-index", "--no-assume-unchanged", "src/demo.py")


def test_workspace_agent_rejects_all_skipped_dynamic_junit_evidence(tmp_path: Path):
    root = _repo(tmp_path)
    command = "python -m pytest tests/test_skipped.py -q"
    (root / "tests" / "test_skipped.py").write_text(
        "import pytest\n\ndef test_skipped():\n    pytest.skip('not executed')\n",
        encoding="utf-8",
    )
    _git(root, "add", "tests/test_skipped.py")
    _git(root, "commit", "-m", "add skipped test fixture")
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "read", "path": "tests/test_skipped.py"},
        {"action": "test", "command": command},
        {"action": "diff"},
        {"action": "finish", "summary": "Skipped tests are not behavioral proof.", "claims": []},
    ])

    result = run_workspace_agent(
        root,
        goal="Reject an all-skipped targeted test as behavioral evidence.",
        planner=planner,
        acceptance_commands=["git diff --check"],
        allow_dynamic_behavioral_tests=True,
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-skipped-junit",
        limits=WorkspaceAgentLimits(max_steps=7),
    )

    test_result = result["test_results"][0]
    assert test_result["exit_code"] == 0
    assert test_result["candidate_junit"]["skipped"] == 1
    assert test_result["candidate_junit"]["executed"] == 0
    assert test_result["candidate_junit"]["passed"] == 0
    assert test_result["passed"] is False
    assert result["evidence"]["failed_tests_this_generation"] == [command]
    assert result["evidence"]["behavioral_acceptance_passed"] is False


def test_workspace_agent_rejects_dynamic_evidence_after_conftest_patch(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "tests" / "conftest.py").write_text("BASELINE = True\n", encoding="utf-8")
    _git(root, "add", "tests/conftest.py")
    _git(root, "commit", "-m", "add pytest configuration fixture")
    patch = _patch(1, 2) + _replace_file_patch(
        "tests/conftest.py",
        "BASELINE = True",
        "BASELINE = False",
    )
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": patch},
        {"action": "read", "path": "tests/test_demo.py"},
        {"action": "test", "command": PYTEST_COMMAND},
    ])

    result = run_workspace_agent(
        root,
        goal="Do not accept candidate-controlled pytest configuration as proof.",
        planner=planner,
        acceptance_commands=["git diff --check"],
        allow_dynamic_behavioral_tests=True,
        allowed_edit_paths=["src/demo.py", "tests/conftest.py"],
        artifact_dir=tmp_path / "artifacts-modified-conftest",
        limits=WorkspaceAgentLimits(max_steps=5),
    )

    rejected = result["steps"][4]["observation"]
    assert rejected["error"] == "unsafe_dynamic_test_command"
    assert rejected["rejection_reason"] == "dynamic_test_support_modified_by_candidate"
    assert result["test_results"] == []
    assert result["evidence"]["dynamic_behavioral_commands"] == []


def test_workspace_agent_latches_exit_127_even_after_same_generation_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = _repo(tmp_path)
    real_run = workspace_agent_module._run
    pytest_runs = 0

    def fail_first_pytest(args, *, cwd, timeout, env=None):
        nonlocal pytest_runs
        if cwd.name == "workspace" and "-I" in args and "-c" in args:
            pytest_runs += 1
            if pytest_runs == 1:
                return subprocess.CompletedProcess(args, 127, stdout="synthetic launch failure", stderr=None)
        return real_run(args, cwd=cwd, timeout=timeout, env=env)

    monkeypatch.setattr(workspace_agent_module, "_run", fail_first_pytest)
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "test", "command": PYTEST_COMMAND},
        {"action": "test", "command": PYTEST_COMMAND},
        {"action": "diff"},
        {"action": "finish", "summary": "A retry must not erase the launch failure.", "claims": []},
    ])

    result = run_workspace_agent(
        root,
        goal="Latch every failed acceptance attempt in the current candidate generation.",
        planner=planner,
        acceptance_commands=[PYTEST_COMMAND],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-exit-127-latch",
        limits=WorkspaceAgentLimits(max_steps=7),
    )

    assert [item["exit_code"] for item in result["test_results"]] == [127, 0]
    assert [item["passed"] for item in result["test_results"]] == [False, True]
    assert result["evidence"]["current_test_status"][PYTEST_COMMAND] is True
    assert result["evidence"]["failed_tests_this_generation"] == [PYTEST_COMMAND]
    assert result["evidence"]["all_acceptance_passed"] is False
    assert result["ok"] is False


def test_workspace_agent_rolls_back_post_apply_intent_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = _repo(tmp_path)
    real_git = workspace_agent_module._git

    def fail_intent(root_path, *args, timeout=120):
        if root_path.name == "worktree" and args[:2] == ("add", "--intent-to-add"):
            return subprocess.CompletedProcess(args, 125, stdout="synthetic intent failure", stderr=None)
        return real_git(root_path, *args, timeout=timeout)

    monkeypatch.setattr(workspace_agent_module, "_git", fail_intent)
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _new_file_patch("src/answer.py", "ANSWER = 42\n")},
    ])

    result = run_workspace_agent(
        root,
        goal="Rollback a patch when its post-apply index step fails.",
        planner=planner,
        acceptance_commands=["git diff --check"],
        allowed_edit_paths=["src/answer.py"],
        artifact_dir=tmp_path / "artifacts-patch-rollback",
        limits=WorkspaceAgentLimits(max_steps=3),
    )

    failed_patch = result["steps"][2]["observation"]
    assert failed_patch["error"] == "intent_to_add_failed"
    assert failed_patch["rollback_ok"] is True
    assert result["stop_reason"] == "patch_transaction_failed"
    assert result["evidence"]["patch_generation"] == 0
    assert result["evidence"]["current_test_status"] == {}
    assert result["final_diff"] == ""
    assert (root / "src" / "answer.py").exists() is False


@pytest.mark.parametrize("dynamic", [False, True])
def test_workspace_agent_trusted_pytest_bootstrap_rejects_runner_shadowing(
    tmp_path: Path,
    dynamic: bool,
):
    root = _repo(tmp_path)
    fake_runner = (
        "from pathlib import Path\n"
        "import sys\n"
        "for argument in sys.argv:\n"
        "    if argument.startswith('--junitxml='):\n"
        "        Path(argument.split('=', 1)[1]).write_text("
        "'<testsuite tests=\"1\" failures=\"0\" errors=\"0\" skipped=\"0\"/>', "
        "encoding='utf-8')\n"
        "raise SystemExit(0)\n"
    )
    patch = _patch(1, 999) + _new_file_patch("pytest.py", fake_runner)
    actions = [
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": patch},
    ]
    if dynamic:
        actions.extend([
            {"action": "test", "command": "git diff --check"},
            {"action": "read", "path": "tests/test_demo.py"},
            {"action": "test", "command": PYTEST_COMMAND},
        ])
        acceptance_commands = ["git diff --check"]
    else:
        actions.append({"action": "test", "command": PYTEST_COMMAND})
        acceptance_commands = [PYTEST_COMMAND]

    result = run_workspace_agent(
        root,
        goal="Do not allow candidate modules to replace the trusted pytest runner.",
        planner=ScriptedPlanner(actions),
        acceptance_commands=acceptance_commands,
        allow_dynamic_behavioral_tests=dynamic,
        allowed_edit_paths=["src/demo.py", "pytest.py"],
        artifact_dir=tmp_path / f"artifacts-pytest-shadow-{dynamic}",
        limits=WorkspaceAgentLimits(max_steps=len(actions)),
    )

    pytest_result = result["test_results"][-1]
    assert pytest_result["passed"] is False
    assert pytest_result["candidate_junit"]["failures"] == 1
    assert result["evidence"]["behavioral_acceptance_passed"] is False
    assert result["ok"] is False


def test_workspace_agent_rejects_candidate_modified_pytest_plugin(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "tests" / "conftest.py").write_text(
        "pytest_plugins = ['fakeplugin']\n",
        encoding="utf-8",
    )
    (root / "src" / "fakeplugin.py").write_text(
        "def pytest_configure(config):\n    return None\n",
        encoding="utf-8",
    )
    _git(root, "add", "tests/conftest.py", "src/fakeplugin.py")
    _git(root, "commit", "-m", "add baseline pytest plugin")
    plugin_patch = (
        "diff --git a/src/fakeplugin.py b/src/fakeplugin.py\n"
        "--- a/src/fakeplugin.py\n"
        "+++ b/src/fakeplugin.py\n"
        "@@ -1,2 +1,5 @@\n"
        "-def pytest_configure(config):\n"
        "-    return None\n"
        "+import pytest\n"
        "+\n"
        "+def pytest_collection_modifyitems(items):\n"
        "+    for item in items:\n"
        "+        item.add_marker(pytest.mark.skip(reason='candidate plugin'))\n"
    )
    patch = _patch(1, 999) + plugin_patch
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": patch},
        {"action": "test", "command": PYTEST_COMMAND},
    ])

    result = run_workspace_agent(
        root,
        goal="Do not accept behavior evidence from a candidate-modified pytest plugin.",
        planner=planner,
        acceptance_commands=[PYTEST_COMMAND],
        allowed_edit_paths=["src/demo.py", "src/fakeplugin.py"],
        artifact_dir=tmp_path / "artifacts-pytest-plugin",
        limits=WorkspaceAgentLimits(max_steps=4),
    )

    test_result = result["test_results"][0]
    assert test_result["pytest_surface_baseline_identical"] is False
    assert test_result["behavioral_oracle_passed"] is False
    assert result["evidence"]["behavioral_acceptance_passed"] is False
    assert result["ok"] is False


def test_workspace_agent_blocks_patch_action_toctou_in_prior_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = _repo(tmp_path)
    (root / "src" / "other.py").write_text("OTHER = 0\n", encoding="utf-8")
    _git(root, "add", "src/other.py")
    _git(root, "commit", "-m", "add second patch target")
    real_git = workspace_agent_module._git
    injected = False

    def inject_during_second_apply(root_path, *args, timeout=120):
        nonlocal injected
        patch_name = Path(str(args[-1])).name if args else ""
        if (
            not injected
            and root_path.name == "worktree"
            and args
            and args[0] == "apply"
            and "--check" not in args
            and patch_name == "04.diff"
        ):
            (root_path / "src" / "demo.py").write_text(
                "VALUE = 2\nBACKDOOR = True\n",
                encoding="utf-8",
            )
            injected = True
        return real_git(root_path, *args, timeout=timeout)

    monkeypatch.setattr(workspace_agent_module, "_git", inject_during_second_apply)
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {
            "action": "patch",
            "unified_diff": _replace_file_patch("src/other.py", "OTHER = 0", "OTHER = 1"),
        },
    ])

    result = run_workspace_agent(
        root,
        goal="Reject undeclared mutations that race a later patch action.",
        planner=planner,
        acceptance_commands=["git diff --check"],
        allowed_edit_paths=["src/demo.py", "src/other.py"],
        artifact_dir=tmp_path / "artifacts-patch-toctou",
        limits=WorkspaceAgentLimits(max_steps=4),
    )

    failed = result["steps"][3]["observation"]
    assert injected is True
    assert failed["error"] == "patch_transaction_mismatch"
    assert failed["rollback_ok"] is True
    assert result["ok"] is False
    assert "BACKDOOR" not in result["final_diff"]


def test_workspace_agent_binds_final_diff_to_tested_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = _repo(tmp_path)

    class DiffRacePlanner(ScriptedPlanner):
        diff_started = False

        def __call__(self, context: dict) -> dict:
            action = super().__call__(context)
            if action.get("action") == "diff":
                self.diff_started = True
            return action

    planner = DiffRacePlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "test", "command": PYTEST_COMMAND},
        {"action": "diff"},
    ])
    real_snapshot = workspace_agent_module._candidate_snapshot
    mutated = False

    def race_snapshot(root_path, *, timeout, content_paths=()):
        nonlocal mutated
        snapshot = real_snapshot(root_path, timeout=timeout, content_paths=content_paths)
        if planner.diff_started and root_path.name == "worktree" and not mutated:
            (root_path / "src" / "demo.py").write_text("VALUE = 999\n", encoding="utf-8")
            mutated = True
        return snapshot

    monkeypatch.setattr(workspace_agent_module, "_candidate_snapshot", race_snapshot)
    result = run_workspace_agent(
        root,
        goal="Bind the emitted diff to the exact candidate that passed tests.",
        planner=planner,
        acceptance_commands=[PYTEST_COMMAND],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-final-diff-race",
        limits=WorkspaceAgentLimits(max_steps=5),
    )

    failed = result["steps"][4]["observation"]
    assert mutated is True
    assert failed["error"] == "candidate_snapshot_mismatch"
    assert result["final_diff"] == ""
    assert result["ok"] is False


def test_workspace_agent_candidate_and_baseline_use_same_runtime_identity(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "tests" / "test_demo.py").write_text(
        "from pathlib import Path\n\n"
        "def test_runtime_identity():\n"
        "    assert 'baseline-worktree' not in str(Path.cwd())\n",
        encoding="utf-8",
    )
    _git(root, "add", "tests/test_demo.py")
    _git(root, "commit", "-m", "add cwd-sensitive regression fixture")
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 99)},
        {"action": "test", "command": PYTEST_COMMAND},
    ])

    result = run_workspace_agent(
        root,
        goal="Do not let tests distinguish candidate and baseline by runtime paths.",
        planner=planner,
        acceptance_commands=[PYTEST_COMMAND],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-runtime-identity",
        limits=WorkspaceAgentLimits(max_steps=4),
    )

    test_result = result["test_results"][0]
    assert test_result["candidate_passed"] is True
    assert test_result["baseline_exit_code"] == 0
    assert test_result["runner_argv_equivalent"] is True
    assert test_result["baseline_differential_validated"] is False
    assert result["ok"] is False


def test_workspace_agent_cleans_candidate_phase_state_before_baseline(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "tests" / "test_demo.py").write_text(
        "from pathlib import Path\n\n"
        "def test_no_phase_marker():\n"
        "    marker = Path.cwd().parent / 'candidate-ran.marker'\n"
        "    assert not marker.exists()\n"
        "    marker.write_text('ran', encoding='utf-8')\n",
        encoding="utf-8",
    )
    _git(root, "add", "tests/test_demo.py")
    _git(root, "commit", "-m", "add phase-state regression fixture")
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 99)},
        {"action": "test", "command": PYTEST_COMMAND},
    ])

    result = run_workspace_agent(
        root,
        goal="Do not share candidate phase files with the baseline run.",
        planner=planner,
        acceptance_commands=[PYTEST_COMMAND],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-phase-state",
        limits=WorkspaceAgentLimits(max_steps=4),
    )

    test_result = result["test_results"][0]
    assert test_result["candidate_passed"] is True
    assert test_result["baseline_exit_code"] == 0
    assert test_result["baseline_differential_validated"] is False
    assert result["ok"] is False


def test_workspace_agent_blocks_baseline_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = _repo(tmp_path)
    real_git = workspace_agent_module._git
    removes = 0

    def fail_second_runtime_remove(root_path, *args, timeout=120):
        nonlocal removes
        if args[:3] == ("worktree", "remove", "--force") and "test-execution-" in str(args[3]):
            removes += 1
            result = real_git(root_path, *args, timeout=timeout)
            if removes == 2:
                return subprocess.CompletedProcess(args, 1, stdout="synthetic cleanup failure", stderr=None)
            return result
        return real_git(root_path, *args, timeout=timeout)

    monkeypatch.setattr(workspace_agent_module, "_git", fail_second_runtime_remove)
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "test", "command": PYTEST_COMMAND},
    ])

    result = run_workspace_agent(
        root,
        goal="Treat every candidate and baseline cleanup failure as blocking.",
        planner=planner,
        acceptance_commands=[PYTEST_COMMAND],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-baseline-cleanup",
        limits=WorkspaceAgentLimits(max_steps=4),
    )

    assert removes == 2
    assert result["test_results"][0]["baseline_worktree_cleanup_ok"] is False
    assert result["ok"] is False
    assert result["stop_reason"] == "workspace_cleanup_failed"


def test_workspace_agent_invalidates_behavioral_pass_after_new_patch_generation(tmp_path: Path):
    root = _repo(tmp_path)
    planner = ScriptedPlanner([
        {"action": "search", "query": "VALUE", "path": "src"},
        {"action": "read", "path": "src/demo.py"},
        {"action": "patch", "unified_diff": _patch(1, 2)},
        {"action": "test", "command": "all"},
        {"action": "patch", "unified_diff": _patch(2, 3)},
        {"action": "diff"},
        {"action": "finish", "summary": "The old test pass should not validate this patch."},
    ])

    result = run_workspace_agent(
        root,
        goal="Require behavioral evidence for the final patch generation.",
        planner=planner,
        acceptance_commands=[PYTEST_COMMAND, "git diff --check"],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-stale-behavioral-pass",
        limits=WorkspaceAgentLimits(max_steps=7),
    )

    assert result["ok"] is False
    assert result["status"] == "needs_continuation"
    assert result["evidence"]["patch_generation"] == 2
    assert result["evidence"]["current_test_status"] == {}
    assert result["evidence"]["behavioral_acceptance_passed"] is False
    assert result["evidence"]["behavioral_acceptance_patch_generation"] is None
    assert {item["patch_generation"] for item in result["test_results"]} == {1}
    assert "behavioral_test" in result["steps"][-1]["observation"]["missing_gates"]


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
        acceptance_commands=[PYTEST_COMMAND, "git diff --check"],
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
        acceptance_commands=[PYTEST_COMMAND, "git diff --check"],
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
        acceptance_commands=[PYTEST_COMMAND, "git diff --check"],
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
        {"action": "test", "command": "all"},
        {"action": "diff"},
        {"action": "finish", "summary": "New module is ready for review.", "claims": ["review_candidate_ready"]},
    ])

    result = run_workspace_agent(
        root,
        goal="Add a small answer module and produce a replayable candidate.",
        planner=planner,
        acceptance_commands=[ANSWER_PYTEST_COMMAND, "git diff --check"],
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
    (root / "tests" / "test_data.py").write_text(
        "from pathlib import Path\n\n"
        "def test_data_value():\n"
        "    assert Path('data/value.txt').read_text(encoding='utf-8') == 'new\\n'\n",
        encoding="utf-8",
    )
    _git(root, "add", "data/value.txt", "tests/test_data.py")
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
        {"action": "test", "command": "all"},
        {"action": "diff"},
        {"action": "finish", "summary": "Explicitly scoped data edit is ready.", "claims": ["review_candidate_ready"]},
    ])

    result = run_workspace_agent(
        root,
        goal="Update the explicitly authorized data file.",
        planner=planner,
        acceptance_commands=[DATA_PYTEST_COMMAND, "git diff --check"],
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
    assert test_result["candidate_mutated_by_test"] is False
    assert test_result["test_snapshot_mutated"] is True
    assert test_result["test_status_mutated"] is True
    assert test_result["disposable_test_worktree"] is True
    assert test_result["status_sha256_before"] == test_result["status_sha256_after"]
    diff_observation = result["steps"][4]["observation"]
    assert diff_observation["ok"] is True
    assert not (root / "generated-by-test.txt").exists()
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


class FlakyScriptedClient(ScriptedClient):
    def __init__(self, turns: list[AssistantTurn], *, failures: int) -> None:
        super().__init__(turns)
        self.failures = failures

    def send(self, messages, **_kwargs):
        self.messages_seen.append(list(messages))
        self.systems_seen.append(str(_kwargs.get("system") or ""))
        if self.failures > 0:
            self.failures -= 1
            raise LLMError("injected transient provider failure")
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
        _tool_turn(4, "workspace_test", {"command": "all"}),
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
        acceptance_commands=[PYTEST_COMMAND, "git diff --check"],
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


def test_workspace_agent_does_not_trust_model_validation_claim_without_behavioral_test(tmp_path: Path):
    root = _repo(tmp_path)
    claimed_summary = "Validated, minimal, and correct."
    claimed_review = "The patch is fully behaviorally validated."
    client = ScriptedClient([
        _tool_turn(1, "workspace_search", {"query": "VALUE", "path": "src"}),
        _tool_turn(2, "workspace_read", {"path": "src/demo.py"}),
        _tool_turn(3, "workspace_patch", {"unified_diff": _patch(1, 2)}),
        _tool_turn(4, "workspace_test", {"command": "git diff --check"}),
        _tool_turn(5, "workspace_diff", {}),
        _tool_turn(6, "workspace_finish", {
            "summary": claimed_summary,
            "review": claimed_review,
            "claims": ["review_candidate_ready"],
        }),
    ])

    result = run_workspace_agent(
        root,
        goal="Do not trust a model's validation claim without a behavioral test.",
        client=client,
        acceptance_commands=["git diff --check"],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-untrusted-model-claim",
    )

    assert result["ok"] is False
    assert result["status"] == "format_validated_only"
    assert result["summary"] != claimed_summary
    assert result["semantic_review"] == ""
    assert result["untrusted_model_summary"] == claimed_summary
    assert result["untrusted_model_semantic_review"] == claimed_review
    assert "review_candidate_ready" not in {item["claim"] for item in result["claims"]}
    assert result["unsupported_claims"] == [
        {"claim": "review_candidate_ready", "reason": "runtime_evidence_missing"}
    ]


def test_workspace_agent_quarantines_model_text_for_any_incomplete_status(tmp_path: Path):
    root = _repo(tmp_path)
    claimed_summary = "UNVERIFIED COMPLETE"
    claimed_review = "UNVERIFIED REVIEW"
    planner = ScriptedPlanner([{
        "action": "finish",
        "summary": claimed_summary,
        "review": claimed_review,
        "claims": ["review_candidate_ready"],
    }])

    result = run_workspace_agent(
        root,
        goal="Keep incomplete model claims out of trusted result fields.",
        planner=planner,
        acceptance_commands=[PYTEST_COMMAND],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-incomplete-model-claim",
        limits=WorkspaceAgentLimits(max_steps=1),
    )

    assert result["ok"] is False
    assert result["status"] == "needs_continuation"
    assert result["summary"] != claimed_summary
    assert result["summary"].startswith("Workspace agent did not complete")
    assert result["semantic_review"] == ""
    assert result["untrusted_model_summary"] == claimed_summary
    assert result["untrusted_model_semantic_review"] == claimed_review


def test_workspace_agent_retries_transient_model_failure_with_audited_budget(tmp_path: Path):
    root = _repo(tmp_path)
    client = FlakyScriptedClient([
        _tool_turn(1, "workspace_search", {"query": "VALUE", "path": "src"}),
        _tool_turn(2, "workspace_read", {"path": "src/demo.py"}),
        _tool_turn(3, "workspace_patch", {"unified_diff": _patch(1, 2)}),
        _tool_turn(4, "workspace_test", {"command": "all"}),
        _tool_turn(5, "workspace_diff", {}),
        _tool_turn(6, "workspace_finish", {
            "summary": "Candidate ready after a transient provider failure.",
            "review": "The final diff changes only VALUE from 1 to 2.",
        }),
    ], failures=1)

    result = run_workspace_agent(
        root,
        goal="Recover from one transient provider failure and finish the candidate.",
        client=client,
        acceptance_commands=[PYTEST_COMMAND, "git diff --check"],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-model-retry",
        limits=WorkspaceAgentLimits(max_steps=8, max_decision_retries=1),
    )

    assert result["ok"] is True
    assert result["steps"][0]["action"] == "decision_retry"
    assert result["steps"][0]["observation"]["retrying"] is True
    assert result["budget"]["decision_failures"] == 1
    assert len(client.messages_seen[0]) == 2
    assert len(client.messages_seen[1]) == 2


def test_workspace_agent_stops_after_bounded_model_failure_retries(tmp_path: Path):
    root = _repo(tmp_path)
    client = FlakyScriptedClient([], failures=3)

    result = run_workspace_agent(
        root,
        goal="Stop honestly when the provider remains unavailable.",
        client=client,
        acceptance_commands=["git diff --check"],
        allowed_edit_paths=["src/demo.py"],
        artifact_dir=tmp_path / "artifacts-model-retry-exhausted",
        limits=WorkspaceAgentLimits(max_steps=4, max_decision_retries=1),
    )

    assert result["ok"] is False
    assert result["status"] == "needs_continuation"
    assert result["stop_reason"] == "decision_error:LLMError"
    assert [item["action"] for item in result["steps"]] == ["decision_retry", "decision_error"]
    assert result["budget"]["decision_failures"] == 2


def test_workspace_agent_model_prompt_marks_artifact_paths_as_read_only_evidence(tmp_path: Path):
    root = _repo(tmp_path)
    client = ScriptedClient([
        _tool_turn(1, "workspace_search", {"query": "VALUE", "path": "src"}),
        _tool_turn(2, "workspace_read", {"path": "src/demo.py"}),
        _tool_turn(3, "workspace_patch", {"unified_diff": _patch(1, 2)}),
        _tool_turn(4, "workspace_test", {"command": "all"}),
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
        acceptance_commands=[PYTEST_COMMAND, "git diff --check"],
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
