from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from research_os.agent import messaging
from xsci import agentic_capability_benchmark as benchmark
from xsci import config as xcfg
from xsci import kaggle as ak
from xsci import workspace_agent


@pytest.fixture()
def isolated_gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home_xsci"
    monkeypatch.setattr(xcfg, "GLOBAL_DIR", home)
    monkeypatch.setattr(xcfg, "GLOBAL_CONFIG", home / "config.toml")
    monkeypatch.setattr(xcfg, "SECRETS_FILE", home / "secrets.toml")
    monkeypatch.setattr(xcfg, "ONBOARDED_MARKER", home / "onboarded.json")
    monkeypatch.setattr(ak, "GLOBAL_DIR", home)
    for env_name in list(xcfg._ENV_MAP):
        monkeypatch.delenv(env_name, raising=False)
    for env_name in (
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "EVOLUTION_PRIMARY_PROVIDER",
        "EVOLUTION_PROVIDER_STRICT",
    ):
        monkeypatch.delenv(env_name, raising=False)
    return home


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "plain-git-repo"
    nested = root / "src" / "nested"
    nested.mkdir(parents=True)
    (root / "README.md").write_text("baseline\n", encoding="utf-8")
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "EvoMind Test")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "initial")
    return root, nested


class AvailableClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def is_available(self) -> bool:
        return True


class UnavailableClient(AvailableClient):
    def is_available(self) -> bool:
        return False


def test_help_lists_real_workspace_and_behavior_benchmark(isolated_gateway, capsys):
    assert ak.main(["--help"]) == 0
    output = capsys.readouterr().out
    assert 'evomind workspace "goal"' in output
    assert "evomind benchmark-agent --all" in output


def test_workspace_command_uses_current_git_top_level_and_strict_provider(
    isolated_gateway, tmp_path, monkeypatch, capsys
):
    root, nested = _repo(tmp_path)
    monkeypatch.chdir(nested)
    calls: list[dict] = []

    def fake_run(workspace_root, **kwargs):
        calls.append({"workspace_root": Path(workspace_root), **kwargs})
        assert os.environ["EVOLUTION_PRIMARY_PROVIDER"] == "deepseek"
        assert os.environ["EVOLUTION_PROVIDER_STRICT"] == "1"
        return {
            "ok": True,
            "completed": True,
            "status": "completed",
            "provider": "deepseek",
            "model": "deepseek-chat",
            "candidate_diff_path": str(tmp_path / "candidate.diff"),
            "artifact_path": str(tmp_path / "manifest.json"),
            "human_gate": "review_candidate_before_merge",
            "final_diff": "diff --git a/README.md b/README.md\n",
        }

    monkeypatch.setattr(messaging, "AgentMessageClient", AvailableClient)
    monkeypatch.setattr(workspace_agent, "run_workspace_agent", fake_run)

    rc = ak.main([
        "workspace",
        "--json",
        "--provider",
        "deepseek",
        "--allow",
        "src",
        "--check",
        "git diff --check",
        "repair",
        "the",
        "project",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["workspace_root"] == str(root.resolve())
    assert calls[0]["workspace_root"] == root.resolve()
    assert calls[0]["goal"] == "repair the project"
    assert calls[0]["allowed_edit_paths"] == ["src"]
    assert calls[0]["acceptance_commands"] == ["git diff --check"]
    assert calls[0]["require_post_patch_read"] is True
    assert os.environ.get("EVOLUTION_PROVIDER_STRICT") is None


def test_workspace_json_output_redacts_nested_and_inline_secrets(
    isolated_gateway, tmp_path, monkeypatch, capsys
):
    marker = "virtual-cli-secret-3c87bfe9"
    _root, nested = _repo(tmp_path)
    monkeypatch.chdir(nested)
    monkeypatch.setattr(messaging, "AgentMessageClient", AvailableClient)
    monkeypatch.setattr(
        workspace_agent,
        "run_workspace_agent",
        lambda *_args, **_kwargs: {
            "ok": True,
            "completed": True,
            "status": "completed",
            "api_key": marker,
            "nested": {"password": marker},
            "summary": f"Authorization: Bearer {marker}",
            "final_diff": "diff --git a/README.md b/README.md\n",
        },
    )

    rc = ak.main(["workspace", "--json", "--provider", "deepseek", "inspect"])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert rc == 0
    assert marker not in output
    assert payload["api_key"] == "[redacted]"
    assert payload["nested"]["password"] == "[redacted]"
    assert payload["summary"] == "Authorization: Bearer [redacted]"


def test_workspace_command_fails_closed_before_agent_when_provider_is_missing(
    isolated_gateway, tmp_path, monkeypatch, capsys
):
    _root, nested = _repo(tmp_path)
    monkeypatch.chdir(nested)
    monkeypatch.setattr(messaging, "AgentMessageClient", UnavailableClient)

    def must_not_run(*args, **kwargs):
        raise AssertionError("workspace agent must not run without the selected provider")

    monkeypatch.setattr(workspace_agent, "run_workspace_agent", must_not_run)
    rc = ak.main(["workspace", "--json", "--provider", "deepseek", "inspect", "repo"])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert payload["status"] == "blocked_provider_unavailable"
    assert payload["stop_reason"] == "provider_unavailable"
    assert payload["final_diff"] == ""


def test_benchmark_agent_defaults_to_one_real_case_and_preserves_provider_scope(
    isolated_gateway, monkeypatch, capsys
):
    calls: list[dict] = []

    def fake_benchmark(**kwargs):
        calls.append(kwargs)
        assert os.environ["EVOLUTION_PRIMARY_PROVIDER"] == "deepseek"
        assert os.environ["EVOLUTION_PROVIDER_STRICT"] == "1"
        return {
            "execution_status": "completed",
            "provider": "deepseek",
            "cases_run": 1,
            "passed_cases": 1,
            "failed_cases": 0,
            "task_success_rate": 1.0,
            "scope_violations": 0,
            "unsupported_claims": 0,
            "report_path": str(isolated_gateway / "report.json"),
        }

    monkeypatch.setattr(benchmark, "run_workspace_agent_benchmark", fake_benchmark, raising=False)
    rc = ak.main(["benchmark-agent", "--json", "--provider", "deepseek"])
    report = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert report["passed_cases"] == 1
    assert calls[0]["case_ids"] == ["retrieval_exact_release_token"]
    assert calls[0]["provider"] == "deepseek"
    assert calls[0]["timeout_seconds"] == 180
    assert calls[0]["limits"]["max_steps"] == 24
    assert calls[0]["limits"]["max_patch_attempts"] == 5
    assert os.environ.get("EVOLUTION_PROVIDER_STRICT") is None


def test_benchmark_agent_requires_explicit_all_for_full_suite(
    isolated_gateway, monkeypatch, capsys
):
    calls: list[dict] = []

    def fake_benchmark(**kwargs):
        calls.append(kwargs)
        return {
            "execution_status": "completed",
            "cases_run": 12,
            "passed_cases": 12,
            "failed_cases": 0,
            "task_success_rate": 1.0,
            "scope_violations": 0,
            "unsupported_claims": 0,
        }

    monkeypatch.setattr(benchmark, "run_workspace_agent_benchmark", fake_benchmark, raising=False)
    assert ak.main(["benchmark-agent", "--all", "--json"]) == 0
    json.loads(capsys.readouterr().out)
    assert calls[0]["case_ids"] is None


def test_benchmark_agent_uses_configured_default_provider_when_flag_is_omitted(
    isolated_gateway, monkeypatch, capsys
):
    isolated_gateway.mkdir(parents=True, exist_ok=True)
    xcfg.set_global("llm", "provider", "anthropic")
    calls: list[dict] = []

    def fake_benchmark(**kwargs):
        calls.append(kwargs)
        assert os.environ["EVOLUTION_PRIMARY_PROVIDER"] == "anthropic"
        assert os.environ["EVOLUTION_PROVIDER_STRICT"] == "1"
        return {
            "execution_status": "completed",
            "provider": "anthropic",
            "cases_run": 1,
            "passed_cases": 1,
            "failed_cases": 0,
            "task_success_rate": 1.0,
            "scope_violations": 0,
            "unsupported_claims": 0,
        }

    monkeypatch.setattr(benchmark, "run_workspace_agent_benchmark", fake_benchmark, raising=False)
    assert ak.main(["benchmark-agent", "--json"]) == 0
    json.loads(capsys.readouterr().out)
    assert calls[0]["provider"] == "anthropic"
