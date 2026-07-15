from __future__ import annotations

import json
import subprocess
from pathlib import Path

from xsci import kaggle
from xsci.kaggle_conversation import _forced_tool_hints, _terminal_tool_specs
from xsci.kaggle_intent import classify
from xsci.kaggle_session import SessionState
from xsci.scientist_upgrade_gateway import (
    build_activation_callback,
    initialize_upgrade_repository,
    resolve_upgrade_repository,
    run_upgrade_campaign_cli,
)
from xsci.terminal_tools import TerminalTools


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "EvoMind Test")
    _git(repository, "config", "user.email", "evomind@example.invalid")
    (repository / ".gitignore").write_text(".xsci/\n", encoding="utf-8")
    (repository / "source.txt").write_text("candidate\n", encoding="utf-8")
    _git(repository, "add", ".gitignore", "source.txt")
    _git(repository, "commit", "-q", "-m", "candidate")
    return repository


def test_status_cli_fails_closed_but_is_operational(tmp_path: Path, capsys) -> None:
    repository = _repository(tmp_path)

    exit_code = run_upgrade_campaign_cli(["status", "--json"], repository)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "blocked"
    assert payload["parity_claim_allowed"] is False


def test_evomind_dispatch_exposes_campaign_and_certification_commands(tmp_path: Path, capsys) -> None:
    repository = _repository(tmp_path)

    campaign_exit = kaggle._dispatch(["upgrade-campaign", "status", "--json"], repository)
    campaign_payload = json.loads(capsys.readouterr().out)
    certification_exit = kaggle._dispatch(["certification-status"], repository)
    certification_output = capsys.readouterr().out

    assert campaign_exit == 0
    assert campaign_payload["status"] == "blocked"
    assert certification_exit == 1
    assert "not_certified" in certification_output


def test_real_entrypoint_prefers_current_source_repo_over_global_workspace(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    repository = _repository(tmp_path)
    global_workspace = tmp_path / "global-workspace"
    global_workspace.mkdir()
    monkeypatch.chdir(repository)
    monkeypatch.setattr(kaggle, "active_root", lambda: global_workspace)

    assert resolve_upgrade_repository(global_workspace) == repository
    exit_code = kaggle.main(["upgrade-campaign", "status", "--json"])
    payload = json.loads(capsys.readouterr().out)
    certification_exit = kaggle.main(["certification-status"])
    certification_output = capsys.readouterr().out

    assert exit_code == 0
    assert payload["certification"]["artifact_path"].startswith(str(repository))
    assert certification_exit == 1
    assert "capability_certification_result.json" in certification_output
    assert str(global_workspace) not in certification_output


def test_terminal_tools_expose_certification_campaign_and_parity_status(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    state = SessionState.from_root(repository)

    certification = TerminalTools.dispatch("scientist_capability_certification", state, repository)
    campaign = TerminalTools.dispatch("scientist_upgrade_campaign", state, repository)
    parity = TerminalTools.dispatch("scientist_research_parity_gate", state, repository)

    assert certification["status"] == "not_certified"
    assert campaign["status"] == "not_run"
    assert parity["status"] == "blocked"
    assert {
        "scientist_capability_certification",
        "scientist_upgrade_campaign",
        "scientist_research_parity_gate",
    }.issubset(TerminalTools.list_tool_names())


def test_release_evidence_intents_route_to_read_only_tools() -> None:
    assert classify("show external capability certification").payload == "scientist_capability_certification"
    assert classify("show upgrade campaign status").payload == "scientist_upgrade_campaign"
    assert classify("show research parity gate").payload == "scientist_research_parity_gate"
    assert _forced_tool_hints("show external certification status") == ["scientist_capability_certification"]
    assert _forced_tool_hints("show upgrade campaign status") == ["scientist_upgrade_campaign"]
    assert _forced_tool_hints("show research parity gate") == ["scientist_research_parity_gate"]
    names = {spec.name for spec in _terminal_tool_specs()}
    assert {
        "scientist_capability_certification",
        "scientist_upgrade_campaign",
        "scientist_research_parity_gate",
    }.issubset(names)


def test_invalid_request_is_rejected_without_traceback(tmp_path: Path, capsys) -> None:
    repository = _repository(tmp_path)
    request = repository / ".xsci" / "scientist_upgrade_campaign_request.json"
    request.parent.mkdir(parents=True)
    request.write_text('{"schema":"wrong"}', encoding="utf-8")

    exit_code = run_upgrade_campaign_cli(["run", "--request", str(request), "--json"], repository)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert "unsupported upgrade campaign request schema" in payload["error"]


def test_ambiguous_request_json_is_rejected_without_traceback(tmp_path: Path, capsys) -> None:
    repository = _repository(tmp_path)
    request = repository / ".xsci" / "scientist_upgrade_campaign_request.json"
    request.parent.mkdir(parents=True)
    request.write_text('{"schema":"one","schema":"two"}\n', encoding="utf-8")

    exit_code = run_upgrade_campaign_cli(["run", "--request", str(request), "--json"], repository)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "blocked"
    assert "request is missing or invalid" in payload["error"]


def test_certification_install_cli_rejects_unverified_result(tmp_path: Path, capsys) -> None:
    repository = _repository(tmp_path)
    result_path = tmp_path / "invalid-result.json"
    result_path.write_text("{}", encoding="utf-8")

    exit_code = kaggle._dispatch([
        "certification-install",
        str(result_path),
        "--expected-sha256",
        "0" * 64,
        "--repository",
        str(repository),
    ], repository)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "rejected"


def test_activation_callback_runs_against_exact_detached_tree(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    commit = _git(repository, "rev-parse", "HEAD")
    tree = _git(repository, "rev-parse", "HEAD^{tree}")
    callback = build_activation_callback(
        [["{python}", "-c", "from pathlib import Path; assert Path('source.txt').read_text() == 'candidate\\n'"]]
    )

    result = callback(repository, commit, tree)

    assert result["passed"] is True
    assert result["runtime_tree_sha"] == tree
    assert result["checks"][0]["returncode"] == 0
    assert "stdout" not in result["checks"][0]
    assert "stderr" not in result["checks"][0]


def test_extracted_source_archive_can_initialize_clean_campaign_repository(tmp_path: Path) -> None:
    source = tmp_path / "extracted-source"
    (source / "src" / "xsci").mkdir(parents=True)
    (source / "pyproject.toml").write_text("[project]\nname='xcientist'\n", encoding="utf-8")
    (source / "README.md").write_text("# EvoMind\n", encoding="utf-8")
    (source / "LICENSE").write_text("MIT\n", encoding="utf-8")
    (source / ".gitignore").write_text(".xsci/\n", encoding="utf-8")
    (source / "src" / "xsci" / "scientist_upgrade_controller.py").write_text(
        "CONTROLLER_SCHEMA = 'fixture'\n",
        encoding="utf-8",
    )

    result = initialize_upgrade_repository(source)

    assert result["status"] == "initialized"
    assert (source / ".git").is_dir()
    assert _git(source, "status", "--porcelain=v1", "--untracked-files=all") == ""
    assert resolve_upgrade_repository(source) == source
