"""Phase-3 tests: `xsci task add/list`, run-plan resolution, credential env
injection, and the `--dry-run` path (no engine execution, no network)."""
from __future__ import annotations

import json
import os

import pytest

from xsci import config as xcfg
from xsci import tasks as xtasks
from xsci.__main__ import main


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home_xsci"
    monkeypatch.setattr(xcfg, "GLOBAL_DIR", home)
    monkeypatch.setattr(xcfg, "GLOBAL_CONFIG", home / "config.toml")
    monkeypatch.setattr(xcfg, "SECRETS_FILE", home / "secrets.toml")
    for env in list(xcfg._ENV_MAP) + ["EVOLUTION_PRIMARY_PROVIDER", "ANTHROPIC_BASE_URL"]:
        monkeypatch.delenv(env, raising=False)
    return home


@pytest.fixture()
def project(tmp_path, monkeypatch, isolated_home):
    from xsci.project import run_init
    monkeypatch.chdir(tmp_path)
    run_init()
    return tmp_path


def _sample_task(path, name="titanic"):
    path.write_text(json.dumps({
        "task_name": name, "modality": "tabular", "metric": "accuracy",
        "target_column": "Survived", "local_data_dir": "/data/titanic",
        "n_train": 891, "n_test": 418,
    }), encoding="utf-8")
    return path


def test_task_add_from_json(project, tmp_path):
    src = _sample_task(tmp_path / "t.json")
    dest = xtasks.add_task(str(src))
    assert dest.name == "titanic.json"
    assert xtasks.resolve_task("titanic").exists()


def test_task_add_from_kaggle_url_scaffolds(project):
    dest = xtasks.add_task("https://www.kaggle.com/c/spaceship-titanic")
    data = json.loads(dest.read_text(encoding="utf-8"))
    assert data["task_name"] == "spaceship-titanic"
    assert data["remote_data_dirname"] == "spaceship-titanic"


def test_task_add_duplicate_needs_force(project, tmp_path):
    src = _sample_task(tmp_path / "t.json")
    xtasks.add_task(str(src))
    with pytest.raises(FileExistsError):
        xtasks.add_task(str(src))
    assert xtasks.add_task(str(src), force=True).exists()


def test_task_list_cli(project, tmp_path, capsys):
    xtasks.add_task(str(_sample_task(tmp_path / "t.json")))
    rc = main(["task", "list"])
    out = capsys.readouterr().out
    assert rc == 0 and "titanic" in out


def test_inject_engine_env_sets_names_only(isolated_home, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    xcfg.write_secret("anthropic_api_key", "sk-secret")
    xcfg.set_global("llm", "provider", "anthropic")
    cfg = xcfg.load_config()
    names = xcfg.inject_engine_env(cfg)
    assert "ANTHROPIC_API_KEY" in names and "EVOLUTION_PRIMARY_PROVIDER" in names
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-secret"
    # value is never returned, only the name
    assert "sk-secret" not in names


def test_inject_does_not_override_existing_env(isolated_home, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "pre-existing")
    xcfg.write_secret("anthropic_api_key", "sk-new")
    names = xcfg.inject_engine_env(xcfg.load_config())
    assert "ANTHROPIC_API_KEY" not in names
    assert os.environ["ANTHROPIC_API_KEY"] == "pre-existing"


def test_run_dry_run_resolves_without_executing(project, tmp_path, monkeypatch, capsys):
    xcfg.write_secret("anthropic_api_key", "sk-x")
    xcfg.set_global("llm", "provider", "anthropic")
    xtasks.add_task(str(_sample_task(tmp_path / "t.json")))
    monkeypatch.setattr("xsci.engine.execute_plan",
                        lambda plan: pytest.fail("execute_plan must NOT run in dry-run"))
    rc = main(["run", "titanic", "--dry-run", "--compute", "local"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "run plan:" in out and "dry-run" in out
    assert "data dir    : /data/titanic" in out
    assert "Scientist execution contract:" in out
    contract_path = project / ".xsci" / "scientist_execution_contract.json"
    assert contract_path.exists()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["execution_gate_decision"]["status"] == "blocked"
    assert contract["execution_gate_decision"]["no_training_started"] is True


def test_run_refuses_when_scientist_contract_is_not_training_ready(project, tmp_path, monkeypatch, capsys):
    xcfg.write_secret("anthropic_api_key", "sk-x")
    xcfg.set_global("llm", "provider", "anthropic")
    xtasks.add_task(str(_sample_task(tmp_path / "t.json")))
    monkeypatch.setattr("xsci.engine.execute_plan",
                        lambda plan: pytest.fail("execute_plan must NOT run when contract is not training-ready"))
    rc = main(["run", "titanic", "--compute", "local"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Scientist execution contract:" in out
    assert "model training  : blocked" in out
    assert "gate decision   : blocked" in out
    assert "blocked by      :" in out
    assert "safe next:" in out


def test_execution_gate_decision_is_structured_and_safe():
    from xsci.scientist_execution_gate import (
        build_execution_gate_decision,
        contract_blocks_training,
        render_execution_contract_lines,
    )

    contract = {
        "ok": True,
        "go_no_go": "no_go",
        "agent_session_ready": False,
        "model_training_ready": False,
        "data_contract_status": "blocked",
        "root_causes": ["gpu_blocked", "data_missing"],
        "setup_blockers": ["GPU smoke is stale token=LEAKME"],
        "artifact_path": "D:/workspace/.xsci/scientist_execution_contract.json",
    }

    decision = build_execution_gate_decision(contract)
    rendered = "\n".join(render_execution_contract_lines(contract))
    serialized = json.dumps(decision, ensure_ascii=False)

    assert decision["blocked"] is True
    assert decision["status"] == "blocked"
    assert "execution_contract_no_go" in decision["blocked_by"]
    assert "model_training_not_ready" in decision["blocked_by"]
    assert "evomind repair" in decision["safe_next_commands"]
    assert contract_blocks_training(contract) is True
    assert "gate decision   : blocked" in rendered
    assert "safe next       : evomind repair" in rendered
    assert "token=LEAKME" not in serialized


def test_run_refuses_without_llm_key(project, tmp_path, monkeypatch, capsys):
    for env in ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    xtasks.add_task(str(_sample_task(tmp_path / "t.json")))
    monkeypatch.setattr("xsci.engine.execute_plan",
                        lambda plan: pytest.fail("must refuse before executing"))
    rc = main(["run", "titanic", "--compute", "local"])
    out = capsys.readouterr().out
    assert rc == 1 and "no LLM key" in out


def test_task_add_missing_json_errors_not_scaffold(project):
    # a .json-looking source that doesn't exist must error, never scaffold a
    # task named after the broken path
    with pytest.raises(FileNotFoundError):
        xtasks.add_task("configs/evolution/nope.json")
    with pytest.raises(FileNotFoundError):
        xtasks.add_task("/some/where/missing.json")
    assert xtasks.list_tasks() == []


def test_run_unknown_task(project, capsys):
    rc = main(["run", "does-not-exist"])
    assert rc == 1 and "no task" in capsys.readouterr().out
