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
