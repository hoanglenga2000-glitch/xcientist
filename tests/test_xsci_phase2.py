"""Phase-2 tests: `xsci init` scaffolding + gitignore guard, and `xsci login`
credential saving (with the invariant that secrets never touch the project)."""
from __future__ import annotations

import os

import pytest

from xsci import config as xcfg
from xsci import login as xlogin
from xsci.__main__ import main
from xsci.project import run_init


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home_xsci"
    monkeypatch.setattr(xcfg, "GLOBAL_DIR", home)
    monkeypatch.setattr(xcfg, "GLOBAL_CONFIG", home / "config.toml")
    monkeypatch.setattr(xcfg, "SECRETS_FILE", home / "secrets.toml")
    for env in list(xcfg._ENV_MAP):
        monkeypatch.delenv(env, raising=False)
    return home


def test_init_scaffolds_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = run_init(compute="gpu")
    assert rc == 0
    assert (tmp_path / ".xsci" / "config.toml").exists()
    assert (tmp_path / ".xsci" / "tasks").is_dir()
    assert (tmp_path / "experiments").is_dir()
    cfg_text = (tmp_path / ".xsci" / "config.toml").read_text(encoding="utf-8")
    assert 'backend = "gpu"' in cfg_text


def test_init_gitignore_guards_secrets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_init()
    gi = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert ".xsci/secrets*" in gi
    assert "experiments/" in gi


def test_init_appends_to_existing_gitignore_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    run_init()
    run_init(force=True)  # second run must not duplicate the block
    gi = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert gi.count(".xsci/secrets*") == 1
    assert "node_modules/" in gi  # preserved existing content


def test_init_refuses_overwrite_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_init() == 0
    assert run_init() == 1  # already initialized
    assert run_init(force=True) == 0


def test_login_non_interactive_saves_and_isolates_secrets(isolated_home, tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    (proj / ".xsci").mkdir(parents=True)
    monkeypatch.chdir(proj)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-abc")
    monkeypatch.setenv("KAGGLE_KEY", "kg-xyz")
    rc = main([
        "login", "--non-interactive",
        "--provider", "anthropic", "--kaggle-username", "alice",
    ])
    assert rc == 0
    cfg = xcfg.load_config()
    assert cfg.get("secrets.anthropic_api_key") == "sk-abc"
    assert cfg.get("secrets.kaggle_key") == "kg-xyz"
    assert cfg.get("llm.provider") == "anthropic"
    # invariant: no secret file anywhere under the project
    assert not (proj / ".xsci" / "secrets.toml").exists()


def test_login_rejects_command_line_secret_values(isolated_home, capsys):
    rc = main(["login", "--non-interactive", "--provider", "anthropic", "--api-key", "fixture-value"])

    assert rc == 1
    assert "command-line secret values are disabled" in capsys.readouterr().out
    expected = isolated_home / ("secrets.dpapi.json" if os.name == "nt" else "secrets.toml")
    assert not expected.exists()


def test_login_rejects_unknown_provider(isolated_home):
    with pytest.raises(ValueError):
        xlogin.save_llm_credentials("gpt5", "sk-x")


def test_import_kaggle_json(isolated_home, tmp_path):
    kj = tmp_path / "kaggle.json"
    kj.write_text('{"username": "bob", "key": "kk-1"}', encoding="utf-8")
    xlogin.import_kaggle_json(kj)
    cfg = xcfg.load_config()
    assert cfg.get("secrets.kaggle_username") == "bob"
    assert cfg.get("secrets.kaggle_key") == "kk-1"


def test_login_non_interactive_nothing_to_do(isolated_home):
    assert main(["login", "--non-interactive"]) == 1
