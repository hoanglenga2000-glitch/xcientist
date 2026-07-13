"""Phase-1 tests for the xsci terminal-agent skeleton: config layering,
secure secret writing, doctor self-check, and the CLI dispatch surface."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from xsci import config as xcfg
from xsci.__main__ import main


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    """Point the global config dir at a temp location and clear env overrides."""
    home = tmp_path / "home_xsci"
    monkeypatch.setattr(xcfg, "GLOBAL_DIR", home)
    monkeypatch.setattr(xcfg, "GLOBAL_CONFIG", home / "config.toml")
    monkeypatch.setattr(xcfg, "SECRETS_FILE", home / "secrets.toml")
    for env in list(xcfg._ENV_MAP):
        monkeypatch.delenv(env, raising=False)
    return home


def test_precedence_project_overrides_global(isolated_home, tmp_path, monkeypatch):
    isolated_home.mkdir(parents=True)
    (isolated_home / "config.toml").write_text(
        '[llm]\nprovider = "anthropic"\nmodel = "global-model"\n', encoding="utf-8"
    )
    proj = tmp_path / "proj"
    (proj / ".xsci").mkdir(parents=True)
    (proj / ".xsci" / "config.toml").write_text('[llm]\nmodel = "proj-model"\n', encoding="utf-8")
    monkeypatch.chdir(proj)

    cfg = xcfg.load_config()
    assert cfg.get("llm.provider") == "anthropic"     # from global
    assert cfg.get("llm.model") == "proj-model"        # project wins


def test_env_overrides_files(isolated_home, monkeypatch):
    isolated_home.mkdir(parents=True)
    (isolated_home / "config.toml").write_text('[llm]\nprovider = "anthropic"\n', encoding="utf-8")
    monkeypatch.setenv("XSCI_LLM_PROVIDER", "deepseek")
    cfg = xcfg.load_config()
    assert cfg.get("llm.provider") == "deepseek"       # env wins over file


def test_write_secret_goes_to_global_not_project(isolated_home, tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    (proj / ".xsci").mkdir(parents=True)
    monkeypatch.chdir(proj)

    path = xcfg.write_secret("anthropic_api_key", "sk-test-123")
    expected = isolated_home / ("secrets.dpapi.json" if os.name == "nt" else "secrets.toml")
    assert path == expected
    # secret must NOT be written anywhere under the project dir
    assert not (proj / ".xsci" / "secrets.toml").exists()
    cfg = xcfg.load_config()
    assert cfg.get("secrets.anthropic_api_key") == "sk-test-123"
    if os.name == "nt":
        assert b"sk-test-123" not in path.read_bytes()


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI migration only")
def test_legacy_plaintext_secret_is_migrated_to_dpapi(isolated_home):
    isolated_home.mkdir(parents=True)
    legacy = isolated_home / "secrets.toml"
    legacy.write_text('[secrets]\nanthropic_api_key = "legacy-test-value"\n', encoding="utf-8")

    cfg = xcfg.load_config()

    encrypted = isolated_home / "secrets.dpapi.json"
    assert cfg.get("secrets.anthropic_api_key") == "legacy-test-value"
    assert encrypted.exists()
    assert b"legacy-test-value" not in encrypted.read_bytes()
    assert not legacy.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI validation only")
def test_malformed_dpapi_store_fails_closed(isolated_home):
    isolated_home.mkdir(parents=True)
    (isolated_home / "secrets.dpapi.json").write_text('{"schema":"wrong","secrets":{}}', encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        xcfg.load_config()


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI concurrency only")
def test_concurrent_dpapi_writers_preserve_both_secrets(isolated_home):
    env = os.environ.copy()
    env["XSCI_HOME"] = str(isolated_home)
    env["PYTHONPATH"] = str(Path(xcfg.__file__).resolve().parents[1])
    script = "from xsci.config import write_secret; import sys; write_secret(sys.argv[1], sys.argv[2])"
    first = subprocess.Popen([sys.executable, "-c", script, "anthropic_api_key", "fixture-alpha"], env=env)
    second = subprocess.Popen([sys.executable, "-c", script, "deepseek_api_key", "fixture-beta"], env=env)

    assert first.wait(timeout=45) == 0
    assert second.wait(timeout=45) == 0
    cfg = xcfg.load_config()
    assert cfg.get("secrets.anthropic_api_key") == "fixture-alpha"
    assert cfg.get("secrets.deepseek_api_key") == "fixture-beta"
    payload = (isolated_home / "secrets.dpapi.json").read_bytes()
    assert b"fixture-alpha" not in payload
    assert b"fixture-beta" not in payload


@pytest.mark.skipif(os.name == "nt", reason="POSIX perms not enforced on Windows")
def test_secret_file_is_0600(isolated_home):
    path = xcfg.write_secret("kaggle_key", "abc")
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600


def test_require_raises_on_missing(isolated_home):
    cfg = xcfg.load_config()
    with pytest.raises(KeyError):
        cfg.require("llm.model")


def test_doctor_runs_and_reports(isolated_home, capsys):
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert "environment self-check" in out
    assert "research_os engine importable" in out
    assert "compute" in out
    # HPC-only doctor blocks when the gated GPU/SSH runtime is not configured.
    assert rc == 1


def test_config_prints_sources(isolated_home, capsys):
    rc = main(["config"])
    out = capsys.readouterr().out
    assert "config sources" in out and "compute.backend" in out
    assert rc == 0


def test_config_repr_and_cli_never_display_secret_values(isolated_home, capsys):
    secret = "fixture-secret-must-not-print"
    xcfg.write_secret("anthropic_api_key", secret)
    cfg = xcfg.load_config()

    assert secret not in repr(cfg)
    assert "[redacted]" in repr(cfg)
    assert main(["config", "secrets.anthropic_api_key"]) == 0
    output = capsys.readouterr().out
    assert secret not in output
    assert "redacted" in output


def test_no_command_prints_help(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert "XCIENTIST" in out and "doctor" in out
    assert rc == 0


def test_watch_renders_events(tmp_path, monkeypatch, capsys):
    from xsci.project import run_init

    monkeypatch.chdir(tmp_path)
    run_init()
    run_dir = tmp_path / "experiments" / "evolution" / "demo_local_20260101_000000"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({
            "task": "demo", "best_exp_id": "EXP000", "best_cv_score": 0.9,
            "metric": "accuracy", "metric_direction": "maximize",
            "n_iterations": 1, "n_promotions": 1,
        }),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        json.dumps({"seq": 1, "type": "run_begin", "task": "demo", "metric": "accuracy"}) + "\n"
        + json.dumps({"seq": 2, "type": "run_end", "best_exp_id": "EXP000", "best_cv_score": 0.9}) + "\n",
        encoding="utf-8",
    )

    rc = main(["watch", "demo"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "watching:" in out and "demo" in out and "EXP000" in out


def test_memory_lists_retrospective_lessons(tmp_path, monkeypatch, capsys):
    from research_os.retrospective_memory import MemoryRecord, RetrospectiveMemoryStore
    from xsci.project import run_init

    monkeypatch.chdir(tmp_path)
    run_init()
    mem = tmp_path / "experiments" / "evolution" / "retrospective_memory.json"
    RetrospectiveMemoryStore(mem).add_memory(MemoryRecord(
        memory_id="demo:EXP000", task_type="classification", dataset_profile={},
        method="Base", what_worked="clean baseline", what_failed="",
        metric_delta=0.01, reusable_strategy="stratified kfold",
        failure_pattern="", linked_exp_ids=["EXP000"],
    ))

    rc = main(["memory", "successes", "--task-type", "classification"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "demo:EXP000" in out and "stratified kfold" in out


def test_dashboard_dispatches_manager(monkeypatch):
    calls = []

    def fake_dashboard(command, *, port, timeout, build, force):
        calls.append((command, port, timeout, build, force))
        return 0

    monkeypatch.setattr("xsci.dashboard.run_dashboard", fake_dashboard)
    rc = main(["dashboard", "status", "--port", "8090"])
    assert rc == 0
    assert calls == [("status", 8090, 45.0, False, False)]
