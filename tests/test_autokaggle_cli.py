"""Smoke tests for the installable Kaggle Research Agent terminal."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from xsci import config as xcfg
from xsci import kaggle as ak
from xsci import kaggle_intent as ki
from xsci.kaggle_session import SessionState


@pytest.fixture()
def isolated_autokaggle(tmp_path, monkeypatch):
    home = tmp_path / "home_xsci"
    monkeypatch.setattr(xcfg, "GLOBAL_DIR", home)
    monkeypatch.setattr(xcfg, "GLOBAL_CONFIG", home / "config.toml")
    monkeypatch.setattr(xcfg, "SECRETS_FILE", home / "secrets.toml")
    monkeypatch.setattr(xcfg, "ONBOARDED_MARKER", home / "onboarded.json")
    monkeypatch.setattr(ak, "GLOBAL_DIR", home)
    for env in list(xcfg._ENV_MAP):
        monkeypatch.delenv(env, raising=False)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    return home


def test_autokaggle_help_mentions_product_shell(isolated_autokaggle, capsys):
    rc = ak.main(["--help"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Kaggle Research Agent" in out
    assert "kaggle official" in out
    assert "autokaggle" in out
    assert "http://127.0.0.1:8088/?page=control" in out


def test_active_root_scaffolds_global_workspace(isolated_autokaggle):
    root = xcfg.active_root()
    assert root == isolated_autokaggle / "workspace"
    assert (root / ".xsci" / "tasks").is_dir()
    assert (root / "experiments").is_dir()


def test_task_list_works_from_outside_project(isolated_autokaggle, capsys):
    rc = ak.main(["task", "list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no tasks yet" in out
    assert (isolated_autokaggle / "workspace" / ".xsci" / "tasks").is_dir()


def test_add_url_registers_global_task(isolated_autokaggle, capsys):
    rc = ak.main(["task", "add", "https://www.kaggle.com/competitions/spaceship-titanic"])
    out = capsys.readouterr().out
    task = isolated_autokaggle / "workspace" / ".xsci" / "tasks" / "spaceship-titanic.json"
    assert rc == 0
    assert "registered task: spaceship-titanic" in out
    assert task.exists()


def test_register_existing_url_returns_clean_slug(isolated_autokaggle):
    root = xcfg.active_root()
    url = "https://www.kaggle.com/competitions/spaceship-titanic"
    assert ak._register_task(url, root) == "spaceship-titanic"
    assert ak._register_task(url, root) == "spaceship-titanic"


def test_agent_dispatch_uses_autokaggle_agent(monkeypatch, isolated_autokaggle):
    calls = []

    def fake_run_agent(task, root, *, goal="", compute="", resume=False):
        calls.append((task, Path(root), goal, compute, resume))
        return 0

    monkeypatch.setattr(ak, "_run_agent", fake_run_agent)
    rc = ak.main(["agent", "spaceship-titanic", "improve", "cv"])
    assert rc == 0
    assert calls == [("spaceship-titanic", isolated_autokaggle / "workspace", "improve cv", "", False)]


def test_greeting_in_console_replies_without_running_agent(monkeypatch, isolated_autokaggle, capsys):
    calls = []
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/nyc-taxi-fare-prediction", root)

    def fake_run_agent(*args, **kwargs):
        calls.append((args, kwargs))
        return 0

    monkeypatch.setattr(ak, "_run_agent", fake_run_agent)
    rc, selected, should_exit = ak._handle_console_command("你好", root, "nyc-taxi-fare-prediction")
    out = capsys.readouterr().out
    assert rc == 0
    assert selected == "nyc-taxi-fare-prediction"
    assert not should_exit
    assert not calls
    assert "Kaggle Agent" in out
    assert "对话终端" in out


def test_run_intent_without_llm_guides_setup(isolated_autokaggle, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/nyc-taxi-fare-prediction", root)
    rc, selected, should_exit = ak._handle_console_command("开始训练第二轮自进化", root, "nyc-taxi-fare-prediction")
    out = capsys.readouterr().out
    assert rc == 1
    assert selected == "nyc-taxi-fare-prediction"
    assert not should_exit
    assert "Setup needed" in out
    assert "LLM API" in out
    assert "不会贸然开始训练" in out


def test_status_command_reports_missing_setup(isolated_autokaggle, capsys):
    root = xcfg.active_root()
    rc, selected, should_exit = ak._handle_console_command("status", root, None)
    out = capsys.readouterr().out
    assert rc == 0
    assert selected is None
    assert not should_exit
    assert "System status" in out
    assert "LLM API" in out
    assert "Kaggle API" in out


def test_official_passthrough_restores_argv(monkeypatch):
    calls = []
    fake_pkg = types.ModuleType("kaggle")
    fake_cli = types.ModuleType("kaggle.cli")

    def fake_official_main():
        calls.append(list(sys.argv))

    fake_cli.main = fake_official_main
    monkeypatch.setitem(sys.modules, "kaggle", fake_pkg)
    monkeypatch.setitem(sys.modules, "kaggle.cli", fake_cli)
    before = list(sys.argv)
    rc = ak.official_main(["competitions", "list"])
    assert rc == 0
    assert calls == [["kaggle", "competitions", "list"]]
    assert sys.argv == before


def test_real_chinese_intents_are_not_mojibake():
    assert ki.classify("帮我规划第二轮自进化").kind == ki.PLANNING
    assert ki.classify("先想一下这个比赛的思路").kind == ki.PLANNING
    assert ki.classify("开始训练第二轮自进化").kind == ki.EXECUTION
    assert ki.classify("运行自进化").kind == ki.EXECUTION
    assert ki.classify("执行下一轮").kind == ki.EXECUTION
    assert ki.classify("自进化").kind == ki.EXECUTION
    assert ki.classify("你好").kind == ki.GREETING


def test_hard_now_overrides_planning_verb():
    assert ki.classify("现在就按这个规划开跑").kind == ki.EXECUTION


def test_command_and_query_intents():
    assert ki.classify("task add https://kaggle.com/c/x").kind == ki.TASK_ADD
    assert ki.classify("official competitions list").kind == ki.OFFICIAL
    assert ki.classify("status").kind == ki.STATUS
    assert ki.classify("你能做什么").kind == ki.CAPABILITY
    assert ki.classify("这个数据长什么样").kind == ki.CHAT


def test_planning_in_console_plans_without_running_agent(monkeypatch, isolated_autokaggle, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    calls = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: calls.append((a, k)) or 0)
    rc, selected, should_exit = ak._handle_console_command("帮我规划第二轮自进化", root, "spaceship-titanic")
    out = capsys.readouterr().out
    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not calls
    assert "Research plan" in out


def test_slash_status_matches_bare_status(isolated_autokaggle, capsys):
    root = xcfg.active_root()
    rc, _, _ = ak._handle_console_command("/status", root, None)
    out = capsys.readouterr().out
    assert rc == 0
    assert "System status" in out
    assert "LLM API" in out


def test_execution_without_task_refuses_cleanly(monkeypatch, isolated_autokaggle, capsys):
    root = xcfg.active_root()
    calls = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: calls.append((a, k)) or 0)
    rc, selected, should_exit = ak._handle_console_command("开始训练", root, None)
    out = capsys.readouterr().out
    assert rc == 1
    assert selected is None
    assert not should_exit
    assert not calls
    assert "还没有选中比赛" in out


def test_session_state_persists_snapshot(isolated_autokaggle):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)
    path = state.persist(root)
    assert path is not None and path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["selected_task"] == "spaceship-titanic"
    assert payload["llm_ready"] is False
    assert "anthropic_api_key" not in json.dumps(payload)
    assert "secrets" not in payload


def test_session_marks_configured_gpu_blocked_by_external_manifest(isolated_autokaggle):
    root = xcfg.active_root()
    manifest = Path.cwd() / "configs" / "external_resources.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "\n".join([
            "resources:",
            "  hpc_gpu_ssh:",
            "    status: \"configured_channels_closed\"",
            "    current_blocker: \"All local HPC SSH/SOCKS channels were intentionally closed.\"",
        ]),
        encoding="utf-8",
    )
    xcfg.set_global("compute", "backend", "gpu")
    xcfg.set_global("gpu_ssh", "host", "127.0.0.1")
    xcfg.set_global("gpu_ssh", "user", "aimslab-test")

    state = SessionState.from_root(root)
    assert state.gpu_ready is True
    assert state.gpu_blocked is True
    assert state.can_execute() is False
    assert ("gpu/ssh", "blocked (configured_channels_closed)") in state.status_rows()
    assert any("fresh GPU smoke" in gap for gap in state.missing_setup())


def test_run_console_end_to_end_smoke(isolated_autokaggle, monkeypatch, capsys):
    import builtins

    from xsci import agent as xagent

    def _tripwire(*args, **kwargs):
        raise AssertionError("deep research agent ran without an LLM key")

    monkeypatch.setattr(xagent, "run_agent", _tripwire)
    root = xcfg.active_root()
    scripted = iter(
        [
            "你好",
            "status",
            "task add https://www.kaggle.com/competitions/spaceship-titanic",
            "帮我规划第二轮自进化",
            "开始训练",
            "help",
            "exit",
        ]
    )

    def fake_input(prompt=""):
        try:
            return next(scripted)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(builtins, "input", fake_input)
    rc = ak.run_console(root)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Kaggle Research Agent" in out
    assert "对话终端" in out
    assert "LLM API" in out and "Kaggle API" in out
    assert "selected task: spaceship-titanic" in out
    assert "Research plan" in out
    assert "Setup needed" in out
    assert "Commands" in out
    snap = root / ".xsci" / "session.json"
    payload = json.loads(snap.read_text(encoding="utf-8"))
    assert payload["selected_task"] == "spaceship-titanic"
    assert "anthropic_api_key" not in json.dumps(payload)


def test_slash_task_add_routes_and_selects(isolated_autokaggle, capsys):
    root = xcfg.active_root()
    rc, selected, should_exit = ak._handle_console_command(
        "/task add https://www.kaggle.com/competitions/spaceship-titanic", root, None
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert not should_exit
    assert selected == "spaceship-titanic"
    assert "selected task: spaceship-titanic" in out


def test_slash_use_selects_existing_task(isolated_autokaggle, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    rc, selected, _ = ak._handle_console_command("/use spaceship-titanic", root, None)
    out = capsys.readouterr().out
    assert rc == 0
    assert selected == "spaceship-titanic"
    assert "selected task: spaceship-titanic" in out


def test_task_list_lists_registered_tasks(isolated_autokaggle, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    ak._register_task("https://www.kaggle.com/competitions/titanic", root)
    rc, _, should_exit = ak._handle_console_command("task list", root, "titanic")
    out = capsys.readouterr().out
    assert rc == 0 and not should_exit
    assert "Registered tasks" in out
    assert "spaceship-titanic" in out and "titanic" in out


def test_session_reloads_selected_task_across_restart(isolated_autokaggle):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    ak._register_task("https://www.kaggle.com/competitions/titanic", root)
    s1 = SessionState.from_root(root)
    s1.selected_task = "titanic"
    s1.last_goal = "lift valid submission rate"
    s1.persist(root)
    s2 = SessionState.from_root(root)
    assert s2.selected_task == "titanic"
    assert s2.last_goal == "lift valid submission rate"


def test_session_drops_deleted_selected_task(isolated_autokaggle):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)
    state.selected_task = "a-task-that-was-deleted"
    state.persist(root)
    reloaded = SessionState.from_root(root)
    assert reloaded.selected_task == "spaceship-titanic"


def test_task_brief_grounds_context(isolated_autokaggle):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)
    assert "metric=accuracy" in state.task_brief
    assert "UNFILLED" in state.task_brief
    from xsci.kaggle_conversation import ConversationAgent

    block = ConversationAgent()._context_block(state)
    assert "task = " in block
    assert "accuracy" in block


class _FakeStdin:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_bare_kaggle_without_llm_leads_setup_before_chat(isolated_autokaggle, monkeypatch):
    calls = []
    monkeypatch.setattr(ak, "run_setup", lambda **kw: calls.append(("setup", kw.get("reason"))) or 0)
    monkeypatch.setattr(ak, "run_console", lambda root=None: calls.append(("console", None)) or 0)
    monkeypatch.setattr(ak, "_has_llm", lambda cfg=None: False)
    monkeypatch.setattr(ak.sys, "stdin", _FakeStdin(True))
    rc = ak.main([])
    assert rc == 0
    assert [c[0] for c in calls] == ["setup", "console"]


def test_bare_kaggle_with_llm_opens_chat_directly(isolated_autokaggle, monkeypatch):
    calls = []
    monkeypatch.setattr(ak, "run_setup", lambda **kw: calls.append("setup") or 0)
    monkeypatch.setattr(ak, "run_console", lambda root=None: calls.append("console") or 0)
    monkeypatch.setattr(ak, "_has_llm", lambda cfg=None: True)
    monkeypatch.setattr(ak.sys, "stdin", _FakeStdin(True))
    rc = ak.main([])
    assert rc == 0
    assert calls == ["console"]


def test_setup_gate_never_fires_without_a_tty(isolated_autokaggle, monkeypatch):
    calls = []
    monkeypatch.setattr(ak, "run_setup", lambda **kw: calls.append("setup") or 0)
    monkeypatch.setattr(ak, "run_console", lambda root=None: calls.append("console") or 0)
    monkeypatch.setattr(ak, "_has_llm", lambda cfg=None: False)
    monkeypatch.setattr(ak.sys, "stdin", _FakeStdin(False))
    ak.main([])
    assert calls == ["console"]


def test_setup_llm_saves_provider_model_and_endpoint(isolated_autokaggle, monkeypatch):
    answers = iter(["1", "http://127.0.0.1:62446/anthropic", "1"])
    monkeypatch.setattr(ak, "_safe_input", lambda prompt, default="": next(answers, default))
    monkeypatch.setattr(ak.getpass, "getpass", lambda *a, **k: "sk-TESTKEY")
    assert ak._setup_llm() is True
    cfg = xcfg.load_config()
    assert cfg.get("llm.provider") == "anthropic"
    assert cfg.get("llm.anthropic_base_url") == "http://127.0.0.1:62446/anthropic"
    assert cfg.get("llm.model") == "claude-opus-4-8"
    assert cfg.get("secrets.anthropic_api_key") == "sk-TESTKEY"


def test_setup_llm_skip_saves_no_key(isolated_autokaggle, monkeypatch):
    monkeypatch.setattr(ak, "_safe_input", lambda prompt, default="": "s")
    assert ak._setup_llm() is False
    assert xcfg.load_config().get("secrets.anthropic_api_key") is None


def test_llm_model_exports_to_active_provider_env(monkeypatch):
    fake_env: dict = {}
    monkeypatch.setattr(xcfg.os, "environ", fake_env)
    cfg = xcfg.Config(data={"llm": {"provider": "anthropic", "model": "claude-opus-4-8"}})
    injected = xcfg.inject_engine_env(cfg, override=True)
    assert fake_env.get("CLAUDE_CODE_MODEL") == "claude-opus-4-8"
    assert "CLAUDE_CODE_MODEL" in injected
    assert "DEEPSEEK_MODEL" not in fake_env


def test_find_project_dir_skips_the_global_home(tmp_path, monkeypatch):
    home = tmp_path / "user"
    global_home = home / ".xsci"
    global_home.mkdir(parents=True)
    monkeypatch.setattr(xcfg, "GLOBAL_DIR", global_home)
    assert xcfg.find_project_dir(home) is None
    proj = tmp_path / "proj"
    (proj / ".xsci").mkdir(parents=True)
    assert xcfg.find_project_dir(proj) == proj


def test_active_root_from_home_uses_global_workspace_not_home(tmp_path, monkeypatch):
    home = tmp_path / "user"
    global_home = home / ".xsci"
    global_home.mkdir(parents=True)
    monkeypatch.setattr(xcfg, "GLOBAL_DIR", global_home)
    root = xcfg.active_root(home)
    assert root == global_home / "workspace"
    assert root != home
    assert (root / ".xsci" / "tasks").is_dir()
