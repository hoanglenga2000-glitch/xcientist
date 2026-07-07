"""Smoke tests for the installable EvoMind research terminal."""
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
    assert "EvoMind" in out
    assert "evomind official" in out
    assert "autokaggle" in out
    assert "http://127.0.0.1:8088/?page=control" in out


def test_console_welcome_shows_default_panel_url(isolated_autokaggle, capsys):
    root = xcfg.active_root()
    state = SessionState.from_root(root, cfg=xcfg.load_config(root))
    ak._print_welcome(state, xcfg.load_config(root))
    out = capsys.readouterr().out
    assert "Panel" in out
    assert "http://127.0.0.1:8088/?page=control" in out
    assert "evomind dashboard start" in out


def test_console_welcome_uses_user_configured_panel_url(isolated_autokaggle, monkeypatch, capsys):
    monkeypatch.setenv("EVOMIND_DASHBOARD_URL", "http://127.0.0.1:8099/?page=control")
    root = xcfg.active_root()
    cfg = xcfg.load_config(root)
    state = SessionState.from_root(root, cfg=cfg)
    ak._print_welcome(state, cfg)
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8099/?page=control" in out
    assert "http://127.0.0.1:8088/?page=control" not in out


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
    assert "EvoMind" in out
    assert ("对话终端" in out or "研究" in out or "任务" in out)


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
    assert "will not start training" in out


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


def test_model_status_query_is_deterministic_not_generic_chat(isolated_autokaggle, capsys):
    xcfg.set_global("llm", "provider", "anthropic")
    xcfg.set_global("llm", "model", "claude-opus-4-8")
    root = xcfg.active_root()
    rc, selected, should_exit = ak._handle_console_command("你现在使用的什么模型", root, None)
    out = capsys.readouterr().out
    assert rc == 0
    assert selected is None
    assert not should_exit
    assert ("Model status" in out) or ("EvoMind Tool" in out)
    assert "claude-opus-4-8" in out
    assert "provider" in out.lower() or "anthropic" in out.lower()
    # Never exposes the API key
    assert "sk-" not in out


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


def test_local_compute_override_bypasses_gpu_manifest_blocker(isolated_autokaggle, monkeypatch, capsys):
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
    xcfg.write_secret("anthropic_api_key", "sk-TEST")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    calls = []

    def fake_run_agent(task, root_arg, *, goal="", compute=None, resume=False, cfg=None):
        calls.append((task, Path(root_arg), goal, compute, resume))
        return 0

    monkeypatch.setattr(ak, "_run_agent", fake_run_agent)
    rc, selected, should_exit = ak._handle_console_command(
        "开始这个比赛的训练，目前使用本地算力就好了", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert calls and calls[0][3] == "local"
    assert "EvoMind is preparing an audited research run" in out
    assert "Selecting compute" in out
    assert "local" in out
    assert "Setup needed" not in out


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
    assert "EvoMind" in out
    assert ("对话终端" in out or "研究" in out or "任务" in out)
    assert "LLM API" in out and "Kaggle API" in out
    assert "selected task: spaceship-titanic" in out
    assert "Research plan" in out
    assert ("Setup needed" in out or "需要配置" in out)
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

    block = ConversationAgent()._build_task_aware_reply(state, "spaceship-titanic", state.missing_setup())
    assert "spaceship-titanic" in block
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


# ── EvoMind Terminal Agent upgrade tests ─────────────────────────────────

def test_model_status_tool_query_returns_structured_info(isolated_autokaggle, capsys):
    """你现在使用的什么模型 → deterministic tool output, no API key."""
    xcfg.set_global("llm", "provider", "anthropic")
    xcfg.set_global("llm", "model", "claude-opus-4-8")
    root = xcfg.active_root()
    rc, selected, should_exit = ak._handle_console_command("你现在使用的什么模型", root, None)
    out = capsys.readouterr().out
    assert rc == 0
    assert not should_exit
    assert "claude-opus-4-8" in out
    assert "sk-" not in out
    assert "provider" in out.lower() or "anthropic" in out.lower()


def test_tool_status_query_returns_tool_list(isolated_autokaggle, capsys):
    """你有哪些工具 → lists available tools, no training triggered."""
    root = xcfg.active_root()
    rc, selected, should_exit = ak._handle_console_command("你有哪些工具", root, None)
    out = capsys.readouterr().out
    assert rc == 0
    assert not should_exit
    assert ("model_status" in out or "model" in out.lower() or "tool" in out.lower())
    # Must NOT trigger training
    assert "Setup needed" not in out


def test_task_list_tool_query(isolated_autokaggle, capsys):
    """我有哪些任务 → lists registered tasks without training."""
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    rc, selected, should_exit = ak._handle_console_command("我有哪些任务", root, None)
    out = capsys.readouterr().out
    assert rc == 0
    assert not should_exit
    assert "spaceship-titanic" in out


def test_data_availability_tool_query(isolated_autokaggle, capsys):
    """这个任务数据准备好了吗 → data check, no training."""
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    rc, selected, should_exit = ak._handle_console_command(
        "这个任务数据准备好了吗", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert not should_exit
    # Either returns tool output or mentions data directory
    assert ("data" in out.lower() or "数据" in out or "not set" in out.lower())


def test_gpu_blocked_shows_repair_suggestion(isolated_autokaggle, monkeypatch, capsys):
    """用GPU服务器训练 with manifest blocked → shows repair, doesn't train."""
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
    xcfg.write_secret("anthropic_api_key", "sk-TEST")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    calls = []

    def fake_run_agent(*args, **kwargs):
        calls.append("trained")
        return 0
    monkeypatch.setattr(ak, "_run_agent", fake_run_agent)

    rc, selected, should_exit = ak._handle_console_command(
        "用GPU服务器训练", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out
    assert rc == 1
    # Should be blocked by GPU manifest
    assert not calls
    assert ("GPU" in out or "gpu" in out.lower() or "blocked" in out.lower() or "Setup needed" in out)


def test_resume_intent_routes_to_execute(isolated_autokaggle, monkeypatch, capsys):
    """继续上次实验 → routes to execution with resume=True (not chat/plan).

    A configured (even if placeholder) key can't be validated offline, so we
    assert the *routing*: the utterance reaches _run_agent for the selected
    task with the resume flag propagated.
    """
    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    calls = {}

    def fake_run_agent(task, root=None, *, goal="", compute=None, resume=False, cfg=None):
        calls["task"] = task
        calls["resume"] = resume
        return 0
    monkeypatch.setattr(ak, "_run_agent", fake_run_agent)

    rc, selected, should_exit = ak._handle_console_command(
        "继续上次实验", root, "spaceship-titanic"
    )
    assert calls.get("task") == "spaceship-titanic"
    assert calls.get("resume") is True  # resume intent propagates to the agent
    assert "spaceship-titanic" in str(selected or "")


def test_switch_task_only_switches_without_training(isolated_autokaggle, monkeypatch):
    """Bug #2: "切换到 <task>" switches the selected task and does NOT train."""
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    ak._register_task("https://www.kaggle.com/competitions/house-prices", root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "切换到 house-prices", root, "spaceship-titanic"
    )
    assert selected == "house-prices"  # switched
    assert not trained             # did NOT start training
    assert rc == 0


def test_switch_task_then_train_in_one_utterance(isolated_autokaggle, monkeypatch):
    """Bug #2: "切换到 <task> 开始训练" switches first, then runs on the NEW task."""
    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    ak._register_task("https://www.kaggle.com/competitions/house-prices", root)
    calls = {}

    def fake_run_agent(task, root=None, *, goal="", compute=None, resume=False, cfg=None):
        calls["task"] = task
        return 0
    monkeypatch.setattr(ak, "_run_agent", fake_run_agent)

    rc, selected, should_exit = ak._handle_console_command(
        "切换到 house-prices 开始训练，用本地算力", root, "spaceship-titanic"
    )
    assert selected == "house-prices"          # switched first
    assert calls.get("task") == "house-prices"  # trained the NEW task, not the old one


def test_new_chinese_intents_classify_correctly():
    """Verify all new Chinese intents route as expected."""
    assert ki.classify("你现在使用的什么模型").kind == ki.TOOL_QUERY
    assert ki.classify("当前模型").kind == ki.TOOL_QUERY
    assert ki.classify("你有哪些工具").kind == ki.TOOL_QUERY
    assert ki.classify("可以调用什么工具").kind == ki.TOOL_QUERY
    assert ki.classify("检查数据").kind == ki.TOOL_QUERY
    assert ki.classify("这个任务数据准备好了吗").kind == ki.TOOL_QUERY
    assert ki.classify("继续上次实验").kind == ki.EXECUTION
    assert ki.classify("看进度").kind == ki.TOOL_QUERY
    assert ki.classify("开始训练，用本地算力").kind == ki.EXECUTION
    assert ki.classify("用GPU服务器训练").kind == ki.EXECUTION
    assert ki.classify("查看报告").kind == ki.REPORT
    assert ki.classify("你好").kind == ki.GREETING


def test_preflight_stages_appear_in_output(isolated_autokaggle, monkeypatch, capsys):
    """Verify preflight stages are printed before training."""
    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    calls = []

    def fake_run_agent(*args, **kwargs):
        calls.append(args)
        return 0
    monkeypatch.setattr(ak, "_run_agent", fake_run_agent)

    rc, selected, should_exit = ak._handle_console_command(
        "开始这个比赛的训练，目前使用本地算力就好了", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "EvoMind is preparing an audited research run" in out
    # The 6 preflight stages
    for stage in ("Inspecting task", "Checking data", "Checking config",
                  "Selecting compute", "Planning experiment", "Entering workstation agent"):
        assert stage in out, f"missing preflight stage: {stage}"


# ── Persistence verification tests (P0/P1 gap fixes) ────────────────────

def test_recovery_guard_produces_artifact(isolated_autokaggle):
    """Verify RecoveryGuard actually writes recovery_guard.md to disk."""
    from xsci.recovery_guard import RecoveryGuard, GUARD_START, GUARD_END
    from pathlib import Path

    root = xcfg.active_root()
    state = SessionState.from_root(root)
    state.selected_task = "test_task"

    guard = RecoveryGuard()
    guard.set_state_file(root / ".xsci" / "recovery_guard.md")
    guard.record_tool("model_status: ok")
    guard.record_tool("data_check: found")
    path = guard.emit(state, event="UserPromptSubmit")

    assert path is not None, "recovery_guard emit returned None"
    assert path.exists(), f"recovery_guard.md not created at {path}"
    content = path.read_text(encoding="utf-8")
    assert GUARD_START in content, "missing RECOVERY_GUARD_AUTO start marker"
    assert GUARD_END in content, "missing RECOVERY_GUARD_AUTO end marker"
    assert "Recovery rules" in content, "missing recovery rules"
    assert "test_task" in content, "missing selected task in guard"
    # Never contains API keys
    assert "sk-" not in content, "guard contains API key pattern"


def test_tool_ledger_produces_artifact(isolated_autokaggle):
    """Verify ToolLedger writes tool_ledger.jsonl."""
    from xsci.tool_ledger import ToolLedger
    import json
    from pathlib import Path

    root = xcfg.active_root()
    ledger = ToolLedger(root)
    ledger.record("model_status", {"provider": "anthropic"}, ok=True, summary="model ready")
    ledger.record("data_check", {"train_csv": True}, ok=True, summary="data found")
    ledger.record("start_training", {}, ok=False, summary="blocked: no LLM key")

    path = root / "tool_ledger.jsonl"
    assert path.exists(), f"tool_ledger.jsonl not created at {path}"

    entries = ledger.recent(limit=10)
    assert len(entries) == 3, f"Expected 3 entries, got {len(entries)}"
    assert entries[0]["tool"] == "model_status"
    assert entries[1]["tool"] == "data_check"
    assert entries[2]["tool"] == "start_training"
    assert entries[2]["ok"] is False

    # Verify summary lines
    summary = ledger.summary_lines(limit=3)
    assert len(summary) == 3
    assert "model_status" in summary[0]


def test_evolution_tracker_produces_artifact(isolated_autokaggle):
    """Verify EvolutionTracker tracks metrics correctly and produces a report."""
    from xsci.evolution_tracker import EvolutionTracker
    import json
    from pathlib import Path

    root = xcfg.active_root()
    tracker = EvolutionTracker(root)
    tracker.record_run(success=True, cv_score=0.85, promotions=2, task="test")
    tracker.record_repair(success=True)
    tracker.record_innovation(success=True, strategy="target_encoding + oof_stacking")
    tracker.record_task_completed("test")
    tracker.record_cross_task_transfer("titanic", "house_prices")

    snapshot = tracker.current_snapshot()
    assert snapshot.total_runs == 1
    assert snapshot.repair_attempts == 1
    assert snapshot.repair_successes == 1
    assert snapshot.innovations_tried == 1
    assert snapshot.innovation_successes == 1
    assert snapshot.tasks_completed == 1
    assert snapshot.cross_task_transfers == 1
    # With 2*2 + 3 + 5 + 3 + 2 = 17 score → should be competent
    assert snapshot.skill_level in ("competent", "expert", "master")

    # Verify the report method works
    report = tracker.report()
    assert "Self-Evolution" in report
    assert str(snapshot.total_runs) in report


def test_recovery_guard_never_leaks_secrets(isolated_autokaggle):
    """RecoveryGuard must NEVER write API keys, tokens, or passwords."""
    import os
    from xsci.recovery_guard import RecoveryGuard
    from pathlib import Path

    root = xcfg.active_root()
    state = SessionState.from_root(root)

    # Set up a scenario with API key-like text in the session
    # (simulating what could happen with config data)
    guard = RecoveryGuard()
    guard.set_state_file(root / ".xsci" / "recovery_guard.md")
    guard.record_tool("model_status: provider=anthropic, key=sk-TESTKEY12345")
    path = guard.emit(state, event="UserPromptSubmit")

    assert path and path.exists()
    content = path.read_text(encoding="utf-8")
    # The guard should NOT write the key (redaction happens at the output layer)
    assert "sk-TESTKEY12345" not in content, "API key leaked in recovery guard!"


def test_terminal_agent_integration_persists_all(isolated_autokaggle):
    """TerminalAgent.handle() should persist: recovery_guard + tool_ledger."""
    from xsci.terminal_agent import TerminalAgent
    from pathlib import Path

    root = xcfg.active_root()
    xcfg.set_global("llm", "provider", "anthropic")
    xcfg.set_global("llm", "model", "claude-opus-4-8")
    state = SessionState.from_root(root)

    agent = TerminalAgent(colour=False)
    result = agent.handle("你现在使用的什么模型", state, root)

    assert result.rc == 0
    assert result.action == "tool_call"

    # Verify tool_ledger.jsonl was created
    ledger_path = root / "tool_ledger.jsonl"
    assert ledger_path.exists(), "TerminalAgent did not persist tool_ledger.jsonl!"

    # Verify recovery_guard.md was created
    guard_path = root / ".xsci" / "recovery_guard.md"
    assert guard_path.exists(), "TerminalAgent did not persist recovery_guard.md!"


def test_context_rescue_handles_edge_cases(isolated_autokaggle):
    """Context rescue: below threshold keeps all, above trims correctly."""
    from xsci.context_rescue import auto_rescue_context, estimate_body_bytes

    # Below threshold: nothing dropped
    short_msgs = [
        {"role": "system", "content": "Hi"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    kept, report = auto_rescue_context(short_msgs, target_bytes=100000, min_keep_messages=1)
    assert len(kept) == 3
    assert report["dropped"] == 0
    assert report["reason"] == "under_target"

    # Long messages: force trimming
    long_msg = {"role": "user", "content": "X" * 10000}
    many_msgs = [long_msg] * 50
    kept, report = auto_rescue_context(many_msgs, target_bytes=10000, min_keep_messages=2)
    assert len(kept) <= len(many_msgs)
    assert report["dropped"] > 0

    # Empty input: no crash
    kept, report = auto_rescue_context([], target_bytes=1000)
    assert len(kept) == 0


def test_auto_repair_diagnosis_for_all_patterns(isolated_autokaggle):
    """Auto-repair: every failure pattern maps to a repair strategy."""
    from xsci.auto_repair import diagnose_failure, _REPAIR_TEMPLATES

    patterns = [
        ("Timeout after 1800s", "timeout"),
        ("CUDA out of memory", "oom"),
        ("ModuleNotFoundError: No module named 'xgboost'", "import_error"),
        ("FileNotFoundError: train.csv", "file_not_found"),
        ("KeyError: 'target_column'", "schema_mismatch"),
        ("ValueError: invalid literal", "value_error"),
        ("did not emit CV_SCORE", "contract_violation"),
        ("SyntaxError: invalid syntax", "syntax_error"),
    ]
    for error, expected_pattern in patterns:
        diag = diagnose_failure("EXP001", error)
        assert diag.failure_pattern, f"No pattern for: {error}"
        # Every pattern must have a repair strategy
        assert diag.repair_strategy, f"No strategy for pattern: {diag.failure_pattern}"
