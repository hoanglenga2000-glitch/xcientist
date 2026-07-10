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


def test_next_action_aliases_are_cli_commands_not_task_names(isolated_autokaggle, monkeypatch):
    calls = []

    def fake_next_action(session, root):
        calls.append((session.selected_task, str(root)))
        return 0

    monkeypatch.setattr(ak, "_show_scientist_next_action", fake_next_action)

    for alias in ("next-action", "safe-next", "act-next"):
        assert ak.main([alias]) == 0

    assert len(calls) == 3


def test_continuation_status_aliases_are_cli_commands_not_task_names(isolated_autokaggle, monkeypatch):
    calls = []

    def fake_continuation_status(session, root):
        calls.append((session.selected_task, str(root)))
        return 0

    monkeypatch.setattr(ak, "_show_scientist_continuation_status", fake_continuation_status)

    for alias in ("continuation", "continuation-status", "continue-status", "turn-status"):
        assert ak.main([alias]) == 0

    assert len(calls) == 4


def test_continuation_resume_aliases_are_cli_commands_not_task_names(isolated_autokaggle, monkeypatch):
    calls = []

    def fake_continuation_resume(session, root):
        calls.append((session.selected_task, str(root)))
        return 0

    monkeypatch.setattr(ak, "_show_scientist_continuation_resume", fake_continuation_resume)

    for alias in ("resume-continuation", "resume-safe", "finish-safe-tools", "continue-tools"):
        assert ak.main([alias]) == 0

    assert len(calls) == 4


def test_readiness_report_aliases_are_cli_commands_not_task_names(isolated_autokaggle, monkeypatch):
    calls = []

    def fake_readiness_report(session, root):
        calls.append((session.selected_task, str(root)))
        return 0

    monkeypatch.setattr(ak, "_show_scientist_readiness_report", fake_readiness_report)

    for alias in ("readiness-report", "launch-readiness", "scientist-readiness", "agent-readiness"):
        assert ak.main([alias]) == 0

    assert len(calls) == 4


def test_causal_diagnosis_aliases_are_cli_commands_not_task_names(isolated_autokaggle, monkeypatch):
    calls = []

    def fake_causal_diagnosis(session, root):
        calls.append((session.selected_task, str(root)))
        return 0

    monkeypatch.setattr(ak, "_show_scientist_causal_diagnosis", fake_causal_diagnosis)

    for alias in ("causal-diagnosis", "cause-map", "root-cause-map", "causal-graph"):
        assert ak.main([alias]) == 0

    assert len(calls) == 4


def test_strategy_optimizer_aliases_are_cli_commands_not_task_names(isolated_autokaggle, monkeypatch):
    calls = []

    def fake_strategy_optimizer(session, root):
        calls.append((session.selected_task, str(root)))
        return 0

    monkeypatch.setattr(ak, "_show_scientist_strategy_optimizer", fake_strategy_optimizer)

    for alias in ("strategy", "strategy-optimizer", "priority-plan", "intervention-plan", "decision-matrix"):
        assert ak.main([alias]) == 0

    assert len(calls) == 5


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
    assert "Scientist decision:" in calls[0][2]
    assert (root / ".xsci" / "scientist_decision.json").exists()
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
    assert "live" in out
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


def test_recent_run_is_filtered_to_selected_task(isolated_autokaggle):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    ak._register_task("https://www.kaggle.com/competitions/house-prices", root)

    base = root / "experiments" / "evolution"
    spaceship = base / "spaceship_titanic_gpu_20990101_000000"
    house = base / "house_prices_gpu_20260706_230022"
    spaceship.mkdir(parents=True)
    house.mkdir(parents=True)
    (spaceship / "summary.json").write_text(json.dumps({
        "task": "spaceship_titanic",
        "best_exp_id": "EXP999",
        "best_cv_score": 0.8,
    }), encoding="utf-8")
    (house / "summary.json").write_text(json.dumps({
        "task": "house_prices",
        "best_exp_id": "EXP001",
        "best_cv_score": 0.123,
    }), encoding="utf-8")

    state = SessionState.from_root(root)
    state.selected_task = "house-prices"
    state.refresh_recent_run(root)

    assert state.recent_run_id == "house_prices_gpu_20260706_230022"
    assert state.recent_best_cv == 0.123


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
    evolution = ki.classify("它有没有学到经验")
    assert evolution.kind == ki.TOOL_QUERY
    assert evolution.payload == "evolution_status"
    checkpoint = ki.classify("像科学家一样分析下一步怎么提升")
    assert checkpoint.kind == ki.TOOL_QUERY
    assert checkpoint.payload == "scientist_checkpoint"
    autopilot = ki.classify("全面诊断当前系统为什么不够智能")
    assert autopilot.kind == ki.TOOL_QUERY
    assert autopilot.payload == "scientist_autopilot"
    workplan = ki.classify("帮我生成工作计划路线图")
    assert workplan.kind == ki.TOOL_QUERY
    assert workplan.payload == "scientist_workplan"
    repair = ki.classify("哪里卡住了，生成修复计划")
    assert repair.kind == ki.TOOL_QUERY
    assert repair.payload == "scientist_repair_plan"
    contract = ki.classify("执行前检查一下现在能不能跑")
    assert contract.kind == ki.TOOL_QUERY
    assert contract.payload == "scientist_execution_contract"
    trace = ki.classify("查看科学家工具调用过程和步骤轨迹")
    assert trace.kind == ki.TOOL_QUERY
    assert trace.payload == "scientist_step_trace"
    recovery = ki.classify("上下文丢了，帮我恢复现场看看从哪里继续")
    assert recovery.kind == ki.TOOL_QUERY
    assert recovery.payload == "scientist_recovery"
    queue = ki.classify("查看科学家行动队列和下一步命令")
    assert queue.kind == ki.TOOL_QUERY
    assert queue.payload == "scientist_action_queue"
    continuation = ki.classify("上轮没跑完，看看续跑状态和还剩哪些工具")
    assert continuation.kind == ki.TOOL_QUERY
    assert continuation.payload == "scientist_continuation_status"
    continuation_resume = ki.classify("把上轮没跑完的剩余安全工具自动跑完")
    assert continuation_resume.kind == ki.TOOL_QUERY
    assert continuation_resume.payload == "scientist_continuation_resume"
    next_action = ki.classify("执行安全下一步")
    assert next_action.kind == ki.TOOL_QUERY
    assert next_action.payload == "scientist_next_action"
    situation = ki.classify("综合分析当前局势，告诉我为什么卡住")
    assert situation.kind == ki.TOOL_QUERY
    assert situation.payload == "scientist_situation_model"
    loop = ki.classify("继续优化这个 agent，让它像 Claude Code 一样自主回合推进")
    assert loop.kind == ki.TOOL_QUERY
    assert loop.payload == "scientist_loop"
    live = ki.classify("show the real-time scientist live stream")
    assert live.kind == ki.TOOL_QUERY
    assert live.payload == "scientist_step_trace"


def test_scientist_checkpoint_tool_summarizes_state_without_secret(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)

    data_dir = root / "data" / "spaceship-titanic"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("PassengerId,Transported\n1,True\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("PassengerId\n2\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("PassengerId,Transported\n2,False\n", encoding="utf-8")

    task_path = root / ".xsci" / "tasks" / "spaceship-titanic.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["local_data_dir"] = str(data_dir)
    task["target_column"] = "Transported"
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_checkpoint", state, root)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["ok"] is True
    assert result["tool"] == "scientist_checkpoint"
    assert result["gate"]["can_execute"] is True
    assert any("train.csv" in line for line in result["observe"])
    assert result["propose"], "checkpoint should propose next research actions"
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_checkpoint_console_query_does_not_train(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "像科学家一样分析下一步怎么提升", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "scientist_checkpoint" in out
    assert "observe" in out


def test_scientist_autopilot_runs_multi_tool_chain_without_secret(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools
    from xsci.scientist_trace import load_recent_scientist_step_events
    from xsci.scientist_turns import load_recent_scientist_turns

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)

    data_dir = root / "data" / "spaceship-titanic"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("PassengerId,Transported\n1,True\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("PassengerId\n2\n", encoding="utf-8")
    task_path = root / ".xsci" / "tasks" / "spaceship-titanic.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["local_data_dir"] = str(data_dir)
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_autopilot", state, root)
    serialized = json.dumps(result, ensure_ascii=False)
    trace_names = [row["tool"] for row in result["tool_trace"]]

    assert result["ok"] is True
    assert result["tool"] == "scientist_autopilot"
    assert "system_status" in trace_names
    assert "scientist_checkpoint" in trace_names
    assert "research_decision" in trace_names
    assert "scientist_hypothesis_review" in trace_names
    assert "scientist_repair_plan" in trace_names
    assert "scientist_execution_contract" in trace_names
    for row in result["tool_trace"]:
        assert isinstance(row.get("rationale"), str)
        assert row["rationale"]
        assert isinstance(row.get("confidence"), float)
        assert 0.0 < row["confidence"] <= 1.0
        assert isinstance(row.get("evidence_signal"), str)
        assert row["evidence_signal"]
    assert result["decision"]["selected_action"] == "run_audited_baseline"
    assert result["selected_hypothesis"]
    assert result["hypothesis_review_artifact_path"].endswith("scientist_hypothesis_review.json")
    assert result["action_queue"]
    assert result["memory_reuse_plan"]["reuse_rules"]
    assert any(action["id"] == "apply_memory_reuse_plan" for action in result["action_queue"])
    memory_action = next(action for action in result["action_queue"] if action["id"] == "apply_memory_reuse_plan")
    assert memory_action["gate"] == "memory_reuse_gate"
    assert memory_action["status"] == "applied"
    assert memory_action["metadata"]["memory_reuse_plan"]["reuse_rules"]
    assert any(action["id"] == "run_gated_candidate" for action in result["action_queue"])
    run_action = next(action for action in result["action_queue"] if action["id"] == "run_gated_candidate")
    assert run_action["command"] == "evomind run spaceship-titanic"
    assert run_action["gate"] == "human_run_command_required"
    assert run_action["autonomy"] == "requires_user_run_command"
    assert run_action["metadata"]["selected_hypothesis"]
    assert run_action["metadata"]["memory_reuse_plan"]["reuse_rules"]
    assert (root / ".xsci" / "scientist_autopilot.json").exists()
    assert (root / ".xsci" / "scientist_action_queue.json").exists()
    assert (root / ".xsci" / "scientist_hypothesis_review.json").exists()
    assert (root / ".xsci" / "scientist_repair_plan.json").exists()
    assert (root / ".xsci" / "scientist_execution_contract.json").exists()
    assert (root / ".xsci" / "scientist_step_trace.jsonl").exists()
    assert (root / ".xsci" / "scientist_turns.jsonl").exists()
    assert load_recent_scientist_turns(root)[-1]["route"] == "scientist_autopilot"
    step_events = load_recent_scientist_step_events(root, limit=80)
    phases = [event.get("phase") for event in step_events]
    tools = [event.get("tool") for event in step_events]
    assert "autopilot_start" in phases
    assert "tool_completed" in phases
    assert "autopilot_complete" in phases
    assert "scientist_checkpoint" in tools
    assert "scientist_repair_plan" in tools
    assert "scientist_execution_contract" in tools
    assert all(event.get("no_training_started") is True for event in step_events)
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized

    audit = TerminalTools.dispatch("scientist_self_audit", state, root)
    orchestration = next(item for item in audit["capabilities"] if item["name"] == "tool_orchestration")
    assert "tool-choice confidence and rationale are recorded" not in orchestration["missing_checks"]
    assert not any(item["id"] == "streaming_tool_confidence" for item in audit["upgrade_backlog"])


def test_scientist_autopilot_observer_streams_live_tool_events(isolated_autokaggle):
    from xsci.terminal_tools import run_scientist_autopilot

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)
    events = []

    result = run_scientist_autopilot(state, root, observer=events.append)
    phases = [event.get("phase") for event in events]
    tools = [event.get("tool") for event in events]

    assert result["ok"] is True
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert result["action_queue_artifact_path"].endswith("scientist_action_queue.json")
    assert any(action["id"] == "preserve_claim_boundary" for action in result["action_queue"])
    assert phases[0] == "autopilot_start"
    assert phases[-1] == "autopilot_complete"
    assert "tool_started" in phases
    assert "tool_completed" in phases
    assert "system_status" in tools
    assert "scientist_execution_contract" in tools
    tool_events = [event for event in events if event.get("phase") in {"tool_started", "tool_completed"}]
    assert tool_events
    for event in tool_events:
        details = event.get("details")
        assert isinstance(details, dict)
        assert isinstance(details.get("rationale"), str)
        assert details.get("rationale")
        assert isinstance(details.get("confidence"), float)
        assert 0.0 < details["confidence"] <= 1.0
        assert isinstance(details.get("evidence_signal"), str)
        assert details.get("evidence_signal")
    assert all(event.get("no_training_started") is True for event in events)


def test_scientist_autopilot_console_query_does_not_train(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "全面诊断当前系统为什么不够智能", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "scientist_autopilot" in out
    assert "tool_trace" in out
    assert "Observe system" in out
    assert "Execution contract" in out
    assert "confidence=" in out
    assert "why:" in out
    assert "action_queue" in out
    assert "command: evomind" in out
    assert "no_training_started: True" in out


def test_scientist_live_timeline_renders_gate_artifact_without_training(isolated_autokaggle, monkeypatch, capsys):
    from xsci.scientist_trace import record_scientist_step_event

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    record_scientist_step_event(root, {
        "trace_run_id": "trace-test",
        "source": "scientist_execution_contract",
        "task": "spaceship-titanic",
        "phase": "execution_contract_snapshot",
        "tool": "scientist_execution_contract",
        "status": "blocked",
        "message": "contract blocked until data gate clears",
        "artifact_path": str(root / ".xsci" / "scientist_execution_contract.json"),
        "gate": "execution_contract_gate",
        "evidence": ["agent_trace", "metrics.json"],
        "no_training_started": True,
    })

    rc, selected, should_exit = ak._handle_console_command("live", root, "spaceship-titanic")
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "Scientist live timeline" in out
    assert "timeline:" in out
    assert "execution_contract_snapshot / scientist_execution_contract" in out
    assert "gate=execution_contract_gate" in out
    assert "artifact:" in out
    assert "no_training_started: True" in out


def test_scientist_action_queue_tool_reads_current_queue(isolated_autokaggle, monkeypatch):
    from xsci.terminal_tools import TerminalTools

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-for-readiness")
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    data_dir = root / "data" / "spaceship-titanic"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("PassengerId,Transported\n1,True\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("PassengerId\n2\n", encoding="utf-8")
    task_path = root / ".xsci" / "tasks" / "spaceship-titanic.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["local_data_dir"] = str(data_dir)
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_action_queue", state, root)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["ok"] is True
    assert result["tool"] == "scientist_action_queue"
    assert result["actions"]
    assert any(action["id"] == "run_gated_candidate" for action in result["actions"])
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert "api_key" not in serialized.lower()


def test_scientist_next_action_blocks_training_gate_without_training(isolated_autokaggle, monkeypatch, capsys):
    from xsci.terminal_tools import TerminalTools

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-for-readiness")
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    data_dir = root / "data" / "spaceship-titanic"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("PassengerId,Transported\n1,True\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("PassengerId\n2\n", encoding="utf-8")
    task_path = root / ".xsci" / "tasks" / "spaceship-titanic.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["local_data_dir"] = str(data_dir)
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    state = SessionState.from_root(root)
    TerminalTools.dispatch("scientist_autopilot", state, root)
    result = TerminalTools.dispatch("scientist_next_action", state, root)
    rc, selected, should_exit = ak._handle_console_command("执行安全下一步", root, "spaceship-titanic")
    out = capsys.readouterr().out

    assert result["ok"] is True
    assert result["status"] == "blocked_by_gate"
    assert result["selected_action"]["id"] == "run_gated_candidate"
    assert result["selected_action"]["command"] == "evomind run spaceship-titanic"
    assert result["no_training_started"] is True
    assert (root / ".xsci" / "scientist_next_action.json").exists()
    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "scientist_next_action" in out
    assert "blocked_by_gate" in out


def test_scientist_next_action_executes_read_only_repair_when_setup_blocked(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    data_dir = root / "data" / "spaceship-titanic"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("PassengerId,Transported\n1,True\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("PassengerId\n2\n", encoding="utf-8")
    task_path = root / ".xsci" / "tasks" / "spaceship-titanic.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["local_data_dir"] = str(data_dir)
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    stale_queue_path = root / ".xsci" / "scientist_action_queue.json"
    stale_queue_path.write_text(json.dumps({
        "ok": True,
        "tool": "scientist_action_queue",
        "selected_task": "spaceship-titanic",
        "actions": [
            {
                "id": "clear_blockers",
                "title": "Clear setup/data gates before execution",
                "status": "ready",
                "command": "evomind ready",
                "gate": "setup_or_data_gate",
                "why": "stale shallow readiness command",
                "risk": "would repeat system_status",
                "rollback_condition": "refresh queue",
                "expected_artifacts": [".xsci/scientist_repair_plan.json"],
                "evidence": ["scientist_repair_plan"],
                "autonomy": "read_only_repair_guidance",
            }
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_next_action", state, root)

    assert result["ok"] is True
    assert result["status"] == "executed_read_only_tool"
    assert result["selected_action"]["id"] == "clear_blockers"
    assert result["selected_action"]["command"] == "evomind repair"
    assert result["selected_action"]["metadata"]["diagnostic_escalation"] is True
    assert result["executed_tool"] == "scientist_repair_plan"
    assert result["tool_result"]["tool"] == "scientist_repair_plan"
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"


def test_scientist_next_action_executes_read_only_queue_item(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    queue_path = root / ".xsci" / "scientist_action_queue.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps({
        "ok": True,
        "tool": "scientist_action_queue",
        "selected_task": "",
        "actions": [
            {
                "id": "watch_evidence",
                "title": "Watch live evidence and step trace",
                "status": "ready",
                "command": "evomind trace",
                "gate": "observability_gate",
                "why": "Need trace evidence.",
                "risk": "none",
                "rollback_condition": "regenerate autopilot if trace is empty",
                "expected_artifacts": [".xsci/scientist_step_trace.jsonl"],
                "evidence": ["scientist_step_trace"],
                "autonomy": "read_only",
            }
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_next_action", state, root)

    assert result["ok"] is True
    assert result["status"] == "executed_read_only_tool"
    assert result["executed_tool"] == "scientist_step_trace"
    assert result["tool_result"]["tool"] == "scientist_step_trace"
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"


def test_scientist_next_action_executes_hypothesis_review_read_only(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    queue_path = root / ".xsci" / "scientist_action_queue.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps({
        "ok": True,
        "tool": "scientist_action_queue",
        "selected_task": "spaceship-titanic",
        "actions": [
            {
                "id": "review_hypotheses",
                "title": "Review proposed hypotheses before execution",
                "status": "ready",
                "command": "evomind review-hypotheses",
                "gate": "hypothesis_review_gate",
                "why": "review before training",
                "risk": "none",
                "rollback_condition": "stay read-only",
                "expected_artifacts": [".xsci/scientist_hypothesis_review.json"],
                "evidence": ["scientist_hypothesis_review"],
                "autonomy": "read_only",
            }
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_next_action", state, root)

    assert result["ok"] is True
    assert result["status"] == "executed_read_only_tool"
    assert result["selected_action"]["id"] == "review_hypotheses"
    assert result["executed_tool"] == "scientist_hypothesis_review"
    assert result["tool_result"]["tool"] == "scientist_hypothesis_review"
    assert (root / ".xsci" / "scientist_hypothesis_review.json").exists()
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"


def test_scientist_next_action_executes_experiment_blueprint_read_only(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    queue_path = root / ".xsci" / "scientist_action_queue.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps({
        "ok": True,
        "tool": "scientist_action_queue",
        "selected_task": "spaceship-titanic",
        "actions": [
            {
                "id": "prepare_experiment_blueprint",
                "title": "Prepare gated experiment blueprint from reviewed hypothesis",
                "status": "ready",
                "command": "evomind blueprint",
                "gate": "experiment_blueprint_gate",
                "why": "turn the selected hypothesis into an auditable experiment plan",
                "risk": "none",
                "rollback_condition": "stay read-only until blueprint exists",
                "expected_artifacts": [".xsci/scientist_experiment_blueprint.json"],
                "evidence": ["scientist_hypothesis_review", "scientist_experiment_blueprint"],
                "autonomy": "read_only",
            }
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_next_action", state, root)

    assert result["ok"] is True
    assert result["status"] == "executed_read_only_tool"
    assert result["selected_action"]["id"] == "prepare_experiment_blueprint"
    assert result["executed_tool"] == "scientist_experiment_blueprint"
    assert result["tool_result"]["tool"] == "scientist_experiment_blueprint"
    assert (root / ".xsci" / "scientist_experiment_blueprint.json").exists()
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"


def test_scientist_loop_runs_bounded_safe_cycle_and_records_lesson(isolated_autokaggle, monkeypatch, capsys):
    from xsci.scientist_trace import load_recent_scientist_step_events
    from xsci.scientist_turns import load_recent_scientist_turns
    from xsci.terminal_tools import TerminalTools

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-for-readiness")
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    data_dir = root / "data" / "spaceship-titanic"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("PassengerId,Transported\n1,True\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("PassengerId\n2\n", encoding="utf-8")
    task_path = root / ".xsci" / "tasks" / "spaceship-titanic.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["local_data_dir"] = str(data_dir)
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_loop", state, root)
    rc, selected, should_exit = ak._handle_console_command(
        "继续优化这个 agent，让它像 Claude Code 一样自主回合推进", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["ok"] is True
    assert result["tool"] == "scientist_loop"
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert result["final_next_action"]["status"] == "blocked_by_gate"
    assert result["final_next_action"]["selected_action"]["id"] == "run_gated_candidate"
    assert any(step["tool"] == "scientist_autopilot" for step in result["steps"])
    assert any(step["tool"] == "scientist_next_action" for step in result["steps"])
    assert any(step["step"] == "memory_writeback" for step in result["steps"])
    assert result["memory_consolidation"]["tool"] == "scientist_memory_consolidation"
    assert result["memory_consolidation"]["records_added"] > 0
    assert result["memory_records_total"] >= result["memory_consolidation"]["records_added"]
    assert "AgentSession" in result["lesson"]["lesson"]
    assert (root / ".xsci" / "scientist_loop.json").exists()
    assert (root / ".xsci" / "scientist_loop_lessons.jsonl").exists()
    assert (root / ".xsci" / "scientist_memory_consolidation.json").exists()
    assert (root / "experiments" / "evolution" / "retrospective_memory.json").exists()
    assert load_recent_scientist_turns(root)[-1]["route"] == "scientist_loop"
    phases = [event.get("phase") for event in load_recent_scientist_step_events(root, limit=120)]
    assert "loop_start" in phases
    assert "loop_next_action" in phases
    assert "loop_memory_consolidation" in phases
    assert "loop_complete" in phases
    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "scientist_loop" in out
    assert "no_training_started: True" in out
    assert "test-key-for-readiness" not in serialized


def test_scientist_loop_escalates_repeated_read_only_action_to_planning_artifacts(isolated_autokaggle):
    from xsci.scientist_trace import load_recent_scientist_step_events
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_loop", state, root)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["ok"] is True
    assert result["stop_reason"] == "repetition_escalated_to_planning_artifacts"
    assert result["mode"] == "repetition_escalated_to_planning_artifacts"
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert sum(1 for step in result["steps"] if str(step.get("step") or "").startswith("safe_next_")) == 1
    assert any(step["step"] == "predicted_repetition" for step in result["steps"])
    assert result["final_next_action"]["status"] == "predicted_repeated_read_only_action"
    assert any(step["step"] == "repetition_escalation" for step in result["steps"])
    assert any(step["tool"] == "scientist_repair_plan" for step in result["steps"])
    assert any(step["tool"] == "scientist_execution_contract" for step in result["steps"])
    assert any(step["tool"] == "scientist_workplan" for step in result["steps"])
    assert any(step["step"] == "memory_writeback" for step in result["steps"])
    assert result["memory_consolidation"]["tool"] == "scientist_memory_consolidation"
    assert "spinning in place" in result["lesson"]["lesson"]
    assert (root / ".xsci" / "scientist_repair_plan.json").exists()
    assert (root / ".xsci" / "scientist_execution_contract.json").exists()
    assert (root / ".xsci" / "scientist_workplan.json").exists()
    phases = [event.get("phase") for event in load_recent_scientist_step_events(root, limit=120)]
    assert "loop_predicted_repetition" in phases
    assert "loop_repetition_escalation" in phases
    assert "sk-" not in serialized.lower()


def test_scientist_memory_consolidation_intent_routes_before_training():
    query = "\u628a\u8fd9\u8f6e\u79d1\u5b66\u5bb6\u5faa\u73af\u7684\u7ecf\u9a8c\u6c89\u6dc0\u8fdb\u957f\u671f\u8bb0\u5fc6"
    intent = ki.classify(query)

    assert intent.kind == ki.TOOL_QUERY
    assert intent.payload == "scientist_memory_consolidation"


def test_scientist_memory_consolidation_writes_retrospective_memory_without_training(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)

    loop = TerminalTools.dispatch("scientist_loop", state, root)
    result = TerminalTools.dispatch("scientist_memory_consolidation", state, root)
    serialized = json.dumps({"loop": loop, "manual": result}, ensure_ascii=False)
    memory_path = root / "experiments" / "evolution" / "retrospective_memory.json"
    memory_payload = json.loads(memory_path.read_text(encoding="utf-8"))

    assert loop["tool"] == "scientist_loop"
    assert loop["memory_consolidation"]["ok"] is True
    assert loop["memory_consolidation"]["records_added"] > 0
    assert result["ok"] is True
    assert result["tool"] == "scientist_memory_consolidation"
    assert result["records_added"] == 0
    assert result["records_total"] >= loop["memory_consolidation"]["records_added"]
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert (root / ".xsci" / "scientist_memory_consolidation.json").exists()
    assert memory_path.exists()
    assert isinstance(memory_payload, list)
    assert memory_payload
    required = {
        "memory_id", "task_type", "dataset_profile", "method", "what_worked",
        "what_failed", "metric_delta", "reusable_strategy", "failure_pattern",
        "linked_exp_ids",
    }
    assert required <= set(memory_payload[-1])
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized
    assert "sk-TEST-SHOULD-NOT-LEAK" not in json.dumps(memory_payload, ensure_ascii=False)


def test_scientist_memory_consolidation_is_idempotent(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)

    TerminalTools.dispatch("scientist_loop", state, root)
    first = TerminalTools.dispatch("scientist_memory_consolidation", state, root)
    second = TerminalTools.dispatch("scientist_memory_consolidation", state, root)
    memory_payload = json.loads((root / "experiments" / "evolution" / "retrospective_memory.json").read_text(encoding="utf-8"))
    memory_ids = [item["memory_id"] for item in memory_payload]

    assert first["records_total"] == second["records_total"]
    assert first["records_added"] == 0
    assert second["records_added"] == 0
    assert len(memory_ids) == len(set(memory_ids))


def test_scientist_memory_consolidation_feeds_innovation_backlog(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)

    TerminalTools.dispatch("scientist_loop", state, root)
    memory = TerminalTools.dispatch("scientist_memory_consolidation", state, root)
    backlog = TerminalTools.dispatch("scientist_innovation_backlog", state, root)

    assert memory["records_total"] > 0
    assert memory["records_added"] == 0
    assert backlog["memory_summary"]["retrospective_memory_records"] >= memory["records_total"]
    assert backlog["innovation_hypotheses"]
    assert backlog["no_training_started"] is True


def test_scientist_memory_consolidation_learns_patch_work_order(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    xsci = root / ".xsci"
    xsci.mkdir(parents=True, exist_ok=True)
    (xsci / "scientist_repair_plan.json").write_text(json.dumps({
        "tool": "scientist_repair_plan",
        "root_causes": ["gpu_blocked"],
        "safe_next_command": "evomind ready",
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    state = SessionState.from_root(root)

    patch = TerminalTools.dispatch("scientist_patch_work_order", state, root)
    memory = TerminalTools.dispatch("scientist_memory_consolidation", state, root)
    memory_payload = json.loads((root / "experiments" / "evolution" / "retrospective_memory.json").read_text(encoding="utf-8"))
    serialized = json.dumps({"patch": patch, "memory": memory, "records": memory_payload}, ensure_ascii=False)
    patch_records = [item for item in memory_payload if item.get("method") == "scientist_patch_work_order"]
    trial_records = [item for item in memory_payload if item.get("method") == "scientist_patch_trial_lesson"]

    assert patch["status"] == "blocked_external_gate"
    assert memory["ok"] is True
    assert memory["source_counts"]["patch_work_order_present"] is True
    assert memory["source_counts"]["patch_trials"] >= 1
    assert memory["records_added"] >= 2
    assert patch_records
    assert trial_records
    assert patch_records[-1]["failure_pattern"] == "blocked_external_gate"
    assert "clear the external gate" in patch_records[-1]["reusable_strategy"]
    assert memory["no_training_started"] is True
    assert memory["official_submit"] == "blocked_until_explicit_human_approval"
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_memory_consolidation_console_query_does_not_train(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)
    from xsci.terminal_tools import TerminalTools
    TerminalTools.dispatch("scientist_loop", state, root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "\u5de9\u56fa\u8bb0\u5fc6\uff0c\u628a\u8fd9\u8f6e\u7ecf\u9a8c\u5199\u5165\u957f\u671f\u8bb0\u5fc6",
        root,
        "spaceship-titanic",
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "scientist_memory_consolidation" in out
    assert "no_training_started: True" in out


def test_scientist_self_audit_intent_routes_before_loop():
    query = "\u50cf Claude Code \u8fd8\u5dee\u4ec0\u4e48\uff0c\u505a\u81ea\u6211\u5ba1\u8ba1"
    intent = ki.classify(query)

    assert intent.kind == ki.TOOL_QUERY
    assert intent.payload == "scientist_self_audit"


def test_scientist_readiness_report_intent_routes_before_training():
    intent = ki.classify("做一次上线前检查，告诉我现在能不能训练和安全上线")

    assert intent.kind == ki.TOOL_QUERY
    assert intent.payload == "scientist_readiness_report"


def test_scientist_causal_diagnosis_intent_routes_before_training():
    intent = ki.classify("为什么不能训练，做因果诊断和根因图")

    assert intent.kind == ki.TOOL_QUERY
    assert intent.payload == "scientist_causal_diagnosis"


def test_scientist_strategy_optimizer_intent_routes_before_generic_planning():
    intent = ki.classify("帮我做下一步策略优化，看看应该先做哪个动作")

    assert intent.kind == ki.TOOL_QUERY
    assert intent.payload == "scientist_strategy_optimizer"


def test_scientist_context_packet_intent_routes_before_generic_recovery():
    intent = ki.classify("帮我生成科学家上下文包，整理当前上下文")

    assert intent.kind == ki.TOOL_QUERY
    assert intent.payload == "scientist_context_packet"


def test_scientist_readiness_report_writes_unified_report_without_training(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_readiness_report", state, root)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["ok"] is True
    assert result["tool"] == "scientist_readiness_report"
    assert result["schema"] == "evomind.ai_scientist.readiness_report.v1"
    assert isinstance(result["overall_score"], int)
    assert result["claim_readiness"]["rank_or_medal_claim"] == "blocked_without_kaggle_response_artifact"
    assert result["claim_readiness"]["official_submit_claim"] == "blocked_until_explicit_human_approval"
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert any(item["name"] == "compute_resource_gate" for item in result["readiness_matrix"])
    assert any(item["name"] == "rank_medal_claim_gate" and item["status"] == "blocked" for item in result["readiness_matrix"])
    assert "evomind self-audit" in result["recommended_next_commands"]
    assert (root / ".xsci" / "scientist_readiness_report.json").exists()
    assert (root / ".xsci" / "scientist_readiness_report.md").exists()
    assert (root / ".xsci" / "scientist_self_audit.json").exists()
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_causal_diagnosis_writes_graph_without_training(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_causal_diagnosis", state, root)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["ok"] is True
    assert result["tool"] == "scientist_causal_diagnosis"
    assert result["schema"] == "evomind.ai_scientist.causal_diagnosis.v1"
    assert result["symptoms"]
    assert result["root_causes"]
    assert result["interventions"]
    assert result["causal_graph"]["nodes"]
    assert result["next_safe_command"]
    assert result["claim_boundary"]["rank_or_medal"] == "blocked_without_kaggle_response_artifact"
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert result["no_training_started"] is True
    assert (root / ".xsci" / "scientist_causal_diagnosis.json").exists()
    assert (root / ".xsci" / "scientist_causal_diagnosis.md").exists()
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_strategy_optimizer_ranks_interventions_without_training(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)

    causal = TerminalTools.dispatch("scientist_causal_diagnosis", state, root)
    result = TerminalTools.dispatch("scientist_strategy_optimizer", state, root)
    serialized = json.dumps(result, ensure_ascii=False)

    assert causal["tool"] == "scientist_causal_diagnosis"
    assert result["ok"] is True
    assert result["tool"] == "scientist_strategy_optimizer"
    assert result["schema"] == "evomind.ai_scientist.strategy_optimizer.v1"
    assert result["intervention_ranking"]
    assert result["selected_strategy"]["id"]
    assert result["selected_strategy"]["total_score"] >= result["intervention_ranking"][-1]["total_score"]
    assert result["decision_matrix"]["candidate_count"] >= len(result["intervention_ranking"])
    assert result["next_decision"]["selected_command"] == result["next_safe_command"]
    assert result["claim_boundary"]["rank_or_medal"] == "blocked_without_kaggle_response_artifact"
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert result["no_training_started"] is True
    assert (root / ".xsci" / "scientist_strategy_optimizer.json").exists()
    assert (root / ".xsci" / "scientist_strategy_optimizer.md").exists()
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_context_packet_compacts_state_without_training(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)

    TerminalTools.dispatch("scientist_strategy_optimizer", state, root)
    result = TerminalTools.dispatch("scientist_context_packet", state, root)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["ok"] is True
    assert result["tool"] == "scientist_context_packet"
    assert result["schema"] == "evomind.ai_scientist.context_packet.v1"
    assert result["task_profile"]["task_slug"] == "spaceship-titanic"
    assert result["readiness"]["llm_ready"] is True
    assert result["requirement_context"]["schema"] == "evomind.ai_scientist.self_evolution_context.v1"
    assert "execution_partition" in result["requirement_context"]
    assert result["active_strategy"]["present"] is True
    assert result["context_quality"]["score"] >= 0
    assert result["next_safe_command"]
    assert result["response_contract"]["must_use_context_packet"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert result["no_training_started"] is True
    assert (root / ".xsci" / "scientist_context_packet.json").exists()
    assert (root / ".xsci" / "scientist_context_packet.md").exists()
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_self_audit_writes_capability_backlog_without_training(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_self_audit", state, root)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["ok"] is True
    assert result["tool"] == "scientist_self_audit"
    assert isinstance(result["overall_score"], int)
    assert result["capability_readiness"]
    assert result["claim_readiness"]["rank_or_medal_claim"] == "blocked_without_kaggle_response_artifact"
    assert result["claim_readiness"]["official_submit_claim"] == "blocked_until_explicit_human_approval"
    assert result["capabilities"]
    assert result["capability_trend"]["records_after"] == 1
    assert result["capability_trend"]["previous_score"] is None
    assert result["capability_trend"]["current_score"] == result["overall_score"]
    assert any(item["name"] == "tool_orchestration" for item in result["capabilities"])
    assert result["upgrade_backlog"]
    assert any(item["id"] == "streaming_tool_confidence" for item in result["upgrade_backlog"])
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert (root / ".xsci" / "scientist_self_audit.json").exists()
    assert (root / ".xsci" / "scientist_upgrade_backlog.json").exists()
    assert (root / ".xsci" / "scientist_capability_trend.jsonl").exists()
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_self_audit_separates_capability_score_from_execution_claim(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_self_audit", state, root)

    assert result["overall_score"] >= 0
    assert result["claim_readiness"]["training_readiness_claim"] in {
        "ready_for_gated_training",
        "blocked_by_external_resource_or_data_gate",
    }
    if result["execution_readiness"]["runtime_execution_ready"] is False:
        assert result["launch_readiness"] != "strong_local_agent_ready"
        assert result["claim_readiness"]["training_readiness_claim"] == "blocked_by_external_resource_or_data_gate"
        assert result["claim_readiness"]["ai_scientist_parity_claim"] == "blocked_without_end_to_end_training_and_recovery_evidence"
    assert result["claim_readiness"]["rank_or_medal_claim"] == "blocked_without_kaggle_response_artifact"
    assert result["no_training_started"] is True


def test_scientist_self_audit_appends_capability_trend(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)

    first = TerminalTools.dispatch("scientist_self_audit", state, root)
    second = TerminalTools.dispatch("scientist_self_audit", state, root)
    trend_path = root / ".xsci" / "scientist_capability_trend.jsonl"
    records = [
        json.loads(line)
        for line in trend_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert first["capability_trend"]["records_after"] == 1
    assert second["capability_trend"]["records_after"] == 2
    assert second["capability_trend"]["previous_score"] == first["overall_score"]
    assert isinstance(second["capability_trend"]["score_delta"], int)
    assert len(records) == 2
    assert records[-1]["overall_score"] == second["overall_score"]
    assert records[-1]["no_training_started"] is True
    assert records[-1]["official_submit"] == "blocked_until_explicit_human_approval"


def test_scientist_self_audit_rejects_artifact_only_parity_claims(isolated_autokaggle, monkeypatch):
    import xsci.terminal_tools as tt
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    xsci = root / ".xsci"
    xsci.mkdir(parents=True, exist_ok=True)

    def write_json(name, payload):
        (xsci / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    write_json("scientist_autopilot.json", {
        "tool": "scientist_autopilot",
        "mode": "repair_first",
        "decision": {"selected_action": "observe_before_training"},
        "tool_trace": [
            {"tool": f"tool_{idx}", "rationale": "evidence-driven", "confidence": 0.8, "evidence_signal": "artifact"}
            for idx in range(6)
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    })
    write_json("scientist_loop.json", {
        "tool": "scientist_loop",
        "steps": [{"step": "observe"}, {"step": "plan"}, {"step": "reflect"}],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    })
    write_json("scientist_action_queue.json", {
        "tool": "scientist_action_queue",
        "actions": [{"id": "repair_compute", "gate": "human_gate"}],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    })
    write_json("scientist_next_action.json", {"tool": "scientist_next_action", "no_training_started": True})
    write_json("scientist_recovery_snapshot.json", {
        "tool": "scientist_recovery",
        "recovery_decision": "resume_from_selected_action",
        "resume_commands": ["evomind ready"],
    })
    write_json("scientist_workplan.json", {"tool": "scientist_workplan", "steps": [{"id": "gate_preflight"}]})
    write_json("scientist_repair_plan.json", {"tool": "scientist_repair_plan", "repair_steps": []})
    write_json("scientist_execution_contract.json", {
        "tool": "scientist_execution_contract",
        "go_no_go": "no_go",
        "human_gate": {"training": "blocked"},
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    })
    write_json("innovation_log.json", {"proposals": [{"id": "h1"}], "tried": []})
    write_json("scientist_hypothesis_review.json", {"selected_hypothesis": {"hypothesis_id": "h1", "strategy_name": "feature search"}})
    write_json("scientist_experiment_blueprint.json", {"experiment_blueprint": {"blueprint_id": "bp1"}})
    (xsci / "scientist_turns.jsonl").write_text(json.dumps({"route": "scientist_terminal_turn"}) + "\n", encoding="utf-8")
    (xsci / "scientist_step_trace.jsonl").write_text(json.dumps({"phase": "observe"}) + "\n", encoding="utf-8")
    (xsci / "scientist_loop_lessons.jsonl").write_text(json.dumps({"lesson": "gate before train"}) + "\n", encoding="utf-8")
    (xsci / "recovery_guard.md").write_text("guard", encoding="utf-8")

    ui = root / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "AiControlConsole.tsx"
    route = root / "web" / "research-agent-workstation" / "src" / "app" / "api" / "scientist" / "upgrade-plan" / "route.ts"
    summary = root / "web" / "research-agent-workstation" / "src" / "lib" / "server" / "summary.ts"
    route.parent.mkdir(parents=True, exist_ok=True)
    ui.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    ui.write_text("scientist_self_audit scientist_upgrade_plan", encoding="utf-8")
    route.write_text("export {}", encoding="utf-8")
    (route.parent.parent / "autopilot").mkdir(parents=True, exist_ok=True)
    (route.parent.parent / "autopilot" / "route.ts").write_text("export {}", encoding="utf-8")
    (route.parent.parent / "loop").mkdir(parents=True, exist_ok=True)
    (route.parent.parent / "loop" / "route.ts").write_text("export {}", encoding="utf-8")
    summary.write_text("scientist_loop scientist_autopilot", encoding="utf-8")

    monkeypatch.setattr(tt, "inspect_evolution_status", lambda session, root: {
        "retrospective_memory": {"records": 12},
        "tracker": {"reusable_lessons": 0},
    })
    monkeypatch.setattr(tt, "get_system_status", lambda session, root: {
        "blockers": ["GPU/SSH smoke is stale token=LEAKME"],
        "gpu_blocked": True,
    })

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_self_audit", state, root)
    backlog_ids = {item["id"] for item in result["upgrade_backlog"]}
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["overall_score"] < 100
    assert "ai_scientist_parity" in {item["name"] for item in result["capabilities"]}
    assert "innovation_trial_feedback_loop" in backlog_ids
    assert "resource_gate_truthfulness" in backlog_ids
    assert "codex_claude_parity_loop" in backlog_ids
    assert result["evidence_sources"]["parity"]["gpu_blocked"] is True
    assert result["evidence_sources"]["parity"]["setup_gate_enforced"] is True
    assert result["execution_readiness"]["status"] == "blocked_by_external_resource_or_data_gate"
    parity_cap = next(item for item in result["capabilities"] if item["name"] == "ai_scientist_parity")
    assert "execute: compute/data gates are either clear or hard-blocked with a contract" in parity_cap["passed_checks"]
    assert "LEAKME" not in serialized

def test_scientist_self_audit_console_query_does_not_train(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "self-audit", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "scientist_self_audit" in out
    assert "overall_score" in out
    assert "execution_readiness" in out
    assert "no_training_started: True" in out


def test_scientist_self_upgrade_loop_creates_work_order_without_training(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_self_upgrade_loop", state, root)
    serialized = json.dumps(result, ensure_ascii=False)
    work_order = result.get("work_order") or {}

    assert result["ok"] is True
    assert result["tool"] == "scientist_self_upgrade_loop"
    assert result["status"] in {"ready_for_code_agent", "no_open_upgrade_backlog"}
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert (root / ".xsci" / "scientist_self_upgrade_loop.json").exists()
    assert (root / ".xsci" / "scientist_self_upgrade_work_order.json").exists()
    assert (root / ".xsci" / "scientist_self_upgrade_action_queue.json").exists()
    assert (root / ".xsci" / "scientist_self_upgrade_trials.jsonl").exists()
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized
    if result["status"] == "ready_for_code_agent":
        assert result["selected_backlog_id"]
        assert work_order.get("acceptance_checks")
        assert work_order.get("human_gate") == "review_patch_before_merge"


def test_scientist_self_upgrade_loop_carries_requirement_context(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    xsci = root / ".xsci"
    xsci.mkdir(parents=True, exist_ok=True)
    ledger = {
        "schema": "evomind.ai_scientist.requirement_ledger.v1",
        "goal": "Make EvoMind act like a stronger AI Scientist",
        "requirements": [
            {
                "id": "memory_consolidation",
                "status": "pending",
                "gate": "memory_writeback_gate",
                "reason": "Recent lessons must be consolidated before the next branch selection",
                "evidence_needed": [".xsci/scientist_memory_consolidation.json"],
            }
        ],
        "open_requirements": ["memory_consolidation"],
        "blocked_requirements": [],
        "satisfied_requirements": [],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    (xsci / "scientist_terminal_turn.json").write_text(json.dumps({
        "tool": "scientist_terminal_turn",
        "selected_task": "spaceship-titanic",
        "requirement_ledger": ledger,
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_self_upgrade_loop", state, root)
    context = result["work_order"]["self_evolution_context"]
    partition = result["work_order"]["execution_partition"]

    assert result["ok"] is True
    assert context["schema"] == "evomind.ai_scientist.self_evolution_context.v1"
    assert "memory_consolidation" in context["open_requirements"]
    assert partition["read_only_next_actions"][0]["next_safe_command"] == "evomind memory-consolidation"
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"


def test_scientist_self_upgrade_loop_intent_routes_before_training():
    intent = ki.classify("执行自升级闭环，把 P0 能力缺口转成工单")

    assert intent.kind == ki.TOOL_QUERY
    assert intent.payload == "scientist_self_upgrade_loop"


def test_scientist_self_upgrade_console_query_does_not_train(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "self-upgrade", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "scientist_self_upgrade_loop" in out
    assert "no_training_started: True" in out


def test_scientist_patch_work_order_from_budget_exhaustion(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    xsci = root / ".xsci"
    xsci.mkdir(parents=True, exist_ok=True)
    (xsci / "scientist_terminal_turn.json").write_text(json.dumps({
        "tool": "scientist_terminal_turn",
        "selected_task": "spaceship-titanic",
        "budget_exhausted": True,
        "deferred_tools": ["scientist_autopilot"],
        "must_run_deferred_tools": ["scientist_autopilot"],
        "parity_lifecycle": {
            "schema": "evomind.ai_scientist.parity_lifecycle.v1",
            "phase_status": {
                "observe": "passed",
                "plan": "passed",
                "act": "passed",
                "reflect": "passed",
                "improve": "needs_more_tools",
            },
        },
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_patch_work_order", state, root)
    serialized = json.dumps(result, ensure_ascii=False)
    work_order = result["work_order"]

    assert result["ok"] is True
    assert result["tool"] == "scientist_patch_work_order"
    assert result["status"] == "ready_for_code_agent"
    assert result["selected_issue_id"] == "scientist_turn_budget_exhausted"
    assert "src/xsci/terminal_agent.py" in work_order["files_to_edit"]
    assert "tests/test_autokaggle_cli.py" in work_order["files_to_edit"]
    assert any("scientist_patch_work_order" in check for check in work_order["acceptance_checks"])
    assert work_order["human_gate"] == "review_patch_before_merge"
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert (root / ".xsci" / "scientist_patch_work_order.json").exists()
    assert (root / ".xsci" / "scientist_patch_action_queue.json").exists()
    assert (root / ".xsci" / "scientist_patch_trials.jsonl").exists()
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_patch_work_order_blocks_external_gate_as_non_code_patch(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    xsci = root / ".xsci"
    xsci.mkdir(parents=True, exist_ok=True)
    (xsci / "scientist_repair_plan.json").write_text(json.dumps({
        "tool": "scientist_repair_plan",
        "root_causes": ["gpu_blocked"],
        "safe_next_command": "evomind ready",
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_patch_work_order", state, root)
    work_order = result["work_order"]

    assert result["ok"] is True
    assert result["status"] == "blocked_external_gate"
    assert result["selected_issue_id"] == "external_gate_not_code_patch"
    assert work_order["files_to_edit"] == []
    assert work_order["acceptance_checks"] == []
    assert work_order["human_gate"] == "clear_external_gate_before_patch"
    assert work_order["safe_next_command"] == "evomind ready"
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"


def test_scientist_patch_work_order_uses_requirement_progress_to_block_external_gate(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    xsci = root / ".xsci"
    xsci.mkdir(parents=True, exist_ok=True)
    ledger = {
        "schema": "evomind.ai_scientist.requirement_ledger.v1",
        "goal": "Continue AI Scientist upgrade without leaking token=LEAKME",
        "requirements": [
            {
                "id": "setup_gate_clearance",
                "status": "blocked",
                "gate": "setup_gate",
                "reason": "GPU/HPC SSH smoke is still blocked password=LEAKME",
                "evidence_needed": [".xsci/scientist_repair_plan.json"],
                "execution_evidence": {
                    "mapped_tool_hits": ["scientist_repair_plan"],
                    "artifact_hits": [".xsci/scientist_repair_plan.json"],
                },
            }
        ],
        "open_requirements": ["setup_gate_clearance"],
        "blocked_requirements": ["setup_gate_clearance"],
        "satisfied_requirements": [],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    (xsci / "scientist_terminal_turn.json").write_text(json.dumps({
        "tool": "scientist_terminal_turn",
        "selected_task": "spaceship-titanic",
        "requirement_ledger": ledger,
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    (xsci / "scientist_requirement_progress.json").write_text(json.dumps({
        "schema": "evomind.ai_scientist.requirement_progress.v1",
        "requirement_id": "setup_gate_clearance",
        "before_status": "blocked",
        "after_status": "blocked",
        "safe_tool": "scientist_repair_plan",
        "tool_ok": True,
        "open_requirements": ["setup_gate_clearance"],
        "blocked_requirements": ["setup_gate_clearance"],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_patch_work_order", state, root)
    serialized = json.dumps(result, ensure_ascii=False)
    context = result["work_order"]["self_evolution_context"]
    partition = context["execution_partition"]

    assert result["ok"] is True
    assert result["status"] == "blocked_external_gate"
    assert result["selected_issue_id"] == "external_gate_not_code_patch"
    assert result["work_order"]["files_to_edit"] == []
    assert result["evidence"]["requirement_context"]["latest_requirement_progress"]["after_status"] == "blocked"
    assert "setup_gate_clearance" in context["blocked_requirements"]
    assert partition["external_resource_blockers"][0]["requirement_id"] == "setup_gate_clearance"
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert "LEAKME" not in serialized


def test_scientist_patch_work_order_separates_attempted_code_gap_from_external_gate(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    xsci = root / ".xsci"
    xsci.mkdir(parents=True, exist_ok=True)
    ledger = {
        "schema": "evomind.ai_scientist.requirement_ledger.v1",
        "goal": "Strengthen observe-plan-act-reflect-improve parity",
        "requirements": [
            {
                "id": "parity_lifecycle",
                "status": "pending",
                "gate": "scientist_parity_gate",
                "reason": "Mapped safe tool ran but lifecycle evidence is incomplete",
                "evidence_needed": [".xsci/scientist_parity_loop.jsonl"],
                "execution_evidence": {
                    "mapped_tool_hits": ["scientist_turn_plan"],
                    "artifact_hits": [".xsci/scientist_turn_plan.json"],
                },
            }
        ],
        "open_requirements": ["parity_lifecycle"],
        "blocked_requirements": [],
        "satisfied_requirements": [],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    (xsci / "scientist_terminal_turn.json").write_text(json.dumps({
        "tool": "scientist_terminal_turn",
        "selected_task": "spaceship-titanic",
        "requirement_ledger": ledger,
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_patch_work_order", state, root)
    partition = result["work_order"]["execution_partition"]

    assert result["ok"] is True
    assert result["status"] == "ready_for_code_agent"
    assert result["work_order"]["files_to_edit"]
    assert partition["external_resource_blockers"] == []
    assert partition["code_agent_fixable_requirements"][0]["requirement_id"] == "parity_lifecycle"
    assert "self_evolution_context" in result["work_order"]
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"


def test_scientist_patch_work_order_console_query_does_not_train(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "生成代码补丁工单",
        root,
        "spaceship-titanic",
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "scientist_patch_work_order" in out
    assert "no_training_started: True" in out
    assert (root / ".xsci" / "scientist_patch_work_order.json").exists()


def test_scientist_upgrade_plan_intent_routes_before_training():
    query = "把自我审计 upgrade backlog 转成工程升级计划"
    intent = ki.classify(query)

    assert intent.kind == ki.TOOL_QUERY
    assert intent.payload == "scientist_upgrade_plan"


def test_scientist_upgrade_plan_writes_engineering_plan_without_training(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools
    from xsci.scientist_turns import load_recent_scientist_turns

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    xsci = root / ".xsci"
    xsci.mkdir(parents=True, exist_ok=True)
    (xsci / "scientist_upgrade_backlog.json").write_text(json.dumps({
        "generated_at": "2026-07-09T00:00:00+00:00",
        "tool": "scientist_upgrade_backlog",
        "source": "scientist_self_audit",
        "overall_score": 58,
        "items": [
            {
                "id": "planner_executor_observer_loop",
                "title": "Strengthen planner executor observer loop password=LEAKME",
                "priority": "P0",
                "status": "proposed",
                "why": "Agent needs visible continuation before training token=LEAKME",
                "safe_next_command": "evomind loop",
                "expected_artifacts": [".xsci/scientist_loop.json"],
                "gate": "engineering_review_required",
            },
            {
                "id": "frontend_self_audit_card",
                "title": "Expose self audit on UI",
                "priority": "P1",
                "status": "done",
                "why": "closed item should not be planned",
            },
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_upgrade_plan", state, root)
    serialized = json.dumps(result, ensure_ascii=False)
    artifact = root / ".xsci" / "scientist_upgrade_plan.json"
    artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
    turns = load_recent_scientist_turns(root)

    assert result["ok"] is True
    assert result["tool"] == "scientist_upgrade_plan"
    assert result["readiness"] == "ready_for_engineering_review"
    assert result["open_backlog_count"] == 1
    assert result["planned_steps"]
    assert result["planned_steps"][0]["backlog_id"] == "planner_executor_observer_loop"
    assert "src/xsci/terminal_agent.py" in result["planned_steps"][0]["files_to_inspect"]
    assert any("pytest" in check for check in result["planned_steps"][0]["acceptance_checks"])
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert artifact.exists()
    assert artifact_payload["tool"] == "scientist_upgrade_plan"
    assert turns[-1]["route"] == "scientist_upgrade_plan"
    assert turns[-1]["no_training_started"] is True
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized
    assert "LEAKME" not in serialized


def test_scientist_upgrade_plan_console_query_does_not_train(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "升级计划", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "scientist_upgrade_plan" in out
    assert "no_training_started: True" in out


def test_scientist_innovation_backlog_intent_routes_before_training():
    query = "\u751f\u6210\u521b\u65b0\u5047\u8bbe\uff0c\u6839\u636e\u8bb0\u5fc6\u590d\u7528\u63d0\u51fa\u4e0b\u4e00\u8f6e\u5206\u652f"
    intent = ki.classify(query)

    assert intent.kind == ki.TOOL_QUERY
    assert intent.payload == "scientist_innovation_backlog"


def test_scientist_innovation_backlog_writes_proposals_without_training(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    memory_dir = root / "experiments" / "evolution"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "retrospective_memory.json").write_text(json.dumps({
        "records": [
            {
                "memory_id": "mem_001",
                "task": "spaceship_titanic",
                "task_type": "classification",
                "reusable_strategy": "target encoding + calibrated OOF ensemble",
                "what_worked": "fold stability audit reduced CV-public gap risk",
                "what_failed": "uncontrolled single-fold CV repeated public gap regressions",
                "failure_pattern": "single-fold overfit without OOF stability check",
            },
            {
                "memory_id": "mem_002",
                "task": "titanic",
                "task_type": "classification",
                "reusable_strategy": "model-family diversity blend + probability calibration",
                "what_worked": "simple feature interactions improved accuracy",
            },
        ]
    }), encoding="utf-8")
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_innovation_backlog", state, root)
    serialized = json.dumps(result, ensure_ascii=False)
    log_payload = json.loads((root / ".xsci" / "innovation_log.json").read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert result["tool"] == "scientist_innovation_backlog"
    assert result["innovation_hypotheses"]
    assert result["memory_reuse_plan"]["gate"] == "memory_reuse_before_execution"
    assert "mem_001" in result["memory_reuse_plan"]["supporting_memory_ids"]
    assert any("target encoding" in item["strategy"] for item in result["memory_reuse_plan"]["reuse_rules"])
    assert any("single-fold" in item["pattern"] for item in result["memory_reuse_plan"]["avoid_patterns"])
    assert result["innovation_hypotheses"][0]["memory_reuse_plan"]["reuse_rules"]
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert (root / ".xsci" / "scientist_innovation_backlog.json").exists()
    assert log_payload["proposals"]
    assert log_payload["tried"] == []
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized

    audit = TerminalTools.dispatch("scientist_self_audit", state, root)
    self_evolution = next(item for item in audit["capabilities"] if item["name"] == "self_evolution_memory")
    assert "innovation proposals or trials exist" not in self_evolution["missing_checks"]


def test_scientist_innovation_backlog_console_query_does_not_train(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "innovate-plan", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "scientist_innovation_backlog" in out
    assert "no_training_started: True" in out


def test_scientist_hypothesis_review_intent_routes_before_training():
    query = "\u8bc4\u5ba1\u5047\u8bbe\uff0c\u770b\u770b\u54ea\u4e2a\u65b9\u6848\u6700\u597d"
    intent = ki.classify(query)

    assert intent.kind == ki.TOOL_QUERY
    assert intent.payload == "scientist_hypothesis_review"


def test_scientist_hypothesis_review_ranks_proposals_without_training(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    memory_dir = root / "experiments" / "evolution"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "retrospective_memory.json").write_text(json.dumps({
        "records": [
            {
                "memory_id": "mem_001",
                "task": "spaceship_titanic",
                "task_type": "classification",
                "reusable_strategy": "OOF blend with fold stability audit",
                "what_worked": "probability calibration reduced leaderboard gap risk",
                "failure_pattern": "public gap widened when calibration was skipped",
            },
            {
                "memory_id": "mem_002",
                "task": "titanic",
                "task_type": "classification",
                "reusable_strategy": "feature interactions plus model-family ensemble",
                "what_worked": "diverse tree/logistic blend improved local validation",
            },
        ]
    }), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_hypothesis_review", state, root)
    serialized = json.dumps(result, ensure_ascii=False)
    artifact = root / ".xsci" / "scientist_hypothesis_review.json"

    assert result["ok"] is True
    assert result["tool"] == "scientist_hypothesis_review"
    assert result["reviews"]
    assert result["selected_hypothesis"]
    assert result["memory_reuse_plan"]["reuse_rules"]
    assert result["memory_reuse_plan"]["avoid_patterns"]
    assert result["selected_hypothesis"]["memory_reuse_plan"]["gate"] == "memory_reuse_before_execution"
    assert result["hypotheses_reviewed"] == len(result["reviews"])
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert artifact.exists()
    assert (root / ".xsci" / "scientist_innovation_backlog.json").exists()
    assert result["gate_summary"]["memory_records"] >= 2
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_review_refreshes_stale_backlog_memory_plan(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    memory_dir = root / "experiments" / "evolution"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "retrospective_memory.json").write_text(json.dumps({
        "records": [
            {
                "memory_id": "mem_stale_001",
                "task_type": "classification",
                "reusable_strategy": "target encoding with OOF stability audit",
                "what_worked": "OOF stability prevented leaderboard gap drift",
                "what_failed": "single split validation created false promotion",
                "failure_pattern": "single split validation without OOF audit",
            }
        ]
    }), encoding="utf-8")
    xsci = root / ".xsci"
    xsci.mkdir(parents=True, exist_ok=True)
    (xsci / "scientist_innovation_backlog.json").write_text(json.dumps({
        "ok": True,
        "tool": "scientist_innovation_backlog",
        "selected_task": "spaceship-titanic",
        "memory_summary": {},
        "memory_reuse_plan": {},
        "innovation_hypotheses": [
            {
                "id": "stale_h1",
                "strategy_name": "stale_without_memory",
                "components": ["baseline"],
                "evidence_records": [],
                "proposed_branch_type": "feature_engineering",
                "code_generation_mode": "stepwise",
            }
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    state = SessionState.from_root(root)

    review = TerminalTools.dispatch("scientist_hypothesis_review", state, root)
    (xsci / "scientist_hypothesis_review.json").write_text(json.dumps({
        "ok": True,
        "tool": "scientist_hypothesis_review",
        "selected_task": "spaceship-titanic",
        "memory_reuse_plan": {},
        "selected_hypothesis": {
            "hypothesis_id": "stale_h1",
            "strategy_name": "stale_without_memory",
            "branch_type": "feature_engineering",
            "code_generation_mode": "stepwise",
            "memory_reuse_plan": {},
        },
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    blueprint = TerminalTools.dispatch("scientist_experiment_blueprint", state, root)
    audit = TerminalTools.dispatch("scientist_self_audit", state, root)
    serialized = json.dumps({"review": review, "blueprint": blueprint, "audit": audit}, ensure_ascii=False)
    backlog_ids = {item["id"] for item in audit["upgrade_backlog"]}

    assert review["memory_reuse_plan"]["reuse_rules"]
    assert review["memory_reuse_plan"]["avoid_patterns"]
    assert "mem_stale_001" in review["memory_reuse_plan"]["supporting_memory_ids"]
    assert review["selected_hypothesis"]["memory_reuse_plan"]["reuse_rules"]
    assert blueprint["memory_reuse_plan"]["reuse_rules"]
    assert blueprint["experiment_blueprint"]["memory_reuse_plan"]["avoid_patterns"]
    assert audit["evidence_sources"]["memory"]["active_reuse_plan"] is True
    assert audit["evidence_sources"]["memory"]["reuse_rules"] > 0
    assert "memory_reuse_before_each_run" not in backlog_ids
    assert blueprint["no_training_started"] is True
    assert blueprint["official_submit"] == "blocked_until_explicit_human_approval"
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_hypothesis_review_console_query_does_not_train(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "评审假设哪个方案最好", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "scientist_hypothesis_review" in out
    assert "ranked_reviews" in out
    assert "no_training_started: True" in out


def test_scientist_experiment_blueprint_intent_routes_before_training():
    intent = ki.classify("generate experiment blueprint from the reviewed hypothesis")

    assert intent.kind == ki.TOOL_QUERY
    assert intent.payload == "scientist_experiment_blueprint"


def test_scientist_experiment_blueprint_writes_gated_plan_without_training(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    memory_dir = root / "experiments" / "evolution"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "retrospective_memory.json").write_text(json.dumps({
        "records": [
            {
                "memory_id": "mem_001",
                "task": "spaceship_titanic",
                "task_type": "classification",
                "reusable_strategy": "OOF blend with fold stability audit",
                "what_worked": "probability calibration reduced leaderboard gap risk",
                "what_failed": "leaderboard gap widened when calibration was skipped",
                "failure_pattern": "missing probability calibration before blend",
            }
        ]
    }), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_experiment_blueprint", state, root)
    artifact = root / ".xsci" / "scientist_experiment_blueprint.json"
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["ok"] is True
    assert result["tool"] == "scientist_experiment_blueprint"
    assert result["selected_hypothesis"]
    assert result["experiment_blueprint"]["blueprint_id"]
    assert result["experiment_blueprint"]["branch_type"]
    assert result["experiment_blueprint"]["code_generation_mode"]
    assert result["experiment_blueprint"]["resource_mode"]
    assert result["experiment_blueprint"]["required_artifacts"]
    assert result["experiment_blueprint"]["promotion_gates"]
    assert result["experiment_blueprint"]["memory_writeback_plan"]
    assert result["experiment_blueprint"]["memory_reuse_plan"]["reuse_rules"]
    assert result["experiment_blueprint"]["memory_reuse_plan"]["avoid_patterns"]
    assert any("memory_reuse_plan" in item for item in result["experiment_blueprint"]["validation_plan"])
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert artifact.exists()
    assert (root / ".xsci" / "scientist_hypothesis_review.json").exists()
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_experiment_blueprint_console_query_does_not_train(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "experiment blueprint gated plan", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "scientist_experiment_blueprint" in out
    assert "experiment_blueprint" in out
    assert "no_training_started: True" in out


def test_scientist_innovation_trial_feedback_intent_routes_before_training():
    intent = ki.classify("把假设结果写回创新日志，记录门禁反馈")

    assert intent.kind == ki.TOOL_QUERY
    assert intent.payload == "scientist_innovation_trial_feedback"

    english = ki.classify("record innovation trial feedback before training")
    assert english.kind == ki.TOOL_QUERY
    assert english.payload == "scientist_innovation_trial_feedback"


def test_scientist_innovation_trial_feedback_writes_idempotent_memory_without_training(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    memory_dir = root / "experiments" / "evolution"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "retrospective_memory.json").write_text(json.dumps({
        "records": [
            {
                "memory_id": "mem_feedback_001",
                "task": "spaceship_titanic",
                "task_type": "classification",
                "reusable_strategy": "OOF blend with calibration and fold stability",
                "what_worked": "gate-first review avoided wasting GPU on weak candidates",
                "what_failed": "training before data contract caused invalid artifacts",
                "failure_pattern": "missing data contract before execution",
            }
        ]
    }), encoding="utf-8")

    state = SessionState.from_root(root)
    first = TerminalTools.dispatch("scientist_innovation_trial_feedback", state, root)
    second = TerminalTools.dispatch("scientist_innovation_trial_feedback", state, root)
    log_payload = json.loads((root / ".xsci" / "innovation_log.json").read_text(encoding="utf-8"))
    tried = log_payload["tried"]
    audit = TerminalTools.dispatch("scientist_self_audit", state, root)
    backlog_ids = {item["id"] for item in audit["upgrade_backlog"]}
    serialized = json.dumps({"first": first, "second": second, "audit": audit}, ensure_ascii=False)

    assert first["ok"] is True
    assert first["tool"] == "scientist_innovation_trial_feedback"
    assert first["trial_feedback"]["trial_id"]
    assert first["trial_feedback"]["outcome"] in {"blocked_by_gate", "ready_for_gated_execution"}
    assert first["trial_feedback"]["memory_reuse_rule_count"] > 0
    assert first["trial_feedback"]["avoid_pattern_count"] > 0
    assert first["no_training_started"] is True
    assert first["official_submit"] == "blocked_until_explicit_human_approval"
    assert second["idempotent_update"] is True
    assert len(tried) == 1
    assert tried[0]["trial_id"] == first["trial_feedback"]["trial_id"]
    assert log_payload["last_trial_feedback"]["trial_id"] == first["trial_feedback"]["trial_id"]
    assert (root / ".xsci" / "scientist_innovation_trial_feedback.json").exists()
    assert "innovation_trial_feedback_loop" not in backlog_ids
    assert audit["evidence_sources"]["innovation"]["tried"] == 1
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_innovation_trial_feedback_console_query_does_not_train(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "innovation-feedback", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "scientist_innovation_trial_feedback" in out
    assert "no_training_started: True" in out


def test_scientist_next_action_executes_innovation_feedback_read_only(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    queue_path = root / ".xsci" / "scientist_action_queue.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps({
        "ok": True,
        "tool": "scientist_action_queue",
        "selected_task": "spaceship-titanic",
        "actions": [
            {
                "id": "record_innovation_trial_feedback",
                "title": "Record innovation gate feedback before execution",
                "status": "ready",
                "command": "evomind innovation-feedback",
                "gate": "innovation_feedback_gate",
                "why": "write gate feedback",
                "risk": "none",
                "rollback_condition": "stay read-only",
                "expected_artifacts": [".xsci/scientist_innovation_trial_feedback.json"],
                "evidence": ["scientist_innovation_trial_feedback"],
                "autonomy": "read_only",
            }
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_next_action", state, root)

    assert result["ok"] is True
    assert result["status"] == "executed_read_only_tool"
    assert result["selected_action"]["id"] == "record_innovation_trial_feedback"
    assert result["executed_tool"] == "scientist_innovation_trial_feedback"
    assert result["tool_result"]["tool"] == "scientist_innovation_trial_feedback"
    assert (root / ".xsci" / "scientist_innovation_trial_feedback.json").exists()
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"


def test_scientist_next_action_prioritizes_open_requirement_ledger_item(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    xsci = root / ".xsci"
    queue_path = xsci / "scientist_action_queue.json"
    queue_path.write_text(json.dumps({
        "ok": True,
        "tool": "scientist_action_queue",
        "selected_task": "spaceship-titanic",
        "actions": [
            {
                "id": "run_gated_candidate",
                "title": "Run candidate",
                "status": "ready",
                "command": "evomind run spaceship-titanic",
                "gate": "human_run_command_required",
                "why": "stale queue item should not outrank open requirements",
                "risk": "would require training gate",
                "rollback_condition": "hold",
                "expected_artifacts": ["metrics.json"],
                "evidence": ["metrics.json"],
                "autonomy": "requires_user_run_command",
            }
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    (xsci / "scientist_terminal_turn.json").write_text(json.dumps({
        "ok": True,
        "tool": "scientist_terminal_turn",
        "selected_task": "spaceship-titanic",
        "requirement_ledger": {
            "schema": "evomind.ai_scientist.requirement_ledger.v1",
            "requirements": [
                {
                    "id": "agent_self_audit",
                    "description": "Agent parity requires a capability audit before claims.",
                    "status": "pending",
                    "gate": "capability_audit_gate",
                    "reason": "No self-audit artifact yet.",
                    "evidence_needed": [".xsci/scientist_self_audit.json"],
                    "mapped_tools": ["scientist_self_audit"],
                }
            ],
            "open_requirements": ["agent_self_audit"],
            "blocked_requirements": [],
        },
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_next_action", state, root)
    serialized = json.dumps(result, ensure_ascii=False)
    refreshed_queue = json.loads(queue_path.read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert result["status"] == "executed_read_only_tool"
    assert result["selected_action"]["id"] == "satisfy_agent_self_audit"
    assert result["selected_action"]["metadata"]["source"] == "requirement_ledger"
    assert result["selected_action"]["metadata"]["requirement_id"] == "agent_self_audit"
    assert result["executed_tool"] == "scientist_self_audit"
    updated_turn = json.loads((xsci / "scientist_terminal_turn.json").read_text(encoding="utf-8"))
    progress = json.loads((xsci / "scientist_requirement_progress.json").read_text(encoding="utf-8"))
    updated_req = updated_turn["requirement_ledger"]["requirements"][0]
    assert updated_req["id"] == "agent_self_audit"
    assert updated_req["status"] == "satisfied"
    assert updated_turn["requirement_ledger"]["open_requirements"] == []
    assert progress["requirement_id"] == "agent_self_audit"
    assert progress["after_status"] == "satisfied"
    assert "requirement_ledger_summary" not in refreshed_queue
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_next_action_uses_requirement_ledger_to_repair_setup_blocker(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    xsci = root / ".xsci"
    queue_path = xsci / "scientist_action_queue.json"
    queue_path.write_text(json.dumps({
        "ok": True,
        "tool": "scientist_action_queue",
        "selected_task": "spaceship-titanic",
        "actions": [
            {
                "id": "run_gated_candidate",
                "title": "Run candidate",
                "status": "ready",
                "command": "evomind run spaceship-titanic",
                "gate": "human_run_command_required",
                "why": "stale queue item should not run while setup gate is blocked",
                "risk": "would require training gate",
                "rollback_condition": "hold",
                "expected_artifacts": ["metrics.json"],
                "evidence": ["metrics.json"],
                "autonomy": "requires_user_run_command",
            }
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    (xsci / "scientist_terminal_turn.json").write_text(json.dumps({
        "ok": True,
        "tool": "scientist_terminal_turn",
        "selected_task": "spaceship-titanic",
        "requirement_ledger": {
            "schema": "evomind.ai_scientist.requirement_ledger.v1",
            "requirements": [
                {
                    "id": "setup_gate_clearance",
                    "description": "Setup gates must be clear before execution-like work can start.",
                    "status": "blocked",
                    "gate": "setup_gate",
                    "reason": "HPC SSH fresh smoke is missing.",
                    "evidence_needed": ["evomind ready", ".xsci/scientist_repair_plan.json"],
                    "mapped_tools": ["system_status", "scientist_repair_plan"],
                }
            ],
            "open_requirements": ["setup_gate_clearance"],
            "blocked_requirements": ["setup_gate_clearance"],
        },
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_next_action", state, root)
    refreshed_queue = json.loads(queue_path.read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert result["status"] == "executed_read_only_tool"
    assert result["selected_action"]["id"] == "clear_blockers"
    assert result["selected_action"]["metadata"]["source"] == "requirement_ledger"
    assert result["selected_action"]["metadata"]["requirement_id"] == "setup_gate_clearance"
    assert result["executed_tool"] == "scientist_repair_plan"
    progress = json.loads((xsci / "scientist_requirement_progress.json").read_text(encoding="utf-8"))
    updated_turn = json.loads((xsci / "scientist_terminal_turn.json").read_text(encoding="utf-8"))
    assert progress["requirement_id"] == "setup_gate_clearance"
    assert progress["after_status"] == "blocked"
    assert updated_turn["requirement_ledger"]["blocked_requirements"] == ["setup_gate_clearance"]
    assert refreshed_queue["requirement_ledger_summary"]["blocked_requirements"] == ["setup_gate_clearance"]
    assert refreshed_queue["actions"][0]["id"] == "clear_blockers"
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"


def test_scientist_next_action_moves_past_repeated_external_blocker_requirement(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    xsci = root / ".xsci"
    queue_path = xsci / "scientist_action_queue.json"
    queue_path.write_text(json.dumps({
        "ok": True,
        "tool": "scientist_action_queue",
        "selected_task": "spaceship-titanic",
        "actions": [
            {
                "id": "run_gated_candidate",
                "title": "Run candidate",
                "status": "ready",
                "command": "evomind run spaceship-titanic",
                "gate": "human_run_command_required",
                "why": "training gate must stay behind requirements",
                "risk": "would require training gate",
                "rollback_condition": "hold",
                "expected_artifacts": ["metrics.json"],
                "evidence": ["metrics.json"],
                "autonomy": "requires_user_run_command",
            }
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    (xsci / "scientist_terminal_turn.json").write_text(json.dumps({
        "ok": True,
        "tool": "scientist_terminal_turn",
        "selected_task": "spaceship-titanic",
        "requirement_ledger": {
            "schema": "evomind.ai_scientist.requirement_ledger.v1",
            "requirements": [
                {
                    "id": "setup_gate_clearance",
                    "description": "Setup gates must be clear before execution-like work can start.",
                    "status": "blocked",
                    "gate": "setup_gate",
                    "reason": "External GPU allocation is not available yet.",
                    "evidence_needed": ["evomind ready", ".xsci/scientist_repair_plan.json"],
                    "mapped_tools": ["system_status", "scientist_repair_plan"],
                },
                {
                    "id": "agent_self_audit",
                    "description": "Agent parity requires a capability audit before claims.",
                    "status": "pending",
                    "gate": "capability_audit_gate",
                    "reason": "No self-audit artifact yet.",
                    "evidence_needed": [".xsci/scientist_self_audit.json"],
                    "mapped_tools": ["scientist_self_audit"],
                },
            ],
            "open_requirements": ["setup_gate_clearance", "agent_self_audit"],
            "blocked_requirements": ["setup_gate_clearance"],
        },
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    (xsci / "scientist_next_action.json").write_text(json.dumps({
        "ok": True,
        "tool": "scientist_next_action",
        "status": "executed_read_only_tool",
        "selected_task": "spaceship-titanic",
        "selected_action": {
            "id": "clear_blockers",
            "status": "ready",
            "command": "evomind repair",
            "metadata": {
                "source": "requirement_ledger",
                "requirement_id": "setup_gate_clearance",
                "requirement_status": "blocked",
            },
        },
        "executed_tool": "scientist_repair_plan",
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_next_action", state, root)
    refreshed_queue = json.loads(queue_path.read_text(encoding="utf-8"))
    action_ids = [item["id"] for item in refreshed_queue["actions"][:2]]
    updated_turn = json.loads((xsci / "scientist_terminal_turn.json").read_text(encoding="utf-8"))
    progress = json.loads((xsci / "scientist_requirement_progress.json").read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert result["status"] == "executed_read_only_tool"
    assert result["selected_action"]["id"] == "satisfy_agent_self_audit"
    assert result["selected_action"]["metadata"]["requirement_id"] == "agent_self_audit"
    assert result["executed_tool"] == "scientist_self_audit"
    assert action_ids[0] == "clear_blockers"
    assert "satisfy_agent_self_audit" not in action_ids
    assert updated_turn["requirement_ledger"]["blocked_requirements"] == ["setup_gate_clearance"]
    assert "agent_self_audit" not in updated_turn["requirement_ledger"]["open_requirements"]
    assert progress["requirement_id"] == "agent_self_audit"
    assert progress["after_status"] == "satisfied"
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"


def test_scientist_next_action_stops_at_gate_when_only_attempted_external_blocker_remains(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    xsci = root / ".xsci"
    queue_path = xsci / "scientist_action_queue.json"
    queue_path.write_text(json.dumps({
        "ok": True,
        "tool": "scientist_action_queue",
        "selected_task": "spaceship-titanic",
        "actions": [
            {
                "id": "run_gated_candidate",
                "title": "Run candidate",
                "status": "ready",
                "command": "evomind run spaceship-titanic",
                "gate": "human_run_command_required",
                "why": "Only the user/workstation run gate remains after diagnostics.",
                "risk": "would require training gate",
                "rollback_condition": "hold",
                "expected_artifacts": ["metrics.json"],
                "evidence": ["metrics.json"],
                "autonomy": "requires_user_run_command",
            }
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    (xsci / "scientist_terminal_turn.json").write_text(json.dumps({
        "ok": True,
        "tool": "scientist_terminal_turn",
        "selected_task": "spaceship-titanic",
        "requirement_ledger": {
            "schema": "evomind.ai_scientist.requirement_ledger.v1",
            "requirements": [
                {
                    "id": "setup_gate_clearance",
                    "description": "Setup gates must be clear before execution-like work can start.",
                    "status": "blocked",
                    "gate": "setup_gate",
                    "reason": "External GPU allocation is still unavailable.",
                    "evidence_needed": ["evomind ready", ".xsci/scientist_repair_plan.json"],
                    "mapped_tools": ["system_status", "scientist_repair_plan"],
                }
            ],
            "open_requirements": ["setup_gate_clearance"],
            "blocked_requirements": ["setup_gate_clearance"],
        },
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    (xsci / "scientist_next_action.json").write_text(json.dumps({
        "ok": True,
        "tool": "scientist_next_action",
        "status": "executed_read_only_tool",
        "selected_task": "spaceship-titanic",
        "selected_action": {
            "id": "clear_blockers",
            "status": "ready",
            "command": "evomind repair",
            "metadata": {
                "source": "requirement_ledger",
                "requirement_id": "setup_gate_clearance",
                "requirement_status": "blocked",
            },
        },
        "executed_tool": "scientist_repair_plan",
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_next_action", state, root)
    refreshed_queue = json.loads(queue_path.read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert result["status"] == "blocked_by_gate"
    assert result["selected_action"]["id"] == "run_gated_candidate"
    assert result["executed_tool"] is None
    assert refreshed_queue["actions"][0]["id"] == "clear_blockers"
    assert refreshed_queue["actions"][1]["id"] == "run_gated_candidate"
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"


def test_scientist_situation_model_intent_routes_before_training():
    intent = ki.classify("analyze the current situation and why are we blocked")

    assert intent.kind == ki.TOOL_QUERY
    assert intent.payload == "scientist_situation_model"


def test_scientist_situation_model_synthesizes_context_without_training(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    memory_dir = root / "experiments" / "evolution"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "retrospective_memory.json").write_text(json.dumps({
        "records": [
            {
                "memory_id": "mem_001",
                "task": "spaceship_titanic",
                "reusable_strategy": "OOF blend with fold stability audit",
                "what_worked": "probability calibration reduced leaderboard gap risk",
            }
        ]
    }), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_situation_model", state, root)
    artifact = root / ".xsci" / "scientist_situation_model.json"
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["ok"] is True
    assert result["tool"] == "scientist_situation_model"
    assert result["selected_task"] == "spaceship-titanic"
    assert result["situation_model"]["reasoning_mode"] == "observe_orient_decide_act_with_gates"
    assert isinstance(result["readiness_score"], int)
    assert result["next_safe_commands"]
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert artifact.exists()
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_situation_model_console_query_does_not_train(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "situation model current research state", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "scientist_situation_model" in out
    assert "readiness_score" in out
    assert "no_training_started: True" in out


def test_scientist_next_action_executes_situation_model_read_only(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    queue_path = root / ".xsci" / "scientist_action_queue.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps({
        "ok": True,
        "tool": "scientist_action_queue",
        "selected_task": "spaceship-titanic",
        "actions": [
            {
                "id": "refresh_situation_model",
                "title": "Refresh Scientist Situation Model",
                "status": "ready",
                "command": "evomind situation",
                "gate": "read_only_situation_gate",
                "why": "synthesize evidence and blockers before any training action",
                "risk": "none",
                "rollback_condition": "read-only artifact only",
                "expected_artifacts": [".xsci/scientist_situation_model.json"],
                "evidence": ["scientist_situation_model"],
                "autonomy": "read_only",
            }
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_next_action", state, root)

    assert result["ok"] is True
    assert result["status"] == "executed_read_only_tool"
    assert result["selected_action"]["id"] == "refresh_situation_model"
    assert result["executed_tool"] == "scientist_situation_model"
    assert result["tool_result"]["tool"] == "scientist_situation_model"
    assert (root / ".xsci" / "scientist_situation_model.json").exists()
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"


def test_scientist_turn_plan_builds_per_turn_tool_plan_without_training(isolated_autokaggle):
    from xsci.scientist_turn_planner import build_scientist_turn_plan

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)

    result = build_scientist_turn_plan(
        state,
        root,
        "start the second self-evolution training round after checking gates",
        persist=True,
        record_turn=True,
    )
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["ok"] is True
    assert result["tool"] == "scientist_turn_plan"
    assert result["intent"]["kind"] == ki.EXECUTION
    assert result["autonomy_level"] in {"gated_executor_pending", "repair_first"}
    assert "scientist_execution_contract" in result["tool_sequence"]
    ledger = result["requirement_ledger"]
    requirement_ids = {item["id"] for item in ledger["requirements"]}
    assert ledger["schema"] == "evomind.ai_scientist.requirement_ledger.v1"
    assert "execution_contract" in requirement_ids
    assert "data_and_validation_contract" in requirement_ids
    assert "no_unapproved_training_or_submit" in ledger["satisfied_requirements"]
    assert "secret_safety" in ledger["satisfied_requirements"]
    evidence_values = [
        value
        for item in ledger["requirements"]
        for value in item.get("evidence_needed", [])
    ]
    assert ".xsci/scientist_execution_contract.json" in evidence_values
    assert "[truncated]" not in evidence_values
    assert result["response_contract"]["must_report_open_requirements"] is True
    lifecycle = result["parity_lifecycle"]
    phases = [item["phase"] for item in lifecycle["phases"]]
    assert lifecycle["schema"] == "evomind.ai_scientist.parity_lifecycle.v1"
    assert phases == ["observe", "plan", "act", "reflect", "improve"]
    assert "open_requirements" in lifecycle["phases"][0]["evidence"]
    assert lifecycle["phases"][1]["requirement_count"] == len(ledger["requirements"])
    assert lifecycle["completion_gate"]["must_record_artifact"] == ".xsci/scientist_parity_loop.jsonl"
    assert lifecycle["completion_gate"]["must_preserve_no_training"] is True
    assert lifecycle["completion_gate"]["must_preserve_submit_block"] is True
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert (root / ".xsci" / "scientist_turn_plan.json").exists()
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_turn_plan_uses_scientific_critique_for_agent_capability_goal(isolated_autokaggle):
    from xsci.scientist_turn_planner import build_scientist_turn_plan

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)

    result = build_scientist_turn_plan(
        state,
        root,
        "我的 EvoMind agent 还是不够智能，要像 Claude Code 一样解决复杂科研问题",
        persist=True,
    )

    critique = result["scientific_critique"]
    tools = result["tool_sequence"]
    gap_names = [gap["gap"] for gap in critique["evidence_gaps"]]

    assert critique["decision"] == "self_audit_then_consolidate_memory"
    assert critique["actionability_score"] < 100
    assert "agent_capability_audit_needed" in gap_names
    assert "memory_consolidation_needed" in gap_names
    ledger = result["requirement_ledger"]
    requirement_ids = {item["id"] for item in ledger["requirements"]}
    assert "agent_self_audit" in requirement_ids
    assert "memory_consolidation" in requirement_ids
    assert "parity_lifecycle" in requirement_ids
    assert tools.index("scientist_self_audit") < tools.index("scientist_autopilot")
    assert tools.index("scientist_memory_consolidation") < tools.index("scientist_autopilot")
    assert any("rank" in item or "medal" in item for item in critique["claim_boundaries"])
    assert result["no_training_started"] is True


def test_conversation_reply_records_turn_plan_for_generic_scientist_question(isolated_autokaggle):
    from xsci.kaggle_conversation import ConversationAgent
    from xsci.scientist_turns import load_recent_scientist_turns

    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)
    agent = ConversationAgent(client=None)

    agent.reply("Before answering, plan your tools and explain the next safe step.", state)
    artifact = root / ".xsci" / "scientist_turn_plan.json"
    turns = load_recent_scientist_turns(root)
    latest = turns[-1]

    assert artifact.exists()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["tool"] == "scientist_turn_plan"
    assert payload["selected_tools"]
    assert latest["no_training_started"] is True
    assert "scientist_turn_plan" in latest["forced_tools"]
    assert str(artifact) in latest["artifacts"]


def test_generic_chat_runs_observable_scientist_turn_without_training(isolated_autokaggle, monkeypatch, capsys):
    from xsci import agent as xagent
    from xsci.scientist_trace import load_recent_scientist_step_events
    from xsci.scientist_turns import load_recent_scientist_parity_loops, load_recent_scientist_turns

    def _tripwire(*args, **kwargs):
        raise AssertionError("training must not start from a generic scientist turn")

    monkeypatch.setattr(xagent, "run_agent", _tripwire)
    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)

    rc, selected, should_exit = ak._handle_console_command(
        "请给我一个稳妥的研究判断",
        root,
        "spaceship-titanic",
    )
    out = capsys.readouterr().out
    turns = load_recent_scientist_turns(root)
    parity_loops = load_recent_scientist_parity_loops(root)
    steps = load_recent_scientist_step_events(root, limit=80)
    terminal_payload = json.loads((root / ".xsci" / "scientist_terminal_turn.json").read_text(encoding="utf-8"))
    serialized = json.dumps(turns + steps, ensure_ascii=False)

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert "AI Scientist Turn" in out
    assert "tools_executed" in out
    assert "no_training_started: True" in out
    assert (root / ".xsci" / "scientist_turn_plan.json").exists()
    assert (root / ".xsci" / "scientist_terminal_turn.json").exists()
    assert (root / ".xsci" / "scientist_parity_loop.jsonl").exists()
    assert (root / ".xsci" / "scientist_latest_parity_loop.json").exists()
    assert turns[-1]["route"] == "scientist_terminal_turn"
    assert turns[-1]["no_training_started"] is True
    assert turns[-1]["parity_lifecycle"]["schema"] == "evomind.ai_scientist.parity_lifecycle.v1"
    assert terminal_payload["parity_lifecycle"]["phase_status"]["observe"] == "passed"
    assert terminal_payload["parity_lifecycle"]["phase_status"]["plan"] == "passed"
    assert {item["phase"] for item in terminal_payload["parity_lifecycle"]["phases"]} == {
        "observe", "plan", "act", "reflect", "improve"
    }
    assert parity_loops[-1]["phase_status"]["observe"] == "passed"
    assert parity_loops[-1]["official_submit"] == "blocked_until_explicit_human_approval"
    assert any(step.get("phase") == "terminal_turn_complete" for step in steps)
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized
    assert "sk-TEST-SHOULD-NOT-LEAK" not in json.dumps([terminal_payload, parity_loops], ensure_ascii=False)


def test_ask_command_runs_one_scientist_turn_without_training(isolated_autokaggle, monkeypatch, capsys):
    from xsci import agent as xagent
    from xsci.scientist_turns import load_recent_scientist_turns

    def _tripwire(*args, **kwargs):
        raise AssertionError("training must not start from evomind ask")

    monkeypatch.setattr(xagent, "run_agent", _tripwire)
    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)

    rc = ak.main(["ask", "分析当前任务，给出下一步安全科研判断"])
    out = capsys.readouterr().out
    turns = load_recent_scientist_turns(root)
    serialized = json.dumps(turns, ensure_ascii=False)

    assert rc == 0
    assert "AI Scientist Turn" in out
    assert "tools_executed" in out
    assert "no_training_started: True" in out
    assert (root / ".xsci" / "scientist_terminal_turn.json").exists()
    assert turns[-1]["route"] == "scientist_terminal_turn"
    assert turns[-1]["no_training_started"] is True
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_ask_command_json_returns_machine_readable_result(isolated_autokaggle, monkeypatch, capsys):
    from xsci import agent as xagent

    def _tripwire(*args, **kwargs):
        raise AssertionError("training must not start from evomind ask --json")

    monkeypatch.setattr(xagent, "run_agent", _tripwire)
    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)

    rc = ak.main(["ask", "--json", "--max-tools", "2", "检查证据并给出下一步"])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload["ok"] is True
    assert payload["action"] == "scientist_turn"
    assert payload["selected_task"] == "spaceship-titanic"
    assert "execution_ready" in payload
    assert "execution_blocked" in payload
    assert isinstance(payload["blocking_gates"], list)
    assert payload["scientific_critique"]["evidence_gaps"]
    assert payload["scientific_critique"]["claim_boundaries"]
    assert payload["requirement_ledger"]["schema"] == "evomind.ai_scientist.requirement_ledger.v1"
    assert payload["requirement_ledger"]["resolution"]["mode"] == "post_tool_execution"
    assert "no_unapproved_training_or_submit" in payload["requirement_ledger"]["satisfied_requirements"]
    assert "context_packet" in payload["requirement_ledger"]["satisfied_requirements"]
    assert payload["parity_lifecycle"]["schema"] == "evomind.ai_scientist.parity_lifecycle.v1"
    assert payload["parity_loop_artifact"].endswith(".xsci\\scientist_parity_loop.jsonl") or payload["parity_loop_artifact"].endswith(".xsci/scientist_parity_loop.jsonl")
    assert [item["phase"] for item in payload["parity_lifecycle"]["phases"]] == ["observe", "plan", "act", "reflect", "improve"]
    assert payload["scientist_reasoning_synthesis"]["reasoning_quality"]["score"] >= 70
    assert payload["answer_markdown"]
    assert payload["tool_budget"]["reasoning_synthesis_auto_executed"] is True
    assert payload["no_training_started"] is True
    assert payload["official_submit"] == "blocked_until_explicit_human_approval"
    assert payload["artifacts"]
    assert any(str(path).endswith("scientist_context_packet.json") for path in payload["artifacts"])
    assert "sk-TEST-SHOULD-NOT-LEAK" not in out


def test_ask_json_expands_default_budget_for_meta_scientist_turn(isolated_autokaggle, monkeypatch, capsys):
    from xsci import agent as xagent
    from xsci.scientist_turns import load_recent_scientist_turns

    def _tripwire(*args, **kwargs):
        raise AssertionError("meta scientist audit must not start training")

    monkeypatch.setattr(xagent, "run_agent", _tripwire)
    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)

    rc = ak.main([
        "ask",
        "--json",
        "我的 EvoMind agent 还是不够智能，要像 Claude Code 一样解决复杂科研问题",
    ])
    out = capsys.readouterr().out
    payload = json.loads(out)
    turn_artifact = root / ".xsci" / "scientist_terminal_turn.json"
    turn_payload = json.loads(turn_artifact.read_text(encoding="utf-8"))
    continuation_payload = json.loads((root / ".xsci" / "scientist_continuation.json").read_text(encoding="utf-8"))
    turns = load_recent_scientist_turns(root)
    executed = {
        item.get("tool")
        for item in turn_payload.get("executed_tools", [])
        if item.get("tool") != "scientist_turn_plan"
    }
    must_run = set(turn_payload["tool_budget"]["must_run_tools"])

    assert rc == 0
    assert payload["ok"] is True
    assert payload["tool_budget"]["recommended_min_tools"] > 4
    assert payload["tool_budget"]["effective_max_tools"] >= payload["tool_budget"]["recommended_min_tools"]
    assert payload["budget_exhausted"] is False
    assert payload["continuation"]["status"] == "closed"
    assert continuation_payload["status"] == "closed"
    assert continuation_payload["remaining_safe_tools"] == []
    assert payload["parity_lifecycle"]["phase_status"]["improve"] == "passed"
    assert "scientist_context_packet" in executed
    assert "scientist_self_audit" in executed
    assert "scientist_memory_consolidation" in executed
    assert "scientist_autopilot" in executed
    assert "scientist_workplan" in executed
    assert payload["requirement_ledger"]["open_requirements"]
    requirement_by_id = {
        item["id"]: item
        for item in payload["requirement_ledger"]["requirements"]
    }
    assert requirement_by_id["agent_self_audit"]["status"] == "satisfied"
    assert requirement_by_id["context_packet"]["status"] == "satisfied"
    assert requirement_by_id["memory_consolidation"]["status"] == "satisfied"
    assert requirement_by_id["recoverable_workplan"]["status"] == "satisfied"
    assert requirement_by_id["parity_lifecycle"]["status"] == "satisfied"
    assert "scientist_memory_consolidation" in requirement_by_id["memory_consolidation"]["execution_evidence"]["mapped_tool_hits"]
    assert payload["requirement_ledger"]["resolution"]["mode"] == "post_tool_execution"
    assert "memory_consolidation" in {
        item["id"] for item in requirement_by_id.values()
    }
    assert must_run.issubset(executed)
    assert not (must_run & set(payload["deferred_tools"]))
    assert turns[-1]["decision"]["tool_budget"]["effective_max_tools"] >= turns[-1]["decision"]["tool_budget"]["recommended_min_tools"]
    assert turns[-1]["no_training_started"] is True
    assert turns[-1]["official_submit"] == "blocked_until_explicit_human_approval"
    assert "sk-TEST-SHOULD-NOT-LEAK" not in json.dumps([payload, turn_payload, turns[-1]], ensure_ascii=False)


def test_scientist_continuation_status_reports_no_artifact(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_continuation_status", state, root)

    assert result["ok"] is True
    assert result["status"] == "no_continuation"
    assert result["completion_ratio"] == 1.0
    assert result["remaining_safe_tools"] == []
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert (root / ".xsci" / "scientist_continuation_status.json").exists()


def test_scientist_continuation_status_reports_remaining_progress(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xsci_dir = root / ".xsci"
    xsci_dir.mkdir(parents=True, exist_ok=True)
    (xsci_dir / "scientist_continuation.json").write_text(json.dumps({
        "schema": "evomind.ai_scientist.continuation.v1",
        "tool": "scientist_continuation",
        "status": "needs_more_tools",
        "must_run_deferred_tools": ["scientist_self_audit", "scientist_memory_consolidation"],
        "remaining_safe_tools": ["scientist_memory_consolidation"],
        "safe_next_command": "evomind ask --json --max-tools 4 \"continue\"",
        "progress_history": [{
            "schema": "evomind.ai_scientist.continuation_progress.v1",
            "updated_at": "2026-07-09T00:00:00+00:00",
            "safe_tool": "scientist_self_audit",
            "tool_ok": True,
            "tool_artifact_path": str(xsci_dir / "scientist_self_audit.json"),
            "before_remaining_safe_tools": ["scientist_self_audit", "scientist_memory_consolidation"],
            "after_remaining_safe_tools": ["scientist_memory_consolidation"],
            "status": "needs_more_tools",
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        }],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_continuation_status", state, root)

    assert result["ok"] is True
    assert result["status"] == "needs_more_tools"
    assert result["completion_ratio"] == 0.5
    assert result["completed_required_tools"] == 1
    assert result["remaining_safe_tools"] == ["scientist_memory_consolidation"]
    assert result["executed_or_completed_tools"] == ["scientist_self_audit"]
    assert result["next_safe_action_command"] == "evomind memory-consolidation"
    assert result["progress_history"][0]["safe_tool"] == "scientist_self_audit"
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"


def test_scientist_continuation_resume_reports_no_artifact(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_continuation_resume", state, root)

    assert result["ok"] is True
    assert result["status"] == "no_continuation"
    assert result["steps_executed"] == 0
    assert result["remaining_safe_tools"] == []
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert (root / ".xsci" / "scientist_continuation_resume.json").exists()


def test_scientist_continuation_resume_closes_remaining_safe_tools(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xsci_dir = root / ".xsci"
    xsci_dir.mkdir(parents=True, exist_ok=True)
    continuation_path = xsci_dir / "scientist_continuation.json"
    continuation_path.write_text(json.dumps({
        "schema": "evomind.ai_scientist.continuation.v1",
        "tool": "scientist_continuation",
        "status": "needs_more_tools",
        "must_run_deferred_tools": ["scientist_self_audit", "scientist_memory_consolidation"],
        "remaining_safe_tools": ["scientist_self_audit", "scientist_memory_consolidation"],
        "action_queue_hint": [
            {"safe_tool": "scientist_self_audit", "command": "evomind self-audit"},
            {"safe_tool": "scientist_memory_consolidation", "command": "evomind memory-consolidation"},
        ],
        "safe_next_command": "evomind resume-continuation",
        "progress_history": [],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_continuation_resume", state, root)
    updated = json.loads(continuation_path.read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert result["status"] == "closed"
    assert result["stop_reason"] == "closed"
    assert result["steps_executed"] == 2
    assert result["executed_tools"] == ["scientist_self_audit", "scientist_memory_consolidation"]
    assert result["remaining_safe_tools"] == []
    assert updated["status"] == "closed"
    assert updated["remaining_safe_tools"] == []
    assert len(updated["progress_history"]) == 2
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert (root / ".xsci" / "scientist_continuation_resume.json").exists()


def test_scientist_continuation_resume_observer_streams_steps(isolated_autokaggle):
    from xsci.terminal_tools import run_scientist_continuation_resume

    root = xcfg.active_root()
    xsci_dir = root / ".xsci"
    xsci_dir.mkdir(parents=True, exist_ok=True)
    (xsci_dir / "scientist_continuation.json").write_text(json.dumps({
        "schema": "evomind.ai_scientist.continuation.v1",
        "tool": "scientist_continuation",
        "status": "needs_more_tools",
        "must_run_deferred_tools": ["scientist_self_audit", "scientist_memory_consolidation"],
        "remaining_safe_tools": ["scientist_self_audit", "scientist_memory_consolidation"],
        "action_queue_hint": [
            {"safe_tool": "scientist_self_audit", "command": "evomind self-audit"},
            {"safe_tool": "scientist_memory_consolidation", "command": "evomind memory-consolidation"},
        ],
        "progress_history": [],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    state = SessionState.from_root(root)
    events = []

    result = run_scientist_continuation_resume(state, root, observer=events.append)
    phases = [event.get("phase") for event in events]

    assert result["status"] == "closed"
    assert phases[0] == "continuation_resume_start"
    assert phases.count("continuation_resume_step_started") == 2
    assert phases.count("continuation_resume_step_completed") == 2
    assert phases[-1] == "continuation_resume_complete"
    assert all(event.get("no_training_started") is True for event in events)
    assert all(event.get("official_submit") == "blocked_until_explicit_human_approval" for event in events)


def test_memory_consolidation_learns_continuation_resume(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xsci_dir = root / ".xsci"
    xsci_dir.mkdir(parents=True, exist_ok=True)
    (xsci_dir / "scientist_continuation_resume.json").write_text(json.dumps({
        "ok": True,
        "tool": "scientist_continuation_resume",
        "generated_at": "2026-07-10T00:00:00+00:00",
        "selected_task": "demo-task",
        "status": "closed",
        "stop_reason": "closed",
        "steps_executed": 2,
        "steps": [
            {"index": 1, "status": "executed_read_only_tool", "executed_tool": "scientist_self_audit", "after_remaining_safe_tools": ["scientist_memory_consolidation"]},
            {"index": 2, "status": "executed_read_only_tool", "executed_tool": "scientist_memory_consolidation", "after_remaining_safe_tools": []},
        ],
        "remaining_safe_tools": [],
        "executed_tools": ["scientist_self_audit", "scientist_memory_consolidation"],
        "artifact_path": str(xsci_dir / "scientist_continuation_resume.json"),
        "continuation_status_artifact_path": str(xsci_dir / "scientist_continuation_status.json"),
        "continuation_artifact_path": str(xsci_dir / "scientist_continuation.json"),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")
    state = SessionState.from_root(root)

    result = TerminalTools.dispatch("scientist_memory_consolidation", state, root)
    memory_payload = json.loads((root / "experiments" / "evolution" / "retrospective_memory.json").read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert result["source_counts"]["continuation_resume_present"] is True
    assert result["records_added"] >= 1
    assert any(item.get("method") == "scientist_continuation_resume" for item in memory_payload)
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"


def test_ask_json_respects_explicit_small_budget_and_records_deferred_tools(isolated_autokaggle, monkeypatch, capsys):
    from xsci import agent as xagent
    from xsci.terminal_tools import TerminalTools

    def _tripwire(*args, **kwargs):
        raise AssertionError("small-budget scientist ask must not start training")

    monkeypatch.setattr(xagent, "run_agent", _tripwire)
    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)

    rc = ak.main([
        "ask",
        "--json",
        "--max-tools",
        "2",
        "我的 EvoMind agent 还是不够智能，要像 Claude Code 一样解决复杂科研问题",
    ])
    out = capsys.readouterr().out
    payload = json.loads(out)
    turn_payload = json.loads((root / ".xsci" / "scientist_terminal_turn.json").read_text(encoding="utf-8"))
    continuation_path = root / ".xsci" / "scientist_continuation.json"
    continuation_payload = json.loads(continuation_path.read_text(encoding="utf-8"))
    executed = [
        item.get("tool")
        for item in turn_payload.get("executed_tools", [])
        if item.get("tool") != "scientist_turn_plan"
    ]

    assert rc == 0
    assert payload["ok"] is True
    assert payload["tool_budget"]["requested_max_tools"] == 2
    assert payload["tool_budget"]["effective_max_tools"] == 2
    assert payload["tool_budget"]["recommended_min_tools"] > 2
    assert payload["budget_exhausted"] is True
    assert payload["continuation"]["status"] == "needs_more_tools"
    assert payload["continuation"]["explicit_user_budget_cap"] is True
    assert continuation_payload["status"] == "needs_more_tools"
    assert continuation_payload["must_run_deferred_tools"]
    assert continuation_payload["remaining_safe_tools"] == continuation_payload["must_run_deferred_tools"]
    assert continuation_payload["safe_next_command"].startswith("evomind ask --json --max-tools")
    assert payload["parity_lifecycle"]["phase_status"]["improve"] == "needs_more_tools"
    assert payload["parity_lifecycle"]["budget_exhausted"] is True
    assert payload["deferred_tools"]
    assert "scientist_context_packet" in executed
    budgeted_executed = [
        tool for tool in executed
        if tool not in {"scientist_context_packet", "scientist_reasoning_synthesis"}
    ]
    assert len(budgeted_executed) == 2
    assert payload["tool_budget"]["context_packet_auto_executed"] is True
    assert payload["tool_budget"]["reasoning_synthesis_auto_executed"] is True
    assert "scientist_autopilot" in payload["deferred_tools"]
    small_budget_requirements = {
        item["id"]: item
        for item in payload["requirement_ledger"]["requirements"]
    }
    assert small_budget_requirements["memory_consolidation"]["status"] != "satisfied"
    assert "memory_consolidation" in payload["requirement_ledger"]["open_requirements"]
    assert payload["requirement_ledger"]["resolution"]["mode"] == "post_tool_execution"
    assert payload["no_training_started"] is True
    assert payload["official_submit"] == "blocked_until_explicit_human_approval"
    assert "sk-TEST-SHOULD-NOT-LEAK" not in out

    state = SessionState.from_root(root)
    queue = TerminalTools.dispatch("scientist_action_queue", state, root)
    first = queue["actions"][0]
    assert first["metadata"]["source"] == "scientist_continuation"
    assert first["metadata"]["safe_tool"] in continuation_payload["remaining_safe_tools"]

    next_action = TerminalTools.dispatch("scientist_next_action", state, root)
    updated_continuation = json.loads(continuation_path.read_text(encoding="utf-8"))

    assert next_action["status"] == "executed_read_only_tool"
    assert next_action["selected_action"]["metadata"]["source"] == "scientist_continuation"
    assert next_action["continuation_progress"]["safe_tool"] == next_action["executed_tool"]
    assert next_action["executed_tool"] not in updated_continuation["remaining_safe_tools"]
    assert updated_continuation["last_progress"]["tool_ok"] is True
    assert updated_continuation["no_training_started"] is True
    assert updated_continuation["official_submit"] == "blocked_until_explicit_human_approval"
    status = TerminalTools.dispatch("scientist_continuation_status", state, root)
    assert status["status"] in {"needs_more_tools", "closed"}
    assert next_action["executed_tool"] in status["executed_or_completed_tools"]
    assert status["no_training_started"] is True
    assert status["official_submit"] == "blocked_until_explicit_human_approval"


def test_scientist_recovery_snapshot_rebuilds_resume_context(isolated_autokaggle, capsys):
    from xsci.scientist_trace import load_recent_scientist_step_events
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)

    TerminalTools.dispatch("scientist_loop", state, root)
    result = TerminalTools.dispatch("scientist_recovery", state, root)
    rc, selected, should_exit = ak._handle_console_command("恢复现场看看从哪里继续", root, "spaceship-titanic")
    out = capsys.readouterr().out
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["ok"] is True
    assert result["tool"] == "scientist_recovery"
    assert result["selected_task"] == "spaceship-titanic"
    assert result["recovery_decision"] in {"blocked_clear_gates", "resume_from_selected_action", "refresh_scientist_loop"}
    assert result["recent_step_count"] > 0
    assert result["recent_turn_count"] > 0
    assert result["selected_resume_action"]
    assert "evomind recovery" in result["resume_commands"]
    assert "evomind next" in result["resume_commands"]
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert (root / ".xsci" / "scientist_recovery_snapshot.json").exists()
    assert (root / ".xsci" / "recovery_guard.md").exists()
    phases = [event.get("phase") for event in load_recent_scientist_step_events(root, limit=160)]
    assert "recovery_snapshot" in phases
    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert "scientist_recovery" in out
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_conversation_autopilot_query_records_scientist_artifact(isolated_autokaggle):
    from xsci.kaggle_conversation import ConversationAgent, _terminal_tool_specs
    from xsci.scientist_turns import load_recent_scientist_turns

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    state = SessionState.from_root(root)
    agent = ConversationAgent(client=None)

    out = agent.reply("全面诊断当前系统为什么不够智能", state)
    tool_names = [spec.name for spec in _terminal_tool_specs()]
    turns = load_recent_scientist_turns(root)
    serialized_turns = json.dumps(turns, ensure_ascii=False)

    assert "scientist_autopilot" in tool_names
    assert "scientist_recovery" in tool_names
    assert "Scientist Autopilot" in out
    assert "No training or official Kaggle submission was started." in out
    assert (root / ".xsci" / "scientist_autopilot.json").exists()
    assert (root / ".xsci" / "scientist_turns.jsonl").exists()
    assert turns
    assert any("scientist_autopilot" in turn.get("forced_tools", []) for turn in turns)
    assert turns[-1]["no_training_started"] is True
    assert turns[-1]["official_submit"] == "blocked_until_explicit_human_approval"
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized_turns


def test_research_decision_persists_baseline_artifact_without_secret(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    data_dir = root / "data" / "spaceship-titanic"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("PassengerId,Transported\n1,True\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("PassengerId\n2\n", encoding="utf-8")

    task_path = root / ".xsci" / "tasks" / "spaceship-titanic.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["local_data_dir"] = str(data_dir)
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")

    unrelated = root / "experiments" / "evolution" / "other-task_local_20260708_000000"
    unrelated.mkdir(parents=True)
    (unrelated / "summary.json").write_text(
        json.dumps({
            "task": "other-task",
            "best_exp_id": "",
            "best_cv_score": None,
            "n_promotions": 0,
            "n_iterations": 3,
        }),
        encoding="utf-8",
    )

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("research_decision", state, root)
    serialized = json.dumps(result, ensure_ascii=False)
    artifact = root / ".xsci" / "scientist_decision.json"

    assert result["ok"] is True
    assert result["tool"] == "research_decision"
    assert result["decision"]["selected_action"] == "run_audited_baseline"
    assert result["decision"]["selected_branch"] == "baseline"
    assert result["decision"]["code_generation_mode"] == "Base"
    assert result["decision"]["latest_run_signal"]["signal"] == "no_prior_run"
    assert artifact.exists()
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized
    assert "blocked_until_explicit_user_approval" in serialized


def test_research_decision_reuses_retrospective_memory(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    data_dir = root / "data" / "spaceship-titanic"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("PassengerId,Transported\n1,True\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("PassengerId\n2\n", encoding="utf-8")
    task_path = root / ".xsci" / "tasks" / "spaceship-titanic.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["local_data_dir"] = str(data_dir)
    task["task_type"] = "classification"
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")

    memory_path = root / "experiments" / "evolution" / "retrospective_memory.json"
    memory_path.parent.mkdir(parents=True)
    memory_path.write_text(json.dumps([{
        "memory_id": "prior:EXP001",
        "task_type": "classification",
        "dataset_profile": {},
        "method": "Stepwise",
        "what_worked": "calibrated GBDT with stable folds",
        "what_failed": "",
        "metric_delta": 0.012,
        "reusable_strategy": "calibrated_gbdt,stable_stratified_folds",
        "failure_pattern": "",
        "linked_exp_ids": ["EXP001"],
    }], ensure_ascii=False), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("research_decision", state, root)

    assert result["decision"]["selected_action"] == "run_memory_guided_baseline"
    assert result["decision"]["selected_branch"] == "baseline_with_retrospective_transfer"
    assert result["research_brief"]["memory_reuse_records"]
    assert result["research_brief"]["hypotheses"]
    assert result["research_brief"]["experiment_plan"]


def test_research_decision_reuses_scientist_turn_memory_without_secret(isolated_autokaggle):
    from xsci.scientist_turns import record_scientist_turn
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    data_dir = root / "data" / "spaceship-titanic"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("PassengerId,Transported\n1,True\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("PassengerId\n2\n", encoding="utf-8")
    task_path = root / ".xsci" / "tasks" / "spaceship-titanic.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["local_data_dir"] = str(data_dir)
    task["task_type"] = "classification"
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")

    record_scientist_turn(root, {
        "task": "spaceship-titanic",
        "route": "scientist_autopilot",
        "user": "全面诊断下一轮实验 api_key=LEAKME",
        "executed_tools": [{"tool": "data_check"}, {"tool": "research_decision"}],
        "mode": "needs_attention",
        "decision": {
            "selected_action": "prepare_data_or_schema",
            "selected_branch": "data_readiness",
            "code_generation_mode": "none",
        },
        "blockers": ["schema audit missing; password=LEAKME"],
        "next_actions": ["Run schema audit before model search."],
        "answer_preview": "cookie=LEAKME should be redacted",
        "no_training_started": True,
    })

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("research_decision", state, root)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["decision"]["selected_action"] == "run_turn_memory_guided_baseline"
    assert result["decision"]["selected_branch"] == "baseline_with_scientist_turn_reuse"
    assert result["research_brief"]["scientist_turn_reuse_records"]
    assert result["research_brief"]["turn_derived_failure_avoidance"]
    assert "Run schema audit before model search" in serialized
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized
    assert "LEAKME" not in serialized


def test_research_decision_prioritizes_self_audit_upgrade_backlog(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    data_dir = root / "data" / "spaceship-titanic"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("PassengerId,Transported\n1,True\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("PassengerId\n2\n", encoding="utf-8")
    task_path = root / ".xsci" / "tasks" / "spaceship-titanic.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["local_data_dir"] = str(data_dir)
    task["task_type"] = "classification"
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    xsci = root / ".xsci"
    xsci.mkdir(parents=True, exist_ok=True)
    (xsci / "scientist_upgrade_backlog.json").write_text(json.dumps({
        "generated_at": "2026-07-09T00:00:00+00:00",
        "tool": "scientist_upgrade_backlog",
        "source": "scientist_self_audit",
        "overall_score": 61,
        "items": [
            {
                "id": "planner_executor_observer_loop",
                "title": "Close planner/executor gap token=LEAKME",
                "priority": "P0",
                "status": "proposed",
                "why": "Agent must finish self-upgrade before training api_key=LEAKME",
                "safe_next_command": "evomind loop",
                "expected_artifacts": [".xsci/scientist_loop.json"],
                "gate": "engineering_review_required",
            }
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }, ensure_ascii=False), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("research_decision", state, root)
    serialized = json.dumps(result, ensure_ascii=False)
    brief = result["research_brief"]

    assert result["decision"]["selected_action"] == "close_agent_upgrade_backlog"
    assert result["decision"]["selected_branch"] == "scientist_capability_upgrade"
    assert result["decision"]["code_generation_mode"] == "none"
    assert brief["agent_capability_gate"]["status"] == "upgrade_required_before_training"
    assert brief["agent_capability_gate"]["p0_count"] == 1
    assert brief["agent_upgrade_backlog"][0]["id"] == "planner_executor_observer_loop"
    assert "planner_executor_observer_loop" in serialized
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized
    assert "LEAKME" not in serialized


def test_research_decision_closes_scientist_critique_budget_before_training(isolated_autokaggle):
    from xsci.scientist_turns import record_scientist_turn
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    data_dir = root / "data" / "spaceship-titanic"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("PassengerId,Transported\n1,True\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("PassengerId\n2\n", encoding="utf-8")
    task_path = root / ".xsci" / "tasks" / "spaceship-titanic.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["local_data_dir"] = str(data_dir)
    task["task_type"] = "classification"
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")

    record_scientist_turn(root, {
        "task": "spaceship-titanic",
        "route": "scientist_terminal_turn",
        "user": "让系统像 AI Scientist 一样复盘上一轮 token=LEAKME",
        "executed_tools": [{"tool": "scientist_turn_plan"}, {"tool": "scientist_self_audit"}],
        "mode": "meta_scientist",
        "decision": {
            "scientific_critique": {
                "evidence_gaps": [
                    {
                        "gap": "missing_execution_contract token=LEAKME",
                        "severity": "blocking",
                        "suggested_tool": "scientist_execution_contract",
                    },
                    {
                        "gap": "agent_capability_audit_needed",
                        "severity": "high",
                        "suggested_tool": "scientist_self_audit",
                    },
                ],
                "uncertainty_drivers": ["No execution contract before training."],
                "claim_boundaries": ["No top30/medal claim without Kaggle response artifact."],
            },
            "tool_budget": {
                "recommended_min_tools": 6,
                "requested_max_tools": 2,
                "effective_max_tools": 2,
                "executed_tool_count": 2,
                "must_run_deferred_count": 2,
            },
            "deferred_tools": ["scientist_autopilot", "scientist_workplan"],
            "must_run_deferred_tools": ["scientist_autopilot", "scientist_workplan"],
            "budget_exhausted": True,
        },
        "blockers": [],
        "next_actions": ["Run scientist_execution_contract before model search."],
        "answer_preview": "password=LEAKME should be redacted",
        "no_training_started": True,
    })

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("research_decision", state, root)
    serialized = json.dumps(result, ensure_ascii=False)
    brief = result["research_brief"]

    assert result["decision"]["selected_action"] == "complete_scientist_turn_closure"
    assert result["decision"]["selected_branch"] == "scientist_turn_budget_repair"
    assert result["decision"]["code_generation_mode"] == "none"
    assert brief["turn_derived_evidence_gaps"]
    assert brief["turn_derived_budget_risks"]["budget_exhausted_turns"] == 1
    assert brief["turn_derived_budget_risks"]["must_run_deferred_tools"]
    assert brief["turn_derived_claim_boundaries"]
    assert "scientist_autopilot" in serialized
    assert "missing_execution_contract" in serialized
    assert "Kaggle response artifact" in serialized
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized
    assert "LEAKME" not in serialized


def test_scientist_workplan_persists_steps_and_gates_without_secret(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools
    from xsci.scientist_trace import load_recent_scientist_step_events

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    data_dir = root / "data" / "spaceship-titanic"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("PassengerId,Transported\n1,True\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("PassengerId\n2\n", encoding="utf-8")
    task_path = root / ".xsci" / "tasks" / "spaceship-titanic.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["local_data_dir"] = str(data_dir)
    task["task_type"] = "classification"
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_workplan", state, root)
    serialized = json.dumps(result, ensure_ascii=False)
    artifact = root / ".xsci" / "scientist_workplan.json"

    assert result["ok"] is True
    assert result["tool"] == "scientist_workplan"
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert result["summary"]["steps_total"] >= 8
    assert result["current_focus"]["step_id"] != "official_submit"
    assert any(step["id"] == "execute_candidate" for step in result["steps"])
    assert artifact.exists()
    step_events = load_recent_scientist_step_events(root, limit=40)
    assert any(event.get("phase") == "workplan_snapshot" for event in step_events)
    assert any(event.get("phase") == "workplan_step" and event.get("step_id") == "execute_candidate" for event in step_events)
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_repair_plan_diagnoses_data_blocker_without_training(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools
    from xsci.scientist_trace import load_recent_scientist_step_events
    from xsci.scientist_turns import load_recent_scientist_turns

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)

    state = SessionState.from_root(root)
    state.selected_task = "spaceship-titanic"
    result = TerminalTools.dispatch("scientist_repair_plan", state, root)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["ok"] is True
    assert result["tool"] == "scientist_repair_plan"
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert result["mode"] in {"blocked_repair", "quality_improvement"}
    assert "data_missing" in result["root_causes"]
    assert any(step["id"] == "repair_data_contract" for step in result["repair_steps"])
    assert (root / ".xsci" / "scientist_repair_plan.json").exists()
    step_events = load_recent_scientist_step_events(root, limit=60)
    assert any(event.get("phase") == "repair_plan_snapshot" for event in step_events)
    assert any(event.get("phase") == "repair_step" and event.get("step_id") == "repair_data_contract" for event in step_events)
    turns = load_recent_scientist_turns(root)
    assert turns[-1]["route"] == "scientist_repair_plan"
    assert turns[-1]["no_training_started"] is True
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_repair_plan_console_query_does_not_train(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "哪里卡住了，给我修复计划", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "scientist_repair_plan" in out
    assert "root_causes" in out


def test_scientist_execution_contract_goes_ready_without_training(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools
    from xsci.scientist_trace import load_recent_scientist_step_events
    from xsci.scientist_turns import load_recent_scientist_turns

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    data_dir = root / "data" / "spaceship-titanic"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("PassengerId,Transported\n1,True\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("PassengerId\n2\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("PassengerId,Transported\n2,False\n", encoding="utf-8")
    task_path = root / ".xsci" / "tasks" / "spaceship-titanic.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["local_data_dir"] = str(data_dir)
    task["target_column"] = "Transported"
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("scientist_execution_contract", state, root)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["ok"] is True
    assert result["tool"] == "scientist_execution_contract"
    assert result["go_no_go"] == "go"
    assert result["agent_session_ready"] is True
    assert result["model_training_ready"] is True
    assert result["data_contract_status"] == "ready"
    assert result["execution_gate_decision"]["status"] == "ready_for_gated_training"
    assert result["execution_gate_decision"]["blocked"] is False
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert "rank" in result["claim_boundary"].lower()
    assert "Scientist decision:" in result["enriched_goal"]
    assert (root / ".xsci" / "scientist_execution_contract.json").exists()
    step_events = load_recent_scientist_step_events(root, limit=80)
    assert any(event.get("phase") == "execution_contract_snapshot" for event in step_events)
    turns = load_recent_scientist_turns(root)
    assert turns[-1]["route"] == "scientist_execution_contract"
    assert turns[-1]["no_training_started"] is True
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_scientist_execution_contract_console_query_does_not_train(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "执行前检查一下现在能不能跑", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "scientist_execution_contract" in out
    assert "go_no_go" in out


def test_evolution_tracker_record_run_persists(isolated_autokaggle):
    from xsci.evolution_tracker import EvolutionTracker

    root = xcfg.active_root()
    EvolutionTracker(root).record_run(success=True, cv_score=0.81, promotions=1, task="demo")

    snap = EvolutionTracker(root).current_snapshot()
    assert snap.total_runs == 1
    assert snap.total_promotions == 1
    assert snap.best_cv_ever == 0.81


def test_evolution_status_tool_reports_durable_learning(isolated_autokaggle):
    from xsci.evolution_tracker import EvolutionTracker
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST-SHOULD-NOT-LEAK")
    EvolutionTracker(root).record_run(success=True, cv_score=0.8, promotions=1, task="demo")
    memory_path = root / "experiments" / "evolution" / "retrospective_memory.json"
    memory_path.parent.mkdir(parents=True)
    memory_path.write_text(json.dumps([{
        "memory_id": "demo:EXP000",
        "task_type": "classification",
        "dataset_profile": {},
        "method": "Base",
        "what_worked": "clean baseline",
        "what_failed": "",
        "metric_delta": 0.01,
        "reusable_strategy": "stratified_kfold",
        "failure_pattern": "",
        "linked_exp_ids": ["EXP000"],
    }], ensure_ascii=False), encoding="utf-8")

    state = SessionState.from_root(root)
    result = TerminalTools.dispatch("evolution_status", state, root)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["ok"] is True
    assert result["tracker"]["total_runs"] == 1
    assert result["retrospective_memory"]["records"] == 1
    assert "Self-evolution evidence is present" in result["message"]
    assert "sk-TEST-SHOULD-NOT-LEAK" not in serialized


def test_evolution_status_console_query_does_not_train(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "它有没有学到经验", root, None
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected is None
    assert not should_exit
    assert not trained
    assert "evolution_status" in out


def test_research_decision_console_query_does_not_train(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    trained = []
    monkeypatch.setattr(ak, "_run_agent", lambda *a, **k: trained.append(1) or 0)

    rc, selected, should_exit = ak._handle_console_command(
        "下一轮实验决策怎么做", root, "spaceship-titanic"
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert not trained
    assert "research_decision" in out
    assert (root / ".xsci" / "scientist_decision.json").exists()


def test_auto_mode_uses_research_decision_goal(isolated_autokaggle, monkeypatch, capsys):
    root = xcfg.active_root()
    xcfg.write_secret("anthropic_api_key", "sk-TEST")
    ak._register_task("https://www.kaggle.com/competitions/spaceship-titanic", root)
    data_dir = root / "data" / "spaceship-titanic"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("PassengerId,Transported\n1,True\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("PassengerId\n2\n", encoding="utf-8")
    task_path = root / ".xsci" / "tasks" / "spaceship-titanic.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["local_data_dir"] = str(data_dir)
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    calls = {}

    def fake_run_agent(task_name, root_arg, *, goal="", compute=None, resume=False, cfg=None):
        calls["task"] = task_name
        calls["goal"] = goal
        calls["compute"] = compute
        return 0

    monkeypatch.setattr(ak, "_run_agent", fake_run_agent)
    rc, selected, should_exit = ak._handle_console_command("auto spaceship-titanic", root, "spaceship-titanic")
    out = capsys.readouterr().out

    assert rc == 0
    assert selected == "spaceship-titanic"
    assert not should_exit
    assert calls["task"] == "spaceship-titanic"
    assert "branch=baseline" in calls["goal"]
    assert "code_generation_mode=Base" in calls["goal"]
    assert "Decision artifact" in out


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
    tracker.record_lesson(reusable=True, failure=True)
    tracker.record_task_completed("test")
    tracker.record_cross_task_transfer("titanic", "house_prices")

    snapshot = tracker.current_snapshot()
    assert snapshot.total_runs == 1
    assert snapshot.repair_attempts == 1
    assert snapshot.repair_successes == 1
    assert snapshot.innovations_tried == 1
    assert snapshot.innovation_successes == 1
    assert snapshot.lessons_recorded == 1
    assert snapshot.reusable_lessons == 1
    assert snapshot.failure_lessons == 1
    assert snapshot.tasks_completed == 1
    assert snapshot.cross_task_transfers == 1
    # With 2*2 + 3 + 5 + 3 + 2 = 17 score → should be competent
    assert snapshot.skill_level in ("competent", "expert", "master")

    # Verify the report method works
    report = tracker.report()
    assert "Self-Evolution" in report
    assert "Lessons recorded" in report
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


def test_negated_training_request_routes_to_planning():
    goal = (
        "\u4e0d\u8981\u5f00\u59cb\u8bad\u7ec3\uff0c\u53ea\u5206\u6790\u5f53\u524d\u8bc1\u636e"
        "\u5e76\u63d0\u51fa\u4e09\u4e2a\u53ef\u8bc1\u4f2a\u5047\u8bbe"
    )
    assert ki.classify(goal).kind == ki.PLANNING
    assert ki.classify("\u73b0\u5728\u5f00\u59cb\u8bad\u7ec3").kind == ki.EXECUTION


def test_reasoning_synthesis_fulfills_falsifiable_hypothesis_contract(isolated_autokaggle):
    from xsci.scientist_reasoning import build_scientist_reasoning_synthesis

    root = xcfg.active_root()
    state = SessionState.from_root(root)
    state.selected_task = "house-prices"
    state.llm_ready = False
    goal = (
        "\u4e0d\u8981\u5f00\u59cb\u8bad\u7ec3\uff0c\u53ea\u5206\u6790\u5f53\u524d\u8bc1\u636e\uff0c"
        "\u63d0\u51fa\u4e09\u4e2a\u53ef\u8bc1\u4f2a\u5047\u8bbe\uff0c"
        "\u6bd4\u8f83\u8bc1\u636e\u3001\u98ce\u9669\u548c\u6210\u672c\uff0c"
        "\u9009\u62e9\u4e0b\u4e00\u6b65\u5b89\u5168\u52a8\u4f5c\u3002"
    )
    evidence = {
        "task_profile": {
            "task_slug": "house-prices",
            "task_type": "regression",
            "modality": "tabular",
            "metric": "rmse",
            "target_column": "SalePrice",
        },
        "readiness": {"blocking_gates": ["gpu gate blocked"]},
    }
    result = build_scientist_reasoning_synthesis(
        state,
        root,
        goal=goal,
        evidence=evidence,
    )
    cached = build_scientist_reasoning_synthesis(state, root, goal=goal, evidence=evidence)

    assert result["reasoning_quality"]["score"] >= 85
    assert result["reasoning_quality"]["hypotheses_requested"] == 3
    assert result["reasoning_quality"]["complete_falsifiable_hypotheses"] >= 3
    assert len(result["hypotheses"]) >= 3
    assert all(item["disconfirming_result"] for item in result["hypotheses"][:3])
    assert result["next_safe_action"]["command"] == "evomind repair"
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
    assert cached["cache_hit"] is True
    assert cached["cache_stats"]["hits"] >= 1
    assert cached["cache_stats"]["hit_ratio"] > 0
    assert (root / ".xsci" / "scientist_reasoning_synthesis.json").exists()
    persisted = json.loads((root / ".xsci" / "scientist_reasoning_synthesis.json").read_text(encoding="utf-8"))
    assert persisted["cache_hit"] is True
    assert persisted["cache_stats"]["hits"] >= 1
    assert (root / ".xsci" / "scientist_reasoning_synthesis.md").exists()
    assert (root / ".xsci" / "scientist_reasoning_cache_stats_deterministic.json").exists()


def test_memory_relevance_prefers_same_task_and_penalizes_cross_modality():
    from xsci.terminal_tools import _memory_relevance

    task = {
        "task_slug": "house-prices",
        "task_type": "regression",
        "modality": "tabular",
        "metric": "rmse",
    }
    same_task = {
        "task_type": "regression",
        "dataset_profile": {
            "task_slug": "house-prices",
            "modality": "tabular",
            "metric": "rmse",
        },
        "reusable_strategy": "log1p target and leakage-safe OOF blend",
    }
    cross_task = {
        "task_type": "regression",
        "dataset_profile": {
            "task_slug": "stanford-covid-vaccine",
            "modality": "sequence",
            "metric": "mcrmse",
        },
        "reusable_strategy": "RNA base-pair features and sequence window expansion",
    }

    same_score, same_reasons = _memory_relevance(same_task, task)
    cross_score, cross_reasons = _memory_relevance(cross_task, task)
    assert same_score > cross_score
    assert "same_task" in same_reasons
    assert "cross_modality_method_penalty" in cross_reasons


def test_engineering_aliases_route_to_isolated_loop(isolated_autokaggle, monkeypatch):
    calls = []

    def fake_engineering(argv, root):
        calls.append((list(argv), str(root)))
        return 0

    monkeypatch.setattr(ak, "_run_scientist_engineering_command", fake_engineering)
    for alias in ("engineer", "engineering-loop", "validate-patch", "execute-upgrade"):
        assert ak.main([alias]) == 0
    assert len(calls) == 4


def test_engineering_intent_routes_before_self_upgrade():
    for text in (
        "validate patch in isolated worktree",
        "\u9a8c\u8bc1\u8865\u4e01",
        "\u6267\u884c\u5de5\u7a0b\u95ed\u73af",
        "\u6267\u884c\u81ea\u5347\u7ea7\u8865\u4e01",
    ):
        intent = ki.classify(text)
        assert intent.kind == ki.TOOL_QUERY
        assert intent.payload == "scientist_engineering_loop"


def test_memory_consolidation_records_isolated_engineering_evidence(isolated_autokaggle):
    from xsci.terminal_tools import TerminalTools

    root = xcfg.active_root()
    state = SessionState.from_root(root)
    state.selected_task = "house-prices"
    xsci = root / ".xsci"
    xsci.mkdir(parents=True, exist_ok=True)
    engineering = {
        "tool": "scientist_engineering_loop",
        "run_id": "engineering_test",
        "status": "passed_review_candidate",
        "changed_files": ["src/xsci/demo.py"],
        "acceptance_checks": [
            {"command": "python -m py_compile src/xsci/demo.py", "passed": True},
            {"command": "git diff --check", "passed": True},
        ],
        "main_worktree_modified": False,
        "merge_ready": True,
        "candidate_diff_path": ".xsci/engineering_runs/engineering_test/candidate.diff",
        "run_manifest_path": ".xsci/engineering_runs/engineering_test/manifest.json",
        "human_gate": "review_candidate_before_merge",
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    (xsci / "scientist_engineering_loop.json").write_text(
        json.dumps(engineering),
        encoding="utf-8",
    )
    with (xsci / "scientist_engineering_trials.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "run_id": "engineering_test",
            "status": "passed_review_candidate",
            "acceptance_passed": True,
            "main_worktree_modified": False,
            "candidate_diff_path": engineering["candidate_diff_path"],
        }) + "\n")

    result = TerminalTools.dispatch("scientist_memory_consolidation", state, root)
    records = json.loads(
        (root / "experiments" / "evolution" / "retrospective_memory.json").read_text(encoding="utf-8")
    )

    assert result["ok"] is True
    assert result["source_counts"]["engineering_loop_present"] is True
    assert result["source_counts"]["engineering_trials"] == 1
    assert any(item.get("method") == "isolated_engineering_validation" for item in records)
    assert any(item.get("method") == "isolated_engineering_trial" for item in records)
    assert result["no_training_started"] is True
    assert result["official_submit"] == "blocked_until_explicit_human_approval"
