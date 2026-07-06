"""Kaggle Research Agent shell — Claude-Code-like research terminal.

`kaggle` enters the research-agent conversation. `kaggle official ...` passes
through to the official Kaggle CLI.
"""
from __future__ import annotations

import contextlib
import getpass
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import kaggle_menu
from .config import (
    GLOBAL_DIR, active_root, inject_engine_env, is_onboarded,
    load_config, mark_onboarded, set_global, write_secret,
)
from .kaggle_conversation import ConversationAgent
from .kaggle_intent import (
    CAPABILITY, CHAT, EXECUTION, GREETING, MEMORY,
    OFFICIAL, PLANNING, REPORT, STATUS, TASK_ADD, TASK_USE, classify,
)
from .kaggle_session import MODE_CHAT, MODE_EXECUTING, MODE_PLANNING, SessionState
from .kaggle_stream import StageRenderer, thinking
from .login import import_kaggle_json, save_kaggle_api_token, save_kaggle_credentials, save_llm_credentials
from .tasks import add_task, list_tasks, resolve_task, slugify

_XSCI_COMMANDS = {"doctor", "config", "init", "login", "task", "run", "report", "watch", "dashboard", "memory"}
_CONVERSATION: Optional[ConversationAgent] = None


def _conversation() -> ConversationAgent:
    global _CONVERSATION
    if _CONVERSATION is None:
        _CONVERSATION = ConversationAgent()
    return _CONVERSATION


def _ansi(code: str, text: str) -> str:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _dim(text: str) -> str:
    return _ansi("90", text)


def _accent(text: str) -> str:
    return _ansi("96", text)


def _strong(text: str) -> str:
    return _ansi("97;1", text)


def logo() -> str:
    return "\n".join([
        _accent("  __ __                 __        "),
        _accent(" / //_/_ ____ ____ ____/ /__      "),
        _accent("/ ,< / _ `/ _ `/ _ `/ _  / -_)    "),
        _accent("/_/|_|\\_,_/\\_, /\\_,_/\\_,_/\\__/  "),
        _accent("          /___/                   "),
        f"{_strong('Kaggle Research Agent')}  {_dim('XSCI self-evolving MLE workstation')}",
    ])


def _agent_reply(text: str, *, title: str = "Kaggle Agent") -> None:
    print()
    print(_strong(title))
    for paragraph in text.strip().split("\n"):
        if not paragraph.strip():
            print()
            continue
        wrapped = textwrap.wrap(paragraph, width=88, replace_whitespace=False) or [""]
        for line in wrapped:
            print(f"  {line}")


def _has_llm(cfg=None) -> bool:
    cfg = cfg or load_config()
    return bool(cfg.get("secrets.anthropic_api_key") or cfg.get("secrets.deepseek_api_key")
                or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"))


def _has_kaggle(cfg=None) -> bool:
    cfg = cfg or load_config()
    return bool(cfg.get("secrets.kaggle_api_token")
                or (cfg.get("secrets.kaggle_username") and cfg.get("secrets.kaggle_key"))
                or os.environ.get("KAGGLE_API_TOKEN")
                or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")))


def _print_readiness(root: Path) -> None:
    cfg = load_config(root)
    inject_engine_env(cfg)
    session = SessionState.from_root(root, cfg=cfg)
    print(logo())
    print()
    print(_strong("Readiness"))
    for label, value in session.status_rows():
        print("  " + f"{label:<11} {value}")
    print()
    print(_strong("Dashboard"))
    print("  http://127.0.0.1:8088/?page=control")
    gaps = session.missing_setup()
    if gaps:
        print()
        print(_strong("Setup gaps"))
        for gap in gaps:
            print(f"  - {gap}")
    else:
        print()
        print("  core config ready; official Kaggle submit remains human-gated.")


@contextlib.contextmanager
def _inside_workspace(root: Path):
    old = Path.cwd()
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(old)


def _register_task(source: str, root: Optional[Path] = None, force: bool = False) -> str:
    root = root or active_root()
    if source.startswith(("http://", "https://")):
        from urllib.parse import urlparse
        path = urlparse(source).path
        parts = [p for p in path.split("/") if p]
        slug_candidate = parts[-1] if parts else source
    else:
        slug_candidate = source
    slug = slugify(slug_candidate)
    add_task(slug, url=source if source.startswith("http") else "", root=root)
    return slug


def official_main(argv: Optional[list[str]] = None) -> int:
    import subprocess
    args = list(argv or [])
    cmd = ["kaggle-official"] + args
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("Official Kaggle CLI not found. Install: pip install kaggle")
        return 1


def _competitions_cmd(args: list[str]) -> int:
    from . import kaggle_competitions
    sub = args[0].lower() if args else "list"
    query = " ".join(args[1:]) if len(args) > 1 else ""
    if sub in {"search", "find"} and not query:
        print("Usage: kaggle competitions search <keyword>")
        return 1
    result = kaggle_competitions.list_competitions(query=query if sub in {"search", "find", "list"} else f"{sub} {' '.join(args[1:])}".strip())
    if not result.get("ok"):
        print(f"Error: {result.get('message', 'Unknown error')}")
        return 1
    comps = result.get("competitions", [])
    if not comps:
        print("No competitions found.")
        return 0
    print(f"\n  Found {len(comps)} competition(s):\n")
    for i, c in enumerate(comps, 1):
        title = c.get("title", c.get("ref", "?"))
        slug = c.get("ref", "?")
        cat = c.get("category", "")
        deadline = c.get("deadline", "")
        print(f"  {i:2d}. {title}")
        print(f"      slug: {slug}  |  category: {cat}  |  deadline: {deadline}")
        if c.get("reward"):
            print(f"      reward: {c['reward']}")
        print()
    print(f"  To start: kaggle task add https://www.kaggle.com/c/<slug>")
    return 0


def _run_agent(task: str, root: Optional[Path] = None, *, goal: str = "",
               compute: Optional[str] = None, resume: bool = False, cfg=None) -> int:
    from .agent import run_agent
    return run_agent(task, goal=goal, compute=compute, resume=resume, cfg=cfg,
                     event_renderer=StageRenderer(), show_plan=False)


def _delegate_xsci(argv: list[str], root: Path) -> int:
    from . import __main__ as xsci_main
    return xsci_main.main(argv)


# ── Console loop ────────────────────────────────────────────

def _dispatch_intent(line: str, root: Path, session: SessionState) -> tuple[int, bool]:
    raw = (line or "").strip()
    if not raw:
        if session.selected_task:
            _agent_reply(
                "Current task: " + str(session.selected_task) + ". "
                "Describe your research goal or type /help."
            )
        else:
            _agent_reply("No task selected. Use `competitions` to browse, or `task add <url>` to register one. Type /help.")
        return 0, False

    stripped = raw[1:].strip() if raw.startswith("/") else raw
    parts = stripped.split()
    verb = parts[0].lower() if parts else ""
    rest = parts[1:]

    if verb in {"exit", "quit", "q"}:
        return 0, True
    if verb in {"help", "?"}:
        _print_console_help(session.selected_task)
        return 0, False
    if verb == "setup":
        return run_setup(force=True), False
    if verb == "official":
        return official_main(rest), False
    if verb in {"dashboard", "open"}:
        return _delegate_xsci(["dashboard", *(rest or ["start"])], root), False
    if verb in {"resume", "continue"}:
        task = (rest[0] if rest else None) or session.selected_task
        if not task:
            _agent_reply("No task selected for resume. First `task add <url>` or `use <task>`.")
            return 1, False
        session.current_mode = MODE_EXECUTING
        rc = _run_agent(task, root, goal=session.last_goal or "Continue from best-so-far.", resume=True)
        session.current_mode = MODE_CHAT
        return rc, False
    if verb in {"watch", "report", "memory", "run", "doctor", "config", "init", "login"}:
        if verb == "run":
            task = (rest[0] if rest else None) or session.selected_task
            if not task:
                _agent_reply("No task selected. `task add <url>` first.")
                return 1, False
            return _run_agent(task, root, goal=session.last_goal), False
        return _delegate_xsci([verb, *rest], root), False
    if verb in {"competitions", "comps"}:
        return _competitions_cmd(rest), False
    if verb in {"download", "dl"}:
        from . import kaggle_actions
        slug = rest[0] if rest else (session.selected_task or "")
        if not slug:
            _agent_reply("Usage: download <task-slug>. E.g. `download titanic`.")
            return 1, False
        with thinking("downloading"):
            result = kaggle_actions.download_competition_data(slug)
        if result.get("ok"):
            _agent_reply("Downloaded: " + slug + ". Files: " + ", ".join(result.get("files", [])))
        else:
            _agent_reply("Download failed: " + result.get("message", ""))
        return 0 if result.get("ok") else 1, False
    if verb in {"task", "tasks"}:
        sub = rest[0].lower() if rest else "list"
        if sub in {"list", "ls"}:
            _print_task_list(root, session.selected_task)
            return 0, False
        if sub == "add" and len(rest) >= 2:
            slug = _register_task(rest[1], root)
            if slug:
                session.selected_task = slug
                print(f"selected task: {slug}")
                return 0, False
            return 1, False
        _agent_reply("Usage: task list | task add <url>")
        return 1, False
    if verb == "use":
        task = slugify(rest[0]) if rest else ""
        if not task:
            _agent_reply("Usage: use <task-name>")
            return 1, False
        try:
            resolve_task(task, project_root=root)
            session.selected_task = task
            print(f"selected task: {task}")
            return 0, False
        except FileNotFoundError:
            print(f"task not found: {task}")
            return 1, False

    intent = classify(stripped)
    if intent.kind == GREETING:
        _agent_reply("Hello. I am the Kaggle Research Agent. Browse competitions, pick one, and let us get started. Type `competitions` or ask me anything.")
        return 0, False
    if intent.kind == STATUS:
        _agent_reply("System status: check `kaggle ready` for details.", title="System status")
        return 0, False
    if intent.kind == CAPABILITY:
        with thinking("thinking"):
            reply = _conversation().capability(session)
        _agent_reply(reply)
        return 0, False
    if intent.kind == TASK_ADD:
        if not intent.payload:
            _agent_reply("Usage: `task add <kaggle-url|name>`")
            return 1, False
        slug = _register_task(intent.payload, root)
        if slug:
            session.selected_task = slug
            print(f"selected task: {slug}")
            return 0, False
        return 1, False
    if intent.kind == TASK_USE:
        task = slugify(intent.payload)
        try:
            resolve_task(task, project_root=root)
            session.selected_task = task
            print(f"selected task: {task}")
            return 0, False
        except FileNotFoundError:
            print(f"task not found: {task}")
            return 1, False
    if intent.kind == OFFICIAL:
        return official_main(intent.args), False
    if intent.kind == REPORT:
        return _delegate_xsci(["report"], root), False
    if intent.kind == MEMORY:
        return _delegate_xsci(["memory"], root), False
    if intent.kind == PLANNING:
        session.current_mode = MODE_PLANNING
        session.last_goal = raw
        if not session.selected_task:
            _agent_reply("I can plan, but no competition is selected. Browse with `competitions` or `task add <url>` first.")
            return 0, False
        with thinking("planning"):
            plan_text = _conversation().plan(raw, session)
        _agent_reply(plan_text, title="Research plan")
        return 0, False
    if intent.kind == EXECUTION:
        session.last_goal = raw
        if not session.selected_task:
            _agent_reply("I see you want to train, but no competition is selected. Browse with `competitions` first.")
            return 1, False
        session.current_mode = MODE_EXECUTING
        rc = _run_agent(session.selected_task, root, goal=raw)
        session.current_mode = MODE_CHAT
        return rc, False

    session.current_mode = MODE_CHAT
    with thinking("thinking"):
        reply = _conversation().reply(raw, session)
    _agent_reply(reply)
    return 0, False


def _print_task_list(root: Path, selected_task: Optional[str]) -> None:
    tasks = list_tasks(root)
    if not tasks:
        _agent_reply("No competitions registered. Use `competitions` to browse Kaggle, then `task add <url>` to register one.")
        return
    print()
    print(_strong("Registered tasks"))
    for slug, _path in tasks:
        mark = _accent("*") if slug == selected_task else " "
        print(f"  {mark} {slug}")
    print(_dim("  `use <task>` to select; `task add <url>` to register a new one."))


def _print_console_help(selected_task: Optional[str]) -> None:
    print(_strong("Commands (also work with a leading /)"))
    print("  competitions [query]     browse/search Kaggle competitions")
    print("  task add <url>           register a competition")
    print("  task list                list registered tasks")
    print("  download <task>          download competition data")
    print("  use <task>               select a task")
    print("  status                   show setup gaps")
    print("  setup                    configuration wizard")
    print("  run [task]               start training")
    print("  resume [task]            continue previous run")
    print("  watch / report / memory  engine views")
    print("  dashboard                manage the 8088 workstation")
    print("  official <args...>       Kaggle CLI passthrough")
    print("  exit                     quit")
    print()
    print(_dim("Natural language: /competitions titanic /download /plan /start training"))
    print(f"{_dim('Selected task:')} {selected_task or '(none)'}")


def _print_welcome(session: SessionState) -> None:
    print(logo())
    print()
    if session.selected_task:
        print(_strong("Current task"))
        print(f"  {session.selected_task}")
        if session.task_brief:
            print(_dim(f"  {session.task_brief}"))
        print()
    gaps = session.missing_setup()
    if gaps:
        print(_strong("Setup checklist"))
        for gap in gaps:
            head = gap.split(":", 1)[0]
            print(f"  [ ] {head}")
        print(_dim("  Run setup to configure. You can skip any of them."))
    elif session.selected_task:
        print(_dim("Type a research goal, or /run when you want to start execution."))
    print(_dim("Type help for commands, setup to configure, exit to quit."))


def run_console(root: Optional[Path] = None) -> int:
    root = root or active_root()
    cfg = load_config(root)
    inject_engine_env(cfg)
    session = SessionState.from_root(root, cfg=cfg)
    _print_welcome(session)
    session.persist(root)
    print()
    while True:
        try:
            line = input("kaggle> ").strip()
        except EOFError:
            print("\nbye.")
            return 0
        except KeyboardInterrupt:
            print(_dim("\n(use exit to quit)"))
            continue
        prev_task = session.selected_task
        try:
            rc, should_exit = _dispatch_intent(line, root, session)
        except KeyboardInterrupt:
            print()
            _agent_reply("Interrupted. Back to conversation terminal.", title="Interrupted")
            continue
        if session.selected_task != prev_task:
            session.refresh_recent_run(root)
            session.refresh_task_brief(root)
        session.persist(root)
        if should_exit:
            print("bye.")
            return rc


def _print_help() -> None:
    print(logo())
    print()
    print("Usage:")
    print("  kaggle                       enter the research-agent conversation")
    print("  kaggle setup                 first-run setup wizard")
    print("  kaggle ready                 show terminal/model/Kaggle/GPU readiness")
    print("  kaggle status                same as ready")
    print("  kaggle competitions [query]  browse/search Kaggle competitions")
    print("  kaggle task add <url>        register a Kaggle/MLE-Bench task")
    print("  kaggle download <task>       download competition data")
    print("  kaggle agent <task>          open the deep research agent on a task")
    print("  kaggle run <task>            run the audited evolution loop")
    print("  kaggle watch -f              follow the latest event stream")
    print("  kaggle memory                inspect retrospective memory")
    print("  kaggle dashboard start       open/manage the 8088 workstation")
    print("  kaggle official ...          pass through to the official Kaggle CLI")
    print()
    print("Alias:")
    print("  autokaggle                   same agent shell without overriding Kaggle CLI")
    print("  kaggle-official ...          direct official Kaggle CLI passthrough")
    print()
    print("Default dashboard gateway:")
    print("  http://127.0.0.1:8088/?page=control")
    print()
    print(_dim("Official Kaggle submit remains human-gated by the workstation policy."))


def _dispatch(argv: list[str], root: Path) -> int:
    if not argv:
        return run_console(root)
    cmd = argv[0].lower()
    if cmd in {"-h", "--help", "help"}:
        _print_help()
        return 0
    if cmd in {"ready", "status"}:
        _print_readiness(root)
        return 0
    if cmd in {"setup", "configure", "--setup"}:
        return run_setup(force=True)
    if cmd == "official":
        return official_main(argv[1:])
    if cmd == "agent":
        if not argv[1:]:
            return run_console(root)
        return _run_agent(argv[1], root, goal=" ".join(argv[2:]))
    if cmd == "competitions":
        return _competitions_cmd(argv[1:])
    if cmd == "download":
        from . import kaggle_actions
        slug = argv[1] if len(argv) > 1 else ""
        if not slug:
            print("Usage: kaggle download <task-slug>")
            return 1
        result = kaggle_actions.download_competition_data(slug)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok") else 1
    if cmd in _XSCI_COMMANDS:
        return _delegate_xsci(argv, root)
    if cmd in {"add", "register"} and len(argv) >= 2:
        slug = _register_task(argv[1], root)
        return 0 if slug else 1
    source = argv[0]
    goal = ""
    if "--goal" in argv:
        idx = argv.index("--goal")
        goal = " ".join(argv[idx + 1:])
    task = _register_task(source, root) if source.startswith(("http://", "https://")) else slugify(source)
    return _run_agent(task, root, goal=goal)


# ── Setup wizard ─────────────────────────────────────────────

def _setup_guidance(cfg=None, selected_task: Optional[str] = None) -> str:
    root = active_root()
    state = SessionState.from_root(root, cfg=cfg or load_config(root))
    if selected_task is not None:
        state.selected_task = selected_task
    gaps = state.missing_setup()
    if not gaps:
        return "All core config ready. You can directly input a research goal, or use /run to start the audited loop."
    return "Missing setup:\n" + "\n".join(f"- {item}" for item in gaps)


def _safe_input(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"  {prompt}{suffix}> ").strip()
    return value or default


def _yes(prompt: str, default: bool = True) -> bool:
    flag = "Y/n" if default else "y/N"
    value = input(f"  {prompt} [{flag}]> ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes", "1", "true"}


def _setup_step(index: int, title: str, detail: str) -> None:
    print()
    print(_strong(f"{index}/3 {title}"))
    print(_dim(f"    {detail}"))


def run_setup(*, force: bool = False, reason: str = "") -> int:
    if not sys.stdin.isatty() and not force:
        print("Setup needs an interactive terminal. Run `kaggle setup` in a TTY.")
        return 1
    print(logo())
    print()
    if reason == "first_run":
        print(_strong("Welcome. Let us set up your research agent before we open the terminal."))
    elif reason == "llm_missing":
        print(_strong("Almost there - the terminal needs an LLM brain before it can chat."))
    elif reason:
        print(_strong(reason))
    else:
        print(_strong("Configuration wizard"))

    # Step 1: LLM
    _setup_step(1, "LLM brain", "Drives planning, code generation, audit, and self-evolution.")
    print("  Available providers:")
    print("    1. Anthropic (Claude) - native tool-use, recommended for deep research")
    print("    2. DeepSeek - OpenAI-compatible chat/planning/code generation")
    print("    3. OpenAI (GPT) - /v1/chat/completions endpoint")
    choice = _safe_input("Select provider [1-3]", "1")
    if choice == "1":
        key = _safe_input("Anthropic API key (sk-ant-...)")
        if key:
            save_llm_credentials("anthropic", api_key=key)
            set_global("llm", "provider", "anthropic")
            print("  saved Anthropic API key to secure storage")
    elif choice == "2":
        key = _safe_input("DeepSeek API key (sk-...)")
        if key:
            save_llm_credentials("deepseek", api_key=key)
            set_global("llm", "provider", "deepseek")
            print("  saved DeepSeek API key to secure storage")
    elif choice == "3":
        key = _safe_input("OpenAI API key (sk-...)")
        if key:
            save_llm_credentials("deepseek", api_key=key)
            set_global("llm", "provider", "deepseek")
            print("  saved OpenAI API key (routed as DeepSeek-compatible)")
    else:
        print("  skipped - you can configure later with `kaggle setup`")

    # Step 2: Kaggle
    _setup_step(2, "Kaggle account", "Used for data access and human-gated submissions.")
    if _yes("Configure Kaggle API now?", True):
        token = _safe_input("Kaggle API token (kaggle.json format)")
        if token:
            save_kaggle_api_token(token)
            print("  saved Kaggle API token")
    else:
        print("  skipped - local/offline tasks still work.")

    # Step 3: Compute
    _setup_step(3, "Compute backend", "Choose local for small tests, or gpu for SSH/HPC execution.")
    compute = _safe_input("Default compute backend: local/gpu", "local").lower()
    if compute in {"local", "gpu"}:
        set_global("compute", "backend", compute)
        print(f"  set compute backend to {compute}")
    else:
        print("  using default: local")

    mark_onboarded()
    print()
    _agent_reply("Setup complete. Type a research goal or `competitions` to browse Kaggle tasks.", title="Ready")
    return 0


# ── Entry points ─────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = active_root()
    if args and args[0] in {"-h", "--help", "help", "official"}:
        return _dispatch(args, root)
    if not args and sys.stdin.isatty() and not _has_llm():
        reason = "first_run" if not is_onboarded() else "llm_missing"
        rc = run_setup(force=True, reason=reason)
        if rc != 0:
            return rc
    return _dispatch(args, root)


if __name__ == "__main__":
    raise SystemExit(main())
