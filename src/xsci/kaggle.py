"""EvoMind shell - Claude-Code-like research terminal.

`evomind` enters the research-agent conversation. `evomind official ...`
passes through to the official Kaggle CLI. Legacy `kaggle` and `autokaggle`
shims may exist as compatibility aliases, but EvoMind is the product name.
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
    OFFICIAL, PLANNING, REPORT, STATUS, TASK_ADD, TASK_USE, TOOL_QUERY, classify,
)
from .terminal_agent import TerminalAgent, _agent_reply as _term_agent_reply
from .recovery_guard import RecoveryGuard
from .kaggle_session import MODE_CHAT, MODE_EXECUTING, MODE_PLANNING, SessionState
from .kaggle_stream import StageRenderer, thinking
from .login import import_kaggle_json, save_kaggle_api_token, save_kaggle_credentials, save_llm_credentials
from .tasks import add_task, list_tasks, resolve_task, slugify

_XSCI_COMMANDS = {"doctor", "config", "init", "login", "task", "run", "report", "watch", "dashboard", "memory", "evolution", "innovate"}
_CONVERSATION: Optional[ConversationAgent] = None
_DEFAULT_DASHBOARD_URL = "http://127.0.0.1:8088/?page=control"


@dataclass
class _Provider:
    label: str
    hint: str
    family: str
    base_url: str = ""
    models: tuple[str, ...] = ()


_PROVIDERS = (
    _Provider("Anthropic (Claude)", "Native tool-use; recommended for the full deep research agent.",
              "anthropic", "", ("claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5-20251001")),
    _Provider("Anthropic-compatible gateway", "Local/proxy /v1/messages endpoint.", "anthropic", "", ()),
    _Provider("DeepSeek", "OpenAI-compatible chat/planning/code generation.",
              "deepseek", "https://api.deepseek.com", ("deepseek-chat", "deepseek-reasoner")),
    _Provider("OpenAI (GPT)", "OpenAI /v1/chat/completions endpoint.",
              "deepseek", "https://api.openai.com", ("gpt-4o", "gpt-4o-mini", "gpt-4.1")),
    _Provider("OpenAI-compatible gateway", "Qwen/Kimi/GLM/vLLM/LM Studio style gateway.", "deepseek", "", ()),
)


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
        _accent("   ______           __  ___ _           __"),
        _accent("  / ____/   ______ /  |/  /(_)___  ____/ /"),
        _accent(" / __/ | | / / __ `/ /|_/ // / __ \\/ __  / "),
        _accent("/ /___ | |/ / /_/ / /  / // / / / / /_/ /  "),
        _accent("\\____/ |___/\\__,_/_/  /_//_/_/ /_/\\__,_/   "),
        f"{_strong('EvoMind')}  {_dim('auditable self-evolving AI scientist')}",
    ])


def _agent_reply(text: str, *, title: str = "EvoMind") -> None:
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


def _infer_compute_override(text: str) -> Optional[str]:
    low = (text or "").lower()
    local_terms = (
        "本地算力", "本地", "local", "cpu", "本机", "不用gpu", "不用 gpu",
        "不要gpu", "不要 gpu", "小任务", "小数据",
    )
    gpu_terms = (
        "gpu", "hpc", "服务器", "集群", "远端", "远程", "算力服务器",
        "a800", "cuda",
    )
    if any(term in low for term in local_terms):
        return "local"
    if any(term in low for term in gpu_terms):
        return "gpu"
    return None


def _wants_model_status(text: str) -> bool:
    low = (text or "").lower()
    return (
        ("模型" in low and any(term in low for term in ("什么", "哪个", "使用", "当前", "现在")))
        or "model" in low
        or "llm" in low
    )


# ── Task-switch detection (Bug #2) ─────────────────────────────────────
# "切换到 house_prices" / "切到 X 开始训练" classify as EXECUTION, but the
# EXECUTION branch used to train the *currently* selected task and never
# switch. These helpers let the dispatcher switch first, then decide whether
# the same utterance also asked to start training.
_SWITCH_CUES = (
    "切换到", "切换成", "切到", "换到", "换成", "转到",
    "switch to", "change to", "use task",
)
_TRAIN_CUES = (
    "训练", "train", "开始", "开跑", "跑一", "跑起来", "跑通", "跑一遍",
    "run", "执行", "execute", "baseline", "基线", "建模",
    "自进化", "evolve", "进化", "继续", "resume",
)


def _has_switch_cue(text: str) -> bool:
    low = (text or "").lower()
    return any(cue in low for cue in _SWITCH_CUES)


def _mentions_training(text: str) -> bool:
    low = (text or "").lower()
    return any(cue in low for cue in _TRAIN_CUES)


def _normalize_slug(text: str) -> str:
    return (text or "").lower().replace(" ", "").replace("-", "").replace("_", "")


def _match_task_in_text(text: str, root: Path) -> Optional[str]:
    """Return the registered task slug mentioned in ``text``, or None.

    Matches on the raw slug substring or a punctuation-insensitive form so that
    "house_prices", "house-prices", and "house prices" all resolve to the same
    registered task. Prefers the longest matching slug.
    """
    low = (text or "").lower()
    norm = _normalize_slug(text)
    best: Optional[str] = None
    for slug, _ in list_tasks(root):
        s = slug.lower()
        if s in low or _normalize_slug(slug) in norm:
            if best is None or len(slug) > len(best):
                best = slug
    return best


def _model_status_text(session: SessionState) -> str:
    cfg = load_config(Path(session.workspace_root) if session.workspace_root else None)
    provider = str(cfg.get("llm.brand") or cfg.get("llm.provider") or session.llm_provider or "unset")
    model = str(cfg.get("llm.model") or "")
    if not model:
        family = str(cfg.get("llm.provider") or "").lower()
        model = (
            os.environ.get("CLAUDE_CODE_MODEL")
            if family == "anthropic"
            else os.environ.get("DEEPSEEK_MODEL")
        ) or "(provider default)"
    base_url = (
        cfg.get("llm.anthropic_base_url")
        if str(cfg.get("llm.provider") or "").lower() == "anthropic"
        else cfg.get("llm.deepseek_base_url")
    ) or "(provider default)"
    return "\n".join([
        f"当前 LLM provider：{provider}",
        f"当前模型：{model}",
        f"Base URL：{base_url}",
        f"状态：{'ready' if session.llm_ready else 'setup needed'}",
        "说明：不会显示 API key；训练和提交仍走工作站审计门禁。",
    ])


def _dashboard_url(cfg=None) -> str:
    cfg = cfg or load_config()
    value = (
        cfg.get("workstation.dashboard_url")
        or cfg.get("dashboard.url")
        or cfg.get("ui.dashboard_url")
        or _DEFAULT_DASHBOARD_URL
    )
    return str(value).strip() or _DEFAULT_DASHBOARD_URL


def _print_dashboard_hint(cfg=None) -> None:
    print(_strong("Panel"))
    print("  " + _dashboard_url(cfg))
    print(_dim("  Open this URL after `evomind dashboard start` or the workstation launcher starts."))


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
    _print_dashboard_hint(cfg)
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
    try:
        add_task(source, project_root=root, force=force)
    except FileExistsError:
        return slug
    return slug


def official_main(argv: Optional[list[str]] = None) -> int:
    args = list(argv or [])
    if not args or args[0] in {"-h", "--help", "help"}:
        print("Official Kaggle CLI passthrough")
        print("")
        print("Usage:")
        print("  evomind official competitions <args...>")
        print("  evomind official datasets <args...>")
        print("  evomind official kernels <args...>")
        print("  evomind official config <args...>")
        print("")
        print("Direct command:")
        print("  kaggle-official <args...>")
        print("")
        print("Networked official Kaggle commands may require configured credentials.")
        return 0
    before = list(sys.argv)
    sys.argv = ["kaggle", *args]
    try:
        from kaggle.cli import main as kaggle_cli_main
        result = kaggle_cli_main()
        return int(result or 0)
    except ModuleNotFoundError:
        print("Official Kaggle CLI not found. Install: pip install kaggle")
        return 1
    finally:
        sys.argv = before


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
    print(f"  To start: evomind task add https://www.kaggle.com/c/<slug>")
    return 0


def _run_agent(task: str, root: Optional[Path] = None, *, goal: str = "",
               compute: Optional[str] = None, resume: bool = False, cfg=None) -> int:
    from .agent import run_agent
    return run_agent(task, goal=goal, compute=compute, resume=resume, cfg=cfg,
                     event_renderer=StageRenderer(), show_plan=False)


def _auto_research_pipeline(task: str, root: Path, session: SessionState, *,
                            compute: Optional[str] = None) -> int:
    """Autonomous research mode: inspect data → baseline → train → report.

    EvoMind takes the initiative — it checks what's available, decides on a
    sensible first experiment, and runs it.  Every step is gated.
    """
    from .terminal_tools import TerminalTools

    print()
    print(_strong("EvoMind Autonomous Research Mode"))
    print(_dim(f"  Task: {task} | Compute: {compute or session.compute_backend}"))
    print()

    # Step 1: Inspect task
    task_result = TerminalTools.dispatch("inspect_task", session, root)
    if not task_result.get("ok"):
        _agent_reply(f"Cannot inspect task: {task_result.get('message', '')}", title="Auto blocked")
        return 1
    task_info = task_result
    print(_dim(f"  Task: {task_info.get('name', task)}, "
               f"modality={task_info.get('modality')}, "
               f"metric={task_info.get('metric')}({task_info.get('metric_direction')})"))

    # Step 2: Check data
    effective_compute = compute or session.compute_backend
    if effective_compute == "gpu":
        # Data is on remote GPU — check gpu_data_dir config
        gpu_dir = task_info.get("gpu_data_dir", "")
        if not gpu_dir:
            _agent_reply(
                f"GPU data directory not configured for {task}. "
                f"Set gpu_data_dir in the task config.",
                title="Auto blocked"
            )
            return 1
        print(_dim(f"  Data: GPU remote path={gpu_dir}"))
    else:
        data_result = TerminalTools.dispatch("data_check", session, root)
        if not data_result.get("train_csv"):
            _agent_reply(
                f"Training data not found for {task}. "
                f"Run `evomind download {task}` first.",
                title="Auto blocked"
            )
            return 1
        print(_dim(f"  Data: train.csv found, test.csv {'found' if data_result.get('test_csv') else 'missing'}"))

    # Step 3: Check gates
    effective_compute = compute or session.compute_backend
    blockers = session.blocking_setup(compute_override=effective_compute)
    if blockers:
        _agent_reply(
            "Setup needed before autonomous research:\n" + "\n".join(f"- {b}" for b in blockers),
            title="Auto blocked"
        )
        return 1

    # Step 4: Run preflight + training
    goal = (
        f"Autonomous research on {task}: inspect data, establish a strong baseline "
        f"with appropriate preprocessing and model selection, then report results."
    )
    _print_preflight_stream(session, compute=effective_compute, goal=goal)

    if _execution_blocker_reply(session, compute=effective_compute):
        return 1

    session.current_mode = MODE_EXECUTING
    rc = _run_agent(task, root, goal=goal, compute=effective_compute)
    session.current_mode = MODE_CHAT

    # Step 5: Brief post-run analysis
    session.refresh_recent_run(root)
    if session.recent_run_id:
        cv_str = f"{session.recent_best_cv:.4f}" if session.recent_best_cv is not None else "N/A"
        _agent_reply(
            f"Autonomous research completed.\n\n"
            f"Run: {session.recent_run_id}\n"
            f"Best CV: {cv_str}\n\n"
            f"Run `evomind report` for the full research report, or describe your next "
            f"research goal to continue improving.",
            title="EvoMind Auto"
        )
    return rc


def _print_preflight_stream(session: SessionState, *, compute: Optional[str], goal: str) -> None:
    """Render a 6-stage preflight narrative using the StageRenderer.

    Each stage checks the real system state and reports passed/blocked.
    Imitates Claude Code's staged startup output.
    """
    from .terminal_tools import TerminalTools

    effective_compute = compute or session.compute_backend
    renderer = StageRenderer()
    root = Path(session.workspace_root) if session.workspace_root else active_root()

    print()
    print(_strong("EvoMind is preparing an audited research run"))

    # Stage 1: Inspect task
    task_result = TerminalTools.dispatch("inspect_task", session, root)
    task_ok = task_result.get("ok")
    renderer.preflight("Inspecting task",
                       f"task={session.selected_task}, metric={task_result.get('metric', '?')}, "
                       f"modality={task_result.get('modality', '?')}",
                       status="passed" if task_ok else "blocked")
    if not task_ok:
        print(_dim(f"  blocked: {task_result.get('message', '')}"))
        return

    # Stage 2: Check data
    effective_compute = compute or session.compute_backend
    if effective_compute == "gpu":
        # Data is on remote GPU — check gpu_data_dir config instead
        gpu_dir = task_result.get("gpu_data_dir", "")
        renderer.preflight("Checking data",
                           f"GPU remote path={gpu_dir or '(from task config)'}",
                           status="passed" if gpu_dir else "blocked")
    else:
        data_result = TerminalTools.dispatch("data_check", session, root)
        data_ok = data_result.get("ok") and data_result.get("train_csv")
        renderer.preflight("Checking data",
                           f"train.csv={'found' if data_result.get('train_csv') else 'missing'}, "
                           f"test.csv={'found' if data_result.get('test_csv') else 'missing'}",
                           status="passed" if data_ok else "blocked")

    # Stage 3: Check config
    model_result = TerminalTools.dispatch("model_status", session, root)
    model_ready = model_result.get("ready") and model_result.get("ok")
    renderer.preflight("Checking config",
                       f"provider={model_result.get('provider')}, model={model_result.get('model')}, "
                       f"ready={'yes' if model_ready else 'no'}",
                       status="passed" if model_ready else "blocked")

    # Stage 4: Select compute
    if effective_compute == "gpu" and session.gpu_blocked:
        renderer.preflight("Selecting compute",
                           f"compute=gpu BLOCKED: {session.gpu_blocker or session.gpu_status}",
                           status="blocked")
    else:
        note = ""
        if effective_compute == "local" and session.compute_backend == "gpu" and session.gpu_blocked:
            note = " (GPU blocker ignored for local run)"
        renderer.preflight("Selecting compute", f"compute={effective_compute}{note}",
                           status="passed")

    # Stage 5: Planning experiment
    blockers = session.blocking_setup(compute_override=effective_compute)
    if blockers:
        renderer.preflight("Planning experiment",
                           f"blocked: {', '.join(blockers[:3])}",
                           status="blocked")
    else:
        renderer.preflight("Planning experiment",
                           f"goal={goal[:60] if goal else '(default plan)'}",
                           status="passed")

    # Stage 6: Entering agent
    renderer.preflight("Entering workstation agent",
                       f"compute={effective_compute}, events → events.jsonl, dashboard → :8088",
                       status="passed")


def _execution_blocker_reply(session: SessionState, *, compute: Optional[str] = None) -> bool:
    gaps = session.blocking_setup(compute_override=compute)
    if session.can_execute(compute_override=compute) and not gaps:
        return False
    _agent_reply(
        "Setup needed before execution. EvoMind will not start training until every gate below is clear:\n"
        + "\n".join(f"- {gap}" for gap in gaps),
        title="Setup needed",
    )
    return True


def _delegate_xsci(argv: list[str], root: Path) -> int:
    from . import __main__ as xsci_main
    return xsci_main.main(argv)


# ── Console loop ────────────────────────────────────────────

def _handle_console_command(line: str, root: Path, selected_task: Optional[str]) -> tuple[int, Optional[str], bool]:
    """Test-compatible wrapper around _dispatch_intent."""
    cfg = load_config(root)
    session = SessionState.from_root(root, cfg=cfg)
    session.selected_task = selected_task
    rc, should_exit = _dispatch_intent(line, root, session)
    return rc, session.selected_task, should_exit


def _pick_model(preset: "_Provider") -> str:
    if not preset.models:
        return _safe_input("Default model id (blank = provider default)", "").strip()
    choices = [kaggle_menu.Choice(m) for m in preset.models]
    choices.append(kaggle_menu.Choice("Custom model id (type your own)"))
    idx = kaggle_menu.select("    Default model:", choices, default=0, reader=_safe_input)
    if 0 <= idx < len(preset.models):
        return preset.models[idx]
    return _safe_input("Custom model id (blank = provider default)", "").strip()


def _setup_llm() -> bool:
    """Standalone LLM setup — test-compatible entry point."""
    _setup_step(1, "LLM brain", "Drives planning, code generation, audit, and self-evolution.")
    choices = [kaggle_menu.Choice(p.label, p.hint) for p in _PROVIDERS]
    idx = kaggle_menu.select(
        "    Choose your LLM provider:",
        choices,
        default=0,
        allow_skip=True,
        reader=_safe_input,
    )
    if idx is None or idx < 0:
        return False
    preset = _PROVIDERS[idx]
    base_url = _safe_input("Base URL (blank = provider default)", preset.base_url).strip()
    model = _pick_model(preset)
    key = getpass.getpass(f"  {preset.label} API key (hidden)> ").strip()
    if key:
        save_llm_credentials(
            preset.family,
            api_key=key,
            base_url=base_url or None,
            model=model or None,
            brand=preset.label,
        )
        return True
    return False


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
    if verb in {"auto", "autonomous"}:
        task = (rest[0] if rest else None) or session.selected_task
        if not task:
            _agent_reply("Usage: `auto <task-slug>` — autonomous research mode. EvoMind will inspect data, generate a baseline, train it, and report results.")
            return 1, False
        compute = _infer_compute_override(raw)
        return _auto_research_pipeline(task, root, session, compute=compute), False
    if verb in {"resume", "continue"}:
        task = (rest[0] if rest else None) or session.selected_task
        if not task:
            _agent_reply("No task selected for resume. First `task add <url>` or `use <task>`.")
            return 1, False
        compute = _infer_compute_override(raw)
        _print_preflight_stream(session, compute=compute, goal=session.last_goal or "Continue from best-so-far.")
        if _execution_blocker_reply(session, compute=compute):
            return 1, False
        session.current_mode = MODE_EXECUTING
        rc = _run_agent(task, root, goal=session.last_goal or "Continue from best-so-far.", compute=compute, resume=True)
        session.current_mode = MODE_CHAT
        return rc, False
    if verb in {"evolution", "evolve", "self-evolution"}:
        from .evolution_tracker import EvolutionTracker
        tracker = EvolutionTracker(Path(session.workspace_root) if session.workspace_root else root)
        _agent_reply(tracker.report(), title="Self-Evolution")
        return 0, False
    if verb in {"innovate", "innovation"}:
        return _show_innovations(session, root), False
    if verb in {"watch", "report", "memory", "run", "doctor", "config", "init", "login"}:
        if verb == "run":
            task = (rest[0] if rest else None) or session.selected_task
            if not task:
                _agent_reply("No task selected. `task add <url>` first.")
                return 1, False
            compute = _infer_compute_override(raw)
            _print_preflight_stream(session, compute=compute, goal=session.last_goal or "Start audited evolution loop.")
            if _execution_blocker_reply(session, compute=compute):
                return 1, False
            return _run_agent(task, root, goal=session.last_goal, compute=compute), False
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

    if _wants_model_status(stripped):
        result = TerminalAgent().handle(raw, session, root)
        _agent_reply(result.summary, title="Model status")
        session.last_action = result.action
        return result.rc, False

    intent = classify(stripped)
    # Bug #5: record the turn's action so status/recovery show real history.
    # Specific branches below (TOOL_QUERY, EXECUTION, switch) refine this.
    session.last_action = intent.kind

    # ── TOOL_QUERY: lightweight tool calls (no training) ──────────
    if intent.kind == TOOL_QUERY:
        result = TerminalAgent().handle(raw, session, root)
        _agent_reply(result.summary, title="EvoMind Tool")
        session.last_action = result.action
        return result.rc, False

    if intent.kind == GREETING:
        reply = _conversation()._build_greeting(session)
        _agent_reply(reply, title="EvoMind")
        return 0, False
    if intent.kind == STATUS:
        # Delegate to ConversationAgent's richer status report
        reply = _conversation()._build_status_reply(session, session.missing_setup())
        _agent_reply(reply, title="System status")
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
        # Bug #2: "切换到 X [开始训练]" classifies as EXECUTION. Switch the task
        # first, then decide whether the same utterance also asked to train.
        if _has_switch_cue(raw):
            target = _match_task_in_text(raw, root)
            if target is None:
                names = [s for s, _ in list_tasks(root)]
                _agent_reply(
                    "没找到要切换到的比赛。已注册：" + (", ".join(names) if names else "(无)")
                    + "。用 `task add <kaggle-url>` 注册，或 `use <task>` 选择已有任务。"
                )
                session.last_action = "switch_task"
                return 1, False
            if target != session.selected_task:
                session.selected_task = target
                session.refresh_task_brief(root)
                session.refresh_recent_run(root)
            session.last_action = "switch_task"
            session.last_artifact = target
            if not _mentions_training(raw):
                _agent_reply(
                    f"已切换到 **{target}**。"
                    + (f"\n{session.task_brief}" if session.task_brief else "")
                    + "\n\n告诉我研究目标，或者说“开始训练”。",
                    title="Task switched",
                )
                return 0, False
        if not session.selected_task:
            _agent_reply(
                "还没有选中比赛，所以不会启动训练。"
                "请先用 `competitions` 浏览，或用 `task add <kaggle-url>` 注册并选择一个任务。"
            )
            session.last_action = "training_blocked"
            return 1, False
        compute = _infer_compute_override(raw)
        resume = (intent.payload == "resume")
        _print_preflight_stream(session, compute=compute, goal=raw)
        if _execution_blocker_reply(session, compute=compute):
            session.last_action = "training_blocked"
            return 1, False
        session.current_mode = MODE_EXECUTING
        rc = _run_agent(session.selected_task, root, goal=raw, compute=compute, resume=resume)
        session.current_mode = MODE_CHAT
        session.last_action = "training"
        session.refresh_recent_run(root)
        session.last_artifact = session.recent_run_id or ""
        return rc, False

    session.current_mode = MODE_CHAT
    with thinking("thinking"):
        reply = _conversation().reply(raw, session)
    _agent_reply(reply)
    return 0, False


def _show_innovations(session: SessionState, root: Path) -> int:
    """Display innovation proposals for the current task."""
    from .innovation_engine import InnovationEngine
    from .agent import _load_context
    from .tasks import resolve_task

    if not session.selected_task:
        _agent_reply("No task selected. Select a task first: `use <task-name>`.", title="Innovation")
        return 1

    try:
        task_config = resolve_task(session.selected_task, project_root=root)
        ctx, _ = _load_context(task_config)
        task_type = ctx.task_type
    except Exception:
        task_type = "classification"

    try:
        from research_os.retrospective_memory import RetrospectiveMemoryStore
        from research_os.agent.memory_library import MemoryLibrary
        store = RetrospectiveMemoryStore(root / "experiments" / "evolution" / "retrospective_memory.json")
        library = MemoryLibrary(store)
        engine = InnovationEngine(memory_library=library, workspace_root=root)

        if not engine.ready_for_innovation(task_type):
            _agent_reply(
                f"Not enough experience yet for innovation on task_type={task_type}. "
                f"Complete at least 5 successful experiments with reusable strategies first.",
                title="Innovation"
            )
            return 0

        proposals = engine.propose_innovations(task_type, n=4)

        if not proposals:
            _agent_reply(
                f"No novel combinations found for task_type={task_type}. "
                f"All known strategies have been exhausted or tried on this task.",
                title="Innovation"
            )
            return 0

        lines = [f"💡 Innovation Proposals for task_type={task_type}:\n"]
        for i, p in enumerate(proposals, 1):
            lines.append(f"{i}. {p.strategy_name}")
            lines.append(f"   Novelty: {p.novelty_score:.0%} | Confidence: {p.confidence:.0%}")
            lines.append(f"   Components: {', '.join(p.components)}")
            lines.append(f"   Rationale: {p.rationale}")
            if p.source_tasks:
                lines.append(f"   Based on: {', '.join(p.source_tasks[:3])}")
            lines.append("")

        stats = engine.stats()
        lines.append(f"📊 Innovation stats: {stats['innovations_tried']} tried, "
                     f"{stats['successes']} succeeded, hit rate {stats['hit_rate']}")

        _agent_reply("\n".join(lines), title="Innovation Engine")
        return 0
    except Exception as exc:
        _agent_reply(f"Innovation engine unavailable: {exc}", title="Innovation")
        return 1


def _print_task_list(root: Path, selected_task: Optional[str]) -> None:
    tasks = list_tasks(root)
    if not tasks:
        _agent_reply("No competitions registered; no tasks yet. Use `competitions` to browse Kaggle, then `task add <url>` to register one.")
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
    print("  auto [task]              autonomous research: inspect→baseline→train→report")
    print("  watch / report / memory  engine views")
    print("  dashboard                manage the 8088 workstation")
    print("  official <args...>       Kaggle CLI passthrough")
    print("  exit                     quit")
    print()
    print(_dim("Natural language: /competitions titanic /download /plan /start training"))
    print(f"{_dim('Selected task:')} {selected_task or '(none)'}")


def _print_welcome(session: SessionState, cfg=None) -> None:
    print(logo())
    print()
    _print_dashboard_hint(cfg)
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
    # Bug #4: wire the recovery guard so EVERY turn (chat/plan/execute, not just
    # tool queries) writes a durable recovery section for compaction/restart.
    guard = RecoveryGuard()
    guard.set_state_file(Path(root) / ".xsci" / "recovery_guard.md")
    _print_welcome(session, cfg)
    session.persist(root)
    guard.emit(session, event="SessionStart")
    print()
    while True:
        try:
            line = input("evomind> ").strip()
        except EOFError:
            print("\nbye.")
            return 0
        except KeyboardInterrupt:
            print(_dim("\n(use exit to quit)"))
            continue
        prev_task = session.selected_task
        guard.emit(session, event="UserPromptSubmit")
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
        guard.record_tool(f"{session.last_action or 'turn'}: rc={rc} task={session.selected_task or '(none)'}")
        guard.emit(session, event="PostReply")
        if should_exit:
            print("bye.")
            return rc


def _print_help() -> None:
    print(logo())
    print()
    print("Usage:")
    print("  evomind                     enter the EvoMind research terminal")
    print("  evomind setup               first-run setup wizard")
    print("  evomind ready               show terminal/model/Kaggle/GPU readiness")
    print("  evomind status              same as ready")
    print("  evomind competitions [q]    browse/search Kaggle competitions")
    print("  evomind task add <url>      register a Kaggle/MLE-Bench task")
    print("  evomind download <task>     download competition data")
    print("  evomind agent <task>        open the deep research agent on a task")
    print("  evomind run <task>          run the audited evolution loop")
    print("  evomind watch -f            follow the latest event stream")
    print("  evomind memory              inspect retrospective memory")
    print("  evomind dashboard start     open/manage the 8088 workstation")
    print("  evomind official ...        pass through to the official Kaggle CLI")
    print()
    print("Alias:")
    print("  autokaggle                  compatibility alias for EvoMind")
    print("  kaggle-official ...          direct official Kaggle CLI passthrough")
    print()
    print("Default EvoMind gateway:")
    print("  " + _dashboard_url())
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
            print("Usage: evomind download <task-slug>")
            return 1
        result = kaggle_actions.download_competition_data(slug)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok") else 1
    if cmd in {"evolution", "evolve", "self-evolution"}:
        from .evolution_tracker import EvolutionTracker
        tracker = EvolutionTracker(root)
        print(tracker.report())
        return 0
    if cmd in {"innovate", "innovation"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_innovations(session, root)
    if cmd == "task":
        sub = argv[1].lower() if len(argv) > 1 else "list"
        if sub in {"add", "register"} and len(argv) >= 3:
            slug = _register_task(argv[2], root, force="--force" in argv[3:])
            if slug:
                print(f"registered task: {slug}")
                return 0
            return 1
        if sub in {"list", "ls"}:
            _print_task_list(root, None)
            return 0
        if sub in {"use", "select"} and len(argv) >= 3:
            task = slugify(argv[2])
            try:
                resolve_task(task, project_root=root)
            except FileNotFoundError:
                print(f"task not found: {task}")
                return 1
            print(f"selected task: {task}")
            return 0
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
        print("Setup needs an interactive terminal. Run `evomind setup` in a TTY.")
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
        print("  skipped - you can configure later with `evomind setup`")

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
    _agent_reply("Setup complete. Type a research goal or `competitions` to browse Kaggle tasks.", title="EvoMind Ready")
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
