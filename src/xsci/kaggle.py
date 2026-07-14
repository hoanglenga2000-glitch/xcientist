"""EvoMind shell - Claude-Code-like research terminal.

`evomind` enters the research-agent conversation. `evomind official ...`
passes through to the official Kaggle CLI. Legacy `kaggle` and `autokaggle`
shims may exist as compatibility aliases, but EvoMind is the product name.
"""
from __future__ import annotations

import argparse
import contextlib
import getpass
import io
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from research_os.hpc_policy import HPCPolicyError, require_hpc_compute

from . import kaggle_menu
from .config import (
    GLOBAL_DIR as GLOBAL_DIR,
)
from .config import (
    active_root,
    inject_engine_env,
    is_onboarded,
    load_config,
    mark_onboarded,
    set_global,
)
from .kaggle_conversation import ConversationAgent
from .kaggle_intent import (
    CAPABILITY,
    EXECUTION,
    GREETING,
    MEMORY,
    OFFICIAL,
    PLANNING,
    REPORT,
    STATUS,
    TASK_ADD,
    TASK_USE,
    TOOL_QUERY,
    classify,
)
from .kaggle_session import MODE_CHAT, MODE_EXECUTING, MODE_PLANNING, SessionState
from .kaggle_stream import StageRenderer, thinking
from .login import save_kaggle_api_token, save_llm_credentials
from .recovery_guard import RecoveryGuard
from .tasks import add_task, list_tasks, resolve_task, slugify
from .terminal_agent import TerminalAgent
from .terminal_tools import TerminalTools

_XSCI_COMMANDS = {
    "doctor", "config", "init", "login", "task", "run", "report", "watch",
    "dashboard", "memory", "evolution", "innovate", "scientist", "checkpoint",
    "think", "brief", "decide", "decision", "autopilot", "diagnose",
    "self-audit", "audit-agent", "capability", "intelligence",
    "readiness-report", "launch-readiness", "scientist-readiness", "agent-readiness",
    "causal-diagnosis", "cause-map", "root-cause-map", "causal-graph",
    "strategy", "strategy-optimizer", "priority-plan", "intervention-plan", "decision-matrix",
    "briefing", "context-packet", "scientist-context", "state-briefing",
    "patch-order", "patch-work-order", "repair-order", "code-patch", "code-work-order",
    "engineer", "engineering-loop", "validate-patch", "execute-upgrade",
    "learn", "memory-consolidate", "consolidate-memory", "memory-writeback",
    "innovate-plan", "innovation-backlog", "hypotheses", "proposals",
    "review-hypotheses", "hypothesis-review", "rank-hypotheses", "critique",
    "blueprint", "experiment-blueprint", "plan-experiment", "candidate-blueprint",
    "innovation-feedback", "trial-feedback", "feedback-innovation", "scientist-feedback",
    "situation", "situation-model", "state-model", "scientist-state", "orient",
    "workplan", "roadmap", "agenda", "repair", "fixplan", "diagnose-repair",
    "self-repair", "contract", "execution-contract", "run-contract", "preflight-contract",
    "trace", "steptrace", "steps", "live", "stream", "evidence-stream", "recovery", "recover", "context", "resume-context",
    "ask", "turn", "turn-plan", "tool-plan", "plan-turn", "scientist-turn",
    "loop", "scientist-loop", "autonomous-loop",
    "continuation", "continuation-status", "continue-status", "turn-status",
    "resume-continuation", "resume-safe", "finish-continuation", "finish-safe-tools",
    "continue-tools", "auto-continue-tools",
    "queue", "action-queue", "actions", "next", "next-action", "safe-next",
    "act", "act-next", "continue-scientist",
    "workspace", "code-workspace", "benchmark-agent", "agent-benchmark",
}
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
              "openai", "https://api.openai.com", ("gpt-4o", "gpt-4o-mini", "gpt-4.1")),
    _Provider("OpenAI-compatible gateway", "Qwen/Kimi/GLM/vLLM/LM Studio style gateway.", "openai", "", ()),
)


def _conversation() -> ConversationAgent:
    global _CONVERSATION
    if _CONVERSATION is None:
        _CONVERSATION = ConversationAgent()
    return _CONVERSATION


class _CliArgumentParser(argparse.ArgumentParser):
    """Argument parser that reports errors to the caller instead of exiting."""

    def error(self, message: str) -> None:
        raise ValueError(message)


@contextlib.contextmanager
def _provider_selection(provider: str = ""):
    """Temporarily select one provider without leaking the override to later turns."""

    normalized = str(provider or "").strip().lower()
    keys = ("EVOLUTION_PRIMARY_PROVIDER", "EVOLUTION_PROVIDER_STRICT")
    previous = {key: os.environ.get(key) for key in keys}
    if normalized:
        os.environ["EVOLUTION_PRIMARY_PROVIDER"] = normalized
        os.environ["EVOLUTION_PROVIDER_STRICT"] = "1"
    try:
        yield
    finally:
        if normalized:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _git_top_level(start: Path) -> tuple[Optional[Path], str]:
    """Resolve the real Git top-level for workspace commands."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"git workspace lookup failed: {type(exc).__name__}"
    candidate = completed.stdout.strip()
    if completed.returncode != 0 or not candidate:
        return None, "current directory is not inside a Git repository"
    try:
        return Path(candidate).resolve(), ""
    except OSError:
        return None, "Git returned an invalid workspace path"


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


def _safe_print(text: str = "") -> None:
    from .workspace_agent import _safe_text

    rendered = _safe_text(text, limit=1_000_000)
    try:
        print(rendered)  # lgtm[py/clear-text-logging-sensitive-data] _safe_text redacts credential-bearing values.
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe = rendered.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe)  # lgtm[py/clear-text-logging-sensitive-data] safe is derived only from redacted rendered text.


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
    _safe_print()
    _safe_print(_strong(title))
    for paragraph in text.strip().split("\n"):
        if not paragraph.strip():
            _safe_print()
            continue
        wrapped = textwrap.wrap(paragraph, width=88, replace_whitespace=False) or [""]
        for line in wrapped:
            _safe_print(f"  {line}")


def _has_llm(cfg=None) -> bool:
    cfg = cfg or load_config()
    return bool(
        cfg.get("secrets.anthropic_api_key")
        or cfg.get("secrets.deepseek_api_key")
        or cfg.get("secrets.openai_api_key")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


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


def _effective_compute(session: SessionState, compute: Optional[str] = None) -> str:
    return str(compute or session.compute_backend or "gpu").strip().lower()


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
        model = {
            "anthropic": os.environ.get("CLAUDE_CODE_MODEL"),
            "deepseek": os.environ.get("DEEPSEEK_MODEL"),
            "openai": os.environ.get("OPENAI_MODEL"),
        }.get(family) or "(provider default)"
    family = str(cfg.get("llm.provider") or "").lower()
    base_url = cfg.get(f"llm.{family}_base_url") or "(provider default)"
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
    args = list(sys.argv[1:] if argv is None else argv)
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
    print("  To start: evomind task add https://www.kaggle.com/c/<slug>")
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
    print(_dim(f"  Task: {task} | Compute: {_effective_compute(session, compute)}"))
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
    effective_compute = _effective_compute(session, compute)
    session.current_compute_override = effective_compute
    if effective_compute == "gpu":
        # Data is on remote GPU — check gpu_data_dir config
        gpu_dir = task_info.get("gpu_data_dir", "")
        if not gpu_dir:
            print(_dim(f"  Data: GPU remote path not configured for {task}; planning remains read-only."))
        else:
            print(_dim(f"  Data: GPU remote path={gpu_dir}"))

    # Step 3: Record execution gates without suppressing read-only planning.
    effective_compute = _effective_compute(session, compute)
    blockers = session.blocking_setup(compute_override=effective_compute)
    if blockers:
        print(_dim(
            "  Execution gate blocked; planning remains read-only: "
            + "; ".join(blockers[:3])
        ))

    # Step 4: Ask the scientist decision layer for the next branch/code mode.
    decision_result = TerminalTools.dispatch("research_decision", session, root)
    decision = decision_result.get("decision", {}) if isinstance(decision_result, dict) else {}
    artifact = decision_result.get("artifact_path", "") if isinstance(decision_result, dict) else ""
    if artifact:
        print(_dim(f"  Decision artifact: {artifact}"))
    contract_result = TerminalTools.dispatch("scientist_execution_contract", session, root)
    contract_artifact = contract_result.get("artifact_path", "") if isinstance(contract_result, dict) else ""
    if contract_artifact:
        print(_dim(f"  Execution contract: {contract_artifact}"))
    if contract_result.get("go_no_go") == "no_go":
        _agent_reply(
            "Autonomous research is blocked by the execution contract.\n"
            f"Root causes: {', '.join(contract_result.get('root_causes', []) or ['unknown'])}",
            title="Auto blocked",
        )
        return 1

    # Step 5: Run preflight + training
    goal = str(contract_result.get("enriched_goal") or (
        f"Autonomous research on {task}: action={decision.get('selected_action', 'run_audited_baseline')}; "
        f"branch={decision.get('selected_branch', 'baseline')}; "
        f"code_generation_mode={decision.get('code_generation_mode', 'Base')}. "
        "Produce all workstation artifacts and do not submit to official Kaggle."
    ))
    _print_preflight_stream(session, compute=effective_compute, goal=goal)

    if _execution_blocker_reply(session, compute=effective_compute):
        return 1

    session.current_mode = MODE_EXECUTING
    rc = _run_agent(task, root, goal=goal, compute=effective_compute)
    session.current_mode = MODE_CHAT

    # Step 6: Brief post-run analysis
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

    effective_compute = _effective_compute(session, compute)
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
    effective_compute = _effective_compute(session, compute)
    try:
        require_hpc_compute(effective_compute)
        policy_error = ""
    except HPCPolicyError as exc:
        policy_error = str(exc)
    if effective_compute == "gpu":
        # Data is on remote GPU — check gpu_data_dir config instead
        gpu_dir = task_result.get("gpu_data_dir", "")
        renderer.preflight("Checking data",
                           f"GPU remote path={gpu_dir or '(from task config)'}",
                           status="passed" if gpu_dir else "blocked")
    else:
        renderer.preflight("Checking data",
                           "not evaluated: local training is disabled",
                           status="blocked")

    # Stage 3: Check config
    model_result = TerminalTools.dispatch("model_status", session, root)
    model_ready = model_result.get("ready") and model_result.get("ok")
    renderer.preflight("Checking config",
                       f"provider={model_result.get('provider')}, model={model_result.get('model')}, "
                       f"ready={'yes' if model_ready else 'no'}",
                       status="passed" if model_ready else "blocked")

    # Stage 4: Select compute
    if policy_error:
        renderer.preflight("Selecting compute",
                           f"compute={effective_compute} BLOCKED: {policy_error}",
                           status="blocked")
    elif session.gpu_blocked:
        renderer.preflight("Selecting compute",
                           f"compute=gpu BLOCKED: {session.gpu_blocker or session.gpu_status}",
                           status="blocked")
    else:
        renderer.preflight("Selecting compute", "compute=gpu", status="passed")

    # Stage 5: Planning experiment
    blockers = ([policy_error] if policy_error else []) + session.blocking_setup(
        compute_override=effective_compute
    )
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
                       status="blocked" if blockers else "passed")


def _execution_blocker_reply(session: SessionState, *, compute: Optional[str] = None) -> bool:
    effective_compute = _effective_compute(session, compute)
    try:
        require_hpc_compute(effective_compute)
    except HPCPolicyError as exc:
        _agent_reply(str(exc), title="Setup needed")
        return True
    gaps = session.blocking_setup(compute_override=effective_compute)
    if session.can_execute(compute_override=effective_compute) and not gaps:
        return False
    _agent_reply(
        "Setup needed before execution. EvoMind will not start training until every gate below is clear:\n"
        + "\n".join(f"- {gap}" for gap in gaps),
        title="Setup needed",
    )
    return True


def _show_scientist_checkpoint(session: SessionState, root: Path) -> int:
    from .terminal_events import render_tool_result_as_lines
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_checkpoint", session, root)
    _agent_reply("\n".join(render_tool_result_as_lines(result)), title="Scientist checkpoint")
    return 0 if result.get("ok", True) else 1


def _show_research_decision(session: SessionState, root: Path) -> int:
    from .terminal_events import render_tool_result_as_lines
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("research_decision", session, root)
    _agent_reply("\n".join(render_tool_result_as_lines(result)), title="Research decision")
    return 0 if result.get("ok", True) else 1


def _show_scientist_autopilot(session: SessionState, root: Path) -> int:
    from .terminal_events import TerminalEventEmitter, render_scientist_autopilot_summary
    from .terminal_tools import run_scientist_autopilot

    stage_labels = {
        "system_status": "Observe system",
        "inspect_task": "Inspect task",
        "data_check": "Check data",
        "recent_run": "Read latest run",
        "evolution_status": "Read memory",
        "scientist_checkpoint": "Checkpoint",
        "research_decision": "Choose branch",
        "scientist_hypothesis_review": "Review hypotheses",
        "scientist_experiment_blueprint": "Build experiment blueprint",
        "scientist_workplan": "Build workplan",
        "scientist_repair_plan": "Repair plan",
        "scientist_execution_contract": "Execution contract",
    }
    emitter = TerminalEventEmitter(root, colour=True, jsonl_path=root / ".xsci" / "terminal_events.jsonl")

    def live_event(event: dict) -> None:
        phase = str(event.get("phase") or "")
        tool = str(event.get("tool") or "")
        if phase == "autopilot_start":
            emitter.emit(
                "AI Scientist",
                "starting bounded multi-tool diagnosis; no training or Kaggle submit will start",
                status="running",
            )
            return
        if phase == "tool_started":
            emitter.emit(stage_labels.get(tool, tool), "calling tool", status="running")
            return
        if phase in {"tool_completed", "tool_blocked"}:
            status = "passed" if phase == "tool_completed" else "blocked"
            message = str(event.get("message") or "completed").replace("\n", " ")[:220]
            artifact = str(event.get("artifact_path") or "") or None
            emitter.emit(stage_labels.get(tool, tool), message, status=status, artifact=artifact)
            return
        if phase == "autopilot_complete":
            status = "passed" if event.get("status") == "completed" else "blocked"
            artifact = str(event.get("artifact_path") or "") or None
            emitter.emit("AI Scientist", "diagnosis complete; artifacts persisted", status=status, artifact=artifact)

    result = run_scientist_autopilot(session, root, observer=live_event)
    _agent_reply("\n".join(render_scientist_autopilot_summary(result)), title="Scientist autopilot")
    return 0 if result.get("ok", True) else 1


def _show_scientist_loop(session: SessionState, root: Path) -> int:
    from .terminal_events import TerminalEventEmitter, render_scientist_loop_summary
    from .terminal_tools import run_scientist_loop

    emitter = TerminalEventEmitter(root, colour=True, jsonl_path=root / ".xsci" / "terminal_events.jsonl")

    def live_event(event: dict) -> None:
        phase = str(event.get("phase") or "")
        status = str(event.get("status") or "running")
        message = str(event.get("message") or "").replace("\n", " ")[:260]
        artifact = str(event.get("artifact_path") or "") or None
        stage = {
            "loop_start": "AI Scientist loop",
            "loop_observe": "Observe and decide",
            "loop_next_action": "Safe next action",
            "loop_refresh": "Refresh evidence",
            "loop_repetition_escalation": "Escalate repeated action",
            "loop_complete": "Learn and stop",
        }.get(phase, "AI Scientist loop")
        rendered_status = "passed" if status in {"passed", "completed"} else "blocked" if status == "blocked" else "running"
        emitter.emit(stage, message or phase, status=rendered_status, artifact=artifact)

    result = run_scientist_loop(session, root, observer=live_event)
    _agent_reply("\n".join(render_scientist_loop_summary(result)), title="Scientist loop")
    return 0 if result.get("ok", True) else 1


def _show_scientist_self_audit(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_self_audit_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_self_audit", session, root)
    _agent_reply("\n".join(render_scientist_self_audit_summary(result)), title="Scientist self-audit")
    return 0 if result.get("ok", True) else 1


def _show_scientist_readiness_report(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_readiness_report_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_readiness_report", session, root)
    _agent_reply("\n".join(render_scientist_readiness_report_summary(result)), title="Scientist readiness report")
    return 0 if result.get("ok", True) else 1


def _show_scientist_causal_diagnosis(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_causal_diagnosis_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_causal_diagnosis", session, root)
    _agent_reply("\n".join(render_scientist_causal_diagnosis_summary(result)), title="Scientist causal diagnosis")
    return 0 if result.get("ok", True) else 1


def _show_scientist_strategy_optimizer(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_strategy_optimizer_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_strategy_optimizer", session, root)
    _agent_reply("\n".join(render_scientist_strategy_optimizer_summary(result)), title="Scientist strategy optimizer")
    return 0 if result.get("ok", True) else 1


def _show_scientist_context_packet(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_context_packet_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_context_packet", session, root)
    _agent_reply("\n".join(render_scientist_context_packet_summary(result)), title="Scientist context packet")
    return 0 if result.get("ok", True) else 1


def _show_scientist_upgrade_plan(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_upgrade_plan_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_upgrade_plan", session, root)
    _agent_reply("\n".join(render_scientist_upgrade_plan_summary(result)), title="Scientist upgrade plan")
    return 0 if result.get("ok", True) else 1


def _show_scientist_self_upgrade_loop(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_self_upgrade_loop_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_self_upgrade_loop", session, root)
    _agent_reply("\n".join(render_scientist_self_upgrade_loop_summary(result)), title="Scientist self-upgrade loop")
    return 0 if result.get("ok", True) else 1


def _show_scientist_patch_work_order(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_patch_work_order_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_patch_work_order", session, root)
    _agent_reply("\n".join(render_scientist_patch_work_order_summary(result)), title="Scientist patch work order")
    return 0 if result.get("ok", True) else 1


def _run_scientist_engineering_command(argv: list[str], root: Path) -> int:
    from .scientist_engineering import run_scientist_engineering_loop
    from .terminal_events import render_scientist_engineering_loop_summary

    generate_patch = "--generate" in argv
    json_mode = "--json" in argv
    dashboard_url = "http://127.0.0.1:8088"
    patch_path: Path | None = None
    work_order_path: Path | None = None
    timeout_seconds = 180
    for index, value in enumerate(argv):
        if value == "--patch" and index + 1 < len(argv):
            patch_path = Path(argv[index + 1])
        elif value == "--work-order" and index + 1 < len(argv):
            work_order_path = Path(argv[index + 1])
        elif value == "--dashboard-url" and index + 1 < len(argv):
            dashboard_url = argv[index + 1]
        elif value == "--timeout" and index + 1 < len(argv):
            with contextlib.suppress(ValueError):
                timeout_seconds = max(30, min(900, int(argv[index + 1])))

    cfg = load_config(root)
    inject_engine_env(cfg)
    session = SessionState.from_root(root, cfg=cfg)
    if work_order_path is None:
        TerminalTools.dispatch("scientist_patch_work_order", session, root)
    result = run_scientist_engineering_loop(
        session,
        root,
        work_order_path=work_order_path,
        patch_path=patch_path,
        generate_patch=generate_patch,
        dashboard_url=dashboard_url,
        timeout_seconds=timeout_seconds,
    )
    if json_mode:
        _safe_print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _agent_reply(
            "\n".join(render_scientist_engineering_loop_summary(result)),
            title="Scientist engineering loop",
        )
    return 0 if result.get("ok", False) else 1


def _show_scientist_memory_consolidation(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_memory_consolidation_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_memory_consolidation", session, root)
    _agent_reply("\n".join(render_scientist_memory_consolidation_summary(result)), title="Scientist memory consolidation")
    return 0 if result.get("ok", True) else 1


def _show_scientist_innovation_backlog(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_innovation_backlog_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_innovation_backlog", session, root)
    _agent_reply("\n".join(render_scientist_innovation_backlog_summary(result)), title="Scientist innovation backlog")
    return 0 if result.get("ok", True) else 1


def _show_scientist_hypothesis_review(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_hypothesis_review_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_hypothesis_review", session, root)
    _agent_reply("\n".join(render_scientist_hypothesis_review_summary(result)), title="Scientist hypothesis review")
    return 0 if result.get("ok", True) else 1


def _show_scientist_experiment_blueprint(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_experiment_blueprint_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_experiment_blueprint", session, root)
    _agent_reply("\n".join(render_scientist_experiment_blueprint_summary(result)), title="Scientist experiment blueprint")
    return 0 if result.get("ok", True) else 1


def _show_scientist_innovation_trial_feedback(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_innovation_trial_feedback_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_innovation_trial_feedback", session, root)
    _agent_reply("\n".join(render_scientist_innovation_trial_feedback_summary(result)), title="Scientist innovation feedback")
    return 0 if result.get("ok", True) else 1


def _show_scientist_situation_model(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_situation_model_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_situation_model", session, root)
    _agent_reply("\n".join(render_scientist_situation_model_summary(result)), title="Scientist situation model")
    return 0 if result.get("ok", True) else 1


def _show_scientist_turn_plan(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_turn_plan_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_turn_plan", session, root)
    _agent_reply("\n".join(render_scientist_turn_plan_summary(result)), title="Scientist turn plan")
    return 0 if result.get("ok", True) else 1


def _show_scientist_workplan(session: SessionState, root: Path) -> int:
    from .terminal_events import render_tool_result_as_lines
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_workplan", session, root)
    _agent_reply("\n".join(render_tool_result_as_lines(result)), title="Scientist workplan")
    return 0 if result.get("ok", True) else 1


def _show_scientist_repair_plan(session: SessionState, root: Path) -> int:
    from .terminal_events import render_tool_result_as_lines
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_repair_plan", session, root)
    _agent_reply("\n".join(render_tool_result_as_lines(result)), title="Scientist repair plan")
    return 0 if result.get("ok", True) else 1


def _show_scientist_execution_contract(session: SessionState, root: Path) -> int:
    from .terminal_events import render_tool_result_as_lines
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_execution_contract", session, root)
    _agent_reply("\n".join(render_tool_result_as_lines(result)), title="Scientist execution contract")
    return 0 if result.get("ok", True) else 1


def _show_scientist_step_trace(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_step_trace_timeline
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_step_trace", session, root)
    _agent_reply("\n".join(render_scientist_step_trace_timeline(result)), title="Scientist live timeline")
    return 0 if result.get("ok", True) else 1


def _show_scientist_recovery(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_recovery_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_recovery", session, root)
    _agent_reply("\n".join(render_scientist_recovery_summary(result)), title="Scientist recovery")
    return 0 if result.get("ok", True) else 1


def _show_scientist_action_queue(session: SessionState, root: Path) -> int:
    from .terminal_events import render_tool_result_as_lines
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_action_queue", session, root)
    _agent_reply("\n".join(render_tool_result_as_lines(result)), title="Scientist action queue")
    return 0 if result.get("ok", True) else 1


def _show_scientist_continuation_status(session: SessionState, root: Path) -> int:
    from .terminal_events import render_scientist_continuation_status_summary
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_continuation_status", session, root)
    _agent_reply("\n".join(render_scientist_continuation_status_summary(result)), title="Scientist continuation")
    return 0 if result.get("ok", True) else 1


def _show_scientist_next_action(session: SessionState, root: Path) -> int:
    from .terminal_events import render_tool_result_as_lines
    from .terminal_tools import TerminalTools

    result = TerminalTools.dispatch("scientist_next_action", session, root)
    _agent_reply("\n".join(render_tool_result_as_lines(result)), title="Scientist next action")
    return 0 if result.get("ok", True) else 1


def _show_scientist_continuation_resume(session: SessionState, root: Path) -> int:
    from .terminal_events import TerminalEventEmitter, render_scientist_continuation_resume_summary
    from .terminal_tools import run_scientist_continuation_resume

    emitter = TerminalEventEmitter(root, colour=True, jsonl_path=root / ".xsci" / "terminal_events.jsonl")

    def live_event(event: dict) -> None:
        phase = str(event.get("phase") or "")
        status = str(event.get("status") or "running")
        message = str(event.get("message") or "").replace("\n", " ")[:260]
        artifact = str(event.get("artifact_path") or "") or None
        stage = {
            "continuation_resume_start": "Continuation resume",
            "continuation_resume_step_started": "Safe continuation step",
            "continuation_resume_step_completed": "Safe continuation step",
            "continuation_resume_complete": "Continuation resume",
        }.get(phase, "Continuation resume")
        rendered_status = "passed" if status in {"passed", "closed", "completed"} else "blocked" if status in {"blocked", "blocked_by_gate", "stalled"} else "running"
        emitter.emit(stage, message or phase, status=rendered_status, artifact=artifact)

    result = run_scientist_continuation_resume(session, root, observer=live_event)
    _agent_reply("\n".join(render_scientist_continuation_resume_summary(result)), title="Scientist continuation resume")
    return 0 if result.get("ok", True) else 1


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
    if verb in {"innovate", "innovation", "innovate-plan", "innovation-backlog", "hypotheses", "proposals"}:
        return _show_scientist_innovation_backlog(session, root), False
    if verb in {"review-hypotheses", "hypothesis-review", "rank-hypotheses", "critique"}:
        return _show_scientist_hypothesis_review(session, root), False
    if verb in {"blueprint", "experiment-blueprint", "plan-experiment", "candidate-blueprint"}:
        return _show_scientist_experiment_blueprint(session, root), False
    if verb in {"innovation-feedback", "trial-feedback", "feedback-innovation", "scientist-feedback"}:
        return _show_scientist_innovation_trial_feedback(session, root), False
    if verb in {"situation", "situation-model", "state-model", "scientist-state", "orient"}:
        return _show_scientist_situation_model(session, root), False
    if verb in {"turn-plan", "tool-plan", "plan-turn", "scientist-turn"}:
        return _show_scientist_turn_plan(session, root), False
    if verb in {"scientist", "checkpoint", "think", "brief"}:
        return _show_scientist_checkpoint(session, root), False
    if verb in {"decide", "decision"}:
        return _show_research_decision(session, root), False
    if verb in {"autopilot", "diagnose"}:
        return _show_scientist_autopilot(session, root), False
    if verb in {"loop", "scientist-loop", "autonomous-loop"}:
        return _show_scientist_loop(session, root), False
    if verb in {"self-audit", "audit-agent", "capability", "intelligence"}:
        return _show_scientist_self_audit(session, root), False
    if verb in {"readiness-report", "launch-readiness", "scientist-readiness", "agent-readiness"}:
        return _show_scientist_readiness_report(session, root), False
    if verb in {"causal-diagnosis", "cause-map", "root-cause-map", "causal-graph"}:
        return _show_scientist_causal_diagnosis(session, root), False
    if verb in {"strategy", "strategy-optimizer", "priority-plan", "intervention-plan", "decision-matrix"}:
        return _show_scientist_strategy_optimizer(session, root), False
    if verb in {"briefing", "context-packet", "scientist-context", "state-briefing"}:
        return _show_scientist_context_packet(session, root), False
    if verb in {"upgrade-plan", "agent-upgrade", "capability-upgrade", "upgrade-backlog"}:
        return _show_scientist_upgrade_plan(session, root), False
    if verb in {"self-upgrade", "self-upgrade-loop", "upgrade-loop", "capability-loop"}:
        return _show_scientist_self_upgrade_loop(session, root), False
    if verb in {"patch-order", "patch-work-order", "repair-order", "code-patch", "code-work-order"}:
        return _show_scientist_patch_work_order(session, root), False
    if verb in {"engineer", "engineering-loop", "validate-patch", "execute-upgrade"}:
        return _run_scientist_engineering_command(parts[1:], root), False
    if verb in {"workspace", "code-workspace"}:
        return _run_workspace_command(rest, root), False
    if verb in {"benchmark-agent", "agent-benchmark"}:
        return _run_benchmark_agent_command(rest, root), False
    if verb in {"learn", "memory-consolidate", "consolidate-memory", "memory-writeback"}:
        return _show_scientist_memory_consolidation(session, root), False
    if verb in {"workplan", "roadmap", "agenda"}:
        return _show_scientist_workplan(session, root), False
    if verb in {"repair", "fixplan", "diagnose-repair", "self-repair"}:
        return _show_scientist_repair_plan(session, root), False
    if verb in {"contract", "execution-contract", "run-contract", "preflight-contract"}:
        return _show_scientist_execution_contract(session, root), False
    if verb in {"trace", "steptrace", "steps", "live", "stream", "evidence-stream"}:
        return _show_scientist_step_trace(session, root), False
    if verb in {"recovery", "recover", "context", "resume-context"}:
        return _show_scientist_recovery(session, root), False
    if verb in {"queue", "action-queue", "actions"}:
        return _show_scientist_action_queue(session, root), False
    if verb in {"continuation", "continuation-status", "continue-status", "turn-status"}:
        return _show_scientist_continuation_status(session, root), False
    if verb in {"resume-continuation", "resume-safe", "finish-continuation", "finish-safe-tools", "continue-tools", "auto-continue-tools"}:
        return _show_scientist_continuation_resume(session, root), False
    if verb in {"next", "next-action", "safe-next", "act", "act-next", "continue-scientist"}:
        return _show_scientist_next_action(session, root), False
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
        session.current_compute_override = _effective_compute(session, compute)
        resume = (intent.payload == "resume")
        _print_preflight_stream(session, compute=compute, goal=raw)
        if _execution_blocker_reply(session, compute=compute):
            session.last_action = "training_blocked"
            return 1, False
        decision_result = TerminalTools.dispatch("research_decision", session, root)
        artifact = decision_result.get("artifact_path", "") if isinstance(decision_result, dict) else ""
        if artifact:
            print(_dim(f"  Decision artifact: {artifact}"))
        contract_result = TerminalTools.dispatch("scientist_execution_contract", session, root)
        contract_artifact = contract_result.get("artifact_path", "") if isinstance(contract_result, dict) else ""
        if contract_artifact:
            print(_dim(f"  Execution contract: {contract_artifact}"))
        if contract_result.get("go_no_go") == "no_go":
            _agent_reply(
                "Execution contract returned no_go, so EvoMind will not start training.\n"
                f"Root causes: {', '.join(contract_result.get('root_causes', []) or ['unknown'])}",
                title="Execution blocked",
            )
            session.last_action = "training_blocked"
            return 1, False
        enriched_goal = f"{raw}\n\n{contract_result.get('enriched_goal', '')}".strip()
        session.current_mode = MODE_EXECUTING
        rc = _run_agent(session.selected_task, root, goal=enriched_goal, compute=compute, resume=resume)
        session.current_mode = MODE_CHAT
        session.last_action = "training"
        session.refresh_recent_run(root)
        session.last_artifact = session.recent_run_id or ""
        return rc, False

    session.current_mode = MODE_CHAT
    result = TerminalAgent().handle_scientist_turn(raw, session, root)
    _agent_reply(result.summary, title="AI Scientist Turn")
    session.last_action = result.action
    if result.artifacts:
        session.last_artifact = result.artifacts[-1]
    return result.rc, False


def _show_innovations(session: SessionState, root: Path) -> int:
    """Display innovation proposals for the current task."""
    from .agent import _load_context
    from .innovation_engine import InnovationEngine
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
        from research_os.agent.memory_library import MemoryLibrary
        from research_os.retrospective_memory import RetrospectiveMemoryStore
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
    print("  scientist / think        show Observe→Analyze→Propose→Gate→Act checkpoint")
    print("  decide                   persist next experiment branch/code-mode decision")
    print("  loop                     run bounded safe Scientist loop and write lesson")
    print("  self-audit               score EvoMind capabilities and write upgrade backlog")
    print("  readiness-report         write unified capability/execution/claim readiness report")
    print("  causal-diagnosis         build symptom/root-cause/evidence/intervention graph")
    print("  strategy                 rank safe interventions by impact/evidence/cost/risk/gates")
    print("  briefing                 build per-turn Scientist context packet")
    print("  upgrade-plan             convert self-audit backlog into an engineering plan")
    print("  self-upgrade             create a safe work order for the next P0 capability upgrade")
    print("  patch-order              turn latest failure/blocker evidence into a code-agent patch work order")
    print("  engineer [--generate]    validate a patch in an isolated Git worktree; never auto-merge")
    print("  workspace <goal>         run bounded search/read/patch/test/diff loop on the current Git repo")
    print("  benchmark-agent          run one hidden-oracle workspace case (use --all for all 12)")
    print("  memory-consolidate       write Scientist lessons into retrospective memory")
    print("  innovate-plan            generate memory-guided proposal backlog before training")
    print("  review-hypotheses        rank proposed research hypotheses before training")
    print("  blueprint                generate gated experiment blueprint from reviewed hypothesis")
    print("  innovation-feedback      write hypothesis/blueprint gate outcome into innovation memory")
    print("  situation                synthesize evidence, blockers, uncertainty, strategy, and memory")
    print("  live                     show recent Scientist tool/gate/artifact timeline")
    print("  recovery                 rebuild restart/context recovery snapshot")
    print("  queue / continuation     show action queue or incomplete-turn continuation status")
    print("  next                     execute safe read-only next action")
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
    print("  evomind ask \"goal\"          run one auditable AI Scientist turn")
    print("  evomind turn \"goal\"         alias for `evomind ask`")
    print("  evomind run <task>          run the audited evolution loop")
    print("  evomind scientist           show the AI Scientist checkpoint")
    print("  evomind autopilot           live multi-tool AI Scientist diagnosis chain")
    print("  evomind loop                run bounded safe Scientist loop and write lesson")
    print("  evomind self-audit          score EvoMind capabilities and write upgrade backlog")
    print("  evomind readiness-report    write unified capability/execution/claim readiness report")
    print("  evomind causal-diagnosis    build symptom/root-cause/evidence/intervention graph")
    print("  evomind strategy            rank safe interventions by impact/evidence/cost/risk/gates")
    print("  evomind briefing            build per-turn Scientist context packet")
    print("  evomind upgrade-plan        convert self-audit backlog into engineering plan")
    print("  evomind self-upgrade        create a safe work order for the next P0 capability upgrade")
    print("  evomind patch-order         create a code-agent patch work order from latest evidence")
    print("  evomind engineer            validate the latest patch in an isolated Git worktree")
    print("  evomind engineer --generate ask Code Agent for a diff, validate it, and stop before merge")
    print("  evomind workspace \"goal\"  build a tested candidate diff from the current Git repo")
    print("  evomind benchmark-agent    run one real hidden-oracle workspace case")
    print("  evomind benchmark-agent --all run the full 12-case behavior suite")
    print("  evomind memory-consolidate  write Scientist lessons into retrospective memory")
    print("  evomind innovate-plan       generate memory-guided proposal backlog before training")
    print("  evomind review-hypotheses   rank proposed hypotheses against evidence/gates")
    print("  evomind blueprint           turn reviewed hypothesis into gated experiment plan")
    print("  evomind innovation-feedback record gate feedback into innovation memory")
    print("  evomind situation           synthesize the current AI Scientist situation model")
    print("  evomind decide              persist the next audited experiment decision")
    print("  evomind workplan            persist the multi-step Scientist workplan")
    print("  evomind repair              build the self-repair/root-cause plan")
    print("  evomind contract            build the pre-execution go/no-go contract")
    print("  evomind live                show the recent Scientist tool/gate/artifact timeline")
    print("  evomind trace               alias for `evomind live`")
    print("  evomind recovery            build a restart/context recovery snapshot")
    print("  evomind queue               show the Scientist action queue")
    print("  evomind continuation-status show remaining tools from an incomplete Scientist turn")
    print("  evomind resume-continuation run remaining safe continuation tools until closed/gated")
    print("  evomind next                execute the next safe read-only action or block at gates")
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
    if cmd in {"workspace", "code-workspace"}:
        return _run_workspace_command(argv[1:], root)
    if cmd in {"benchmark-agent", "agent-benchmark"}:
        return _run_benchmark_agent_command(argv[1:], root)
    if cmd in {"ask", "turn"}:
        return _run_scientist_turn_command(argv[1:], root)
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
        _safe_print(tracker.report())
        return 0
    if cmd in {"innovate", "innovation", "innovate-plan", "innovation-backlog", "hypotheses", "proposals"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_innovation_backlog(session, root)
    if cmd in {"review-hypotheses", "hypothesis-review", "rank-hypotheses", "critique"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_hypothesis_review(session, root)
    if cmd in {"blueprint", "experiment-blueprint", "plan-experiment", "candidate-blueprint"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_experiment_blueprint(session, root)
    if cmd in {"innovation-feedback", "trial-feedback", "feedback-innovation", "scientist-feedback"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_innovation_trial_feedback(session, root)
    if cmd in {"situation", "situation-model", "state-model", "scientist-state", "orient"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_situation_model(session, root)
    if cmd in {"turn-plan", "tool-plan", "plan-turn", "scientist-turn"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_turn_plan(session, root)
    if cmd in {"scientist", "checkpoint", "think", "brief"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_checkpoint(session, root)
    if cmd in {"decide", "decision"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_research_decision(session, root)
    if cmd in {"autopilot", "diagnose"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_autopilot(session, root)
    if cmd in {"loop", "scientist-loop", "autonomous-loop"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_loop(session, root)
    if cmd in {"self-audit", "audit-agent", "capability", "intelligence"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_self_audit(session, root)
    if cmd in {"readiness-report", "launch-readiness", "scientist-readiness", "agent-readiness"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_readiness_report(session, root)
    if cmd in {"causal-diagnosis", "cause-map", "root-cause-map", "causal-graph"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_causal_diagnosis(session, root)
    if cmd in {"strategy", "strategy-optimizer", "priority-plan", "intervention-plan", "decision-matrix"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_strategy_optimizer(session, root)
    if cmd in {"briefing", "context-packet", "scientist-context", "state-briefing"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_context_packet(session, root)
    if cmd in {"upgrade-plan", "agent-upgrade", "capability-upgrade", "upgrade-backlog"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_upgrade_plan(session, root)
    if cmd in {"self-upgrade", "self-upgrade-loop", "upgrade-loop", "capability-loop"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_self_upgrade_loop(session, root)
    if cmd in {"patch-order", "patch-work-order", "repair-order", "code-patch", "code-work-order"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_patch_work_order(session, root)
    if cmd in {"engineer", "engineering-loop", "validate-patch", "execute-upgrade"}:
        return _run_scientist_engineering_command(argv[1:], root)
    if cmd in {"learn", "memory-consolidate", "consolidate-memory", "memory-writeback"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_memory_consolidation(session, root)
    if cmd in {"workplan", "roadmap", "agenda"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_workplan(session, root)
    if cmd in {"repair", "fixplan", "diagnose-repair", "self-repair"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_repair_plan(session, root)
    if cmd in {"contract", "execution-contract", "run-contract", "preflight-contract"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_execution_contract(session, root)
    if cmd in {"trace", "steptrace", "steps", "live", "stream", "evidence-stream"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_step_trace(session, root)
    if cmd in {"recovery", "recover", "context", "resume-context"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_recovery(session, root)
    if cmd in {"queue", "action-queue", "actions"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_action_queue(session, root)
    if cmd in {"continuation", "continuation-status", "continue-status", "turn-status"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_continuation_status(session, root)
    if cmd in {"resume-continuation", "resume-safe", "finish-continuation", "finish-safe-tools", "continue-tools", "auto-continue-tools"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_continuation_resume(session, root)
    if cmd in {"next", "next-action", "safe-next", "act", "act-next", "continue-scientist"}:
        session = SessionState.from_root(root, cfg=load_config(root))
        return _show_scientist_next_action(session, root)
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


def _run_workspace_command(argv: list[str], state_root: Path) -> int:
    parser = _CliArgumentParser(prog="evomind workspace", add_help=False)
    parser.add_argument("--help", "-h", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-diff", action="store_true")
    parser.add_argument("--provider", choices=("anthropic", "deepseek", "openai"), default="")
    parser.add_argument("--root", default="")
    parser.add_argument("--allow", "--allow-path", dest="allowed_paths", action="append", default=[])
    parser.add_argument("--check", dest="checks", action="append", default=[])
    parser.add_argument("--max-steps", type=int, default=18)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("goal", nargs="*")
    try:
        args = parser.parse_args(argv)
    except ValueError as exc:
        print(f"evomind workspace: {exc}")
        print("Usage: evomind workspace [--provider NAME] [--allow PATH] [--check CMD] <goal>")
        return 1

    if args.help:
        print("Usage: evomind workspace [options] <goal>")
        print("  --provider anthropic|deepseek|openai  use only the selected configured provider")
        print("  --allow PATH                   restrict edits to a path; repeat as needed")
        print("  --check CMD                    allow and require a test/check command")
        print("  --max-steps N                  bounded model/tool decisions (1-40)")
        print("  --timeout SECONDS              total workspace loop budget")
        print("  --json                         machine-readable result")
        print("  --show-diff                    print the audited candidate diff")
        return 0

    goal = " ".join(args.goal).strip()
    if not goal and not sys.stdin.isatty():
        with contextlib.suppress(OSError):
            goal = sys.stdin.read().strip()
    if not goal:
        print("Usage: evomind workspace [options] <goal>")
        return 1

    start = Path(args.root).expanduser() if args.root else Path.cwd()
    workspace_root, lookup_error = _git_top_level(start.resolve())
    if workspace_root is None:
        payload = {
            "ok": False,
            "completed": False,
            "status": "blocked_workspace_not_git",
            "stop_reason": "workspace_not_git",
            "message": lookup_error,
            "goal": goal,
            "workspace_root": str(start.resolve()),
            "human_gate": "no_candidate_created",
        }
        if args.json:
            _safe_print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _agent_reply(lookup_error, title="Workspace blocked")
        return 1

    max_steps = max(1, min(int(args.max_steps), 40))
    timeout_seconds = max(5, min(int(args.timeout), 3600))
    checks = list(dict.fromkeys(str(item).strip() for item in args.checks if str(item).strip()))
    if not checks:
        checks = ["git diff --check"]
    allowed_paths = list(dict.fromkeys(str(item).strip() for item in args.allowed_paths if str(item).strip()))

    cfg = load_config(workspace_root)
    inject_engine_env(cfg)
    with _provider_selection(args.provider):
        from research_os.agent.messaging import AgentMessageClient

        from .workspace_agent import WorkspaceAgentLimits, run_workspace_agent, sanitize_workspace_result

        client = AgentMessageClient(max_retries=1, timeout=min(120, timeout_seconds))
        if not client.is_available():
            result = {
                "ok": False,
                "completed": False,
                "status": "blocked_provider_unavailable",
                "stop_reason": "provider_unavailable",
                "message": "No configured model is available for the selected provider.",
                "goal": goal,
                "provider": args.provider,
                "human_gate": "configure_provider_before_workspace_run",
                "final_diff": "",
            }
        else:
            result = run_workspace_agent(
                workspace_root,
                goal=goal,
                client=client,
                acceptance_commands=checks,
                allowed_edit_paths=allowed_paths,
                require_post_patch_read=True,
                allow_dynamic_behavioral_tests=False,
                limits=WorkspaceAgentLimits(
                    max_steps=max_steps,
                    command_timeout_seconds=min(120, timeout_seconds),
                    total_timeout_seconds=timeout_seconds,
                ),
            )

    payload = dict(result)
    payload.setdefault("workspace_root", str(workspace_root))
    payload.setdefault("goal", goal)
    payload.setdefault("requested_provider", args.provider or "configured_default")
    payload = sanitize_workspace_result(payload)
    if args.json:
        _safe_print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        lines = [
            f"Status: {payload.get('status') or 'unknown'}",
            f"Workspace: {workspace_root}",
            f"Provider: {payload.get('provider') or args.provider or 'unavailable'}",
        ]
        if payload.get("model"):
            lines.append(f"Model: {payload['model']}")
        if payload.get("candidate_diff_path"):
            lines.append(f"Candidate diff: {payload['candidate_diff_path']}")
        if payload.get("artifact_path"):
            lines.append(f"Evidence: {payload['artifact_path']}")
        if payload.get("human_gate"):
            lines.append(f"Gate: {payload['human_gate']}")
        if payload.get("message"):
            lines.append(f"Reason: {payload['message']}")
        if payload.get("summary"):
            lines.append(f"Summary: {payload['summary']}")
        _agent_reply("\n".join(lines), title="Workspace candidate")
        if args.show_diff and payload.get("final_diff"):
            _safe_print(str(payload["final_diff"]))
    return 0 if bool(payload.get("ok")) else 1


def _run_benchmark_agent_command(argv: list[str], state_root: Path) -> int:
    parser = _CliArgumentParser(prog="evomind benchmark-agent", add_help=False)
    parser.add_argument("--help", "-h", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--provider", choices=("anthropic", "deepseek", "openai"), default="")
    parser.add_argument("--case", dest="case_ids", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--timeout", type=int, default=180)
    try:
        args = parser.parse_args(argv)
    except ValueError as exc:
        print(f"evomind benchmark-agent: {exc}")
        print("Usage: evomind benchmark-agent [--case ID | --all] [--provider NAME] [--json]")
        return 1

    if args.help:
        print("Usage: evomind benchmark-agent [options]")
        print("  --case ID                      run one public task id; repeat as needed")
        print("  --all                          run the full 12-case hidden-oracle suite")
        print("  --provider anthropic|deepseek|openai  use only the selected configured provider")
        print("  --timeout SECONDS              hard child-process budget per case")
        print("  --seed N                       deterministic fixture seed")
        print("  --json                         machine-readable report")
        return 0
    if args.all and args.case_ids:
        print("Choose either --case or --all, not both.")
        return 1

    case_ids = None if args.all else (args.case_ids or ["retrieval_exact_release_token"])
    timeout_seconds = max(10, min(int(args.timeout), 900))
    report_path = state_root / ".xsci" / "agentic_capability_benchmark.json"
    cfg = load_config(state_root)
    inject_engine_env(cfg)
    selected_provider = str(args.provider or cfg.get("llm.provider") or "").strip().lower()
    if selected_provider not in {"anthropic", "deepseek", "openai"}:
        selected_provider = ""
    with _provider_selection(selected_provider):
        from .agentic_capability_benchmark import run_workspace_agent_benchmark

        report = run_workspace_agent_benchmark(
            workspace_root=state_root,
            case_ids=case_ids,
            seed=int(args.seed),
            timeout_seconds=timeout_seconds,
            provider=selected_provider or None,
            report_path=report_path,
            limits={
                "max_steps": 24,
                "max_patch_attempts": 5,
                "max_test_runs": 8,
                "command_timeout_seconds": min(120, timeout_seconds),
                "total_timeout_seconds": timeout_seconds,
            },
        )

    if args.json:
        _safe_print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        rate = float(report.get("task_success_rate") or 0.0)
        lines = [
            f"Execution: {report.get('execution_status') or 'unknown'}",
            f"Provider: {report.get('provider') or args.provider or 'unavailable'}",
            f"Cases: {report.get('passed_cases', 0)}/{report.get('cases_run', 0)} passed",
            f"Success rate: {rate:.1%}",
            f"Scope violations: {report.get('scope_violations', 0)}",
            f"Unsupported claims: {report.get('unsupported_claims', 0)}",
            f"Report: {report.get('report_path') or report_path}",
        ]
        if report.get("message"):
            lines.append(f"Reason: {report['message']}")
        _agent_reply("\n".join(lines), title="Agent behavior benchmark")

    completed = report.get("execution_status") == "completed"
    clean = int(report.get("failed_cases") or 0) == 0
    clean = clean and int(report.get("scope_violations") or 0) == 0
    clean = clean and int(report.get("unsupported_claims") or 0) == 0
    return 0 if completed and clean and int(report.get("cases_run") or 0) > 0 else 1


def _run_scientist_turn_command(argv: list[str], root: Path) -> int:
    """Run one non-interactive AI Scientist turn from the command line.

    This gives installed users a Claude-Code-like one-shot command:
    `evomind ask "inspect the current task and tell me the next safe step"`.
    It deliberately reuses the safe Scientist Turn path, so it writes evidence
    artifacts and stops before training, downloads, or official submission.
    """
    args = list(argv)
    json_mode = False
    max_tools = 4
    cleaned: list[str] = []
    i = 0
    while i < len(args):
        item = args[i]
        if item == "--json":
            json_mode = True
            i += 1
            continue
        if item == "--max-tools":
            if i + 1 >= len(args):
                print("Usage: evomind ask [--json] [--max-tools N] <research-goal>")
                return 1
            try:
                max_tools = max(1, min(8, int(args[i + 1])))
            except ValueError:
                print("Usage: evomind ask [--json] [--max-tools N] <research-goal>")
                return 1
            i += 2
            continue
        cleaned.append(item)
        i += 1

    prompt = " ".join(cleaned).strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("Usage: evomind ask [--json] [--max-tools N] <research-goal>")
        print('Example: evomind ask "analyze the selected task and propose the next safe experiment"')
        return 1

    cfg = load_config(root)
    inject_engine_env(cfg)
    session = SessionState.from_root(root, cfg=cfg)
    if json_mode:
        with contextlib.redirect_stdout(io.StringIO()):
            result = TerminalAgent().handle_scientist_turn(prompt, session, root, max_tools=max_tools)
    else:
        result = TerminalAgent().handle_scientist_turn(prompt, session, root, max_tools=max_tools)
    session.last_action = result.action
    session.last_goal = prompt
    if result.artifacts:
        session.last_artifact = result.artifacts[-1]
    session.persist(root)

    if json_mode:
        turn_artifact = root / ".xsci" / "scientist_terminal_turn.json"
        turn_payload = {}
        if turn_artifact.exists():
            with contextlib.suppress(Exception):
                turn_payload = json.loads(turn_artifact.read_text(encoding="utf-8"))
        payload = {
            "ok": result.rc == 0 and not result.blocked,
            "action": result.action,
            "selected_task": result.selected_task or "",
            "summary": result.summary,
            "artifacts": result.artifacts,
            "blocked": result.blocked,
            "execution_ready": bool(turn_payload.get("execution_ready")),
            "execution_blocked": bool(turn_payload.get("execution_blocked")),
            "blocking_gates": turn_payload.get("blocking_gates") or [],
            "scientific_critique": turn_payload.get("scientific_critique") or {},
            "requirement_ledger": turn_payload.get("requirement_ledger") or {},
            "tool_budget": turn_payload.get("tool_budget") or {},
            "deferred_tools": turn_payload.get("deferred_tools") or [],
            "must_run_deferred_tools": turn_payload.get("must_run_deferred_tools") or [],
            "budget_exhausted": bool(turn_payload.get("budget_exhausted")),
            "continuation": turn_payload.get("continuation") or {},
            "continuation_artifact_path": turn_payload.get("continuation_artifact_path") or "",
            "parity_lifecycle": turn_payload.get("parity_lifecycle") or {},
            "parity_loop_artifact": turn_payload.get("parity_loop_artifact") or "",
            "scientist_reasoning_synthesis": turn_payload.get("reasoning_synthesis") or {},
            "answer_markdown": turn_payload.get("answer_markdown") or "",
            "reasoning_quality": turn_payload.get("reasoning_quality") or {},
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        }
        _safe_print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _agent_reply(result.summary, title="AI Scientist Turn")
    return result.rc


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
        key = getpass.getpass("  Anthropic API key (hidden)> ").strip()
        if key:
            save_llm_credentials("anthropic", api_key=key)
            set_global("llm", "provider", "anthropic")
            print("  saved Anthropic API key to secure storage")
    elif choice == "2":
        key = getpass.getpass("  DeepSeek API key (hidden)> ").strip()
        if key:
            save_llm_credentials("deepseek", api_key=key)
            set_global("llm", "provider", "deepseek")
            print("  saved DeepSeek API key to secure storage")
    elif choice == "3":
        key = getpass.getpass("  OpenAI API key (hidden)> ").strip()
        if key:
            save_llm_credentials("openai", api_key=key)
            set_global("llm", "provider", "openai")
            print("  saved OpenAI API key to secure storage")
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
        print("  skipped - read-only/offline inspection remains available.")

    # Step 3: Compute
    _setup_step(3, "Compute backend", "Release builds use the gated SSH/HPC GPU runtime.")
    set_global("compute", "backend", "gpu")
    print("  set compute backend to gpu (HPC-only release policy)")

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
