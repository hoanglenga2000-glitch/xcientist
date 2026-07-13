"""`xsci agent` - the deep research agent as an interactive terminal.

This is the "AI Scientist" surface: the model drives the research loop, calling
research_os tools turn by turn. It reuses ``engine.build_plan`` for the exact
same task resolution, credential injection, and Runner selection as ``xsci run``,
then hands control to ``research_os.agent.AgentSession`` instead of the fixed
evolution loop.

Two modes:
  * ``xsci agent <task>`` - interactive REPL: type a goal, watch the agent work,
    then type the next goal (state/tree/memory persist across goals).
  * ``xsci agent <task> --goal "..."`` - one-shot: run a single goal and exit.

Live events stream to the terminal and to ``<exp_dir>/events.jsonl``, which the
:8088 dashboard reads, so the same run is visible in both surfaces at once.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from .config import Config, load_config
from .engine import (
    RunPlan,
    _load_context,
    _record_evolution_summary,
    _workspace_root_from_exp_dir,
    build_plan,
)
from .scientist_execution_gate import (
    build_execution_contract_for_task,
    build_execution_gate_decision,
    contract_blocks_training,
    render_execution_contract_lines,
)


def _make_runner(plan: RunPlan, data: dict[str, Any]):
    """Build the Runner for this plan - identical selection to execute_plan."""
    from research_os.gpu_runner import GPURunner, GPURunnerConfig
    from research_os.hpc_policy import require_hpc_compute

    require_hpc_compute(plan.compute)
    dirname = data.get("remote_data_dirname") or plan.task_name
    return GPURunner(dirname, config=GPURunnerConfig(timeout=int(data.get("timeout", 1800))))


def _latest_run_dir(plan: RunPlan) -> Optional["Path"]:
    """Find the newest prior run dir for this task (a resume target).

    A run dir is ``<experiments>/evolution/<task>_<compute>_<stamp>`` and only
    counts as resumable if it has a ``search_graph.json`` (the audited state to
    continue). Returns None when there's nothing to resume."""

    base = plan.exp_dir.parent
    if not base.is_dir():
        return None
    prefix = f"{plan.task_name}_"
    candidates = [
        d for d in base.iterdir()
        if d.is_dir() and d.name.startswith(prefix) and (d / "search_graph.json").exists()
    ]
    if not candidates:
        return None
    # dir names end in <YYYYmmdd_HHMMSS>, so the last 15 chars sort chronologically.
    return max(candidates, key=lambda d: d.name[-15:])


def _build_session(plan: RunPlan, *, quiet: bool, mcgs: bool = True, budget: int = 40,
                   resume: bool = False, event_renderer: Optional[Callable[[dict], None]] = None):
    """Assemble an AgentSession (toolbox + client) from a resolved plan.

    ``event_renderer``, when given, replaces the built-in raw one-line ``_render``
    as the session's ``on_event`` — the terminal passes a staged renderer here so
    the run reads as a research narrative instead of a log dump. The JSONL sink is
    unaffected either way (events always persist for the dashboard).

    When ``resume`` is set, the plan's ``exp_dir`` is repointed at the newest prior
    run dir for this task, the toolbox rehydrates that run's search graph, and the
    session continues its ledger conversation instead of starting fresh."""
    from research_os import events as ev
    from research_os.agent import AgentSession, ResearchToolbox
    from research_os.retrospective_memory import RetrospectiveMemoryStore

    ctx, data = _load_context(plan.task_config)
    resume_dir = _latest_run_dir(plan) if resume else None
    if resume_dir is not None:
        plan.exp_dir = resume_dir  # continue in the SAME dir (ledger + graph live here)
    plan.exp_dir.mkdir(parents=True, exist_ok=True)
    runner = _make_runner(plan, data)
    memory = RetrospectiveMemoryStore(plan.exp_dir.parent / "retrospective_memory.json")

    # Layered co-governance: the MCGS brain owns topology by default.
    # --no-mcgs drops to model-chosen parents for debugging.
    selector = None
    if mcgs:
        from research_os.mcgs_selector import MCGSSelector

        selector = MCGSSelector(total_steps=max(1, budget))
    toolbox = ResearchToolbox(
        ctx,
        data_dir=plan.data_dir,
        work_dir=plan.exp_dir,
        runner=runner,
        memory=memory,
        selector=selector,
    )

    # Resume: rehydrate the audited research state BEFORE the session starts, so the
    # restored graph is what _finalize re-exports (never overwritten with an empty one).
    resumed = False
    if resume_dir is not None:
        try:
            info = toolbox.restore_from(resume_dir)
            resumed = True
            if not quiet:
                print(f"[resume] {resume_dir.name}: restored {info['restored_nodes']} node(s), "
                      f"best={info['best_exp_id']}, next={info['next_exp_id']}")
        except (FileNotFoundError, ValueError, OSError) as exc:
            if not quiet:
                print(f"[resume] could not restore {resume_dir.name} ({exc}); starting fresh")

    def _render(event: dict) -> None:
        if quiet:
            return
        try:
            print(ev.format_event(event), flush=True)
        except Exception:  # noqa: BLE001 - never let rendering break a run
            pass

    on_event = event_renderer if event_renderer is not None else _render

    session = AgentSession(
        context=ctx,
        toolbox=toolbox,
        exp_dir=plan.exp_dir,
        on_event=on_event,
        run_meta={"compute": plan.compute, "exp_dir": str(plan.exp_dir)},
        resume=resumed,
    )
    return session


def run_agent(
    task: str,
    *,
    goal: Optional[str] = None,
    compute: Optional[str] = None,
    data_dir: str = "",
    quiet: bool = False,
    mcgs: bool = True,
    resume: bool = False,
    cfg: Optional[Config] = None,
    event_renderer: Optional[Callable[[dict], None]] = None,
    show_plan: bool = True,
) -> int:
    """Entry point for the `agent` subcommand.

    ``show_plan`` / ``event_renderer`` default to the standalone ``xsci agent``
    behaviour (print the full plan + banner, stream raw event lines). The Kaggle
    terminal opts out (``show_plan=False`` + a staged ``event_renderer``) so the
    run reads as a research narrative rather than a wall of plan text plus raw
    events. The hard no-key gates below are identical in both paths.
    """
    from .tasks import resolve_task

    try:
        task_config = resolve_task(task)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1

    cfg = cfg or load_config()
    plan = build_plan(task_config, cfg=cfg, compute=compute, data_dir=data_dir, mcgs=False)
    from research_os.hpc_policy import HPCPolicyError, require_hpc_compute

    try:
        require_hpc_compute(plan.compute)
    except HPCPolicyError as exc:
        print(f"\nrefusing to run: {exc}")
        return 1
    if show_plan:
        print(plan.render())

    # Same hard gate as `xsci run`: no LLM key -> stop before spending anything.
    blocking = [w for w in plan.warnings if "no LLM key" in w]
    if blocking:
        print("\nrefusing to run: " + "; ".join(blocking))
        return 1

    contract = build_execution_contract_for_task(
        task,
        root=_workspace_root_from_exp_dir(plan.exp_dir),
        cfg=cfg,
        compute=plan.compute,
        goal=goal or "xsci agent",
    )
    if show_plan:
        print()
        print("\n".join(render_execution_contract_lines(contract)))
    gate_decision = build_execution_gate_decision(contract, require_model_ready=False)
    if contract_blocks_training(contract, require_model_ready=False):
        print(
            "\nrefusing to run: " + str(gate_decision.get("message") or "Scientist execution gate blocked training.")
        )
        safe_next = gate_decision.get("safe_next_commands") or []
        if safe_next:
            print("safe next: " + " | ".join(str(item) for item in safe_next[:3]))
        return 1
    enriched_goal = str(contract.get("enriched_goal") or "").strip()
    if enriched_goal:
        goal = f"{goal or ''}\n\n{enriched_goal}".strip()

    from research_os.agent import AgentMessageClient

    if not AgentMessageClient().is_available():
        print(
            "\nrefusing to run: the deep agent needs an Anthropic-compatible key "
            "(tool-use runs against ANTHROPIC_BASE_URL). Run `xsci login --provider anthropic`."
        )
        return 1

    session = _build_session(plan, quiet=quiet, mcgs=mcgs, resume=resume,
                             event_renderer=event_renderer)
    if show_plan:
        print(_banner(session, plan, mcgs=mcgs))

    # When a staged renderer owns the view it prints its own Report footer on
    # RUN_END, so skip the raw [summary]/[artifacts] lines to avoid doubling up.
    quiet_summary = event_renderer is not None
    if goal:
        return _run_once(
            session, goal,
            quiet_summary=quiet_summary,
            task_name=str(getattr(plan, "task_name", task)),
        )
    return _repl(session)


def _short_middle(value: object, limit: int = 58) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return "..." + text[-(limit - 3):]


def _banner(session, plan: RunPlan, *, mcgs: bool) -> str:
    """Opening card printed once before the first goal."""
    ctx = session.context
    bar = "-" * 78
    compute = "remote HPC/GPU"
    return "\n".join([
        "",
        "EvoMind / XCIENTIST Research Agent",
        "Provider : Anthropic-compatible research gateway",
        "Mode     : MLEvolve search + XCIENTIST audit + memory",
        bar,
        f"Task     : {ctx.task_name}",
        f"Problem  : {ctx.modality}/{ctx.task_type}",
        f"Metric   : {ctx.metric} ({ctx.metric_direction})",
        f"Compute  : {compute} | MCGS {'on' if mcgs else 'off'}",
        f"Data     : {_short_middle(plan.data_dir or '(declared schema only)')}",
        f"Artifacts: {_short_middle(plan.exp_dir)}",
        f"Events   : {_short_middle(plan.exp_dir / 'events.jsonl')}",
        bar,
        "Controls : plan -> code -> train -> score -> gate -> memory -> report",
        "Boundary : official Kaggle submit and medal/rank claims stay behind human gates",
        "Prompt   : type a research goal, or press Enter for the default plan",
        bar,
    ])


def _run_once(session, goal: str, *, quiet_summary: bool = False,
              task_name: str = "") -> int:
    if not quiet_summary:
        print(f"[goal] {goal}\n")
    summary = session.run(goal)
    exp_dir = Path(session.exp_dir)
    _record_evolution_summary(
        _workspace_root_from_exp_dir(exp_dir),
        summary,
        task=task_name,
        events_path=exp_dir / "events.jsonl",
    )
    if not quiet_summary:
        _print_summary(session, summary)
    return 0 if summary.get("finished_by_agent") is True else 2


def _repl(session) -> int:
    default = "Establish a strong baseline, then improve the best CV score with 2-3 informed iterations."
    print("\nInput")
    print("  exit   quit")
    print(f"  enter  {default}\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0
        if line.lower() in {"exit", "quit", "q"}:
            print("bye.")
            return 0
        goal = line or default
        print(f"\n[goal] {goal}\n")
        try:
            summary = session.run(goal)
            task_name = str(getattr(getattr(session, "context", None), "task_name", ""))
            exp_dir = Path(session.exp_dir)
            _record_evolution_summary(
                _workspace_root_from_exp_dir(exp_dir),
                summary,
                task=task_name,
                events_path=exp_dir / "events.jsonl",
            )
        except Exception as exc:  # keep the REPL alive on a run error
            print(f"\n! run error: {type(exc).__name__}: {exc}")
            continue
        _print_summary(session, summary)
        print()


def _print_summary(session, summary: dict[str, Any]) -> None:
    print(
        f"\n[summary] best={summary.get('best_exp_id')}  cv={summary.get('best_cv_score')}  "
        f"promotions={summary.get('n_promotions')}/{summary.get('n_iterations')}  "
        f"turns={summary.get('turns_used')}"
    )
    print(f"[artifacts] {session.exp_dir}")
