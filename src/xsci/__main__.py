"""xsci — the XCIENTIST research-agent CLI entry point.

Thin shell over the research_os engine. Phase 1 implements the installable
skeleton: `doctor` (self-check) and `config` (inspect resolved settings). Other
subcommands are registered so the surface is discoverable and stubbed with the
phase they land in.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from . import __version__
from .config import load_config, GLOBAL_DIR, find_project_dir

_STUBS: dict[str, str] = {}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="xsci",
        description="XCIENTIST - a research-type terminal agent for auditable, "
                    "experience-driven ML research.",
    )
    p.add_argument("-V", "--version", action="version", version=f"xsci {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    sub.add_parser("doctor", help="self-check: config, keys, compute, deps")

    cfg = sub.add_parser("config", help="show the resolved configuration")
    cfg.add_argument("key", nargs="?", help="optional dotted key to print (e.g. llm.provider)")

    ini = sub.add_parser("init", help="scaffold a research project in the current dir")
    ini.add_argument("--compute", choices=["local", "gpu"], default="local",
                     help="default compute backend for this project")
    ini.add_argument("--force", action="store_true", help="overwrite an existing project config")

    log = sub.add_parser("login", help="securely set LLM / Kaggle keys (stored in ~/.xsci)")
    log.add_argument("--provider", choices=["anthropic", "deepseek"], help="LLM provider")
    log.add_argument("--api-key", help="LLM API key (non-interactive)")
    log.add_argument("--base-url", help="custom LLM base URL")
    log.add_argument("--kaggle-username", help="Kaggle username (non-interactive)")
    log.add_argument("--kaggle-key", help="Kaggle API key (non-interactive)")
    log.add_argument("--kaggle-json", help="path to an existing kaggle.json to import")
    log.add_argument("--non-interactive", action="store_true",
                     help="use flags instead of prompting (for CI)")

    tsk = sub.add_parser("task", help="register / list research task configs")
    tsub = tsk.add_subparsers(dest="task_command", metavar="<subcommand>")
    tadd = tsub.add_parser("add", help="add a task from JSON path, Kaggle URL, or name")
    tadd.add_argument("source", help="path/to/task.json | https://kaggle.com/c/<slug> | <name>")
    tadd.add_argument("--force", action="store_true", help="overwrite an existing task")
    tsub.add_parser("list", help="list registered tasks")

    run = sub.add_parser("run", help="run the research loop on a task")
    run.add_argument("task", help="task slug (see `xsci task list`) or path to a task .json")
    run.add_argument("--compute", choices=["local", "gpu"], help="override the compute backend")
    run.add_argument("--iterations", type=int, help="number of search iterations")
    run.add_argument("--data-dir", default="", help="data dir (local) / remote dirname (gpu)")
    run.add_argument("--no-mcgs", action="store_true", help="disable the MCGS selection brain")
    run.add_argument("--quiet", action="store_true",
                     help="don't stream per-event lines to the terminal (events.jsonl is still written)")
    run.add_argument("--dry-run", action="store_true",
                     help="resolve + print the plan without spending tokens or hitting the network")

    rep = sub.add_parser("report", help="show completed run results (read-only)")
    rep.add_argument("run", nargs="?", default="",
                     help="run id or task name; omit to list all runs")

    agt = sub.add_parser("agent", help="interactive deep research agent (the model drives)")
    agt.add_argument("task", help="task slug (see `xsci task list`) or path to a task .json")
    agt.add_argument("--goal", help="run a single goal non-interactively, then exit")
    agt.add_argument("--compute", choices=["local", "gpu"], help="override the compute backend")
    agt.add_argument("--data-dir", default="", help="data dir (local) / remote dirname (gpu)")
    agt.add_argument("--no-mcgs", action="store_true",
                     help="disable the MCGS selection brain (model chooses freely; Phase A behavior)")
    agt.add_argument("--quiet", action="store_true",
                     help="don't stream agent thoughts/tool calls (events.jsonl still written)")
    agt.add_argument("--resume", action="store_true",
                     help="continue the newest prior run of this task (reload its search graph "
                          "+ conversation) instead of starting fresh")

    watch = sub.add_parser("watch", help="replay or follow a run's events.jsonl stream")
    watch.add_argument("run", nargs="?", default="",
                       help="run id, task name, run dir, or events.jsonl path; omit for newest")
    watch.add_argument("-f", "--follow", action="store_true", help="keep following new events")
    watch.add_argument("--lines", type=int, default=80, help="initial number of lines to render")
    watch.add_argument("--interval", type=float, default=1.0, help="poll interval in follow mode")

    dash = sub.add_parser("dashboard", help="manage the existing :8088 workstation dashboard")
    dash.add_argument("dashboard_command", nargs="?", default="status",
                      choices=["start", "stop", "restart", "status"])
    dash.add_argument("--port", type=int, default=8088)
    dash.add_argument("--timeout", type=float, default=45.0)
    dash.add_argument("--build", action="store_true", help="run npm build before start/restart")
    dash.add_argument("--force", action="store_true", help="force restart/port cleanup")

    mem = sub.add_parser("memory", help="inspect retrospective memory lessons")
    mem.add_argument("memory_command", nargs="?", default="list",
                     choices=["list", "successes", "failures"])
    mem.add_argument("--task-type", default="", help="filter by task_type")
    mem.add_argument("--limit", type=int, default=20)
    mem.add_argument("--json", action="store_true", help="emit raw JSON records")
    mem.add_argument("--path", default="", help="override retrospective_memory.json path")
    return p


def cmd_doctor(_args: argparse.Namespace) -> int:
    from .doctor import run_doctor
    return run_doctor()


def cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config()
    if args.key:
        val = cfg.get(args.key)
        print(val if val is not None else f"(unset: {args.key})")
        return 0 if val is not None else 1
    print(f"config sources (low->high precedence):")
    for src in cfg.sources or ["(none - run `xsci login` / `xsci init`)"]:
        print(f"  - {src}")
    print(f"\nglobal dir : {GLOBAL_DIR}")
    proj = find_project_dir()
    print(f"project    : {proj if proj else '(none in this directory tree)'}")
    print(f"llm.provider : {cfg.get('llm.provider', '(unset)')}")
    print(f"compute.backend : {cfg.get('compute.backend', 'local (default)')}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    from .project import run_init
    return run_init(compute=args.compute, force=args.force)


def cmd_login(args: argparse.Namespace) -> int:
    from .login import run_login
    return run_login(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        kaggle_username=args.kaggle_username,
        kaggle_key=args.kaggle_key,
        kaggle_json=args.kaggle_json,
        non_interactive=args.non_interactive,
    )


def cmd_task(args: argparse.Namespace) -> int:
    from .tasks import add_task, list_tasks
    if args.task_command == "add":
        try:
            dest = add_task(args.source, force=args.force)
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        print(f"registered task: {dest.stem}  ({dest})")
        if '"TODO' in dest.read_text(encoding="utf-8"):
            print("  note: scaffold has TODO fields - edit schema/metric before running.")
        return 0
    if args.task_command == "list":
        tasks = list_tasks()
        if not tasks:
            print("no tasks yet - `xsci task add <kaggle-url|path|name>`")
            return 0
        print("registered tasks:")
        for slug, path in tasks:
            print(f"  - {slug}  ({path.name})")
        return 0
    print("usage: xsci task {add|list}")
    return 2


def cmd_run(args: argparse.Namespace) -> int:
    from .tasks import resolve_task
    from .engine import build_plan, execute_plan
    try:
        task_config = resolve_task(args.task)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1
    plan = build_plan(
        task_config,
        compute=args.compute,
        iterations=args.iterations,
        mcgs=not args.no_mcgs,
        data_dir=args.data_dir,
    )
    print(plan.render())
    if args.dry_run:
        print("\n[dry-run] resolved only - nothing executed.")
        return 0
    blocking = [w for w in plan.warnings if "no LLM key" in w]
    if blocking:
        print("\nrefusing to run: " + "; ".join(blocking))
        return 1
    print("\nstarting research loop (this spends API tokens"
          + (" and uses the remote GPU)" if plan.compute == "gpu" else ")") + " ...\n")
    from research_os import events as ev
    quiet = getattr(args, "quiet", False)

    def _render(event: dict) -> None:
        # Live one-line-per-event renderer. The JSONL sink (in execute_plan) is the
        # durable record; this is just the human view, so a print failure is ignored.
        if quiet:
            return
        try:
            print(ev.format_event(event), flush=True)
        except Exception:  # noqa: BLE001 - never let rendering break a run
            pass

    summary = execute_plan(plan, on_event=_render)
    print(f"\nbest={summary.get('best_exp_id')} cv={summary.get('best_cv_score')} "
          f"promotions={summary.get('n_promotions')}/{summary.get('n_iterations')}")
    print(f"artifacts: {plan.exp_dir}")
    print(f"events   : {plan.exp_dir / 'events.jsonl'}")
    return 0


def _fmt_score(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "  -   "


def cmd_agent(args: argparse.Namespace) -> int:
    from .agent import run_agent
    return run_agent(
        args.task, goal=args.goal, compute=args.compute,
        data_dir=args.data_dir, quiet=args.quiet, mcgs=not args.no_mcgs,
        resume=args.resume,
    )


def cmd_watch(args: argparse.Namespace) -> int:
    from .watch import run_watch
    return run_watch(args.run, follow=args.follow, lines=args.lines, interval=args.interval)


def cmd_dashboard(args: argparse.Namespace) -> int:
    from .dashboard import run_dashboard
    return run_dashboard(
        args.dashboard_command,
        port=args.port,
        timeout=args.timeout,
        build=args.build,
        force=args.force,
    )


def cmd_memory(args: argparse.Namespace) -> int:
    from .memory import run_memory
    return run_memory(
        args.memory_command,
        task_type=args.task_type,
        limit=args.limit,
        json_output=args.json,
        path=args.path,
    )


def cmd_report(args: argparse.Namespace) -> int:
    from .results import list_results, find_result
    if not args.run:
        rows = list_results()
        if not rows:
            print("no completed runs yet - `xsci run <task>` first")
            return 0
        print(f"{'run':<48} {'best':>8} {'promo':>7} {'ok%':>5}")
        for r in rows:
            print(f"{r.run_id:<48} {_fmt_score(r.best_cv_score):>8} "
                  f"{r.n_promotions:>3}/{r.n_iterations:<3} {r.success_rate*100:>4.0f}%")
        return 0
    r = find_result(args.run)
    if r is None:
        print(f"no run matching '{args.run}' - see `xsci report`")
        return 1
    print(f"run     : {r.run_id}")
    print(f"task    : {r.task}   metric: {r.metric} ({r.metric_direction})")
    print(f"best    : {r.best_exp_id}  cv={_fmt_score(r.best_cv_score)}")
    print(f"promoted: {r.n_promotions}/{r.n_iterations}   success: {r.success_rate*100:.0f}%")
    print(f"artifacts: {r.run_dir}")
    print("\niterations:")
    print(f"  {'exp':<10} {'mode':<9} {'ok':<3} {'cv':>8} {'promo':<5} model")
    for it in r.iterations:
        mark = "+" if it.promoted else " "
        prov = f"{it.provider}/{it.model}" if it.provider else ""
        print(f"  {it.exp_id:<10} {it.mode:<9} {'Y' if it.success else 'n':<3} "
              f"{_fmt_score(it.cv_score):>8} {mark:<5} {prov}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0
    dispatch = {
        "doctor": cmd_doctor, "config": cmd_config, "init": cmd_init,
        "login": cmd_login, "task": cmd_task, "run": cmd_run, "report": cmd_report,
        "agent": cmd_agent,
        "watch": cmd_watch, "dashboard": cmd_dashboard, "memory": cmd_memory,
    }
    if args.command in dispatch:
        return dispatch[args.command](args)
    if args.command in _STUBS:
        print(f"`xsci {args.command}` - not implemented yet: {_STUBS[args.command]}")
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
