"""`xsci run` — drive the research_os evolution loop from the CLI.

This is the thin bridge between the terminal agent and the engine. It resolves a
task config, injects ~/.xsci credentials into the environment, picks a Runner
(local subprocess or remote GPU) per the resolved compute backend, and runs the
loop. `--dry-run` performs every resolution step and prints the plan WITHOUT
spending API tokens or touching the network, so wiring is verifiable offline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .config import Config, inject_engine_env, load_config, find_project_dir


@dataclass
class RunPlan:
    """Everything resolved for a run, safe to print (no secrets)."""

    task_name: str
    task_config: Path
    compute: str
    iterations: int
    mcgs: bool
    data_dir: str
    exp_dir: Path
    injected_env: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "run plan:",
            f"  task        : {self.task_name}",
            f"  config      : {self.task_config}",
            f"  compute     : {self.compute}",
            f"  iterations  : {self.iterations}",
            f"  mcgs brain  : {'on' if self.mcgs else 'off'}",
            f"  data dir    : {self.data_dir or '(none)'}",
            f"  output      : {self.exp_dir}",
            f"  env injected: {', '.join(self.injected_env) or '(none)'}",
            f"  strategies  : {', '.join(self.strategies) or '(engine default)'}",
        ]
        for w in self.warnings:
            lines.append(f"  ! warning   : {w}")
        return "\n".join(lines)


def _load_context(config_path: Path):
    from research_os.variation_generator import TaskContext

    data = json.loads(config_path.read_text(encoding="utf-8"))
    ctx = TaskContext(
        task_name=data["task_name"],
        modality=data.get("modality", "tabular"),
        task_type=data.get("task_type", "classification"),
        metric=data.get("metric", "accuracy"),
        metric_direction=data.get("metric_direction", "maximize"),
        target_column=data.get("target_column", ""),
        id_column=data.get("id_column", ""),
        data_schema=data.get("data_schema", ""),
        n_train=int(data.get("n_train", 0)),
        n_test=int(data.get("n_test", 0)),
        extra_notes=data.get("extra_notes", ""),
    )
    return ctx, data


def _strategies_for(ctx, data: dict[str, Any]) -> list[str]:
    from research_os.strategy_selector import TaskProfile, recommend_strategies

    profile = TaskProfile(
        modality=ctx.modality, task_type=ctx.task_type,
        train_size=ctx.n_train, test_size=ctx.n_test, metric=ctx.metric,
        n_features=int(data.get("n_features", 0)),
        n_high_cardinality_features=int(data.get("n_high_cardinality_features", 0)),
        n_model_families=int(data.get("n_model_families", 3)),
        has_time_column=bool(data.get("has_time_column", False)),
        target_is_positive=bool(data.get("target_is_positive", False)),
    )
    return recommend_strategies(profile).strategies


def _read_run_events(events_path: Path | None) -> list[dict[str, Any]]:
    if events_path is None:
        return []
    path = Path(events_path)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _record_evolution_summary(root: Path, summary: dict[str, Any], *, task: str = "",
                              events_path: Path | None = None) -> None:
    """Persist high-level learning stats after a completed run.

    This bridges real run artifacts back into the terminal's self-evolution
    tracker.  It deliberately records only summary-level metrics and never
    reads credentials or raw data.
    """
    try:
        from .evolution_tracker import EvolutionTracker

        events = _read_run_events(events_path)
        promotions = int(summary.get("n_promotions") or 0)
        iterations = int(summary.get("n_iterations") or 0)
        tracker = EvolutionTracker(root)
        tracker.record_run(
            success=bool(summary.get("best_exp_id")) or promotions > 0 or iterations > 0,
            cv_score=summary.get("best_cv_score"),
            promotions=promotions,
            task=task or str(summary.get("task") or ""),
        )
        repair_events = [e for e in events if e.get("type") == "repair"]
        promotion_events = [e for e in events if e.get("type") == "promote" and e.get("promoted")]
        repair_succeeded = bool(promotion_events) or promotions > 0
        for _event in repair_events:
            tracker.record_repair(success=repair_succeeded)
        for event in events:
            if event.get("type") != "lesson":
                continue
            reusable = bool(event.get("reusable_strategy"))
            failure = bool(event.get("failure_pattern"))
            tracker.record_lesson(reusable=reusable, failure=failure)
        if promotions > 0:
            tracker.record_task_completed(task or str(summary.get("task") or ""))
    except Exception:
        pass


def _workspace_root_from_exp_dir(exp_dir: Path) -> Path:
    """Return workspace root for root/experiments/evolution/<run> paths."""
    exp_dir = Path(exp_dir)
    try:
        return exp_dir.parents[2]
    except IndexError:
        return exp_dir.parent


def build_plan(
    task_config: Path,
    *,
    cfg: Optional[Config] = None,
    compute: Optional[str] = None,
    iterations: Optional[int] = None,
    mcgs: bool = True,
    data_dir: str = "",
    project_root: Optional[Path] = None,
) -> RunPlan:
    """Resolve a full run plan. Pure/offline: no engine execution, no network."""
    cfg = cfg or load_config(project_root)
    root = project_root or find_project_dir() or Path.cwd()
    ctx, data = _load_context(task_config)

    compute = compute or cfg.get("compute.backend", "local")
    iterations = iterations if iterations is not None else int(cfg.get("run.iterations", 20))
    injected = inject_engine_env(cfg)
    strategies = _strategies_for(ctx, data)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = root / "experiments" / "evolution" / f"{ctx.task_name}_{compute}_{stamp}"

    warnings: list[str] = []
    if compute == "gpu":
        dirname = data.get("remote_data_dirname") or ctx.task_name
        resolved_data = data_dir or data.get("gpu_data_dir") or dirname
    else:
        resolved_data = data_dir or data.get("local_data_dir", "")
        if not resolved_data:
            warnings.append("local compute but no data dir (set local_data_dir in the task or pass --data-dir)")
    if "ANTHROPIC_API_KEY" not in injected and "DEEPSEEK_API_KEY" not in injected:
        import os
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")):
            warnings.append("no LLM key available - run `xsci login`")

    return RunPlan(
        task_name=ctx.task_name, task_config=task_config, compute=compute,
        iterations=iterations, mcgs=mcgs, data_dir=resolved_data, exp_dir=exp_dir,
        injected_env=injected, strategies=strategies, warnings=warnings,
    )


def execute_plan(plan: RunPlan, *, on_event: Callable[[dict], None] | None = None) -> dict[str, Any]:
    """Actually run the loop. Spends API tokens and (if gpu) hits the A40.

    A research-event stream is ALWAYS persisted to ``<exp_dir>/events.jsonl`` so
    ``xsci watch`` / ``dashboard`` / replay have a single source of truth, whether
    or not a live terminal renderer is attached. ``on_event`` is an optional extra
    sink (the live terminal renderer for ``xsci run``); it is fanned out alongside
    the JSONL writer, and an exception in one sink never disturbs the other or the
    run itself.
    """
    from research_os.evolution_loop import EvolutionConfig, EvolutionLoop, LocalSubprocessRunner
    from research_os.retrospective_memory import RetrospectiveMemoryStore
    from research_os import events as ev

    ctx, data = _load_context(plan.task_config)
    plan.exp_dir.mkdir(parents=True, exist_ok=True)
    memory = RetrospectiveMemoryStore(plan.exp_dir.parent / "retrospective_memory.json")

    if plan.compute == "gpu":
        from research_os.gpu_runner import GPURunner, GPURunnerConfig
        dirname = data.get("remote_data_dirname") or ctx.task_name
        runner = GPURunner(dirname, config=GPURunnerConfig(timeout=int(data.get("timeout", 1800))))
    else:
        runner = LocalSubprocessRunner(plan.exp_dir / "runs", timeout=int(data.get("timeout", 1800)))

    selector = None
    if plan.mcgs:
        from research_os.mcgs_selector import MCGSSelector
        selector = MCGSSelector(total_steps=plan.iterations)

    # Always persist the event stream; fan the live renderer in when present.
    sink = ev.fan_out(ev.JsonlEventSink(plan.exp_dir / "events.jsonl"), on_event)
    run_meta = {"compute": plan.compute, "exp_dir": str(plan.exp_dir)}

    loop = EvolutionLoop(
        ctx, data_dir=plan.data_dir, work_dir=plan.exp_dir, runner=runner,
        memory=memory, config=EvolutionConfig(max_iterations=plan.iterations),
        selector=selector, on_event=sink, run_meta=run_meta,
    )
    summary = loop.run(strategies=plan.strategies)
    (plan.exp_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if loop.best_code:
        (plan.exp_dir / "best_solution.py").write_text(loop.best_code, encoding="utf-8")
    loop.graph.export_json(plan.exp_dir / "search_graph.json")
    _record_evolution_summary(
        _workspace_root_from_exp_dir(plan.exp_dir),
        summary,
        task=ctx.task_name,
        events_path=plan.exp_dir / "events.jsonl",
    )
    return summary
