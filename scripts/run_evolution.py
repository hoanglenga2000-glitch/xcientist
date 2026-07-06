#!/usr/bin/env python3
"""Run the evolution engine on one task, locally or on the GPU.

One command unifies both tracks:

    # local (fast, needs sklearn/lightgbm locally)
    python scripts/run_evolution.py --task-config configs/evolution/nomad2018.json --runner local

    # GPU (real training on the A40, data already on the box)
    python scripts/run_evolution.py --task-config configs/evolution/nomad2018.json --runner gpu

The task config is a small JSON describing the TaskContext plus the remote data
dir name. Results (summary + best solution) are written under experiments/evolution/.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (SRC, ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(ROOT / ".env"), override=True)  # .env is authoritative for the engine
except Exception:
    pass

from research_os.evolution_loop import EvolutionConfig, EvolutionLoop, LocalSubprocessRunner
from research_os.retrospective_memory import RetrospectiveMemoryStore
from research_os.strategy_selector import TaskProfile, recommend_strategies
from research_os.variation_generator import TaskContext


def _load_context(config_path: Path) -> tuple[TaskContext, dict]:
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


def _strategies_for(ctx: TaskContext, data: dict) -> list[str]:
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the evolution engine on one task.")
    ap.add_argument("--task-config", required=True, help="JSON task context file")
    ap.add_argument("--runner", choices=["local", "gpu"], default="local")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--data-dir", default="", help="local data dir (local runner) or remote data dirname (gpu)")
    ap.add_argument("--remote-data-dirname", default="", help="task dir under mlebench_raw_data (gpu runner)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--mcgs", action="store_true",
                    help="enable the MCGS selection brain (UCT + multi-branch + cross/aggregation)")
    args = ap.parse_args()

    config_path = Path(args.task_config)
    ctx, data = _load_context(config_path)
    strategies = _strategies_for(ctx, data)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = ROOT / "experiments" / "evolution" / f"{ctx.task_name}_{args.runner}_{stamp}"
    exp_root.mkdir(parents=True, exist_ok=True)
    memory = RetrospectiveMemoryStore(ROOT / "experiments" / "evolution" / "retrospective_memory.json")

    if args.runner == "gpu":
        from research_os.gpu_runner import GPURunner, GPURunnerConfig
        dirname = args.remote_data_dirname or data.get("remote_data_dirname") or ctx.task_name
        runner = GPURunner(dirname, config=GPURunnerConfig(timeout=args.timeout))
        # Prefer an explicit absolute remote data dir from the config JSON. Reading
        # it from the file (not argv) avoids MSYS/Git-bash mangling a Unix absolute
        # path like /hpc2hdd/... into C:/Program Files/Git/hpc2hdd/... on Windows.
        data_dir = args.data_dir or data.get("gpu_data_dir") or dirname
    else:
        runner = LocalSubprocessRunner(exp_root / "runs", timeout=args.timeout)
        data_dir = args.data_dir or data.get("local_data_dir", "")
        if not data_dir:
            print("ERROR: --data-dir is required for the local runner", file=sys.stderr)
            return 2

    selector = None
    if getattr(args, "mcgs", False):
        from research_os.mcgs_selector import MCGSSelector
        selector = MCGSSelector(total_steps=args.iterations)

    loop = EvolutionLoop(
        ctx, data_dir=data_dir, work_dir=exp_root, runner=runner,
        memory=memory, config=EvolutionConfig(max_iterations=args.iterations),
        selector=selector,
    )
    print(f"[evolution] task={ctx.task_name} runner={args.runner} iters={args.iterations} "
          f"mcgs={'on' if selector else 'off'} strategies={strategies}")
    summary = loop.run(strategies=strategies)

    (exp_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if loop.best_code:
        (exp_root / "best_solution.py").write_text(loop.best_code, encoding="utf-8")
    graph_path = exp_root / "search_graph.json"
    loop.graph.export_json(graph_path)

    print("\n=== EVOLUTION SUMMARY ===")
    for it in summary["iterations"]:
        print(f"  {it['exp_id']} mode={it['mode']:8s} ok={it['success']} "
              f"cv={it['cv_score']} promoted={it['promoted']} [{it['provider']}/{it['model']}]")
    print(f"best={summary['best_exp_id']} cv={summary['best_cv_score']} "
          f"promotions={summary['n_promotions']}/{summary['n_iterations']}")
    print(f"artifacts: {exp_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

