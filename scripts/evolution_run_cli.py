#!/usr/bin/env python3
"""Web-callable wrapper around the research_os evolution engine (engine A).

This is the thin, JSON-in/JSON-out entry point the workstation web app spawns to
run REAL evolution training through ``research_os.EvolutionLoop`` (opus-4-8 +
GPURunner). It deliberately REUSES the standalone
``run_evolution.py`` helpers (``_load_context`` / ``_strategies_for``) instead of
duplicating them, so both tracks stay in lockstep.

Contract (mirrors evolution_engine_cli.py conventions):
  * Input: a JSON file via --input, with keys:
      task_id            (required) workstation task id, e.g. "nyc_taxi"
      runner             "gpu"                     (default "gpu")
      iterations         int                        (default 8)
      mcgs               bool                       (default true)
      evolution_config   optional explicit path to a configs/evolution/*.json
      data_dir           optional remote data path or dirname
  * Output: exactly one JSON object on stdout:
      {ok, task_id, runner, summary_path, exp_dir, best_exp_id, best_cv_score,
       metric, metric_direction, n_iterations, n_promotions}
    On failure: {ok:false, error, decision}.

Never fabricates: it only reports what the engine actually wrote to disk.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (SRC, ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Reuse the standalone engine helpers so the two tracks never drift apart.
from run_evolution import _load_context, _strategies_for  # type: ignore  # noqa: E402

from research_os.evolution_loop import EvolutionConfig, EvolutionLoop  # noqa: E402
from research_os.hpc_policy import HPCPolicyError, require_hpc_compute  # noqa: E402
from research_os.retrospective_memory import RetrospectiveMemoryStore  # noqa: E402

EVOLUTION_CONFIG_DIR = ROOT / "configs" / "evolution"

# task_id -> evolution config stem. Web task ids can differ from the JSON stems;
# unknown ids fall through to a direct "<task_id>.json" probe (no guessing beyond
# an exact filename match, so we never run the wrong task's config).
_ALIASES = {
    "nyc_taxi": "nyc_taxi",
    "new-york-city-taxi-fare-prediction": "nyc_taxi",
    "nomad2018": "nomad2018",
    "aerial_cactus": "aerial_cactus",
    "aerial-cactus-identification": "aerial_cactus",
    "champs": "champs",
    "leaf_classification": "leaf_classification",
    "spooky_author": "spooky_author",
    "tps_dec2021": "tps_dec2021",
    "tps_may2022": "tps_may2022",
    "ventilator": "ventilator",
}


def resolve_evolution_config(task_id: str, explicit: str = "") -> Path | None:
    """Return the configs/evolution/*.json for a task_id, or None if none exists.

    Resolution order: explicit path -> alias map -> exact "<task_id>.json".
    Returns None (never a guess) when nothing matches, so the caller can report
    ``no_evolution_config`` instead of running an unintended task.
    """
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / explicit
        return p if p.exists() else None
    stem = _ALIASES.get(task_id, task_id)
    candidate = EVOLUTION_CONFIG_DIR / f"{stem}.json"
    return candidate if candidate.exists() else None


def _fail(error: str, decision: str = "failed") -> int:
    print(json.dumps({"ok": False, "error": error, "decision": decision}, ensure_ascii=False))
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Web-callable research_os evolution runner.")
    ap.add_argument("--input", required=True, help="JSON input file (task_id, runner, iterations, ...)")
    args = ap.parse_args()

    try:
        data_in = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report as JSON, never crash silently
        return _fail(f"bad input file: {type(exc).__name__}")

    task_id = str(data_in.get("task_id", "") or "")
    if not task_id:
        return _fail("task_id is required", "rejected")
    runner_kind = str(data_in.get("runner", "gpu") or "gpu").lower()
    try:
        require_hpc_compute(runner_kind)
    except HPCPolicyError as exc:
        return _fail(str(exc), "blocked_local_training_disabled")
    iterations = int(data_in.get("iterations", 8) or 8)
    use_mcgs = bool(data_in.get("mcgs", True))
    explicit_cfg = str(data_in.get("evolution_config", "") or "")

    config_path = resolve_evolution_config(task_id, explicit_cfg)
    if config_path is None:
        return _fail(
            f"no evolution config for task_id '{task_id}'. Create configs/evolution/<name>.json "
            f"or pass evolution_config.", "no_evolution_config")

    ctx, cfg_data = _load_context(config_path)
    strategies = _strategies_for(ctx, cfg_data)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = ROOT / "experiments" / "evolution" / f"{ctx.task_name}_{runner_kind}_{stamp}"
    exp_root.mkdir(parents=True, exist_ok=True)
    memory = RetrospectiveMemoryStore(ROOT / "experiments" / "evolution" / "retrospective_memory.json")

    from research_os.gpu_runner import GPURunner, GPURunnerConfig

    dirname = cfg_data.get("remote_data_dirname") or ctx.task_name
    runner = GPURunner(dirname, config=GPURunnerConfig())
    data_dir = str(data_in.get("data_dir", "") or cfg_data.get("gpu_data_dir") or dirname)

    selector = None
    if use_mcgs:
        from research_os.mcgs_selector import MCGSSelector
        selector = MCGSSelector(total_steps=iterations)

    loop = EvolutionLoop(
        ctx, data_dir=data_dir, work_dir=exp_root, runner=runner,
        memory=memory, config=EvolutionConfig(max_iterations=iterations),
        selector=selector,
    )

    try:
        summary = loop.run(strategies=strategies)
    except Exception as exc:  # noqa: BLE001 - engine errors reported as JSON
        return _fail(f"evolution loop failed: {type(exc).__name__}: {exc}", "training_failed")

    summary_path = exp_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if loop.best_code:
        (exp_root / "best_solution.py").write_text(loop.best_code, encoding="utf-8")
    loop.graph.export_json(exp_root / "search_graph.json")

    rel = lambda p: str(p).replace(str(ROOT), "").lstrip("/\\").replace("\\", "/")  # noqa: E731
    print(json.dumps({
        "ok": True,
        "task_id": task_id,
        "runner": runner_kind,
        "summary_path": rel(summary_path),
        "exp_dir": rel(exp_root),
        "best_exp_id": summary.get("best_exp_id"),
        "best_cv_score": summary.get("best_cv_score"),
        "metric": summary.get("metric"),
        "metric_direction": summary.get("metric_direction"),
        "n_iterations": summary.get("n_iterations"),
        "n_promotions": summary.get("n_promotions"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
