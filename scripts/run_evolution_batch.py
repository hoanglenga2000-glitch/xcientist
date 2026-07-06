#!/usr/bin/env python3
"""Batch-evolve many tasks and aggregate results into a leaderboard.

Runs the evolution engine over every task config in a directory (or an explicit
list), one after another, and writes a combined leaderboard JSON + Markdown.
Each task is isolated: a failure in one does not stop the batch.

    python scripts/run_evolution_batch.py --config-dir configs/evolution --runner gpu --iterations 3

The shared retrospective memory persists across tasks, so later tasks can reuse
lessons from earlier ones (cross-task learning).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _discover_configs(config_dir: Path, only: list[str]) -> list[Path]:
    configs = sorted(p for p in config_dir.glob("*.json"))
    if only:
        wanted = {name if name.endswith(".json") else f"{name}.json" for name in only}
        configs = [p for p in configs if p.name in wanted]
    return configs


def _run_one(config: Path, runner: str, iterations: int, timeout: int, mcgs: bool = False) -> dict:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_evolution.py"),
           "--task-config", str(config), "--runner", runner,
           "--iterations", str(iterations), "--timeout", str(timeout)]
    if mcgs:
        cmd.append("--mcgs")
    started = datetime.now()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=iterations * (timeout + 120) + 300)
        ok = proc.returncode == 0
        tail = (proc.stdout or "")[-1200:]
        err = (proc.stderr or "")[-600:]
    except subprocess.TimeoutExpired:
        ok, tail, err = False, "", "batch-level timeout"
    # locate the newest summary for this task
    task_name = json.loads(config.read_text(encoding="utf-8"))["task_name"]
    dirs = sorted((ROOT / "experiments" / "evolution").glob(f"{task_name}_{runner}_*/"), reverse=True)
    summary = {}
    if dirs and (dirs[0] / "summary.json").exists():
        summary = json.loads((dirs[0] / "summary.json").read_text(encoding="utf-8"))
    return {
        "task": task_name, "config": config.name, "ok": ok,
        "best_cv_score": summary.get("best_cv_score"),
        "best_exp_id": summary.get("best_exp_id"),
        "metric": summary.get("metric"), "metric_direction": summary.get("metric_direction"),
        "n_promotions": summary.get("n_promotions"), "n_iterations": summary.get("n_iterations"),
        "elapsed_s": round((datetime.now() - started).total_seconds(), 1),
        "stdout_tail": tail, "stderr_tail": err,
    }


def _write_leaderboard(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "batch_leaderboard.json").write_text(
        json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"), "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Evolution Batch Leaderboard", "",
             f"Generated: {datetime.now().isoformat(timespec='seconds')}", "",
             "| Task | Metric | Best CV | Promotions | OK | Elapsed |",
             "|------|--------|---------|------------|----|---------|"]
    for r in results:
        lines.append(f"| {r['task']} | {r.get('metric') or '?'} ({r.get('metric_direction') or '?'}) "
                     f"| {r.get('best_cv_score')} | {r.get('n_promotions')}/{r.get('n_iterations')} "
                     f"| {'yes' if r['ok'] else 'NO'} | {r['elapsed_s']}s |")
    (out_dir / "batch_leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch-evolve many tasks.")
    ap.add_argument("--config-dir", default="configs/evolution")
    ap.add_argument("--only", nargs="*", default=[], help="restrict to these config names")
    ap.add_argument("--runner", choices=["local", "gpu"], default="gpu")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=360)
    ap.add_argument("--mcgs", action="store_true", help="enable MCGS selector brain for every task")
    args = ap.parse_args()

    config_dir = (ROOT / args.config_dir) if not Path(args.config_dir).is_absolute() else Path(args.config_dir)
    configs = _discover_configs(config_dir, args.only)
    if not configs:
        print(f"No task configs found in {config_dir}", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / "evolution" / f"_batch_{args.runner}_{stamp}"
    print(f"[batch] {len(configs)} tasks | runner={args.runner} | iters={args.iterations} | mcgs={args.mcgs}")
    results = []
    for i, config in enumerate(configs, 1):
        print(f"[batch] ({i}/{len(configs)}) {config.name} ...")
        result = _run_one(config, args.runner, args.iterations, args.timeout, args.mcgs)
        results.append(result)
        print(f"[batch]   -> best_cv={result['best_cv_score']} ok={result['ok']} ({result['elapsed_s']}s)")
        _write_leaderboard(results, out_dir)  # incremental, survives interruption
    print(f"\n[batch] done. leaderboard: {out_dir}/batch_leaderboard.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

