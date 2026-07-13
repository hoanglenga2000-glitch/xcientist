from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIGS = {
    "house_prices": "configs/house_prices.yaml",
    "titanic": "configs/titanic.yaml",
    "telco_churn": "configs/telco_churn.yaml",
}


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    parsed = None
    if completed.stdout.strip():
        first_json = completed.stdout.strip().split("\nSummary written to:", 1)[0].strip()
        try:
            parsed = json.loads(first_json)
        except json.JSONDecodeError:
            parsed = None
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "parsed_run": parsed,
    }


def build_commands(tasks: list[str], seeds: list[int], output_base: str) -> list[dict[str, Any]]:
    commands = []
    for task_id in tasks:
        config = TASK_CONFIGS[task_id]
        for seed in seeds:
            commands.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "config": config,
                    "command": [
                        sys.executable,
                        "scripts/run_workstation_orchestrator.py",
                        "--config",
                        config,
                        "--output-base",
                        output_base,
                        "--random-state",
                        str(seed),
                    ],
                }
            )
    return commands


def summarize_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    best_by_task: dict[str, Any] = {}
    for item in results:
        parsed = item.get("parsed_run") or {}
        task_id = item.get("task_id")
        if not task_id or item.get("returncode") != 0:
            continue
        metrics = parsed.get("best_metrics") or {}
        metric_value = metrics.get("cv_rmsle_mean")
        direction = "minimize"
        metric_name = "cv_rmsle_mean"
        if metric_value is None:
            metric_value = metrics.get("cv_accuracy_mean")
            direction = "maximize"
            metric_name = "cv_accuracy_mean"
        if not isinstance(metric_value, (int, float)):
            continue
        current = best_by_task.get(task_id)
        better = (
            current is None
            or (direction == "minimize" and metric_value < current["metric_value"])
            or (direction == "maximize" and metric_value > current["metric_value"])
        )
        if better:
            best_by_task[task_id] = {
                "task_id": task_id,
                "seed": item.get("seed"),
                "metric_name": metric_name,
                "metric_value": metric_value,
                "direction": direction,
                "best_model": parsed.get("best_model"),
                "output_dir": parsed.get("output_dir"),
            }
    return best_by_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run whitelisted GPU/remote training batches for Research Agent Workstation tasks.")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_CONFIGS), choices=sorted(TASK_CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output-base", default="experiments")
    parser.add_argument("--manifest-dir", default="workspace/gpu_training_batches")
    parser.add_argument("--dry-run", action="store_true", help="Only write the manifest of commands without executing training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = ROOT / args.manifest_dir / batch_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    commands = build_commands(args.tasks, args.seeds, args.output_base)
    results = []
    if not args.dry_run:
        for item in commands:
            result = run_command(item["command"])
            results.append({**item, **result})
            if result["returncode"] != 0:
                break

    manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "dry_run" if args.dry_run else "executed",
        "tasks": args.tasks,
        "seeds": args.seeds,
        "output_base": args.output_base,
        "command_count": len(commands),
        "commands": [
            {
                "task_id": item["task_id"],
                "seed": item["seed"],
                "config": item["config"],
                "command": " ".join(item["command"]),
            }
            for item in commands
        ],
        "results": results,
        "best_by_task": summarize_best(results),
        "status": "planned" if args.dry_run else "passed" if results and all(item["returncode"] == 0 for item in results) else "failed",
    }
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": manifest["status"] in {"planned", "passed"}, "manifest_path": str(manifest_path.relative_to(ROOT)), **manifest}, ensure_ascii=False, indent=2))
    if manifest["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
