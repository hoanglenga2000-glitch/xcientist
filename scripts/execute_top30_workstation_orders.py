from __future__ import annotations

"""Execute Top30 evolution orders through the workstation launcher.

This script is a supervisor-side batch controller. It does not train models
directly and does not submit to Kaggle. Every experiment is launched through
``scripts/run_workstation_ensemble.py``, which delegates to AgentOrchestrator.
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ORDERS_PATH = ROOT / "workspace" / "top30_next_evolution_orders_20260625.json"
OUT_DIR = ROOT / "workspace" / "top30_workstation_execution"
REPORTS_DIR = ROOT / "reports"
MEMORY_PATH = ROOT / "workspace" / "top30_retrospective_memory_20260625.json"

LOCAL_TEMPLATE = "sklearn_rf_hgb_et_ensemble"
HPC_TEMPLATES = {"exp007_style_lgb_xgb_cat_blend", "lightgbm_optuna_cv"}
DEFAULT_CONFIGS = {
    "spaceship_titanic": "configs/spaceship_titanic.yaml",
    "house_prices": "configs/house_prices.yaml",
    "titanic": "configs/titanic.yaml",
    "digit_recognizer": "configs/digit_recognizer.yaml",
    "porto_seguro_safe_driver_prediction": "configs/porto_seguro_safe_driver_prediction.yaml",
}


@dataclass
class BranchExecutionResult:
    task_id: str
    branch_id: str
    branch_type: str
    status: str
    command: list[str]
    returncode: int | None = None
    run_dir: str | None = None
    best_score: float | None = None
    score_gate_decision: str | None = None
    rank_gate_decision: str | None = None
    elapsed_seconds: float | None = None
    blocker: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    artifacts: dict[str, bool] = field(default_factory=dict)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def tail(text: str | None, limit: int = 3000) -> str:
    if not text:
        return ""
    return text[-limit:]


def parse_summary(stdout: str) -> dict[str, Any]:
    start = stdout.rfind("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        return json.loads(stdout[start : end + 1])
    except json.JSONDecodeError:
        return {}


def find_latest_run_dir(task_id: str, before_names: set[str]) -> Path | None:
    task_root = ROOT / "experiments" / task_id
    if not task_root.exists():
        return None
    candidates = [path for path in task_root.iterdir() if path.is_dir() and path.name not in before_names]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def artifact_state(run_dir: Path | None) -> dict[str, bool]:
    if run_dir is None:
        return {}
    names = [
        "agent_trace.json",
        "metrics.json",
        "oof_predictions.csv",
        "submission.csv",
        "artifact_manifest.json",
        "score_promotion_gate.json",
        "search_controller_decision.json",
        "validation_contract.json",
        "claim_audit.json",
        "submission_audit.json",
        "rank_promotion_gate.json",
        "benchmark_claim_gate.json",
        "task_benchmark_state.json",
        "workstation_run_registry.json",
        "failure_review.json",
        "timeout_manifest.json",
    ]
    return {name: (run_dir / name).exists() for name in names}


def read_run_quality(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    metrics = read_json(run_dir / "metrics.json")
    gate = read_json(run_dir / "score_promotion_gate.json")
    rank_gate = read_json(run_dir / "rank_promotion_gate.json")
    score = metrics.get("ensemble", {}).get("best_validation_score")
    decision = gate.get("decision", {}) if isinstance(gate.get("decision"), dict) else {}
    return {
        "best_score": score if isinstance(score, (int, float)) else None,
        "score_gate_decision": decision.get("decision"),
        "rank_gate_decision": rank_gate.get("decision"),
    }


def select_template(branch: dict[str, Any], allow_hpc: bool) -> str:
    branch_type = str(branch.get("branch_type") or "")
    if allow_hpc and branch_type in {"model_family", "hyperparameter_search"}:
        return "lightgbm_optuna_cv"
    if allow_hpc and branch_type in {"ensemble_blend", "rank_gate_candidate"}:
        return "exp007_style_lgb_xgb_cat_blend"
    return LOCAL_TEMPLATE


def build_command(
    *,
    task_id: str,
    config: str,
    template: str,
    branch_index: int,
    fast: bool,
    sample_rows: int,
    n_folds: int,
    timeout_seconds: int,
    branch: dict[str, Any],
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_workstation_ensemble.py"),
        "--config",
        config,
        "--template",
        template,
        "--output-base",
        "experiments",
        "--random-state",
        str(20260625 + branch_index),
        "--n-folds",
        str(n_folds),
        "--timeout-seconds",
        str(timeout_seconds),
        "--branch-id",
        str(branch.get("branch_id") or ""),
        "--branch-type",
        str(branch.get("branch_type") or ""),
        "--code-generation-mode",
        str(branch.get("code_generation_mode") or ""),
        "--branch-hypothesis",
        str(branch.get("hypothesis") or ""),
        "--cross-branch-references",
        json.dumps(branch.get("cross_branch_references") or [], ensure_ascii=False),
    ]
    if fast:
        cmd.extend(["--fast", "--sample-rows", str(sample_rows)])
    if task_id == "spaceship_titanic" and branch_index == 2:
        cmd.extend(["--seeds", "42,3407"])
    return cmd


def should_skip_task(order: dict[str, Any], selected_tasks: set[str] | None, max_priority: str) -> bool:
    task_id = str(order.get("task_id"))
    priority = str(order.get("priority") or "P9")
    if selected_tasks is not None and task_id not in selected_tasks:
        return True
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    return priority_rank.get(priority, 99) > priority_rank.get(max_priority, 99)


def update_retrospective_memory(payload: dict[str, Any]) -> None:
    if payload.get("dry_run"):
        return
    existing = read_json(MEMORY_PATH)
    records = existing.get("records", []) if isinstance(existing, dict) else []
    for result in payload.get("results", []):
        record = {
            "memory_id": f"{result['task_id']}::{result['branch_id']}::{payload['batch_id']}",
            "task_type": "kaggle_tabular",
            "dataset_profile": {"task_id": result["task_id"]},
            "method": result.get("branch_type"),
            "what_worked": "workstation branch produced all required artifacts"
            if result.get("status") == "completed" and all(result.get("artifacts", {}).get(name, False) for name in ["metrics.json", "submission.csv", "claim_audit.json"])
            else "",
            "what_failed": result.get("blocker") or ("branch did not promote or remains proxy-only" if result.get("score_gate_decision") != "promote" else ""),
            "metric_delta": None,
            "reusable_strategy": "reuse branch only if score_promotion_gate promotes and claim_audit allows local/proxy claim",
            "failure_pattern": result.get("blocker") or (result.get("score_gate_decision") or "unknown"),
            "linked_exp_ids": [value for value in [result.get("run_dir")] if value],
        }
        records = [item for item in records if item.get("memory_id") != record["memory_id"]]
        records.append(record)
    write_json(
        MEMORY_PATH,
        {
            "schema": "academic_research_os.top30_retrospective_memory.v1",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "records": records[-500:],
        },
    )


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Top30 Workstation Execution Batch",
        "",
        f"- Batch: `{payload['batch_id']}`",
        f"- Created at: `{payload['created_at']}`",
        f"- Dry run: `{payload['dry_run']}`",
        f"- Codex role: `{payload['codex_role']}`",
        f"- Official Kaggle submit: `blocked`",
        f"- Claim boundary: {payload['claim_boundary']}",
        "",
        "## Results",
        "",
        "| task | branch | type | status | run dir | best score | score gate | rank gate |",
        "|---|---|---|---|---|---:|---|---|",
    ]
    for result in payload.get("results", []):
        lines.append(
            "| `{task_id}` | `{branch_id}` | `{branch_type}` | `{status}` | `{run_dir}` | {best_score} | `{score_gate_decision}` | `{rank_gate_decision}` |".format(
                **{
                    **result,
                    "run_dir": result.get("run_dir") or "",
                    "best_score": result.get("best_score"),
                    "score_gate_decision": result.get("score_gate_decision"),
                    "rank_gate_decision": result.get("rank_gate_decision"),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Verification Boundary",
            "",
            "- 本批次只允许通过工作站 AgentOrchestrator 发起训练；脚本本身不直接实现模型训练。",
            "- `rank_promotion_gate` 没有 Kaggle response artifact 时必须保持 `proxy_only` 或 `blocked_by_gate`。",
            "- 未达到前 30% 时只能记录为下一轮自进化输入，不能覆盖官方 best 或宣称 MLEvolve parity。",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Top30 workstation orders through AgentOrchestrator.")
    parser.add_argument("--orders", default=str(ORDERS_PATH), help="Top30 orders JSON path.")
    parser.add_argument("--tasks", default="spaceship_titanic", help="Comma separated task ids; use all for every order.")
    parser.add_argument("--max-priority", default="P0", choices=["P0", "P1", "P2", "P3"], help="Highest numeric priority to execute.")
    parser.add_argument("--max-branches", type=int, default=3, help="Max branches per task.")
    parser.add_argument("--fast", action="store_true", help="Use sampled local runs for quick supervised execution.")
    parser.add_argument("--sample-rows", type=int, default=2000)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--allow-hpc", action="store_true", help="Allow HPC-only templates if currently verified.")
    parser.add_argument("--dry-run", action="store_true", help="Only write the execution plan, do not launch workstation runs.")
    parser.add_argument("--refresh-reports", action="store_true", help="Refresh inventory/alignment leaderboard after execution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    orders_path = Path(args.orders)
    orders_payload = read_json(orders_path)
    selected_tasks = None if args.tasks.strip().lower() == "all" else {item.strip() for item in args.tasks.split(",") if item.strip()}
    batch_id = f"top30_batch_{now_stamp()}"
    results: list[BranchExecutionResult] = []

    for order in orders_payload.get("orders", []):
        task_id = str(order.get("task_id"))
        if should_skip_task(order, selected_tasks, args.max_priority):
            continue
        config = DEFAULT_CONFIGS.get(task_id)
        if not config or not (ROOT / config).exists():
            for branch in order.get("selected_branches", [])[: args.max_branches]:
                results.append(
                    BranchExecutionResult(
                        task_id=task_id,
                        branch_id=str(branch.get("branch_id")),
                        branch_type=str(branch.get("branch_type")),
                        status="blocked",
                        command=[],
                        blocker=f"Missing config mapping for {task_id}.",
                    )
                )
            continue

        before = {path.name for path in (ROOT / "experiments" / task_id).iterdir()} if (ROOT / "experiments" / task_id).exists() else set()
        for branch_index, branch in enumerate(order.get("selected_branches", [])[: args.max_branches], start=1):
            template = select_template(branch, args.allow_hpc)
            if template in HPC_TEMPLATES and not args.allow_hpc:
                template = LOCAL_TEMPLATE
            cmd = build_command(
                task_id=task_id,
                config=config,
                template=template,
                branch_index=branch_index,
                fast=args.fast,
                sample_rows=args.sample_rows,
                n_folds=args.n_folds,
                timeout_seconds=args.timeout_seconds,
                branch=branch,
            )
            if args.dry_run:
                results.append(
                    BranchExecutionResult(
                        task_id=task_id,
                        branch_id=str(branch.get("branch_id")),
                        branch_type=str(branch.get("branch_type")),
                        status="planned",
                        command=cmd,
                    )
                )
                continue
            env = os.environ.copy()
            env.setdefault("PYTHONUTF8", "1")
            env.setdefault("PYTHONIOENCODING", "utf-8")
            started = datetime.now()
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            elapsed = (datetime.now() - started).total_seconds()
            latest_run = find_latest_run_dir(task_id, before)
            if latest_run is not None:
                before.add(latest_run.name)
            summary = parse_summary(completed.stdout)
            quality = read_run_quality(latest_run)
            status = "completed" if completed.returncode == 0 else "failed"
            blocker = None if completed.returncode == 0 else f"workstation launcher returned {completed.returncode}"
            if summary.get("run_id") and latest_run is None:
                candidate = ROOT / "experiments" / task_id / str(summary["run_id"])
                latest_run = candidate if candidate.exists() else None
            results.append(
                BranchExecutionResult(
                    task_id=task_id,
                    branch_id=str(branch.get("branch_id")),
                    branch_type=str(branch.get("branch_type")),
                    status=status,
                    command=cmd,
                    returncode=completed.returncode,
                    run_dir=latest_run.relative_to(ROOT).as_posix() if latest_run else None,
                    best_score=quality.get("best_score"),
                    score_gate_decision=quality.get("score_gate_decision"),
                    rank_gate_decision=quality.get("rank_gate_decision"),
                    elapsed_seconds=elapsed,
                    blocker=blocker,
                    stdout_tail=tail(completed.stdout),
                    stderr_tail=tail(completed.stderr),
                    artifacts=artifact_state(latest_run),
                )
            )

    payload = {
        "schema": "academic_research_os.top30_workstation_execution_batch.v1",
        "batch_id": batch_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_orders": orders_path.relative_to(ROOT).as_posix() if orders_path.is_relative_to(ROOT) else str(orders_path),
        "dry_run": bool(args.dry_run),
        "codex_role": "supervisor_system_engineer_only_no_direct_training_no_direct_submit",
        "execution_subject": "scripts/run_workstation_ensemble.py -> AgentOrchestrator",
        "official_kaggle_submit": False,
        "claim_boundary": "Local/proxy evidence only. Official top30 requires Kaggle response and rank_promotion_gate top30_reached=true.",
        "parameters": {
            "tasks": args.tasks,
            "max_priority": args.max_priority,
            "max_branches": args.max_branches,
            "fast": args.fast,
            "sample_rows": args.sample_rows,
            "n_folds": args.n_folds,
            "timeout_seconds": args.timeout_seconds,
            "allow_hpc": args.allow_hpc,
        },
        "results": [asdict(result) for result in results],
    }
    out_json = OUT_DIR / f"{batch_id}.json"
    out_md = REPORTS_DIR / f"TOP30_WORKSTATION_EXECUTION_{batch_id}.md"
    write_json(out_json, payload)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_report(payload), encoding="utf-8")
    update_retrospective_memory(payload)

    if args.refresh_reports and not args.dry_run:
        for script in [
            "build_kaggle_experiment_inventory.py",
            "export_mlevolve_alignment_matrix.py",
            "build_mlebench_style_leaderboard_report.py",
        ]:
            subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=False)

    print(
        json.dumps(
            {
                "batch_id": batch_id,
                "json": out_json.relative_to(ROOT).as_posix(),
                "md": out_md.relative_to(ROOT).as_posix(),
                "results": len(results),
                "completed": len([item for item in results if item.status == "completed"]),
                "failed": len([item for item in results if item.status == "failed"]),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
