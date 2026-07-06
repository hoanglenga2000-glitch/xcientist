from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "workspace"
REPORTS = ROOT / "reports"
SRC_RESEARCH_OS = ROOT / "src" / "research_os"
PROMPTS = ROOT / "prompts" / "agents"
SCHEMAS = ROOT / "configs" / "schemas"

OUT_JSON = WORKSPACE / "workstation_learning_loop_readiness_20260630.json"
OUT_MD = REPORTS / "WORKSTATION_LEARNING_LOOP_READINESS_20260630.md"

REQUIRED_MODULES = [
    "search_graph.py",
    "retrospective_memory.py",
    "validation_contract.py",
    "claim_audit.py",
    "benchmark_manager.py",
    "mlevolve_controller.py",
    "mlevolve_adapter.py",
]

REQUIRED_PROMPTS = [
    "search_controller_mlevolve_style.md",
    "retrospective_memory_agent.md",
    "validation_contract_agent_xcientist_style.md",
    "claim_audit_agent.md",
    "report_agent_research_harness.md",
]

REQUIRED_SCHEMAS = [
    "experiment_node.schema.json",
    "search_graph.schema.json",
    "retrospective_memory.schema.json",
    "validation_contract.schema.json",
    "claim_audit.schema.json",
    "benchmark_task.schema.json",
    "benchmark_result.schema.json",
    "rank_promotion_gate.schema.json",
    "benchmark_claim_gate.schema.json",
]

EVIDENCE_FILES = {
    "experiment_inventory": WORKSPACE / "kaggle_experiment_inventory_20260624.json",
    "leaderboard": WORKSPACE / "mlebench_style_current_leaderboard_20260625.json",
    "training_progress": WORKSPACE / "workstation_training_progress_20260630.json",
    "task_api_matrix": WORKSPACE / "workstation_task_api_matrix_20260630.json",
    "next_run_queue": WORKSPACE / "workstation_next_run_queue_20260630.json",
    "cache_hit_rate": WORKSPACE / "deepseek_cache_hit_rate_target_verification_20260623.json",
    "kaggle4_self_evolution": WORKSPACE / "kaggle4_self_evolution_verification_20260624.json",
    "kaggle10_self_evolution": WORKSPACE / "kaggle_10_self_evolution_progress_20260623.json",
    "mlevolve_alignment": WORKSPACE / "mlevolve_alignment_matrix_20260625.json",
    "mlevolve_next_orders": WORKSPACE / "mlevolve_next_orders_20260625.json",
}

MEMORY_PATTERNS = [
    "experiment_memory.json",
    "memory_records.jsonl",
    "retrospective_memory_*.json",
    "kaggle_10_retrospective_memory_*.json",
]

SEARCH_ORDER_PATTERNS = [
    "*search*_orders*.json",
    "*next*_orders*.json",
    "*evolution_orders*.json",
    "round*_search_plan_*.json",
    "round*_workstation_branches_*.json",
    "robust_evolution_state_*.json",
]


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {"_missing": True, "_path": str(path)}
    except Exception as exc:
        return {"_error": str(exc), "_path": str(path)}


def count_json_records(path: Path) -> int:
    if not path.exists():
        return 0
    if path.suffix == ".jsonl":
        return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
    payload = read_json(path)
    if isinstance(payload, list):
        return len(payload)
    for key in ("records", "memories", "items", "orders", "branches", "tasks", "runs"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    return 1 if payload and not payload.get("_missing") and not payload.get("_error") else 0


def collect_pattern_files(patterns: list[str]) -> list[Path]:
    files: dict[str, Path] = {}
    for pattern in patterns:
        for path in WORKSPACE.glob(pattern):
            if path.is_file():
                files[str(path)] = path
    return sorted(files.values(), key=lambda item: item.name)


def check_required_files(base: Path, names: list[str]) -> list[dict[str, Any]]:
    checks = []
    for name in names:
        path = base / name
        checks.append({
            "name": name,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0,
        })
    return checks


def source_contains(path: Path, patterns: list[str]) -> dict[str, bool]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    return {pattern: bool(re.search(pattern, text, re.IGNORECASE)) for pattern in patterns}


def build_report() -> dict[str, Any]:
    module_checks = check_required_files(SRC_RESEARCH_OS, REQUIRED_MODULES)
    prompt_checks = check_required_files(PROMPTS, REQUIRED_PROMPTS)
    schema_checks = check_required_files(SCHEMAS, REQUIRED_SCHEMAS)

    evidence = {name: read_json(path) for name, path in EVIDENCE_FILES.items()}
    evidence_checks = [
        {
            "name": name,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "exists": path.exists(),
            "valid_json": not payload.get("_missing") and not payload.get("_error"),
            "error": payload.get("_error"),
        }
        for name, path in EVIDENCE_FILES.items()
        for payload in [evidence[name]]
    ]

    memory_files = collect_pattern_files(MEMORY_PATTERNS)
    search_order_files = collect_pattern_files(SEARCH_ORDER_PATTERNS)
    memory_record_count = sum(count_json_records(path) for path in memory_files)
    search_order_count = sum(count_json_records(path) for path in search_order_files)

    controller_path = SRC_RESEARCH_OS / "mlevolve_controller.py"
    controller_features = source_contains(controller_path, [
        "progressive_mcgs",
        "retrospective_memory",
        "adaptive_code_generation",
        "TOP30_TARGET_PERCENTILE",
        "build_search_controller_decision",
        "build_benchmark_claim_gate",
    ])

    training_summary = evidence["training_progress"].get("summary") or {}
    inventory = evidence["experiment_inventory"]
    next_queue = evidence["next_run_queue"]
    cache = evidence["cache_hit_rate"]

    cache_ratio = cache.get("observed_hit_ratio") or cache.get("cache_hit_ratio") or cache.get("measured_hit_ratio")
    if cache_ratio is None:
        cache_ratio = (cache.get("summary") or {}).get("cache_hit_ratio")
    if cache_ratio is None:
        cache_ratio = (cache.get("manifest_stats") or {}).get("observed_hit_ratio")
    if cache_ratio is None:
        cache_ratio = ((cache.get("conclusion") or {}).get("current_runtime_evidence") or {}).get("observed_hit_ratio")

    next_queue_items = next_queue.get("queue") or next_queue.get("next_run_queue") or next_queue.get("tasks") or []
    recommended = next_queue.get("recommended_first_batch") or next_queue.get("recommended_tasks") or []
    next_queue_blockers = next_queue.get("blockers") or next_queue.get("global_blockers") or []
    next_queue_status = next_queue.get("status")
    if next_queue_status is None:
        next_queue_status = "blocked" if next_queue_blockers or next_queue.get("ready_to_start_now") is False else "ready"

    failures: list[str] = []
    if any(not item["exists"] for item in module_checks):
        failures.append("missing_research_os_module")
    if any(not item["exists"] for item in prompt_checks):
        failures.append("missing_agent_prompt")
    if any(not item["exists"] for item in schema_checks):
        failures.append("missing_schema")
    if any(not item["valid_json"] for item in evidence_checks):
        failures.append("invalid_or_missing_evidence_json")
    if memory_record_count <= 0:
        failures.append("no_retrospective_memory_records")
    if search_order_count <= 0:
        failures.append("no_search_or_evolution_orders")
    if not all(controller_features.values()):
        failures.append("mlevolve_controller_feature_gap")
    if not inventory.get("total_runs_observed"):
        failures.append("no_training_run_inventory")
    if not next_queue_items and not recommended:
        failures.append("no_next_run_queue")

    blocked_by_resources = []
    readiness = read_json(WORKSPACE / "workstation_launch_readiness_20260630.json")
    for blocker in readiness.get("blockers", []):
        if blocker in {"gpu_resource_blocked", "figma_auth_blocked", "figma_access_blocked"}:
            blocked_by_resources.append(blocker)
    cache_batch_gate = cache.get("batch_generation_gate_status")
    if cache_batch_gate == "blocked_below_target":
        blocked_by_resources.append("deepseek_cache_below_80_for_batch_generation")

    status = "passed" if not failures else "failed"
    return {
        "schema": "academic_research_os.workstation_learning_loop_readiness.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "failures": failures,
        "resource_blockers": blocked_by_resources,
        "module_checks": module_checks,
        "prompt_checks": prompt_checks,
        "schema_checks": schema_checks,
        "evidence_checks": evidence_checks,
        "memory": {
            "file_count": len(memory_files),
            "record_count": memory_record_count,
            "files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in memory_files[:40]],
        },
        "search_orders": {
            "file_count": len(search_order_files),
            "record_count": search_order_count,
            "files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in search_order_files[:40]],
        },
        "mlevolve_controller_features": controller_features,
        "training_progress": {
            "tasks_with_experiments": training_summary.get("tasks_with_experiments") or inventory.get("task_count_with_experiments"),
            "observed_runs": training_summary.get("observed_runs") or inventory.get("total_runs_observed"),
            "scored_runs": training_summary.get("scored_runs") or inventory.get("total_scored_runs"),
            "promoted_runs": training_summary.get("promoted_runs") or inventory.get("total_promoted_runs"),
            "held_runs": training_summary.get("held_runs") or inventory.get("total_held_runs"),
            "official_submission_tasks": training_summary.get("official_submission_tasks"),
            "official_top30_tasks": training_summary.get("official_top30_tasks"),
            "medal_count": training_summary.get("medal_count"),
            "benchmark_claim_status": training_summary.get("benchmark_claim_status"),
        },
        "deepseek_cache": {
            "status": cache.get("status"),
            "cache_hit_ratio": cache_ratio,
            "target": cache.get("target") or cache.get("target_hit_ratio") or cache.get("target_cache_hit_ratio") or 0.8,
            "implementation_target_met": cache.get("implementation_target_met"),
            "measured_80_percent_met": cache.get("measured_80_percent_met"),
            "measured_status": cache.get("measured_status"),
            "batch_generation_gate_status": cache_batch_gate,
            "needed_perfect_local_hits_for_80_percent": cache.get("needed_perfect_local_hits_for_80_percent"),
            "cache_warmup_plan": cache.get("cache_warmup_plan"),
        },
        "next_run_queue": {
            "status": next_queue_status,
            "ready_to_start_now": next_queue.get("ready_to_start_now"),
            "blockers": next_queue_blockers,
            "queued_count": len(next_queue_items) if isinstance(next_queue_items, list) else None,
            "recommended_first_batch": recommended,
        },
        "claim_boundary": (
            "This check proves that the workstation has stored evidence for an automated learning loop: "
            "training inventory, retrospective memory, MLEvolve-style search orders, benchmark gates, and next-run queue. "
            "It does not start training, use GPU resources, call Kaggle submit, or claim medals without official evidence."
        ),
    }


def write_markdown(report: dict[str, Any]) -> None:
    progress = report["training_progress"]
    lines = [
        "# 工作站自动化学习闭环 Readiness",
        "",
        f"- 生成时间：`{report['created_at']}`",
        f"- 总状态：`{report['status']}`",
        f"- 失败项：`{', '.join(report['failures']) or 'none'}`",
        f"- 外部资源阻断：`{', '.join(report['resource_blockers']) or 'none'}`",
        "",
        "## 训练进度",
        "",
        f"- 已有实验任务：`{progress.get('tasks_with_experiments')}`",
        f"- observed runs：`{progress.get('observed_runs')}`",
        f"- scored runs：`{progress.get('scored_runs')}`",
        f"- promoted / held：`{progress.get('promoted_runs')}` / `{progress.get('held_runs')}`",
        f"- 官方提交任务：`{progress.get('official_submission_tasks')}`",
        f"- 官方 top30 任务：`{progress.get('official_top30_tasks')}`",
        f"- medal count：`{progress.get('medal_count')}`",
        f"- benchmark claim：`{progress.get('benchmark_claim_status')}`",
        "",
        "## 记忆与搜索",
        "",
        f"- retrospective memory 文件数：`{report['memory']['file_count']}`",
        f"- retrospective memory 记录数：`{report['memory']['record_count']}`",
        f"- search/evolution order 文件数：`{report['search_orders']['file_count']}`",
        f"- search/evolution order 记录数：`{report['search_orders']['record_count']}`",
        "",
        "## MLEvolve Controller 特征",
        "",
        "| feature | present |",
        "| --- | --- |",
    ]
    for key, value in report["mlevolve_controller_features"].items():
        lines.append(f"| `{key}` | `{value}` |")

    cache = report["deepseek_cache"]
    queue = report["next_run_queue"]
    lines.extend([
        "",
        "## DeepSeek 缓存与下一轮队列",
        "",
        f"- 缓存验证状态：`{cache.get('status')}`",
        f"- 缓存命中率：`{cache.get('cache_hit_ratio')}`",
        f"- 缓存目标：`{cache.get('target')}`",
        f"- 批量生成 Gate：`{cache.get('batch_generation_gate_status')}`",
        f"- 达到 80% 仍需本地命中：`{cache.get('needed_perfect_local_hits_for_80_percent')}`",
        f"- 下一轮队列状态：`{queue.get('status')}`",
        f"- 当前是否可启动：`{queue.get('ready_to_start_now')}`",
        f"- 队列阻断：`{', '.join(queue.get('blockers') or []) or 'none'}`",
        f"- queued count：`{queue.get('queued_count')}`",
        "",
        "## 必备文件检查",
        "",
        "| group | name | exists | size |",
        "| --- | --- | --- | ---: |",
    ])
    for group, items in [
        ("module", report["module_checks"]),
        ("prompt", report["prompt_checks"]),
        ("schema", report["schema_checks"]),
    ]:
        for item in items:
            lines.append(f"| {group} | `{item['name']}` | `{item['exists']}` | {item['size']} |")

    lines.extend([
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify automated learning-loop readiness without starting training.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    report = build_report()
    if args.write_report:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(report)

    print(json.dumps({
        "status": report["status"],
        "failures": report["failures"],
        "resource_blockers": report["resource_blockers"],
        "memory_records": report["memory"]["record_count"],
        "search_order_records": report["search_orders"]["record_count"],
        "observed_runs": report["training_progress"]["observed_runs"],
        "next_run_ready": report["next_run_queue"]["ready_to_start_now"],
        "json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
        "md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
