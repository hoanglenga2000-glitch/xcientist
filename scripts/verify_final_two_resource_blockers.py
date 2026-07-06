from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_BLOCKER_GROUPS = {
    "code_agent": ["DEEPSEEK_API_KEY or ANTHROPIC_API_KEY"],
    "gpu_ssh_gateway": ["GPU_SSH_HOST", "GPU_SSH_USER", "GPU_SSH_PASSWORD or GPU_SSH_KEY_PATH", "GPU_REMOTE_WORKSPACE"],
}


def fail(message: str) -> None:
    raise SystemExit(f"FINAL_TWO_RESOURCE_BLOCKERS_FAILED: {message}")


def run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        fail(
            json.dumps(
                {
                    "command": " ".join(command),
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
                ensure_ascii=False,
            )
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        fail(f"could not parse JSON from {' '.join(command)}: {error}\n{completed.stdout}")
    raise AssertionError("unreachable")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def blocker_groups_from_launch(launch: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    external = launch.get("external_resources") or {}
    for group, required_keys in ALLOWED_BLOCKER_GROUPS.items():
        item = external.get(group) or {}
        missing = item.get("missing_keys") or []
        if missing:
            groups.append(
                {
                    "group": group,
                    "missing_keys": missing,
                    "allowed": sorted(missing) == sorted(required_keys),
                }
            )
    unexpected = [
        key
        for key, item in external.items()
        if key not in {*ALLOWED_BLOCKER_GROUPS.keys(), "kaggle_official_api_optional"}
        and (item or {}).get("missing_keys")
    ]
    if unexpected:
        groups.append({"group": "unexpected_external_resources", "missing_keys": unexpected, "allowed": False})
    return groups


def gateway_matches_launch(gateways: dict[str, Any] | None, launch_groups: list[dict[str, Any]]) -> bool:
    if not gateways:
        return True
    missing_by_group = {group["group"]: sorted(group["missing_keys"]) for group in launch_groups}
    code_agent = gateways.get("code_agent") or gateways.get("claude") or {}
    gpu = gateways.get("gpu") or {}
    if code_agent.get("configured") is False and sorted(code_agent.get("missing_env") or []) != missing_by_group.get("code_agent", []):
        return False
    if gpu.get("configured") is False and sorted(gpu.get("missing_env") or []) != missing_by_group.get("gpu_ssh_gateway", []):
        return False
    return True


def write_markdown(report: dict[str, Any], target: Path) -> None:
    lines = [
        "# 最终两类外部资源阻塞审计",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 总体状态：{report['overall_status']}",
        f"- 本地训练状态：{report['local_training_status']}",
        f"- 训练优化完成率：{report['training_completion_rate_percent']}%",
        f"- 阻塞资源组数：{report['blocker_group_count']}",
        "",
        "## 阻塞资源",
        "",
    ]
    if report["blocker_groups"]:
        for group in report["blocker_groups"]:
            lines.append(f"- {group['group']}：{', '.join(group['missing_keys'])}")
    else:
        lines.append("- 无。Claude 与 GPU SSH 均已配置或已通过真实 smoke test。")

    lines.extend(
        [
            "",
            "## Kaggle 训练就绪证据",
            "",
        ]
    )
    for task in report["tasks"]:
        lines.append(
            f"- {task['task_id']}：ready={task['ready']}，best_model={task.get('best_model')}，"
            f"latest={task.get('latest_experiment')}"
        )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            report["conclusion"],
            "",
        ]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate readiness evidence and verify there are no blockers beyond Claude API and GPU SSH.")
    parser.add_argument("--dashboard-url", default=None, help="Optional dashboard URL to verify live Claude/GPU gateway endpoints.")
    parser.add_argument("--container-name", default=None, help="Optional Docker container name for gateway artifact checks.")
    parser.add_argument("--write-report", action="store_true", help="Write final JSON and Markdown audit under docs/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launch = run_json([sys.executable, "scripts/verify_launch_resource_readiness.py"])
    training = run_json([sys.executable, "scripts/verify_training_optimization_readiness.py"])
    gateways = None
    if args.dashboard_url:
        gateway_command = [sys.executable, "scripts/verify_external_resource_gateways.py", "--url", args.dashboard_url]
        if args.container_name:
            gateway_command.extend(["--container-name", args.container_name])
        gateways = run_json(gateway_command)

    local_ready = launch.get("local_training_status") == "ready"
    training_ready = training.get("overall_status") == "passed" and training.get("completion_rate_percent") == 100.0
    blocker_groups = blocker_groups_from_launch(launch)
    allowed_blockers_only = all(group.get("allowed") for group in blocker_groups)
    live_gateway_consistent = gateway_matches_launch(gateways, blocker_groups)
    unexpected_required_external = [
        key
        for key in launch.get("missing_required_external_keys", [])
        if key not in {item for values in ALLOWED_BLOCKER_GROUPS.values() for item in values}
    ]
    kaggle = ((launch.get("external_resources") or {}).get("kaggle_official_api_optional") or {})
    kaggle_optional_not_blocking = bool(kaggle) and set(kaggle.get("missing_keys") or []).issubset({"KAGGLE_USERNAME", "KAGGLE_KEY"})

    task_map = {task["task_id"]: task for task in training.get("tasks", [])}
    tasks = []
    for task_id in ["house_prices", "titanic", "telco_churn"]:
        task = task_map.get(task_id, {})
        tasks.append(
            {
                "task_id": task_id,
                "ready": bool(task.get("ready_for_optimized_local_training")),
                "latest_experiment": task.get("latest_experiment"),
                "best_model": task.get("best_model_recorded"),
                "best_model_selection_valid": task.get("best_model_selection_valid"),
                "candidate_model_count": task.get("candidate_model_count"),
            }
        )

    passed = (
        local_ready
        and training_ready
        and allowed_blockers_only
        and not unexpected_required_external
        and kaggle_optional_not_blocking
        and live_gateway_consistent
        and all(task["ready"] for task in tasks)
    )
    blocker_group_count = len(blocker_groups)
    conclusion = (
        "当前系统已证明本地 Kaggle 风格训练、模型选择和完成率均达标；除 Claude API Key 与 GPU SSH 凭证两类外部资源外，没有发现其他阻塞正式增强训练的前置条件。"
        if passed and blocker_group_count
        else "当前系统已证明本地训练与外部资源 smoke test 均满足要求；没有剩余外部资源阻塞。"
        if passed
        else "最终两类资源阻塞审计未通过，请查看 blocker_groups、unexpected_required_external 或 live_gateway_consistent。"
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_status": "passed" if passed else "failed",
        "local_training_status": launch.get("local_training_status"),
        "training_completion_rate_percent": training.get("completion_rate_percent"),
        "ready_task_count": training.get("ready_task_count"),
        "required_task_count": training.get("required_task_count"),
        "blocker_group_count": blocker_group_count,
        "blocker_groups": blocker_groups,
        "allowed_blocker_groups": ALLOWED_BLOCKER_GROUPS,
        "unexpected_required_external": unexpected_required_external,
        "kaggle_optional_not_blocking": kaggle_optional_not_blocking,
        "live_gateway_checked": gateways is not None,
        "live_gateway_consistent": live_gateway_consistent,
        "gateway_status": {
            "code_agent": (gateways or {}).get("code_agent") or (gateways or {}).get("claude"),
            "gpu": (gateways or {}).get("gpu"),
        },
        "tasks": tasks,
        "conclusion": conclusion,
    }

    if args.write_report:
        json_path = ROOT / "docs" / "final_two_resource_blockers.json"
        md_path = ROOT / "docs" / "最终两类外部资源阻塞审计.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(report, md_path)
        report["report_paths"] = {"json": rel(json_path), "markdown": rel(md_path)}

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
