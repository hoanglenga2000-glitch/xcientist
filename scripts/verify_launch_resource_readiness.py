from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


TASKS = {
    "house_prices": {
        "config": ROOT / "configs" / "house_prices.yaml",
        "experiment_root": ROOT / "experiments" / "house_prices",
        "required_inputs": [
            ROOT / "tasks" / "house_prices" / "overview.txt",
            ROOT / "tasks" / "house_prices" / "data" / "train.csv",
            ROOT / "tasks" / "house_prices" / "data" / "test.csv",
            ROOT / "tasks" / "house_prices" / "data" / "sample_submission.csv",
        ],
        "metric_checks": [
            ("cv_rmsle_mean", "<=", 0.18),
            ("holdout_rmsle", "<=", 0.20),
            ("submission_rows", "==", 1459),
        ],
    },
    "titanic": {
        "config": ROOT / "configs" / "titanic.yaml",
        "experiment_root": ROOT / "experiments" / "titanic",
        "required_inputs": [
            ROOT / "tasks" / "titanic" / "overview.txt",
            ROOT / "tasks" / "titanic" / "data" / "train.csv",
            ROOT / "tasks" / "titanic" / "data" / "test.csv",
            ROOT / "tasks" / "titanic" / "data" / "sample_submission.csv",
        ],
        "metric_checks": [
            ("cv_accuracy_mean", ">=", 0.78),
        ],
    },
    "telco_churn": {
        "config": ROOT / "configs" / "telco_churn.yaml",
        "experiment_root": ROOT / "experiments" / "telco_churn",
        "required_inputs": [
            ROOT / "tasks" / "telco_churn" / "overview.txt",
            ROOT / "tasks" / "telco_churn" / "data" / "train.csv",
            ROOT / "tasks" / "telco_churn" / "data" / "test.csv",
            ROOT / "tasks" / "telco_churn" / "data" / "sample_submission.csv",
        ],
        "metric_checks": [
            ("cv_accuracy_mean", ">=", 0.78),
        ],
    },
}


CODE_AGENT_ENV = ["DEEPSEEK_API_KEY or ANTHROPIC_API_KEY"]
GPU_ENV = ["GPU_SSH_HOST", "GPU_SSH_USER", "GPU_SSH_PASSWORD or GPU_SSH_KEY_PATH", "GPU_REMOTE_WORKSPACE"]
KAGGLE_ENV = ["KAGGLE_USERNAME", "KAGGLE_KEY"]
HPC_PROBE = ROOT / "workspace" / "hpc" / "web_terminal_probe.txt"
LOCAL_READY_ARTIFACTS = [
    "validation_gate.json",
    "experiment_log.json",
    "workflow_stage_audit.json",
    "submission.csv",
    "model_results.json",
]


def read_file_if_present(file_path: str | None) -> str:
    if not file_path:
        return ""
    try:
        return Path(file_path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def secret_dir_file(names: list[str]) -> Path | None:
    secret_dir = os.environ.get("WORKSTATION_SECRET_DIR")
    if not secret_dir:
        return None
    for name in names:
        candidate = Path(secret_dir) / name
        if candidate.exists():
            return candidate
    return None


def secret_value(key: str, aliases: list[str] | None = None) -> str:
    keys = [key, *(aliases or [])]
    for candidate in keys:
        direct = os.environ.get(candidate)
        if direct:
            return direct
        file_value = read_file_if_present(os.environ.get(f"{candidate}_FILE"))
        if file_value:
            return file_value
    dir_file = secret_dir_file(keys)
    return read_file_if_present(str(dir_file) if dir_file else None)


def secret_path(key: str, names: list[str] | None = None) -> str:
    direct = os.environ.get(key) or os.environ.get(f"{key}_FILE")
    if direct:
        return direct
    dir_file = secret_dir_file([key, *(names or [])])
    return str(dir_file) if dir_file else ""


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def hpc_probe_status() -> dict[str, Any]:
    if not HPC_PROBE.is_file():
        return {
            "configured": False,
            "path": str(HPC_PROBE.relative_to(ROOT)),
            "state": "missing_web_terminal_probe",
            "required_keys": [],
            "missing_keys": ["workspace/hpc/web_terminal_probe.txt"],
            "detail": "No Web Terminal nvidia-smi evidence file is present.",
        }
    text = HPC_PROBE.read_text(encoding="utf-8", errors="replace")
    required = ["whoami", "hostname", "pwd", "Python", "NVIDIA-SMI", "Filesystem", "Mem:"]
    missing = [term for term in required if term not in text]
    a800_hits = text.count("NVIDIA A800") + text.count("NVIDIAA800") + text.count("A800-SXM4")
    return {
        "configured": not missing and a800_hits >= 4,
        "path": str(HPC_PROBE.relative_to(ROOT)),
        "state": "gpu_verified_via_login_node_web_terminal" if not missing and a800_hits >= 4 else "incomplete_web_terminal_probe",
        "required_keys": [],
        "missing_keys": missing,
        "missing_terms": missing,
        "a800_text_hits": a800_hits,
        "detail": "4 x NVIDIA A800 is proven by Web Terminal evidence." if not missing and a800_hits >= 4 else "GPU evidence is present but incomplete.",
    }


def latest_experiment(root: Path) -> Path | None:
    if not root.exists():
        return None
    runs = sorted(path for path in root.iterdir() if path.is_dir())
    for run_dir in reversed(runs):
        if all((run_dir / name).exists() and (run_dir / name).stat().st_size > 0 for name in LOCAL_READY_ARTIFACTS):
            return run_dir
    return runs[-1] if runs else None


def metric_pass(value: Any, operator: str, threshold: float) -> bool:
    if not isinstance(value, (int, float)):
        return False
    if operator == "<=":
        return value <= threshold
    if operator == ">=":
        return value >= threshold
    if operator == "==":
        return value == threshold
    raise ValueError(f"unsupported operator: {operator}")


def env_status(keys: list[str], aliases: dict[str, list[str]] | None = None, path_keys: dict[str, list[str]] | None = None) -> dict[str, Any]:
    aliases = aliases or {}
    path_keys = path_keys or {}
    missing = []
    for key in keys:
        if key in path_keys:
            configured = bool(secret_path(key, path_keys[key]))
        else:
            configured = bool(secret_value(key, aliases.get(key, [])))
        if not configured:
            missing.append(key)
    return {
        "configured": not missing,
        "required_keys": keys,
        "missing_keys": missing,
        "accepted_secret_sources": ["direct env", "*_FILE", "WORKSTATION_SECRET_DIR"],
    }


def code_agent_status() -> dict[str, Any]:
    configured = bool(secret_value("DEEPSEEK_API_KEY") or secret_value("ANTHROPIC_API_KEY", ["CLAUDE_API_KEY"]))
    return {
        "configured": configured,
        "required_keys": CODE_AGENT_ENV,
        "missing_keys": [] if configured else CODE_AGENT_ENV,
        "accepted_secret_sources": ["direct env", "*_FILE", "WORKSTATION_SECRET_DIR"],
    }


def gpu_gateway_status() -> dict[str, Any]:
    missing = []
    for key in ["GPU_SSH_HOST", "GPU_SSH_USER", "GPU_REMOTE_WORKSPACE"]:
        if not secret_value(key):
            missing.append(key)
    if not (secret_value("GPU_SSH_PASSWORD", ["HPC_SSH_PASSWORD"]) or secret_path("GPU_SSH_KEY_PATH", ["GPU_SSH_PRIVATE_KEY", "gpu_ssh_private_key", "id_rsa"])):
        missing.append("GPU_SSH_PASSWORD or GPU_SSH_KEY_PATH")
    return {
        "configured": not missing,
        "required_keys": GPU_ENV,
        "missing_keys": missing,
        "accepted_secret_sources": ["direct env", "*_FILE", "WORKSTATION_SECRET_DIR"],
    }


def inspect_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = latest_experiment(spec["experiment_root"])
    validation = read_json(run_dir / "validation_gate.json") if run_dir else None
    experiment_log = read_json(run_dir / "experiment_log.json") if run_dir else None

    metrics = {}
    if validation and isinstance(validation.get("metrics"), dict):
        metrics.update(validation["metrics"])
    if validation:
        for key in ("cv_rmsle_mean", "holdout_rmsle", "submission_rows", "cv_accuracy_mean", "holdout_accuracy"):
            if key in validation:
                metrics[key] = validation[key]
    if experiment_log and isinstance(experiment_log.get("best_metrics"), dict):
        metrics.update(experiment_log["best_metrics"])

    metric_checks = []
    for metric_name, operator, threshold in spec["metric_checks"]:
        value = metrics.get(metric_name)
        metric_checks.append(
            {
                "metric": metric_name,
                "operator": operator,
                "threshold": threshold,
                "value": value,
                "passed": metric_pass(value, operator, threshold),
            }
        )

    required_artifacts = []
    if run_dir:
        for name in [
            "validation_gate.json",
            "experiment_log.json",
            "workflow_stage_audit.json",
            "submission.csv",
            "model_results.json",
        ]:
            required_artifacts.append({"path": str((run_dir / name).relative_to(ROOT)), "exists": (run_dir / name).exists()})

    missing_inputs = [str(path.relative_to(ROOT)) for path in spec["required_inputs"] if not path.exists()]
    missing_artifacts = [item["path"] for item in required_artifacts if not item["exists"]]
    validation_passed = bool(validation and validation.get("status") == "passed")

    return {
        "task_id": task_id,
        "config_exists": spec["config"].exists(),
        "latest_experiment": str(run_dir.relative_to(ROOT)) if run_dir else None,
        "data_ready": not missing_inputs,
        "missing_inputs": missing_inputs,
        "validation_gate_passed": validation_passed,
        "metrics": metrics,
        "metric_checks": metric_checks,
        "required_artifacts": required_artifacts,
        "artifact_ready": not missing_artifacts,
        "missing_artifacts": missing_artifacts,
        "ready_for_local_training": bool(not missing_inputs and validation_passed and not missing_artifacts and all(item["passed"] for item in metric_checks)),
    }


def write_markdown(report: dict[str, Any], target: Path) -> None:
    lines = [
        "# 科研 Agent 工作站上线资源就绪审计",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 总体状态：{report['overall_status']}",
        f"- 本地 Kaggle 风格训练闭环：{report['local_training_status']}",
        "",
        "## 结论",
        "",
        report["conclusion"],
        "",
        "## 外部资源状态",
        "",
    ]
    for key, item in report["external_resources"].items():
        lines.append(f"- {key}: {'已配置' if item['configured'] else '未配置'}")
        if item["missing_keys"]:
            lines.append(f"  缺少：{', '.join(item['missing_keys'])}")
    lines.extend(
        [
            "",
            "## Kaggle 数据训练任务",
            "",
        ]
    )
    for task in report["tasks"]:
        lines.append(f"### {task['task_id']}")
        lines.append(f"- 最新实验：{task['latest_experiment']}")
        lines.append(f"- 数据就绪：{task['data_ready']}")
        lines.append(f"- Validation Gate：{task['validation_gate_passed']}")
        lines.append(f"- 本地训练就绪：{task['ready_for_local_training']}")
        for check in task["metric_checks"]:
            lines.append(
                f"- 指标 {check['metric']} {check['operator']} {check['threshold']}："
                f" 当前 {check['value']}，{'通过' if check['passed'] else '未通过'}"
            )
        if task["missing_inputs"]:
            lines.append(f"- 缺失输入：{', '.join(task['missing_inputs'])}")
        if task["missing_artifacts"]:
            lines.append(f"- 缺失产物：{', '.join(task['missing_artifacts'])}")
        lines.append("")

    lines.extend(
        [
            "## 配置后直接执行",
            "",
            "1. 配置 Claude：`ANTHROPIC_API_KEY`。",
            "2. 配置 GPU SSH：`GPU_SSH_HOST`、`GPU_SSH_USER`、`GPU_SSH_PASSWORD` 或 `GPU_SSH_KEY_PATH`、`GPU_REMOTE_WORKSPACE`。",
            "3. 可选配置 Kaggle 官方下载/提交：`KAGGLE_USERNAME`、`KAGGLE_KEY`。",
            "4. 重新运行：`python scripts\\verify_launch_resource_readiness.py --write-report`。",
            "5. 浏览器打开 `http://127.0.0.1:8088`，在 Code Runner 中启动 Claude Session 或 GPU Job。",
            "",
            "说明：Kaggle token 只影响官方 API 下载/leaderboard 提交；当前本地训练使用已验证的 Kaggle 风格输入文件，因此不把 Kaggle token 计入本轮最后两项阻塞资源。",
            "",
        ]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify launch readiness for Kaggle-style training and external resources.")
    parser.add_argument("--write-report", action="store_true", help="Write JSON and Markdown readiness reports under docs/.")
    args = parser.parse_args()

    tasks = [inspect_task(task_id, spec) for task_id, spec in TASKS.items()]
    local_ready = all(task["ready_for_local_training"] for task in tasks)
    hpc_verified = hpc_probe_status()
    external_resources = {
        "code_agent": code_agent_status(),
        "gpu_ssh_gateway": gpu_gateway_status(),
        "hpc_gpu_verified_container": hpc_verified,
        "kaggle_official_api_optional": env_status(KAGGLE_ENV),
    }
    required_external_ready = external_resources["code_agent"]["configured"] and external_resources["gpu_ssh_gateway"]["configured"]
    missing_required_external = (
        external_resources["code_agent"]["missing_keys"]
        + external_resources["gpu_ssh_gateway"]["missing_keys"]
    )
    conclusion = (
        "本地 Kaggle 风格数据训练、指标阈值、submission 和审计产物已经就绪；"
        "GPU 容器硬件已由 Web Terminal 证据证明；正式进入外部增强训练/自动代码优化前，还缺 Claude API Key 与 GPU 自动作业凭据。"
        if local_ready and not required_external_ready and hpc_verified["configured"]
        else "本地 Kaggle 风格数据训练、指标阈值、submission 和审计产物已经就绪；"
        "正式进入外部增强训练/自动代码优化前，只缺 Claude API Key 与 GPU SSH 资源。"
        if local_ready and not required_external_ready
        else "本地训练或外部资源仍有未满足项，请查看下方明细。"
    )
    if local_ready and required_external_ready:
        conclusion = "本地训练与两项外部资源均已就绪，可以启动 Claude Code 与 GPU SSH 增强训练流程。"

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_status": "ready_for_external_resources" if local_ready and not required_external_ready else "fully_ready" if local_ready and required_external_ready else "not_ready",
        "local_training_status": "ready" if local_ready else "not_ready",
        "required_external_status": "ready" if required_external_ready else "missing",
        "missing_required_external_keys": missing_required_external,
        "external_resources": external_resources,
        "tasks": tasks,
        "conclusion": conclusion,
    }

    if args.write_report:
        json_path = ROOT / "docs" / "launch_resource_readiness.json"
        md_path = ROOT / "docs" / "上线资源接入与Kaggle训练就绪审计.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(report, md_path)
        report["report_paths"] = {
            "json": str(json_path.relative_to(ROOT)),
            "markdown": str(md_path.relative_to(ROOT)),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not local_ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
