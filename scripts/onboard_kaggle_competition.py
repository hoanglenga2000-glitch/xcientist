from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def slug_to_task_id(slug: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in slug.lower()).strip("_")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def kaggle_env_configured() -> bool:
    return bool(os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")))


def kaggle_auth_mode() -> str:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "access_token"
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "legacy_username_key"
    return "not_configured"


def kaggle_cli_available() -> bool:
    return shutil.which("kaggle") is not None or shutil.which("kaggle.exe") is not None or importlib.util.find_spec("kaggle") is not None


def kaggle_command() -> list[str]:
    executable = shutil.which("kaggle") or shutil.which("kaggle.exe")
    if executable:
        return [executable]
    if importlib.util.find_spec("kaggle") is not None:
        return [sys.executable, "-m", "kaggle"]
    return []


def extract_downloaded_archives(target_dir: Path) -> list[str]:
    extracted: list[str] = []
    for archive in target_dir.glob("*.zip"):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.namelist():
                if member.endswith("/"):
                    continue
                handle.extract(member, target_dir)
                extracted.append(member)
    return extracted


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_kaggle_download(slug: str, target_dir: Path) -> dict[str, Any]:
    if not kaggle_env_configured():
        return {
            "attempted": False,
            "status": "not_configured",
            "auth_mode": "not_configured",
            "reason": "KAGGLE_API_TOKEN or KAGGLE_USERNAME/KAGGLE_KEY is required for official competition download.",
        }
    command_prefix = kaggle_command()
    if not command_prefix:
        return {
            "attempted": False,
            "status": "cli_missing",
            "reason": "kaggle CLI or Python package is not installed.",
        }
    target_dir.mkdir(parents=True, exist_ok=True)
    command = [*command_prefix, "competitions", "download", "-c", slug, "-p", str(target_dir), "--force"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300)
    extracted = extract_downloaded_archives(target_dir) if completed.returncode == 0 else []
    return {
        "attempted": True,
        "status": "passed" if completed.returncode == 0 else "failed",
        "auth_mode": kaggle_auth_mode(),
        "command": " ".join(command),
        "extracted_files": extracted,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def copy_local_inputs(data_dir: Path, target_data_dir: Path) -> dict[str, str]:
    required = {
        "train": "train.csv",
        "test": "test.csv",
        "sample_submission": "sample_submission.csv",
    }
    target_data_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for key, filename in required.items():
        source = data_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing {filename} under {data_dir}")
        target = target_data_dir / filename
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied[key] = rel(target)
    for optional in ["overview.txt", "data_description.txt"]:
        source = data_dir / optional
        if source.is_file():
            target = target_data_dir / optional
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
    return copied


def copied_file_manifest(copied: dict[str, str]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for label, relative in copied.items():
        path = ROOT / relative
        files[label] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return files


def infer_target(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame, requested_target: str | None) -> str:
    if requested_target:
        if requested_target not in train.columns:
            raise ValueError(f"Requested target column is not in train.csv: {requested_target}")
        return requested_target
    train_only = [col for col in train.columns if col not in test.columns]
    if len(train_only) == 1:
        return train_only[0]
    prediction_column = sample.columns[1] if len(sample.columns) > 1 else None
    if prediction_column and prediction_column in train.columns:
        return prediction_column
    raise ValueError("Could not infer target column; pass --target.")


def infer_task_type(target: pd.Series, requested_type: str | None) -> str:
    if requested_type:
        return requested_type
    unique_count = int(target.nunique(dropna=True))
    unique_ratio = unique_count / max(len(target), 1)
    if pd.api.types.is_numeric_dtype(target) and unique_count > 20 and unique_ratio > 0.02:
        return "regression"
    return "classification"


def infer_metric(task_type: str, target: pd.Series, requested_metric: str | None) -> str:
    if requested_metric:
        return requested_metric.lower()
    if task_type == "classification":
        return "accuracy"
    if pd.api.types.is_numeric_dtype(target) and float(target.min()) >= 0 and target.skew() > 1.0:
        return "rmsle"
    return "rmse"


def candidate_models(task_type: str, metric: str) -> list[str]:
    if task_type == "classification":
        return ["logistic_regression", "random_forest", "extra_trees", "gradient_boosting"]
    return ["ridge_log_target", "random_forest_log_target", "extra_trees_log_target", "gradient_boosting_log_target"]


def build_config(
    task_id: str,
    slug: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    target: str,
    task_type: str,
    metric: str,
    copied: dict[str, str],
    task_dir: Path,
) -> dict[str, Any]:
    sample_columns = sample.columns.tolist()
    thresholds: dict[str, Any] = {
        "require_submission_schema_valid": True,
        "require_no_missing_predictions": True,
        "require_train_test_features_match": True,
        "expected_submission_rows": int(len(sample)),
        "expected_submission_columns": sample_columns,
    }
    if metric == "rmsle":
        thresholds["require_positive_predictions"] = True
    if task_type == "classification":
        observed = sorted(train[target].dropna().unique().tolist())
        if len(observed) <= 20:
            thresholds["allowed_prediction_values"] = [value.item() if hasattr(value, "item") else value for value in observed]

    return {
        "task": {
            "name": task_id,
            "competition": slug,
            "type": task_type,
            "target": target,
            "metric": metric,
            "id_column": sample_columns[0],
            "prediction_column": sample_columns[1] if len(sample_columns) > 1 else "prediction",
        },
        "data": {
            "task_dir": rel(task_dir),
            "train": copied["train"],
            "test": copied["test"],
            "sample_submission": copied["sample_submission"],
            "overview": rel(task_dir / "overview.txt"),
            "data_source": rel(task_dir / "data" / "DATA_SOURCE.md"),
        },
        "agent_templates": "configs/agent_templates.yaml",
        "workflow": [
            "task_understanding",
            "preliminary_eda",
            "data_quality_check",
            "feature_engineering",
            "model_validation",
            "submission_generation",
            "report_and_review",
        ],
        "thresholds": thresholds,
        "scaffold": {
            "time_budget_minutes": 20,
            "validation_strategy": "5-fold local cross-validation plus holdout validation; official Kaggle leaderboard remains gated.",
            "first_stage_models": candidate_models(task_type, metric),
            "risk_points": [
                "Metric and target inference must be reviewed before official submission.",
                "Local CV is a proxy and may not match hidden leaderboard distribution.",
                "Submission schema must match sample_submission.csv exactly.",
                "Official Kaggle download/submission requires credentials and a human gate.",
            ],
        },
        "feature_engineering": {
            "preset": "generic_tabular",
            "target_transform": "log1p" if metric == "rmsle" else None,
            "drop_columns": [sample_columns[0]] if sample_columns and sample_columns[0] in train.columns else [],
        },
    }


def write_readiness_report(report: dict[str, Any], task_dir: Path) -> None:
    workspace_path = ROOT / "workspace" / "kaggle_onboarding" / f"{report['task_id']}_readiness.json"
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path = ROOT / "docs" / "kaggle_new_competition_readiness.json"
    md_path = ROOT / "docs" / "Kaggle新比赛接入就绪报告.md"
    report["report_paths"] = {
        "workspace_json": rel(workspace_path),
        "docs_json": rel(docs_path),
        "docs_markdown": rel(md_path),
        "task_json": rel(task_dir / "onboarding_report.json"),
    }
    lines = [
        "# Kaggle 新比赛接入就绪报告",
        "",
        f"- task_id: `{report['task_id']}`",
        f"- competition_slug: `{report['competition_slug']}`",
        f"- generated_config: `{report['config_path']}`",
        f"- task_type: `{report['task_type']}`",
        f"- target: `{report['target']}`",
        f"- metric: `{report['metric']}`",
        f"- official_download_status: `{report['official_download']['status']}`",
        f"- local_baseline_ready: `{report['local_baseline_ready']}`",
        "",
        "## 能力结论",
        "",
        report["conclusion"],
        "",
        "## 上线说明",
        "",
        "- 没有 Kaggle token 时，系统仍可用本地上传或镜像数据完成 baseline、submission 和审计链路。",
        "- 有 Kaggle token 后，可切换到官方下载和人工 Gate 后提交。",
        "- 有 GPU 后，可把同一配置提交到 SSH GPU 网关做 seed sweep、超参搜索或更重模型。",
    ]
    workspace_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    docs_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    (task_dir / "onboarding_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_blocked_download_report(task_id: str, slug: str, task_dir: Path, official_download: dict[str, Any]) -> dict[str, Any]:
    task_dir.mkdir(parents=True, exist_ok=True)
    data_dir = task_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    overview_path = task_dir / "overview.txt"
    overview_path.write_text(
        "\n".join([
            f"Competition: {slug}",
            f"Task id: {task_id}",
            "Status: official_download_blocked",
            "This Kaggle competition requires the account to join/accept rules before files can be downloaded.",
        ]),
        encoding="utf-8",
    )
    (data_dir / "DATA_SOURCE.md").write_text(
        "\n".join([
            "# Data Source",
            "",
            f"- competition_slug: `{slug}`",
            f"- official_download_status: `{official_download.get('status')}`",
            "- local files are not present yet.",
            "- Accept the competition rules in Kaggle, then rerun the generated onboarding command.",
        ]),
        encoding="utf-8",
    )
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "blocked_official_download",
        "task_id": task_id,
        "competition_slug": slug,
        "config_path": None,
        "task_dir": rel(task_dir),
        "target": None,
        "task_type": None,
        "metric": None,
        "train_rows": 0,
        "test_rows": 0,
        "sample_submission_rows": 0,
        "official_download": official_download,
        "kaggle_cli_available": kaggle_cli_available(),
        "kaggle_env_configured": kaggle_env_configured(),
        "kaggle_auth_mode": kaggle_auth_mode(),
        "data_files": {},
        "local_baseline_ready": False,
        "next_commands": [
            f"Open https://www.kaggle.com/competitions/{slug} and accept/join the competition rules.",
            f"python scripts/onboard_kaggle_competition.py --competition-slug {slug} --task-id {task_id} --use-kaggle-api",
        ],
        "conclusion": "Kaggle API authentication is working, but the competition files are blocked until the account accepts the competition rules. No leaderboard submission was attempted.",
    }
    write_readiness_report(report, task_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a runnable workstation task from a Kaggle competition or local Kaggle-style files.")
    parser.add_argument("--competition-slug", required=True, help="Kaggle competition slug, for example playground-series-s5e6.")
    parser.add_argument("--task-id", default=None, help="Local task id. Defaults to a slug-derived id.")
    parser.add_argument("--data-dir", default=None, help="Directory containing train.csv, test.csv and sample_submission.csv.")
    parser.add_argument("--target", default=None)
    parser.add_argument("--task-type", choices=["classification", "regression"], default=None)
    parser.add_argument("--metric", default=None, help="accuracy, rmsle, rmse or mae. Defaults from target profile.")
    parser.add_argument("--use-kaggle-api", action="store_true", help="Attempt official Kaggle download before using local files.")
    args = parser.parse_args()

    task_id = args.task_id or slug_to_task_id(args.competition_slug)
    task_dir = ROOT / "tasks" / task_id
    target_data_dir = task_dir / "data"
    official_download = {"attempted": False, "status": "local_files", "reason": "Using local Kaggle-style files."}
    if args.use_kaggle_api:
        official_download = run_kaggle_download(args.competition_slug, target_data_dir)
        if official_download.get("status") != "passed" and not args.data_dir:
            report = write_blocked_download_report(task_id, args.competition_slug, task_dir, official_download)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            raise SystemExit(2)

    source_data_dir = Path(args.data_dir) if args.data_dir else target_data_dir
    if not source_data_dir.is_absolute():
        source_data_dir = ROOT / source_data_dir
    copied = copy_local_inputs(source_data_dir, target_data_dir)
    copied_manifest = copied_file_manifest(copied)

    train = pd.read_csv(ROOT / copied["train"])
    test = pd.read_csv(ROOT / copied["test"])
    sample = pd.read_csv(ROOT / copied["sample_submission"])
    target = infer_target(train, test, sample, args.target)
    task_type = infer_task_type(train[target], args.task_type)
    metric = infer_metric(task_type, train[target], args.metric)

    overview_path = task_dir / "overview.txt"
    overview_path.write_text(
        "\n".join([
            f"Competition: {args.competition_slug}",
            f"Task id: {task_id}",
            f"Target: {target}",
            f"Task type: {task_type}",
            f"Metric: {metric}",
            "This task was onboarded by scripts/onboard_kaggle_competition.py.",
        ]),
        encoding="utf-8",
    )
    data_source_path = target_data_dir / "DATA_SOURCE.md"
    data_source_path.write_text(
        "\n".join([
            "# Data Source",
            "",
            f"- competition_slug: `{args.competition_slug}`",
            f"- official_download_status: `{official_download['status']}`",
            f"- local_source_dir: `{source_data_dir}`",
            "- official Kaggle submission remains disabled until credentials and human gate are configured.",
        ]),
        encoding="utf-8",
    )

    config = build_config(task_id, args.competition_slug, train, test, sample, target, task_type, metric, copied, task_dir)
    config_path = ROOT / "configs" / "generated" / f"{task_id}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "passed",
        "task_id": task_id,
        "competition_slug": args.competition_slug,
        "config_path": rel(config_path),
        "task_dir": rel(task_dir),
        "target": target,
        "task_type": task_type,
        "metric": metric,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "sample_submission_rows": int(len(sample)),
        "official_download": official_download,
        "kaggle_cli_available": kaggle_cli_available(),
        "kaggle_env_configured": kaggle_env_configured(),
        "kaggle_auth_mode": kaggle_auth_mode(),
        "data_files": copied_manifest,
        "local_baseline_ready": True,
        "next_commands": [
            f"python scripts/run_workstation_orchestrator.py --config {rel(config_path)} --output-base experiments --random-state 42",
            f"python scripts/validate_tabular_experiment.py --experiment-dir experiments/{task_id}/<latest> --config {rel(config_path)}",
        ],
        "conclusion": "系统已能把新 Kaggle 风格表格比赛转成可运行科研任务；本地 baseline 可立即训练，官方下载/提交等待 Kaggle 凭证和人工 Gate。",
    }
    write_readiness_report(report, task_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise
