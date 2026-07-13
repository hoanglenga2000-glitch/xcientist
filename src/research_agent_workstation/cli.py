from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research Agent Workstation v1")
    parser.add_argument("--data", required=True, help="Path to a CSV dataset.")
    parser.add_argument("--test-data", default=None, help="Optional Kaggle-style test.csv path.")
    parser.add_argument("--sample-submission", default=None, help="Optional sample_submission.csv path.")
    parser.add_argument("--target", default=None, help="Optional target column for baseline training.")
    parser.add_argument("--output-dir", default="experiments", help="Directory for experiment outputs.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def profile_dataframe(df: pd.DataFrame, target: str | None) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    missing = df.isna().sum().sort_values(ascending=False).head(20).to_dict()

    numeric_summary: dict[str, Any] = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().round(4)
        numeric_summary = {
            col: {idx: to_jsonable(desc.loc[idx, col]) for idx in desc.index}
            for col in desc.columns
        }

    task_type = "eda_only"
    if target:
        if target not in df.columns:
            raise ValueError(f"Target column not found: {target}")
        target_series = df[target].dropna()
        unique_count = int(target_series.nunique())
        if pd.api.types.is_numeric_dtype(target_series) and unique_count > 20:
            task_type = "regression"
        else:
            task_type = "classification"

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target": target,
        "task_type": task_type,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_top20": {k: int(v) for k, v in missing.items()},
        "numeric_summary": numeric_summary,
    }


def build_preprocessor(df: pd.DataFrame, target: str) -> ColumnTransformer:
    feature_df = df.drop(columns=[target])
    numeric_cols = feature_df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in feature_df.columns if col not in numeric_cols]

    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def train_baseline(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    clean_df = df.dropna(subset=[target]).copy()
    if clean_df.empty:
        raise ValueError("No rows left after dropping missing target values.")

    x = clean_df.drop(columns=[target])
    y = clean_df[target]
    stratify = y if task_type == "classification" and y.nunique() > 1 else None

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=160, random_state=random_state, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=160, random_state=random_state, n_jobs=-1)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(clean_df, target)),
            ("model", model),
        ]
    )
    start = time.time()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_valid)
    elapsed = round(time.time() - start, 4)

    if task_type == "classification":
        metrics = {
            "accuracy": round(float(accuracy_score(y_valid, predictions)), 6),
            "macro_f1": round(float(f1_score(y_valid, predictions, average="macro")), 6),
        }
    else:
        metrics = {
            "r2": round(float(r2_score(y_valid, predictions)), 6),
            "mae": round(float(mean_absolute_error(y_valid, predictions)), 6),
        }

    return {
        "pipeline": pipeline,
        "model": model.__class__.__name__,
        "task_type": task_type,
        "train_rows": int(len(x_train)),
        "valid_rows": int(len(x_valid)),
        "metrics": metrics,
        "training_seconds": elapsed,
    }


def make_submission(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    if sample_df.empty:
        raise ValueError("sample_submission.csv is empty.")
    if len(test_df) != len(sample_df):
        raise ValueError(
            f"Row mismatch: test.csv has {len(test_df)} rows, sample_submission has {len(sample_df)} rows."
        )
    if sample_df.shape[1] < 2:
        raise ValueError("sample_submission.csv must contain at least an id column and one prediction column.")

    submission = sample_df.copy()
    prediction_columns = sample_df.columns[1:].tolist()
    predictions = pipeline.predict(test_df)

    if len(prediction_columns) == 1:
        submission[prediction_columns[0]] = predictions
    else:
        if getattr(predictions, "ndim", 1) != 2 or predictions.shape[1] != len(prediction_columns):
            raise ValueError("Model output does not match multi-column sample_submission format.")
        for index, column in enumerate(prediction_columns):
            submission[column] = predictions[:, index]

    output_path = output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)

    checks = {
        "path": str(output_path),
        "rows_match": len(submission) == len(sample_df),
        "columns_match": submission.columns.tolist() == sample_df.columns.tolist(),
        "has_missing_predictions": bool(submission[prediction_columns].isna().any().any()),
        "prediction_columns": prediction_columns,
    }
    checks["valid"] = checks["rows_match"] and checks["columns_match"] and not checks["has_missing_predictions"]
    return checks


def write_report(
    output_dir: Path,
    data_path: Path,
    profile: dict[str, Any],
    baseline: dict[str, Any] | None,
    submission_check: dict[str, Any] | None,
    failure_reason: str | None,
) -> None:
    lines = [
        "# 科研数据任务实验报告",
        "",
        "## 任务背景",
        "",
        f"- 数据文件：`{data_path}`",
        f"- 目标列：`{profile.get('target') or '未指定'}`",
        f"- 任务类型：`{profile['task_type']}`",
        "",
        "## 数据分析",
        "",
        f"- 行数：{profile['rows']}",
        f"- 列数：{profile['columns']}",
        f"- 数值字段数：{len(profile['numeric_columns'])}",
        f"- 非数值字段数：{len(profile['categorical_columns'])}",
        "",
        "## 建模结果",
        "",
    ]

    if baseline:
        display_baseline = {key: value for key, value in baseline.items() if key != "pipeline"}
        lines.extend(
            [
                f"- baseline 模型：`{baseline['model']}`",
                f"- 训练行数：{baseline['train_rows']}",
                f"- 验证行数：{baseline['valid_rows']}",
                f"- 训练耗时：{baseline['training_seconds']} 秒",
                f"- 指标：`{json.dumps(display_baseline['metrics'], ensure_ascii=False)}`",
            ]
        )
    else:
        lines.append("- 未训练模型：未指定目标列或训练失败。")

    lines.extend(["", "## Submission 检查", ""])
    if submission_check:
        lines.extend(
            [
                f"- submission 文件：`{submission_check['path']}`",
                f"- 行数匹配：{submission_check['rows_match']}",
                f"- 列名匹配：{submission_check['columns_match']}",
                f"- 预测缺失：{submission_check['has_missing_predictions']}",
                f"- 是否通过：{submission_check['valid']}",
            ]
        )
    else:
        lines.append("- 未生成 submission：未提供 test.csv/sample_submission.csv，或模型训练未完成。")

    lines.extend(
        [
            "",
            "## 问题复盘",
            "",
            f"- 失败原因：{failure_reason or '本轮未记录失败。'}",
            "",
            "## 下一步计划",
            "",
            "- 检查目标列和评价指标是否与真实任务一致。",
            "- 增加特征工程和模型对比。",
            "- 接入 Kaggle API 后补充真实 leaderboard 或提交记录。",
        ]
    )
    (output_dir / "research_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = data_path.stem.replace(" ", "_")
    output_dir = Path(args.output_dir) / f"{timestamp}_{dataset_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    failure_reason = None
    baseline = None
    submission_check = None

    df = pd.read_csv(data_path)
    profile = profile_dataframe(df, args.target)

    try:
        if args.target:
            baseline = train_baseline(
                df=df,
                target=args.target,
                task_type=profile["task_type"],
                test_size=args.test_size,
                random_state=args.random_state,
            )
    except Exception as exc:
        failure_reason = str(exc)

    try:
        if baseline and args.test_data and args.sample_submission:
            test_df = pd.read_csv(Path(args.test_data).resolve())
            sample_df = pd.read_csv(Path(args.sample_submission).resolve())
            submission_check = make_submission(
                pipeline=baseline["pipeline"],
                test_df=test_df,
                sample_df=sample_df,
                output_dir=output_dir,
            )
    except Exception as exc:
        failure_reason = f"{failure_reason}; submission failed: {exc}" if failure_reason else f"submission failed: {exc}"

    baseline_for_log = None
    if baseline:
        baseline_for_log = {key: value for key, value in baseline.items() if key != "pipeline"}

    profile_path = output_dir / "data_profile.json"
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    experiment_log = {
        "data_path": str(data_path),
        "target": args.target,
        "task_type": profile["task_type"],
        "started_at": timestamp,
        "total_seconds": round(time.time() - started, 4),
        "baseline": baseline_for_log,
        "submission_check": submission_check,
        "failure_reason": failure_reason,
        "outputs": {
            "data_profile": str(profile_path),
            "research_report": str(output_dir / "research_report.md"),
        },
    }
    (output_dir / "experiment_log.json").write_text(
        json.dumps(experiment_log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir, data_path, profile, baseline, submission_check, failure_reason)

    print(f"Experiment written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

