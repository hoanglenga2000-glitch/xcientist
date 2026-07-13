"""Local sklearn ensemble runner for workstation-managed training.

Implements the sklearn_rf_hgb_et_ensemble template:
  - RandomForest + HistGradientBoosting + ExtraTrees
  - 5-fold StratifiedKFold x multi-seed
  - OOF blend grid search
  - Logistic regression stacking
  - Produces workstation-compatible artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, mean_squared_log_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]

BATCH_SIZE = 50_000  # Max rows to transform at once for test predictions


def chunked_predict_proba(model, test_x_df, feature_cols, preprocessor):
    """predict_proba in batches to avoid large dense array allocation."""
    n = len(test_x_df)
    if n <= BATCH_SIZE:
        x_chunk = preprocessor.transform(test_x_df[feature_cols]).astype(np.float32)
        return model.predict_proba(x_chunk)
    preds = []
    for i in range(0, n, BATCH_SIZE):
        chunk = test_x_df.iloc[i:i + BATCH_SIZE]
        x_chunk = preprocessor.transform(chunk[feature_cols]).astype(np.float32)
        preds.append(model.predict_proba(x_chunk))
    return np.concatenate(preds)


def chunked_predict(model, test_x_df, feature_cols, preprocessor):
    """predict in batches to avoid large dense array allocation."""
    n = len(test_x_df)
    if n <= BATCH_SIZE:
        x_chunk = preprocessor.transform(test_x_df[feature_cols]).astype(np.float32)
        return model.predict(x_chunk)
    preds = []
    for i in range(0, n, BATCH_SIZE):
        chunk = test_x_df.iloc[i:i + BATCH_SIZE]
        x_chunk = preprocessor.transform(chunk[feature_cols]).astype(np.float32)
        preds.append(model.predict(x_chunk))
    return np.concatenate(preds)


def elapsed_seconds(started: float) -> float:
    return time.monotonic() - started


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def apply_task_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame, task_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply task-specific feature engineering if a pipeline module exists."""
    pipeline_path = ROOT / "src" / "research_agent_workstation" / f"{task_id}_pipeline.py"
    if not pipeline_path.exists():
        return train_df, test_df
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(f"{task_id}_pipeline", str(pipeline_path))
    if spec is None or spec.loader is None:
        return train_df, test_df
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{task_id}_pipeline"] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "engineer_features", None)
    if fn is None:
        return train_df, test_df
    return fn(train_df), fn(test_df)


def add_features(frame: pd.DataFrame, branch_type: str = "") -> pd.DataFrame:
    df = frame.copy()
    for col in list(df.columns):
        lower = str(col).lower()
        if lower in {"datetime", "date"} or lower.endswith("_date"):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                df[f"{col}_year"] = parsed.dt.year.fillna(-1).astype("int16")
                df[f"{col}_month"] = parsed.dt.month.fillna(-1).astype("int8")
                df[f"{col}_day"] = parsed.dt.day.fillna(-1).astype("int8")
                df[f"{col}_dayofweek"] = parsed.dt.dayofweek.fillna(-1).astype("int8")
                df[f"{col}_hour"] = parsed.dt.hour.fillna(-1).astype("int8")
        if lower == "passengerid":
            parts = df[col].astype(str).str.split("_", n=1, expand=True)
            if parts.shape[1] >= 2:
                df[f"{col}_group"] = pd.to_numeric(parts[0], errors="coerce").fillna(-1).astype("int32")
                df[f"{col}_member"] = pd.to_numeric(parts[1], errors="coerce").fillna(-1).astype("int16")
        if lower == "cabin":
            parts = df[col].astype(str).str.split("/", n=2, expand=True)
            if parts.shape[1] >= 3:
                df[f"{col}_deck"] = parts[0].replace("nan", "missing")
                df[f"{col}_num"] = pd.to_numeric(parts[1], errors="coerce").fillna(-1).astype("int32")
                df[f"{col}_side"] = parts[2].replace("nan", "missing")
    spend_cols = [c for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"] if c in df.columns]
    if spend_cols:
        spend = df[spend_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df["total_spend"] = spend.sum(axis=1).astype("float32")
        df["any_spend"] = (df["total_spend"] > 0).astype("int8")
        df["spend_mean"] = spend.mean(axis=1).astype("float32")
        df["spend_std"] = spend.std(axis=1).fillna(0).astype("float32")
        for col in spend_cols:
            df[f"{col}_log1p"] = np.log1p(np.clip(spend[col].astype(float), 0, None)).astype("float32")
    if branch_type in {"feature_engineering", "ensemble_blend"}:
        if {"HomePlanet", "Destination"}.issubset(df.columns):
            df["homeplanet_destination"] = df["HomePlanet"].astype(str) + "__" + df["Destination"].astype(str)
        if {"CryoSleep", "total_spend"}.issubset(df.columns):
            df["cryo_total_spend"] = df["CryoSleep"].astype(str) + "__" + pd.cut(
                df["total_spend"], bins=[-1, 0, 500, 2000, 1000000], labels=["zero", "low", "mid", "high"]
            ).astype(str)
        if {"VIP", "total_spend"}.issubset(df.columns):
            df["vip_total_spend"] = df["VIP"].astype(str) + "__" + pd.cut(
                df["total_spend"], bins=[-1, 0, 1000, 5000, 1000000], labels=["zero", "low", "mid", "high"]
            ).astype(str)
    pairs = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("u", "r"), ("g", "i"), ("r", "z")]
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            df[f"{a}_minus_{b}"] = df[a] - df[b]
    if "redshift" in df.columns:
        df["redshift_log1p"] = np.log1p(np.clip(df["redshift"].astype(float), 0, None))
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns.tolist()
        if str(c).lower() not in {"id", "imageid", "passengerid"}
    ]
    if numeric_cols:
        numeric_frame = df[numeric_cols]
        for col in numeric_cols:
            series = df[col]
            if series.isna().any():
                df[f"{col}_isna"] = series.isna().astype("int8")
            try:
                if (series == -1).any():
                    df[f"{col}_is_neg1"] = (series == -1).astype("int8")
            except Exception:
                pass
        df["row_missing_count"] = numeric_frame.isna().sum(axis=1).astype("int16")
        df["row_neg1_count"] = (numeric_frame == -1).sum(axis=1).astype("int16")
        df["row_zero_count"] = (numeric_frame == 0).sum(axis=1).astype("int16")
        df["row_numeric_mean"] = numeric_frame.mean(axis=1).astype("float32")
        df["row_numeric_std"] = numeric_frame.std(axis=1).fillna(0).astype("float32")
    return df


def feature_columns_from_config(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    id_col: str,
    config: dict[str, Any],
) -> list[str]:
    """Return leakage-safe feature columns available in both train and test.

    Kaggle tables often contain helper columns in train only, e.g. Bike Sharing
    has casual/registered while test only expects count. The workstation config
    can also declare drop_columns. This helper keeps the generic runner usable
    across benchmark tasks without task-specific training code.
    """

    drop_columns = set(config.get("feature_engineering", {}).get("drop_columns", []) or [])
    drop_columns.add(id_col)
    train_cols = [c for c in train_x.columns if c not in drop_columns]
    test_cols = set(test_x.columns)
    return [c for c in train_cols if c in test_cols]


def rmsle_from_log_target(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))


def binary_positive_index(label_encoder: LabelEncoder, class_names: list[Any]) -> int:
    positive_label = class_names[-1]
    try:
        return int(label_encoder.transform([str(positive_label)])[0])
    except Exception:
        return len(class_names) - 1


def classification_metric_score(metric_name: str, y_true: np.ndarray, proba: np.ndarray, positive_index: int) -> float:
    pred = proba.argmax(axis=1)
    if metric_name == "accuracy":
        return float(accuracy_score(y_true, pred))
    if metric_name in {"auc", "roc_auc", "gini", "normalized_gini"} and proba.shape[1] == 2:
        auc = float(roc_auc_score(y_true, proba[:, positive_index]))
        return float(2 * auc - 1) if metric_name in {"gini", "normalized_gini"} else auc
    return float(balanced_accuracy_score(y_true, pred))


def run_regression_ensemble(args: argparse.Namespace, config: dict[str, Any], output_dir: Path, run_id: str, started: float) -> None:
    task_id = args.task_id
    data_dir = ROOT / "tasks" / task_id / "data"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    target = config["task"].get("target", "SalePrice")
    configured_id_col = config["task"].get("id_column") or config.get("data", {}).get("id_column")
    id_col = configured_id_col if configured_id_col in sample.columns else sample.columns[0]
    prediction_col = config["task"].get("prediction_column")
    if prediction_col not in sample.columns:
        prediction_col = sample.columns[1] if len(sample.columns) > 1 else target
    target_transform = str(config.get("feature_engineering", {}).get("target_transform", "")).lower()

    y_raw = train[target].astype(float).to_numpy()
    y = np.log1p(y_raw) if target_transform == "log1p" else y_raw
    train_sampled = train
    if args.fast and args.sample_rows > 0 and args.sample_rows < len(y):
        sample_idx = np.random.RandomState(42).choice(len(y), args.sample_rows, replace=False)
        train = train.iloc[sample_idx].reset_index(drop=True)
        y_raw = y_raw[sample_idx]
        y = y[sample_idx]
        train_sampled = train
        print(f"[fast mode] Sampled {args.sample_rows} rows before preprocessing", flush=True)

    branch_type = str(getattr(args, "branch_type", "") or "")
    # Apply task-specific feature engineering pipeline if available (fix: was missing from regression path)
    train, test = apply_task_pipeline(train, test, args.task_id)
    train_x = add_features(train.drop(columns=[target], errors="ignore"), branch_type=branch_type)
    test_x = add_features(test, branch_type=branch_type)
    feature_cols = feature_columns_from_config(train_x, test_x, id_col, config)
    if not feature_cols:
        raise ValueError(f"No shared feature columns remain after applying drop_columns for task {task_id}.")
    categorical = [c for c in feature_cols if train_x[c].dtype == "object"]
    numeric = [c for c in feature_cols if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric), ("cat", make_encoder(), categorical)],
        remainder="drop",
    )
    x_all = preprocessor.fit_transform(train_x[feature_cols]).astype(np.float32)
    # x_test is NOT pre-transformed; chunked_predict handles batching to save memory

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    if args.fast:
        seeds = seeds[:1]
        n_folds = min(3, max(2, len(y) // 10))
        rf_est, hgb_iter, et_est = 120, 180, 120
    else:
        n_folds = args.n_folds
        rf_est, hgb_iter, et_est = 500, 700, 500
    if branch_type == "model_family":
        rf_est, hgb_iter, et_est = int(rf_est * 0.8), int(hgb_iter * 1.25), int(et_est * 0.8)
    elif branch_type == "ensemble_blend":
        rf_est, hgb_iter, et_est = int(rf_est * 1.1), int(hgb_iter * 1.1), int(et_est * 1.1)

    model_builders = {
        "rf": lambda rseed: RandomForestRegressor(
            n_estimators=rf_est, max_depth=18, min_samples_leaf=2,
            max_features=0.55, n_jobs=4, random_state=rseed,
        ),
        "hgb": lambda rseed: HistGradientBoostingRegressor(
            max_iter=hgb_iter, learning_rate=0.035, max_leaf_nodes=31,
            l2_regularization=0.05, early_stopping=True,
            validation_fraction=0.12, n_iter_no_change=40, random_state=rseed,
        ),
        "et": lambda rseed: ExtraTreesRegressor(
            n_estimators=et_est, max_depth=None, min_samples_leaf=2,
            max_features=0.75, n_jobs=4, random_state=rseed,
        ),
    }
    model_names = list(model_builders.keys())
    oof = {name: np.zeros(len(y), dtype=np.float64) for name in model_names}
    test_preds = {name: np.zeros(len(test_x), dtype=np.float64) for name in model_names}
    cv_scores = {name: [] for name in model_names}
    event_log = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for seed in seeds:
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_all)):
            X_tr, X_va = x_all[train_idx], x_all[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            for name, build_fn in model_builders.items():
                model = build_fn(seed)
                model.fit(X_tr, y_tr)
                p_val = np.asarray(model.predict(X_va), dtype=np.float64)
                p_test = np.asarray(chunked_predict(model, test_x, feature_cols, preprocessor), dtype=np.float64)
                oof[name][val_idx] += p_val / len(seeds)
                test_preds[name] += p_test / (n_folds * len(seeds))
                fold_rmsle = rmsle_from_log_target(y_va, p_val) if target_transform == "log1p" else float(np.sqrt(mean_squared_log_error(np.clip(y_va,0,None), np.clip(p_val,0,None))))
                cv_scores[name].append(fold_rmsle)
                event_log.append({"model": name, "seed": seed, "fold": fold + 1, "rmsle": fold_rmsle})
                print(f"  [{elapsed_seconds(started):.0f}s] {name} seed={seed} fold={fold+1}: rmsle={fold_rmsle:.5f}", flush=True)

    oof_scores = {name: rmsle_from_log_target(y, oof[name]) if target_transform == "log1p" else float(np.sqrt(mean_squared_log_error(np.clip(y,0,None), np.clip(oof[name],0,None)))) for name in model_names}

    best_blend_score = float("inf")
    best_weights = tuple([1.0 / len(model_names)] * len(model_names))
    for w1 in range(5, 91, 5):
        for w2 in range(5, 91, 5):
            w3 = 100 - w1 - w2
            if w3 < 5:
                continue
            w = (w1 / 100.0, w2 / 100.0, w3 / 100.0)
            blend = w[0] * oof[model_names[0]] + w[1] * oof[model_names[1]] + w[2] * oof[model_names[2]]
            score = rmsle_from_log_target(y, blend) if target_transform == "log1p" else float(np.sqrt(mean_squared_log_error(np.clip(y,0,None), np.clip(blend,0,None))))
            if score < best_blend_score:
                best_blend_score = score
                best_weights = w

    blend_oof = best_weights[0] * oof[model_names[0]] + best_weights[1] * oof[model_names[1]] + best_weights[2] * oof[model_names[2]]
    blend_test = best_weights[0] * test_preds[model_names[0]] + best_weights[1] * test_preds[model_names[1]] + best_weights[2] * test_preds[model_names[2]]

    stack_features = np.column_stack([oof[name] for name in model_names])
    stack_test = np.column_stack([test_preds[name] for name in model_names])
    stacker = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 50.0])
    stacker.fit(stack_features, y)
    stack_oof = np.asarray(stacker.predict(stack_features), dtype=np.float64)
    stack_pred_test = np.asarray(stacker.predict(stack_test), dtype=np.float64)
    stack_score = rmsle_from_log_target(y, stack_oof) if target_transform == "log1p" else float(np.sqrt(mean_squared_log_error(np.clip(y,0,None), np.clip(stack_oof,0,None))))

    if stack_score < best_blend_score:
        final_oof = stack_oof
        final_pred_log = stack_pred_test
        best_method = "ridge_stack"
        best_oof_score = stack_score
    else:
        final_oof = blend_oof
        final_pred_log = blend_test
        best_method = "blend"
        best_oof_score = best_blend_score

    final_pred = np.expm1(final_pred_log) if target_transform == "log1p" else final_pred_log
    final_pred = np.clip(final_pred, 1e-6, None)
    submission = pd.DataFrame({id_col: sample[id_col].to_numpy(), prediction_col: final_pred})
    submission.to_csv(output_dir / "submission.csv", index=False)

    oof_frame = pd.DataFrame({"id": train_sampled[id_col].to_numpy(), "prediction": np.expm1(final_oof) if target_transform == "log1p" else final_oof, "true": y_raw})
    oof_frame.to_csv(output_dir / "oof_predictions.csv", index=False)

    cv_summary = {name: {"mean": float(np.mean(cv_scores[name])), "std": float(np.std(cv_scores[name]))} for name in model_names}
    metrics = {
        "schema": "academic_research_os.local_ensemble_metrics.v1",
        "status": "passed",
        "task_id": args.task_id,
        "run_id": run_id,
        "runner": "local_sklearn_regression_ensemble_rf_hgb_et",
        "search_branch": {
            "branch_id": getattr(args, "branch_id", ""),
            "branch_type": branch_type,
            "code_generation_mode": getattr(args, "code_generation_mode", ""),
            "hypothesis": getattr(args, "branch_hypothesis", ""),
        },
        "task_type": "regression",
        "metric": "rmsle",
        "metric_direction": "minimize",
        "n_folds": n_folds,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features_after_encoding": int(x_all.shape[1]),
        "cv_fold_rmsle": cv_summary,
        "oof_rmsle": oof_scores,
        "ensemble": {
            "blend": {"weights": {model_names[i]: round(best_weights[i], 4) for i in range(len(model_names))}, "rmsle": best_blend_score},
            "stack": {"method": "RidgeCV", "rmsle": stack_score},
            "best_method": best_method,
            "selection_metric": "rmsle",
            "best_validation_score": float(best_oof_score),
            "metric_direction": "minimize",
        },
        "seconds": round(elapsed_seconds(started), 3),
        "submission_rows": int(len(submission)),
        "prediction_summary": {"min": float(np.min(final_pred)), "mean": float(np.mean(final_pred)), "max": float(np.max(final_pred))},
        "human_gate_required_for_official_submission": True,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    artifacts = {}
    for fpath in sorted(output_dir.rglob("*")):
        if fpath.is_file():
            rel = str(fpath.relative_to(output_dir))
            artifacts[rel] = {"path": rel, "sha256": sha256_file(fpath), "size": fpath.stat().st_size}
    manifest = {
        "schema": "academic_research_os.artifact_manifest.v1",
        "task_id": args.task_id,
        "run_id": run_id,
        "template_id": "sklearn_rf_hgb_et_ensemble",
        "created_by_agent": "workstation_orchestrator",
        "stage": "training",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": artifacts,
        "gate_dependency": "SUBMISSION_APPROVAL",
        "claim_binding": f"Local sklearn regression ensemble produced OOF rmsle={best_oof_score:.6f}",
        "metrics": {"best_method": best_method, "best_validation_score": best_oof_score, "rmsle": best_oof_score, "metric_direction": "minimize"},
    }
    (output_dir / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# Local Sklearn Regression Ensemble Run", "",
        f"- task: `{args.task_id}`",
        "- runner: `local_sklearn_regression_ensemble_rf_hgb_et`",
        f"- folds: `{n_folds}` x seeds: `{len(seeds)}`",
        f"- best method: `{best_method}`",
        f"- best OOF RMSLE: `{best_oof_score:.6f}`", "",
        "## CV RMSLE (per-fold, mean +/- std)",
    ]
    for name in model_names:
        report.append(f"- {name.upper()}: {cv_summary[name]['mean']:.6f} +/- {cv_summary[name]['std']:.6f}")
    report += ["", "## OOF Scores", *[f"- {n.upper()}: rmsle={oof_scores[n]:.6f}" for n in model_names], f"- Blend: rmsle={best_blend_score:.6f} (weights: {best_weights[0]:.2f}/{best_weights[1]:.2f}/{best_weights[2]:.2f})", f"- Ridge stack: rmsle={stack_score:.6f}", "", f"- submission rows: `{metrics['submission_rows']}`", "- official Kaggle submission remains behind Human Gate."]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps({"status": "passed", "task_id": args.task_id, "run_id": run_id, "output_dir": str(output_dir), "best_method": best_method, "best_validation_score": best_oof_score, "seconds": metrics["seconds"]}, ensure_ascii=False))


def _run_local_ensemble_for_tests() -> None:
    parser = argparse.ArgumentParser(description="Local sklearn ensemble runner.")
    parser.add_argument("--config", required=True, help="Task config YAML")
    parser.add_argument("--output-base", required=True, help="Output base directory")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seeds", default="42,3407,12345")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--task-id", default="playground_series_s6e6")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--fast", action="store_true", help="Use reduced estimators and sample for quick verification.")
    parser.add_argument("--sample-rows", type=int, default=0, help="Sample N rows for fast mode.")
    parser.add_argument("--branch-id", default="", help="Search-controller branch id.")
    parser.add_argument("--branch-type", default="", help="Search-controller branch type.")
    parser.add_argument("--code-generation-mode", default="", help="Base / Stepwise / Diff mode selected by the Search Controller.")
    parser.add_argument("--branch-hypothesis", default="", help="Branch hypothesis for governance artifacts.")
    parser.add_argument("--cross-branch-references", default="", help="JSON encoded cross-branch reference list.")
    args = parser.parse_args()

    started = time.monotonic()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    if args.fast:
        seeds = seeds[:1]  # Single seed for fast mode
        args.n_folds = 3  # Reduced folds for fast mode
        n_folds = 3
    else:
        n_folds = args.n_folds
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_base) / args.task_id / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import yaml
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    except Exception:
        config = {"task": {"name": args.task_id, "target": "class", "metric": "balanced_accuracy", "type": "classification"}}

    task_type = str(config.get("task", {}).get("type", "classification")).lower()
    metric_name = str(config.get("task", {}).get("metric", "")).lower()
    if task_type == "regression" or metric_name in {"rmsle", "rmse", "mae", "mse"}:
        return run_regression_ensemble(args, config, output_dir, run_id, started)

    data_dir = ROOT / "tasks" / args.task_id / "data"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    target = config["task"].get("target", "class")
    configured_id_col = config["task"].get("id_column") or config.get("data", {}).get("id_column")
    id_col = configured_id_col if configured_id_col in sample.columns else sample.columns[0]
    prediction_col = config["task"].get("prediction_column")
    if prediction_col not in sample.columns:
        prediction_col = sample.columns[1] if len(sample.columns) > 1 else target
    class_names = sorted(train[target].dropna().unique().tolist())
    le = LabelEncoder()
    y = le.fit_transform(train[target].astype(str)).astype("int64")
    train_sampled = train
    if args.fast and args.sample_rows > 0 and args.sample_rows < len(y):
        if task_type == "classification":
            # Stratified sampling: ensure every viable class (>=2 members) gets at least 2 samples
            viable_classes = [c for c in class_names if (train[target].astype(str) == str(c)).sum() >= 2]
            per_class = max(2, args.sample_rows // max(1, len(viable_classes)))
            dfs = []
            for cls_val in viable_classes:
                cls_mask = train[target].astype(str) == str(cls_val)
                cls_df = train[cls_mask]
                n_sample = min(per_class, len(cls_df))
                if n_sample > 0:
                    dfs.append(cls_df.sample(n=n_sample, random_state=42))
            # Also include singleton classes (only 1 total in dataset) to keep n_classes correct
            for cls_val in class_names:
                if cls_val not in viable_classes:
                    cls_mask = train[target].astype(str) == str(cls_val)
                    cls_df = train[cls_mask]
                    if len(cls_df) > 0:
                        dfs.append(cls_df)
            if dfs:
                train = pd.concat(dfs, ignore_index=True)
                if len(train) > args.sample_rows:
                    train = train.sample(n=args.sample_rows, random_state=42)
            else:
                sample_idx = np.random.RandomState(42).choice(len(y), args.sample_rows, replace=False)
                train = train.iloc[sample_idx].reset_index(drop=True)
                y = y[sample_idx]
        else:
            sample_idx = np.random.RandomState(42).choice(len(y), args.sample_rows, replace=False)
            train = train.iloc[sample_idx].reset_index(drop=True)
            y = y[sample_idx]
        train_sampled = train
        y = le.fit_transform(train[target].astype(str)).astype("int64")
        print(f"[fast mode] Sampled {len(train)} rows before preprocessing", flush=True)

    # Apply task-specific feature engineering pipeline if available
    train, test = apply_task_pipeline(train, test, args.task_id)

    branch_type = str(args.branch_type or "")
    train_x = add_features(train.drop(columns=[target], errors="ignore"), branch_type=branch_type)
    test_x = add_features(test, branch_type=branch_type)
    feature_cols = feature_columns_from_config(train_x, test_x, id_col, config)
    if not feature_cols:
        raise ValueError(f"No shared feature columns remain after applying drop_columns for task {args.task_id}.")
    categorical = [c for c in feature_cols if train_x[c].dtype == "object"]
    numeric = [c for c in feature_cols if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric), ("cat", make_encoder(), categorical)],
        remainder="drop",
    )
    x_all = preprocessor.fit_transform(train_x[feature_cols]).astype(np.float32)
    # x_test is NOT pre-transformed; chunked_predict_proba handles batching to save memory

    # Fast-mode sampling is applied before preprocessing to avoid dense encoding full large datasets.
    n_classes = len(class_names)

    if args.fast:
        rf_est, rf_depth = 80, 12
        hgb_iter, hgb_depth = 100, 6
        et_est, et_depth = 80, 12
    else:
        rf_est, rf_depth = 400, 18
        hgb_iter, hgb_depth = 300, 8
        et_est, et_depth = 400, 18
    if branch_type == "model_family":
        rf_est, rf_depth = int(rf_est * 0.75), max(6, rf_depth - 2)
        hgb_iter, hgb_depth = int(hgb_iter * 1.35), hgb_depth + 1
        et_est, et_depth = int(et_est * 0.75), max(8, et_depth - 1)
    elif branch_type == "feature_engineering":
        rf_est, hgb_iter, et_est = int(rf_est * 1.05), int(hgb_iter * 1.05), int(et_est * 1.05)
    elif branch_type == "ensemble_blend":
        rf_est, hgb_iter, et_est = int(rf_est * 1.15), int(hgb_iter * 1.15), int(et_est * 1.15)

    n_jobs = 4
    model_builders = {
        "rf": lambda rseed: RandomForestClassifier(
            n_estimators=rf_est, max_depth=rf_depth, min_samples_leaf=32,
            max_features="sqrt", n_jobs=n_jobs, random_state=rseed,
        ),
        "hgb": lambda rseed: HistGradientBoostingClassifier(
            max_iter=hgb_iter, learning_rate=0.05, max_depth=hgb_depth, max_leaf_nodes=63,
            min_samples_leaf=32, l2_regularization=0.5, early_stopping=not args.fast,
            validation_fraction=0.12, n_iter_no_change=30, random_state=rseed,
        ),
        "et": lambda rseed: ExtraTreesClassifier(
            n_estimators=et_est, max_depth=et_depth, min_samples_leaf=32,
            max_features="sqrt", n_jobs=n_jobs, random_state=rseed,
        ),
    }
    model_names = list(model_builders.keys())

    oof = {name: np.zeros((len(y), n_classes), dtype=np.float64) for name in model_names}
    test_preds = {name: np.zeros((len(test_x), n_classes), dtype=np.float64) for name in model_names}
    cv_scores = {name: [] for name in model_names}
    event_log = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for seed in seeds:
        for fold, (train_idx, val_idx) in enumerate(skf.split(x_all, y)):
            X_tr, X_va = x_all[train_idx], x_all[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            for name, build_fn in model_builders.items():
                model = build_fn(seed)
                model.fit(X_tr, y_tr)
                p_val = model.predict_proba(X_va)
                if p_val.ndim == 1:
                    p_val = np.column_stack([1 - p_val, p_val])
                if p_val.shape[1] != n_classes:
                    full = np.zeros((len(p_val), n_classes), dtype=np.float64)
                    full[:, :p_val.shape[1]] = p_val
                    p_val = full
                p_test = chunked_predict_proba(model, test_x, feature_cols, preprocessor)
                if p_test.ndim == 1:
                    p_test = np.column_stack([1 - p_test, p_test])
                if p_test.shape[1] != n_classes:
                    full = np.zeros((len(p_test), n_classes), dtype=np.float64)
                    full[:, :p_test.shape[1]] = p_test
                    p_test = full

                oof[name][val_idx] += p_val / len(seeds)
                test_preds[name] += p_test / (n_folds * len(seeds))
                acc = float(accuracy_score(y_va, p_val.argmax(axis=1)))
                cv_scores[name].append(acc)
                event_log.append({"model": name, "seed": seed, "fold": fold + 1, "accuracy": acc})
                print(f"  [{elapsed_seconds(started):.0f}s] {name} seed={seed} fold={fold+1}: acc={acc:.4f}", flush=True)

    # OOF scores
    oof_scores = {}
    for name in model_names:
        oof_scores[name] = {
            "accuracy": float(accuracy_score(y, oof[name].argmax(axis=1))),
            "balanced_accuracy": float(balanced_accuracy_score(y, oof[name].argmax(axis=1))),
            "log_loss": float(log_loss(y, oof[name], labels=list(range(n_classes)))),
        }

    task_metric = str(config["task"].get("metric", "balanced_accuracy")).lower()
    score_key = task_metric if task_metric in {"accuracy", "auc", "roc_auc", "gini", "normalized_gini"} else "balanced_accuracy"
    positive_index = binary_positive_index(le, class_names)

    # Blend weight grid search
    best_blend_score = 0.0
    best_weights = tuple([1.0 / len(model_names)] * len(model_names))
    for w1 in range(10, 71, 3):
        for w2 in range(10, 71, 3):
            w3 = 100 - w1 - w2
            if w3 < 5:
                continue
            w = (w1 / 100.0, w2 / 100.0, w3 / 100.0)
            blend = w[0] * oof[model_names[0]] + w[1] * oof[model_names[1]] + w[2] * oof[model_names[2]]
            blend_score = classification_metric_score(score_key, y, blend, positive_index)
            if blend_score > best_blend_score:
                best_blend_score = blend_score
                best_weights = w

    blend_oof = best_weights[0] * oof[model_names[0]] + best_weights[1] * oof[model_names[1]] + best_weights[2] * oof[model_names[2]]
    blend_accuracy = float(accuracy_score(y, blend_oof.argmax(axis=1)))
    blend_bal_acc = float(balanced_accuracy_score(y, blend_oof.argmax(axis=1)))

    # Logistic regression stacking
    stack_features = np.hstack([oof[name] for name in model_names])
    stack_test = np.hstack([test_preds[name] for name in model_names])
    stacker = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
    if branch_type == "model_family":
        stacker = LogisticRegression(max_iter=5000, C=0.5, random_state=42)
    elif branch_type == "ensemble_blend":
        stacker = LogisticRegression(max_iter=5000, C=2.0, random_state=42)
    stacker.fit(stack_features, y)
    stack_oof = stacker.predict_proba(stack_features)
    stack_accuracy = float(accuracy_score(y, stack_oof.argmax(axis=1)))
    stack_bal_acc = float(balanced_accuracy_score(y, stack_oof.argmax(axis=1)))
    blend_metric_score = classification_metric_score(score_key, y, blend_oof, positive_index)
    stack_metric_score = classification_metric_score(score_key, y, stack_oof, positive_index)

    # Choose best method
    stack_score = stack_metric_score
    blend_score = blend_metric_score
    if stack_score > blend_score:
        final_proba_test = stacker.predict_proba(stack_test)
        best_method = "stack"
        best_oof_score = stack_score
    else:
        blend_test = best_weights[0] * test_preds[model_names[0]] + best_weights[1] * test_preds[model_names[1]] + best_weights[2] * test_preds[model_names[2]]
        final_proba_test = blend_test
        best_method = "blend"
        best_oof_score = blend_score

    if score_key in {"auc", "roc_auc", "gini", "normalized_gini"} and n_classes == 2:
        submission_values: list[Any] | np.ndarray = np.clip(final_proba_test[:, positive_index], 0.0, 1.0)
    else:
        final_pred_test = final_proba_test.argmax(axis=1)
        submission_values = [class_names[int(i)] for i in final_pred_test]
    submission = pd.DataFrame({id_col: sample[id_col].to_numpy(), prediction_col: submission_values})
    submission.to_csv(output_dir / "submission.csv", index=False)

    # Save OOF predictions
    final_oof = blend_oof if best_method == "blend" else stack_oof
    oof_ids = train_sampled[id_col].to_numpy() if id_col in train_sampled.columns else np.arange(1, len(train_sampled) + 1)
    oof_frame = pd.DataFrame({"id": oof_ids})
    for idx, cls in enumerate(class_names):
        oof_frame[f"proba_{cls}"] = final_oof[:, idx]
    oof_frame["pred_class"] = [class_names[int(i)] for i in final_oof.argmax(axis=1)]
    oof_frame["true_class"] = train_sampled[target].astype(str).to_numpy()
    oof_frame.to_csv(output_dir / "oof_predictions.csv", index=False)

    cv_summary = {}
    for name in model_names:
        scores = cv_scores[name]
        cv_summary[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}

    if score_key in {"auc", "roc_auc", "gini", "normalized_gini"} and n_classes == 2:
        prediction_summary: dict[str, Any] = {
            "min": float(np.min(submission[prediction_col])),
            "mean": float(np.mean(submission[prediction_col])),
            "max": float(np.max(submission[prediction_col])),
            "q01": float(np.quantile(submission[prediction_col], 0.01)),
            "q50": float(np.quantile(submission[prediction_col], 0.50)),
            "q99": float(np.quantile(submission[prediction_col], 0.99)),
        }
    else:
        prediction_summary = submission[prediction_col].value_counts().to_dict()

    metrics = {
        "schema": "academic_research_os.local_ensemble_metrics.v1",
        "status": "passed",
        "task_id": args.task_id,
        "run_id": run_id,
        "runner": "local_sklearn_ensemble_rf_hgb_et",
        "search_branch": {
            "branch_id": args.branch_id,
            "branch_type": branch_type,
            "code_generation_mode": args.code_generation_mode,
            "hypothesis": args.branch_hypothesis,
            "cross_branch_references": json.loads(args.cross_branch_references) if args.cross_branch_references else [],
        },
        "task_type": "classification",
        "metric": score_key,
        "metric_direction": "maximize",
        "n_folds": n_folds,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features_after_encoding": int(x_all.shape[1]),
        "classes": class_names,
        "cv_fold_accuracy": cv_summary,
        "oof_accuracy": {name: oof_scores[name]["accuracy"] for name in model_names},
        "oof_balanced_accuracy": {name: oof_scores[name]["balanced_accuracy"] for name in model_names},
        "oof_log_loss": {name: oof_scores[name]["log_loss"] for name in model_names},
        "ensemble": {
            "blend": {
                "weights": {model_names[i]: round(best_weights[i], 4) for i in range(len(model_names))},
                "accuracy": blend_accuracy,
                "balanced_accuracy": blend_bal_acc,
                "log_loss": float(log_loss(y, blend_oof, labels=list(range(n_classes)))),
            },
            "stack": {
                "accuracy": stack_accuracy,
                "balanced_accuracy": stack_bal_acc,
                "log_loss": float(log_loss(y, stack_oof, labels=list(range(n_classes)))),
            },
            "best_method": best_method,
            "selection_metric": score_key,
            "best_validation_score": float(best_oof_score),
            "metric_direction": "maximize",
        },
        "seconds": round(elapsed_seconds(started), 3),
        "submission_rows": int(len(submission)),
        "prediction_distribution": prediction_summary,
        "positive_class_index": positive_index,
        "human_gate_required_for_official_submission": True,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build artifact manifest
    artifacts = {}
    for fpath in sorted(output_dir.rglob("*")):
        if fpath.is_file():
            rel = str(fpath.relative_to(output_dir))
            artifacts[rel] = {
                "path": rel,
                "sha256": sha256_file(fpath),
                "size": fpath.stat().st_size,
            }

    manifest = {
        "schema": "academic_research_os.artifact_manifest.v1",
        "task_id": args.task_id,
        "run_id": run_id,
        "template_id": "sklearn_rf_hgb_et_ensemble",
        "created_by_agent": "workstation_orchestrator",
        "stage": "training",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": artifacts,
        "gate_dependency": "SUBMISSION_APPROVAL",
        "claim_binding": f"Local sklearn ensemble produced OOF balanced_accuracy={best_oof_score:.6f}",
        "metrics": {
            "best_method": best_method,
            "best_validation_score": best_oof_score,
            "balanced_accuracy": best_oof_score,
        },
    }
    (output_dir / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Report
    report = [
        "# Local Sklearn Ensemble Run",
        "",
        f"- task: `{args.task_id}`",
        "- runner: `local_sklearn_ensemble_rf_hgb_et`",
        f"- folds: `{n_folds}` x seeds: `{len(seeds)}`",
        f"- best method: `{best_method}`",
        f"- selection metric: `{score_key}`",
        f"- best OOF {score_key}: `{best_oof_score:.6f}`",
        "",
        "## CV Accuracy (per-fold, mean +/- std)",
    ]
    for name in model_names:
        report.append(f"- {name.upper()}: {cv_summary[name]['mean']:.6f} +/- {cv_summary[name]['std']:.6f}")
    report += [
        "",
        "## OOF Scores",
        *[f"- {n.upper()}: bal_acc={oof_scores[n]['balanced_accuracy']:.6f} log_loss={oof_scores[n]['log_loss']:.6f}" for n in model_names],
        f"- Blend: bal_acc={blend_bal_acc:.6f} (weights: {best_weights[0]:.2f}/{best_weights[1]:.2f}/{best_weights[2]:.2f})",
        f"- Stack (LogisticRegression): bal_acc={stack_bal_acc:.6f}",
        "",
        f"- submission rows: `{metrics['submission_rows']}`",
        f"- prediction distribution: {json.dumps(metrics['prediction_distribution'])}",
        "- official Kaggle submission remains behind Human Gate.",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps({
        "status": "passed",
        "task_id": args.task_id,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "best_method": best_method,
        "best_validation_score": best_oof_score,
        "seconds": metrics["seconds"],
    }, ensure_ascii=False))


def main() -> int:
    """Retained command shim that cannot start workstation training."""
    print(json.dumps({
        "status": "blocked_local_training_disabled",
        "training_started": False,
        "required_compute": "gpu",
        "message": "Local training is disabled by release policy. Use the gated HPC/GPU workflow.",
    }, ensure_ascii=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
