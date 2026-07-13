"""Enhanced ensemble runner for spaceship_titanic: HGB+GBC+RF+ET with multi-seed.

Key improvements over the default sklearn_rf_hgb_et_ensemble:
- Adds GradientBoostingClassifier (4th model — stronger than ET alone)
- 5-seed diversity (vs 1 seed in prior best)
- Aggressive HGB hyperparameters
- Feature engineering via add_features()
- OOF blend + stacking with LogisticRegression
- Produces all workstation governance artifacts
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
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]


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


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for col in list(df.columns):
        lower = str(col).lower()
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
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seeds", default="42,260612,3407,12345,99999")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    started = time.monotonic()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    n_folds = args.n_folds
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_base) / "spaceship_titanic" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    task_id = "spaceship_titanic"
    data_dir = ROOT / "tasks" / task_id / "data"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    target = "Transported"
    id_col = "PassengerId"
    prediction_col = "Transported"

    class_names = [False, True]
    le = LabelEncoder()
    y = le.fit_transform(train[target].astype(str)).astype("int64")

    train_x = add_features(train.drop(columns=[target], errors="ignore"))
    test_x = add_features(test)

    drop_cols = {id_col}
    train_cols = [c for c in train_x.columns if c not in drop_cols]
    test_cols_set = set(test_x.columns)
    feature_cols = [c for c in train_cols if c in test_cols_set]

    categorical = [c for c in feature_cols if train_x[c].dtype == "object"]
    numeric = [c for c in feature_cols if c not in categorical]

    print(f"[{elapsed_seconds(started):.0f}s] Features: {len(feature_cols)} ({len(numeric)} numeric + {len(categorical)} categorical)", flush=True)

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric), ("cat", make_encoder(), categorical)],
        remainder="drop",
    )
    x_all = preprocessor.fit_transform(train_x[feature_cols]).astype(np.float32)
    x_test = preprocessor.transform(test_x[feature_cols]).astype(np.float32)
    n_classes = len(class_names)
    print(f"[{elapsed_seconds(started):.0f}s] After encoding: {x_all.shape[1]} features, {x_all.shape[0]} train rows", flush=True)

    # Model builders: HGB (strongest) + GBC + RF + ET
    model_builders = {
        "hgb": lambda rseed: HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.04, max_depth=10, max_leaf_nodes=95,
            min_samples_leaf=24, l2_regularization=0.4, early_stopping=True,
            validation_fraction=0.12, n_iter_no_change=35, random_state=rseed,
        ),
        "gbc": lambda rseed: GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6, min_samples_leaf=24,
            subsample=0.8, max_features="sqrt", random_state=rseed,
        ),
        "rf": lambda rseed: RandomForestClassifier(
            n_estimators=350, max_depth=14, min_samples_leaf=32,
            max_features="sqrt", n_jobs=4, random_state=rseed,
        ),
        "et": lambda rseed: ExtraTreesClassifier(
            n_estimators=350, max_depth=14, min_samples_leaf=32,
            max_features="sqrt", n_jobs=4, random_state=rseed,
        ),
    }
    model_names = list(model_builders.keys())

    # Models that can't handle NaN natively: fill with 0
    nan_sensitive = {"gbc", "rf", "et"}
    nan_imputer = SimpleImputer(strategy="constant", fill_value=0.0)

    oof = {name: np.zeros((len(y), n_classes), dtype=np.float64) for name in model_names}
    test_preds = {name: np.zeros((len(x_test), n_classes), dtype=np.float64) for name in model_names}
    cv_scores = {name: [] for name in model_names}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for seed in seeds:
        for fold, (train_idx, val_idx) in enumerate(skf.split(x_all, y)):
            X_tr_raw, X_va_raw = x_all[train_idx], x_all[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            for name, build_fn in model_builders.items():
                model = build_fn(seed)
                if name in nan_sensitive:
                    X_tr = nan_imputer.fit_transform(X_tr_raw)
                    X_va = nan_imputer.transform(X_va_raw)
                    X_te = nan_imputer.transform(x_test)
                else:
                    X_tr, X_va = X_tr_raw, X_va_raw
                    X_te = x_test
                model.fit(X_tr, y_tr)
                p_val = model.predict_proba(X_va)
                if p_val.ndim == 1:
                    p_val = np.column_stack([1 - p_val, p_val])
                if p_val.shape[1] != n_classes:
                    full = np.zeros((len(p_val), n_classes), dtype=np.float64)
                    full[:, :p_val.shape[1]] = p_val
                    p_val = full
                p_test = model.predict_proba(X_te)
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
                print(f"  [{elapsed_seconds(started):.0f}s] {name} seed={seed} fold={fold+1}: acc={acc:.4f}", flush=True)

    # OOF scores
    oof_scores = {}
    for name in model_names:
        oof_scores[name] = {
            "accuracy": float(accuracy_score(y, oof[name].argmax(axis=1))),
            "balanced_accuracy": float(balanced_accuracy_score(y, oof[name].argmax(axis=1))),
            "log_loss": float(log_loss(y, oof[name], labels=list(range(n_classes)))),
        }
        print(f"[{elapsed_seconds(started):.0f}s] {name} OOF: acc={oof_scores[name]['accuracy']:.4f} bal_acc={oof_scores[name]['balanced_accuracy']:.4f} log_loss={oof_scores[name]['log_loss']:.4f}", flush=True)

    # Blend grid search (4-way weighted blend)
    n_models = len(model_names)
    best_blend_acc = 0.0
    best_weights = tuple([1.0 / n_models] * n_models)
    # Coarse-to-fine grid search
    for step in [5, 3, 1]:
        for w1 in range(0, 101, step):
            for w2 in range(0, 101 - w1, step):
                for w3 in range(0, 101 - w1 - w2, step):
                    w4 = 100 - w1 - w2 - w3
                    if w4 < 0:
                        continue
                    w = (w1 / 100.0, w2 / 100.0, w3 / 100.0, w4 / 100.0)
                    blend = w[0] * oof[model_names[0]] + w[1] * oof[model_names[1]] + w[2] * oof[model_names[2]] + w[3] * oof[model_names[3]]
                    acc = float(accuracy_score(y, blend.argmax(axis=1)))
                    if acc > best_blend_acc:
                        best_blend_acc = acc
                        best_weights = w

    blend_oof = sum(best_weights[i] * oof[model_names[i]] for i in range(n_models))
    blend_acc = float(accuracy_score(y, blend_oof.argmax(axis=1)))
    blend_bal_acc = float(balanced_accuracy_score(y, blend_oof.argmax(axis=1)))
    print(f"[{elapsed_seconds(started):.0f}s] Best blend weights: {dict(zip(model_names, [round(w, 3) for w in best_weights]))}, acc={blend_acc:.4f}", flush=True)

    # Stacking
    stack_features = np.hstack([oof[name] for name in model_names])
    stack_test = np.hstack([test_preds[name] for name in model_names])
    stacker = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
    stacker.fit(stack_features, y)
    stack_oof = stacker.predict_proba(stack_features)
    stack_acc = float(accuracy_score(y, stack_oof.argmax(axis=1)))
    stack_bal_acc = float(balanced_accuracy_score(y, stack_oof.argmax(axis=1)))
    print(f"[{elapsed_seconds(started):.0f}s] Stack acc={stack_acc:.4f} bal_acc={stack_bal_acc:.4f}", flush=True)

    # Select best method
    if stack_acc > blend_acc:
        final_proba_test = stacker.predict_proba(stack_test)
        best_method = "stack"
        best_oof_acc = stack_acc
        final_oof = stack_oof
    else:
        final_proba_test = sum(best_weights[i] * test_preds[model_names[i]] for i in range(n_models))
        best_method = "blend"
        best_oof_acc = blend_acc
        final_oof = blend_oof

    # Submission
    pred_class = final_proba_test.argmax(axis=1)
    submission_values = [class_names[int(i)] for i in pred_class]
    submission = pd.DataFrame({id_col: sample[id_col].to_numpy(), prediction_col: submission_values})
    submission.to_csv(output_dir / "submission.csv", index=False)

    # OOF predictions
    oof_ids = train[id_col].to_numpy() if id_col in train.columns else np.arange(1, len(train) + 1)
    oof_frame = pd.DataFrame({"id": oof_ids})
    for idx, cls in enumerate(class_names):
        oof_frame[f"proba_{cls}"] = final_oof[:, idx]
    oof_frame["pred_class"] = [class_names[int(i)] for i in final_oof.argmax(axis=1)]
    oof_frame["true_class"] = train[target].astype(str).to_numpy()
    oof_frame.to_csv(output_dir / "oof_predictions.csv", index=False)

    # CV summary
    cv_summary = {}
    for name in model_names:
        scores = cv_scores[name]
        cv_summary[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}

    # Metrics
    metrics = {
        "schema": "academic_research_os.local_ensemble_metrics.v2",
        "status": "passed",
        "task_id": task_id,
        "run_id": run_id,
        "runner": "spaceship_enhanced_ensemble_hgb_gbc_rf_et",
        "task_type": "classification",
        "metric": "accuracy",
        "metric_direction": "maximize",
        "n_folds": n_folds,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features_after_encoding": int(x_all.shape[1]),
        "classes": [str(c) for c in class_names],
        "cv_fold_accuracy": cv_summary,
        "oof_accuracy": {name: oof_scores[name]["accuracy"] for name in model_names},
        "oof_balanced_accuracy": {name: oof_scores[name]["balanced_accuracy"] for name in model_names},
        "oof_log_loss": {name: oof_scores[name]["log_loss"] for name in model_names},
        "ensemble": {
            "blend": {
                "weights": {model_names[i]: round(best_weights[i], 4) for i in range(n_models)},
                "accuracy": blend_acc,
                "balanced_accuracy": blend_bal_acc,
                "log_loss": float(log_loss(y, blend_oof, labels=list(range(n_classes)))),
            },
            "stack": {
                "accuracy": stack_acc,
                "balanced_accuracy": stack_bal_acc,
                "log_loss": float(log_loss(y, stack_oof, labels=list(range(n_classes)))),
            },
            "best_method": best_method,
            "selection_metric": "accuracy",
            "best_validation_score": float(best_oof_acc),
            "metric_direction": "maximize",
        },
        "seconds": round(elapsed_seconds(started), 3),
        "submission_rows": int(len(submission)),
        "prediction_distribution": submission[prediction_col].value_counts().to_dict(),
        "positive_class_index": 1,
        "human_gate_required_for_official_submission": True,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Artifact manifest
    artifacts = {}
    for fpath in sorted(output_dir.rglob("*")):
        if fpath.is_file():
            rel = str(fpath.relative_to(output_dir))
            artifacts[rel] = {"path": rel, "sha256": sha256_file(fpath), "size": fpath.stat().st_size}
    manifest = {
        "schema": "academic_research_os.artifact_manifest.v1",
        "task_id": task_id,
        "run_id": run_id,
        "template_id": "spaceship_enhanced_ensemble_hgb_gbc_rf_et",
        "created_by_agent": "workstation_orchestrator",
        "stage": "training",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": artifacts,
        "gate_dependency": "SUBMISSION_APPROVAL",
        "claim_binding": f"Enhanced 4-model ensemble (HGB+GBC+RF+ET) produced OOF accuracy={best_oof_acc:.6f}",
        "metrics": {"best_method": best_method, "best_validation_score": best_oof_acc, "accuracy": best_oof_acc, "metric_direction": "maximize"},
    }
    (output_dir / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Report
    report = [
        f"# Enhanced Spaceship Titanic Ensemble Run",
        "",
        f"- task: `{task_id}`",
        f"- runner: `spaceship_enhanced_ensemble_hgb_gbc_rf_et`",
        f"- models: HGB(500iter,lr=0.04,depth=10) + GBC(400est,lr=0.05) + RF(350est,depth=14) + ET(350est,depth=14)",
        f"- folds: `{n_folds}` x seeds: `{len(seeds)}`",
        f"- best method: `{best_method}`",
        f"- best OOF accuracy: `{best_oof_acc:.6f}`",
        "",
        "## CV Accuracy (per-fold, mean +/- std)",
    ]
    for name in model_names:
        report.append(f"- {name.upper()}: {cv_summary[name]['mean']:.6f} +/- {cv_summary[name]['std']:.6f}")
    report += [
        "",
        "## OOF Scores",
        *[f"- {n.upper()}: acc={oof_scores[n]['accuracy']:.6f} bal_acc={oof_scores[n]['balanced_accuracy']:.6f} log_loss={oof_scores[n]['log_loss']:.6f}" for n in model_names],
        f"- Blend ({len(model_names)}-way): acc={blend_acc:.6f} bal_acc={blend_bal_acc:.6f} weights={{{', '.join(f'{n}:{round(best_weights[i],2)}' for i,n in enumerate(model_names))}}}",
        f"- Stack (LogisticRegression): acc={stack_acc:.6f} bal_acc={stack_bal_acc:.6f}",
        "",
        f"- submission rows: `{metrics['submission_rows']}`",
        f"- prediction distribution: {json.dumps(metrics['prediction_distribution'])}",
        "- official Kaggle submission remains behind Human Gate.",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps({
        "status": "passed", "task_id": task_id, "run_id": run_id,
        "output_dir": str(output_dir), "best_method": best_method,
        "best_validation_score": best_oof_acc, "seconds": metrics["seconds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
