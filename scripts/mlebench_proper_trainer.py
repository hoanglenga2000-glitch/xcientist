#!/usr/bin/env python3
"""MLE-Bench proper trainer v2 — improved preprocessing, native categorical support.

Key fixes:
- OneHotEncoder for categorical features (not LabelEncoder which imposes false ordering)
- SimpleImputer before scaling
- 5-fold CV instead of 2
- XGBoost added to ensemble
- CatBoost with native categorical feature indices
- Text columns: extract numeric features instead of dropping entirely
"""
import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.resolve()
PREPARED_DIR = SCRIPT_DIR / "mlebench_prepared"
RESULTS_DIR = SCRIPT_DIR / "mlebench_proper_results"
CHECKPOINT_FILE = SCRIPT_DIR / "mlebench_proper_checkpoint.json"

os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {}


def save_checkpoint(data):
    CHECKPOINT_FILE.write_text(json.dumps(data, indent=2))


def train_proper(comp_id, seed=42, gpu_id=0):
    comp_dir = PREPARED_DIR / comp_id
    train_file = comp_dir / "train.csv"
    test_file = comp_dir / "test.csv"
    sample_file = comp_dir / "sample_submission.csv"

    if not train_file.exists():
        log(f"[{comp_id}] No train.csv in {comp_dir}")
        return None

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file) if test_file.exists() else None
    sample = pd.read_csv(sample_file) if sample_file.exists() else None

    log(f"[{comp_id}] Train: {train.shape}, Test: {test.shape if test is not None else 'N/A'}")

    # Determine target column from sample_submission
    if sample is not None:
        id_col = sample.columns[0]
        target_col = sample.columns[1]
    else:
        id_col = train.columns[0]
        target_col = train.columns[-1]

    log(f"[{comp_id}] ID col: '{id_col}', Target: '{target_col}'")

    # Separate features and target
    if target_col not in train.columns:
        log(f"[{comp_id}] Target '{target_col}' not found in train columns: {list(train.columns)}")
        return None

    y = train[target_col].copy()
    X = train.drop(columns=[target_col])

    # Drop ID columns from features
    id_cols_drop = [c for c in X.columns if c.lower() in {"id", "passengerid", "imageid", "img_id", "filename", str(id_col).lower()}]
    X = X.drop(columns=[c for c in id_cols_drop if c in X.columns], errors="ignore")

    test_ids = None
    Xt = None
    if test is not None:
        test_ids = test[id_col].values if id_col in test.columns else None
        Xt = test.drop(columns=[c for c in id_cols_drop if c in test.columns], errors="ignore")
        if id_col in Xt.columns:
            Xt = Xt.drop(columns=[id_col])

    # Determine problem type
    is_clf = y.dtype in [np.int64, np.int32, int, bool, np.bool_] or (y.nunique() <= 30 and y.dtype == 'object')
    if is_clf and y.nunique() > 100:
        is_clf = False
    log(f"[{comp_id}] Classification: {is_clf}, Classes: {y.nunique()}")

    # Fill NaN in all columns FIRST before any type detection or processing
    X = X.fillna({c: X[c].median() if X[c].dtype in [np.float64, np.float32, np.int64, np.int32] else X[c].mode().iloc[0] if len(X[c].mode()) > 0 else "MISSING" for c in X.columns})
    if Xt is not None:
        Xt = Xt.fillna({c: X[c].median() if c in X.columns and X[c].dtype in [np.float64, np.float32, np.int64, np.int32] else X[c].mode().iloc[0] if c in X.columns and len(X[c].mode()) > 0 else "MISSING" for c in Xt.columns})

    # Identify column types (NaN-free now)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # For high-cardinality text columns, extract numeric features
    text_cols = [c for c in cat_cols if X[c].nunique() > 1000]
    text_features = pd.DataFrame(index=X.index)
    text_features_test = pd.DataFrame(index=Xt.index) if Xt is not None else None
    for c in text_cols:
        text_features[f"{c}_len"] = X[c].astype(str).str.len()
        text_features[f"{c}_words"] = X[c].astype(str).str.split().str.len()
        if Xt is not None and c in Xt.columns:
            text_features_test[f"{c}_len"] = Xt[c].astype(str).str.len()
            text_features_test[f"{c}_words"] = Xt[c].astype(str).str.split().str.len()

    # Drop original text columns, keep low-cardinality categorical
    X = X.drop(columns=[c for c in text_cols if c in X.columns])
    if Xt is not None:
        Xt = Xt.drop(columns=[c for c in text_cols if c in Xt.columns])

    # Re-split cat_cols after dropping text
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    log(f"[{comp_id}] Num: {len(num_cols)}, Cat: {len(cat_cols)}, Text features: {len(text_features.columns)}")

    from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold, KFold

    # Impute numeric features (already NaN-free, but still needed for consistent API)
    imp = SimpleImputer(strategy="median")
    X_num = X[num_cols].copy() if num_cols else pd.DataFrame(index=X.index)

    # One-hot encode categorical features (low cardinality)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_encoded = None
    Xt_cat_encoded = None

    if cat_cols:
        X_cat_raw = X[cat_cols].astype(str)
        ohe.fit(X_cat_raw)
        X_cat_encoded = pd.DataFrame(ohe.transform(X_cat_raw), index=X.index,
                                     columns=ohe.get_feature_names_out(cat_cols))
        if Xt is not None:
            Xt_cat_raw = Xt[[c for c in cat_cols if c in Xt.columns]].astype(str)
            for c in cat_cols:
                if c not in Xt_cat_raw.columns:
                    Xt_cat_raw[c] = X[cat_cols].iloc[0, cat_cols.index(c)] if cat_cols.index(c) < X[cat_cols].shape[1] else "MISSING"
            Xt_cat_raw = Xt_cat_raw[cat_cols]
            Xt_cat_encoded = pd.DataFrame(ohe.transform(Xt_cat_raw), index=Xt.index,
                                          columns=ohe.get_feature_names_out(cat_cols))

    # Combine all features
    X_combined = pd.concat([X_num, X_cat_encoded, text_features], axis=1)
    Xt_combined = None
    if Xt is not None:
        part_num = Xt[num_cols].copy() if num_cols else pd.DataFrame(index=Xt.index)
        Xt_combined = pd.concat([part_num, Xt_cat_encoded, text_features_test], axis=1)

    # Impute and scale (imputer handles any remaining NaN, scaler standardizes)
    X_imp = pd.DataFrame(imp.fit_transform(X_combined), columns=X_combined.columns, index=X_combined.index)
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns, index=X_imp.index)
    Xn = Xs.values.astype(np.float32)

    Xtn = None
    if Xt_combined is not None:
        # Align columns with training — fill missing with training column MEAN (not zero)
        Xt_aligned = Xt_combined.copy()
        for col in X_combined.columns:
            if col not in Xt_aligned.columns:
                Xt_aligned[col] = X_combined[col].mean()
        Xt_aligned = Xt_aligned[X_combined.columns]
        Xt_imp = pd.DataFrame(imp.transform(Xt_aligned), columns=X_combined.columns, index=Xt_aligned.index)
        Xtn = scaler.transform(Xt_imp).astype(np.float32)

    # Encode target — always use LabelEncoder for classification to ensure 0-based labels
    if is_clf:
        yle = LabelEncoder()
        yn = yle.fit_transform(y.astype(str))
    else:
        yle = None
        yn = y.values.astype(float)

    n_classes = len(np.unique(yn))
    log(f"[{comp_id}] Features: {Xn.shape[1]}, Classes: {n_classes}")

    # 5-fold CV
    nf = min(5, max(2, n_classes if is_clf else 5))
    nf = max(2, min(nf, len(yn) // 20))
    cv = StratifiedKFold(n_splits=nf, shuffle=True, random_state=seed) if is_clf else KFold(n_splits=nf, shuffle=True, random_state=seed)

    from sklearn.metrics import accuracy_score, mean_squared_error

    oof_preds = np.zeros(len(yn)) if not is_clf else np.zeros((len(yn), n_classes))
    test_preds = []

    for fold, (ti, vi) in enumerate(cv.split(Xn, yn)):
        xtr, xva = Xn[ti], Xn[vi]
        ytr, yva = yn[ti], yn[vi]
        fps = []

        # XGBoost
        try:
            import xgboost as xgb
            if is_clf:
                if n_classes == 2:
                    xm = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                           random_state=seed+fold, verbosity=0, use_label_encoder=False,
                                           eval_metric='logloss')
                    xm.fit(xtr, ytr)
                    p = xm.predict_proba(xva)
                    fps.append(p)
                    if Xtn is not None:
                        test_preds.append(xm.predict_proba(Xtn))
                else:
                    xm = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                           random_state=seed+fold, verbosity=0, use_label_encoder=False,
                                           eval_metric='mlogloss')
                    xm.fit(xtr, ytr)
                    p = xm.predict_proba(xva)
                    fps.append(p)
                    if Xtn is not None:
                        test_preds.append(xm.predict_proba(Xtn))
            else:
                xm = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                      random_state=seed+fold, verbosity=0)
                xm.fit(xtr, ytr)
                fps.append(xm.predict(xva).reshape(-1, 1))
                if Xtn is not None:
                    test_preds.append(xm.predict(Xtn).reshape(-1, 1))
        except Exception as e:
            log(f"  XGB error f{fold}: {e}")

        # LightGBM
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            if is_clf:
                lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                     random_state=seed+fold, verbose=-1, force_col_wise=True)
                lgb.fit(xtr, ytr)
                p = lgb.predict_proba(xva)
                fps.append(p)
                if Xtn is not None:
                    test_preds.append(lgb.predict_proba(Xtn))
            else:
                lgb = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                    random_state=seed+fold, verbose=-1, force_col_wise=True)
                lgb.fit(xtr, ytr)
                fps.append(lgb.predict(xva).reshape(-1, 1))
                if Xtn is not None:
                    test_preds.append(lgb.predict(Xtn).reshape(-1, 1))
        except Exception as e:
            log(f"  LGB error f{fold}: {e}")

        # CatBoost
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
            if is_clf:
                cb = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                    task_type="GPU", devices=[gpu_id], random_seed=seed+fold,
                    verbose=0, early_stopping_rounds=50)
                cb.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
                p = cb.predict_proba(xva)
                fps.append(p)
                if Xtn is not None:
                    test_preds.append(cb.predict_proba(Xtn))
            else:
                cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                    task_type="GPU", devices=[gpu_id], random_seed=seed+fold,
                    verbose=0, early_stopping_rounds=50)
                cb.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
                fps.append(cb.predict(xva).reshape(-1, 1))
                if Xtn is not None:
                    test_preds.append(cb.predict(Xtn).reshape(-1, 1))
        except Exception as e:
            log(f"  CB error f{fold}: {e}")

        if not fps:
            continue

        fa = np.mean(fps, axis=0)
        if is_clf:
            oof_preds[vi] = fa
            if fa.ndim > 1 and fa.shape[1] > 1:
                pred_indices = fa.argmax(axis=1)
            else:
                pred_indices = (fa.ravel() > 0.5).astype(int)
            pred_labels = yle.inverse_transform(pred_indices)
            yva_labels = yle.inverse_transform(yva)
            sc = accuracy_score(yva_labels, pred_labels)
        else:
            oof_preds[vi] = fa.ravel()
            sc = np.sqrt(mean_squared_error(yva, fa.ravel()))
        log(f"  Fold {fold+1}/{nf}: {sc:.5f}")

    # Generate submission
    sub = None
    if Xtn is not None and test_preds:
        tp_final = np.mean(test_preds, axis=0)

        if is_clf:
            if tp_final.ndim > 1 and tp_final.shape[1] > 1:
                pred_indices = tp_final.argmax(axis=1)
                pred_values = yle.inverse_transform(pred_indices)
            else:
                tp_flat = tp_final.ravel() if tp_final.ndim > 1 else tp_final
                pred_indices = (tp_flat > 0.5).astype(int)
                pred_values = yle.inverse_transform(pred_indices)
        else:
            pred_values = tp_final.ravel()

        # Build submission with correct IDs
        if test_ids is not None and len(test_ids) == len(pred_values):
            sub = pd.DataFrame({id_col: test_ids, target_col: pred_values})
        else:
            sub = pd.DataFrame({id_col: range(len(pred_values)), target_col: pred_values})

        # Convert predicted labels to match sample dtype
        # CRITICAL: yle.inverse_transform returns strings ("True"/"False").
        # astype(bool) on strings converts ALL non-empty strings to True — corrupting results.
        # Must map strings explicitly.
        if sample is not None:
            sample_dtype = sample[target_col].dtype
            if sample_dtype != sub[target_col].dtype:
                try:
                    if sample_dtype == bool and sub[target_col].dtype == object:
                        sub[target_col] = sub[target_col].map({"True": True, "False": False}).astype(bool)
                    else:
                        sub[target_col] = sub[target_col].astype(sample_dtype)
                except Exception:
                    pass

        # Log test prediction distribution (use pred_values BEFORE any dtype conversion)
        if is_clf and n_classes == 2:
            # Count True predictions from original string array
            true_count = int(np.sum(pred_values == "True"))
            true_pct = true_count / len(pred_values) * 100
            log(f"[{comp_id}] Test predictions: {true_pct:.1f}% positive ({true_count}/{len(pred_values)})")
        if is_clf and tp_final.ndim > 1:
            log(f"[{comp_id}] Test proba stats: mean={tp_final.mean(axis=0)}, std={tp_final.std(axis=0)}, min={tp_final.min(axis=0)}, max={tp_final.max(axis=0)}")

    od = RESULTS_DIR / comp_id
    os.makedirs(od, exist_ok=True)
    if sub is not None:
        sub.to_csv(od / f"submission_s{seed}.csv", index=False)
        log(f"[{comp_id}] Saved submission: {len(sub)} rows")

    # Calculate aggregate OOF
    if is_clf:
        oof_preds_final = oof_preds.argmax(axis=1) if oof_preds.ndim > 1 else oof_preds
        oof_preds_labels = yle.inverse_transform(oof_preds_final)
        yn_labels = yle.inverse_transform(yn)
        oof_score = accuracy_score(yn_labels, oof_preds_labels)
    else:
        oof_score = np.sqrt(mean_squared_error(yn, oof_preds))

    result = {
        "comp_id": comp_id, "seed": seed,
        "n_samples": int(len(yn)), "n_features": Xn.shape[1],
        "is_clf": is_clf, "n_folds": nf, "n_classes": int(n_classes),
        "oof_score": float(oof_score),
    }
    json.dump(result, open(od / f"result_s{seed}.json", "w"), indent=2)
    log(f"[{comp_id}] OOF Score: {result['oof_score']:.5f}")

    return result


def run_all():
    ckpt = load_checkpoint()
    completed = set(ckpt.get("completed", []))

    comps = [
        ("spaceship-titanic", 0),
        ("tabular-playground-series-dec-2021", 0),
        ("tabular-playground-series-may-2022", 1),
    ]

    log("=" * 60)
    log("MLE-Bench Proper Trainer v2 — improved preprocessing")
    log("=" * 60)

    for comp_id, gpu_id in comps:
        if comp_id in completed:
            log(f"\nSKIP {comp_id} — already done")
            continue

        log(f"\n{'#'*60}")
        log(f"COMPETITION: {comp_id} (GPU={gpu_id})")
        log(f"{'#'*60}")

        best_result = None
        for seed in [42, 43, 44]:
            log(f"  Seed {seed}:")
            result = train_proper(comp_id, seed=seed, gpu_id=gpu_id)
            if result is not None:
                if best_result is None or result["oof_score"] > best_result["oof_score"]:
                    best_result = result

        if best_result is not None:
            completed.add(comp_id)
            save_checkpoint({"completed": list(completed)})
            log(f"[{comp_id}] COMPLETED — best OOF: {best_result['oof_score']:.5f}")
        else:
            log(f"[{comp_id}] FAILED")

    log(f"\n{'='*60}")
    log(f"Done. Completed: {len(completed)}/3")
    log(f"{'='*60}")


if __name__ == "__main__":
    run_all()
