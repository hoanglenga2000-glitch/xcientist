#!/usr/bin/env python3
"""Fixed dec-2021 trainer: StratifiedKFold + LGBM-only to avoid CatBoost rare-class crash."""
import os, sys, time, json, gc
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(os.path.expanduser('~/jinghw/scripts/gpu_tra'))
PREPARED_DIR = SCRIPT_DIR / 'mlebench_prepared'
RESULTS_DIR = SCRIPT_DIR / 'mlebench_proper_results'

def log(msg):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}', flush=True)

def train_one(comp_id, seed, gpu=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    comp_dir = PREPARED_DIR / comp_id
    train_file = comp_dir / 'train.csv'
    test_file = comp_dir / 'test.csv'
    sample_file = comp_dir / 'sample_submission.csv'

    if not train_file.exists():
        log(f'[{comp_id}] Train file not found: {train_file}')
        return None

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file) if test_file.exists() else None
    sample = pd.read_csv(sample_file) if sample_file.exists() else None

    log(f'[{comp_id}] Train: {train.shape}, Test: {test.shape if test is not None else "N/A"}')

    if sample is not None:
        id_col = sample.columns[0]
        target_col = sample.columns[1]
    else:
        id_col = train.columns[0]
        target_col = train.columns[-1]

    log(f'[{comp_id}] ID: "{id_col}", Target: "{target_col}"')

    if target_col not in train.columns:
        log(f'[{comp_id}] Target not found!')
        return None

    y = train[target_col].copy()
    X = train.drop(columns=[target_col])

    id_cols_drop = [c for c in X.columns if c.lower() in {'id', 'passengerid', 'imageid', 'img_id', 'filename', str(id_col).lower()}]
    X = X.drop(columns=[c for c in id_cols_drop if c in X.columns], errors='ignore')

    test_ids = None
    Xt = None
    if test is not None:
        test_ids = test[id_col].values if id_col in test.columns else np.arange(len(test))
        Xt = test.drop(columns=[c for c in id_cols_drop if c in test.columns], errors='ignore')
        if id_col in Xt.columns:
            Xt = Xt.drop(columns=[id_col])

    # Fill NaN first
    X = X.fillna({c: X[c].median() if X[c].dtype in [np.float64, np.float32, np.int64, np.int32] else X[c].mode().iloc[0] if len(X[c].mode()) > 0 else 'MISSING' for c in X.columns})
    if Xt is not None:
        Xt = Xt.fillna({c: X[c].median() if c in X.columns and X[c].dtype in [np.float64, np.float32, np.int64, np.int32] else X[c].mode().iloc[0] if c in X.columns and len(X[c].mode()) > 0 else 'MISSING' for c in Xt.columns})

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import accuracy_score, mean_squared_error

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    text_cols = [c for c in cat_cols if X[c].nunique() > 1000]
    if text_cols:
        X = X.drop(columns=text_cols)
        if Xt is not None:
            Xt = Xt.drop(columns=[c for c in text_cols if c in Xt.columns])
        cat_cols = [c for c in cat_cols if c not in text_cols]

    log(f'[{comp_id}] Num: {len(num_cols)}, Cat: {len(cat_cols)}, Text: {len(text_cols)}')

    imp = SimpleImputer(strategy='median')
    X_num = pd.DataFrame(imp.fit_transform(X[num_cols]), columns=num_cols, index=X.index) if num_cols else pd.DataFrame(index=X.index)

    X_cat = pd.DataFrame(index=X.index)
    encs = {}
    for c in cat_cols:
        le = LabelEncoder()
        vals = list(X[c].astype(str).fillna('MISSING'))
        if Xt is not None and c in Xt.columns:
            vals += list(Xt[c].astype(str).fillna('MISSING'))
        le.fit(vals)
        X_cat[c] = le.transform(X[c].astype(str).fillna('MISSING'))
        encs[c] = le

    X_combined = pd.concat([X_num, X_cat], axis=1)

    if Xt is not None:
        tn_cols = [c for c in num_cols if c in Xt.columns]
        tn = pd.DataFrame(imp.transform(Xt[tn_cols]), columns=tn_cols, index=Xt.index) if tn_cols else pd.DataFrame(index=Xt.index)
        tc = pd.DataFrame(index=Xt.index)
        for c, le in encs.items():
            if c in Xt.columns:
                tc[c] = le.transform(Xt[c].astype(str).fillna('MISSING'))
        Xt_combined = pd.concat([tn, tc], axis=1)
        for col in X_combined.columns:
            if col not in Xt_combined.columns:
                Xt_combined[col] = X_combined[col].mean()
        Xt_combined = Xt_combined[X_combined.columns]

    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X_combined), columns=X_combined.columns)
    Xn = Xs.values.astype(np.float32)

    is_clf = y.dtype == object or y.dtype == bool or y.nunique() < 20
    n_classes = y.nunique() if is_clf else 1
    log(f'[{comp_id}] Classification: {is_clf}, Classes: {n_classes}')

    yle = LabelEncoder()
    if is_clf:
        yn = yle.fit_transform(y.astype(str))
    else:
        yn = y.values.astype(np.float32)

    log(f'[{comp_id}] Features: {Xn.shape[1]}, Classes: {n_classes}')

    # StratifiedKFold to ensure all classes present in each fold
    nf = 5 if len(yn) >= 100000 else 3 if len(yn) >= 10000 else 5
    if is_clf and n_classes > 1:
        class_counts = np.bincount(yn)
        if (class_counts >= nf).all():
            kf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=seed)
            log(f'[{comp_id}] Using StratifiedKFold(n_splits={nf})')
        else:
            # Fall back to fewer folds
            nf = int(class_counts.min())
            nf = max(2, min(nf, 5))
            if (class_counts >= nf).all():
                kf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=seed)
                log(f'[{comp_id}] Using StratifiedKFold(n_splits={nf}) - reduced due to rare classes')
            else:
                kf = KFold(n_splits=nf, shuffle=True, random_state=seed)
                log(f'[{comp_id}] Using KFold(n_splits={nf}) - stratification not possible')
    else:
        kf = KFold(n_splits=nf, shuffle=True, random_state=seed)

    oof_preds = np.zeros((len(Xn), n_classes)) if is_clf and n_classes > 1 else np.zeros(len(Xn))
    test_ps = []

    if Xt is not None:
        Xtn_vals = Xt_combined.values.astype(np.float32)
    has_test = Xt is not None

    from lightgbm import LGBMClassifier, LGBMRegressor
    n_est = min(500, max(100, len(Xn) // 1000))

    def safe_predict_proba(model, X_data, n_classes):
        """Handle LGBM returning fewer classes when some are missing from training."""
        p = model.predict_proba(X_data)
        if p.shape[1] < n_classes:
            full_p = np.zeros((len(p), n_classes), dtype=p.dtype)
            full_p[:, model.classes_.astype(int)] = p
            return full_p
        return p

    for fold, (ti, vi) in enumerate(kf.split(Xn, yn)):
        xtr, xva = Xn[ti], Xn[vi]
        ytr, yva = yn[ti], yn[vi]

        try:
            if is_clf and n_classes > 2:
                lgb = LGBMClassifier(n_estimators=n_est, learning_rate=0.05, max_depth=8,
                    num_leaves=127, random_state=seed+fold, verbose=-1, n_jobs=-1)
            elif is_clf:
                lgb = LGBMClassifier(n_estimators=n_est, learning_rate=0.05, max_depth=8,
                    random_state=seed+fold, verbose=-1, n_jobs=-1)
            else:
                lgb = LGBMRegressor(n_estimators=n_est, learning_rate=0.05, max_depth=8,
                    random_state=seed+fold, verbose=-1, n_jobs=-1)

            lgb.fit(xtr, ytr)
            log(f'[{comp_id}]   Fold {fold+1}/{nf} trained on {len(np.unique(ytr))} classes: {np.unique(ytr)}')

            if has_test:
                if is_clf and n_classes > 1:
                    tp = safe_predict_proba(lgb, Xtn_vals, n_classes)
                else:
                    tp = lgb.predict(Xtn_vals).reshape(-1, 1)
                test_ps.append(tp)

            if is_clf and n_classes > 1:
                p = safe_predict_proba(lgb, xva, n_classes)
                if n_classes > 2:
                    oof_preds[vi] = p
                else:
                    oof_preds[vi] = p[:, 1] if p.shape[1] > 1 else p.ravel()
                pred = p.argmax(axis=1)
                acc = (pred == yva).mean()
            else:
                p = lgb.predict(xva).ravel()
                pred = (p > 0.5).astype(int)
                oof_preds[vi] = p
                acc = (pred == yva).mean()

            log(f'[{comp_id}]   Fold {fold+1}/{nf}: {acc:.5f}')
        except Exception as e:
            log(f'[{comp_id}]   Fold {fold+1}/{nf} LGB error: {e}')

        gc.collect()

    # Final prediction
    sub = None
    if has_test and test_ps:
        tp_final = np.mean(test_ps, axis=0)

        if is_clf and n_classes > 2 and tp_final.ndim > 1:
            pred_indices = tp_final.argmax(axis=1)
            pred_values = yle.inverse_transform(pred_indices)
        elif is_clf and tp_final.ndim > 1:
            tp_flat = tp_final[:, 1] if tp_final.shape[1] > 1 else tp_final.ravel()
            pred_indices = (tp_flat > 0.5).astype(int)
            pred_values = yle.inverse_transform(pred_indices)
        else:
            pred_values = tp_final.ravel()

        if test_ids is not None and len(test_ids) == len(pred_values):
            sub = pd.DataFrame({id_col: test_ids, target_col: pred_values})
        else:
            sub = pd.DataFrame({id_col: range(len(pred_values)), target_col: pred_values})

        if sample is not None:
            sample_dtype = sample[target_col].dtype
            if sample_dtype != sub[target_col].dtype:
                try:
                    if sample_dtype == bool and sub[target_col].dtype == object:
                        sub[target_col] = sub[target_col].map({'True': True, 'False': False}).astype(bool)
                    else:
                        sub[target_col] = sub[target_col].astype(sample_dtype)
                except Exception:
                    pass

        if is_clf and n_classes == 2:
            true_count = int(np.sum(pred_values == 'True'))
            log(f'[{comp_id}] Test predictions: {true_count/len(pred_values)*100:.1f}% positive ({true_count}/{len(pred_values)})')

    od = RESULTS_DIR / comp_id
    os.makedirs(od, exist_ok=True)
    if sub is not None:
        sub.to_csv(od / f'submission_s{seed}.csv', index=False)
        log(f'[{comp_id}] Saved submission: {len(sub)} rows')

    # OOF score
    if is_clf:
        if oof_preds.ndim > 1 and oof_preds.shape[1] > 1:
            oof_preds_final = oof_preds.argmax(axis=1)
        else:
            oof_preds_final = (oof_preds > 0.5).astype(int) if oof_preds.ndim == 1 else oof_preds.argmax(axis=1)
        oof_preds_labels = yle.inverse_transform(oof_preds_final)
        yn_labels = yle.inverse_transform(yn)
        oof_score = accuracy_score(yn_labels, oof_preds_labels)
    else:
        oof_score = np.sqrt(mean_squared_error(yn, oof_preds))

    result = {
        'comp_id': comp_id, 'seed': seed,
        'n_samples': int(len(yn)), 'n_features': Xn.shape[1],
        'is_clf': is_clf, 'n_folds': nf, 'n_classes': int(n_classes),
        'oof_score': float(oof_score),
    }

    with open(od / f'result_s{seed}.json', 'w') as f:
        json.dump(result, f)

    log(f'[{comp_id}] DONE: OOF={oof_score:.5f} folds={nf} features={Xn.shape[1]}')
    return result


if __name__ == '__main__':
    import sys
    comp_id = sys.argv[1]
    seed = int(sys.argv[2])
    gpu = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    train_one(comp_id, seed, gpu)
