#!/usr/bin/env python3
"""Debug the prediction pipeline for spaceship-titanic."""
import sys, os
import numpy as np, pandas as pd
from pathlib import Path

SCRIPT_DIR = Path('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra')
PREPARED_DIR = SCRIPT_DIR / 'mlebench_prepared'

comp_id = 'spaceship-titanic'
comp_dir = PREPARED_DIR / comp_id
train = pd.read_csv(comp_dir / 'train.csv')
test = pd.read_csv(comp_dir / 'test.csv')
sample = pd.read_csv(comp_dir / 'sample_submission.csv')

print(f'Train: {train.shape}, Test: {test.shape}')

id_col = 'PassengerId'
target_col = 'Transported'

y = train[target_col].copy()
X = train.drop(columns=[target_col])

id_cols = [c for c in X.columns if c.lower() in {'id', 'passengerid', 'imageid', 'img_id', 'filename'}]
X = X.drop(columns=[c for c in id_cols if c in X.columns], errors='ignore')

test_ids = test[id_col].values
Xt = test.drop(columns=[c for c in id_cols if c in test.columns], errors='ignore')
if id_col in Xt.columns:
    Xt = Xt.drop(columns=[id_col])

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

text_cols = [c for c in cat_cols if X[c].nunique() > 1000]
if text_cols:
    X = X.drop(columns=text_cols)
    Xt = Xt.drop(columns=[c for c in text_cols if c in Xt.columns])
    cat_cols = [c for c in cat_cols if c not in text_cols]

imp = SimpleImputer(strategy='median')
X_num = pd.DataFrame(imp.fit_transform(X[num_cols]), columns=num_cols, index=X.index) if num_cols else pd.DataFrame(index=X.index)

X_cat = pd.DataFrame(index=X.index)
encs = {}
for c in cat_cols:
    le = LabelEncoder()
    vals = list(X[c].astype(str).fillna('MISSING'))
    if c in Xt.columns:
        vals += list(Xt[c].astype(str).fillna('MISSING'))
    le.fit(vals)
    X_cat[c] = le.transform(X[c].astype(str).fillna('MISSING'))
    encs[c] = le

Xp = pd.concat([X_num, X_cat], axis=1)

tn_cols = [c for c in num_cols if c in Xt.columns]
tn = pd.DataFrame(imp.transform(Xt[tn_cols]), columns=tn_cols, index=Xt.index) if tn_cols else pd.DataFrame(index=Xt.index)
tc = pd.DataFrame(index=Xt.index)
for c, le in encs.items():
    if c in Xt.columns:
        tc[c] = le.transform(Xt[c].astype(str).fillna('MISSING'))
Xtp = pd.concat([tn, tc], axis=1)
for col in Xp.columns:
    if col not in Xtp.columns:
        Xtp[col] = 0
Xtp = Xtp[Xp.columns]

scaler = StandardScaler()
Xs = pd.DataFrame(scaler.fit_transform(Xp), columns=Xp.columns)
Xtn_vals = Xtp.values.astype(np.float32)

Xn = Xs.values.astype(np.float32)
yn = y.values.astype(int)

kf = KFold(n_splits=2, shuffle=True, random_state=42)

test_ps = []
for fold, (ti, vi) in enumerate(kf.split(Xn, yn)):
    xtr, xva = Xn[ti], Xn[vi]
    ytr, yva = yn[ti], yn[vi]

    try:
        from lightgbm import LGBMClassifier
        lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42+fold, verbose=-1)
        lgb.fit(xtr, ytr)
        tp = lgb.predict_proba(Xtn_vals)
        test_ps.append(tp)
        p = lgb.predict_proba(xva)
        pred = p.argmax(axis=1)
        acc = (pred == yva).mean()
        print(f'Fold {fold} LGB: Test proba True mean={tp[:,1].mean():.4f}, CV acc={acc:.4f}')
        print(f'  Pred True on test: {(tp[:,1] > 0.5).sum()}')
        print(f'  Argmax True on test: {(tp.argmax(axis=1) == 1).sum()}')
    except Exception as e:
        print(f'LGB error: {e}')

    try:
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
            task_type='GPU', devices=[0], random_seed=42+fold, verbose=0, early_stopping_rounds=50)
        cb.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
        tp = cb.predict_proba(Xtn_vals)
        test_ps.append(tp)
        p = cb.predict_proba(xva)
        pred = p.argmax(axis=1)
        acc = (pred == yva).mean()
        print(f'Fold {fold} CB: Test proba True mean={tp[:,1].mean():.4f}, CV acc={acc:.4f}')
        print(f'  Pred True on test: {(tp[:,1] > 0.5).sum()}')
        print(f'  Argmax True on test: {(tp.argmax(axis=1) == 1).sum()}')
    except Exception as e:
        print(f'CB error: {e}')

print()
print(f'Total test_ps: {len(test_ps)} arrays')
if test_ps:
    tp_final = np.mean(test_ps, axis=0)
    print(f'tp_final shape: {tp_final.shape}')
    print(f'tp_final mean by class: col0={tp_final[:,0].mean():.4f}, col1={tp_final[:,1].mean():.4f}')
    print(f'Argmax True count: {(tp_final.argmax(axis=1) == 1).sum()}')
    print(f'Threshold >0.5 True count: {(tp_final[:,1] > 0.5).sum()}')
