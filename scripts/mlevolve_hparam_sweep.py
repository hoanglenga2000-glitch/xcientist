"""
MLEvolve-style targeted hyperparameter optimization.
Searches 20-50 hyperparameter combinations for best single-model OOF.
Then blends the best variants with optimized weights.
Goal: push spaceship_titanic OOF from 0.8163 to >0.820
"""
import json, os, sys, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]

def load_spaceship():
    train = pd.read_csv(ROOT / "tasks/spaceship_titanic/data/train.csv")
    test = pd.read_csv(ROOT / "tasks/spaceship_titanic/data/test.csv")

    # Feature engineering (same as V2)
    for df in [train, test]:
        df['Cabin_deck'] = df['Cabin'].str.split('/').str[0].fillna('Unknown')
        df['Cabin_num'] = pd.to_numeric(df['Cabin'].str.split('/').str[1], errors='coerce').fillna(-1)
        df['Cabin_side'] = df['Cabin'].str.split('/').str[2].fillna('Unknown')
        df['Group'] = df['PassengerId'].str[:4]
        for col in ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']:
            df[col] = df[col].fillna(0)
        df['TotalSpend'] = df['RoomService']+df['FoodCourt']+df['ShoppingMall']+df['Spa']+df['VRDeck']
        df['HasSpend'] = (df['TotalSpend']>0).astype(int)
        df['SpendPerService'] = df['TotalSpend'] / (df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]>0).sum(axis=1).clip(1)
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Age_bucket'] = pd.cut(df['Age'], bins=[0,12,18,30,50,100], labels=[0,1,2,3,4])
        df['VIP'] = df['VIP'].fillna(False)
        df['CryoSleep'] = df['CryoSleep'].fillna(False)
        df['VIP_Age'] = df['VIP'].astype(int) * df['Age']
        df['CryoSleep_Spend'] = df['CryoSleep'].astype(int) * df['TotalSpend']
        # NEW: spending ratios
        for col in ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']:
            df[f'{col}_ratio'] = df[col] / df['TotalSpend'].clip(1)
        # NEW: group aggregates
        group_counts = df.groupby('Group').size().to_dict()
        df['GroupSize'] = df['Group'].map(group_counts)
        # NEW: deck-based aggregates
        deck_spend = df.groupby('Cabin_deck')['TotalSpend'].mean().to_dict()
        df['DeckAvgSpend'] = df['Cabin_deck'].map(deck_spend)

    target = 'Transported'
    y = (train[target] == True).astype(int).values

    cat_cols = ['HomePlanet','CryoSleep','Destination','VIP','Cabin_deck','Cabin_side']
    drop_cols = [target,'PassengerId','Name','Cabin']

    X = train.drop(columns=[c for c in drop_cols if c in train.columns], errors='ignore')
    X_test = test.drop(columns=[c for c in drop_cols if c in test.columns], errors='ignore')

    # Encode categoricals
    for col in cat_cols:
        if col in X.columns:
            le = LabelEncoder()
            combined = pd.concat([X[col].astype(str), X_test[col].astype(str)])
            le.fit(combined)
            X[col] = le.transform(X[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))

    # Align and handle categoricals
    common = list(set(X.columns) & set(X_test.columns))
    X = X[common]
    X_test = X_test[common]
    # Convert any categorical columns to numeric
    for col in X.columns:
        if X[col].dtype.name == 'category':
            X[col] = X[col].astype(float).fillna(-1)
            X_test[col] = X_test[col].astype(float).fillna(-1)
    X = X.fillna(-1).astype(np.float64)
    X_test = X_test.fillna(-1).astype(np.float64)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_test_s = scaler.transform(X_test)

    return X_s, y, X_test_s, test['PassengerId']

def eval_model(model, X, y, n_folds=3):
    """Quick 3-fold OOF evaluation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, val in skf.split(X, y):
        X_tr, X_val = X[tr], X[val]
        y_tr, y_val = y[tr], y[val]
        model.fit(X_tr, y_tr)
        oof[val] = model.predict_proba(X_val)[:, 1]
    return accuracy_score(y, (oof > 0.5).astype(int)), oof

# Load data once
X, y, X_test, test_ids = load_spaceship()
print(f"Data: {X.shape}, features={X.shape[1]}")

# Baseline: single 5-fold HGB
from sklearn.ensemble import HistGradientBoostingClassifier
base_model = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=None, random_state=42, early_stopping=True, validation_fraction=0.1)
base_acc, base_oof = eval_model(base_model, X, y, n_folds=5)
print(f"\nBaseline HGB (5-fold): {base_acc:.6f}")

# Sweep 1: CatBoost variants (best single model)
print("\n=== CatBoost Sweep ===")
from catboost import CatBoostClassifier
best_cb_acc = 0
best_cb_params = None
cb_configs = [
    {"iterations": 500, "learning_rate": 0.03, "depth": 6},
    {"iterations": 800, "learning_rate": 0.02, "depth": 7},
    {"iterations": 1000, "learning_rate": 0.02, "depth": 8},
    {"iterations": 500, "learning_rate": 0.05, "depth": 5},
    {"iterations": 600, "learning_rate": 0.03, "depth": 7, "l2_leaf_reg": 3},
    {"iterations": 800, "learning_rate": 0.02, "depth": 6, "border_count": 128},
    {"iterations": 500, "learning_rate": 0.04, "depth": 6, "bagging_temperature": 0.5},
    {"iterations": 1000, "learning_rate": 0.015, "depth": 8, "l2_leaf_reg": 5},
]
for i, cfg in enumerate(cb_configs):
    model = CatBoostClassifier(**cfg, random_seed=42, verbose=False, thread_count=-1)
    acc, oof = eval_model(model, X, y, n_folds=3)
    if acc > best_cb_acc:
        best_cb_acc = acc
        best_cb_params = cfg
        best_cb_oof = oof
    print(f"  #{i+1}: acc={acc:.6f} | {cfg}")

print(f"\nBest CatBoost: {best_cb_acc:.6f} | {best_cb_params}")

# Sweep 2: HGB variants
print("\n=== HGB Sweep ===")
best_hgb_acc = 0
best_hgb_params = None
hgb_configs = [
    {"max_iter": 300, "learning_rate": 0.1, "max_depth": None},
    {"max_iter": 500, "learning_rate": 0.05, "max_depth": None},
    {"max_iter": 800, "learning_rate": 0.03, "max_depth": 10},
    {"max_iter": 500, "learning_rate": 0.05, "max_depth": 8, "l2_regularization": 0.1},
    {"max_iter": 600, "learning_rate": 0.04, "max_depth": None, "max_leaf_nodes": 63},
    {"max_iter": 400, "learning_rate": 0.07, "max_depth": 6, "min_samples_leaf": 10},
]
for i, cfg in enumerate(hgb_configs):
    model = HistGradientBoostingClassifier(**cfg, random_state=42, early_stopping=True, validation_fraction=0.1)
    acc, oof = eval_model(model, X, y, n_folds=3)
    if acc > best_hgb_acc:
        best_hgb_acc = acc
        best_hgb_params = cfg
        best_hgb_oof = oof
    print(f"  #{i+1}: acc={acc:.6f} | {cfg}")

print(f"\nBest HGB: {best_hgb_acc:.6f} | {best_hgb_params}")

# Sweep 3: LGB variants
print("\n=== LGB Sweep ===")
import lightgbm as lgb
best_lgb_acc = 0
best_lgb_params = None
lgb_configs = [
    {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 63},
    {"n_estimators": 800, "learning_rate": 0.02, "num_leaves": 95},
    {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 47, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 600, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 15},
    {"n_estimators": 1000, "learning_rate": 0.015, "num_leaves": 127},
]
for i, cfg in enumerate(lgb_configs):
    model = lgb.LGBMClassifier(**cfg, random_state=42, verbose=-1, n_jobs=-1)
    acc, oof = eval_model(model, X, y, n_folds=3)
    if acc > best_lgb_acc:
        best_lgb_acc = acc
        best_lgb_params = cfg
        best_lgb_oof = oof
    print(f"  #{i+1}: acc={acc:.6f} | {cfg}")

print(f"\nBest LGB: {best_lgb_acc:.6f} | {best_lgb_params}")

# Final ensemble blend
print(f"\n{'='*50}")
print("FINAL BLEND")
print(f"{'='*50}")
print(f"Best CatBoost: {best_cb_acc:.6f}")
print(f"Best HGB:      {best_hgb_acc:.6f}")
print(f"Best LGB:      {best_lgb_acc:.6f}")

# Try all weight combinations
best_ensemble_acc = 0
best_ensemble_weights = None
for w_cb in np.arange(0, 1.01, 0.05):
    for w_hgb in np.arange(0, 1.01 - w_cb, 0.05):
        w_lgb = 1.0 - w_cb - w_hgb
        blend = w_cb * best_cb_oof + w_hgb * best_hgb_oof + w_lgb * best_lgb_oof
        acc = accuracy_score(y, (blend > 0.5).astype(int))
        if acc > best_ensemble_acc:
            best_ensemble_acc = acc
            best_ensemble_weights = (w_cb, w_hgb, w_lgb)

print(f"\nBest ensemble OOF: {best_ensemble_acc:.6f}")
print(f"Weights: CatBoost={best_ensemble_weights[0]:.2f}, HGB={best_ensemble_weights[1]:.2f}, LGB={best_ensemble_weights[2]:.2f}")

# Generate test predictions
cb_test = np.zeros(len(X_test))
hgb_test = np.zeros(len(X_test))
lgb_test = np.zeros(len(X_test))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for tr, val in skf.split(X, y):
    # CatBoost
    cb = CatBoostClassifier(**best_cb_params, random_seed=42, verbose=False, thread_count=-1)
    cb.fit(X[tr], y[tr])
    cb_test += cb.predict_proba(X_test)[:, 1] / 5
    # HGB
    hgb = HistGradientBoostingClassifier(**best_hgb_params, random_state=42, early_stopping=True, validation_fraction=0.1)
    hgb.fit(X[tr], y[tr])
    hgb_test += hgb.predict_proba(X_test)[:, 1] / 5
    # LGB
    lgb_m = lgb.LGBMClassifier(**best_lgb_params, random_state=42, verbose=-1, n_jobs=-1)
    lgb_m.fit(X[tr], y[tr])
    lgb_test += lgb_m.predict_proba(X_test)[:, 1] / 5

w_cb, w_hgb, w_lgb = best_ensemble_weights
blend_test = w_cb * cb_test + w_hgb * hgb_test + w_lgb * lgb_test

# Save
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out_dir = ROOT / "experiments" / "spaceship_titanic" / f"mlevolve_hparam_{ts}"
out_dir.mkdir(parents=True, exist_ok=True)

sub = pd.read_csv(ROOT / "tasks/spaceship_titanic/data/sample_submission.csv")
sub['Transported'] = (blend_test > 0.5).astype(bool)
sub.to_csv(out_dir / "submission.csv", index=False)

oof_df = pd.DataFrame({
    'prob_catboost': best_cb_oof,
    'prob_hgb': best_hgb_oof,
    'prob_lgb': best_lgb_oof,
    'true': y,
    'blend': w_cb*best_cb_oof + w_hgb*best_hgb_oof + w_lgb*best_lgb_oof
})
oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

metrics = {
    "task_id": "spaceship_titanic",
    "method": "mlevolve_hparam_sweep",
    "best_catboost": float(best_cb_acc),
    "best_hgb": float(best_hgb_acc),
    "best_lgb": float(best_lgb_acc),
    "best_ensemble": float(best_ensemble_acc),
    "weights": {"catboost": float(w_cb), "hgb": float(w_hgb), "lgb": float(w_lgb)},
    "catboost_params": best_cb_params,
    "hgb_params": best_hgb_params,
    "lgb_params": best_lgb_params,
    "features": X.shape[1],
    "test_pred_mean": float(blend_test.mean()),
}
with open(out_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)

print(f"\nOutput: {out_dir}")
print(f"Improvement over previous best (0.8163): {best_ensemble_acc - 0.8163:+.6f}")
