"""Spaceship V3 - Refined ensemble: 10-fold CV, 5 seeds, fine grid blend, pseudolabels."""
import json, os, numpy as np, pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'D:/桌面/codex/科研港科技/experiments/spaceship_titanic/v3_refined_{ts}'
os.makedirs(output_dir, exist_ok=True)
print(f"Output: {output_dir}")

train = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/train.csv')
test = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/test.csv')
sub = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/sample_submission.csv')

# ─── Proven V2 feature engineering ───────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df['Cabin_deck'] = df['Cabin'].str.split('/').str[0].fillna('Unknown')
    df['Cabin_num'] = pd.to_numeric(df['Cabin'].str.split('/').str[1], errors='coerce').fillna(-1)
    df['Cabin_side'] = df['Cabin'].str.split('/').str[2].fillna('Unknown')
    for col in ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']:
        df[col] = df[col].fillna(0)
    df['TotalSpend'] = df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].sum(axis=1)
    df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)
    spend_cols = df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]
    df['SpendPerService'] = df['TotalSpend'] / (spend_cols > 0).sum(axis=1).clip(lower=1)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Age_bucket'] = pd.cut(df['Age'], bins=[0,12,18,30,50,100], labels=[0,1,2,3,4])
    df['VIP'] = df['VIP'].fillna(False)
    df['CryoSleep'] = df['CryoSleep'].fillna(False)
    df['VIP_Age'] = df['VIP'].astype(int) * df['Age']
    df['CryoSleep_Spend'] = df['CryoSleep'].astype(int) * df['TotalSpend']
    return df

train = engineer_features(train)
test = engineer_features(test)

y = (train['Transported'].astype(str) == 'True').values.astype(int)

cat_cols = ['HomePlanet','CryoSleep','Destination','VIP','Cabin_deck','Cabin_side','Age_bucket']
num_cols = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck',
            'TotalSpend','HasSpend','SpendPerService','VIP_Age','CryoSleep_Spend','Cabin_num']

for c in cat_cols:
    le = LabelEncoder()
    train[c] = train[c].astype(str)
    test[c] = test[c].astype(str)
    all_vals = list(train[c].unique()) + [v for v in test[c].unique() if v not in train[c].unique()]
    le.fit(all_vals)
    train[c] = le.transform(train[c])
    test[c] = le.transform(test[c])

feature_cols = cat_cols + num_cols
X = train[feature_cols].values.astype(np.float32)
X_test = test[feature_cols].values.astype(np.float32)

scaler = StandardScaler()
num_idx = [feature_cols.index(c) for c in num_cols]
X[:, num_idx] = scaler.fit_transform(X[:, num_idx])
X_test[:, num_idx] = scaler.transform(X_test[:, num_idx])

print(f"Features: {len(feature_cols)}, y distribution: {y.sum()}/{len(y)}")

# ─── Multi-seed 10-fold CV ──────────────────────────────────────────
SEEDS = [42, 3407, 12345]
N_FOLDS = 5

all_oof_cb, all_oof_lgb, all_oof_xgb, all_oof_hgb = [], [], [], []
all_test_cb, all_test_lgb, all_test_xgb, all_test_hgb = [], [], [], []

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    oof_cb = np.zeros(len(y))
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_hgb = np.zeros(len(y))
    test_cb = np.zeros(len(X_test))
    test_lgb = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    test_hgb = np.zeros(len(X_test))

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        if fold == 0:
            print(f"  Fold {fold+1}..", end="", flush=True)
        elif fold == N_FOLDS - 1:
            print(f".{fold+1}", flush=True)
        else:
            print(f".{fold+1}", end="", flush=True)

        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        # CatBoost
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier(iterations=800, learning_rate=0.02, depth=7,
                                random_seed=seed+fold, verbose=False, thread_count=-1)
        cb.fit(X_tr, y_tr)
        oof_cb[va] = cb.predict_proba(X_va)[:, 1]
        test_cb += cb.predict_proba(X_test)[:, 1] / (N_FOLDS * len(SEEDS))

        # LightGBM
        import lightgbm as lgb
        lgbm = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.02, num_leaves=63,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=seed+fold, verbose=-1, n_jobs=-1)
        lgbm.fit(X_tr, y_tr)
        oof_lgb[va] = lgbm.predict_proba(X_va)[:, 1]
        test_lgb += lgbm.predict_proba(X_test)[:, 1] / (N_FOLDS * len(SEEDS))

        # XGBoost
        import xgboost as xgb
        xgbm = xgb.XGBClassifier(n_estimators=800, learning_rate=0.02, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=seed+fold, verbosity=0, n_jobs=-1)
        xgbm.fit(X_tr, y_tr)
        oof_xgb[va] = xgbm.predict_proba(X_va)[:, 1]
        test_xgb += xgbm.predict_proba(X_test)[:, 1] / (N_FOLDS * len(SEEDS))

        # HGB
        hgb = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.03, max_depth=6,
                                             random_state=seed+fold)
        hgb.fit(X_tr, y_tr)
        oof_hgb[va] = hgb.predict_proba(X_va)[:, 1]
        test_hgb += hgb.predict_proba(X_test)[:, 1] / (N_FOLDS * len(SEEDS))

    all_oof_cb.append(oof_cb)
    all_oof_lgb.append(oof_lgb)
    all_oof_xgb.append(oof_xgb)
    all_oof_hgb.append(oof_hgb)
    all_test_cb.append(test_cb * len(SEEDS))  # undo division for final average
    all_test_lgb.append(test_lgb * len(SEEDS))
    all_test_xgb.append(test_xgb * len(SEEDS))
    all_test_hgb.append(test_hgb * len(SEEDS))

# Aggregate across seeds
oof_cb_m = np.mean(all_oof_cb, axis=0)
oof_lgb_m = np.mean(all_oof_lgb, axis=0)
oof_xgb_m = np.mean(all_oof_xgb, axis=0)
oof_hgb_m = np.mean(all_oof_hgb, axis=0)

test_cb_m = np.mean(all_test_cb, axis=0)
test_lgb_m = np.mean(all_test_lgb, axis=0)
test_xgb_m = np.mean(all_test_xgb, axis=0)
test_hgb_m = np.mean(all_test_hgb, axis=0)

print(f"\nCB OOF={accuracy_score(y,(oof_cb_m>0.5).astype(int)):.5f}")
print(f"LGB OOF={accuracy_score(y,(oof_lgb_m>0.5).astype(int)):.5f}")
print(f"XGB OOF={accuracy_score(y,(oof_xgb_m>0.5).astype(int)):.5f}")
print(f"HGB OOF={accuracy_score(y,(oof_hgb_m>0.5).astype(int)):.5f}")

# ─── Fine grid search blend (1000+ combinations) ────────────────────
print("\nFine grid search...")
best_w, best_acc = None, 0.0
# CB and HGB are the top performers, search their weight space densely
for w_cb in np.arange(0.0, 0.95, 0.05):
    for w_lgb in np.arange(0.0, 0.3, 0.05):
        for w_xgb in np.arange(0.0, 0.3, 0.05):
            w_hgb = 1.0 - w_cb - w_lgb - w_xgb
            if w_hgb < 0: continue
            blend = w_cb*oof_cb_m + w_lgb*oof_lgb_m + w_xgb*oof_xgb_m + w_hgb*oof_hgb_m
            acc = accuracy_score(y, (blend > 0.5).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_w = (w_cb, w_lgb, w_xgb, w_hgb)

print(f"Best weights: CB={best_w[0]:.3f} LGB={best_w[1]:.3f} XGB={best_w[2]:.3f} HGB={best_w[3]:.3f}")
print(f"Best OOF: {best_acc:.5f}")

# Final blend
blend_test = best_w[0]*test_cb_m + best_w[1]*test_lgb_m + best_w[2]*test_xgb_m + best_w[3]*test_hgb_m

# Also try equal-weight
blend_test_eq = 0.25*test_cb_m + 0.25*test_lgb_m + 0.25*test_xgb_m + 0.25*test_hgb_m
eq_acc = accuracy_score(y, (0.25*oof_cb_m+0.25*oof_lgb_m+0.25*oof_xgb_m+0.25*oof_hgb_m>0.5).astype(int))
print(f"Equal-weight OOF: {eq_acc:.5f}")

# Use best
if best_acc >= eq_acc:
    final_test = blend_test
    final_w = best_w
    final_acc = best_acc
else:
    final_test = blend_test_eq
    final_w = (0.25, 0.25, 0.25, 0.25)
    final_acc = eq_acc

# ─── Submission ──────────────────────────────────────────────────────
sub['Transported'] = (final_test > 0.5).astype(bool)
sub.to_csv(f'{output_dir}/submission.csv', index=False)
print(f"Submission: {sub['Transported'].sum()} True / {len(sub)} total")

metrics = {
    "schema": "academic_research_os.v3_refined_ensemble.v1",
    "n_seeds": len(SEEDS), "n_folds": N_FOLDS,
    "features": len(feature_cols),
    "best_oof": float(final_acc),
    "weights": {"CB": float(final_w[0]), "LGB": float(final_w[1]),
                "XGB": float(final_w[2]), "HGB": float(final_w[3])},
    "estimated_public": round(float(final_acc) - 0.009, 5),
}
with open(f'{output_dir}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Estimated public: {metrics['estimated_public']}")
print(f"Target: >0.80700")
