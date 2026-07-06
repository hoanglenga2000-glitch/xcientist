"""Spaceship Titanic V3 - Feature-bagged diverse ensemble to bridge 0.80640→0.807 bronze gap."""
import json, os, random, numpy as np, pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'D:/桌面/codex/科研港科技/experiments/spaceship_titanic/v3_bagged_{ts}'
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
train = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/train.csv')
test = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/test.csv')
sub = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/sample_submission.csv')

def engineer_features(df):
    df = df.copy()
    df['Cabin_deck'] = df['Cabin'].str.split('/').str[0].fillna('Unknown')
    df['Cabin_num'] = pd.to_numeric(df['Cabin'].str.split('/').str[1], errors='coerce').fillna(-1)
    df['Cabin_side'] = df['Cabin'].str.split('/').str[2].fillna('Unknown')
    df['Group'] = df['PassengerId'].str[:4]
    for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df[col] = df[col].fillna(0)
    df['TotalSpend'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)
    df['SpendPerService'] = df['TotalSpend'] / (df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] > 0).sum(axis=1).clip(lower=1)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Age_bucket'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 100], labels=[0,1,2,3,4])
    df['VIP'] = df['VIP'].fillna(False)
    df['CryoSleep'] = df['CryoSleep'].fillna(False)
    df['VIP_Age'] = df['VIP'].astype(int) * df['Age']
    df['CryoSleep_Spend'] = df['CryoSleep'].astype(int) * df['TotalSpend']
    return df

train = engineer_features(train)
test = engineer_features(test)

target = 'Transported'
id_col = 'PassengerId'
y = (train[target].astype(str) == 'True').values.astype(int)

cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_deck', 'Cabin_side', 'Age_bucket']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
            'TotalSpend', 'HasSpend', 'SpendPerService', 'VIP_Age', 'CryoSleep_Spend', 'Cabin_num']

# Encode
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

print(f"Features: {len(feature_cols)}, X: {X.shape}, y: {y.shape}")

# Scale numeric
scaler = StandardScaler()
num_idx = [feature_cols.index(c) for c in num_cols]
X[:, num_idx] = scaler.fit_transform(X[:, num_idx])
X_test[:, num_idx] = scaler.transform(X_test[:, num_idx])

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# ─── Feature-bagged model definitions ────────────────────────────────
random.seed(42)
np.random.seed(42)
n_bags = 3
n_features = len(feature_cols)
bag_size = int(n_features * 0.8)

bagged_models = []
for bag_id in range(n_bags):
    bag_fidx = sorted(random.sample(range(n_features), bag_size))
    bagged_models.append({'type': 'lgb', 'name': f'lgb_bag{bag_id}', 'fidx': bag_fidx})
    bagged_models.append({'type': 'cb',  'name': f'cb_bag{bag_id}',  'fidx': bag_fidx})
    bagged_models.append({'type': 'xgb', 'name': f'xgb_bag{bag_id}', 'fidx': bag_fidx})
    bagged_models.append({'type': 'hgb', 'name': f'hgb_bag{bag_id}', 'fidx': bag_fidx})

print(f"Bagged models: {len(bagged_models)} ({len(bagged_models)//4} bags x 4 types)")

# ─── Train OOF + Test predictions ────────────────────────────────────
oof_preds = {m['name']: np.zeros(len(y)) for m in bagged_models}
test_preds = {m['name']: np.zeros(len(X_test)) for m in bagged_models}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}/{n_folds}...")
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    for m in bagged_models:
        fidx = m['fidx']
        X_tr_s, X_val_s = X_tr[:, fidx], X_val[:, fidx]
        X_ts_s = X_test[:, fidx]
        try:
            if m['type'] == 'lgb':
                import lightgbm as lgbm
                model = lgbm.LGBMClassifier(n_estimators=500, learning_rate=0.03,
                    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                    random_state=42+fold, verbose=-1, n_jobs=-1)
                model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)],
                         callbacks=[lgbm.early_stopping(50), lgbm.log_evaluation(0)])
                oof_preds[m['name']][val_idx] = model.predict_proba(X_val_s)[:,1]
                test_preds[m['name']] += model.predict_proba(X_ts_s)[:,1] / n_folds

            elif m['type'] == 'xgb':
                import xgboost as xgbm
                model = xgbm.XGBClassifier(n_estimators=500, learning_rate=0.03,
                    max_depth=6, subsample=0.8, colsample_bytree=0.8,
                    random_state=42+fold, verbosity=0, n_jobs=-1)
                model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
                oof_preds[m['name']][val_idx] = model.predict_proba(X_val_s)[:,1]
                test_preds[m['name']] += model.predict_proba(X_ts_s)[:,1] / n_folds

            elif m['type'] == 'cb':
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(iterations=500, learning_rate=0.03,
                    depth=6, random_seed=42+fold, verbose=False, thread_count=-1)
                model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)])
                oof_preds[m['name']][val_idx] = model.predict_proba(X_val_s)[:,1]
                test_preds[m['name']] += model.predict_proba(X_ts_s)[:,1] / n_folds

            elif m['type'] == 'hgb':
                model = HistGradientBoostingClassifier(max_iter=500,
                    learning_rate=0.05, max_depth=6, random_state=42+fold,
                    early_stopping=True, validation_fraction=0.1)
                model.fit(X_tr_s, y_tr)
                oof_preds[m['name']][val_idx] = model.predict_proba(X_val_s)[:,1]
                test_preds[m['name']] += model.predict_proba(X_ts_s)[:,1] / n_folds
        except Exception as e:
            print(f"  {m['name']}: FAILED - {e}")
            oof_preds[m['name']][val_idx] = 0.5
            test_preds[m['name']] += np.full(len(X_test), 0.5) / n_folds

# ─── Model-level OOF scores ──────────────────────────────────────────
print("\n=== Per-model OOF scores ===")
model_scores = {}
for m in bagged_models:
    acc = accuracy_score(y, (oof_preds[m['name']] > 0.5).astype(int))
    model_scores[m['name']] = acc
    print(f"  {m['name']}: OOF={acc:.5f}")

# ─── Equal-weight average ensemble (avoids blend overfitting) ────────
active = [n for n in oof_preds if oof_preds[n].any()]
blend = np.zeros(len(y))
blend_test = np.zeros(len(X_test))
for n in active:
    blend += oof_preds[n] / len(active)
    blend_test += test_preds[n] / len(active)

blend_pred = (blend > 0.5).astype(int)
best_acc = accuracy_score(y, blend_pred)
print(f"\nEqual-weight blend ({len(active)} models): OOF={best_acc:.5f}")

# ─── Also try grid-search blend for comparison ───────────────────────
print("\nGrid search blend weights...")
best_w, best_blend_acc = None, 0.0
for w1 in np.arange(0.0, 0.5, 0.1):
    for w2 in np.arange(0.0, 0.4, 0.1):
        w3 = 1.0 - w1 - w2
        if w3 < 0: continue
        # Group by model type
        lgb_oof = np.mean([oof_preds[n] for n in oof_preds if n.startswith('lgb')], axis=0)
        cb_oof  = np.mean([oof_preds[n] for n in oof_preds if n.startswith('cb')], axis=0)
        xgb_hgb_oof = np.mean([oof_preds[n] for n in oof_preds if n.startswith('xgb') or n.startswith('hgb')], axis=0)
        b = w1*lgb_oof + w2*cb_oof + w3*xgb_hgb_oof
        acc = accuracy_score(y, (b > 0.5).astype(int))
        if acc > best_blend_acc:
            best_blend_acc = acc
            best_w = (w1, w2, w3)

if best_w:
    lgb_test = np.mean([test_preds[n] for n in test_preds if n.startswith('lgb')], axis=0)
    cb_test  = np.mean([test_preds[n] for n in test_preds if n.startswith('cb')], axis=0)
    xgb_hgb_test = np.mean([test_preds[n] for n in test_preds if n.startswith('xgb') or n.startswith('hgb')], axis=0)
    blend_test_grid = best_w[0]*lgb_test + best_w[1]*cb_test + best_w[2]*xgb_hgb_test
    print(f"Grid best: w={best_w}, OOF={best_blend_acc:.5f}")
    # Use whichever is better between equal-weight and grid
    if best_blend_acc > best_acc:
        blend_test = blend_test_grid
        best_acc = best_blend_acc
        best_weights = {"lgb_group": best_w[0], "cb_group": best_w[1], "xgb_hgb_group": best_w[2]}
    else:
        best_weights = {n: 1.0/len(active) for n in active}
else:
    best_weights = {n: 1.0/len(active) for n in active}

# ─── Submission ──────────────────────────────────────────────────────
sub['Transported'] = (blend_test > 0.5).astype(bool)
sub.to_csv(f'{output_dir}/submission.csv', index=False)
print(f"\nSubmission: {len(sub)} rows, {sub['Transported'].sum()} True")

# ─── Metrics ─────────────────────────────────────────────────────────
metrics = {
    "schema": "academic_research_os.v3_bagged_ensemble.v1",
    "timestamp": ts,
    "n_features": len(feature_cols),
    "n_bagged_models": len(bagged_models),
    "n_folds": n_folds,
    "equal_weight_oof": float(best_acc),
    "best_model_oof": max(model_scores.values()),
    "model_scores": model_scores,
    "best_weights": best_weights,
    "strategy": "feature_bagged_diverse_ensemble_12_models",
}
with open(f'{output_dir}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nBest OOF: {best_acc:.5f} | Best single model: {max(model_scores.values()):.5f}")
print(f"Estimated public: ~{best_acc - 0.008:.5f} (target > 0.807)")
print(f"Output: {output_dir}")
