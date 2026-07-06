"""Spaceship Titanic V2 - Improved ensemble with CatBoost + feature engineering."""
import json, os, csv, numpy as np, pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'D:/桌面/codex/科研港科技/experiments/spaceship_titanic/v2_{ts}'
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
train = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/train.csv')
test = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/test.csv')
sub = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/sample_submission.csv')

print(f"Train: {train.shape}, Test: {test.shape}")

# Feature engineering
def engineer_features(df):
    df = df.copy()

    # Cabin features: deck/num/side
    df['Cabin_deck'] = df['Cabin'].str.split('/').str[0].fillna('Unknown')
    df['Cabin_num'] = pd.to_numeric(df['Cabin'].str.split('/').str[1], errors='coerce').fillna(-1)
    df['Cabin_side'] = df['Cabin'].str.split('/').str[2].fillna('Unknown')

    # Group ID (first 4 chars of PassengerId)
    df['Group'] = df['PassengerId'].str[:4]

    # Group features
    for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df[col] = df[col].fillna(0)

    df['TotalSpend'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)
    df['SpendPerService'] = df['TotalSpend'] / (df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] > 0).sum(axis=1).clip(lower=1)

    # Age features
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Age_bucket'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 100], labels=[0,1,2,3,4])

    # Fill NaN for boolean columns
    df['VIP'] = df['VIP'].fillna(False)
    df['CryoSleep'] = df['CryoSleep'].fillna(False)

    # VIP + Age interaction
    df['VIP_Age'] = df['VIP'].astype(int) * df['Age']

    # CryoSleep + spending interaction
    df['CryoSleep_Spend'] = df['CryoSleep'].astype(int) * df['TotalSpend']

    return df

train = engineer_features(train)
test = engineer_features(test)

# Identify features
target = 'Transported'
id_col = 'PassengerId'
drop_cols = [target, id_col, 'Name', 'Cabin']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_deck', 'Cabin_side']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
            'Cabin_num', 'TotalSpend', 'HasSpend', 'SpendPerService', 'VIP_Age', 'CryoSleep_Spend']

# Encode categoricals
for col in cat_cols:
    if col in train.columns:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]])
        le.fit(combined)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

# Prepare feature matrix
feature_cols = [c for c in cat_cols + num_cols if c in train.columns]
X = train[feature_cols].fillna(-1).values
X_test = test[feature_cols].fillna(-1).values
y = (train[target] == True).astype(int).values

print(f"Features: {len(feature_cols)}, Train: {X.shape}, Test: {X_test.shape}")

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# 5-fold CV with per-model OOF
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

models_config = {
    'lgb': {'type': 'lgb'},
    'xgb': {'type': 'xgb'},
    'cat': {'type': 'cat'},
    'hgb': {'type': 'hgb'},
}

oof_preds = {m: np.zeros(len(y)) for m in models_config}
test_preds = {m: np.zeros(len(X_test)) for m in models_config}
individual_scores = {}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}/{n_folds}...")
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    for mname, mcfg in models_config.items():
        try:
            if mname == 'lgb':
                import lightgbm as lgb
                model = lgb.LGBMClassifier(
                    n_estimators=500, learning_rate=0.03, num_leaves=63,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbose=-1, n_jobs=-1
                )
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                oof_preds[mname][val_idx] = model.predict_proba(X_val)[:, 1]
                test_preds[mname] += model.predict_proba(X_test)[:, 1] / n_folds

            elif mname == 'xgb':
                import xgboost as xgb
                model = xgb.XGBClassifier(
                    n_estimators=500, learning_rate=0.03, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbosity=0, n_jobs=-1
                )
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                oof_preds[mname][val_idx] = model.predict_proba(X_val)[:, 1]
                test_preds[mname] += model.predict_proba(X_test)[:, 1] / n_folds

            elif mname == 'cat':
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(
                    iterations=500, learning_rate=0.03, depth=6,
                    random_seed=42, verbose=False, thread_count=-1
                )
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
                oof_preds[mname][val_idx] = model.predict_proba(X_val)[:, 1]
                test_preds[mname] += model.predict_proba(X_test)[:, 1] / n_folds

            elif mname == 'hgb':
                from sklearn.ensemble import HistGradientBoostingClassifier
                model = HistGradientBoostingClassifier(
                    max_iter=500, learning_rate=0.05, max_depth=None,
                    random_state=42, early_stopping=True, validation_fraction=0.1
                )
                model.fit(X_tr, y_tr)
                oof_preds[mname][val_idx] = model.predict_proba(X_val)[:, 1]
                test_preds[mname] += model.predict_proba(X_test)[:, 1] / n_folds

        except Exception as e:
            print(f"  {mname}: FAILED - {e}")
            oof_preds[mname][val_idx] = 0.5
            test_preds[mname] += np.full(len(X_test), 0.5) / n_folds

# Score each model
print("\n=== Individual Model OOF Scores ===")
for mname in models_config:
    if oof_preds[mname].sum() > 0:
        acc = accuracy_score(y, (oof_preds[mname] > 0.5).astype(int))
        individual_scores[mname] = acc
        print(f"  {mname}: OOF accuracy = {acc:.6f}")

# Optimize blend weights
print("\n=== Blend Optimization ===")
available = [m for m in models_config if m in individual_scores]
n_models = len(available)

best_acc = 0
best_weights = None

# Grid search
for w_lgb in np.arange(0, 1.01, 0.1) if 'lgb' in available else [0]:
    for w_xgb in np.arange(0, 1.01 - w_lgb, 0.1) if 'xgb' in available else [0]:
        for w_cat in np.arange(0, 1.01 - w_lgb - w_xgb, 0.1) if 'cat' in available else [0]:
            w_hgb = 1.0 - w_lgb - w_xgb - w_cat
            if w_hgb < 0: continue
            if 'hgb' not in available and w_hgb > 0: continue

            blend = np.zeros(len(y))
            for m, w in zip(available, [w_lgb, w_xgb, w_cat, w_hgb]):
                blend += w * oof_preds[m]
            acc = accuracy_score(y, (blend > 0.5).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_weights = dict(zip(available, [w_lgb, w_xgb, w_cat, w_hgb]))

print(f"Best OOF blend accuracy: {best_acc:.6f}")
print(f"Weights: {best_weights}")

# Generate test blend
blend_test = np.zeros(len(X_test))
for m, w in best_weights.items():
    if w > 0:
        blend_test += w * test_preds[m]

# Create submission
sub['Transported'] = (blend_test > 0.5).astype(bool)
sub.to_csv(f'{output_dir}/submission.csv', index=False)
print(f"\nSubmission: {len(sub)} rows, {sub['Transported'].sum()} True ({sub['Transported'].mean()*100:.1f}%)")

# Save metrics
metrics = {
    'schema': 'academic_research_os.ensemble_metrics.v3',
    'task_id': 'spaceship_titanic',
    'run_id': f'v2_{ts}',
    'runner': 'v2_ensemble_lgb_xgb_cat_hgb',
    'metric': 'accuracy',
    'metric_direction': 'maximize',
    'n_folds': n_folds,
    'train_rows': len(train),
    'test_rows': len(test),
    'features': len(feature_cols),
    'individual_oof_accuracy': {m: float(v) for m, v in individual_scores.items()},
    'best_blend_accuracy': float(best_acc),
    'best_weights': {m: float(w) for m, w in best_weights.items()},
    'test_prediction_mean': float(blend_test.mean()),
    'cv_public_gap_estimate': 0.004,
    'estimated_public_score': float(best_acc - 0.004),
}
with open(f'{output_dir}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save OOF predictions
oof_df = pd.DataFrame({
    'id': range(len(y)),
    'true': y,
    'blend': sum(best_weights[m] * oof_preds[m] for m in available),
    **{f'prob_{m}': oof_preds[m] for m in available}
})
oof_df.to_csv(f'{output_dir}/oof_predictions.csv', index=False)

print(f"\n{'='*50}")
print(f"RESULTS: v2_{ts}")
print(f"Best OOF: {best_acc:.6f}")
print(f"Est. Public: {best_acc - 0.004:.6f}")
print(f"Previous Public: 0.80547")
improvement = (best_acc - 0.004) - 0.80547
print(f"Expected improvement: {improvement:+.6f}")
print(f"Output: {output_dir}")
