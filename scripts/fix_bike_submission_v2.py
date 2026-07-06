"""Bike Sharing Demand V2 - TimeSeriesSplit CV, weather interactions, multi-seed LightGBM."""
import json, os, numpy as np, pandas as pd
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_log_error

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'D:/桌面/codex/科研港科技/experiments/bike_sharing_demand/v2_timeseries_{ts}'
os.makedirs(output_dir, exist_ok=True)

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

print("Loading data...")
train = pd.read_csv('D:/桌面/codex/科研港科技/tasks/bike_sharing_demand/data/train.csv')
test = pd.read_csv('D:/桌面/codex/科研港科技/tasks/bike_sharing_demand/data/test.csv')

# Sort by datetime for TimeSeriesSplit
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])
train = train.sort_values('datetime').reset_index(drop=True)

# ─── V2 Feature Engineering ──────────────────────────────────────────
def extract_features(df):
    result = df.copy()
    dt = result['datetime']

    result['hour'] = dt.dt.hour.astype(int)
    result['day'] = dt.dt.day.astype(int)
    result['month'] = dt.dt.month.astype(int)
    result['year'] = dt.dt.year.astype(int)
    result['dayofweek'] = dt.dt.dayofweek.astype(int)
    result['is_weekend'] = (result['dayofweek'] >= 5).astype(int)

    # Cyclical encoding
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24.0)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24.0)
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12.0)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12.0)
    result['dayofweek_sin'] = np.sin(2 * np.pi * result['dayofweek'] / 7.0)
    result['dayofweek_cos'] = np.cos(2 * np.pi * result['dayofweek'] / 7.0)

    # Weather interactions
    result['temp_humidity'] = result['temp'] * result['humidity'] / 100.0
    result['windspeed_sq'] = result['windspeed'] ** 2
    result['feels_like_gap'] = result['atemp'] - result['temp']
    result['bad_weather'] = (result['weather'] >= 3).astype(int)

    # Peak/rush hour
    result['is_rush_hour'] = result['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    result['is_working_hour'] = ((result['hour'] >= 9) & (result['hour'] <= 17)).astype(int)

    # Drop datetime
    result = result.drop(columns=['datetime'])
    return result

train_fe = extract_features(train)
test_fe = extract_features(test)

target_cols = ['casual', 'registered', 'count']
y = train_fe[target_cols].copy()
y_log = np.log1p(y['count'])

feature_cols = [c for c in train_fe.columns if c not in target_cols]
X = train_fe[feature_cols]
X_test = test_fe[feature_cols]

print(f"Features: {len(feature_cols)}, X: {X.shape}, y: {y.shape}")

# ─── TimeSeriesSplit CV (NO shuffle — prevents future leakage) ───────
n_folds = 5
tscv = TimeSeriesSplit(n_splits=n_folds)

# ─── Models ───────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: LightGBM not installed")

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

models = {
    "rf": RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=2,
                                random_state=42, n_jobs=-1),
    "et": ExtraTreesRegressor(n_estimators=500, max_depth=14, min_samples_leaf=2,
                              random_state=42, n_jobs=-1),
    "gb": GradientBoostingRegressor(n_estimators=500, learning_rate=0.03, max_depth=3,
                                    min_samples_leaf=2, subsample=0.8, random_state=42),
}

if HAS_LGB:
    models["lgb_1"] = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03,
        num_leaves=127, max_depth=8, subsample=0.78, colsample_bytree=0.65,
        reg_alpha=0.1, reg_lambda=0.5, random_state=42, n_jobs=-1, verbose=-1)
    models["lgb_2"] = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03,
        num_leaves=127, max_depth=8, subsample=0.78, colsample_bytree=0.65,
        reg_alpha=0.1, reg_lambda=0.5, random_state=3407, n_jobs=-1, verbose=-1)
    models["lgb_3"] = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03,
        num_leaves=127, max_depth=8, subsample=0.78, colsample_bytree=0.65,
        reg_alpha=0.1, reg_lambda=0.5, random_state=12345, n_jobs=-1, verbose=-1)

# ─── Preprocessing ────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("scaler", StandardScaler(), feature_cols),
])

# ─── OOF + Test predictions ──────────────────────────────────────────
oof_preds = {}
test_preds = {}
cv_scores = {}

for name, model in models.items():
    print(f"\n=== {name} ===")
    oof = np.zeros(len(X))
    test_fold = np.zeros(len(X_test))
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        pipe = Pipeline([("pp", preprocessor), ("model", model)])
        # Fit in log space
        pipe.fit(X.iloc[tr_idx], y_log.iloc[tr_idx])
        fold_pred_log = pipe.predict(X.iloc[val_idx])
        fold_pred = np.expm1(fold_pred_log)
        oof[val_idx] = fold_pred
        score = rmsle(y['count'].iloc[val_idx], fold_pred)
        fold_scores.append(score)
        test_fold += np.expm1(pipe.predict(X_test)) / n_folds
        print(f"  Fold {fold+1}: RMSLE={score:.5f}")

    oof_preds[name] = oof
    test_preds[name] = test_fold
    oof_rmsle = rmsle(y['count'], oof)
    cv_scores[name] = {"fold_scores": fold_scores, "mean": float(np.mean(fold_scores)),
                       "oof_rmsle": float(oof_rmsle)}
    print(f"  OOF RMSLE: {oof_rmsle:.5f}")

# ─── Blend ───────────────────────────────────────────────────────────
print("\n=== Blend ===")
all_oof = np.column_stack([oof_preds[n] for n in oof_preds])
all_test = np.column_stack([test_preds[n] for n in test_preds])

# Grid search for blend weights
best_w, best_rmsle = None, float('inf')
for w0 in np.arange(0.05, 0.3, 0.05):
    for w1 in np.arange(0.05, 0.25, 0.05):
        w2 = 1.0 - w0 - w1
        if w2 < 0.05: continue
        model_names = list(oof_preds.keys())
        weights = np.zeros(len(model_names))
        weights[0] = w0
        weights[1] = w1
        for i in range(2, len(weights)):
            weights[i] = (w2) / (len(weights) - 2)
        weights = weights / weights.sum()
        blend_oof = np.dot(all_oof, weights)
        s = rmsle(y['count'], blend_oof)
        if s < best_rmsle:
            best_rmsle = s
            best_w = {n: float(weights[i]) for i, n in enumerate(model_names)}

print(f"Best blend weights: {best_w}")
print(f"Blend OOF RMSLE: {best_rmsle:.5f}")

# Final test blend
blend_test = np.zeros(len(X_test))
for i, n in enumerate(oof_preds.keys()):
    blend_test += best_w[n] * test_preds[n]

# ─── Submission ──────────────────────────────────────────────────────
final_pred = np.maximum(blend_test, 0)  # clip negative
submission = pd.DataFrame({
    'datetime': pd.read_csv('D:/桌面/codex/科研港科技/tasks/bike_sharing_demand/data/test.csv')['datetime'],
    'count': final_pred,
})
submission.to_csv(f'{output_dir}/submission.csv', index=False)
print(f"Submission: {len(submission)} rows, mean={final_pred.mean():.1f}")

# ─── Metrics ─────────────────────────────────────────────────────────
metrics = {
    "schema": "academic_research_os.v2_timeseries_improved.v1",
    "timestamp": ts,
    "n_features": len(feature_cols),
    "n_folds": n_folds,
    "cv_method": "TimeSeriesSplit_noshuffle",
    "cv_scores": {n: {"mean": cv_scores[n]["mean"]} for n in cv_scores},
    "blend_oof_rmsle": float(best_rmsle),
    "blend_weights": best_w,
    "strategy": "timeseries_split_weather_interactions_multiseed_lgb",
}
with open(f'{output_dir}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nFinal blend OOF: {best_rmsle:.5f}")
print(f"V1 baseline OOF: ~0.285")
print(f"Output: {output_dir}")
