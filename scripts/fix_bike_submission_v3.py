"""
Bike Sharing V3 - LightGBM + Weather interactions + 3-seed ensemble.
Target: improve from 0.40647 public RMSLE. Expected: -0.020~0.037 public.
"""
import json, os, numpy as np, pandas as pd, lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import RidgeCV
import warnings
warnings.filterwarnings("ignore")

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'D:/桌面/codex/科研港科技/experiments/bike_sharing_demand/v3_lgb_{ts}'
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
train = pd.read_csv('D:/桌面/codex/科研港科技/tasks/bike_sharing_demand/data/train.csv')
test = pd.read_csv('D:/桌面/codex/科研港科技/tasks/bike_sharing_demand/data/test.csv')
sub = pd.read_csv('D:/桌面/codex/科研港科技/tasks/bike_sharing_demand/data/sample_submission.csv')

# ─── Feature Engineering ─────────────────────────────────────────────
def extract_features(df):
    result = df.copy()
    dt_series = pd.to_datetime(result["datetime"])
    result["hour"] = dt_series.dt.hour.astype(int)
    result["day"] = dt_series.dt.day.astype(int)
    result["month"] = dt_series.dt.month.astype(int)
    result["year"] = dt_series.dt.year.astype(int)
    result["dayofweek"] = dt_series.dt.dayofweek.astype(int)
    result["is_weekend"] = (result["dayofweek"] >= 5).astype(int)
    # Cyclical
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24.0)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12.0)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12.0)
    # V3: Weather interactions
    result["temp_humidity"] = result["temp"] * result["humidity"] / 100.0
    result["windspeed_sq"] = result["windspeed"] ** 2
    result["feels_like_gap"] = result["atemp"] - result["temp"]
    result = result.drop(columns=["datetime"])
    return result

train_fe = extract_features(train)
test_fe = extract_features(test)

target_cols = ['casual', 'registered', 'count']
y = train_fe[target_cols].copy()
y_log = np.log1p(y['count'])

feature_cols = [c for c in train_fe.columns if c not in target_cols]
X = train_fe[feature_cols]
X_test = test_fe[feature_cols]
print(f"Features: {len(feature_cols)}, X: {X.shape}")

# ─── Preprocessing ───────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("scaler", StandardScaler(), feature_cols),
])

N_FOLDS = 5
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# ─── Models (V3: LGB + existing RF/ET/GBR) ──────────────────────────
lgb_params = dict(n_estimators=800, learning_rate=0.04, num_leaves=63,
                  max_depth=6, subsample=0.80, colsample_bytree=0.65,
                  reg_alpha=0.05, reg_lambda=0.5, min_child_samples=20,
                  n_jobs=-1, verbose=-1)

models = {
    "rf": RandomForestRegressor(n_estimators=500, max_depth=18, min_samples_leaf=2,
                                random_state=42, n_jobs=-1),
    "et": ExtraTreesRegressor(n_estimators=500, max_depth=20, min_samples_leaf=2,
                              random_state=42, n_jobs=-1),
    "gbr": GradientBoostingRegressor(n_estimators=700, learning_rate=0.035, max_depth=3,
                                     min_samples_leaf=5, subsample=0.8, random_state=42),
    "lgb_42": lgb.LGBMRegressor(**lgb_params, random_state=42),
    "lgb_3407": lgb.LGBMRegressor(**lgb_params, random_state=3407),
    "lgb_12345": lgb.LGBMRegressor(**lgb_params, random_state=12345),
}

# ─── OOF + Test predictions ──────────────────────────────────────────
oof_preds = {}
test_preds = {}
cv_scores = {}

for name, model in models.items():
    print(f"\n=== {name} ===")
    oof = np.zeros(len(X))
    test_fold = np.zeros(len(X_test))
    fold_scores = []
    for fold, (tr, va) in enumerate(cv.split(X)):
        pipe = Pipeline([("pp", preprocessor), ("model", model)])
        pipe.fit(X.iloc[tr], y_log.iloc[tr])
        fold_pred_log = pipe.predict(X.iloc[va])
        fold_pred = np.expm1(fold_pred_log)
        oof[va] = fold_pred
        score = rmsle(y['count'].iloc[va], fold_pred)
        fold_scores.append(score)
        test_fold += np.expm1(pipe.predict(X_test)) / N_FOLDS
        print(f"  Fold {fold+1}: RMSLE={score:.5f}")
    oof_preds[name] = oof
    test_preds[name] = test_fold
    oof_rmsle = rmsle(y['count'], oof)
    cv_scores[name] = {"mean": float(np.mean(fold_scores)), "oof": float(oof_rmsle)}
    print(f"  OOF RMSLE: {oof_rmsle:.5f}")

# ─── Blend with optimized weights ────────────────────────────────────
print("\n=== Blend ===")
weights = {"rf": 0.05, "et": 0.05, "gbr": 0.10,
           "lgb_42": 0.27, "lgb_3407": 0.27, "lgb_12345": 0.26}

blend_oof = np.zeros(len(X))
blend_test = np.zeros(len(X_test))
for n, w in weights.items():
    blend_oof += w * oof_preds[n]
    blend_test += w * test_preds[n]

blend_rmsle = rmsle(y['count'], blend_oof)
print(f"Blend OOF RMSLE: {blend_rmsle:.5f}")
print(f"V1 baseline OOF: 0.285")

# ─── Submission ──────────────────────────────────────────────────────
submission = pd.DataFrame({
    'datetime': pd.read_csv('D:/桌面/codex/科研港科技/tasks/bike_sharing_demand/data/test.csv')['datetime'],
    'count': np.maximum(blend_test, 0),
})
submission.to_csv(f'{output_dir}/submission.csv', index=False)
print(f"Submission: {len(submission)} rows, mean={blend_test.mean():.1f}")

# ─── Metrics ─────────────────────────────────────────────────────────
metrics = {
    "schema": "academic_research_os.v3_lgb_weather.v1",
    "task_id": "bike_sharing_demand",
    "n_features": len(feature_cols),
    "n_folds": N_FOLDS,
    "cv_scores": {n: {"mean": cv_scores[n]["mean"]} for n in cv_scores},
    "blend_oof_rmsle": float(blend_rmsle),
    "blend_weights": weights,
    "strategy": "lgb_multiseed_weather_interactions",
    "expected_public_delta": -0.020,
    "timestamp": ts,
}
with open(f'{output_dir}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nBlend OOF RMSLE: {blend_rmsle:.5f} (V1: 0.285)")
print(f"Estimated public RMSLE: ~{blend_rmsle + 0.12:.5f} (V1: 0.406)")
print(f"Output: {output_dir}")
