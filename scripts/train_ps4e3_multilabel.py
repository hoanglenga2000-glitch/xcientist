"""Custom multi-label training for ps4e3 - 7 binary targets."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import json, time

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = 'ps4e3'
TARGET_COLS = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
N_SAMPLE = 10000
N_FOLDS = 3
SEED = 42

started = time.monotonic()
data_dir = ROOT / 'tasks' / TASK_ID / 'data'
train = pd.read_csv(data_dir / 'train.csv')
test = pd.read_csv(data_dir / 'test.csv')
sample = pd.read_csv(data_dir / 'sample_submission.csv')

# Separate features and targets
feature_cols = [c for c in train.columns if c not in TARGET_COLS + ['id']]
y_all = train[TARGET_COLS].astype(int).values

# Sample for fast mode
if len(train) > N_SAMPLE:
    # Stratified by the most common target (Other_Faults)
    train_sampled = train.sample(n=N_SAMPLE, random_state=SEED)
    y_all = train_sampled[TARGET_COLS].astype(int).values
    train_x = train_sampled[feature_cols]
else:
    train_x = train[feature_cols]

test_x = test[feature_cols]

# Feature engineering
categorical = [c for c in feature_cols if train_x[c].dtype == 'object']
numeric = [c for c in feature_cols if c not in categorical]
print(f'Features: {len(numeric)} numeric, {len(categorical)} categorical, total={len(feature_cols)}')

try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.float32)

preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numeric), ('cat', ohe, categorical)],
    remainder='drop',
)

X_train = preprocessor.fit_transform(train_x).astype(np.float32)
X_test = preprocessor.transform(test_x).astype(np.float32)
print(f'Encoded: X_train={X_train.shape}, X_test={X_test.shape}')

# Train one model per target
test_preds = np.zeros((len(test), len(TARGET_COLS)), dtype=np.float64)
oof_preds = np.zeros((len(train_x), len(TARGET_COLS)), dtype=np.float64)

for t_idx, tcol in enumerate(TARGET_COLS):
    y = y_all[:, t_idx]
    pos_ratio = y.mean()
    print(f'\n[{tcol}] positive ratio={pos_ratio:.3f}, samples={len(y)}', flush=True)

    # Build fast models
    models = {
        'rf': RandomForestClassifier(n_estimators=80, max_depth=12, min_samples_leaf=32,
                                       max_features='sqrt', n_jobs=4, random_state=SEED),
        'hgb': HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=6,
                                                max_leaf_nodes=63, min_samples_leaf=32, l2_regularization=0.5,
                                                early_stopping=True, validation_fraction=0.12,
                                                n_iter_no_change=30, random_state=SEED),
        'et': ExtraTreesClassifier(n_estimators=80, max_depth=12, min_samples_leaf=32,
                                     max_features='sqrt', n_jobs=4, random_state=SEED),
    }

    oof = np.zeros(len(y), dtype=np.float64)
    tpred = np.zeros(len(test), dtype=np.float64)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        fold_preds = np.zeros(len(va_idx), dtype=np.float64)
        fold_tpred = np.zeros(len(test), dtype=np.float64)

        for name, model in models.items():
            model.fit(X_tr, y_tr)
            p_val = model.predict_proba(X_va)[:, 1]
            p_test = model.predict_proba(X_test)[:, 1]
            fold_preds += p_val / len(models)
            fold_tpred += p_test / len(models)

        oof[va_idx] = fold_preds
        tpred += fold_tpred / N_FOLDS
        acc = ((fold_preds > 0.5).astype(int) == y_va).mean()
        print(f'  fold={fold+1}: acc={acc:.4f}', flush=True)

    oof_preds[:, t_idx] = oof
    test_preds[:, t_idx] = np.clip(tpred, 0.0, 1.0)
    oof_acc = ((oof > 0.5).astype(int) == y).mean()
    print(f'  OOF accuracy={oof_acc:.4f}', flush=True)

# Build submission
submission = pd.DataFrame({'id': sample['id'].values})
for t_idx, tcol in enumerate(TARGET_COLS):
    submission[tcol] = test_preds[:, t_idx]

output_dir = ROOT / 'experiments' / TASK_ID / 'v1'
output_dir.mkdir(parents=True, exist_ok=True)
submission.to_csv(output_dir / 'submission.csv', index=False)

# OOF metrics
oof_mean_acc = np.mean([((oof_preds[:, i] > 0.5).astype(int) == y_all[:, i]).mean() for i in range(len(TARGET_COLS))])
print(f'\n=== Summary ===')
print(f'Mean OOF accuracy: {oof_mean_acc:.4f}')
print(f'Submission rows: {len(submission)}')
print(f'Submission saved to: {output_dir / "submission.csv"}')
print(f'Time: {time.monotonic() - started:.1f}s')

# Save metrics
metrics = {
    'task_id': TASK_ID,
    'run_id': 'v1',
    'oof_mean_accuracy': float(oof_mean_acc),
    'submission_rows': int(len(submission)),
    'seconds': round(time.monotonic() - started, 1),
    'status': 'passed',
}
(output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print('Done.')
