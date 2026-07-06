"""Digit Recognizer V2 - Improved MLP/CNN for MNIST-style digit classification."""
import json, os, numpy as np, pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'D:/桌面/codex/科研港科技/experiments/digit_recognizer/v2_{ts}'
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
train = pd.read_csv('D:/桌面/codex/科研港科技/tasks/digit_recognizer/data/train.csv')
test = pd.read_csv('D:/桌面/codex/科研港科技/tasks/digit_recognizer/data/test.csv')

X = train.drop('label', axis=1).values.astype(np.float32) / 255.0
y = train['label'].values
X_test = test.values.astype(np.float32) / 255.0

print(f"Train: {X.shape}, Test: {X_test.shape}, Classes: {len(np.unique(y))}")

# Reshape for CNN
X_img = X.reshape(-1, 28, 28, 1)
X_test_img = X_test.reshape(-1, 28, 28, 1)

# 5-fold CV
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Try multiple model types
results = {}

# 1. LightGBM (fast baseline)
print("\n=== LightGBM ===")
import lightgbm as lgb
lgb_oof = np.zeros(len(y))
lgb_test = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X.reshape(len(y), -1), y)):
    X_tr, X_val = X.reshape(len(y), -1)[tr_idx], X.reshape(len(y), -1)[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=127,
                               random_state=42, verbose=-1, n_jobs=-1)
    model.fit(X_tr, y_tr)
    lgb_oof[val_idx] = model.predict(X_val)
    lgb_test += model.predict(X_test.reshape(len(X_test), -1)) / n_folds
lgb_acc = accuracy_score(y, lgb_oof)
print(f"  OOF accuracy: {lgb_acc:.6f}")
results['lgb'] = {'oof': lgb_oof, 'test': lgb_test, 'acc': lgb_acc}

# 2. Simple Neural Network with sklearn MLP
print("\n=== MLP Classifier ===")
from sklearn.neural_network import MLPClassifier
mlp_oof = np.zeros(len(y))
mlp_test = np.zeros(len(X_test))
scaler = StandardScaler()
X_flat = scaler.fit_transform(X.reshape(len(y), -1))
X_test_flat = scaler.transform(X_test.reshape(len(X_test), -1))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_flat, y)):
    X_tr, X_val = X_flat[tr_idx], X_flat[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    model = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu',
                          alpha=0.0001, batch_size=128, learning_rate_init=0.001,
                          max_iter=50, early_stopping=True, random_state=42)
    model.fit(X_tr, y_tr)
    mlp_oof[val_idx] = model.predict(X_val)
    mlp_test += model.predict(X_test_flat) / n_folds
mlp_acc = accuracy_score(y, mlp_oof)
print(f"  OOF accuracy: {mlp_acc:.6f}")
results['mlp'] = {'oof': mlp_oof, 'test': mlp_test, 'acc': mlp_acc}

# 3. Keras/TF CNN (if available)
cnn_oof = None
try:
    print("\n=== CNN (TensorFlow/Keras) ===")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras

    cnn_oof = np.zeros(len(y))
    cnn_test = np.zeros((len(X_test), 10))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_img, y)):
        X_tr, X_val = X_img[tr_idx], X_img[val_idx]
        y_tr = keras.utils.to_categorical(y[tr_idx], 10)
        y_val_cat = keras.utils.to_categorical(y[val_idx], 10)

        model = keras.Sequential([
            keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val_cat),
                 epochs=15, batch_size=64, verbose=0)
        cnn_oof[val_idx] = model.predict(X_val, verbose=0).argmax(axis=1)
        cnn_test += model.predict(X_test_img, verbose=0) / n_folds

    cnn_acc = accuracy_score(y, cnn_oof.astype(int))
    print(f"  OOF accuracy: {cnn_acc:.6f}")
    results['cnn'] = {'oof': cnn_oof.astype(int), 'test': cnn_test.argmax(axis=1), 'acc': cnn_acc}
except Exception as e:
    print(f"  CNN FAILED: {e}")

# 4. XGBoost
print("\n=== XGBoost ===")
import xgboost as xgb
xgb_oof = np.zeros(len(y))
xgb_test = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X.reshape(len(y), -1), y)):
    X_tr, X_val = X.reshape(len(y), -1)[tr_idx], X.reshape(len(y), -1)[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                              random_state=42, verbosity=0, n_jobs=-1)
    model.fit(X_tr, y_tr)
    xgb_oof[val_idx] = model.predict(X_val)
    xgb_test += model.predict(X_test.reshape(len(X_test), -1)) / n_folds
xgb_acc = accuracy_score(y, xgb_oof)
print(f"  OOF accuracy: {xgb_acc:.6f}")
results['xgb'] = {'oof': xgb_oof, 'test': xgb_test, 'acc': xgb_acc}

# Select best model
best_model = max(results, key=lambda m: results[m]['acc'])
print(f"\n=== Best model: {best_model} (OOF={results[best_model]['acc']:.6f}) ===")

# Generate submission
sub = pd.read_csv('D:/桌面/codex/科研港科技/tasks/digit_recognizer/data/sample_submission.csv')
sub['Label'] = results[best_model]['test'].astype(int)
sub.to_csv(f'{output_dir}/submission.csv', index=False)

# Save metrics
metrics = {
    'schema': 'academic_research_os.digit_recognizer_metrics.v2',
    'run_id': f'v2_{ts}',
    'models': {m: float(v['acc']) for m, v in results.items()},
    'best_model': best_model,
    'best_oof_accuracy': float(results[best_model]['acc']),
    'estimated_public_score': float(results[best_model]['acc'] - 0.008),
}
with open(f'{output_dir}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nOutput: {output_dir}")
print(f"Submission: {len(sub)} rows")
for m, v in sorted(results.items(), key=lambda x: x[1]['acc'], reverse=True):
    print(f"  {m}: {v['acc']:.6f}")
