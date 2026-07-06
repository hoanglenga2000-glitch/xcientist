"""
optimize_titanic.py — Fix Kaggle Titanic CV-public gap.

Problem diagnosed:
  - Previous run used 1731 features on 891 rows (extreme overfitting)
  - 3-fold CV reported 0.82 but Kaggle public score was only 0.744
  - RF and ET models collapsed to 0.6 accuracy (balanced_accuracy=0.5 = predicting all zeros)
  - HGB alone carried the ensemble but overfit due to massive one-hot expansion

Fix:
  - Only 7 raw features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
  - Simple median imputation for Age, mode for Embarked
  - LogisticRegression (C=0.1) + RandomForest (max_depth=5, min_samples_leaf=5)
  - 10-fold CV for more stable estimate on small dataset
  - Simple probability averaging ensemble
  - Submission: soft probabilities rounded to 0/1

Usage:
  python D:/桌面/codex/科研港科技/scripts/optimize_titanic.py
"""

import os, sys, json, warnings, hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────
TRAIN_PATH = "D:/桌面/codex/科研港科技/tasks/titanic/data/train.csv"
TEST_PATH  = "D:/桌面/codex/科研港科技/tasks/titanic/data/test.csv"
NOW_TS     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
RUN_HASH   = hashlib.sha256(NOW_TS.encode()).hexdigest()[:8]
OUT_DIR    = f"D:/桌面/codex/科研港科技/experiments/titanic/optimized_{NOW_TS}_{RUN_HASH}"

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Output directory: {OUT_DIR}")

# ─── Load data ───────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Survived distribution: {train['Survived'].value_counts().to_dict()}")

# ─── Features (only 7 raw columns — NO complex engineering) ──────────
NUMERIC_FEATURES   = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = train[ALL_FEATURES].copy()
y = train["Survived"].values
X_test = test[ALL_FEATURES].copy()

# ─── Preprocessing pipeline ──────────────────────────────────────────
# Numeric: median imputation + standard scaling
# Categorical: mode imputation + one-hot encoding (drop first to avoid collinearity)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

# ─── Models (simple, regularized — avoid overfitting) ────────────────
models = {
    "lr": LogisticRegression(
        C=0.1,              # strong L2 regularization
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    ),
    "rf": RandomForestClassifier(
        n_estimators=200,
        max_depth=5,        # shallow — hard to overfit 891 rows
        min_samples_leaf=5, # leaves need at least 5 samples
        max_features=0.7,   # random subspace per split
        random_state=42,
        n_jobs=-1,
    ),
}

# ─── 10-fold CV (better estimate for small datasets than 3-fold) ─────
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\n=== 10-fold Cross-Validation ===")
cv_scores = {}
for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
    cv_scores[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
    print(f"  {name}: mean={scores.mean():.4f}, std={scores.std():.4f}")

# ─── Train on full data, generate OOF predictions via 10-fold ────────
print("\n=== Training full models + OOF predictions ===")
oof_preds = {}
test_preds = {}

preprocessor.fit(X)
X_transformed = preprocessor.transform(X)
X_test_transformed = preprocessor.transform(X_test)

n_features_after_encoding = X_transformed.shape[1]
print(f"Features after encoding: {n_features_after_encoding}")

for name, model in models.items():
    # OOF: cross-validated predictions on training data
    oof = np.zeros(len(y))
    for train_idx, val_idx in skf.split(X_transformed, y):
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_transformed[train_idx], y[train_idx])
        oof[val_idx] = model_clone.predict(X_transformed[val_idx])

    oof_acc = accuracy_score(y, oof)
    oof_preds[name] = {"accuracy": float(oof_acc), "predictions": oof}

    # Train on full data for test predictions
    model.fit(X_transformed, y)
    test_proba = model.predict_proba(X_test_transformed)[:, 1]
    test_preds[name] = test_proba

    print(f"  {name}: OOF accuracy = {oof_acc:.4f}")

# ─── Ensemble: simple average of probabilities ───────────────────────
print("\n=== Ensemble ===")
ensemble_test_proba = np.mean([test_preds[name] for name in models], axis=0)
ensemble_test_pred = (ensemble_test_proba >= 0.5).astype(int)

# OOF ensemble accuracy
ensemble_oof = np.mean([oof_preds[name]["predictions"] for name in models], axis=0) >= 0.5
ensemble_oof_acc = accuracy_score(y, ensemble_oof.astype(int))
print(f"  Ensemble OOF accuracy: {ensemble_oof_acc:.4f}")

# ─── Submission ──────────────────────────────────────────────────────
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": ensemble_test_pred,
})
submission_path = os.path.join(OUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"\nSubmission saved to: {submission_path}")
print(f"Prediction distribution: {submission['Survived'].value_counts().to_dict()}")

# ─── Metrics JSON ────────────────────────────────────────────────────
metrics = {
    "schema": "academic_research_os.local_optimized_metrics.v1",
    "run_id": f"optimized_{NOW_TS}_{RUN_HASH}",
    "task_id": "titanic",
    "train_rows": len(train),
    "test_rows": len(test),
    "n_features_raw": len(ALL_FEATURES),
    "n_features_after_encoding": n_features_after_encoding,
    "n_folds_cv": 10,
    "models": list(models.keys()),
    "cv_accuracy_10fold": cv_scores,
    "cv_best_mean": max(s["mean"] for s in cv_scores.values()),
    "cv_best_model": max(cv_scores, key=lambda k: cv_scores[k]["mean"]),
    "oof_accuracy": {name: v["accuracy"] for name, v in oof_preds.items()},
    "ensemble_oof_accuracy": float(ensemble_oof_acc),
    "ensemble_oof_balanced_accuracy": float(
        ((((ensemble_oof > 0.5).astype(int) & (y == 1)).sum() / max((y == 1).sum(), 1))
         + (((ensemble_oof <= 0.5).astype(int) & (y == 0)).sum() / max((y == 0).sum(), 1)))
        / 2.0
    ),
    "prediction_distribution": {
        "0": int((ensemble_test_pred == 0).sum()),
        "1": int((ensemble_test_pred == 1).sum()),
    },
    "strategy": "simple_features_7_columns_logreg_rf_10fold_cv",
    "fix_target": "reduce_cv_public_gap",
    "timestamp": NOW_TS,
}

metrics_path = os.path.join(OUT_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to: {metrics_path}")

# ─── Print final summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  10-fold CV (LR):         {cv_scores['lr']['mean']:.4f} +/- {cv_scores['lr']['std']:.4f}")
print(f"  10-fold CV (RF):         {cv_scores['rf']['mean']:.4f} +/- {cv_scores['rf']['std']:.4f}")
print(f"  OOF accuracy:            {ensemble_oof_acc:.4f}")
print(f"  Features (raw):          {len(ALL_FEATURES)}")
print(f"  Features (encoded):      {n_features_after_encoding}")
print(f"  Prediction [0/1]:        {metrics['prediction_distribution']}")
print(f"\n  Estimated public score:   ~{ensemble_oof_acc:.3f} (conservative, OOF-based)")
print(f"  Previous run CV:         0.820 (3-fold, 1731 features)")
print(f"  Previous Kaggle public:  0.744")
print(f"  CV-public gap (previous): 0.076")
print(f"  Expected CV-public gap:   < 0.02 (shallower models, 10 features, 10-fold CV)")
