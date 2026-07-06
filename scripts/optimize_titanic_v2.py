"""
optimize_titanic_v2.py — Enhanced Titanic with Title-based Age imputation,
Ticket frequency, SVM+KNN diversity, and correlation-aware ensemble stacking.
"""
import os, json, warnings, hashlib, re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone

warnings.filterwarnings("ignore")

TRAIN_PATH = "D:/桌面/codex/科研港科技/tasks/titanic/data/train.csv"
TEST_PATH  = "D:/桌面/codex/科研港科技/tasks/titanic/data/test.csv"
NOW_TS     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
RUN_HASH   = hashlib.sha256(NOW_TS.encode()).hexdigest()[:8]
OUT_DIR    = f"D:/桌面/codex/科研港科技/experiments/titanic/v2_enhanced_{NOW_TS}_{RUN_HASH}"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output: {OUT_DIR}")

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print(f"Train: {train.shape}, Test: {test.shape}")

# ─── V2 Feature Engineering ──────────────────────────────────────────
def engineer_features(df):
    result = df.copy()

    # Title extraction
    result["Title"] = result["Name"].str.extract(r',\s*([^\.]+)\.', expand=False)
    result["Title"] = result["Title"].replace(
        ["Mlle","Ms","Mme"], ["Miss","Miss","Mrs"]
    )
    rare_titles = result["Title"].value_counts()
    rare_titles = rare_titles[rare_titles < 5].index
    result.loc[result["Title"].isin(rare_titles), "Title"] = "Rare"

    # Family size
    result["FamilySize"] = result["SibSp"] + result["Parch"] + 1
    result["IsAlone"] = (result["FamilySize"] == 1).astype(int)

    # Age imputation by Title group (global median fallback)
    title_age_medians = result.groupby("Title")["Age"].transform("median")
    global_age_median = result["Age"].median()
    result["Age"] = result["Age"].fillna(title_age_medians).fillna(global_age_median)

    # Ticket frequency
    result["TicketFreq"] = result.groupby("Ticket")["Ticket"].transform("count")

    # Fare binning and fillna
    result["Fare"] = result["Fare"].fillna(result["Fare"].median())
    result["FareBin"] = pd.qcut(result["Fare"], q=4, labels=False, duplicates="drop").astype(int)

    # Pclass * Sex interaction
    result["Pclass_Sex"] = result["Pclass"].astype(str) + "_" + result["Sex"].astype(str)

    # Cabin: has cabin or not
    result["HasCabin"] = result["Cabin"].notna().astype(int)

    # Fare per person
    result["FarePerPerson"] = result["Fare"] / result["FamilySize"]

    return result

train_fe = engineer_features(train)
test_fe  = engineer_features(test)

NUMERIC = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone",
           "TicketFreq", "FareBin", "HasCabin", "FarePerPerson"]
CATEGORICAL = ["Pclass", "Sex", "Embarked", "Title", "Pclass_Sex"]

X = train_fe[NUMERIC + CATEGORICAL].copy()
y = train_fe["Survived"].values
X_test = test_fe[NUMERIC + CATEGORICAL].copy()

print(f"Features: {len(NUMERIC)} numeric + {len(CATEGORICAL)} categorical")

# ─── Preprocessing ────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), NUMERIC),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(drop="first", sparse_output=False,
                                               handle_unknown="ignore"))]), CATEGORICAL),
])

# ─── Models (tree + non-tree diversity) ───────────────────────────────
models = {
    "lr": LogisticRegression(C=0.1, penalty="l2", solver="liblinear",
                             max_iter=2000, random_state=42),
    "rf": RandomForestClassifier(n_estimators=400, max_depth=12,
                                 min_samples_leaf=2, random_state=42, n_jobs=-1),
    "et": ExtraTreesClassifier(n_estimators=500, max_depth=14,
                               min_samples_leaf=2, random_state=42, n_jobs=-1),
    "gb": GradientBoostingClassifier(n_estimators=500, learning_rate=0.03,
                                     max_depth=3, min_samples_leaf=2,
                                     subsample=0.8, random_state=42),
    "hgb": HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                          max_depth=6, random_state=42),
    "svm": CalibratedClassifierCV(
        SVC(kernel="rbf", C=2.0, gamma="scale", probability=True,
            random_state=42), cv=3),
    "knn": KNeighborsClassifier(n_neighbors=7, weights="distance"),
}

# ─── 10-fold CV ───────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\n=== 10-fold CV ===")
cv_scores = {}
for name, model in models.items():
    pipe = Pipeline([("pp", preprocessor), ("clf", clone(model))])
    scores = []
    for tr, va in skf.split(X, y):
        pipe.fit(X.iloc[tr], y[tr])
        scores.append(accuracy_score(y[va], pipe.predict(X.iloc[va])))
    cv_scores[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
    print(f"  {name}: {cv_scores[name]['mean']:.4f} +/- {cv_scores[name]['std']:.4f}")

# ─── OOF + Test predictions ──────────────────────────────────────────
print("\n=== OOF predictions ===")
preprocessor.fit(X)
Xt = preprocessor.transform(X)
Xt_test = preprocessor.transform(X_test)
print(f"Encoded features: {Xt.shape[1]}")

oof_proba = {}
test_proba = {}

for name, model in models.items():
    oof = np.zeros((len(y), 2))
    test_fold = np.zeros((len(X_test), 2))
    for tr, va in skf.split(Xt, y):
        m = clone(model)
        m.fit(Xt[tr], y[tr])
        oof[va] = m.predict_proba(Xt[va])
        test_fold += m.predict_proba(Xt_test) / skf.n_splits
    oof_proba[name] = oof
    test_proba[name] = test_fold
    oof_acc = accuracy_score(y, oof[:, 1] >= 0.5)
    print(f"  {name}: OOF acc = {oof_acc:.4f}")

# ─── Correlation-aware stacking ──────────────────────────────────────
print("\n=== Stacking ===")
oof_survived = {n: p[:, 1] for n, p in oof_proba.items()}
test_survived = {n: p[:, 1] for n, p in test_proba.items()}
oof_df = pd.DataFrame(oof_survived)

# Select best model, then add uncorrelated models
best = max(cv_scores, key=lambda k: cv_scores[k]["mean"])
selected = [best]
for name in sorted(cv_scores, key=lambda k: cv_scores[k]["mean"], reverse=True):
    if name == best:
        continue
    max_corr = max(abs(oof_df[name].corr(oof_df[s])) for s in selected)
    if max_corr < 0.95:
        selected.append(name)
        if len(selected) >= 5:
            break

print(f"Selected models: {selected}")

# Stack with LogisticRegressionCV
stacker = LogisticRegressionCV(cv=3, scoring="accuracy", max_iter=2000, random_state=42)
stacker.fit(oof_df[selected], y)
ensemble_oof_proba = stacker.predict_proba(oof_df[selected])[:, 1]
ensemble_oof_pred = (ensemble_oof_proba >= 0.5).astype(int)
ensemble_oof_acc = accuracy_score(y, ensemble_oof_pred)

# Test predictions
test_stack_df = pd.DataFrame(test_survived)[selected]
ensemble_test_proba = stacker.predict_proba(test_stack_df)[:, 1]
ensemble_test_pred = (ensemble_test_proba >= 0.5).astype(int)

print(f"Ensemble OOF: {ensemble_oof_acc:.4f}")
print(f"Test preds: 0={int((ensemble_test_pred==0).sum())}, 1={int((ensemble_test_pred==1).sum())}")

# ─── Submission ──────────────────────────────────────────────────────
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": ensemble_test_pred,
})
submission.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)

# ─── Metrics ─────────────────────────────────────────────────────────
metrics = {
    "schema": "academic_research_os.v2_enhanced_metrics.v1",
    "run_id": f"v2_enhanced_{NOW_TS}_{RUN_HASH}",
    "task_id": "titanic",
    "n_features_raw": len(NUMERIC) + len(CATEGORICAL),
    "n_features_encoded": Xt.shape[1],
    "n_folds": 10,
    "cv_scores": cv_scores,
    "best_cv_model": best,
    "selected_stack_models": selected,
    "ensemble_oof_accuracy": float(ensemble_oof_acc),
    "prediction_distribution": {"0": int((ensemble_test_pred==0).sum()),
                                 "1": int((ensemble_test_pred==1).sum())},
    "strategy": "title_age_ticket_freq_svm_knn_corr_stack",
    "timestamp": NOW_TS,
}
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 60)
print(f"V2 Enhanced OOF: {ensemble_oof_acc:.4f}")
print(f"V1 Baseline OOF: ~0.819")
print(f"Expected Kaggle: ~{ensemble_oof_acc - 0.04:.5f}")
print(f"Output: {OUT_DIR}")
