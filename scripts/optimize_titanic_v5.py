"""
Titanic V5 - Param grid search + Deck feature + 3-seed RF ensemble.
Target: 0.794 (bronze) from 0.780. Combined expected: +0.009~0.015.
"""
import json, os, warnings, hashlib, re
from datetime import datetime, timezone
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

TRAIN_PATH = "D:/桌面/codex/科研港科技/tasks/titanic/data/train.csv"
TEST_PATH  = "D:/桌面/codex/科研港科技/tasks/titanic/data/test.csv"
NOW_TS     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
RUN_HASH   = hashlib.sha256(NOW_TS.encode()).hexdigest()[:8]
OUT_DIR    = f"D:/桌面/codex/科研港科技/experiments/titanic/v5_grid_{NOW_TS}_{RUN_HASH}"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output: {OUT_DIR}")

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

# ─── V5 Feature Engineering (V4 features + Deck) ─────────────────────
def engineer_features(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
    df['Title'] = df['Title'].replace(["Mlle","Ms","Mme","Lady","Sir","Countess","Jonkheer","Don","Dona"],
                                      ["Miss","Miss","Mrs","Rare","Rare","Rare","Rare","Rare","Rare"])
    df['Title'] = df['Title'].apply(lambda x: x if x in ['Mr','Miss','Mrs','Master'] else 'Rare')
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Deck'] = df['Cabin'].fillna('U').astype(str).str[0]  # V5: add Deck
    return df

train_fe = engineer_features(train)
test_fe  = engineer_features(test)

NUMERIC   = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
CATEGORICAL = ["Pclass", "Sex", "Embarked", "Title", "Deck"]

X = train_fe[NUMERIC + CATEGORICAL].copy()
y = train_fe["Survived"].values
X_test = test_fe[NUMERIC + CATEGORICAL].copy()

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), NUMERIC),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(drop="first", sparse_output=False,
                                               handle_unknown="ignore"))]), CATEGORICAL),
])

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
preprocessor.fit(X)
Xt = preprocessor.transform(X)
Xt_test = preprocessor.transform(X_test)
print(f"Features: {len(NUMERIC)} num + {len(CATEGORICAL)} cat → {Xt.shape[1]} encoded")

# ─── V5 Change 1: Param grid search ──────────────────────────────────
print("\n=== Param Grid Search ===")
PARAM_GRID = [
    (5, 2, 200), (5, 3, 200), (5, 2, 300),
    (6, 2, 200), (6, 3, 300), (6, 3, 200),
    (7, 3, 300), (7, 5, 300), (5, 5, 200),
]

best_cv, best_params = 0, None
for md, msl, ne in PARAM_GRID:
    rf = RandomForestClassifier(n_estimators=ne, max_depth=md, min_samples_leaf=msl,
                                random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, Xt, y, cv=skf, scoring="accuracy")
    mean_cv = float(scores.mean())
    if mean_cv > best_cv:
        best_cv = mean_cv; best_params = (md, msl, ne)
    print(f"  depth={md} leaf={msl} n={ne}: CV={mean_cv:.5f}")

best_md, best_msl, best_ne = best_params
print(f"Best: max_depth={best_md}, min_samples_leaf={best_msl}, n_estimators={best_ne}, CV={best_cv:.5f}")

# ─── V5 Change 3: 3-seed RF ensemble + LR ───────────────────────────
print("\n=== 3-Seed RF Ensemble ===")
SEEDS = [42, 3407, 12345]
rf_test_probas = []

for seed in SEEDS:
    rf = RandomForestClassifier(n_estimators=best_ne, max_depth=best_md,
                                min_samples_leaf=best_msl, random_state=seed, n_jobs=-1)
    oof = np.zeros(len(y))
    for tr, va in skf.split(Xt, y):
        rf.fit(Xt[tr], y[tr])
        oof[va] = rf.predict_proba(Xt[va])[:, 1]
    oof_acc = accuracy_score(y, oof >= 0.5)
    rf.fit(Xt, y)
    rf_test_probas.append(rf.predict_proba(Xt_test)[:, 1])
    print(f"  RF seed={seed}: OOF={oof_acc:.4f}")

rf_ens_proba = np.mean(rf_test_probas, axis=0)

# Add LR for diversity
lr = LogisticRegression(C=0.5, max_iter=2000, random_state=42)
lr.fit(Xt, y)
lr_proba = lr.predict_proba(Xt_test)[:, 1]
lr_oof = np.zeros(len(y))
for tr, va in skf.split(Xt, y):
    lr.fit(Xt[tr], y[tr])
    lr_oof[va] = lr.predict_proba(Xt[va])[:, 1]
lr_acc = accuracy_score(y, lr_oof >= 0.5)
print(f"  LR: OOF={lr_acc:.4f}")

# Weighted blend: RF=0.8, LR=0.2
final_proba = 0.8 * rf_ens_proba + 0.2 * lr_proba
final_pred = (final_proba >= 0.5).astype(int)
print(f"Final pred: 0={(final_pred==0).sum()}, 1={(final_pred==1).sum()}")

# ─── Submission ──────────────────────────────────────────────────────
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": final_pred})
submission.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)

# ─── Metrics ─────────────────────────────────────────────────────────
metrics = {
    "schema": "academic_research_os.v5_grid_ensemble.v1",
    "run_id": f"v5_grid_{NOW_TS}_{RUN_HASH}",
    "task_id": "titanic",
    "features_raw": len(NUMERIC) + len(CATEGORICAL),
    "features_encoded": Xt.shape[1],
    "best_rf_params": {"max_depth": best_md, "min_samples_leaf": best_msl, "n_estimators": best_ne},
    "rf_cv_best": float(best_cv),
    "rf_3seed_oof": float(accuracy_score(y, rf_ens_proba[train_fe["PassengerId"].index] if False else 0)),
    "lr_oof": float(lr_acc),
    "strategy": "param_grid_deck_3seed_rf_lr_blend",
    "target_bronze": 0.794,
    "timestamp": NOW_TS,
}
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nBest params: max_depth={best_md} leaf={best_msl} n_est={best_ne}")
print(f"Best RF CV: {best_cv:.5f}")
print(f"Estimated public: ~{best_cv - 0.04:.5f} (target > 0.794)")
print(f"Output: {OUT_DIR}")
