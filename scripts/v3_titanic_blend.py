#!/usr/bin/env python3
"""Titanic 3-seed ensemble: seeds 42, 123, 456 with best params."""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

HOME = "/hpc2hdd/home/aimslab"
DATA_DIR = os.path.join(HOME, "titanic")

def load_and_feature_engineer():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    test_ids = test["PassengerId"].values.copy()
    train["is_train"] = 1; test["is_train"] = 0
    combined = pd.concat([train, test], ignore_index=True)
    combined["Title"] = combined["Name"].str.extract(r',\s*([^\.]+)\.', expand=False)
    title_map = {"Mr":"Mr","Mrs":"Mrs","Miss":"Miss","Master":"Master","Dr":"Dr","Rev":"Rev","Col":"Officer","Major":"Officer","Capt":"Officer","Don":"Royalty","Jonkheer":"Royalty","Sir":"Royalty","Lady":"Royalty","the Countess":"Royalty","Mme":"Mrs","Ms":"Miss","Mlle":"Miss","Dona":"Royalty"}
    combined["Title"] = combined["Title"].map(title_map).fillna("Other")
    combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
    combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)
    combined["Fare"] = combined["Fare"].fillna(combined["Fare"].median())
    combined["FarePerPerson"] = combined["Fare"] / combined["FamilySize"]
    combined["FareBin"] = pd.qcut(combined["Fare"], 5, labels=False, duplicates="drop")
    for title in combined["Title"].unique():
        mask = combined["Title"] == title
        combined.loc[mask, "Age"] = combined.loc[mask, "Age"].fillna(combined.loc[mask, "Age"].median())
    combined["Age"] = combined["Age"].fillna(combined["Age"].median())
    combined["AgeBin"] = pd.cut(combined["Age"], bins=[0,12,18,25,35,50,65,100], labels=False)
    combined["CabinDeck"] = combined["Cabin"].str[0].fillna("U")
    combined["HasCabin"] = combined["Cabin"].notna().astype(int)
    combined["CabinNum"] = combined["Cabin"].str.extract(r"(\d+)", expand=False)
    combined["CabinNum"] = combined["CabinNum"].str.split().str[0]
    combined["CabinNum"] = pd.to_numeric(combined["CabinNum"], errors="coerce").fillna(-1)
    combined["TicketPrefix"] = combined["Ticket"].apply(lambda x: x.split()[0].rstrip(".") if not x.split()[0].replace(".","").isdigit() else "NUM")
    pc = combined["TicketPrefix"].value_counts()
    combined["TicketPrefix"] = combined["TicketPrefix"].apply(lambda x: x if pc.get(x,0) >= 3 else "RARE")
    combined["Sex"] = combined["Sex"].map({"male":0, "female":1})
    combined["Pclass_Sex"] = combined["Pclass"].astype(str) + "_" + combined["Sex"].astype(str)
    combined["Age_Pclass"] = combined["AgeBin"].astype(str) + "_" + combined["Pclass"].astype(str)
    combined["Embarked"] = combined["Embarked"].fillna("S")
    drop_cols = ["PassengerId","Name","Ticket","Cabin","is_train"]
    train_feat = combined[combined["is_train"]==1].drop(columns=drop_cols)
    test_feat = combined[combined["is_train"]==0].drop(columns=drop_cols)
    y = train_feat.pop("Survived").astype(int)
    combined_feat = pd.concat([train_feat, test_feat], ignore_index=True)
    n_train = len(train_feat)
    for col in list(combined_feat.columns):
        if combined_feat[col].dtype == "object":
            if combined_feat[col].nunique() > 200:
                combined_feat.drop(columns=[col], inplace=True)
            else:
                combined_feat[col] = combined_feat[col].fillna("MISSING")
                combined_feat[col] = LabelEncoder().fit_transform(combined_feat[col].astype(str))
        elif combined_feat[col].dtype in ("float64","int64"):
            combined_feat[col] = combined_feat[col].fillna(combined_feat[col].median())
    X_train = combined_feat.iloc[:n_train].copy()
    X_test = combined_feat.iloc[n_train:].copy()
    return X_train, y, X_test, test_ids

X_train, y, X_test, test_ids = load_and_feature_engineer()
print(f"Features: {X_train.shape[1]}", flush=True)

SEEDS = [42, 123, 456]
all_oof = np.zeros((len(SEEDS), len(X_train)))
all_test = np.zeros((len(SEEDS), len(X_test)))

for si, seed in enumerate(SEEDS):
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train, y))
    scores = []
    t0 = time.time()
    oof_seed = np.zeros(len(X_train))
    test_seed = np.zeros(len(X_test))
    for fi, (tr, va) in enumerate(folds):
        m = CatBoostClassifier(
            iterations=2000, learning_rate=0.015, depth=6, l2_leaf_reg=8,
            bootstrap_type="Bayesian", bagging_temperature=0.5, random_strength=1.0,
            task_type="GPU", devices="0", verbose=0, random_seed=seed,
            allow_writing_files=False, early_stopping_rounds=80, use_best_model=True
        )
        m.fit(X_train.iloc[tr], y.iloc[tr], eval_set=(X_train.iloc[va], y.iloc[va]), verbose=False)
        p = m.predict_proba(X_train.iloc[va])[:, 1]
        oof_seed[va] = p
        test_seed += m.predict_proba(X_test)[:, 1] / 5
        scores.append(accuracy_score(y.iloc[va], (p > 0.5).astype(int)))
    mean_score = float(np.mean(scores))
    oof_acc = accuracy_score(y, (oof_seed > 0.5).astype(int))
    all_oof[si] = oof_seed
    all_test[si] = test_seed
    print(f"Seed {seed}: fold_mean={mean_score:.4f} oof_acc={oof_acc:.4f} [{time.time()-t0:.0f}s]", flush=True)

# Ensemble all 3 seeds
ens_oof = all_oof.mean(axis=0)
ens_acc = accuracy_score(y, (ens_oof > 0.5).astype(int))
ens_test = all_test.mean(axis=0)

print(f"\nEnsemble (3 seeds): OOF={ens_acc:.4f}", flush=True)

# Submission
pred_int = (ens_test > 0.5).astype(int)
sub = pd.DataFrame({"PassengerId": test_ids, "Survived": pred_int})
sub_path = "/hpc2hdd/home/aimslab/results/v3_submission_titanic.csv"
sub.to_csv(sub_path, index=False)

result_json = {
    "task_id": "titanic", "status": "completed",
    "metric": "accuracy", "direction": "max",
    "oof_score": round(float(ens_acc), 6),
    "seeds": SEEDS, "n_features": X_train.shape[1],
    "submission_path": sub_path,
}
with open("/hpc2hdd/home/aimslab/results/v3_result_titanic.json", "w") as f:
    json.dump(result_json, f)

print(f"Ensemble OOF: {ens_acc:.4f}")
print(f"Submission saved to {sub_path}")
