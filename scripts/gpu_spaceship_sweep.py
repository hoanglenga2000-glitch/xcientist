"""GPU parallel training on spaceship-titanic across 4 A800 GPUs."""
import pandas as pd, numpy as np, json, sys, os
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from datetime import datetime

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MODEL_TYPE = sys.argv[2] if len(sys.argv) > 2 else "cb"

print(f"GPU {gpu_id}: {MODEL_TYPE} training on spaceship_titanic")

home = "/hpc2hdd/home/aimslab"
train = pd.read_csv(f"{home}/spaceship_titanic/train.csv")
test = pd.read_csv(f"{home}/spaceship_titanic/test.csv")

for df in [train, test]:
    df["Cabin_deck"] = df["Cabin"].str.split("/").str[0].fillna("Unknown")
    df["Cabin_num"] = pd.to_numeric(df["Cabin"].str.split("/").str[1], errors="coerce").fillna(-1)
    df["Cabin_side"] = df["Cabin"].str.split("/").str[2].fillna("Unknown")
    for c in ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]:
        df[c] = df[c].fillna(0)
    df["TotalSpend"] = df[["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]].sum(axis=1)
    df["Age"] = df["Age"].fillna(df["Age"].median())

cat_cols = ["HomePlanet","CryoSleep","Destination","VIP","Cabin_deck","Cabin_side"]
for c in cat_cols:
    train[c] = train[c].astype(str); test[c] = test[c].astype(str)

y = (train["Transported"].astype(str) == "True").astype(int)
features = cat_cols + ["Age","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck","TotalSpend","Cabin_num"]
X = train[features].fillna(0)
X_test = test[features].fillna(0)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42+gpu_id)
oof = np.zeros(len(y))
test_preds = np.zeros(len(test))

# Build catboost feature indices for categorical columns
cb_cat_idx = [features.index(c) for c in cat_cols if c in features]

for fold, (tr, va) in enumerate(skf.split(X, y)):
    if MODEL_TYPE == "cb":
        model = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=7,
                                   task_type="GPU", devices=str(gpu_id),
                                   random_seed=42+fold, verbose=100)
        model.fit(X.iloc[tr], y.iloc[tr], cat_features=cb_cat_idx)
    elif MODEL_TYPE == "lgb":
        model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.02, num_leaves=63,
                                   device="gpu", gpu_device_id=0,
                                   random_state=42+fold, verbose=-1)
        model.fit(X.iloc[tr], y.iloc[tr])
    elif MODEL_TYPE == "xgb":
        # XGBoost needs encoded categoricals
        X_enc = X.copy()
        for c in cat_cols:
            if c in X_enc.columns:
                X_enc[c] = X_enc[c].astype('category').cat.codes
        X_test_enc = X_test.copy()
        for c in cat_cols:
            if c in X_test_enc.columns:
                X_test_enc[c] = X_test_enc[c].astype('category').cat.codes
        model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=6,
                                  tree_method="gpu_hist", gpu_id=0,
                                  random_state=42+fold, enable_categorical=True)
        model.fit(X_enc.iloc[tr], y.iloc[tr])
    else:
        model = CatBoostClassifier(iterations=1500, learning_rate=0.02, depth=8,
                                   task_type="GPU", devices=str(gpu_id),
                                   random_seed=42+fold, verbose=100)
        model.fit(X.iloc[tr], y.iloc[tr], cat_features=cb_cat_idx)

    # Predict
    if MODEL_TYPE == "xgb":
        oof[va] = model.predict_proba(X_enc.iloc[va])[:, 1]
        test_preds += model.predict_proba(X_test_enc)[:, 1] / skf.n_splits
    else:
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
    oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
    acc = accuracy_score(y.iloc[va], (oof[va] > 0.5).astype(int))
    print(f"  Fold {fold+1}/5: acc={acc:.4f}")

oof_acc = accuracy_score(y, (oof > 0.5).astype(int))
pred = (test_preds > 0.5).astype(int)
print(f"GPU{gpu_id} {MODEL_TYPE} FINAL: OOF={oof_acc:.5f}, True={pred.sum()}/{len(pred)}")

result = {"gpu": gpu_id, "model": MODEL_TYPE, "oof_acc": float(oof_acc),
          "pred_true": int(pred.sum()), "time": datetime.now().isoformat()}
with open(f"/tmp/gpu{gpu_id}_{MODEL_TYPE}_result.json", "w") as f:
    json.dump(result, f)
