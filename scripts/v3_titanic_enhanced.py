#!/usr/bin/env python3
"""
V3 Titanic Enhanced: Better features + tuned CatBoost + 5-fold CV.
Target: Kaggle >= 0.794 (bronze). Current best: Kaggle 0.78947.
"""
import sys, os, json, time, warnings, argparse
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

HOME = '/hpc2hdd/home/aimslab'
DATA_DIR = os.path.join(HOME, 'titanic')


def load_and_feature_engineer():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    test_ids = test['PassengerId'].values.copy()

    # Combine for feature engineering
    train['is_train'] = 1
    test['is_train'] = 0
    combined = pd.concat([train, test], ignore_index=True)

    # ---- Title ----
    combined['Title'] = combined['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
    title_map = {
        'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
        'Dr': 'Dr', 'Rev': 'Rev', 'Col': 'Officer', 'Major': 'Officer',
        'Capt': 'Officer', 'Don': 'Royalty', 'Jonkheer': 'Royalty',
        'Sir': 'Royalty', 'Lady': 'Royalty', 'the Countess': 'Royalty',
        'Mme': 'Mrs', 'Ms': 'Miss', 'Mlle': 'Miss', 'Dona': 'Royalty'
    }
    combined['Title'] = combined['Title'].map(title_map).fillna('Other')

    # ---- Family features ----
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
    combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)

    # ---- Fare ----
    combined['Fare'] = combined['Fare'].fillna(combined['Fare'].median())
    combined['FarePerPerson'] = combined['Fare'] / combined['FamilySize']
    combined['FareBin'] = pd.qcut(combined['Fare'], 5, labels=False, duplicates='drop')

    # ---- Age ----
    # Fill missing age using Title group median
    for title in combined['Title'].unique():
        mask = combined['Title'] == title
        median_age = combined.loc[mask, 'Age'].median()
        combined.loc[mask, 'Age'] = combined.loc[mask, 'Age'].fillna(median_age)
    combined['Age'] = combined['Age'].fillna(combined['Age'].median())
    combined['AgeBin'] = pd.cut(combined['Age'], bins=[0, 12, 18, 25, 35, 50, 65, 100], labels=False)

    # ---- Cabin ----
    combined['CabinDeck'] = combined['Cabin'].str[0].fillna('U')
    combined['HasCabin'] = combined['Cabin'].notna().astype(int)
    combined['CabinNum'] = combined['Cabin'].str.extract(r'(\d+)', expand=False)
    # Multiple cabin numbers -> take first
    combined['CabinNum'] = combined['CabinNum'].str.split().str[0]
    combined['CabinNum'] = pd.to_numeric(combined['CabinNum'], errors='coerce').fillna(-1)

    # ---- Ticket ----
    combined['TicketPrefix'] = combined['Ticket'].apply(
        lambda x: x.split()[0].rstrip('.') if not x.split()[0].replace('.','').isdigit() else 'NUM'
    )
    # Group rare prefixes
    prefix_counts = combined['TicketPrefix'].value_counts()
    combined['TicketPrefix'] = combined['TicketPrefix'].apply(
        lambda x: x if prefix_counts[x] >= 3 else 'RARE'
    )

    # ---- Interactions ----
    combined['Sex'] = combined['Sex'].map({'male': 0, 'female': 1})
    combined['Pclass_Sex'] = combined['Pclass'].astype(str) + '_' + combined['Sex'].astype(str)
    combined['Age_Pclass'] = combined['AgeBin'].astype(str) + '_' + combined['Pclass'].astype(str)

    # ---- Embarked ----
    combined['Embarked'] = combined['Embarked'].fillna('S')

    # ---- Drop raw columns ----
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'is_train']

    # Separate back
    train_feat = combined[combined['is_train'] == 1].drop(columns=drop_cols)
    test_feat = combined[combined['is_train'] == 0].drop(columns=drop_cols)
    y = train_feat.pop('Survived').astype(int)

    # Re-combine for encoding
    combined_feat = pd.concat([train_feat, test_feat], ignore_index=True)
    n_train = len(train_feat)

    # Encode
    for col in list(combined_feat.columns):
        if combined_feat[col].dtype == 'object':
            if combined_feat[col].nunique() > 200:
                combined_feat.drop(columns=[col], inplace=True)
            else:
                combined_feat[col] = combined_feat[col].fillna('MISSING')
                combined_feat[col] = LabelEncoder().fit_transform(combined_feat[col].astype(str))
        elif combined_feat[col].dtype in ('float64', 'int64'):
            combined_feat[col] = combined_feat[col].fillna(combined_feat[col].median())

    X_train = combined_feat.iloc[:n_train].copy()
    X_test = combined_feat.iloc[n_train:].copy()

    return X_train, y, X_test, test_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    t_start = time.time()

    X_train, y, X_test, test_ids = load_and_feature_engineer()

    print(f"\n{'='*60}")
    print(f"TASK: titanic (enhanced features) | GPU: {args.gpu_device} | FOLDS: {args.n_folds}")
    print(f"{'='*60}")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")

    n_folds = args.n_folds
    n_iter = 500 if args.fast else 2000

    folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X_train, y))

    oof = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    scores = []

    cb_params = {
        'iterations': n_iter,
        'learning_rate': 0.015,
        'depth': 6,
        'l2_leaf_reg': 8,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 0.5,
        'random_strength': 1.0,
        'task_type': 'GPU',
        'devices': str(args.gpu_device),
        'verbose': 0,
        'random_seed': 42,
        'allow_writing_files': False,
        'early_stopping_rounds': 80,
        'use_best_model': True,
    }

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        print(f"  Fold {fold_idx+1}/{n_folds}...", end=' ', flush=True)
        t0 = time.time()
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = CatBoostClassifier(loss_function='Logloss', eval_metric='Accuracy', **cb_params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        if test_preds is not None:
            test_preds += model.predict_proba(X_test)[:, 1] / n_folds

        fold_score = accuracy_score(y_val, (val_pred > 0.5).astype(int))
        scores.append(fold_score)
        print(f"score={fold_score:.4f} [{time.time()-t0:.0f}s]")

    oof_score = float(np.mean(scores))
    oof_std = float(np.std(scores))
    final_metric = accuracy_score(y, (oof > 0.5).astype(int))

    print(f"\n  CV Fold Scores: {[f'{s:.4f}' for s in scores]}")
    print(f"  OOF (fold mean): {oof_score:.4f} +/- {oof_std:.4f}")
    print(f"  OOF (accuracy): {final_metric:.4f}")

    # GATE
    bronze, margin = 0.794, 0.010
    passed = final_metric >= bronze + margin
    gap = bronze + margin - final_metric
    gate_status = "PASS" if passed else "FAIL"
    print(f"  GATE: {gate_status} | accuracy={final_metric:.4f} vs bronze={bronze} margin={margin} | gap={gap:+.4f}")

    # Submission
    pred_int = (test_preds > 0.5).astype(int)
    sub = pd.DataFrame({'PassengerId': test_ids, 'Survived': pred_int})
    sub_path = f"/hpc2hdd/home/aimslab/results/v3_submission_titanic.csv"
    sub.to_csv(sub_path, index=False)

    elapsed = time.time() - t_start
    result_json = {
        "task_id": "titanic", "status": "completed",
        "metric": "accuracy", "direction": "max",
        "oof_score": round(final_metric, 6),
        "oof_std": round(oof_std, 6),
        "cv_scores": [round(float(s), 6) for s in scores],
        "bronze_threshold": bronze,
        "gate_passed": passed,
        "gate_gap": round(float(gap), 6),
        "n_folds": n_folds, "n_features": X_train.shape[1],
        "elapsed_seconds": round(elapsed, 1),
        "submission_path": sub_path,
    }
    result_path = f"/hpc2hdd/home/aimslab/results/v3_result_titanic.json"
    with open(result_path, 'w') as f:
        json.dump(result_json, f)

    print(f"  RESULT: {json.dumps({k: result_json[k] for k in ['task_id', 'oof_score', 'gate_passed', 'gate_gap', 'elapsed_seconds']})}")
    print(f"SUMMARY: titanic | accuracy={final_metric:.4f} | GATE={gate_status} | {elapsed:.0f}s | {X_train.shape[1]} features")


if __name__ == '__main__':
    main()
