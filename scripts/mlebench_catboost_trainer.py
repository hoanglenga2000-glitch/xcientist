#!/usr/bin/env python3
"""
MLE-Bench CatBoost GPU Trainer — Production-grade training for 3 competitions.
Replaces the simplified LGBM mlebench_proper_trainer.py with CatBoost GPU + feature engineering.
"""
import sys, os, time, warnings, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score

HOME = '/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra'
PREPARED = f'{HOME}/mlebench_prepared'
RESULTS = f'{HOME}/mlebench_proper_results'

# ============================================================
# Competition configs
# ============================================================
COMPETITIONS = {
    'spaceship-titanic': {
        'target': 'Transported',
        'id_col': 'PassengerId',
        'type': 'binary',
        'metric': 'accuracy',
        'output': 'bool',  # True/False
        'n_splits': 5,
        'catboost_params': {
            'iterations': 2000,
            'depth': 7,
            'learning_rate': 0.03,
            'l2_leaf_reg': 3,
            'border_count': 254,
            'random_strength': 0.5,
            'bagging_temperature': 0.5,
            'task_type': 'GPU',
            'devices': '0',
            'verbose': 200,
            'random_seed': 42,
            'eval_metric': 'Accuracy',
            'loss_function': 'Logloss',
            'early_stopping_rounds': 200,
        }
    },
    'tabular-playground-series-dec-2021': {
        'target': 'Cover_Type',
        'id_col': 'Id',
        'type': 'multiclass',
        'metric': 'accuracy',
        'output': 'int',  # integer class labels
        'n_splits': 5,
        'catboost_params': {
            'iterations': 3000,
            'depth': 8,
            'learning_rate': 0.03,
            'l2_leaf_reg': 5,
            'border_count': 254,
            'random_strength': 1.0,
            'bagging_temperature': 0.5,
            'task_type': 'GPU',
            'devices': '0',
            'verbose': 200,
            'random_seed': 42,
            'eval_metric': 'Accuracy',
            'loss_function': 'MultiClass',
            'early_stopping_rounds': 300,
            'classes_count': 7,
        }
    },
    'tabular-playground-series-may-2022': {
        'target': 'target',
        'id_col': 'id',
        'type': 'binary',
        'metric': 'auc_roc',
        'output': 'prob',  # probability [0,1]
        'n_splits': 5,
        'catboost_params': {
            'iterations': 3000,
            'depth': 7,
            'learning_rate': 0.03,
            'l2_leaf_reg': 3,
            'border_count': 254,
            'random_strength': 0.5,
            'bagging_temperature': 0.5,
            'task_type': 'GPU',
            'devices': '0',
            'verbose': 200,
            'random_seed': 42,
            'eval_metric': 'AUC',
            'loss_function': 'Logloss',
            'early_stopping_rounds': 300,
        }
    },
}

# ============================================================
# Feature engineering
# ============================================================
def spaceship_features(train, test):
    for df in [df for df in [train, test] if df is not None]:
        if 'CryoSleep' in df.columns and 'Age' in df.columns:
            df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(bool)
        spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
        if all(c in df.columns for c in spend_cols):
            for c in spend_cols:
                df[c] = df[c].fillna(0)
            df['TotalSpend'] = df[spend_cols].sum(axis=1)
            df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)
            df['LogTotalSpend'] = np.log1p(df['TotalSpend'])
        if 'PassengerId' in df.columns:
            df['Group'] = df['PassengerId'].str.split('_').str[0]
            df['GroupSize'] = df['Group'].map(df['Group'].value_counts())
        if 'Cabin' in df.columns:
            df['Deck'] = df['Cabin'].str[0].fillna('U')
            df['CabinNum'] = df['Cabin'].str.extract(r'(\d+)', expand=False).fillna(0).astype(int)
            df['CabinSide'] = df['Cabin'].str[-1].fillna('U')
        # Boolean conversion
        bool_cols = ['CryoSleep', 'VIP']
        for c in bool_cols:
            if c in df.columns:
                df[c] = df[c].fillna(False).astype(bool)
    return train, test

def dec2021_features(train, test):
    for df in [df for df in [train, test] if df is not None]:
        # Feature interactions for soil types
        soil_cols = [c for c in df.columns if c.startswith('Soil_Type')]
        if len(soil_cols) >= 40:
            df['Soil_Count'] = df[soil_cols].sum(axis=1)
            # Group soil types
            df['Soil_Group1_10'] = df[[f'Soil_Type{i}' for i in range(1, 11) if f'Soil_Type{i}' in df.columns]].sum(axis=1)
            df['Soil_Group11_20'] = df[[f'Soil_Type{i}' for i in range(11, 21) if f'Soil_Type{i}' in df.columns]].sum(axis=1)
            df['Soil_Group21_30'] = df[[f'Soil_Type{i}' for i in range(21, 31) if f'Soil_Type{i}' in df.columns]].sum(axis=1)
            df['Soil_Group31_40'] = df[[f'Soil_Type{i}' for i in range(31, 41) if f'Soil_Type{i}' in df.columns]].sum(axis=1)
        # Wilderness area interactions
        wa_cols = [c for c in df.columns if c.startswith('Wilderness_Area')]
        if len(wa_cols) >= 4:
            df['WA_Count'] = df[wa_cols].sum(axis=1)
        # Feature interactions
        if 'Elevation' in df.columns and 'Horizontal_Distance_To_Hydrology' in df.columns:
            df['Elevation_Hydrology'] = df['Elevation'] * df['Horizontal_Distance_To_Hydrology']
        if 'Horizontal_Distance_To_Roadways' in df.columns and 'Horizontal_Distance_To_Fire_Points' in df.columns:
            df['Road_Fire_Dist'] = df['Horizontal_Distance_To_Roadways'] + df['Horizontal_Distance_To_Fire_Points']
        if 'Hillshade_9am' in df.columns and 'Hillshade_Noon' in df.columns and 'Hillshade_3pm' in df.columns:
            df['Hillshade_Mean'] = df[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].mean(axis=1)
            df['Hillshade_Range'] = df[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].max(axis=1) - df[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].min(axis=1)
    return train, test

def may2022_features(train, test):
    for df in [df for df in [train, test] if df is not None]:
        # Feature interactions for 28 anonymized features (f_00 to f_27)
        feature_cols = [c for c in df.columns if c.startswith('f_')]
        if len(feature_cols) >= 28:
            # Force numeric: some columns may contain strings due to train/test distribution differences
            for c in feature_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            # Sums and means of groups
            df['f_sum_00_09'] = df[[f'f_{i:02d}' for i in range(10)]].sum(axis=1)
            df['f_sum_10_19'] = df[[f'f_{i:02d}' for i in range(10, 20)]].sum(axis=1)
            df['f_sum_20_27'] = df[[f'f_{i:02d}' for i in range(20, 28)]].sum(axis=1)
            df['f_mean'] = df[feature_cols].mean(axis=1)
            df['f_std'] = df[feature_cols].std(axis=1)
            df['f_max'] = df[feature_cols].max(axis=1)
            df['f_min'] = df[feature_cols].min(axis=1)
    return train, test

FEATURE_HOOKS = {
    'spaceship-titanic': spaceship_features,
    'tabular-playground-series-dec-2021': dec2021_features,
    'tabular-playground-series-may-2022': may2022_features,
}

# ============================================================
# Training
# ============================================================
def safe_predict_proba(model, X_data, n_classes):
    """Handle CatBoost returning fewer classes when some are missing."""
    p = model.predict_proba(X_data)
    if p.shape[1] < n_classes:
        full_p = np.zeros((len(p), n_classes), dtype=p.dtype)
        full_p[:, model.classes_.astype(int)] = p
        return full_p
    return p

def train_one(comp_id, config):
    data_dir = f'{PREPARED}/{comp_id}'
    train_path = f'{data_dir}/train.csv'
    test_path = f'{data_dir}/test.csv'
    os.makedirs(f'{RESULTS}/{comp_id}', exist_ok=True)

    if not os.path.exists(train_path):
        print(f'  SKIP: {train_path} not found')
        return None

    print(f'\n  Loading {train_path}...')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f'  Train: {train.shape}, Test: {test.shape}')

    target_col = config['target']
    id_col = config['id_col']
    output_type = config['output']
    metric = config['metric']
    cb_params = config['catboost_params'].copy()

    # Feature engineering
    hook = FEATURE_HOOKS.get(comp_id)
    if hook:
        train, test = hook(train, test)

    # Extract IDs and target
    test_ids = test[id_col].values if id_col in test.columns else None
    y = train[target_col].copy()
    if y.dtype == 'bool':
        y = y.astype(int)
    # CatBoost MultiClass requires 0-based labels
    if config['type'] == 'multiclass' and y.min() > 0:
        y = y - y.min()

    # Drop non-feature columns
    drop_cols = [target_col, id_col]
    drop_cols += [c for c in train.columns if c == 'sample_weight']
    train_feat = train.drop(columns=[c for c in drop_cols if c in train.columns], errors='ignore')
    test_feat = test.drop(columns=[c for c in drop_cols if c in test.columns], errors='ignore')

    # Combine for consistent encoding
    combined = pd.concat([train_feat, test_feat], ignore_index=True)
    cat_features = []
    for i, col in enumerate(combined.columns):
        if combined[col].dtype == 'object' or combined[col].dtype == 'bool':
            combined[col] = combined[col].astype(str)
            cat_features.append(i)

    train_feat = combined.iloc[:len(train_feat)].reset_index(drop=True)
    test_feat = combined.iloc[len(train_feat):].reset_index(drop=True)

    n_train = len(train_feat)
    print(f'  Features: {train_feat.shape[1]}, Categorical: {len(cat_features)}')

    # Cross-validation setup
    n_splits = config['n_splits']
    if config['type'] == 'multiclass':
        # Use KFold for multiclass (StratifiedKFold struggles with rare classes)
        try:
            folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(train_feat, y))
        except:
            folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(train_feat))
    else:
        folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(train_feat, y))
        # Fallback
        if len(folds) < n_splits:
            folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(train_feat))

    print(f'  KFold: {len(folds)} folds')

    # Train with CV
    if config['type'] == 'multiclass':
        n_classes = y.nunique()
        oof_preds = np.zeros((n_train, n_classes), dtype=np.float64)
        test_preds = np.zeros((len(test_feat), n_classes), dtype=np.float64)
        cb_params['classes_count'] = n_classes
    else:
        oof_preds = np.zeros(n_train, dtype=np.float64)
        test_preds = np.zeros(len(test_feat), dtype=np.float64)

    oof_scores = []

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        X_tr = train_feat.iloc[train_idx]
        X_val = train_feat.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        cb_params['random_seed'] = 42 + fold_i
        model = CatBoostClassifier(**cb_params)

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        model.fit(train_pool, eval_set=val_pool, verbose=cb_params.get('verbose', 200))
        print(f'  Fold {fold_i+1}/{len(folds)}: best_iter={model.get_best_iteration()}, best_score={model.get_best_score()}')

        if config['type'] == 'multiclass':
            val_prob = safe_predict_proba(model, X_val, n_classes)
            val_pred = val_prob.argmax(axis=1)
            oof_preds[val_idx] = val_prob
            test_prob = safe_predict_proba(model, test_feat, n_classes)
            test_preds += test_prob / len(folds)
        else:
            val_prob = model.predict_proba(X_val)[:, 1]
            if output_type == 'prob':
                oof_preds[val_idx] = val_prob
            else:
                oof_preds[val_idx] = (val_prob > 0.5).astype(int)
            test_prob = model.predict_proba(test_feat)[:, 1]
            test_preds += test_prob / len(folds)

        if metric == 'accuracy':
            if config['type'] == 'multiclass':
                score = accuracy_score(y_val, val_pred)
            else:
                score = accuracy_score(y_val, (val_prob > 0.5).astype(int))
        elif metric == 'auc_roc':
            score = roc_auc_score(y_val, val_prob)
        oof_scores.append(score)
        print(f'    Fold {fold_i+1} {metric}: {score:.6f}')

    mean_oof = np.mean(oof_scores)
    print(f'  Mean OOF {metric}: {mean_oof:.6f}')
    print(f'  Individual folds: {[f"{s:.6f}" for s in oof_scores]}')

    # Build final submission
    sub = pd.DataFrame()
    sub[id_col] = test_ids

    if config['type'] == 'multiclass':
        final_pred = test_preds.argmax(axis=1).astype(int) + 1 if n_classes > 0 else test_preds.argmax(axis=1)
        sub[target_col] = final_pred.astype(int)
    elif output_type == 'bool':
        final_pred = (test_preds > 0.5).astype(bool)
        sub[target_col] = final_pred
    elif output_type == 'prob':
        sub[target_col] = test_preds.clip(0.0, 1.0)
    else:
        final_pred = (test_preds > 0.5).astype(int)
        sub[target_col] = final_pred

    out_path = f'{RESULTS}/{comp_id}/submission_s44.csv'
    sub.to_csv(out_path, index=False)
    print(f'  Saved: {out_path} ({sub.shape})')
    print(f'  Preview: {sub.head(3).to_string()}')

    return {'competition': comp_id, 'oof_score': mean_oof, 'scores': oof_scores, 'path': out_path}

def main():
    print('='*60)
    print('MLE-Bench CatBoost GPU Trainer')
    print(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*60)

    all_results = []
    for comp_id in ['tabular-playground-series-dec-2021', 'tabular-playground-series-may-2022']:
        config = COMPETITIONS[comp_id]
        print(f'\n{"="*60}')
        print(f'  {comp_id} ({config["type"]}, {config["metric"]})')
        print(f'{"="*60}')
        try:
            result = train_one(comp_id, config)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f'  ERROR: {e}')
            import traceback
            traceback.print_exc()

    print(f'\n{"="*60}')
    print('  SUMMARY')
    print(f'{"="*60}')
    for r in all_results:
        print(f'  {r["competition"]:<45s} OOF={r["oof_score"]:.6f}  {r["scores"]}')

    with open(f'{RESULTS}/training_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nDone. Results saved to {RESULTS}/training_results.json')

if __name__ == '__main__':
    main()
