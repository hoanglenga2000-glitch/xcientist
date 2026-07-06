#!/usr/bin/env python3
"""
MLE-Bench CatBoost GPU Trainer v2 — Systematic optimization.
Enhancements: GroupKFold, multi-seed, advanced feature engineering, calibration.
"""
import sys, os, time, warnings, json
os.environ['PYTHONUNBUFFERED'] = '1'
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

HOME = '/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra'
PREPARED = f'{HOME}/mlebench_prepared'
RESULTS = f'{HOME}/mlebench_proper_results'

# Multi-seed list: each model trained with every seed, predictions averaged
SEEDS = [42, 123, 256, 789, 1024]

# ============================================================
# Competition configs — tuned parameters
# ============================================================
COMPETITIONS = {
    'spaceship-titanic': {
        'target': 'Transported',
        'id_col': 'PassengerId',
        'type': 'binary',
        'metric': 'accuracy',
        'output': 'bool',
        'n_splits': 5,
        'use_group_kfold': False,
        'group_col': None,
        'catboost_params': {
            'iterations': 5000,
            'depth': 8,
            'learning_rate': 0.02,
            'l2_leaf_reg': 3,
            'border_count': 254,
            'random_strength': 1.0,
            'bagging_temperature': 0.5,
            'min_data_in_leaf': 20,
            'grow_policy': 'Lossguide',
            'gpu_ram_part': 0.75,
            'task_type': 'GPU',
            'devices': '0',
            'verbose': 500,
            'random_seed': 42,
            'eval_metric': 'Accuracy',
            'loss_function': 'Logloss',
            'early_stopping_rounds': 300,
        }
    },
    'tabular-playground-series-dec-2021': {
        'target': 'Cover_Type',
        'id_col': 'Id',
        'type': 'multiclass',
        'metric': 'accuracy',
        'output': 'int',
        'n_splits': 5,
        'use_group_kfold': False,
        'group_col': None,
        'catboost_params': {
            'iterations': 8000,
            'depth': 10,
            'learning_rate': 0.02,
            'l2_leaf_reg': 5,
            'border_count': 254,
            'random_strength': 1.0,
            'bagging_temperature': 0.5,
            'min_data_in_leaf': 20,
            'grow_policy': 'Lossguide',
            'gpu_ram_part': 0.70,
            'task_type': 'GPU',
            'devices': '0',
            'verbose': 500,
            'random_seed': 42,
            'eval_metric': 'Accuracy',
            'loss_function': 'MultiClass',
            'early_stopping_rounds': 400,
        }
    },
    'tabular-playground-series-may-2022': {
        'target': 'target',
        'id_col': 'id',
        'type': 'binary',
        'metric': 'auc_roc',
        'output': 'prob',
        'n_splits': 5,
        'use_group_kfold': True,
        'group_col': 'f_27',
        'catboost_params': {
            'iterations': 6000,
            'depth': 8,
            'learning_rate': 0.01,
            'l2_leaf_reg': 1,
            'border_count': 254,
            'random_strength': 1.0,
            'bagging_temperature': 0.3,
            'min_data_in_leaf': 20,
            'grow_policy': 'Lossguide',
            'gpu_ram_part': 0.75,
            'rsm': 0.85,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.85,
            'task_type': 'GPU',
            'devices': '0',
            'verbose': 500,
            'random_seed': 42,
            'eval_metric': 'AUC',
            'loss_function': 'Logloss',
            'early_stopping_rounds': 500,
        }
    },
}

# ============================================================
# Enhanced feature engineering
# ============================================================

def spaceship_features(train, test):
    """Enhanced spaceship-titanic features with logical imputation & interactions."""
    for df in [df for df in [train, test] if df is not None]:
        # --- Logical imputation: CryoSleep implies zero spending ---
        if 'CryoSleep' in df.columns:
            df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(bool)

        spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        if all(c in df.columns for c in spend_cols):
            for c in spend_cols:
                df[c] = df[c].fillna(0)
            # Logical: if CryoSleep, spending must be 0
            if 'CryoSleep' in df.columns:
                cryo_mask = df['CryoSleep'] == True
                for c in spend_cols:
                    df.loc[cryo_mask, c] = 0
            df['TotalSpend'] = df[spend_cols].sum(axis=1)
            df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)
            df['LogTotalSpend'] = np.log1p(df['TotalSpend'])
            # Individual spend ratios (which service dominates)
            df['SpendRatio_RoomService'] = df['RoomService'] / (df['TotalSpend'] + 1e-8)
            df['SpendRatio_ShoppingMall'] = df['ShoppingMall'] / (df['TotalSpend'] + 1e-8)
            df['SpendRatio_VRDeck'] = df['VRDeck'] / (df['TotalSpend'] + 1e-8)

        # --- Group features ---
        if 'PassengerId' in df.columns:
            df['Group'] = df['PassengerId'].str.split('_').str[0]
            df['GroupSize'] = df['Group'].map(df['Group'].value_counts())
            df['IsSolo'] = (df['GroupSize'] == 1).astype(int)

        # --- Cabin features ---
        if 'Cabin' in df.columns:
            df['Deck'] = df['Cabin'].str[0].fillna('U')
            df['CabinNum'] = df['Cabin'].str.extract(r'(\d+)', expand=False).fillna(0).astype(int)
            df['CabinSide'] = df['Cabin'].str[-1].fillna('U')
            # Deck × Side interaction
            df['Deck_Side'] = df['Deck'].astype(str) + '_' + df['CabinSide'].astype(str)
            # Cabin bin (group by 100)
            df['CabinBin'] = (df['CabinNum'] // 100 * 100).astype(int)

        # --- Age features ---
        if 'Age' in df.columns:
            df['Age'] = df['Age'].fillna(df['Age'].median())
            df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 25, 35, 50, 80], labels=False).fillna(3).astype(int)
            df['IsChild'] = (df['Age'] < 12).astype(int)
            df['IsTeen'] = ((df['Age'] >= 12) & (df['Age'] < 18)).astype(int)

        # --- Route ---
        if 'HomePlanet' in df.columns and 'Destination' in df.columns:
            df['Route'] = df['HomePlanet'].fillna('Unknown').astype(str) + '_' + df['Destination'].fillna('Unknown').astype(str)

        # --- Interactions ---
        if 'VIP' in df.columns and 'CryoSleep' in df.columns:
            df['VIP'] = df['VIP'].fillna(False).astype(bool)
            df['VIP_Cryo'] = (df['VIP'] & df['CryoSleep']).astype(int)

        # Boolean conversion
        for c in ['CryoSleep', 'VIP']:
            if c in df.columns:
                df[c] = df[c].fillna(False).astype(bool)

    return train, test


def dec2021_features(train, test):
    """Enhanced dec-2021 features: trig encoding, euclidean dist, hillshade ratios, binning, interactions."""
    for df in [df for df in [train, test] if df is not None]:
        # --- Trigonometric encoding of Aspect (circular feature) ---
        if 'Aspect' in df.columns:
            df['Aspect_sin'] = np.sin(np.radians(df['Aspect']))
            df['Aspect_cos'] = np.cos(np.radians(df['Aspect']))

        # --- Euclidean distance features ---
        if 'Elevation' in df.columns:
            if 'Horizontal_Distance_To_Hydrology' in df.columns:
                df['Euc_Dist_Hydrology'] = np.sqrt(df['Elevation']**2 + df['Horizontal_Distance_To_Hydrology']**2)
            if 'Horizontal_Distance_To_Roadways' in df.columns:
                df['Euc_Dist_Roadways'] = np.sqrt(df['Elevation']**2 + df['Horizontal_Distance_To_Roadways']**2)
            if 'Horizontal_Distance_To_Fire_Points' in df.columns:
                df['Euc_Dist_Fire'] = np.sqrt(df['Elevation']**2 + df['Horizontal_Distance_To_Fire_Points']**2)

        # --- Hillshade ratios ---
        if 'Hillshade_Noon' in df.columns:
            if 'Hillshade_9am' in df.columns:
                df['Hillshade_Ratio_9am'] = df['Hillshade_Noon'] / (df['Hillshade_9am'] + 1)
            if 'Hillshade_3pm' in df.columns:
                df['Hillshade_Ratio_3pm'] = df['Hillshade_Noon'] / (df['Hillshade_3pm'] + 1)
            # Max-min hillshade spread
            hill_cols = [c for c in ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'] if c in df.columns]
            if len(hill_cols) == 3:
                df['Hillshade_Mean'] = df[hill_cols].mean(axis=1)
                df['Hillshade_Range'] = df[hill_cols].max(axis=1) - df[hill_cols].min(axis=1)
                df['Hillshade_Std'] = df[hill_cols].std(axis=1)

        # --- Elevation binning ---
        if 'Elevation' in df.columns:
            df['Elevation_Bin'] = pd.qcut(df['Elevation'], q=20, labels=False, duplicates='drop')

        # --- Soil type grouping ---
        soil_cols = [c for c in df.columns if c.startswith('Soil_Type')]
        if len(soil_cols) >= 40:
            df['Soil_Count'] = df[soil_cols].sum(axis=1)
            for g_start in range(1, 41, 10):
                g_cols = [f'Soil_Type{i}' for i in range(g_start, g_start+10) if f'Soil_Type{i}' in df.columns]
                if g_cols:
                    df[f'Soil_Group{g_start}_{g_start+9}'] = df[g_cols].sum(axis=1)

        # --- Wilderness area ---
        wa_cols = [c for c in df.columns if c.startswith('Wilderness_Area')]
        if len(wa_cols) >= 4:
            df['WA_Count'] = df[wa_cols].sum(axis=1)
            # Wilderness × Soil cross interactions
            if len(soil_cols) >= 40:
                for wa_i, wa in enumerate(wa_cols[:4]):
                    for sg in ['Soil_Group1_10', 'Soil_Group11_20', 'Soil_Group21_30', 'Soil_Group31_40']:
                        if sg in df.columns:
                            df[f'{wa}_{sg}'] = df[wa] * df[sg]

        # --- Feature interactions ---
        if 'Elevation' in df.columns and 'Slope' in df.columns:
            df['Elevation_Slope'] = df['Elevation'] * df['Slope']
        if 'Horizontal_Distance_To_Roadways' in df.columns and 'Horizontal_Distance_To_Fire_Points' in df.columns:
            df['Road_Fire_Dist'] = df['Horizontal_Distance_To_Roadways'] + df['Horizontal_Distance_To_Fire_Points']
            df['Road_Fire_Ratio'] = df['Horizontal_Distance_To_Roadways'] / (df['Horizontal_Distance_To_Fire_Points'] + 1)
        if 'Elevation' in df.columns and 'Horizontal_Distance_To_Roadways' in df.columns:
            df['Elevation_Road'] = df['Elevation'] * df['Horizontal_Distance_To_Roadways']

    return train, test


def may2022_features(train, test):
    """Enhanced may-2022 features: group aggregations, interactions, polynomial, KMeans, PCA."""
    for df in [df for df in [train, test] if df is not None]:
        feature_cols = [c for c in df.columns if c.startswith('f_') and c != 'f_27']

        # Force numeric
        for c in feature_cols + (['f_27'] if 'f_27' in df.columns else []):
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        if len(feature_cols) < 27:
            continue

        # --- Group aggregations by f_27 (CRITICAL: f_27 is the group key) ---
        if 'f_27' in df.columns and not df['f_27'].isna().all():
            group_col = df['f_27']
            for fc in feature_cols:
                gmap_mean = df.groupby(group_col)[fc].transform('mean')
                gmap_std  = df.groupby(group_col)[fc].transform('std').fillna(0)
                gmap_min  = df.groupby(group_col)[fc].transform('min')
                gmap_max  = df.groupby(group_col)[fc].transform('max')
                df[f'{fc}_gmean'] = gmap_mean
                df[f'{fc}_gstd']  = gmap_std
                df[f'{fc}_gmin']  = gmap_min
                df[f'{fc}_gmax']  = gmap_max
                df[f'{fc}_grange'] = gmap_max - gmap_min

        # --- Group sums and means ---
        df['f_sum_00_09'] = df[[f'f_{i:02d}' for i in range(10)]].sum(axis=1)
        df['f_sum_10_19'] = df[[f'f_{i:02d}' for i in range(10, 20)]].sum(axis=1)
        df['f_sum_20_27'] = df[[f'f_{i:02d}' for i in range(20, 28)]].sum(axis=1)
        df['f_mean'] = df[feature_cols].mean(axis=1)
        df['f_std'] = df[feature_cols].std(axis=1)
        df['f_max'] = df[feature_cols].max(axis=1)
        df['f_min'] = df[feature_cols].min(axis=1)
        df['f_skew'] = df[feature_cols].skew(axis=1)
        df['f_kurt'] = df[feature_cols].kurtosis(axis=1)
        df['f_range'] = df['f_max'] - df['f_min']

        # --- Interaction features: top-10 features pairwise (45 interactions) ---
        top_cols = [f'f_{i:02d}' for i in range(10)]
        if all(c in df.columns for c in top_cols):
            for i in range(10):
                for j in range(i+1, 10):
                    ci, cj = f'f_{i:02d}', f'f_{j:02d}'
                    df[f'inter_{i}_{j}'] = df[ci] * df[cj]

        # --- Polynomial features: square/cube of top-10 ---
        for i in range(10):
            c = f'f_{i:02d}'
            if c in df.columns:
                df[f'{c}_sq'] = df[c] ** 2
                df[f'{c}_cube'] = df[c] ** 3

        # --- Unsupervised features ---
        # K-Means clustering (k=10)
        km = KMeans(n_clusters=10, random_state=42, n_init='auto')
        df['kmeans_cluster'] = km.fit_predict(df[feature_cols].fillna(0))
        # PCA (5 components)
        pca = PCA(n_components=5, random_state=42)
        pca_result = pca.fit_transform(df[feature_cols].fillna(0))
        for i in range(5):
            df[f'pca_{i}'] = pca_result[:, i]

    return train, test


FEATURE_HOOKS = {
    'spaceship-titanic': spaceship_features,
    'tabular-playground-series-dec-2021': dec2021_features,
    'tabular-playground-series-may-2022': may2022_features,
}

# ============================================================
# Training utilities
# ============================================================

def safe_predict_proba(model, X_data, n_classes):
    """Handle CatBoost returning fewer classes when some are missing."""
    p = model.predict_proba(X_data)
    if p.shape[1] < n_classes:
        full_p = np.zeros((len(p), n_classes), dtype=p.dtype)
        full_p[:, model.classes_.astype(int)] = p
        return full_p
    return p


def calibrate_proba(oof_preds, y_true, test_preds):
    """Platt scaling calibration using OOF predictions."""
    from sklearn.linear_model import LogisticRegression
    cal = LogisticRegression(C=10, random_state=42)
    cal.fit(oof_preds.reshape(-1, 1), y_true)
    oof_cal = cal.predict_proba(oof_preds.reshape(-1, 1))[:, 1]
    test_cal = cal.predict_proba(test_preds.reshape(-1, 1))[:, 1]
    return oof_cal, test_cal


# ============================================================
# Training pipeline
# ============================================================

def train_one(comp_id, config, seeds):
    data_dir = f'{PREPARED}/{comp_id}'
    train_path = f'{data_dir}/train.csv'
    test_path  = f'{data_dir}/test.csv'
    os.makedirs(f'{RESULTS}/{comp_id}', exist_ok=True)

    if not os.path.exists(train_path):
        print(f'  SKIP: {train_path} not found')
        return None

    print(f'\n  Loading {train_path}...')
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    print(f'  Train: {train.shape}, Test: {test.shape}')

    target_col  = config['target']
    id_col      = config['id_col']
    output_type = config['output']
    metric      = config['metric']

    # Feature engineering
    hook = FEATURE_HOOKS.get(comp_id)
    if hook:
        train, test = hook(train, test)

    # Extract IDs and target
    test_ids = test[id_col].values if id_col in test.columns else None
    y = train[target_col].copy()
    if y.dtype == 'bool':
        y = y.astype(int)
    if config['type'] == 'multiclass' and y.min() > 0:
        y = y - y.min()

    # Drop non-feature columns
    drop_cols = [target_col, id_col, 'sample_weight']
    feat_drop = [c for c in drop_cols if c in train.columns]

    train_feat = train.drop(columns=feat_drop, errors='ignore')
    test_feat  = test.drop(columns=[c for c in feat_drop if c in test.columns], errors='ignore')

    # Categorical encoding: only combine if actual categorical cols exist
    cat_features = []
    has_cat = False
    for i, col in enumerate(train_feat.columns):
        if train_feat[col].dtype == 'object' or train_feat[col].dtype == 'bool':
            has_cat = True
            train_feat[col] = train_feat[col].astype(str)
            test_feat[col] = test_feat[col].astype(str)
            cat_features.append(i)

    # Combine only if needed (avoid large memory allocation for numeric-only data)
    if has_cat and len(cat_features) > 0:
        combined = pd.concat([train_feat, test_feat], ignore_index=True)
        for i in cat_features:
            combined.iloc[:, i] = combined.iloc[:, i].astype(str)
        train_feat = combined.iloc[:len(train_feat)].reset_index(drop=True)
        test_feat  = combined.iloc[len(train_feat):].reset_index(drop=True)
        del combined

    n_train = len(train_feat)
    print(f'  Features: {train_feat.shape[1]}, Categorical: {len(cat_features)}')
    print(f'  Multi-seed: {len(seeds)} seeds: {seeds}')

    # --- Cross-validation setup ---
    n_splits = config['n_splits']
    use_group = config.get('use_group_kfold', False)
    group_col = config.get('group_col', None)

    if use_group and group_col and group_col in train.columns:
        groups = train[group_col].fillna(0).values
        try:
            folds = list(GroupKFold(n_splits=n_splits).split(train_feat, y, groups=groups))
            print(f'  GroupKFold (group={group_col}): {len(folds)} folds')
        except Exception as e:
            print(f'  GroupKFold failed ({e}), falling back to StratifiedKFold')
            use_group = False
            folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(train_feat, y))
    if not use_group:
        if config['type'] == 'multiclass':
            try:
                folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(train_feat, y))
            except:
                folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(train_feat))
        else:
            folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(train_feat, y))

    # --- Multi-seed training ---
    cb_params = config['catboost_params'].copy()
    if config['type'] == 'multiclass':
        n_classes = y.nunique()
        cb_params['classes_count'] = n_classes
        all_test_preds = np.zeros((len(seeds), len(test_feat), n_classes), dtype=np.float64)
        all_oof_preds = np.zeros((len(seeds), n_train, n_classes), dtype=np.float64)
    else:
        n_classes = 2
        all_test_preds = np.zeros((len(seeds), len(test_feat)), dtype=np.float64)
        all_oof_preds = np.zeros((len(seeds), n_train), dtype=np.float64)
    all_seed_scores = {}

    for seed_i, seed in enumerate(seeds):
        print(f'\n  --- Seed {seed} ({seed_i+1}/{len(seeds)}) ---')
        cb_params['random_seed'] = seed

        if config['type'] == 'multiclass':
            oof_preds = np.zeros((n_train, n_classes), dtype=np.float64)
        else:
            oof_preds = np.zeros(n_train, dtype=np.float64)

        fold_scores = []

        for fold_i, (train_idx, val_idx) in enumerate(folds):
            X_tr  = train_feat.iloc[train_idx]
            X_val = train_feat.iloc[val_idx]
            y_tr  = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = CatBoostClassifier(**cb_params)
            train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
            val_pool   = Pool(X_val, y_val, cat_features=cat_features)

            model.fit(train_pool, eval_set=val_pool, verbose=cb_params.get('verbose', 500))
            best_iter = model.get_best_iteration()
            best_score = model.get_best_score()

            if config['type'] == 'multiclass':
                val_prob = safe_predict_proba(model, X_val, n_classes)
                val_pred = val_prob.argmax(axis=1)
                oof_preds[val_idx] = val_prob
                test_prob = safe_predict_proba(model, test_feat, n_classes)
                fold_score = accuracy_score(y_val, val_pred)
            else:
                val_prob = model.predict_proba(X_val)[:, 1]
                if output_type == 'prob':
                    oof_preds[val_idx] = val_prob
                else:
                    oof_preds[val_idx] = (val_prob > 0.5).astype(int)
                test_prob = model.predict_proba(test_feat)[:, 1]
                if metric == 'accuracy':
                    fold_score = accuracy_score(y_val, (val_prob > 0.5).astype(int))
                else:
                    fold_score = roc_auc_score(y_val, val_prob)

            all_test_preds[seed_i] += test_prob / len(folds)
            fold_scores.append(fold_score)
            print(f'    Fold {fold_i+1}: best_iter={best_iter}, {metric}={fold_score:.6f}')
            del model, train_pool, val_pool

        # --- Calibration for probability outputs ---
        if output_type == 'prob' and config['type'] == 'binary':
            oof_cal, test_cal = calibrate_proba(oof_preds, y.values, all_test_preds[seed_i])
            oof_preds_calibrated = oof_cal
            all_test_preds[seed_i] = test_cal
            cal_score = roc_auc_score(y.values, oof_preds_calibrated)
            print(f'    After calibration AUC: {cal_score:.6f} (was: {np.mean(fold_scores):.6f})')

        mean_fold = np.mean(fold_scores)
        all_oof_preds[seed_i] = oof_preds
        all_seed_scores[seed] = {'fold_scores': fold_scores, 'mean': mean_fold}
        print(f'    Seed {seed} mean OOF {metric}: {mean_fold:.6f}')

    # --- Ensemble across seeds ---
    if config['type'] == 'multiclass':
        test_preds_avg = all_test_preds.mean(axis=0)
        final_pred = test_preds_avg.argmax(axis=1).astype(int) + 1  # back to 1-based
    else:
        test_preds_avg = all_test_preds.mean(axis=0)

    # Print seed ensemble stats
    print(f'\n  === Multi-Seed Ensemble Summary ===')
    for seed, sc in all_seed_scores.items():
        print(f'    Seed {seed}: mean={sc["mean"]:.6f} folds={[f"{s:.6f}" for s in sc["fold_scores"]]}')

    # --- Build submission ---
    sub = pd.DataFrame()
    sub[id_col] = test_ids

    if config['type'] == 'multiclass':
        sub[target_col] = final_pred.astype(int)
    elif output_type == 'bool':
        sub[target_col] = (test_preds_avg > 0.5).astype(bool)
    elif output_type == 'prob':
        sub[target_col] = np.clip(test_preds_avg, 0.001, 0.999)
    else:
        sub[target_col] = (test_preds_avg > 0.5).astype(int)

    out_path = f'{RESULTS}/{comp_id}/submission_s44.csv'
    sub.to_csv(out_path, index=False)
    print(f'  Saved: {out_path} ({sub.shape})')
    print(f'  Preview:\n{sub.head(3).to_string()}')

    return {
        'competition': comp_id,
        'seed_scores': all_seed_scores,
        'seed_ensemble_mean': float(np.mean([s['mean'] for s in all_seed_scores.values()])),
        'path': out_path
    }


def main():
    print('=' * 60)
    print('MLE-Bench CatBoost GPU Trainer v2 — Systematic Optimization')
    print(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Seeds: {SEEDS}')
    print('=' * 60)

    all_results = []
    # Train all three; order: longest first for time efficiency
    for comp_id in ['tabular-playground-series-dec-2021', 'tabular-playground-series-may-2022', 'spaceship-titanic']:
        config = COMPETITIONS[comp_id]
        print(f'\n{"=" * 60}')
        print(f'  {comp_id} ({config["type"]}, {config["metric"]}, group_kfold={config.get("use_group_kfold", False)})')
        print(f'{"=" * 60}')
        try:
            result = train_one(comp_id, config, SEEDS)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f'  ERROR: {e}')
            import traceback
            traceback.print_exc()

    print(f'\n{"=" * 60}')
    print('  FINAL SUMMARY')
    print(f'{"=" * 60}')
    for r in all_results:
        print(f'  {r["competition"]:<45s} Ensemble OOF={r["seed_ensemble_mean"]:.6f}')

    with open(f'{RESULTS}/training_results_v2.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nDone. Results saved to {RESULTS}/training_results_v2.json')


if __name__ == '__main__':
    main()
