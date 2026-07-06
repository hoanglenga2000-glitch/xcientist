#!/usr/bin/env python3
"""
MLE-Bench CatBoost GPU Trainer v3 — 7 competitions.
Adds: playground-series-s3e18, leaf-classification, taxi-fare-prediction, nomad2018.
Supports: binary, multiclass, regression, multi-label, multi-target regression.
"""
import sys, os, time, warnings, json
os.environ['PYTHONUNBUFFERED'] = '1'
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

HOME = '/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra'
PREPARED = f'{HOME}/mlebench_prepared'
RESULTS = f'{HOME}/mlebench_proper_results'

SEEDS = [42, 123, 256, 789, 1024]

# ============================================================
# Competition configs — 7 competitions
# ============================================================
COMPETITIONS = {
    # --- Original 3 ---
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
            'iterations': 5000, 'depth': 8, 'learning_rate': 0.02,
            'l2_leaf_reg': 3, 'border_count': 254, 'random_strength': 1.0,
            'bagging_temperature': 0.5, 'min_data_in_leaf': 20,
            'grow_policy': 'Lossguide', 'gpu_ram_part': 0.75,
            'task_type': 'GPU', 'devices': '0', 'verbose': 500,
            'random_seed': 42, 'eval_metric': 'Accuracy',
            'loss_function': 'Logloss', 'early_stopping_rounds': 300,
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
            'iterations': 8000, 'depth': 10, 'learning_rate': 0.02,
            'l2_leaf_reg': 5, 'border_count': 254, 'random_strength': 1.0,
            'bagging_temperature': 0.5, 'min_data_in_leaf': 20,
            'grow_policy': 'Lossguide', 'gpu_ram_part': 0.70,
            'task_type': 'GPU', 'devices': '0', 'verbose': 500,
            'random_seed': 42, 'eval_metric': 'Accuracy',
            'loss_function': 'MultiClass', 'early_stopping_rounds': 400,
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
            'iterations': 10000, 'depth': 9, 'learning_rate': 0.005,
            'l2_leaf_reg': 0.5, 'border_count': 254, 'random_strength': 0.5,
            'bagging_temperature': 0.5,
            'grow_policy': 'Lossguide',
            'gpu_ram_part': 0.75, 'min_data_in_leaf': 10,
            'task_type': 'GPU', 'devices': '0', 'verbose': 500,
            'random_seed': 42, 'eval_metric': 'AUC',
            'loss_function': 'Logloss', 'early_stopping_rounds': 500,
        }
    },

    # --- New 1: Multi-label classification (playground-series-s3e18) ---
    'playground-series-s3e18': {
        'target': 'EC1',          # Primary target; EC2 trained separately
        'targets': ['EC1', 'EC2'],  # Both scored targets
        'id_col': 'id',
        'type': 'multilabel',
        'metric': 'auc_roc',
        'output': 'prob',
        'n_splits': 5,
        'use_group_kfold': False,
        'group_col': None,
        'catboost_params': {
            'iterations': 4000, 'depth': 7, 'learning_rate': 0.02,
            'l2_leaf_reg': 3, 'border_count': 254, 'random_strength': 1.0,
            'bagging_temperature': 0.5, 'min_data_in_leaf': 20,
            'grow_policy': 'Lossguide', 'gpu_ram_part': 0.75,
            'task_type': 'GPU', 'devices': '0', 'verbose': 500,
            'random_seed': 42, 'eval_metric': 'AUC',
            'loss_function': 'Logloss', 'early_stopping_rounds': 300,
        }
    },

    # --- New 2: Multiclass 99-class (leaf-classification) ---
    'leaf-classification': {
        'target': 'species',
        'id_col': 'id',
        'type': 'multiclass',
        'metric': 'accuracy',
        'output': 'class_prob',
        'n_splits': 5,
        'use_group_kfold': False,
        'group_col': None,
        'catboost_params': {
            'iterations': 8000, 'depth': 10, 'learning_rate': 0.015,
            'l2_leaf_reg': 3, 'border_count': 254, 'random_strength': 1.0,
            'bagging_temperature': 0.5, 'min_data_in_leaf': 10,
            'grow_policy': 'Lossguide', 'gpu_ram_part': 0.70,
            'task_type': 'GPU', 'devices': '0', 'verbose': 500,
            'random_seed': 42, 'eval_metric': 'Accuracy',
            'loss_function': 'MultiClass', 'early_stopping_rounds': 500,
        }
    },

    # --- New 3: Regression (taxi-fare-prediction) ---
    'new-york-city-taxi-fare-prediction': {
        'target': 'fare_amount',
        'id_col': 'key',
        'type': 'regression',
        'metric': 'rmse',
        'output': 'value',
        'n_splits': 5,
        'use_group_kfold': False,
        'group_col': None,
        'catboost_params': {
            'iterations': 8000, 'depth': 10, 'learning_rate': 0.02,
            'l2_leaf_reg': 5, 'border_count': 254,
            'bagging_temperature': 0.5, 'min_data_in_leaf': 10,
            'grow_policy': 'Lossguide', 'gpu_ram_part': 0.70,
            'task_type': 'GPU', 'devices': '0', 'verbose': 500,
            'random_seed': 42, 'eval_metric': 'RMSE',
            'loss_function': 'RMSE', 'early_stopping_rounds': 400,
        }
    },

    # --- New 4: Multi-target regression (nomad2018) ---
    'nomad2018-predict-transparent-conductors': {
        'target': 'formation_energy_ev_natom',
        'targets': ['formation_energy_ev_natom', 'bandgap_energy_ev'],
        'id_col': 'id',
        'type': 'multireg',
        'metric': 'rmse',
        'output': 'value',
        'n_splits': 5,
        'use_group_kfold': False,
        'group_col': None,
        'catboost_params': {
            'iterations': 5000, 'depth': 8, 'learning_rate': 0.02,
            'l2_leaf_reg': 3, 'border_count': 254,
            'bagging_temperature': 0.5, 'min_data_in_leaf': 10,
            'grow_policy': 'Lossguide', 'gpu_ram_part': 0.75,
            'task_type': 'GPU', 'devices': '0', 'verbose': 500,
            'random_seed': 42, 'eval_metric': 'RMSE',
            'loss_function': 'RMSE', 'early_stopping_rounds': 300,
        }
    },
}

# ============================================================
# Feature engineering hooks
# ============================================================

def spaceship_features(train, test):
    for df in [df for df in [train, test] if df is not None]:
        if 'CryoSleep' in df.columns:
            df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(bool)
        spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        if all(c in df.columns for c in spend_cols):
            for c in spend_cols:
                df[c] = df[c].fillna(0)
            if 'CryoSleep' in df.columns:
                cryo_mask = df['CryoSleep'] == True
                for c in spend_cols:
                    df.loc[cryo_mask, c] = 0
            df['TotalSpend'] = df[spend_cols].sum(axis=1)
            df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)
            df['LogTotalSpend'] = np.log1p(df['TotalSpend'])
            df['SpendRatio_RoomService'] = df['RoomService'] / (df['TotalSpend'] + 1e-8)
            df['SpendRatio_ShoppingMall'] = df['ShoppingMall'] / (df['TotalSpend'] + 1e-8)
            df['SpendRatio_VRDeck'] = df['VRDeck'] / (df['TotalSpend'] + 1e-8)
        if 'PassengerId' in df.columns:
            df['Group'] = df['PassengerId'].str.split('_').str[0]
            df['GroupSize'] = df['Group'].map(df['Group'].value_counts())
            df['IsSolo'] = (df['GroupSize'] == 1).astype(int)
        if 'Cabin' in df.columns:
            df['Deck'] = df['Cabin'].str[0].fillna('U')
            df['CabinNum'] = df['Cabin'].str.extract(r'(\d+)', expand=False).fillna(0).astype(int)
            df['CabinSide'] = df['Cabin'].str[-1].fillna('U')
            df['Deck_Side'] = df['Deck'].astype(str) + '_' + df['CabinSide'].astype(str)
            df['CabinBin'] = (df['CabinNum'] // 100 * 100).astype(int)
        if 'Age' in df.columns:
            df['Age'] = df['Age'].fillna(df['Age'].median())
            df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 25, 35, 50, 80], labels=False).fillna(3).astype(int)
            df['IsChild'] = (df['Age'] < 12).astype(int)
            df['IsTeen'] = ((df['Age'] >= 12) & (df['Age'] < 18)).astype(int)
        if 'HomePlanet' in df.columns and 'Destination' in df.columns:
            df['Route'] = df['HomePlanet'].fillna('Unknown').astype(str) + '_' + df['Destination'].fillna('Unknown').astype(str)
        if 'VIP' in df.columns and 'CryoSleep' in df.columns:
            df['VIP'] = df['VIP'].fillna(False).astype(bool)
            df['VIP_Cryo'] = (df['VIP'] & df['CryoSleep']).astype(int)
        for c in ['CryoSleep', 'VIP']:
            if c in df.columns:
                df[c] = df[c].fillna(False).astype(bool)
    return train, test


def dec2021_features(train, test):
    for df in [df for df in [train, test] if df is not None]:
        if 'Aspect' in df.columns:
            df['Aspect_sin'] = np.sin(np.radians(df['Aspect']))
            df['Aspect_cos'] = np.cos(np.radians(df['Aspect']))
        if 'Elevation' in df.columns:
            for dcol in ['Hydrology', 'Roadways', 'Fire_Points']:
                hcol = f'Horizontal_Distance_To_{dcol}'
                if hcol in df.columns:
                    df[f'Euc_Dist_{dcol}'] = np.sqrt(df['Elevation']**2 + df[hcol]**2)
        if 'Hillshade_Noon' in df.columns:
            for shade in ['9am', '3pm']:
                scol = f'Hillshade_{shade}'
                if scol in df.columns:
                    df[f'Hillshade_Ratio_{shade}'] = df['Hillshade_Noon'] / (df[scol] + 1)
            hill_cols = [c for c in ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'] if c in df.columns]
            if len(hill_cols) == 3:
                df['Hillshade_Mean'] = df[hill_cols].mean(axis=1)
                df['Hillshade_Range'] = df[hill_cols].max(axis=1) - df[hill_cols].min(axis=1)
                df['Hillshade_Std'] = df[hill_cols].std(axis=1)
        if 'Elevation' in df.columns:
            df['Elevation_Bin'] = pd.qcut(df['Elevation'], q=20, labels=False, duplicates='drop')
        soil_cols = [c for c in df.columns if c.startswith('Soil_Type')]
        if len(soil_cols) >= 40:
            df['Soil_Count'] = df[soil_cols].sum(axis=1)
            for g_start in range(1, 41, 10):
                g_cols = [f'Soil_Type{i}' for i in range(g_start, g_start+10) if f'Soil_Type{i}' in df.columns]
                if g_cols:
                    df[f'Soil_Group{g_start}_{g_start+9}'] = df[g_cols].sum(axis=1)
        wa_cols = [c for c in df.columns if c.startswith('Wilderness_Area')]
        if len(wa_cols) >= 4:
            df['WA_Count'] = df[wa_cols].sum(axis=1)
            if len(soil_cols) >= 40:
                for wa in wa_cols[:4]:
                    for sg in ['Soil_Group1_10', 'Soil_Group11_20', 'Soil_Group21_30', 'Soil_Group31_40']:
                        if sg in df.columns:
                            df[f'{wa}_{sg}'] = df[wa] * df[sg]
        if 'Elevation' in df.columns and 'Slope' in df.columns:
            df['Elevation_Slope'] = df['Elevation'] * df['Slope']
        if 'Horizontal_Distance_To_Roadways' in df.columns and 'Horizontal_Distance_To_Fire_Points' in df.columns:
            df['Road_Fire_Dist'] = df['Horizontal_Distance_To_Roadways'] + df['Horizontal_Distance_To_Fire_Points']
            df['Road_Fire_Ratio'] = df['Horizontal_Distance_To_Roadways'] / (df['Horizontal_Distance_To_Fire_Points'] + 1)
        if 'Elevation' in df.columns and 'Horizontal_Distance_To_Roadways' in df.columns:
            df['Elevation_Road'] = df['Elevation'] * df['Horizontal_Distance_To_Roadways']
    return train, test


def may2022_features(train, test):
    """Feature engineering for TPS May 2022.

    Key structural insight: f_00-f_14 are generated independently,
    f_15-f_30 are generated WITHIN groups defined by f_27.
    KMeans/PCA are fit on train only and transformed on both to avoid
    non-comparable feature spaces.
    """
    # Ensure numeric types
    for df in [df for df in [train, test] if df is not None]:
        for c in df.columns:
            if c.startswith('f_'):
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    feature_cols = [c for c in train.columns if c.startswith('f_') and c != 'f_27']
    if len(feature_cols) < 27:
        return train, test

    # Feature groups based on data generation structure
    indep_cols = [f'f_{i:02d}' for i in range(15)]    # f_00-f_14: independent
    group_cols = [f'f_{i:02d}' for i in range(15, 27)]  # f_15-f_26: group-dependent

    for df in [df for df in [train, test] if df is not None]:
        # Row-level statistics computed per-feature-group
        if all(c in df.columns for c in indep_cols):
            df['f_indep_sum'] = df[indep_cols].sum(axis=1)
            df['f_indep_mean'] = df[indep_cols].mean(axis=1)
            df['f_indep_std'] = df[indep_cols].std(axis=1)
            df['f_indep_max'] = df[indep_cols].max(axis=1)
            df['f_indep_min'] = df[indep_cols].min(axis=1)
        if all(c in df.columns for c in group_cols):
            df['f_group_sum'] = df[group_cols].sum(axis=1)
            df['f_group_mean'] = df[group_cols].mean(axis=1)
            df['f_group_std'] = df[group_cols].std(axis=1)
            df['f_group_max'] = df[group_cols].max(axis=1)
            df['f_group_min'] = df[group_cols].min(axis=1)
        # Cross-group ratio
        if 'f_indep_mean' in df.columns and 'f_group_mean' in df.columns:
            df['f_indep_group_ratio'] = df['f_indep_mean'] / (df['f_group_mean'] + 1e-8)

        # Overall row statistics
        df['f_mean'] = df[feature_cols].mean(axis=1)
        df['f_std'] = df[feature_cols].std(axis=1)
        df['f_max'] = df[feature_cols].max(axis=1)
        df['f_min'] = df[feature_cols].min(axis=1)
        df['f_skew'] = df[feature_cols].skew(axis=1)
        df['f_kurt'] = df[feature_cols].kurtosis(axis=1)
        df['f_range'] = df['f_max'] - df['f_min']

        # Group (f_27) aggregations — computed per-dataset but safe with GroupKFold
        if 'f_27' in df.columns and not df['f_27'].isna().all():
            group_col = df['f_27']
            for fc in feature_cols[:10]:  # Limit to first 10 to control feature count
                df[f'{fc}_gmean'] = df.groupby(group_col)[fc].transform('mean')
                df[f'{fc}_gstd'] = df.groupby(group_col)[fc].transform('std').fillna(0)

        # Polynomial features on independent cols (most important)
        for i in range(10):
            c = f'f_{i:02d}'
            if c in df.columns:
                df[f'{c}_sq'] = df[c] ** 2
                df[f'{c}_cube'] = df[c] ** 3

        # Interactions among top 10 features
        top_cols = [f'f_{i:02d}' for i in range(10)]
        if all(c in df.columns for c in top_cols):
            for i in range(10):
                for j in range(i+1, 10):
                    ci, cj = f'f_{i:02d}', f'f_{j:02d}'
                    df[f'inter_{i}_{j}'] = df[ci] * df[cj]

    # KMeans and PCA — fit on train, transform on both
    km = KMeans(n_clusters=10, random_state=42, n_init='auto')
    km.fit(train[feature_cols].fillna(0))
    for df in [df for df in [train, test] if df is not None]:
        df['kmeans_cluster'] = km.predict(df[feature_cols].fillna(0))

    pca = PCA(n_components=5, random_state=42)
    pca.fit(train[feature_cols].fillna(0))
    for df in [df for df in [train, test] if df is not None]:
        pca_result = pca.transform(df[feature_cols].fillna(0))
        for i in range(5):
            df[f'pca_{i}'] = pca_result[:, i]

    return train, test


def s3e18_features(train, test):
    """Basic feature engineering for playground-series-s3e18."""
    for df in [df for df in [train, test] if df is not None]:
        feature_cols = [c for c in df.columns if c not in ('id', 'EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6')]
        # Add basic interactions
        if len(feature_cols) >= 10:
            top10 = feature_cols[:10]
            for i in range(5):
                for j in range(i+1, 5):
                    if top10[i] in df.columns and top10[j] in df.columns:
                        df[f'{top10[i]}_{top10[j]}'] = df[top10[i]] * df[top10[j]]
        # Add row statistics
        if feature_cols:
            df['feat_mean'] = df[feature_cols].mean(axis=1)
            df['feat_std'] = df[feature_cols].std(axis=1)
            df['feat_max'] = df[feature_cols].max(axis=1)
            df['feat_min'] = df[feature_cols].min(axis=1)
    return train, test


def leaf_features(train, test):
    """Feature engineering for leaf-classification (192 pre-extracted features)."""
    for df in [df for df in [train, test] if df is not None]:
        feature_cols = [c for c in df.columns if c not in ('id', 'species')]
        # Add ratios and interactions for top features
        margin_cols = [c for c in feature_cols if c.startswith('margin')]
        shape_cols = [c for c in feature_cols if c.startswith('shape')]
        texture_cols = [c for c in feature_cols if c.startswith('texture')]
        for name, cols in [('margin', margin_cols), ('shape', shape_cols), ('texture', texture_cols)]:
            if cols:
                df[f'{name}_mean'] = df[cols].mean(axis=1)
                df[f'{name}_std'] = df[cols].std(axis=1)
                df[f'{name}_max'] = df[cols].max(axis=1)
                df[f'{name}_min'] = df[cols].min(axis=1)
        # Cross-family interactions
        for m_col in margin_cols[:3]:
            for s_col in shape_cols[:3]:
                if m_col in df.columns and s_col in df.columns:
                    df[f'{m_col}_{s_col}'] = df[m_col] * df[s_col]
    return train, test


def taxi_features(train, test):
    """Feature engineering for NYC taxi fare prediction — proper haversine + temporal."""
    for df in [df for df in [train, test] if df is not None]:
        # Datetime features
        if 'pickup_datetime' in df.columns:
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
            df['pickup_hour'] = df['pickup_datetime'].dt.hour
            df['pickup_day'] = df['pickup_datetime'].dt.day
            df['pickup_month'] = df['pickup_datetime'].dt.month
            df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
            df['pickup_year'] = df['pickup_datetime'].dt.year
            df['pickup_dayofyear'] = df['pickup_datetime'].dt.dayofyear
            df['is_weekend'] = (df['pickup_weekday'] >= 5).astype(int)
            df['is_rush_hour'] = ((df['pickup_hour'].between(7, 10)) | (df['pickup_hour'].between(16, 19))).astype(int)
            df['is_night'] = ((df['pickup_hour'] >= 22) | (df['pickup_hour'] < 6)).astype(int)
            df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)
            df['month_sin'] = np.sin(2 * np.pi * df['pickup_month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['pickup_month'] / 12)
        # Proper haversine distance
        if all(c in df.columns for c in ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']):
            lat1, lon1 = np.radians(df['pickup_latitude'].values), np.radians(df['pickup_longitude'].values)
            lat2, lon2 = np.radians(df['dropoff_latitude'].values), np.radians(df['dropoff_longitude'].values)
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            df['haversine_dist'] = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            df['abs_dlon'] = np.abs(np.degrees(dlon))
            df['abs_dlat'] = np.abs(np.degrees(dlat))
            df['log_dist'] = np.log1p(df['haversine_dist'])
            # Distance bins
            df['dist_bin'] = pd.cut(df['haversine_dist'], bins=[0, 1, 3, 5, 10, 20, 50, 1000], labels=False).fillna(0).astype(int)
            # Airport & landmark proximity features
            df['pickup_near_jfk'] = ((np.abs(df['pickup_longitude'] + 73.78) < 0.05) & (np.abs(df['pickup_latitude'] - 40.64) < 0.05)).astype(int)
            df['dropoff_near_jfk'] = ((np.abs(df['dropoff_longitude'] + 73.78) < 0.05) & (np.abs(df['dropoff_latitude'] - 40.64) < 0.05)).astype(int)
            df['pickup_near_manhattan'] = ((np.abs(df['pickup_longitude'] + 73.97) < 0.1) & (np.abs(df['pickup_latitude'] - 40.78) < 0.1)).astype(int)
            df['dropoff_near_manhattan'] = ((np.abs(df['dropoff_longitude'] + 73.97) < 0.1) & (np.abs(df['dropoff_latitude'] - 40.78) < 0.1)).astype(int)
        # Distance-time interactions
        if 'haversine_dist' in df.columns and 'pickup_hour' in df.columns:
            df['dist_x_hour'] = df['haversine_dist'] * df['pickup_hour']
            df['dist_x_rush'] = df['haversine_dist'] * df['is_rush_hour']
            df['dist_x_weekend'] = df['haversine_dist'] * df['is_weekend']
            df['dist_x_night'] = df['haversine_dist'] * df['is_night']
        # Passenger count
        if 'passenger_count' in df.columns:
            df['passenger_count'] = df['passenger_count'].fillna(1).clip(1, 6).astype(int)
        # Drop datetime (CatBoost can't handle it)
        if 'pickup_datetime' in df.columns:
            df.drop(columns=['pickup_datetime'], inplace=True, errors='ignore')
    return train, test


def nomad2018_features(train, test):
    """Feature engineering for nomad2018 materials prediction."""
    for df in [df for df in [train, test] if df is not None]:
        # Add pairwise element interactions
        element_cols = [c for c in df.columns if c.endswith('_fraction') and c not in ('id', 'formation_energy_ev_natom', 'bandgap_energy_ev')]
        if len(element_cols) >= 2:
            # Sum of element fractions
            if element_cols:
                df['sum_fraction'] = df[element_cols].sum(axis=1)
            # Ratios between key elements
            if len(element_cols) >= 3:
                for i in range(min(3, len(element_cols))):
                    for j in range(i+1, min(3, len(element_cols))):
                        denom = df[element_cols[j]] + 1e-8
                        df[f'{element_cols[i]}_{element_cols[j]}_ratio'] = df[element_cols[i]] / denom
        # Numeric features stats
        num_cols = [c for c in df.columns if df[c].dtype in ('float64', 'int64') and c not in ['id'] + element_cols]
        if num_cols:
            df['num_mean'] = df[num_cols].mean(axis=1)
            df['num_std'] = df[num_cols].std(axis=1)
    return train, test


FEATURE_HOOKS = {
    'spaceship-titanic': spaceship_features,
    'tabular-playground-series-dec-2021': dec2021_features,
    'tabular-playground-series-may-2022': may2022_features,
    'playground-series-s3e18': s3e18_features,
    'leaf-classification': leaf_features,
    'new-york-city-taxi-fare-prediction': taxi_features,
    'nomad2018-predict-transparent-conductors': nomad2018_features,
}

# ============================================================
# Training utilities
# ============================================================

def safe_predict_proba(model, X_data, n_classes):
    p = model.predict_proba(X_data)
    if p.shape[1] < n_classes:
        full_p = np.zeros((len(p), n_classes), dtype=p.dtype)
        full_p[:, model.classes_.astype(int)] = p
        return full_p
    return p


def calibrate_proba(oof_preds, y_true, test_preds):
    from sklearn.linear_model import LogisticRegression
    cal = LogisticRegression(C=10, random_state=42)
    cal.fit(oof_preds.reshape(-1, 1), y_true)
    oof_cal = cal.predict_proba(oof_preds.reshape(-1, 1))[:, 1]
    test_cal = cal.predict_proba(test_preds.reshape(-1, 1))[:, 1]
    return oof_cal, test_cal


def prepare_features(train, test, comp_id):
    hook = FEATURE_HOOKS.get(comp_id)
    if hook:
        train, test = hook(train, test)
    return train, test


def setup_categorical(train_feat, test_feat):
    cat_features = []
    has_cat = False
    for i, col in enumerate(train_feat.columns):
        if train_feat[col].dtype == 'object' or train_feat[col].dtype == 'bool':
            has_cat = True
            train_feat[col] = train_feat[col].astype(str)
            test_feat[col] = test_feat[col].astype(str)
            cat_features.append(i)
    if has_cat and len(cat_features) > 0:
        combined = pd.concat([train_feat, test_feat], ignore_index=True)
        for i in cat_features:
            combined.iloc[:, i] = combined.iloc[:, i].astype(str)
        train_feat = combined.iloc[:len(train_feat)].reset_index(drop=True)
        test_feat = combined.iloc[len(train_feat):].reset_index(drop=True)
        del combined
    return train_feat, test_feat, cat_features


def setup_folds(train_feat, y, config):
    n_splits = config['n_splits']
    use_group = config.get('use_group_kfold', False)
    group_col = config.get('group_col', None)
    comp_type = config['type']

    if use_group and group_col:
        groups = train_feat.get(group_col)
        if groups is not None:
            groups = groups.fillna(0).values
            try:
                return list(GroupKFold(n_splits=n_splits).split(train_feat, y, groups=groups)), f'group={group_col}'
            except:
                pass

    if comp_type in ('multiclass',):
        try:
            return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(train_feat, y)), 'stratified'
        except:
            return list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(train_feat)), 'kfold'
    elif comp_type in ('binary', 'multilabel'):
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(train_feat, y)), 'stratified'
    else:
        return list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(train_feat)), 'kfold'


def train_classifier(train_feat, test_feat, y, config, cat_features, seeds):
    """Train CatBoostClassifier with multi-seed, multi-fold CV."""
    n_train = len(train_feat)
    cb_params = config['catboost_params'].copy()
    output_type = config['output']
    metric = config['metric']

    if config['type'] == 'multiclass':
        n_classes = y.nunique()
        cb_params['classes_count'] = n_classes
        all_test_preds = np.zeros((len(seeds), len(test_feat), n_classes), dtype=np.float64)
        all_oof_preds = np.zeros((len(seeds), n_train, n_classes), dtype=np.float64)
    else:
        n_classes = 2
        all_test_preds = np.zeros((len(seeds), len(test_feat)), dtype=np.float64)
        all_oof_preds = np.zeros((len(seeds), n_train), dtype=np.float64)

    folds, fold_desc = setup_folds(train_feat, y, config)
    print(f'  CV: {fold_desc}, {len(folds)} folds')

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
            X_tr = train_feat.iloc[train_idx]
            X_val = train_feat.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = CatBoostClassifier(**cb_params)
            train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
            val_pool = Pool(X_val, y_val, cat_features=cat_features)

            model.fit(train_pool, eval_set=val_pool, verbose=cb_params.get('verbose', 500))
            best_iter = model.get_best_iteration()

            if config['type'] == 'multiclass':
                val_prob = safe_predict_proba(model, X_val, n_classes)
                oof_preds[val_idx] = val_prob
                test_prob = safe_predict_proba(model, test_feat, n_classes)
                fold_score = accuracy_score(y_val, val_prob.argmax(axis=1))
            else:
                val_prob = model.predict_proba(X_val)[:, 1]
                if output_type == 'prob':
                    oof_preds[val_idx] = val_prob
                else:
                    oof_preds[val_idx] = (val_prob > 0.5).astype(int)
                test_prob = model.predict_proba(test_feat)[:, 1]
                fold_score = accuracy_score(y_val, (val_prob > 0.5).astype(int)) if metric == 'accuracy' else roc_auc_score(y_val, val_prob)

            all_test_preds[seed_i] += test_prob / len(folds)
            fold_scores.append(fold_score)
            print(f'    Fold {fold_i+1}: best_iter={best_iter}, {metric}={fold_score:.6f}')
            del model, train_pool, val_pool

        if output_type == 'prob' and config['type'] == 'binary':
            oof_cal, test_cal = calibrate_proba(oof_preds, y.values, all_test_preds[seed_i])
            all_test_preds[seed_i] = test_cal
            cal_score = roc_auc_score(y.values, oof_cal)
            print(f'    After calibration AUC: {cal_score:.6f} (was: {np.mean(fold_scores):.6f})')

        mean_fold = np.mean(fold_scores)
        all_oof_preds[seed_i] = oof_preds
        all_seed_scores[seed] = {'fold_scores': fold_scores, 'mean': mean_fold}
        print(f'    Seed {seed} mean OOF {metric}: {mean_fold:.6f}')

    # Ensemble across seeds
    if config['type'] == 'multiclass':
        test_preds_avg = all_test_preds.mean(axis=0)
        final_pred = test_preds_avg.argmax(axis=1).astype(int) + 1
    else:
        test_preds_avg = all_test_preds.mean(axis=0)

    return test_preds_avg, all_seed_scores, (final_pred if config['type'] == 'multiclass' else None)


def train_regressor(train_feat, test_feat, y, config, cat_features, seeds):
    """Train CatBoostRegressor with multi-seed, multi-fold CV."""
    n_train = len(train_feat)
    cb_params = config['catboost_params'].copy()
    all_test_preds = np.zeros((len(seeds), len(test_feat)), dtype=np.float64)
    all_oof_preds = np.zeros((len(seeds), n_train), dtype=np.float64)

    folds, fold_desc = setup_folds(train_feat, y, config)
    print(f'  CV: {fold_desc}, {len(folds)} folds')

    all_seed_scores = {}
    for seed_i, seed in enumerate(seeds):
        print(f'\n  --- Seed {seed} ({seed_i+1}/{len(seeds)}) ---')
        cb_params['random_seed'] = seed
        oof_preds = np.zeros(n_train, dtype=np.float64)
        fold_scores = []

        for fold_i, (train_idx, val_idx) in enumerate(folds):
            X_tr = train_feat.iloc[train_idx]
            X_val = train_feat.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = CatBoostRegressor(**cb_params)
            train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
            val_pool = Pool(X_val, y_val, cat_features=cat_features)

            model.fit(train_pool, eval_set=val_pool, verbose=cb_params.get('verbose', 500))
            best_iter = model.get_best_iteration()

            val_pred = model.predict(X_val)
            oof_preds[val_idx] = val_pred
            test_pred = model.predict(test_feat)
            all_test_preds[seed_i] += test_pred / len(folds)

            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            fold_scores.append(rmse)
            print(f'    Fold {fold_i+1}: best_iter={best_iter}, rmse={rmse:.6f}')
            del model, train_pool, val_pool

        mean_fold = np.mean(fold_scores)
        all_oof_preds[seed_i] = oof_preds
        all_seed_scores[seed] = {'fold_scores': fold_scores, 'mean': mean_fold}
        print(f'    Seed {seed} mean OOF rmse: {mean_fold:.6f}')

    test_preds_avg = all_test_preds.mean(axis=0)
    return test_preds_avg, all_seed_scores


# ============================================================
# Training pipeline
# ============================================================

def train_one(comp_id, config, seeds):
    data_dir = f'{PREPARED}/{comp_id}'
    train_path = f'{data_dir}/train.csv'
    test_path = f'{data_dir}/test.csv'
    os.makedirs(f'{RESULTS}/{comp_id}', exist_ok=True)

    if not os.path.exists(train_path):
        print(f'  SKIP: {train_path} not found')
        return None

    comp_type = config['type']

    # --- Multi-label: train each target separately ---
    if comp_type == 'multilabel':
        return train_multilabel(comp_id, config, seeds)

    # --- Multi-target regression: train each target separately ---
    if comp_type == 'multireg':
        return train_multitarget_reg(comp_id, config, seeds)

    # --- Single target ---
    print(f'\n  Loading {train_path}...')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f'  Train: {train.shape}, Test: {test.shape}')

    train, test = prepare_features(train, test, comp_id)

    target_col = config['target']
    id_col = config['id_col']
    output_type = config['output']
    metric = config['metric']

    test_ids = test[id_col].values if id_col in test.columns else None
    y = train[target_col].copy()
    if y.dtype == 'bool':
        y = y.astype(int)
    if y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f'  Encoded string labels ({len(le.classes_)} classes)')
    if comp_type == 'multiclass' and y.min() > 0:
        y = y - y.min()

    drop_cols = [target_col, id_col, 'sample_weight']
    feat_drop = [c for c in drop_cols if c in train.columns]
    train_feat = train.drop(columns=feat_drop, errors='ignore')
    test_feat = test.drop(columns=[c for c in feat_drop if c in test.columns], errors='ignore')

    # Also drop multiclass prob target column names from features (for leaf-classification)
    if comp_type == 'multiclass' and output_type == 'class_prob':
        extra_drop = [c for c in test_feat.columns if c not in train_feat.columns and c != id_col]
        test_feat = test_feat.drop(columns=[c for c in extra_drop if c in test_feat.columns], errors='ignore')

    train_feat, test_feat, cat_features = setup_categorical(train_feat, test_feat)
    n_train = len(train_feat)
    print(f'  Features: {train_feat.shape[1]}, Categorical: {len(cat_features)}')
    print(f'  Multi-seed: {len(seeds)} seeds: {seeds}')

    # --- Train ---
    if comp_type == 'regression':
        test_preds_avg, all_seed_scores = train_regressor(train_feat, test_feat, y, config, cat_features, seeds)
    else:
        test_preds_avg, all_seed_scores, multiclass_final = train_classifier(train_feat, test_feat, y, config, cat_features, seeds)

    # --- Summary ---
    print(f'\n  === Multi-Seed Ensemble Summary ===')
    for seed, sc in all_seed_scores.items():
        print(f'    Seed {seed}: mean={sc["mean"]:.6f} folds={[f"{s:.6f}" for s in sc["fold_scores"]]}')

    # --- Build submission ---
    sub = pd.DataFrame()
    sub[id_col] = test_ids

    if comp_type == 'multiclass':
        sub[target_col] = multiclass_final.astype(int)
    elif output_type == 'bool':
        sub[target_col] = (test_preds_avg > 0.5).astype(bool)
    elif output_type == 'prob':
        sub[target_col] = np.clip(test_preds_avg, 0.001, 0.999)
    else:
        sub[target_col] = np.clip(test_preds_avg, 0, 9999)

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


def train_multilabel(comp_id, config, seeds):
    """Train multi-label: separate binary classifier for each target."""
    target_cols = config['targets']
    id_col = config['id_col']
    data_dir = f'{PREPARED}/{comp_id}'

    print(f'\n  Loading {data_dir}/train.csv...')
    train = pd.read_csv(f'{data_dir}/train.csv')
    test = pd.read_csv(f'{data_dir}/test.csv')
    print(f'  Train: {train.shape}, Test: {test.shape}')

    train, test = prepare_features(train, test, comp_id)
    test_ids = test[id_col].values if id_col in test.columns else None

    all_target_preds = {}
    all_results = {}

    for tcol in target_cols:
        print(f'\n  {"="*50}')
        print(f'  Training target: {tcol}')
        print(f'  {"="*50}')

        y = train[tcol].copy()
        if y.dtype == 'bool':
            y = y.astype(int)

        drop_cols = [id_col, 'sample_weight'] + target_cols + [c for c in train.columns if c.startswith('EC') and c not in target_cols]
        feat_drop = [c for c in drop_cols if c in train.columns and c != tcol]
        train_feat = train.drop(columns=[c for c in feat_drop if c != tcol], errors='ignore')
        test_feat = test.drop(columns=[c for c in feat_drop if c in test.columns], errors='ignore')

        train_feat, test_feat, cat_features = setup_categorical(train_feat, test_feat)

        # Use binary classifier config
        binary_config = config.copy()
        binary_config['type'] = 'binary'
        binary_config['target'] = tcol

        test_preds_avg, all_seed_scores, _ = train_classifier(train_feat, test_feat, y, binary_config, cat_features, seeds)
        all_target_preds[tcol] = test_preds_avg
        all_results[tcol] = float(np.mean([s['mean'] for s in all_seed_scores.values()]))

    # Build multi-label submission
    sub = pd.DataFrame()
    sub[id_col] = test_ids
    for tcol in target_cols:
        sub[tcol] = np.clip(all_target_preds[tcol], 0.001, 0.999)

    out_path = f'{RESULTS}/{comp_id}/submission_s44.csv'
    sub.to_csv(out_path, index=False)
    print(f'\n  Multi-label Saved: {out_path} ({sub.shape})')

    return {
        'competition': comp_id,
        'target_results': all_results,
        'seed_ensemble_mean': float(np.mean(list(all_results.values()))),
        'path': out_path
    }


def train_multitarget_reg(comp_id, config, seeds):
    """Train multi-target regression: separate regressor for each target."""
    target_cols = config['targets']
    id_col = config['id_col']
    data_dir = f'{PREPARED}/{comp_id}'

    print(f'\n  Loading {data_dir}/train.csv...')
    train = pd.read_csv(f'{data_dir}/train.csv')
    test = pd.read_csv(f'{data_dir}/test.csv')
    print(f'  Train: {train.shape}, Test: {test.shape}')

    train, test = prepare_features(train, test, comp_id)
    test_ids = test[id_col].values if id_col in test.columns else None

    all_target_preds = {}
    all_results = {}

    for tcol in target_cols:
        print(f'\n  {"="*50}')
        print(f'  Training target: {tcol}')
        print(f'  {"="*50}')

        y = train[tcol].copy()
        drop_cols = [id_col, 'sample_weight'] + [c for c in target_cols if c != tcol]
        feat_drop = [c for c in drop_cols if c in train.columns]
        train_feat = train.drop(columns=[c for c in feat_drop if c != tcol], errors='ignore')
        test_feat = test.drop(columns=[c for c in feat_drop if c in test.columns], errors='ignore')

        train_feat, test_feat, cat_features = setup_categorical(train_feat, test_feat)

        reg_config = config.copy()
        reg_config['type'] = 'regression'
        reg_config['target'] = tcol

        test_preds_avg, all_seed_scores = train_regressor(train_feat, test_feat, y, reg_config, cat_features, seeds)
        all_target_preds[tcol] = test_preds_avg
        all_results[tcol] = float(np.mean([s['mean'] for s in all_seed_scores.values()]))

    # Build submission
    sub = pd.DataFrame()
    sub[id_col] = test_ids
    for tcol in target_cols:
        sub[tcol] = np.clip(all_target_preds[tcol], 0, 9999)

    out_path = f'{RESULTS}/{comp_id}/submission_s44.csv'
    sub.to_csv(out_path, index=False)
    print(f'\n  Multi-target Saved: {out_path} ({sub.shape})')

    return {
        'competition': comp_id,
        'target_results': all_results,
        'seed_ensemble_mean': float(np.mean(list(all_results.values()))),
        'path': out_path
    }


# ============================================================
def main():
    print('=' * 60)
    print('MLE-Bench CatBoost GPU Trainer v3 — 7 Competitions')
    print(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Seeds: {SEEDS}')
    print('=' * 60)

    all_results = []
    # Order: longest first, then new competitions
    # Skip the 3 already-trained ones if running just the new 4
    comp_order = [
        'tabular-playground-series-dec-2021',
        'tabular-playground-series-may-2022',
        'spaceship-titanic',
        # New competitions:
        'playground-series-s3e18',
        'leaf-classification',
        'new-york-city-taxi-fare-prediction',
        'nomad2018-predict-transparent-conductors',
    ]

    for comp_id in comp_order:
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
        if 'target_results' in r:
            targets_str = ', '.join(f'{k}={v:.6f}' for k, v in r['target_results'].items())
            print(f'  {r["competition"]:<45s} [{targets_str}]')
        else:
            print(f'  {r["competition"]:<45s} Ensemble OOF={r["seed_ensemble_mean"]:.6f}')

    with open(f'{RESULTS}/training_results_v3.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nDone. Results saved to {RESULTS}/training_results_v3.json')


if __name__ == '__main__':
    main()
