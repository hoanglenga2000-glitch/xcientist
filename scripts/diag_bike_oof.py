"""Diagnose bike_sharing OOF RMSLE discrepancy: per-fold ~0.2 vs global 1.67"""
import pandas as pd, numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_train_v3 import compute_oof_metric, load_and_preprocess

data_dir = '/hpc2hdd/home/aimslab/bike_sharing_demand'
train = pd.read_csv(f'{data_dir}/train.csv')
y = train['count']
print(f'Train shape: {train.shape}')
print(f'Index type: {train.index.dtype}, monotonic: {train.index.is_monotonic_increasing}')
print(f'Index range: [{train.index.min()}, {train.index.max()}]')
print(f'Count: min={y.min()}, max={y.max()}, mean={y.mean():.1f}')
print(f'y has NaN: {y.isna().sum()}')

# Load result
with open('/hpc2hdd/home/aimslab/results/v3_result_bike_sharing_demand.json') as f:
    r = json.load(f)
print(f'\nCV scores: {r["cv_scores"]}')
print(f'OOF score: {r["oof_score"]}')

# Load the actual data through load_and_preprocess to get y used during training
result = load_and_preprocess('bike_sharing_demand')
if result:
    X_train, y_train, X_test, task_type, metric, direction, bronze, margin, test_ids, id_col_out, pred_col_out, val_fmt, target_encoder, cat_features = result
    print(f'\nLoaded via load_and_preprocess:')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_train index type: {y_train.index.dtype}')
    print(f'y_train index range: [{y_train.index.min()}, {y_train.index.max()}]')
    print(f'y_train head indices: {y_train.index[:5].tolist()}')
    print(f'y_train tail indices: {y_train.index[-5:].tolist()}')

    # Simulate 5-fold TimeSeriesSplit and check oof assembly
    from sklearn.model_selection import TimeSeriesSplit
    folds = list(TimeSeriesSplit(n_splits=5).split(X_train))

    # Simulate oof like training does
    oof = np.zeros(len(X_train))
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        # val_idx from TimeSeriesSplit
        oof[val_idx] = 1.0  # dummy values to see coverage
        if fold_idx == 0:
            print(f'\nFold 0 val_idx: {val_idx[:5]}...{val_idx[-5:]}, n={len(val_idx)}')

    uncovered = np.where(oof == 0)[0]
    print(f'\nUncovered oof indices: {len(uncovered)} (should be 0)')
    if len(uncovered) > 0:
        print(f'First 10 uncovered: {uncovered[:10]}')

    # Now do the actual computation the way train_catboost does it
    oof_real = np.zeros(len(X_train))
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        # Simulate: each val prediction equals the true value → perfect RMSLE=0
        y_val = y_train.iloc[val_idx].values
        oof_real[val_idx] = y_val  # perfect prediction

    # Check alignment: compute RMSLE on perfect predictions
    global_rmsle = compute_oof_metric(y_train, oof_real, 'rmsle', 'regression')
    print(f'\nGlobal RMSLE with PERFECT predictions: {global_rmsle:.6f} (should be 0.0)')

    # Also check if oof alignment matches y
    for i in [0, 100, 5000, -1]:
        print(f'  y[{i}]={y_train.iloc[i]:.1f}, oof[{i}]={oof_real[i]:.1f}')
