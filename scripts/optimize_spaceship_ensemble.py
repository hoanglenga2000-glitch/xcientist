"""Optimize spaceship_titanic ensemble weights using OOF predictions."""
import json, csv, os, numpy as np
from datetime import datetime

exp_dir = 'D:/桌面/codex/科研港科技/experiments/spaceship_titanic/20260625_193537'

# Read metrics
with open(f'{exp_dir}/metrics.json') as f:
    m = json.load(f)

# Read train labels
import pandas as pd
train = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/train.csv')
y_true = (train['Transported'] == True).astype(int).values

# Read OOF predictions
with open(f'{exp_dir}/oof_predictions.csv') as f:
    reader = csv.DictReader(f)
    oof_data = list(reader)

oof_cols = [c for c in oof_data[0].keys() if c not in ('PassengerId','Transported','actual','True','False')]
print(f'OOF columns: {oof_cols}')

oof_preds = {}
for col in oof_cols:
    try:
        oof_preds[col] = np.array([float(row[col]) for row in oof_data])
        if len(oof_preds[col]) == len(y_true):
            acc = ((oof_preds[col] > 0.5) == y_true).mean()
            print(f'  {col}: OOF acc={acc:.6f}')
    except:
        pass

# Compute optimal blend weights
from scipy.optimize import minimize

def blend_score(weights, preds_dict, cols, y):
    w = np.maximum(np.array(weights), 0)
    w = w / (w.sum() + 1e-10)
    blend = sum(w[i] * preds_dict[cols[i]] for i in range(len(cols)))
    return -((blend > 0.5) == y).mean()

best_acc = 0
best_w = None
usable_cols = [c for c in oof_cols if len(oof_preds.get(c, [])) == len(y_true)]
print(f'\nUsable OOF columns: {usable_cols}')

for seed in range(30):
    np.random.seed(seed)
    w0 = np.random.dirichlet(np.ones(len(usable_cols)))
    try:
        result = minimize(blend_score, w0, args=(oof_preds, usable_cols, y_true),
                         method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        w = np.maximum(result.x, 0)
        w = w / w.sum()
        acc = -result.fun
        if acc > best_acc:
            best_acc = acc
            best_w = w
    except Exception as e:
        pass

print(f'\nOptimized blend weights:')
if best_w is not None:
    for col, w in zip(usable_cols, best_w):
        if w > 0.01:
            print(f'  {col}: {w:.4f}')
    print(f'  OOF accuracy: {best_acc:.6f}')

# Also try simple grid search
print('\nGrid search over weight combinations:')
best_grid_acc = 0
best_grid_weights = None
for hgb_w in np.arange(0.5, 1.01, 0.05):
    rem = 1.0 - hgb_w
    for gbc_w in np.arange(0, rem + 0.01, 0.05):
        rf_w = rem - gbc_w
        et_w = 0
        w = {'hgb': hgb_w, 'gbc': gbc_w, 'rf': rf_w, 'et': et_w}
        blend = np.zeros(len(y_true))
        for m, wt in w.items():
            key = [c for c in usable_cols if m in c.lower()]
            if key and wt > 0:
                blend += wt * oof_preds[key[0]]
        acc = ((blend > 0.5) == y_true).mean()
        if acc > best_grid_acc:
            best_grid_acc = acc
            best_grid_weights = dict(w)

print(f'Grid search best: acc={best_grid_acc:.6f}')
print(f'Weights: {best_grid_weights}')

# Compare with current
current_acc = m['ensemble']['blend']['accuracy']
print(f'\nCurrent ensemble: {current_acc:.6f}')
print(f'Improvement: {best_grid_acc - current_acc:.6f}')

# Generate optimized submission using original test predictions
# We need per-model test predictions to reblend
# Read the submission to get original ensemble prediction
with open(f'{exp_dir}/submission.csv') as f:
    reader = csv.reader(f)
    sub_headers = next(reader)
    sub_rows = list(reader)

# Since we don't have per-model test preds, compute them from OOF models
# For now, report findings
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'D:/桌面/codex/科研港科技/experiments/spaceship_titanic/reblend_{ts}'
os.makedirs(output_dir, exist_ok=True)

# Save analysis
with open(f'{output_dir}/blend_analysis.json', 'w') as f:
    json.dump({
        'current_weights': m['ensemble']['blend']['weights'],
        'current_accuracy': current_acc,
        'optimized_weights': {c: float(w) for c, w in zip(usable_cols, best_w)} if best_w is not None else None,
        'optimized_oof_accuracy': float(best_acc),
        'grid_best_weights': best_grid_weights,
        'grid_best_accuracy': float(best_grid_acc),
        'potential_improvement': float(best_grid_acc - current_acc),
        'recommendation': 'Use HGB-dominated blend (0.7-0.9 HGB + 0.1-0.3 GBC). RF and ET drag score down.'
    }, f, indent=2)

print(f'\nSaved to {output_dir}/blend_analysis.json')
