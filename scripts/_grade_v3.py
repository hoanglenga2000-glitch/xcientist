"""Grade V3 submissions against test_private labels for all 7 competitions."""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from pathlib import Path

PREPARED = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_prepared")
RESULTS = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_proper_results")

MEDAL_THRESHOLDS = {
    'spaceship-titanic': {
        'metric': 'accuracy',
        'gold': 0.82066, 'silver': 0.81388, 'bronze': 0.80967,
    },
    'tabular-playground-series-dec-2021': {
        'metric': 'accuracy',
        'gold': 0.95660, 'silver': 0.95658, 'bronze': 0.95658,
    },
    'tabular-playground-series-may-2022': {
        'metric': 'auc_roc',
        'gold': 0.79500, 'silver': 0.79000, 'bronze': 0.78500,
    },
    # New competitions — thresholds from mle-bench
    'leaf-classification': {
        'metric': 'accuracy',
        'gold': 0.98800, 'silver': 0.94500, 'bronze': 0.90000,
    },
    'new-york-city-taxi-fare-prediction': {
        'metric': 'rmse',
        'gold': 2.80, 'silver': 2.90, 'bronze': 3.00,
    },
    'nomad2018-predict-transparent-conductors': {
        'metric': 'rmse',
        'gold': 0.15, 'silver': 0.20, 'bronze': 0.25,
    },
    'playground-series-s3e18': {
        'metric': 'auc_roc',
        'gold': 0.96000, 'silver': 0.94000, 'bronze': 0.92000,
    },
}

def get_medal(score, thresholds, higher_is_better=True):
    """Determine medal from score. higher_is_better: True for accuracy/AUC, False for RMSE."""
    if higher_is_better:
        for medal in ['gold', 'silver', 'bronze']:
            if score >= thresholds[medal]:
                return medal.upper()
    else:
        for medal in ['gold', 'silver', 'bronze']:
            if score <= thresholds[medal]:
                return medal.upper()
    return 'NO MEDAL'

def grade_binary_accuracy(comp_name, id_col, target_col):
    """Grade binary classification by accuracy."""
    sub_path = RESULTS / comp_name / "submission_s44.csv"
    priv_path = PREPARED / comp_name / "test_private.csv"
    if not sub_path.exists():
        return None, "submission not found"
    if not priv_path.exists():
        return None, "test_private.csv not found"
    sub = pd.read_csv(sub_path)
    priv = pd.read_csv(priv_path)
    # Handle bool conversion
    if sub[target_col].dtype == 'object':
        sub[target_col] = sub[target_col].map({"True": True, "False": False}).astype(bool)
    if priv[target_col].dtype == 'object':
        priv[target_col] = priv[target_col].map({"True": True, "False": False}).astype(bool)
    m = sub.merge(priv[[id_col, target_col]], on=id_col, suffixes=("_pred", "_true"))
    score = accuracy_score(m[f"{target_col}_true"], m[f"{target_col}_pred"])
    return score, None

def grade_multiclass_accuracy(comp_name, id_col, target_col):
    """Grade multiclass by accuracy (label-aligned)."""
    sub_path = RESULTS / comp_name / "submission_s44.csv"
    priv_path = PREPARED / comp_name / "test_private.csv"
    if not sub_path.exists():
        return None, "submission not found"
    if not priv_path.exists():
        return None, "test_private.csv not found"
    sub = pd.read_csv(sub_path)
    priv = pd.read_csv(priv_path)
    m = sub.merge(priv[[id_col, target_col]], on=id_col, suffixes=("_pred", "_true"))
    # Align labels
    y_pred = m[f"{target_col}_pred"]
    y_true = m[f"{target_col}_true"]
    if y_true.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_true = le.fit_transform(y_true)
        y_pred = le.transform(y_pred)
    score = accuracy_score(y_true, y_pred)
    return score, None

def grade_auc(comp_name, id_col, target_col):
    """Grade binary classification by AUC."""
    sub_path = RESULTS / comp_name / "submission_s44.csv"
    priv_path = PREPARED / comp_name / "test_private.csv"
    if not sub_path.exists():
        return None, "submission not found"
    if not priv_path.exists():
        return None, "test_private.csv not found"
    sub = pd.read_csv(sub_path)
    priv = pd.read_csv(priv_path)
    m = sub.merge(priv[[id_col, target_col]], on=id_col, suffixes=("_pred", "_true"))
    score = roc_auc_score(m[f"{target_col}_true"], m[f"{target_col}_pred"])
    return score, None

def grade_rmse(comp_name, id_col, target_col):
    """Grade regression by RMSE."""
    sub_path = RESULTS / comp_name / "submission_s44.csv"
    priv_path = PREPARED / comp_name / "test_private.csv"
    if not sub_path.exists():
        return None, "submission not found"
    if not priv_path.exists():
        return None, "test_private.csv not found"
    sub = pd.read_csv(sub_path)
    priv = pd.read_csv(priv_path)
    m = sub.merge(priv[[id_col, target_col]], on=id_col, suffixes=("_pred", "_true"))
    score = np.sqrt(mean_squared_error(m[f"{target_col}_true"], m[f"{target_col}_pred"]))
    return score, None

def grade_multitarget_rmse(comp_name, id_col, target_cols):
    """Grade multi-target regression by mean RMSE."""
    sub_path = RESULTS / comp_name / "submission_s44.csv"
    priv_path = PREPARED / comp_name / "test_private.csv"
    if not sub_path.exists():
        return None, "submission not found"
    if not priv_path.exists():
        return None, "test_private.csv not found"
    sub = pd.read_csv(sub_path)
    priv = pd.read_csv(priv_path)
    m = sub.merge(priv[[id_col] + target_cols], on=id_col, suffixes=("_pred", "_true"))
    rmses = {}
    for tc in target_cols:
        rmses[tc] = np.sqrt(mean_squared_error(m[f"{tc}_true"], m[f"{tc}_pred"]))
    mean_rmse = np.mean(list(rmses.values()))
    return mean_rmse, rmses

# ================================================================
print("=" * 60)
print("V3 GRADING REPORT")
print("=" * 60)

results = []

# 1. spaceship-titanic
print("\n--- 1. spaceship-titanic (binary, accuracy) ---")
score, err = grade_binary_accuracy('spaceship-titanic', 'PassengerId', 'Transported')
if score is not None:
    th = MEDAL_THRESHOLDS['spaceship-titanic']
    medal = get_medal(score, th, higher_is_better=True)
    print(f"  Score: {score:.6f} -> {medal} (G={th['gold']} S={th['silver']} B={th['bronze']})")
    results.append(('spaceship-titanic', score, medal, 'accuracy'))
else:
    print(f"  SKIP: {err}")
    results.append(('spaceship-titanic', None, 'SKIP', 'accuracy'))

# 2. dec-2021
print("\n--- 2. tabular-playground-series-dec-2021 (multiclass, accuracy) ---")
score, err = grade_multiclass_accuracy('tabular-playground-series-dec-2021', 'Id', 'Cover_Type')
if score is not None:
    th = MEDAL_THRESHOLDS['tabular-playground-series-dec-2021']
    medal = get_medal(score, th, higher_is_better=True)
    print(f"  Score: {score:.6f} -> {medal} (G={th['gold']} S={th['silver']} B={th['bronze']})")
    results.append(('dec-2021', score, medal, 'accuracy'))
else:
    print(f"  SKIP: {err}")
    results.append(('dec-2021', None, 'SKIP', 'accuracy'))

# 3. may-2022
print("\n--- 3. tabular-playground-series-may-2022 (binary, AUC) ---")
score, err = grade_auc('tabular-playground-series-may-2022', 'id', 'target')
if score is not None:
    th = MEDAL_THRESHOLDS['tabular-playground-series-may-2022']
    medal = get_medal(score, th, higher_is_better=True)
    print(f"  Score: {score:.6f} -> {medal} (G={th['gold']} S={th['silver']} B={th['bronze']})")
    results.append(('may-2022', score, medal, 'auc_roc'))
else:
    print(f"  SKIP: {err}")
    results.append(('may-2022', None, 'SKIP', 'auc_roc'))

# 4. s3e18
print("\n--- 4. playground-series-s3e18 (multilabel, AUC) ---")
sub_path = RESULTS / "playground-series-s3e18" / "submission_s44.csv"
priv_path = PREPARED / "playground-series-s3e18" / "test_private.csv"
if sub_path.exists() and priv_path.exists():
    sub = pd.read_csv(sub_path)
    priv = pd.read_csv(priv_path)
    aucs = {}
    for tc in ['EC1', 'EC2']:
        m = sub.merge(priv[['id', tc]], on='id', suffixes=("_pred", "_true"))
        aucs[tc] = roc_auc_score(m[f"{tc}_true"], m[f"{tc}_pred"])
    mean_auc = np.mean(list(aucs.values()))
    th = MEDAL_THRESHOLDS['playground-series-s3e18']
    medal = get_medal(mean_auc, th, higher_is_better=True)
    print(f"  EC1 AUC: {aucs.get('EC1', '?'):.6f}, EC2 AUC: {aucs.get('EC2', '?'):.6f}")
    print(f"  Mean AUC: {mean_auc:.6f} -> {medal} (G={th['gold']} S={th['silver']} B={th['bronze']})")
    results.append(('s3e18', mean_auc, medal, 'auc_roc'))
else:
    print(f"  SKIP: submission or test_private not found")
    results.append(('s3e18', None, 'SKIP', 'auc_roc'))

# 5. leaf-classification
print("\n--- 5. leaf-classification (multiclass 99, accuracy) ---")
score, err = grade_multiclass_accuracy('leaf-classification', 'id', 'species')
if score is not None:
    th = MEDAL_THRESHOLDS['leaf-classification']
    medal = get_medal(score, th, higher_is_better=True)
    print(f"  Score: {score:.6f} -> {medal} (G={th['gold']} S={th['silver']} B={th['bronze']})")
    results.append(('leaf-classification', score, medal, 'accuracy'))
else:
    print(f"  SKIP: {err}")
    results.append(('leaf-classification', None, 'SKIP', 'accuracy'))

# 6. taxi-fare
print("\n--- 6. taxi-fare (regression, RMSE) ---")
score, err = grade_rmse('new-york-city-taxi-fare-prediction', 'key', 'fare_amount')
if score is not None:
    th = MEDAL_THRESHOLDS['new-york-city-taxi-fare-prediction']
    medal = get_medal(score, th, higher_is_better=False)
    print(f"  Score: {score:.6f} -> {medal} (G={th['gold']} S={th['silver']} B={th['bronze']})")
    results.append(('taxi-fare', score, medal, 'rmse'))
else:
    print(f"  SKIP: {err}")
    results.append(('taxi-fare', None, 'SKIP', 'rmse'))

# 7. nomad2018
print("\n--- 7. nomad2018 (multi-target regression, RMSE) ---")
score, err = grade_multitarget_rmse(
    'nomad2018-predict-transparent-conductors', 'id',
    ['formation_energy_ev_natom', 'bandgap_energy_ev']
)
if score is not None:
    th = MEDAL_THRESHOLDS['nomad2018-predict-transparent-conductors']
    medal = get_medal(score, th, higher_is_better=False)
    print(f"  Mean RMSE: {score:.6f} -> {medal} (G={th['gold']} S={th['silver']} B={th['bronze']})")
    if err:
        for tc, rmse in err.items():
            print(f"    {tc}: {rmse:.6f}")
    results.append(('nomad2018', score, medal, 'rmse'))
else:
    print(f"  SKIP: {err}")
    results.append(('nomad2018', None, 'SKIP', 'rmse'))

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
medal_counts = {'GOLD': 0, 'SILVER': 0, 'BRONZE': 0, 'NO MEDAL': 0, 'SKIP': 0}
for comp, score, medal, metric in results:
    if score is not None:
        print(f"  {comp:<30s} {score:>10.6f} ({metric}) -> {medal}")
    else:
        print(f"  {comp:<30s} {'N/A':>10s} -> {medal}")
    if medal in medal_counts:
        medal_counts[medal] += 1

print(f"\n  TOTAL: {sum(medal_counts.values())} competitions")
for medal, count in medal_counts.items():
    if count > 0:
        print(f"    {medal}: {count}")
