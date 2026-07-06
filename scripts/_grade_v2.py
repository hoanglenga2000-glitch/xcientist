"""Grade v2 submissions against test_private labels on server."""
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path

PREPARED = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_prepared")
RESULTS = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_proper_results")

# First list what's available
print("=== Prepared data ===")
for d in sorted(PREPARED.iterdir()):
    if d.is_dir():
        csvs = list(d.glob("*.csv"))
        print(f"  {d.name}: {[c.name for c in csvs]}")

print("\n=== Results ===")
for d in sorted(RESULTS.iterdir()):
    if d.is_dir():
        csvs = list(d.glob("*.csv"))
        print(f"  {d.name}: {[c.name for c in csvs]}")

print("\n=== GRADING ===")

# 1. spaceship-titanic
print("\n--- spaceship-titanic ---")
sub = pd.read_csv(RESULTS / "spaceship-titanic" / "submission_s44.csv")
priv_path = PREPARED / "spaceship-titanic" / "test_private.csv"
if priv_path.exists():
    priv = pd.read_csv(priv_path)
    m = sub.merge(priv[["PassengerId", "Transported"]], on="PassengerId", suffixes=("_pred", "_true"))
    score = accuracy_score(m["Transported_true"], m["Transported_pred"])
else:
    # Try test.csv (which might be the test split)
    test_path = PREPARED / "spaceship-titanic" / "test.csv"
    print(f"  test_private not found, trying {test_path}")
    # Check if test.csv has labels
    test_df = pd.read_csv(test_path)
    print(f"  test.csv cols: {list(test_df.columns)}")
    # If test.csv doesn't have Transported, use train_test split
    # Use the original train.csv split
    print(f"  WARNING: Cannot grade without test_private.csv with true labels")
    print(f"  Using OOF scores from log instead")
    score = None

if score is not None:
    print(f"  Score: {score:.6f}")
    thresholds = [("GOLD", 0.82066), ("SILVER", 0.81388), ("BRONZE", 0.80967)]
    for medal, th in thresholds:
        if score >= th:
            print(f"  -> {medal}")
            break
    else:
        print(f"  -> NO MEDAL")

# 2. dec-2021
print("\n--- tabular-playground-series-dec-2021 ---")
sub = pd.read_csv(RESULTS / "tabular-playground-series-dec-2021" / "submission_s44.csv")
priv_path = PREPARED / "tabular-playground-series-dec-2021" / "test_private.csv"
if priv_path.exists():
    priv = pd.read_csv(priv_path)
    m = sub.merge(priv[["Id", "Cover_Type"]], on="Id", suffixes=("_pred", "_true"))
    score = accuracy_score(m["Cover_Type_true"], m["Cover_Type_pred"])
    print(f"  Score: {score:.6f}")
    thresholds = [("GOLD", 0.95660), ("SILVER", 0.95658), ("BRONZE", 0.95658)]
    for medal, th in thresholds:
        if score >= th:
            print(f"  -> {medal}")
            break
    else:
        print(f"  -> NO MEDAL")
else:
    print(f"  test_private not found!")
    score = None

# 3. may-2022
print("\n--- tabular-playground-series-may-2022 ---")
sub = pd.read_csv(RESULTS / "tabular-playground-series-may-2022" / "submission_s44.csv")
priv_path = PREPARED / "tabular-playground-series-may-2022" / "test_private.csv"
if priv_path.exists():
    priv = pd.read_csv(priv_path)
    m = sub.merge(priv[["id", "target"]], on="id", suffixes=("_pred", "_true"))
    score = roc_auc_score(m["target_true"], m["target_pred"])
    print(f"  Score: {score:.6f}")
    thresholds = [("GOLD", 0.99823), ("SILVER", 0.99822), ("BRONZE", 0.99818)]
    for medal, th in thresholds:
        if score >= th:
            print(f"  -> {medal}")
            break
    else:
        print(f"  -> NO MEDAL")
else:
    print(f"  test_private not found!")
    score = None

print("\nDone!")
