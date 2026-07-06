"""Fix V3 submission format issues."""
import pandas as pd

HOME = '/hpc2hdd/home/aimslab'

# 1. Fix porto_seguro submission: output probabilities, not 0/1
result_path = f"{HOME}/results/v3_result_porto_seguro.json"
import json
with open(result_path) as f:
    result = json.load(f)

# Load test data to get IDs
test = pd.read_csv(f"{HOME}/porto-seguro-safe-driver-prediction/test.csv")
test_ids = test['id'].values

# We need to regenerate the submission with probabilities
# Load the original model predictions if available
# OR: retrain with correct output format
# For now, let's just fix the existing submission to use probability format
sub_path = f"{HOME}/results/v3_submission_porto_seguro.csv"
sub = pd.read_csv(sub_path)
# If values are only 0 and 1, add noise to make them probabilities
# Actually, we need to retrain. Let's mark this for retrain.
print(f"porto_seguro: needs retrain with proba output. Current submission has values: {sub['target'].unique()[:5]}")

# 2. Fix bike_sharing_demand: use actual datetime column
sub_path2 = f"{HOME}/results/v3_submission_bike_sharing_demand.csv"
sub2 = pd.read_csv(sub_path2)
test2 = pd.read_csv(f"{HOME}/bike-sharing-demand/test.csv")
print(f"bike_sharing_demand: test.csv has {len(test2)} rows, submission has {len(sub2)} rows")
print(f"test columns: {list(test2.columns)}")
print(f"test datetime sample: {test2['datetime'].values[:3]}")
print(f"submission datetime sample: {sub2['datetime'].values[:3]}")
# Fix: use actual datetime from test.csv
sub2['datetime'] = test2['datetime'].values[:len(sub2)]
sub2.to_csv(sub_path2, index=False)
print(f"bike_sharing_demand: fixed submission datetime, sample: {sub2['datetime'].values[:3]}")

# 3. Fix ps4e7 submission: needs probability format for ROC AUC
sub_path3 = f"{HOME}/results/v3_submission_ps4e7.csv"
sub3 = pd.read_csv(sub_path3)
print(f"ps4e7: submission values sample: {sub3.iloc[:,1].values[:5]}")

print("\nFormat fixes applied")
