"""Fix V3 competition registry with correct metrics."""
with open('/hpc2hdd/home/aimslab/gpu_train_v3.py', 'r') as f:
    code = f.read()

# ps3e25: Hardness is continuous -> regression
code = code.replace(
    '"ps3e25": ("playground-series-s3e25", "Hardness", "multiclass", "accuracy", "max", 0.700, 0.010)',
    '"ps3e25": ("playground-series-s3e25", "Hardness", "regression", "rmse", "min", 0.700, 0.020)')

# porto_seguro: metric is normalized_gini, not accuracy
code = code.replace(
    '"porto_seguro":          ("porto-seguro-safe-driver-prediction", "target", "binary", "accuracy", "max", 0.285, 0.010)',
    '"porto_seguro":          ("porto-seguro-safe-driver-prediction", "target", "binary", "normalized_gini", "max", 0.285, 0.010)')

# ps4e7: metric is roc_auc, not accuracy
code = code.replace(
    '"ps4e7":  ("playground-series-s4e7", "Response", "binary", "accuracy", "max", 0.600, 0.010)',
    '"ps4e7":  ("playground-series-s4e7", "Response", "binary", "roc_auc", "max", 0.600, 0.010)')

# ps5e2: Price regression - metric should be rmsle
code = code.replace(
    '"ps5e2":  ("playground-series-s5e2", "Price", "regression", "rmse", "min", 0.800, 0.020)',
    '"ps5e2":  ("playground-series-s5e2", "Price", "regression", "rmsle", "min", 0.800, 0.020)')

with open('/hpc2hdd/home/aimslab/gpu_train_v3.py', 'w') as f:
    f.write(code)
print('Registry fixes applied successfully')
