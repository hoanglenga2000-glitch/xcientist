with open('/hpc2hdd/home/aimslab/gpu_train_v3.py', 'r') as f:
    code = f.read()

old = "    elif task_type == 'binary':\n        pred_raw = (test_preds > 0.5).astype(int)\n        if val_fmt == 'bool':\n            pred_values = ['True' if p == 1 else 'False' for p in pred_raw]\n        else:\n            pred_values = pred_raw"

new = "    elif task_type == 'binary':\n        if val_fmt == 'prob':\n            pred_values = test_preds\n        elif val_fmt == 'bool':\n            pred_raw = (test_preds > 0.5).astype(int)\n            pred_values = ['True' if p == 1 else 'False' for p in pred_raw]\n        else:\n            pred_values = (test_preds > 0.5).astype(int)"

if old in code:
    code = code.replace(old, new)
    with open('/hpc2hdd/home/aimslab/gpu_train_v3.py', 'w') as f:
        f.write(code)
    print('FIX_APPLIED')
else:
    print('OLD_NOT_FOUND - already fixed or different code')
    # Show what's there
    import re
    m = re.search(r"elif task_type == 'binary':.*?pred_values = pred_raw", code, re.DOTALL)
    if m:
        print('Current code:', repr(m.group()[:200]))
