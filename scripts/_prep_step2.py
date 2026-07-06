"""Step 2: train_test_split for leaf and nomad2018 (CSVs now extracted from nested zips)."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

HOME = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
RAW = HOME / "mlebench_raw_data"
PREPARED = HOME / "mlebench_prepared"

# === leaf-classification ===
name = "leaf-classification"
out_dir = PREPARED / name
out_dir.mkdir(exist_ok=True, parents=True)
raw_dir = RAW / name
train = pd.read_csv(raw_dir / "train.csv")
print(f"leaf: loaded {train.shape}, cols={list(train.columns)[:10]}")
new_train, new_test = train_test_split(train, test_size=0.1, random_state=0)
new_train.to_csv(out_dir / "train.csv", index=False)
test_x = new_test.drop(columns=["species"])
test_x.to_csv(out_dir / "test.csv", index=False)
new_test.to_csv(out_dir / "test_private.csv", index=False)
sub = test_x[["id"]].copy()
sub["species"] = "Acer_Capillipes"
sub.to_csv(out_dir / "sample_submission.csv", index=False)
print(f"leaf: DONE train={new_train.shape} test={new_test.shape}")

# === nomad2018 ===
name = "nomad2018-predict-transparent-conductors"
out_dir = PREPARED / name
out_dir.mkdir(exist_ok=True, parents=True)
raw_dir = RAW / name
train = pd.read_csv(raw_dir / "train.csv")
print(f"nomad2018: loaded {train.shape}, cols={list(train.columns)[:10]}")
targets = ["formation_energy_ev_natom", "bandgap_energy_ev"]
new_train, new_test = train_test_split(train, test_size=0.1, random_state=0)
new_train.to_csv(out_dir / "train.csv", index=False)
test_x = new_test.drop(columns=targets)
test_x.to_csv(out_dir / "test.csv", index=False)
new_test.to_csv(out_dir / "test_private.csv", index=False)
sub = test_x[["id"]].copy()
for t in targets:
    sub[t] = 0.5
sub.to_csv(out_dir / "sample_submission.csv", index=False)
print(f"nomad2018: DONE train={new_train.shape} test={new_test.shape}")

print("\n=== Summary ===")
for d in sorted(PREPARED.iterdir()):
    if d.is_dir():
        files = list(d.glob("*.csv"))
        total = sum(f.stat().st_size for f in files) / 1024**2
        print(f"  {d.name}: {len(files)} files, {total:.1f} MB")
