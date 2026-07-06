#!/usr/bin/env python3
"""Check target values and sample submission format for broken competitions."""
import pandas as pd
import os

BASE = os.path.expanduser("~/jinghw/scripts/gpu_tra/data")

COMPETITIONS = [
    "tabular-playground-series-feb-2022",
    "playground-series-s6e2",
    "playground-series-s6e3",
    "playground-series-s4e7",
]

for comp in COMPETITIONS:
    print(f"\n{'='*60}")
    print(f"COMPETITION: {comp}")

    train_path = os.path.join(BASE, comp, "train.csv")
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        t = df.iloc[:, -1]
        print(f"  Target column: {df.columns[-1]}")
        print(f"  Target dtype: {t.dtype}")
        uniq = list(t.unique())
        print(f"  Unique values (first 30): {uniq[:30]}")
        print(f"  N unique: {t.nunique()}")
        print(f"  Sample values: {uniq[:10]}")
    else:
        print(f"  train.csv NOT FOUND at {train_path}")

    sub_paths = [
        os.path.join(BASE, comp, "sample_submission.csv"),
        os.path.join(BASE, comp, "sampleSubmission.csv"),
    ]
    for sp in sub_paths:
        if os.path.exists(sp):
            sub = pd.read_csv(sp)
            print(f"  Sample submission: {sp}")
            print(f"    Columns: {list(sub.columns)}")
            print(f"    Dtypes: {dict(sub.dtypes)}")
            print(f"    Head:")
            print(sub.head(3).to_string())
            break
    else:
        print(f"  No sample_submission found")
