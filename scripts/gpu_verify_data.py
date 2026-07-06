#!/usr/bin/env python3
"""Quick verify: check each dir has train/test/sample files (non-zip)."""
import os, glob

BASE = os.path.expanduser("~/jinghw/scripts/gpu_tra/mlebench_raw_data")

complete = 0
no_train = []
no_test = []
no_sample = []
empty = []

for d in sorted(os.listdir(BASE)):
    dp = os.path.join(BASE, d)
    if not os.path.isdir(dp):
        continue

    # Quick check: any non-zip files?
    top_files = [f for f in os.listdir(dp) if not f.endswith(".zip")]
    top_names = " ".join(top_files).lower()

    has_train = "train" in top_names
    has_test = "test" in top_names
    has_sample = "sample" in top_names

    if not top_files:
        empty.append(d)
    else:
        if not has_train: no_train.append(d)
        if not has_test: no_test.append(d)
        if not has_sample: no_sample.append(d)
        if has_train and has_test and has_sample:
            complete += 1

print(f"COMPLETE (train+test+sample): {complete}")
print(f"NO_TRAIN: {len(no_train)}")
print(f"NO_TEST: {len(no_test)}")
print(f"NO_SAMPLE: {len(no_sample)}")
print(f"EMPTY: {len(empty)}")

if no_train:
    print("\n--- Missing TRAIN ---")
    for d in no_train: print(f"  {d}")
if no_test:
    print("\n--- Missing TEST ---")
    for d in no_test: print(f"  {d}")
if no_sample:
    print("\n--- Missing SAMPLE ---")
    for d in no_sample: print(f"  {d}")
if empty:
    print("\n--- EMPTY ---")
    for d in empty: print(f"  {d}")
