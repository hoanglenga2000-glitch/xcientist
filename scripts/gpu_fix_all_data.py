#!/usr/bin/env python3
"""Fix all data issues: extract zips, verify subdirs, report final status."""
import os, zipfile, glob

BASE = os.path.expanduser("~/jinghw/scripts/gpu_tra/mlebench_raw_data")

print("=== STEP 1: Extract remaining zips ===")
for d in sorted(os.listdir(BASE)):
    dp = os.path.join(BASE, d)
    if not os.path.isdir(dp): continue
    for z in glob.glob(os.path.join(dp, "*.zip")):
        zsize = os.path.getsize(z) // (1024*1024)
        try:
            print(f"  {d}/{os.path.basename(z)} ({zsize}MB)...")
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(dp)
            os.remove(z)
            print(f"    OK")
        except Exception as e:
            print(f"    BAD: {e}")
            os.remove(z)

print("\n=== STEP 2: Deep verify - check ALL files (top + subdirs) ===")
issues = []
ok = 0

for d in sorted(os.listdir(BASE)):
    dp = os.path.join(BASE, d)
    if not os.path.isdir(dp): continue

    # Find ALL non-zip files recursively
    all_files = []
    for root, dirs, files in os.walk(dp):
        for f in files:
            if not f.endswith(".zip"):
                all_files.append(os.path.join(root, f))

    if not all_files:
        issues.append(f"EMPTY: {d}")
        continue

    all_lower = " ".join(f.lower() for f in all_files)
    has_train = "train" in all_lower
    has_test = "test" in all_lower
    has_sample = "sample" in all_lower

    total_size = sum(os.path.getsize(f) for f in all_files)
    size_mb = total_size // (1024*1024)

    missing = []
    if not has_train: missing.append("train")
    if not has_test: missing.append("test")
    if not has_sample: missing.append("sample")

    if missing:
        issues.append(f"MISSING: {d} - {','.join(missing)} - {len(all_files)} files, {size_mb}MB")
    else:
        ok += 1

print(f"\n=== FINAL ===")
print(f"FULLY READY: {ok}")
print(f"ISSUES: {len(issues)}")
for i in issues:
    print(f"  {i}")
