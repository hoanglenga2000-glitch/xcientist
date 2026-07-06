#!/usr/bin/env python3
import os

BASE = os.path.expanduser("~/jinghw/scripts/gpu_tra/mlebench_raw_data")

have = []
empty = []
zip_only = []

for d in sorted(os.listdir(BASE)):
    dp = os.path.join(BASE, d)
    if not os.path.isdir(dp): continue
    all_files = os.listdir(dp)
    non_zips = [f for f in all_files if not f.endswith(".zip")]
    zips = [f for f in all_files if f.endswith(".zip")]
    total = sum(os.path.getsize(os.path.join(dp, f)) for f in all_files)

    if total > 1024:
        have.append((d, total//1024//1024, len(non_zips)))
    elif zips:
        zsize = sum(os.path.getsize(os.path.join(dp, f)) for f in zips) // (1024*1024)
        zip_only.append((d, zsize))
    else:
        empty.append(d)

print("=== HAVE DATA ===")
for d, s, n in have:
    print(f"  {d}: {s}MB, {n} files")

print(f"\n=== ZIP ONLY (need extract) ===")
for d, s in zip_only:
    print(f"  {d}: {s}MB zip")

print(f"\n=== EMPTY (need download) ===")
for d in empty:
    print(f"  {d}")

print(f"\nTOTAL: have={len(have)}, zip_only={len(zip_only)}, empty={len(empty)}")
