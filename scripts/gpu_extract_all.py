#!/usr/bin/env python3
"""Extract all zips in mlebench_raw_data, then report remaining empty dirs."""
import os, zipfile, glob

BASE = os.path.expanduser("~/jinghw/scripts/gpu_tra/mlebench_raw_data")

extracted = 0
failed = 0
still_empty = []
still_zip = []

for d in sorted(os.listdir(BASE)):
    dp = os.path.join(BASE, d)
    if not os.path.isdir(dp):
        continue
    zips = glob.glob(os.path.join(dp, "*.zip"))
    if not zips:
        non_zips = [f for f in os.listdir(dp) if not f.endswith(".zip")]
        if not non_zips:
            still_empty.append(d)
        continue

    for z in zips:
        try:
            size_mb = os.path.getsize(z) // (1024*1024)
            print(f"Extracting {d}/{os.path.basename(z)} ({size_mb}MB)...")
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(dp)
            os.remove(z)
            extracted += 1
            print(f"  OK")
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
            # Remove corrupt zip
            try:
                os.remove(z)
                print(f"  Removed corrupt zip")
            except:
                pass

# Re-scan for still-empty
for d in sorted(os.listdir(BASE)):
    dp = os.path.join(BASE, d)
    if not os.path.isdir(dp):
        continue
    all_files = os.listdir(dp)
    zips_left = [f for f in all_files if f.endswith(".zip")]
    non_zips = [f for f in all_files if not f.endswith(".zip")]
    if not non_zips and not zips_left:
        if d not in still_empty:
            still_empty.append(d)
    if zips_left and not non_zips:
        if d not in still_zip:
            still_zip.append(d)

print(f"\n=== RESULTS ===")
print(f"Extracted: {extracted}, Failed: {failed}")
print(f"\nStill empty ({len(still_empty)}):")
for d in sorted(still_empty):
    print(f"  {d}")
if still_zip:
    print(f"\nStill has unextractable zips ({len(still_zip)}):")
    for d in sorted(still_zip):
        print(f"  {d}")
