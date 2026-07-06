"""Extract nested .csv.zip files for leaf and nomad2018, then run prep."""
import zipfile, os
from pathlib import Path

RAW = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data")

for comp_name in ["leaf-classification", "nomad2018-predict-transparent-conductors"]:
    raw_dir = RAW / comp_name
    print(f"\n=== {comp_name} ===")

    # Extract all nested .csv.zip files
    for f in sorted(raw_dir.glob("*.csv.zip")):
        out_name = f.stem  # removes .zip only, leaves .csv
        out_path = raw_dir / out_name
        if not out_path.exists():
            with zipfile.ZipFile(f, "r") as z:
                z.extractall(raw_dir)
            print(f"  Extracted {f.name} -> {out_name}")
        else:
            print(f"  {out_name} already exists, skip")

    # Also check for other nested zips
    for f in sorted(raw_dir.glob("*.zip")):
        if f.name == f"{comp_name}.zip":
            continue
        with zipfile.ZipFile(f, "r") as z:
            for n in z.namelist():
                if n.endswith(".csv"):
                    out_path = raw_dir / Path(n).name
                    if not out_path.exists():
                        z.extract(n, raw_dir)
                        print(f"  Extracted {n} from {f.name}")

    csvs = list(raw_dir.glob("*.csv"))
    print(f"  CSVs: {[c.name for c in csvs]}")

print("\nDone extracting nested zips!")
