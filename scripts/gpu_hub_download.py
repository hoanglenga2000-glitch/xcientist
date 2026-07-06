#!/usr/bin/env python3
"""Download comps via kagglehub where kaggle CLI fails. Handles extraction."""
import os, shutil, kagglehub, zipfile, glob

BASE = os.path.expanduser("~/jinghw/scripts/gpu_tra/mlebench_raw_data")

# Competitions confirmed accessible via kaggle CLI 'files' listing
# But download blocked (403). kagglehub can download them.
COMPS = [
    "detecting-insults-in-social-commentary",
    "hms-harmful-brain-activity-classification",
    "kuzushiji-recognition",
    "osic-pulmonary-fibrosis-progression",
    "ranzcr-clip-catheter-line-classification",
    "herbarium-2020-fgvc7",
    # "seti-breakthrough-listen",  # 131GB - too large for proxy
    "playground-series-s3e18",
    "the-icml-2013-whale-challenge-right-whale-redux",
    "tensorflow-speech-recognition-challenge",
    "jigsaw-unintended-bias-in-toxicity-classification",
    "tensorflow2-question-answering",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
]

OK = 0
FAIL = 0

for comp in COMPS:
    target = os.path.join(BASE, comp)
    os.makedirs(target, exist_ok=True)

    # Skip if already has usable data
    usable = [f for f in os.listdir(target) if not f.endswith(".zip")]
    if usable:
        size = sum(os.path.getsize(os.path.join(target, f)) for f in usable)
        if size > 1024:
            print(f"SKIP {comp}: {len(usable)} files")
            OK += 1
            continue

    print(f"DOWNLOAD {comp}...")
    try:
        path = kagglehub.competition_download(comp)
        print(f"  cached: {path}")
        for f in os.listdir(path):
            src = os.path.join(path, f)
            dst = os.path.join(target, f)
            if not os.path.exists(dst):
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

        # Extract zips
        for z in glob.glob(os.path.join(target, "*.zip")):
            try:
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(target)
                os.remove(z)
                print(f"  extracted: {os.path.basename(z)}")
            except Exception as e:
                print(f"  bad zip: {e}")
                os.remove(z)

        files = [f for f in os.listdir(target) if not f.endswith(".zip")]
        size = sum(os.path.getsize(os.path.join(target, f)) for f in files)
        print(f"  OK {comp}: {len(files)} files, {size//1024//1024}MB")
        OK += 1

    except Exception as e:
        err = str(e)
        if "403" in err:
            print(f"  BLOCKED {comp}: 403")
        elif "404" in err:
            print(f"  OFFLINE {comp}: 404")
        else:
            print(f"  FAIL {comp}: {err[:150]}")
        FAIL += 1

print(f"\nDONE: OK={OK} FAIL={FAIL}")
