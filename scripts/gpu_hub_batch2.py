#!/usr/bin/env python3
"""Batch 2: Download remaining accessible competitions via kagglehub."""
import os, shutil, kagglehub, zipfile, glob

BASE = os.path.expanduser("~/jinghw/scripts/gpu_tra/mlebench_raw_data")

COMPS = [
    "freesound-audio-tagging-2019",
    "google-research-identify-contrails-reduce-global-warming",
    "h-and-m-personalized-fashion-recommendations",
    "herbarium-2021-fgvc8",
    "herbarium-2022-fgvc9",
    "histopathologic-cancer-detection",
    "hotel-id-2021-fgvc8",
    "icecube-neutrinos-in-deep-ice",
    "imet-2020-fgvc7",
    "inaturalist-2019-fgvc6",
    "iwildcam-2019-fgvc6",
    "iwildcam-2020-fgvc7",
    "nfl-player-contact-detection",
    "petfinder-pawpularity-score",
    "plant-pathology-2021-fgvc8",
    "predict-volcanic-eruptions-ingv-oe",
    "rsna-2022-cervical-spine-fracture-detection",
    "rsna-breast-cancer-detection",
    "rsna-miccai-brain-tumor-radiogenomic-classification",
    "siim-isic-melanoma-classification",
    "smartphone-decimeter-2022",
    "uw-madison-gi-tract-image-segmentation",
    "vesuvius-challenge-ink-detection",
]

OK = 0
FAIL = 0

for comp in COMPS:
    target = os.path.join(BASE, comp)
    os.makedirs(target, exist_ok=True)

    usable = [f for f in os.listdir(target) if not f.endswith(".zip")]
    if usable:
        size = sum(os.path.getsize(os.path.join(target, f)) for f in usable)
        if size > 1024:
            print(f"SKIP {comp}: {len(usable)} files, {size//1024//1024}MB")
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
        else:
            print(f"  FAIL {comp}: {err[:150]}")
        FAIL += 1

print(f"\nDONE: OK={OK} FAIL={FAIL}")
