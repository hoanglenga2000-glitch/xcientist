#!/usr/bin/env python3
"""Download ALL MLE-Bench 75 comps using kagglehub. Handles resume, cache, extraction."""
import os, json, time, shutil, glob, zipfile

BASE = os.path.expanduser("~/jinghw/scripts/gpu_tra/mlebench_raw_data")
LOG = os.path.expanduser("~/jinghw/scripts/gpu_tra/download_hub.log")
RESULTS = os.path.expanduser("~/jinghw/scripts/gpu_tra/download_hub.json")

ALL = [
    "random-acts-of-pizza","spooky-author-identification",
    "tweet-sentiment-extraction","us-patent-phrase-to-phrase-matching",
    "facebook-recruiting-iii-keyword-extraction","kuzushiji-recognition",
    "dogs-vs-cats-redux-kernels-edition","google-quest-challenge",
    "jigsaw-toxic-comment-classification-challenge",
    "jigsaw-unintended-bias-in-toxicity-classification",
    "tensorflow-speech-recognition-challenge",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
    "statoil-iceberg-classifier-challenge","mlsp-2013-birds",
    "learning-agency-lab-automated-essay-scoring-2",
    "ventilator-pressure-prediction","tgs-salt-identification-challenge",
    "playground-series-s3e18","whale-categorization-playground",
    "histopathologic-cancer-detection","hubmap-kidney-segmentation",
    "vinbigdata-chest-xray-abnormalities-detection","siim-covid19-detection",
    "siim-isic-melanoma-classification","ranzcr-clip-catheter-line-classification",
    "rsna-2022-cervical-spine-fracture-detection","rsna-breast-cancer-detection",
    "rsna-miccai-brain-tumor-radiogenomic-classification",
    "uw-madison-gi-tract-image-segmentation","stanford-covid-vaccine",
    "hms-harmful-brain-activity-classification","osic-pulmonary-fibrosis-progression",
    "herbarium-2020-fgvc7","herbarium-2021-fgvc8","herbarium-2022-fgvc9",
    "plant-pathology-2020-fgvc7","plant-pathology-2021-fgvc8",
    "inaturalist-2019-fgvc6","iwildcam-2019-fgvc6","iwildcam-2020-fgvc7",
    "imet-2020-fgvc7","hotel-id-2021-fgvc8","seti-breakthrough-listen",
    "vesuvius-challenge-ink-detection","icecube-neutrinos-in-deep-ice",
    "google-research-identify-contrails-reduce-global-warming",
    "nfl-player-contact-detection","predict-volcanic-eruptions-ingv-oe",
    "smartphone-decimeter-2022","freesound-audio-tagging-2019",
    "petfinder-pawpularity-score","h-and-m-personalized-fashion-recommendations",
    "tensorflow2-question-answering","detecting-insults-in-social-commentary",
]

results = {}

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def extract_zips(directory):
    """Extract all zip files in a directory, remove zips after."""
    for z in glob.glob(os.path.join(directory, "*.zip")):
        try:
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(directory)
            os.remove(z)
            log(f"    Extracted: {os.path.basename(z)}")
        except Exception as e:
            log(f"    Bad zip {os.path.basename(z)}: {e}")
            try:
                os.remove(z)
            except:
                pass

def try_download(comp):
    target = os.path.join(BASE, comp)
    os.makedirs(target, exist_ok=True)

    # Skip if already has data
    existing = [f for f in os.listdir(target) if not f.endswith(".zip")]
    if existing:
        total = sum(os.path.getsize(os.path.join(target, f)) for f in existing)
        if total > 1024:
            log(f"SKIP {comp}: {len(existing)}f/{total//1024//1024}MB")
            results[comp] = "had"
            return

    # Extract any existing zips first
    extract_zips(target)

    log(f"DOWNLOAD {comp}")

    try:
        import kagglehub
        cache_path = kagglehub.competition_download(comp)
        log(f"    Cached: {cache_path}")

        # Copy from cache to target
        copied = 0
        for f in os.listdir(cache_path):
            src = os.path.join(cache_path, f)
            dst = os.path.join(target, f)
            if not os.path.exists(dst):
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
                copied += 1
        log(f"    Copied {copied} files")

        # Extract any zips
        extract_zips(target)

        files = os.listdir(target)
        size = sum(os.path.getsize(os.path.join(target, f)) for f in files) // (1024*1024)
        log(f"  OK {comp}: {len(files)}f/{size}MB")
        results[comp] = f"ok_{size}MB"

    except Exception as e:
        err_str = str(e)
        if "403" in err_str or "Forbidden" in err_str or "permission" in err_str.lower():
            log(f"  BLOCKED(403): {comp}")
            results[comp] = "blocked_403"
        elif "404" in err_str or "not found" in err_str.lower():
            log(f"  NOT_FOUND: {comp}")
            results[comp] = "not_found"
        else:
            log(f"  FAIL {comp}: {err_str[:200]}")
            results[comp] = f"error"

# Main
log("=== KAGGLEHUB DOWNLOAD ALL ===")
start = time.time()

for i, comp in enumerate(ALL, 1):
    log(f"[{i}/{len(ALL)}] {comp}")
    try:
        try_download(comp)
    except Exception as e:
        log(f"  EXCEPTION: {e}")
        results[comp] = "exception"
    time.sleep(1)

elapsed = (time.time() - start) / 60

with open(RESULTS, "w") as f:
    json.dump(results, f, indent=2)

ok = sum(1 for v in results.values() if v.startswith("ok"))
had = sum(1 for v in results.values() if v == "had")
blocked = sum(1 for v in results.values() if v == "blocked_403")
nf = sum(1 for v in results.values() if v == "not_found")
err = sum(1 for v in results.values() if v.startswith("error") or v == "exception")
log(f"=== DONE in {elapsed:.1f}min: OK={ok} HAD={had} BLOCKED={blocked} NOTFOUND={nf} ERR={err} ===")
