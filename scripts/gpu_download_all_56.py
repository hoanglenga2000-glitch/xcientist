#!/usr/bin/env python3
"""Download all 56 MLE-Bench 75 competitions. Run on GPU server via nohup."""
import subprocess, os, json, glob, zipfile, time, shutil

BASE = os.path.expanduser("~/jinghw/scripts/gpu_tra/mlebench_raw_data")
LOG = os.path.expanduser("~/jinghw/scripts/gpu_tra/download_all_56.log")
RESULTS = os.path.expanduser("~/jinghw/scripts/gpu_tra/download_all_56.json")

ALL_56 = [
    "detecting-insults-in-social-commentary","random-acts-of-pizza",
    "spooky-author-identification","tweet-sentiment-extraction",
    "us-patent-phrase-to-phrase-matching","facebook-recruiting-iii-keyword-extraction",
    "kuzushiji-recognition","dogs-vs-cats-redux-kernels-edition",
    "google-quest-challenge","jigsaw-toxic-comment-classification-challenge",
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
    "tensorflow2-question-answering",
]

results = {}
already_had = ["random-acts-of-pizza","spooky-author-identification","tweet-sentiment-extraction","us-patent-phrase-to-phrase-matching"]

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def try_download(comp):
    target = os.path.join(BASE, comp)
    os.makedirs(target, exist_ok=True)

    # Check if already has meaningful data
    existing = [f for f in os.listdir(target) if not f.endswith(".zip")]
    if existing:
        total = sum(os.path.getsize(os.path.join(target, f)) for f in existing)
        if total > 1024:  # more than 1KB
            log(f"SKIP {comp}: already {len(existing)} files ({total//1024//1024}MB)")
            results[comp] = "already_had"
            return

    # Try kaggle CLI
    log(f"DOWNLOAD {comp}...")
    r = subprocess.run(
        ["kaggle", "competitions", "download", "-c", comp, "-p", target],
        capture_output=True, text=True, timeout=180
    )

    if r.returncode == 0:
        for z in glob.glob(os.path.join(target, "*.zip")):
            try:
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(target)
                os.remove(z)
            except Exception as e:
                log(f"  unzip error: {e}")
        files = os.listdir(target)
        size = sum(os.path.getsize(os.path.join(target, f)) for f in files) // (1024*1024)
        log(f"  OK: {len(files)} files ({size}MB)")
        results[comp] = f"ok_{size}MB"
        return

    err_msg = (r.stderr + " " + r.stdout)[:300]

    if "403" in err_msg.lower() or "forbidden" in err_msg.lower():
        log(f"  BLOCKED (403): rules still not accepted or competition offline")
        results[comp] = "blocked_403"
        return

    if "404" in err_msg.lower() or "not found" in err_msg.lower():
        log(f"  NOT FOUND: competition may have been removed")
        results[comp] = "not_found"
        return

    # Fallback: try kagglehub
    try:
        import kagglehub
        log(f"  Trying kagglehub for {comp}...")
        path = kagglehub.competition_download(comp)
        for f in os.listdir(path):
            src = os.path.join(path, f)
            dst = os.path.join(target, f)
            if not os.path.exists(dst):
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        files = os.listdir(target)
        size = sum(os.path.getsize(os.path.join(target, f)) for f in files) // (1024*1024)
        log(f"  OK (kagglehub): {len(files)} files ({size}MB)")
        results[comp] = f"ok_hub_{size}MB"
    except Exception as e:
        log(f"  FAIL: {str(e)[:150]}")
        results[comp] = f"error: {str(e)[:100]}"

# Main
log("=== BATCH DOWNLOAD ALL 56 ===")
start_time = time.time()

for i, comp in enumerate(ALL_56, 1):
    log(f"[{i}/56] {comp}")
    try:
        try_download(comp)
    except Exception as e:
        log(f"  EXCEPTION: {e}")
        results[comp] = f"exception: {str(e)[:100]}"
    time.sleep(1)

elapsed = (time.time() - start_time) / 60

with open(RESULTS, "w") as f:
    json.dump(results, f, indent=2)

ok = sum(1 for v in results.values() if v.startswith("ok"))
had = sum(1 for v in results.values() if v == "already_had")
blocked = sum(1 for v in results.values() if v == "blocked_403")
nf = sum(1 for v in results.values() if v == "not_found")
err = sum(1 for v in results.values() if v.startswith("error") or v.startswith("exception"))
log(f"=== DONE in {elapsed:.1f}min: OK={ok} HAD={had} BLOCKED={blocked} NOTFOUND={nf} ERR={err} ===")
