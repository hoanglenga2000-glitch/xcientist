#!/usr/bin/env python3
"""Download small MLE-Bench comps (<200MB), skip large ones for later."""
import subprocess, os, json, glob, zipfile, time

BASE = os.path.expanduser("~/jinghw/scripts/gpu_tra/mlebench_raw_data")
LOG = os.path.expanduser("~/jinghw/scripts/gpu_tra/download_small.log")
RESULTS = os.path.expanduser("~/jinghw/scripts/gpu_tra/download_small.json")

SMALL = [
    "random-acts-of-pizza","spooky-author-identification",
    "tweet-sentiment-extraction","us-patent-phrase-to-phrase-matching",
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
    "histopathologic-cancer-detection","siim-covid19-detection",
    "stanford-covid-vaccine","ranzcr-clip-catheter-line-classification",
    "rsna-miccai-brain-tumor-radiogenomic-classification",
    "hotel-id-2021-fgvc8","nfl-player-contact-detection",
    "smartphone-decimeter-2022","freesound-audio-tagging-2019",
    "petfinder-pawpularity-score","predict-volcanic-eruptions-ingv-oe",
    "tensorflow2-question-answering","detecting-insults-in-social-commentary",
    "hms-harmful-brain-activity-classification",
    "osic-pulmonary-fibrosis-progression",
    "plant-pathology-2020-fgvc7","plant-pathology-2021-fgvc8",
]

results = {}

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def try_dl(comp):
    target = os.path.join(BASE, comp)
    os.makedirs(target, exist_ok=True)

    existing = [f for f in os.listdir(target) if not f.endswith(".zip")]
    if existing:
        total = sum(os.path.getsize(os.path.join(target, f)) for f in existing)
        if total > 1024:
            log(f"SKIP {comp}: {len(existing)}f/{total//1024//1024}MB")
            results[comp] = "had"
            return

    for z in glob.glob(os.path.join(target, "*.zip")):
        try:
            with zipfile.ZipFile(z, "r") as zf:
                zf.testzip()
        except:
            os.remove(z)
            log(f"  Del corrupt {os.path.basename(z)}")

    log(f"DL {comp}")
    r = subprocess.run(
        ["kaggle", "competitions", "download", "-c", comp, "-p", target],
        capture_output=True, text=True, timeout=300
    )

    if r.returncode == 0:
        for z in glob.glob(os.path.join(target, "*.zip")):
            try:
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(target)
                os.remove(z)
            except Exception as e:
                log(f"  Bad zip: {e}")
                os.remove(z)
                results[comp] = "bad_zip"
                return
        files = os.listdir(target)
        size = sum(os.path.getsize(os.path.join(target, f)) for f in files) // (1024*1024)
        log(f"  OK: {len(files)}f/{size}MB")
        results[comp] = f"ok_{size}MB"
    else:
        err = (r.stderr + " " + r.stdout)[:150]
        tag = "403" if ("403" in err.lower() or "forbidden" in err.lower()) else "err"
        log(f"  FAIL({tag}): {err}")
        results[comp] = f"fail_{tag}"

log("=== SMALL FILES DOWNLOAD ===")
for i, comp in enumerate(SMALL, 1):
    log(f"[{i}/{len(SMALL)}]")
    try:
        try_dl(comp)
    except Exception as e:
        log(f"EXC: {e}")
        results[comp] = "exc"
    time.sleep(0.5)

with open(RESULTS, "w") as f:
    json.dump(results, f, indent=2)

ok = sum(1 for v in results.values() if v.startswith("ok"))
had = sum(1 for v in results.values() if v == "had")
fail = len(results) - ok - had
log(f"=== DONE: OK={ok} HAD={had} FAIL={fail} ===")
