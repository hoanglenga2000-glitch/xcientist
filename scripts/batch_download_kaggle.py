#!/usr/bin/env python3
"""Batch download all MLE-Bench competitions from Kaggle."""
import subprocess, os, zipfile, shutil, time

DATA_DIR = os.path.expanduser("~/jinghw/scripts/gpu_tra/data/")
os.makedirs(DATA_DIR, exist_ok=True)

COMPETITIONS = [
    # Getting Started / Classic
    "titanic", "house-prices-advanced-regression-techniques", "digit-recognizer",
    "spaceship-titanic", "store-sales-time-series-forecasting",
    "bike-sharing-demand", "nlp-getting-started",
    "home-data-for-ml-course",

    # Playground S3
    "playground-series-s3e1", "playground-series-s3e7", "playground-series-s3e25",

    # Playground S4
    "playground-series-s4e1", "playground-series-s4e2", "playground-series-s4e3",
    "playground-series-s4e4", "playground-series-s4e6", "playground-series-s4e7",

    # Playground S5
    "playground-series-s5e1", "playground-series-s5e2", "playground-series-s5e3",
    "playground-series-s5e4", "playground-series-s5e5",

    # Playground S6
    "playground-series-s6e2", "playground-series-s6e3", "playground-series-s6e6",
    "playground-series-s6e7",

    # Tabular Playground Series
    "tabular-playground-series-dec-2021", "tabular-playground-series-jan-2022",
    "tabular-playground-series-feb-2022", "tabular-playground-series-mar-2022",
    "tabular-playground-series-may-2022", "tabular-playground-series-aug-2022",

    # Classic tabular
    "porto-seguro-safe-driver-prediction",
    "santander-customer-transaction-prediction",
    "telco-customer-churn",

    # Image
    "aerial-cactus-identification", "aptos2019-blindness-detection",
    "dog-breed-identification", "denoising-dirty-documents",
    "histopathologic-cancer-detection", "cassava-leaf-disease-classification",

    # NLP/Text
    "jigsaw-toxic-comment-classification-challenge",
    "feedback-prize-effectiveness", "commonlit-readability-prize",
    "quora-insincere-questions-classification",

    # Time Series
    "web-traffic-time-series-forecasting",

    # Multi-modal / Other
    "petfinder-pawpularity-score",
    "multi-modal-gesture-recognition",

    # MLE-Bench specific (from prepared data)
    "leaf-classification", "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",

    # Additional MLE-Bench
    "allstate-claims-severity", "mercedes-benz-greener-manufacturing",
    "rossmann-store-sales", "favorita-grocery-sales-forecasting",
    "m5-forecasting-accuracy",
    "siim-isic-melanoma-classification",
    "ranzcr-clip-catheter-line-classification",
    "google-quest-challenge",
    "feedback-prize-english-language-learning",
    "commonlit-evaluate-student-summaries",
    "jigsaw-multilingual-toxic-comment-classification",
    "jigsaw-unintended-bias-in-toxicity-classification",
    "tensorflow-great-barrier-reef",
    "rsna-2022-cervical-spine-fracture-detection",
    "chaii-hindi-and-tamil-question-answering",
]

downloaded = []
failed = []
skipped = []

for comp in COMPETITIONS:
    target_dir = os.path.join(DATA_DIR, comp)
    train_csv = os.path.join(target_dir, "train.csv")
    if os.path.exists(train_csv):
        skipped.append(comp)
        continue

    print(f"{comp}...", end=" ", flush=True)
    try:
        os.makedirs(target_dir, exist_ok=True)
        r = subprocess.run(
            ["kaggle", "competitions", "download", "-c", comp, "-p", target_dir],
            capture_output=True, text=True, timeout=60
        )
        if r.returncode == 0:
            for f in os.listdir(target_dir):
                if f.endswith(".zip"):
                    with zipfile.ZipFile(os.path.join(target_dir, f)) as zf:
                        zf.extractall(target_dir)
            print("OK")
            downloaded.append(comp)
        elif "403" in (r.stderr + r.stdout) or "404" in (r.stderr + r.stdout):
            print("403/404")
            failed.append(comp)
            shutil.rmtree(target_dir, ignore_errors=True)
        elif "accept" in (r.stderr + r.stdout).lower():
            print("NEEDS_ACCEPT")
            failed.append(comp)
            shutil.rmtree(target_dir, ignore_errors=True)
        else:
            print("ERR: " + r.stderr[:80].replace("\n", " "))
            failed.append(comp)
            shutil.rmtree(target_dir, ignore_errors=True)
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        failed.append(comp)
    except Exception as e:
        print("EXC: " + str(e)[:80])
        failed.append(comp)

print("")
print("=== DONE ===")
print("Skipped (existing): " + str(len(skipped)))
print("Downloaded: " + str(len(downloaded)))
print("Failed: " + str(len(failed)))
for c in downloaded:
    print("  + " + c)
for c in failed:
    print("  - " + c)

# Count total data dirs
all_dirs = [d for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d))
            and os.path.exists(os.path.join(DATA_DIR, d, "train.csv"))]
print("")
print("Total competitions with train.csv: " + str(len(all_dirs)))
