#!/usr/bin/env python3
"""Batch download ALL 75 MLE-Bench split tasks to mlebench_raw_data.
Run AFTER accept_kaggle_rules_all75.py has completed.
Runs on GPU server via SSH.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

BASE = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
RAW = BASE / "mlebench_raw_data"
LOGS = BASE / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

SPLIT75 = [
    "3d-object-detection-for-autonomous-vehicles",
    "AI4Code",
    "aerial-cactus-identification",
    "alaska2-image-steganalysis",
    "aptos2019-blindness-detection",
    "billion-word-imputation",
    "bms-molecular-translation",
    "cassava-leaf-disease-classification",
    "cdiscount-image-classification-challenge",
    "chaii-hindi-and-tamil-question-answering",
    "champs-scalar-coupling",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "facebook-recruiting-iii-keyword-extraction",
    "freesound-audio-tagging-2019",
    "google-quest-challenge",
    "google-research-identify-contrails-reduce-global-warming",
    "h-and-m-personalized-fashion-recommendations",
    "herbarium-2020-fgvc7",
    "herbarium-2021-fgvc8",
    "herbarium-2022-fgvc9",
    "histopathologic-cancer-detection",
    "hms-harmful-brain-activity-classification",
    "hotel-id-2021-fgvc8",
    "hubmap-kidney-segmentation",
    "icecube-neutrinos-in-deep-ice",
    "imet-2020-fgvc7",
    "inaturalist-2019-fgvc6",
    "iwildcam-2019-fgvc6",
    "iwildcam-2020-fgvc7",
    "jigsaw-toxic-comment-classification-challenge",
    "jigsaw-unintended-bias-in-toxicity-classification",
    "kuzushiji-recognition",
    "leaf-classification",
    "learning-agency-lab-automated-essay-scoring-2",
    "lmsys-chatbot-arena",
    "mlsp-2013-birds",
    "multi-modal-gesture-recognition",
    "new-york-city-taxi-fare-prediction",
    "nfl-player-contact-detection",
    "nomad2018-predict-transparent-conductors",
    "osic-pulmonary-fibrosis-progression",
    "petfinder-pawpularity-score",
    "plant-pathology-2020-fgvc7",
    "plant-pathology-2021-fgvc8",
    "playground-series-s3e18",
    "predict-volcanic-eruptions-ingv-oe",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "rsna-2022-cervical-spine-fracture-detection",
    "rsna-breast-cancer-detection",
    "rsna-miccai-brain-tumor-radiogenomic-classification",
    "seti-breakthrough-listen",
    "siim-covid19-detection",
    "siim-isic-melanoma-classification",
    "smartphone-decimeter-2022",
    "spooky-author-identification",
    "stanford-covid-vaccine",
    "statoil-iceberg-classifier-challenge",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "tensorflow-speech-recognition-challenge",
    "tensorflow2-question-answering",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "tgs-salt-identification-challenge",
    "the-icml-2013-whale-challenge-right-whale-redux",
    "tweet-sentiment-extraction",
    "us-patent-phrase-to-phrase-matching",
    "uw-madison-gi-tract-image-segmentation",
    "ventilator-pressure-prediction",
    "vesuvius-challenge-ink-detection",
    "vinbigdata-chest-xray-abnormalities-detection",
    "whale-categorization-playground",
]


def has_complete_data(comp_dir: Path) -> bool:
    """Check if directory already has train + test files."""
    train = list(comp_dir.glob("train*"))
    test = list(comp_dir.glob("test*"))
    return len(train) > 0 and len(test) > 0


def download_one(comp: str) -> dict:
    """Download a single competition. Returns status dict."""
    comp_dir = RAW / comp
    comp_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "competition": comp,
        "status": "unknown",
        "files_before": 0,
        "files_after": 0,
        "elapsed_sec": 0,
        "error": None,
    }

    # Skip if already has data
    if has_complete_data(comp_dir):
        result["status"] = "already_complete"
        result["files_before"] = len(list(comp_dir.iterdir()))
        result["files_after"] = result["files_before"]
        return result

    result["files_before"] = len(list(comp_dir.iterdir()))
    t0 = time.time()

    try:
        proc = subprocess.run(
            ["kaggle", "competitions", "download", "-c", comp, "-p", str(comp_dir)],
            capture_output=True, text=True, timeout=600,
        )
        result["elapsed_sec"] = time.time() - t0
        result["files_after"] = len(list(comp_dir.iterdir()))

        if proc.returncode == 0:
            result["status"] = "downloaded"
        elif "403" in proc.stderr or "Forbidden" in proc.stderr:
            result["status"] = "forbidden_403"
            result["error"] = "Rules not accepted"
        elif "404" in proc.stderr:
            result["status"] = "not_found_404"
            result["error"] = "Competition not found"
        else:
            result["status"] = f"error_{proc.returncode}"
            result["error"] = proc.stderr[:300]
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Download timed out after 600s"
        result["elapsed_sec"] = time.time() - t0
    except Exception as e:
        result["status"] = "exception"
        result["error"] = str(e)[:300]
        result["elapsed_sec"] = time.time() - t0

    # Mark as extracted if download succeeded
    if result["status"] == "downloaded":
        (comp_dir / ".extracted.ok").touch()

    return result


def main():
    results = []
    log_file = LOGS / f"batch_download_all75_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

    print(f"Starting batch download of {len(SPLIT75)} competitions")
    print(f"Target: {RAW}")
    print(f"Log: {log_file}")

    for i, comp in enumerate(SPLIT75):
        print(f"\n[{i+1}/{len(SPLIT75)}] {comp} ...", end=" ", flush=True)
        result = download_one(comp)
        results.append(result)
        print(f"{result['status']} ({result['elapsed_sec']:.0f}s)")

        # Append to log
        with open(log_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    # Summary
    status_counts = {}
    for r in results:
        s = r["status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE: {len(results)} competitions")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    # Save full results
    summary_path = LOGS / f"batch_download_all75_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps({
        "total": len(results),
        "status_counts": status_counts,
        "results": results,
    }, indent=2))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
