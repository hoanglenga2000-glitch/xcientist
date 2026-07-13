#!/usr/bin/env python3
"""Batch test Kaggle competition accessibility."""
import subprocess, json

slugs = [
    "aerial-cactus-identification", "aptos2019-blindness-detection",
    "denoising-dirty-documents", "detecting-insults-in-social-commentary",
    "dog-breed-identification", "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection", "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification", "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction", "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7", "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification", "siim-isic-melanoma-classification",
    "spooky-author-identification", "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022", "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language", "the-icml-2013-whale-challenge-right-whale-redux",
    "3d-object-detection-for-autonomous-vehicles", "AI4Code",
    "alaska2-image-steganalysis", "billion-word-imputation",
    "bms-molecular-translation", "cassava-leaf-disease-classification",
    "cdiscount-image-classification-challenge", "chaii-hindi-tamil-question-answering",
    "champs-scalar-coupling", "facebook-recruiting-iii-keyword-extraction",
    "freesound-audio-tagging-2019", "google-quest-challenge",
    "google-research-identify-contrails-reduce-global-warming", "h-and-m-personalized-fashion-recommendations",
    "herbarium-2020-fgvc7", "herbarium-2021-fgvc8", "herbarium-2022-fgvc9",
    "hms-harmful-brain-activity-classification", "hotel-id-2021-fgvc8",
    "hubmap-kidney-segmentation", "icecube-neutrinos-in-deep-ice",
    "imet-2020-fgvc7", "inaturalist-2019-fgvc6",
    "invasive-species-monitoring", "iwildcam-2019-fgvc6", "iwildcam-2020-fgvc7",
    "jigsaw-unintended-bias-in-toxicity-classification", "kuzushiji-recognition",
    "learning-agency-lab-automated-essay-scoring-2", "lmsys-chatbot-arena",
    "ml2021spring-hw2", "movie-review-sentiment-analysis-kernels-only",
    "multi-modal-gesture-recognition", "nfl-player-contact-detection",
    "osic-pulmonary-fibrosis-progression", "paddy-disease-classification",
    "petfinder-pawpularity-score", "plant-pathology-2021-fgvc8",
    "plant-seedlings-classification", "playground-series-s3e18",
    "predict-volcanic-eruptions-ingv-oe", "rsna-2022-cervical-spine-fracture-detection",
    "rsna-breast-cancer-detection", "rsna-miccai-brain-tumor-radiogenomic-classification",
    "seti-breakthrough-listen", "siim-covid19-detection",
    "smartphone-decimeter-2022", "spaceship-titanic",
    "stanford-covid-vaccine", "statoil-iceberg-classifier-challenge",
    "tensorflow-speech-recognition-challenge", "tensorflow2-question-answering",
    "tgs-salt-identification-challenge", "tweet-sentiment-extraction",
    "us-patent-phrase-to-phrase-matching", "uw-madison-gi-tract-image-segmentation",
    "ventilator-pressure-prediction", "vesuvius-challenge-ink-detection",
    "vinbigdata-chest-xray-abnormalities-detection", "whale-categorization-playground",
]

import os, shutil

ok, bad = [], []
for slug in slugs:
    try:
        r = subprocess.run(["kaggle", "competitions", "download", "-c", slug, "-p", "/tmp/mle_test"],
                           capture_output=True, text=True, timeout=30)
        if os.path.exists("/tmp/mle_test"):
            shutil.rmtree("/tmp/mle_test", ignore_errors=True)
        if "Downloading" in r.stdout:
            ok.append(slug)
            print(f"OK:{slug}", flush=True)
        else:
            bad.append(slug)
            print(f"NO:{slug}", flush=True)
    except subprocess.TimeoutExpired:
        ok.append(slug)
        print(f"OK_SLOW:{slug}", flush=True)
        if os.path.exists("/tmp/mle_test"):
            shutil.rmtree("/tmp/mle_test", ignore_errors=True)
    except Exception as e:
        bad.append(slug)
        print(f"ERR:{slug}:{e}", flush=True)

print("___RESULTS___")
print("ACCESSIBLE:", json.dumps(ok))
print("BLOCKED:", json.dumps(bad))
print(f"Total accessible: {len(ok)}/{len(slugs)}")
