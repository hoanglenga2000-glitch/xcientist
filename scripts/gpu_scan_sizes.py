#!/usr/bin/env python3
"""Scan all remaining empty competitions to check file count and total size."""
import subprocess

comps = [
    "detecting-insults-in-social-commentary", "freesound-audio-tagging-2019",
    "google-research-identify-contrails-reduce-global-warming",
    "h-and-m-personalized-fashion-recommendations", "herbarium-2020-fgvc7",
    "herbarium-2021-fgvc8", "herbarium-2022-fgvc9", "histopathologic-cancer-detection",
    "hms-harmful-brain-activity-classification", "hotel-id-2021-fgvc8",
    "hubmap-kidney-segmentation", "icecube-neutrinos-in-deep-ice", "imet-2020-fgvc7",
    "inaturalist-2019-fgvc6", "iwildcam-2019-fgvc6", "iwildcam-2020-fgvc7",
    "jigsaw-unintended-bias-in-toxicity-classification", "kuzushiji-recognition",
    "nfl-player-contact-detection", "osic-pulmonary-fibrosis-progression",
    "petfinder-pawpularity-score", "plant-pathology-2021-fgvc8",
    "playground-series-s3e18", "predict-volcanic-eruptions-ingv-oe",
    "ranzcr-clip-catheter-line-classification",
    "rsna-2022-cervical-spine-fracture-detection", "rsna-breast-cancer-detection",
    "rsna-miccai-brain-tumor-radiogenomic-classification", "seti-breakthrough-listen",
    "siim-isic-melanoma-classification", "smartphone-decimeter-2022",
    "tensorflow-speech-recognition-challenge", "tensorflow2-question-answering",
    "text-normalization-challenge-english-language", "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux", "uw-madison-gi-tract-image-segmentation",
    "vesuvius-challenge-ink-detection", "vinbigdata-chest-xray-abnormalities-detection",
]

for comp in comps:
    r = subprocess.run(["kaggle", "competitions", "files", "-c", comp],
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        err = (r.stderr + r.stdout)[:120].replace("\n", " ")
        if "404" in err:
            print(f"OFFLINE  {comp}")
        elif "403" in err:
            print(f"403      {comp}")
        else:
            print(f"ERR      {comp}: {err}")
    else:
        total = 0
        count = 0
        for line in r.stdout.split("\n"):
            parts = line.split()
            if len(parts) >= 2 and parts[-2].isdigit():
                total += int(parts[-2])
                count += 1
        gb = total / (1024**3)
        if gb < 1:
            print(f"SMALL    {comp}: {count}f, {total/(1024**2):.0f}MB")
        elif gb < 10:
            print(f"MEDIUM   {comp}: {count}f, {gb:.1f}GB")
        else:
            print(f"LARGE    {comp}: {count}f, {gb:.1f}GB")
