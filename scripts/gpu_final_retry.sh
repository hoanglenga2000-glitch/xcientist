#!/bin/bash
# Final retry: ALL remaining empty/corrupt dirs with 1800s timeout
BASE="$HOME/jinghw/scripts/gpu_tra/mlebench_raw_data"
LOG="$HOME/jinghw/scripts/gpu_tra/final_retry.log"

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }

# All competitions that have 0 extracted files (corrupt zip or empty)
# We clean corrupt zips first, then re-download
ALL_REMAINING=(
    "siim-covid19-detection"
    "histopathologic-cancer-detection"
    "vinbigdata-chest-xray-abnormalities-detection"
    "iwildcam-2019-fgvc6"
    "hubmap-kidney-segmentation"
    "kuzushiji-recognition"
    "tensorflow-speech-recognition-challenge"
    "google-research-identify-contrails-reduce-global-warming"
    "hms-harmful-brain-activity-classification"
    "hotel-id-2021-fgvc8"
    "nfl-player-contact-detection"
    "osic-pulmonary-fibrosis-progression"
    "ranzcr-clip-catheter-line-classification"
    "rsna-miccai-brain-tumor-radiogenomic-classification"
    "tensorflow2-question-answering"
    "rsna-2022-cervical-spine-fracture-detection"
    "rsna-breast-cancer-detection"
    "siim-isic-melanoma-classification"
    "uw-madison-gi-tract-image-segmentation"
    "facebook-recruiting-iii-keyword-extraction"
    "learning-agency-lab-automated-essay-scoring-2"
    "stanford-covid-vaccine"
    "text-normalization-challenge-russian-language"
    "tgs-salt-identification-challenge"
    "plant-pathology-2020-fgvc7"
    "statoil-iceberg-classifier-challenge"
    "ventilator-pressure-prediction"
    "mlsp-2013-birds"
    "whale-categorization-playground"
    "herbarium-2020-fgvc7"
)

log "=== FINAL RETRY: ${#ALL_REMAINING[@]} dirs, ALL with 1800s timeout ==="

# Phase 1: Clean all corrupt zips
log "Cleaning corrupt zips..."
for comp in "${ALL_REMAINING[@]}"; do
    rm -f "$BASE/$comp"/*.zip 2>/dev/null
done
log "Cleaned."

# Phase 2: Download all with generous timeout
OK=0; FAIL=0
for comp in "${ALL_REMAINING[@]}"; do
    target="$BASE/$comp"
    mkdir -p "$target"

    # Check if already has usable files (non-zip + non-empty)
    usable=$(find "$target" -type f ! -name '*.zip' -size +0c 2>/dev/null | wc -l)
    if [ "$usable" -gt 0 ]; then
        log "SKIP $comp ($usable usable files)"
        continue
    fi

    log "DL $comp"
    if timeout 1800 kaggle competitions download -c "$comp" -p "$target" -q 2>&1; then
        # Extract all zips
        for z in "$target"/*.zip; do
            [ -f "$z" ] && unzip -o "$z" -d "$target" 2>/dev/null && rm "$z"
        done
        # If unzip failed, try python
        if [ -f "$target"/*.zip 2>/dev/null ]; then
            python3 -c "
import zipfile, glob, os
for z in glob.glob('$target/*.zip'):
    try:
        with zipfile.ZipFile(z) as zf: zf.extractall('$target')
        os.remove(z)
    except: pass
" 2>/dev/null
        fi
        fcount=$(find "$target" -type f | wc -l)
        log "OK $comp ($fcount files)"
        OK=$((OK+1))
    else
        log "FAIL $comp"
        FAIL=$((FAIL+1))
        rm -f "$target"/*.zip 2>/dev/null
    fi
    sleep 2
done

log "=== ALL DONE: OK=$OK FAIL=$FAIL ==="
