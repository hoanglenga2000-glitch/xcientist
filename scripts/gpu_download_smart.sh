#!/bin/bash
# Smart download: small first, skip known 403s, mark large for later
BASE="$HOME/jinghw/scripts/gpu_tra/mlebench_raw_data"
LOG="$HOME/jinghw/scripts/gpu_tra/download_smart.log"

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }

# Already have data: SKIP
# Known 403/offline: icml-whale, playground-s3e18, histopathologic-cancer, detecting-insults

SMALL_COMPS=(
    "siim-covid19-detection"
    "ranzcr-clip-catheter-line-classification"
    "rsna-miccai-brain-tumor-radiogenomic-classification"
    "stanford-covid-vaccine"
    "hms-harmful-brain-activity-classification"
    "osic-pulmonary-fibrosis-progression"
    "hotel-id-2021-fgvc8"
    "nfl-player-contact-detection"
    "smartphone-decimeter-2022"
    "freesound-audio-tagging-2019"
    "petfinder-pawpularity-score"
    "predict-volcanic-eruptions-ingv-oe"
    "tensorflow2-question-answering"
    "google-research-identify-contrails-reduce-global-warming"
    "plant-pathology-2020-fgvc7"
    "plant-pathology-2021-fgvc8"
    "inaturalist-2019-fgvc6"
    "iwildcam-2019-fgvc6"
    "iwildcam-2020-fgvc7"
    "imet-2020-fgvc7"
    "herbarium-2020-fgvc7"
    "herbarium-2021-fgvc8"
    "herbarium-2022-fgvc9"
    "seti-breakthrough-listen"
    "vesuvius-challenge-ink-detection"
    "uw-madison-gi-tract-image-segmentation"
    "h-and-m-personalized-fashion-recommendations"
    "rsna-2022-cervical-spine-fracture-detection"
    "rsna-breast-cancer-detection"
    "siim-isic-melanoma-classification"
    "icecube-neutrinos-in-deep-ice"
)

# Large files that need special handling (>1GB, longer timeout)
LARGE_COMPS=(
    "kuzushiji-recognition"
    "tensorflow-speech-recognition-challenge"
    "hubmap-kidney-segmentation"
    "vinbigdata-chest-xray-abnormalities-detection"
    "ventilator-pressure-prediction"
    "whale-categorization-playground"
)

log "=== SMART DOWNLOAD: SMALL FILES FIRST (${#SMALL_COMPS[@]} comps) ==="
OK=0; SKIP=0; FAIL=0

for comp in "${SMALL_COMPS[@]}"; do
    target="$BASE/$comp"
    mkdir -p "$target"

    # Skip if already has non-zip data
    count=$(ls "$target" 2>/dev/null | grep -v '\.zip$' | wc -l)
    if [ "$count" -gt 0 ]; then
        log "SKIP $comp ($count files)"
        SKIP=$((SKIP+1))
        continue
    fi

    # Clean corrupt zips
    rm -f "$target"/*.zip 2>/dev/null

    log "DL $comp"
    if timeout 600 kaggle competitions download -c "$comp" -p "$target" -q 2>&1; then
        for z in "$target"/*.zip; do
            [ -f "$z" ] && unzip -o "$z" -d "$target" 2>/dev/null && rm "$z"
        done
        fcount=$(ls "$target" 2>/dev/null | wc -l)
        log "OK $comp ($fcount files)"
        OK=$((OK+1))
    else
        log "FAIL $comp"
        FAIL=$((FAIL+1))
    fi
    sleep 1
done

log "=== SMALL DONE: OK=$OK SKIP=$SKIP FAIL=$FAIL ==="

log "=== LARGE FILES (${#LARGE_COMPS[@]} comps, 900s timeout) ==="
for comp in "${LARGE_COMPS[@]}"; do
    target="$BASE/$comp"
    mkdir -p "$target"

    count=$(ls "$target" 2>/dev/null | grep -v '\.zip$' | wc -l)
    if [ "$count" -gt 0 ]; then
        log "SKIP $comp ($count files)"
        continue
    fi

    rm -f "$target"/*.zip 2>/dev/null

    log "DL-LARGE $comp"
    if timeout 900 kaggle competitions download -c "$comp" -p "$target" -q 2>&1; then
        for z in "$target"/*.zip; do
            [ -f "$z" ] && unzip -o "$z" -d "$target" 2>/dev/null && rm "$z"
        done
        fcount=$(ls "$target" 2>/dev/null | wc -l)
        log "OK $comp ($fcount files)"
    else
        log "FAIL-LARGE $comp"
    fi
    sleep 1
done

log "=== ALL DONE ==="
