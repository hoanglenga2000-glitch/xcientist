#!/bin/bash
# Retry all corrupt/incomplete downloads with proper timeout
BASE="$HOME/jinghw/scripts/gpu_tra/mlebench_raw_data"
LOG="$HOME/jinghw/scripts/gpu_tra/retry_corrupt.log"

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }

# Small files (<300MB) - 300s timeout
SMALL=(
    "facebook-recruiting-iii-keyword-extraction"
    "learning-agency-lab-automated-essay-scoring-2"
    "stanford-covid-vaccine"
    "text-normalization-challenge-russian-language"
    "tgs-salt-identification-challenge"
    "tensorflow2-question-answering"
    "rsna-miccai-brain-tumor-radiogenomic-classification"
    "plant-pathology-2020-fgvc7"
    "hms-harmful-brain-activity-classification"
    "hotel-id-2021-fgvc8"
    "nfl-player-contact-detection"
    "statoil-iceberg-classifier-challenge"
    "ventilator-pressure-prediction"
    "mlsp-2013-birds"
    "ranzcr-clip-catheter-line-classification"
    "google-research-identify-contrails-reduce-global-warming"
    "osic-pulmonary-fibrosis-progression"
    "whale-categorization-playground"
)

# Large files (>300MB) - 1800s timeout
LARGE=(
    "siim-covid19-detection"
    "histopathologic-cancer-detection"
    "vinbigdata-chest-xray-abnormalities-detection"
    "iwildcam-2019-fgvc6"
    "hubmap-kidney-segmentation"
    "kuzushiji-recognition"
    "tensorflow-speech-recognition-challenge"
)

log "=== RETRY CORRUPT: SMALL (${#SMALL[@]}) + LARGE (${#LARGE[@]}) ==="

# Phase 1: Delete all corrupt zips
log "--- Cleaning corrupt zips ---"
for comp in "${SMALL[@]}" "${LARGE[@]}"; do
    target="$BASE/$comp"
    rm -f "$target"/*.zip 2>/dev/null
done
log "Cleaned."

# Phase 2: Small files (300s timeout)
OK=0; FAIL=0
for comp in "${SMALL[@]}"; do
    target="$BASE/$comp"
    mkdir -p "$target"
    count=$(ls "$target" 2>/dev/null | grep -v '\.zip$' | wc -l)
    if [ "$count" -gt 0 ] && [ -s "$target"/* 2>/dev/null ]; then
        log "SKIP $comp ($count usable files)"
        continue
    fi
    log "DL $comp"
    if timeout 300 kaggle competitions download -c "$comp" -p "$target" -q 2>&1; then
        for z in "$target"/*.zip; do
            [ -f "$z" ] && unzip -o "$z" -d "$target" 2>/dev/null && rm "$z"
        done
        fcount=$(ls "$target" 2>/dev/null | wc -l)
        log "OK $comp ($fcount files)"
        OK=$((OK+1))
    else
        log "FAIL $comp"
        FAIL=$((FAIL+1))
        rm -f "$target"/*.zip 2>/dev/null
    fi
    sleep 1
done
log "--- SMALL DONE: OK=$OK FAIL=$FAIL ---"

# Phase 3: Large files (1800s = 30 min timeout)
OK2=0; FAIL2=0
for comp in "${LARGE[@]}"; do
    target="$BASE/$comp"
    mkdir -p "$target"
    count=$(ls "$target" 2>/dev/null | grep -v '\.zip$' | wc -l)
    if [ "$count" -gt 0 ] && [ -s "$target"/* 2>/dev/null ]; then
        log "SKIP $comp ($count usable files)"
        continue
    fi
    log "DL-LARGE $comp (30min timeout)"
    if timeout 1800 kaggle competitions download -c "$comp" -p "$target" -q 2>&1; then
        for z in "$target"/*.zip; do
            [ -f "$z" ] && unzip -o "$z" -d "$target" 2>/dev/null && rm "$z"
        done
        fcount=$(ls "$target" 2>/dev/null | wc -l)
        log "OK-LARGE $comp ($fcount files)"
        OK2=$((OK2+1))
    else
        log "FAIL-LARGE $comp"
        FAIL2=$((FAIL2+1))
        rm -f "$target"/*.zip 2>/dev/null
    fi
    sleep 1
done

log "=== ALL DONE: SMALL_OK=$OK SMALL_FAIL=$FAIL LARGE_OK=$OK2 LARGE_FAIL=$FAIL2 ==="
