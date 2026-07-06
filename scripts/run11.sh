#!/bin/bash
# run11: Fix string target encoding bug for broken competitions
# Competitions: feb-2022 (multi-class bacteria), s6e2 (Heart Disease), s6e3 (Churn), s4e7 (Response, 3-fold)
cd ~/jinghw/scripts/gpu_tra
echo "=== RUN11 START: $(date) ==="
echo "Competitions: feb-2022, s6e2, s6e3, s4e7"
echo "Script hash: $(md5sum ~/gpu_batch_trainer_v1.py | cut -d' ' -f1)"

python3 ~/gpu_batch_trainer_v1.py \
    --competitions tabular-playground-series-feb-2022 playground-series-s6e2 playground-series-s6e3 playground-series-s4e7

echo "=== RUN11 END: $(date) ==="
