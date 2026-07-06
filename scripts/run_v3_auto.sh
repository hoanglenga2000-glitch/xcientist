#!/bin/bash
# Auto-launch v3 trainer after v2 finishes
# Run: nohup bash run_v3_after_v2.sh > run_v3_launcher.log 2>&1 &

cd /hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra

# Wait for v2 to finish (check every 5 min)
echo "Waiting for v2 trainer to finish..."
while pgrep -f "mlebench_catboost_trainer_v2" > /dev/null 2>&1; do
    echo "$(date): v2 still running..."
    sleep 300
done

echo "$(date): v2 finished! Starting v3..."
/opt/miniconda3/bin/python mlebench_catboost_trainer_v3.py > mlebench_trainer_v3_87729.log 2>&1

echo "$(date): v3 completed!"
