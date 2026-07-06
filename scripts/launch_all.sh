#!/bin/bash
# Launch all remaining training jobs in background
cd /hpc2hdd/home/aimslab

echo "=== Launching all training jobs ==="

# Titanic V2 ensemble on 87557 (use ssh to other node)
# Actually all jobs share same filesystem, just need different GPU devices
# But each job instance has its own GPU

# On this node: start available jobs on different GPUs
# GPU 0: titanic V2 ensemble 10fold
nohup python3 gpu_train_v2.py titanic --gpu-device 0 --n-folds 10 > results/log_titanic_v2_10fold.txt 2>&1 &
echo "titanic V2 10fold: PID $!"

# GPU 1: ps4e1 V2 ensemble 7fold
nohup python3 gpu_train_v2.py ps4e1 --gpu-device 1 --n-folds 7 > results/log_ps4e1_v2_7fold.txt 2>&1 &
echo "ps4e1 V2 7fold: PID $!"

# GPU 2: tps_aug2022 V3
nohup python3 gpu_train_v3.py tps_aug2022 --gpu-device 2 > results/log_tps_aug2022_v3.txt 2>&1 &
echo "tps_aug2022 V3: PID $!"

# GPU 3: tps_feb2022 V3
nohup python3 gpu_train_v3.py tps_feb2022 --gpu-device 3 > results/log_tps_feb2022_v3.txt 2>&1 &
echo "tps_feb2022 V3: PID $!"

echo "=== All jobs launched ==="
