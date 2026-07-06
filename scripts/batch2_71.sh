#!/bin/bash
cd /hpc2hdd/home/aimslab
nohup python3 gpu_train_v3.py ps5e1 --gpu-device 0 > results/log_ps5e1_v3_b2.txt 2>&1 &
echo "ps5e1 PID $!"
nohup python3 gpu_train_v3.py ps5e2 --gpu-device 1 > results/log_ps5e2_v3_b2.txt 2>&1 &
echo "ps5e2 PID $!"
nohup python3 gpu_train_v3.py tps_feb2022 --gpu-device 2 --n-folds 3 > results/log_tps_feb2022_v3_b2.txt 2>&1 &
echo "tps_feb2022 PID $!"
nohup python3 gpu_train_v3.py tps_may2022 --gpu-device 3 --n-folds 3 > results/log_tps_may2022_v3_b2.txt 2>&1 &
echo "tps_may2022 PID $!"
echo "DONE"
