#!/bin/bash
cd /hpc2hdd/home/aimslab
nohup python3 gpu_train_v3.py ps5e4 --gpu-device 0 > results/log_ps5e4_v3_b2.txt 2>&1 &
echo "ps5e4 PID $!"
nohup python3 gpu_train_v3.py ps5e5 --gpu-device 0 > results/log_ps5e5_v3_b2.txt 2>&1 &
echo "ps5e5 PID $!"
echo "DONE"
