#!/usr/bin/env python3
"""Launch script for MLE-Bench proper trainer on server."""
import os, sys
os.environ['GPU_SSH_PASSWORD'] = '31PFmLLb1f'
sys.path.insert(0, r'D:\桌面\codex\科研港科技\scripts')
from hpc_connect import hpc_exec

# Upload and launch the trainer
out, err = hpc_exec('87739', 'cd ~/jinghw/scripts/gpu_tra && nohup python3 mlebench_proper_trainer.py > mlebench_proper_nohup.log 2>&1 & echo PID=$!')
print('=== LAUNCH RESULT ===')
print(out)
if err:
    print('STDERR:', err)
