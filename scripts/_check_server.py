#!/usr/bin/env python3
import os, sys
os.environ['GPU_SSH_PASSWORD'] = '31PFmLLb1'
from hpc_connect import hpc_exec

cmds = [
    ("HOME", "echo $HOME"),
    ("GPU_TRA files", "ls ~/jinghw/scripts/gpu_tra/ 2>/dev/null || echo 'NOT FOUND'"),
    ("MLEBench results", "ls ~/jinghw/scripts/gpu_tra/mlebench_proper_results/ 2>/dev/null || echo 'EMPTY'"),
    ("MLEBench data", "ls ~/.cache/mle-bench/data/ 2>/dev/null || echo 'NOT FOUND'"),
    ("GPU", "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo 'NO GPU'"),
    ("Python path", "ls /opt/miniconda3/bin/python && /opt/miniconda3/bin/python --version"),
    ("Disk", "df -h /hpc2hdd 2>/dev/null | tail -1"),
]

for label, cmd in cmds:
    out, err = hpc_exec('87739', cmd)
    print(f"=== {label} ===")
    if out: print(out.strip())
    if err: print(f"ERR: {err.strip()}")
    print()
