#!/usr/bin/env python3
import subprocess, sys, os, json, time

WORK = '/hpc2hdd/home/aimslab/jinghw'
HOME = '/hpc2hdd/home/aimslab'
PYTHON = '/opt/miniconda3/bin/python'
SCRIPT = WORK + '/scripts/gpu_train_v3.py'

QUEUE = [
    ('digit_recognizer', '--depth=10 --l2=3 --lr=0.04 --iter=5000 --n-folds=5'),
    ('spaceship_titanic', '--depth=6 --l2=5 --lr=0.03 --iter=3000 --n-folds=5'),
    ('titanic', '--depth=6 --l2=5 --lr=0.03 --iter=3000 --n-folds=8'),
    ('store_sales', '--depth=6 --l2=3 --lr=0.03 --iter=2000 --n-folds=3'),
    ('bike_sharing_demand', '--depth=6 --l2=5 --lr=0.03 --iter=2000 --n-folds=5 --no-log1p'),
    ('ps4e1', '--depth=6 --l2=5 --lr=0.03 --iter=2000 --n-folds=5'),
    ('ps4e7', '--depth=6 --l2=5 --lr=0.03 --iter=800 --n-folds=5'),
    ('ps6e6', '--depth=6 --l2=5 --lr=0.03 --iter=800 --n-folds=5'),
    ('porto_seguro', '--depth=5 --l2=5 --lr=0.02 --iter=500 --n-folds=3'),
    ('ps5e2', '--depth=6 --l2=5 --lr=0.03 --iter=800 --n-folds=5'),
    ('tps_jan2022', '--depth=6 --l2=5 --lr=0.03 --iter=800 --n-folds=5'),
    ('tps_mar2022', '--depth=6 --l2=5 --lr=0.03 --iter=800 --n-folds=5'),
]

LOG_FILE = WORK + '/logs/batch_trainer.log'

def log(msg):
    timestamp = time.strftime('%H:%M:%S')
    line = '[' + timestamp + '] ' + msg
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + chr(10))

log('Batch trainer starting')
log('Queue: ' + str(len(QUEUE)) + ' comps')
log('Work dir: ' + WORK)

success_count = 0
for task, opts in QUEUE:
    log('START: ' + task)
    t0 = time.time()
    cmd = [PYTHON, SCRIPT, task] + opts.split()
    result = subprocess.run(cmd, cwd=WORK, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    if result.returncode == 0:
        for line in result.stdout.split(chr(10)):
            if 'GATE: PASS' in line:
                success_count += 1
        log('DONE: ' + task + ' (' + str(int(elapsed)) + 's)')
    else:
        log('FAIL: ' + task + ' code=' + str(result.returncode))
        log('STDERR: ' + result.stderr[:200])

log('ALL DONE. ' + str(success_count) + '/' + str(len(QUEUE)) + ' GATE PASS.')

log('Auto-submitting...')
for task, _ in QUEUE:
    result_file = WORK + '/results/v3_result_' + task + '.json'
    if os.path.exists(result_file):
        with open(result_file) as f:
            data = json.load(f)
        if data.get('gate_passed'):
            sub_file = data.get('submission_path', '')
            if sub_file and os.path.exists(sub_file):
                comp_name = task.replace('_', '-')
                subprocess.run(['kaggle', 'competitions', 'submit', '-c', comp_name, '-f', sub_file, '-m', 'batch_v3'], cwd=WORK, timeout=120)
                log('SUBMITTED: ' + task)
