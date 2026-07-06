#!/usr/bin/env python3
"""Download submissions from server and grade them with MLE-Bench."""
import os, sys, base64, json, time
from pathlib import Path

os.environ['GPU_SSH_PASSWORD'] = '31PFmLLb1f'
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from hpc_connect import hpc_connect

HOME = '/hpc2hdd/home/aimslab'
RESULTS = f'{HOME}/jinghw/scripts/gpu_tra/mlebench_proper_results'
GRADING_DIR = SCRIPTS_DIR / 'mlebench_grading'

REPO_DIR = Path(r'D:\桌面\codex\科研港科技\external-projects\mle-bench')
sys.path.insert(0, str(REPO_DIR))
from mlebench.grade import grade_csv
from mlebench.registry import registry

# Write downloader to server once
downloader = '''
import sys, base64
path = sys.argv[1]
data = base64.b64encode(open(path, 'rb').read()).decode()
print(data)
'''

def ensure_downloader(ssh):
    _, stdout, stderr = ssh.exec_command('cat > /tmp/dl.py << "EOF"\n' + downloader + '\nEOF')
    err = stderr.read().decode().strip()
    return not err

def download_file(ssh, remote_path, local_path):
    _, stdout, stderr = ssh.exec_command(f'python3 /tmp/dl.py {remote_path}')
    b64 = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if b64 and not err:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(base64.b64decode(b64))
        return os.path.getsize(local_path)
    return 0

def grade_one(csv_path, comp_id):
    comp = registry.get_competition(comp_id)
    return grade_csv(Path(csv_path), comp)


def main():
    competitions = {
        'spaceship-titanic': ['42', '43', '44'],
        'tabular-playground-series-dec-2021': ['43', '44'],  # s42 had wrong target
        'tabular-playground-series-may-2022': ['42', '43', '44'],
    }

    ssh = hpc_connect('87739')
    ensure_downloader(ssh)

    all_reports = []

    for comp_id, seeds in competitions.items():
        print(f'\n{"="*60}')
        print(f'  {comp_id}')
        print(f'{"="*60}')

        for seed in seeds:
            remote = f'{RESULTS}/{comp_id}/submission_s{seed}.csv'
            local = GRADING_DIR / comp_id / f'submission_s{seed}.csv'

            size = download_file(ssh, remote, str(local))
            if size == 0:
                print(f'  s{seed}: DOWNLOAD FAILED')
                continue
            print(f'  s{seed}: {size} bytes downloaded')

            # Check content
            import pandas as pd
            sub = pd.read_csv(local)
            print(f'    Shape: {sub.shape}, Columns: {list(sub.columns)}')

            # Grade
            try:
                report = grade_one(str(local), comp_id)
                medal = 'GOLD' if report.gold_medal else 'SILVER' if report.silver_medal else 'BRONZE' if report.bronze_medal else 'NONE'
                above = 'ABOVE' if report.above_median else 'BELOW'
                print(f'    Score: {report.score:.5f}, Medal: {medal}, {above} median ({report.median_threshold:.5f})')
                all_reports.append(report)
            except Exception as e:
                print(f'    GRADE ERROR: {e}')

    ssh.close()

    # Summary
    print(f'\n{"="*60}')
    print(f'  FINAL SUMMARY')
    print(f'{"="*60}')
    for r in all_reports:
        medal = 'GOLD' if r.gold_medal else 'SILVER' if r.silver_medal else 'BRONZE' if r.bronze_medal else '-'
        print(f'  {r.competition_id:<45s} {r.score:>10.5f}  {medal:>6s}  {"> median" if r.above_median else "< median"}')

    # Save JSON
    summary = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'reports': [
            {
                'competition_id': r.competition_id,
                'score': r.score,
                'medal': 'GOLD' if r.gold_medal else 'SILVER' if r.silver_medal else 'BRONZE' if r.bronze_medal else 'NONE',
                'above_median': r.above_median,
                'valid': r.valid_submission,
                'thresholds': {
                    'gold': r.gold_threshold,
                    'silver': r.silver_threshold,
                    'bronze': r.bronze_threshold,
                    'median': r.median_threshold,
                }
            } for r in all_reports
        ]
    }
    out_path = GRADING_DIR / 'grading_results.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved to: {out_path}')


if __name__ == '__main__':
    main()
