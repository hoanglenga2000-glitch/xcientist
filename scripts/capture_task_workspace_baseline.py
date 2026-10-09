"""Read-only source inventory; writes evidence only, never a release candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = Path('D:/EV12/system-repair-20261008-gates')
EVIDENCE = ROOT / 'artifacts/advanced-tools-migration-20261008'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def capture():
    target = EVIDENCE / 'baseline.json'
    if target.exists():
        raise SystemExit('Existing baseline preserved; use it, do not replace it.')
    rows = []
    for prefix, deployed in [('web/research-agent-workstation/src', 'web/src'), ('src/evomind_runtime', 'runtime/evomind_runtime')]:
        for path in sorted((ROOT / prefix).rglob('*')):
            if path.suffix not in {'.py', '.ts', '.tsx', '.mjs', '.css'} or '__pycache__' in path.parts:
                continue
            relative = path.relative_to(ROOT / prefix)
            rows.append({'path': path.relative_to(ROOT).as_posix(), 'before': digest(path),
                         'published': digest(PUBLISHED / deployed / relative)})
    with urllib.request.urlopen('https://evomind.zhjjq.tech/api/healthz', timeout=12) as response:
        health = json.load(response)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'production_build': health.get('build_id'), 'production_status': health.get('status'),
        'published_source_receipt_sha256': digest(PUBLISHED / 'source-receipt.json'),
        'git_status_before': subprocess.check_output(['git', 'status', '--porcelain=v1', '--untracked-files=no'], cwd=ROOT, text=True).splitlines(),
        'files': rows, 'production_mutated': False}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'files': len(rows), 'evidence': str(target), 'production_build': health.get('build_id')}))


if __name__ == '__main__':
    capture()
