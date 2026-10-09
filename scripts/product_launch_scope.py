"""Local source audit only; no databases, credentials or deployment writes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'artifacts/product-launch-20261008'
PUBLISHED = Path('D:/EV12/system-repair-20261008-gates')
PREFIXES = {
    'web/research-agent-workstation/src': 'web/src',
    'src/evomind_runtime': 'runtime/evomind_runtime',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    target = EVIDENCE / 'baseline.json'
    if target.exists():
        raise RuntimeError('baseline_already_frozen')
    rows = []
    for prefix, published in PREFIXES.items():
        for path in sorted((ROOT / prefix).rglob('*')):
            if path.suffix not in {'.py', '.ts', '.tsx', '.mjs', '.css'} or '__pycache__' in path.parts:
                continue
            name = path.relative_to(ROOT).as_posix()
            copy = EVIDENCE / 'before' / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copy)
            rows.append({'path': name, 'before': sha(path), 'published': sha(PUBLISHED / published / path.relative_to(ROOT / prefix))})
    for name in ('scripts/invitation_release_transaction.py',):
        path = ROOT / name
        copy = EVIDENCE / 'before' / name
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)
        rows.append({'path': name, 'before': sha(path), 'published': None})
    target.write_text(json.dumps({'published_receipt_sha256': sha(PUBLISHED / 'source-receipt.json'), 'files': rows}, indent=2), encoding='utf-8')
    print(json.dumps({'baseline': str(target), 'files': len(rows), 'preexisting_differences': sum(r['before'] != r['published'] for r in rows)}))


if __name__ == '__main__':
    main()
