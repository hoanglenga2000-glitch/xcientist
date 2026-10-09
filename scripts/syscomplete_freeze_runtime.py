"""Freeze reviewed EvoMind files for both service soak and release assembly."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('syscomplete_assembler', ROOT / 'scripts/syscomplete_release_assemble.py')
ASSEMBLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSEMBLER)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    parent = Path('D:/EV12/ack07-11727a59')
    ASSEMBLER.BASE.verify_parent(parent)
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / 'evomind_runtime'
    output.mkdir()
    rows, attribution = [], []
    for name in sorted(ASSEMBLER.RUNTIME_ALLOWLIST):
        source = ASSEMBLER.BASE.checked_path(ROOT, 'src/evomind_runtime/' + name)
        target = output / name
        previous = parent / 'runtime/evomind_runtime' / name
        before = previous.read_bytes() if previous.is_file() else None
        start = ROOT / 'artifacts/system-completeness-20260907/step1-baseline/src/evomind_runtime' / name
        if start.is_file() and before is not None:
            if start.read_text(encoding='utf-8').replace('\r\n', '\n') != before.decode('utf-8').replace('\r\n', '\n'):
                raise ValueError('unreviewed_runtime_baseline_drift_' + name)
            attribution.append({'file': name, 'starting_snapshot_matches_parent': True})
        data = source.read_bytes()
        with target.open('xb') as handle:
            handle.write(data)
        rows.append({'path': 'evomind_runtime/' + name, 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(),
            'before_sha256': hashlib.sha256(before).hexdigest() if before is not None else None})
    manifest = {'schema': 'evomind.syscomplete_runtime_delta.v1', 'frozen': True,
        'model': 'gpt-5.5', 'base_source_tree_sha256': ASSEMBLER.BASE.BASE_TREE,
        'files': rows, 'snapshot_attribution': attribution,
        'scope': 'reviewed_system_completeness_delta_not_entire_dirty_tree'}
    path = destination / 'runtime-delta-manifest.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=True, indent=2, sort_keys=True)
    print(json.dumps({'destination': str(destination), 'files': len(rows),
        'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'production_changed': False}))


if __name__ == '__main__':
    main()
