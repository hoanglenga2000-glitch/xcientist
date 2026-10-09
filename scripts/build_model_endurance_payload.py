"""Build an immutable real-model test ZIP from an already frozen runtime delta."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.runtime_root.resolve(strict=True)
    manifest = root / 'runtime-delta-manifest.json'
    data = manifest.read_bytes()
    if digest(data) != args.manifest_sha256:
        raise ValueError('frozen_runtime_manifest_drift')
    source = json.loads(data)
    if source.get('schema') != 'evomind.syscomplete_runtime_delta.v1' or source.get('frozen') is not True:
        raise ValueError('frozen_runtime_required')
    payloads = {}
    for row in source['files']:
        name = PurePosixPath(row['path'])
        if (name.is_absolute() or '..' in name.parts or '\\' in row['path'] or ':' in row['path']
                or len(name.parts) != 2 or name.parts[0] != 'evomind_runtime' or name.suffix != '.py'
                or name.as_posix() in payloads):
            raise ValueError('frozen_runtime_path_rejected')
        path = root.joinpath(*name.parts)
        if path.is_symlink() or path.resolve().parent != root / 'evomind_runtime':
            raise ValueError('frozen_runtime_path_alias')
        data = path.read_bytes()
        if len(data) != row['bytes'] or digest(data) != row['sha256']:
            raise ValueError('frozen_runtime_file_drift')
        payloads[name.as_posix()] = data
    harness = Path(__file__).with_name('run_model_endurance_acceptance.py')
    payloads[harness.name] = harness.read_bytes()
    if args.output.exists():
        raise ValueError('endurance_payload_already_exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payloads.items()):
            archive.writestr(name, data)
    print(json.dumps({'schema': 'evomind.endurance_payload.v1', 'output': str(args.output),
        'sha256': digest(args.output.read_bytes()), 'file_count': len(payloads),
        'runtime_manifest_sha256': args.manifest_sha256, 'harness_sha256': digest(payloads[harness.name]),
        'production_changed': False}))


if __name__ == '__main__':
    main()
