"""Derive an isolated test candidate from the active package plus an exact ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--candidate-name', choices=[f'endurance-candidate-v{version}' for version in range(1, 12)], default='endurance-candidate-v1')
    args = parser.parse_args()
    payload = Path(args.payload).resolve()
    if sha(payload) != args.sha256: raise SystemExit('payload_integrity_failed')
    root = Path('C:/ProgramData/EvoMind/staging/system-completeness-20260907') / args.candidate_name
    if root.exists(): raise SystemExit('candidate_already_exists')
    package = Path('C:/ProgramData/EvoMind/bundle/runtime/evomind_runtime')
    research = Path('C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src/research_os')
    expected = {
        'model_transport.py': 'a22ceb1d2132e797d771dfa3ddb90f4844f32d5df91dcf3958c667fc2c70e186',
        'runtime.py': '260ddf65656f67016bd7b62891426d7f850d14db8bc1eb05c2fe7c1c29d1a1be',
        'store.py': '576633c747e0648416c481570a0afc8d8e97616f8853cd33ffb1b1ad7d04514d',
        'assistant_runs.py': '11727a59e8e339f5c3cf24d0b59967131ba5e85e5e8af444822d7eb1b560bcf6',
        'aibuild_engine.py': '0bb6cfc506929fd854ea6b2c6c62d3a81c06e2962bd9b4f29a8e4c637e5d8b28',
        'tools.py': 'be45a86ed9d56b46883806fdb6f31e237dbef266c04388be43d949df42cdb1e2',
        'policy.py': '958304cac8e15f0e5b062853937064de4beb7fc78e49031ed4d308dc4a37fff3',
        'http_server.py': '56017af35ec9968636449fb8239ba4cea7cd153008f0f50af00a15d6ee973164',
    }
    for name, digest in expected.items():
        if hashlib.sha256((package/name).read_text(encoding='utf-8').encode()).hexdigest() != digest:
            raise SystemExit('active_source_drift_'+name)
    for source in (package, research):
        if any(path.is_symlink() for path in source.rglob('*')):
            raise SystemExit('source_package_symlink')
    root.mkdir(parents=True)
    shutil.copytree(package, root/'evomind_runtime', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(research, root/'research_os', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    with zipfile.ZipFile(payload) as archive:
        for item in archive.infolist():
            if item.is_dir(): continue
            relative = Path(item.filename)
            if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.py':
                raise SystemExit('payload_member_rejected')
            destination = (root/relative).resolve()
            destination.relative_to(root.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(item))
    files = [{'path':path.relative_to(root).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size}
             for path in sorted(root.rglob('*')) if path.is_file()]
    manifest = {'schema':'evomind.isolated_system_candidate.v1','scope':'acceptance_only_not_deployed',
        'base_source_tree_sha256':'3800faca9fc507c17152904af09629ccc98a74769723e4f7d1ae02ed6f1086be',
        'base_sources':expected,'payload_sha256':args.sha256,'files':files}
    path = root/'candidate-manifest.json'
    path.write_text(json.dumps(manifest,ensure_ascii=True,sort_keys=True,indent=2),encoding='utf-8')
    print(json.dumps({'candidate_root':str(root),'files':len(files),'manifest_sha256':sha(path),'production_changed':False}))


if __name__ == '__main__': main()
