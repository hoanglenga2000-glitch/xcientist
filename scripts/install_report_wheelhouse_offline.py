"""Install an exact public-wheel lock into an isolated, non-production target."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',required=True);parser.add_argument('--sha256',required=True);args=parser.parse_args()
    archive=Path(args.archive)
    if sha(archive)!=args.sha256:raise SystemExit('wheel_archive_hash_mismatch')
    root=Path('C:/EMQA/sys0907-report-deps-v1');lock=json.loads((root/'dependency-lock.json').read_text(encoding='utf-8'))
    expected={row['url'].rsplit('/',1)[-1]:row['sha256'] for row in lock['files']}
    wheels=root/'wheels';wheels.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members=[item for item in bundle.infolist() if not item.is_dir()]
        if {item.filename for item in members}!=set(expected):raise SystemExit('wheel_archive_members_mismatch')
        for item in members:
            if Path(item.filename).name!=item.filename:raise SystemExit('wheel_path_rejected')
            data=bundle.read(item)
            if hashlib.sha256(data).hexdigest()!=expected[item.filename]:raise SystemExit('wheel_hash_mismatch')
            path=wheels/item.filename
            if path.exists():
                if sha(path)!=expected[item.filename]:raise SystemExit('existing_wheel_conflict')
            else:
                with path.open('xb') as handle:handle.write(data)
    target=root/'site-packages'
    if target.exists() and any(target.iterdir()):raise SystemExit('existing_target_requires_reconciliation')
    env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONUTF8']='1'
    result=subprocess.run([sys.executable,'-m','pip','--isolated','--disable-pip-version-check','install','--no-index','--find-links',str(wheels),'--require-hashes','--no-compile','--target',str(target),'-r',str(root/'requirements.lock')],env=env,capture_output=True,timeout=300)
    if result.returncode:raise SystemExit('offline_wheel_install_failed')
    files=[{'name':path.relative_to(target).as_posix(),'sha256':sha(path)} for path in sorted(target.rglob('*')) if path.is_file()]
    (root/'installed-files.json').write_text(json.dumps({'files':files},sort_keys=True),encoding='utf-8')
    receipt={'status':'ready','offline_install':True,'wheel_count':len(expected),'file_count':len(files),'lock_sha256':sha(root/'dependency-lock.json'),'installed_manifest_sha256':sha(root/'installed-files.json'),'target':str(target),'production_changed':False,'gpu_actions':0}
    (root/'offline-install-receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
