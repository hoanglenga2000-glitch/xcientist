"""Prepare a narrow, hash-bound web bridge from the live application overlay."""
import difflib
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import paramiko

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/siim-mlebench-calibration-20260908/web-entry-repair'
REMOTE = 'C:/ProgramData/EvoMind'
DEST = REMOTE + '/staging/siim-mlebench-calibration-20260908/web-entry-repair'
NEW_MODULES = {'siim_calibration_web.py': 'bundle/runtime/evomind_runtime/siim_calibration_web.py'}
SNAPSHOT_EXTRA = {}
EXTRA_INSTALLERS = {}
GENERATE_BRIDGE_CONFIG = True
TARGETS = {
    'tools.py': 'bundle/runtime/evomind_runtime/tools.py',
    'siim_calibration_control.py': 'bundle/runtime/evomind_runtime/siim_calibration_control.py',
    'siim_calibration_suite.py': 'staging/siim-mlebench-calibration-20260908/runtime-extension/siim_calibration_suite.py',
    'siim_aide_controller.py': 'staging/siim-mlebench-calibration-20260908/runtime-extension/siim_aide_controller.py',
}


def connect():
    settings = {}
    for line in subprocess.check_output(['ssh', '-G', 'evomind-shanghai'], text=True, stderr=subprocess.DEVNULL).splitlines():
        key, _, value = line.partition(' ')
        if key in {'hostname', 'port', 'user', 'identityfile', 'userknownhostsfile', 'proxycommand', 'proxyjump'}:
            settings.setdefault(key, value)
    if (settings['hostname'] != '101.43.51.15' or settings['user'] != 'Administrator' or int(settings['port']) != 22
            or any(settings.get(k) not in {None, 'none'} for k in ['proxycommand', 'proxyjump'])):
        raise ValueError('application_host_route_changed')
    client = paramiko.SSHClient()
    client.load_host_keys(settings['userknownhostsfile'])
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(settings['hostname'], port=22, username=settings['user'],
                   pkey=paramiko.Ed25519Key.from_private_key_file(settings['identityfile']),
                   look_for_keys=False, allow_agent=False, timeout=20)
    return client


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    base, payload = OUT / 'base', OUT / 'payload'
    base.mkdir(parents=True, exist_ok=True)
    payload.mkdir(exist_ok=True)
    client = connect()
    try:
        sftp = client.open_sftp()
        sources = {**TARGETS, **SNAPSHOT_EXTRA, 'seal.json': 'state/bundle-integrity.json',
                   'policy.json': 'config/official-calibration/siim-mlebench-20260908.json'}
        for name, remote in sources.items():
            with sftp.open(REMOTE + '/' + remote, 'rb') as stream:
                raw = stream.read()
            path = base / name
            if path.exists() and path.read_bytes() != raw:
                raise ValueError('live_baseline_changed:' + name)
            if not path.exists():
                path.write_bytes(raw)
    finally:
        client.close()
    targets = dict(TARGETS)
    targets.update(NEW_MODULES)
    diffs = []
    for name in targets:
        local = ROOT / ('scripts' if name in {'siim_calibration_suite.py', 'siim_aide_controller.py'} else 'src/evomind_runtime') / name
        current = local.read_text(encoding='utf-8')
        if name in TARGETS:
            previous = (base / name).read_text(encoding='utf-8')
            diff = list(difflib.unified_diff(previous.splitlines(), current.splitlines(), fromfile='live/'+name, tofile='candidate/'+name, lineterm=''))
            changes = [line for line in diff if line.startswith(('+', '-')) and not line.startswith(('+++', '---'))]
            if len(changes) > 65:
                print('\n'.join(diff))
                raise ValueError('unrelated_local_changes_require_review:' + name)
            diffs.extend(diff)
        (payload / name).write_text(current, encoding='utf-8')
    (OUT / 'scoped.patch').write_text('\n'.join(diffs), encoding='utf-8')
    config = {'policy_sha256': sha((base/'policy.json').read_bytes())}
    if GENERATE_BRIDGE_CONFIG:
        config.update(schema='evomind.siim_web_bridge.v1',
                      coordinator_hashes={name: sha((payload/name).read_bytes()) for name in ['siim_calibration_suite.py','siim_aide_controller.py']})
        (payload/'siim-web-bridge.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        targets['siim-web-bridge.json'] = 'config/official-calibration/siim-web-bridge.json'
    manifest = {'schema': 'evomind.siim_web_patch.v1',
                'baseline_seal_sha256': sha((base/'seal.json').read_bytes()),
                'policy_sha256_unchanged': config['policy_sha256'],
                'files': {name: {'target': target, 'sha256': sha((payload/name).read_bytes()),
                           'before_sha256': sha((base/name).read_bytes()) if (base/name).exists() else None}
                          for name, target in targets.items()}}
    (payload/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if args.upload:
        client = connect()
        try:
            sftp = client.open_sftp()
            destination = DEST
            try:
                sftp.stat(destination)
            except FileNotFoundError:
                sftp.mkdir(destination)
            upload = {p.name: p for p in payload.iterdir() if p.is_file()}
            upload['apply_siim_web_bridge.py'] = ROOT/'scripts/apply_siim_web_bridge.py'
            upload.update({name: ROOT/'scripts'/name for name in EXTRA_INSTALLERS})
            for name, source in upload.items():
                target = destination+'/'+name
                sftp.put(str(source), target)
                with sftp.open(target, 'rb') as stream:
                    if sha(stream.read()) != sha(source.read_bytes()):
                        raise ValueError('upload_hash_mismatch')
        finally:
            client.close()
    print(json.dumps({'status': 'payload_prepared', 'manifest_sha256': sha((payload/'manifest.json').read_bytes()),
                      'files': len(targets), 'uploaded': args.upload, 'policy_changed': False, 'deployment_or_training_started': False}))


if __name__ == '__main__':
    main()
