"""Stage the approved production baseline plus this patch, then serve isolated data.

This is local integration verification, NOT a production release transaction.
No inherited provider credentials, historical databases, recovery or GPU worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = Path('D:/EV12/system-repair-20261008-gates')
STAGE = Path('D:/EV12/task-ui-local-20261008')
WEB = ROOT / 'web/research-agent-workstation'
NODE = Path('D:/下载/node.exe')
EVIDENCE = ROOT / 'artifacts/advanced-tools-migration-20261008'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_sources():
    baseline = json.loads((EVIDENCE / 'baseline.json').read_text(encoding='utf-8'))
    if sha(PUBLISHED / 'source-receipt.json') != baseline['published_source_receipt_sha256']:
        raise RuntimeError('published_source_changed')
    before = {row['path']: row for row in baseline['files']}
    changed = []
    for source_prefix, published_prefix, destination in [
        ('web/research-agent-workstation/src', 'web/src', STAGE / 'web/src'),
        ('src/evomind_runtime', 'runtime/evomind_runtime', STAGE / 'runtime/evomind_runtime'),
    ]:
        shutil.copytree(PUBLISHED / published_prefix, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
        for source in sorted((ROOT / source_prefix).rglob('*')):
            if source.suffix not in {'.py', '.tsx', '.ts', '.mjs', '.css'} or '__pycache__' in source.parts:
                continue
            relative = source.relative_to(ROOT).as_posix()
            previous = before.get(relative)
            current = sha(source)
            if previous and previous['before'] == current:
                continue
            if previous and previous['before'] != previous['published']:
                raise RuntimeError('preexisting_change_overlap:' + relative)
            if not previous and not (
                '/task-workspace/' in relative or '/api/assistant/tasks/' in relative
                or '/app/workspace/' in relative or '/api/assistant/model-profiles/' in relative
                or source.name in {'user_tasks.py', 'model_profiles.py', 'model_profile_secrets.py', 'personal_model_client.py', 'personal_model_http.py'}
            ):
                continue
            target = destination / source.relative_to(ROOT / source_prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            changed.append({'path': relative, 'before': previous['before'] if previous else None, 'sha256': current})
    for directory in ('public', 'support', 'scripts', 'test-fixtures'):
        source = PUBLISHED / 'web' / directory
        if source.is_dir():
            shutil.copytree(source, STAGE / 'web' / directory, dirs_exist_ok=True)
    for name in ('package.json', 'package-lock.json', 'next.config.mjs', 'tsconfig.json', 'postcss.config.mjs', 'tailwind.config.ts', 'next-env.d.ts'):
        source = PUBLISHED / 'web' / name
        if not source.is_file():
            source = WEB / name
        if source.is_file():
            shutil.copy2(source, STAGE / 'web' / name)
    modules = STAGE / 'web/node_modules'
    if not modules.exists():
        subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
            f"New-Item -ItemType Junction -Path '{modules}' -Target '{WEB / 'node_modules'}' | Out-Null"],
            check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if modules.resolve() != (WEB / 'node_modules').resolve():
        raise RuntimeError('preview_dependency_path_mismatch')
    receipt = {'schema': 'evomind.task_ui_local_patch.v1', 'published_build': baseline['production_build'],
        'source_receipt_sha256': baseline['published_source_receipt_sha256'], 'changes': changed,
        'production_deployed': False, 'provider_execution': 'not_configured', 'hpc_accessed': False}
    (EVIDENCE / 'source-scope.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return receipt


def backend():
    sys.path[:0] = [str(STAGE / 'runtime'), str(ROOT / 'src')]
    # This seam lives outside both production runtime and web source packages.
    # Only the explicitly named fixture model works; all other network is blocked.
    if Path(os.environ.get('EVOMIND_TEST_FIXTURE_ROOT', '')).resolve() != (STAGE / 'data').resolve():
        raise RuntimeError('isolated_fixture_root_required')
    from task_workspace_provider_fixture import respond
    from evomind_runtime import personal_model_client
    personal_model_client.post_public_json = respond
    from evomind_runtime.http_server import ensure_token, make_handler
    from evomind_runtime.runtime import AgentRuntime
    runtime = AgentRuntime(STAGE / 'data')
    server = ThreadingHTTPServer(('127.0.0.1', 8877), make_handler(runtime, ensure_token(runtime.runtime_root)))
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        runtime.close()


def start():
    for port in (8100, 8877):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(('127.0.0.1', port))
    receipt = stage_sources()
    data = STAGE / 'data'
    data.mkdir(parents=True, exist_ok=True)
    logs = STAGE / 'logs'
    logs.mkdir(exist_ok=True)
    # This password is a deliberately public local-only fixture, not a secret.
    password, salt = 'local-ui-fixture-only', secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32).hex()
    keep = {'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATH', 'PATHEXT', 'TEMP', 'TMP', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'PROGRAMDATA', 'NUMBER_OF_PROCESSORS'}
    env = {key: value for key, value in os.environ.items() if key.upper() in keep}
    env.update({
        'WORKSTATION_ROOT': str(data), 'WORKSTATION_DATA_DIR': str(data),
        'DATABASE_URL': 'file:' + (data / 'web-fixture.db').as_posix(),
        'EVOMIND_TEST_FIXTURE_ROOT': str(data), 'EVOMIND_RUNTIME_PORT': '8877',
        'WORKSTATION_SESSION_SECRET': secrets.token_hex(32), 'WORKSTATION_LOCAL_HTTPS': '0',
        'WORKSTATION_ADMIN_USERNAME': 'preview-user',
        'WORKSTATION_ADMIN_PASSWORD_SCRYPT': f'scrypt$16384$8$1${salt.hex()}${digest}',
        'NEXT_TELEMETRY_DISABLED': '1', 'HOSTNAME': '127.0.0.1', 'PORT': '8100', 'PYTHONUTF8': '1',
    })
    # Production CSP intentionally forbids webpack eval. Test the production
    # build rather than loosening that policy merely to make dev mode work.
    with (logs / 'build.log').open('ab') as log:
        subprocess.run([str(NODE), str(WEB / 'node_modules/next/dist/bin/next'), 'build', '--webpack'],
            cwd=STAGE / 'web', env=env, stdout=log, stderr=log, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    standalone = STAGE / 'web/.next/standalone'
    if not (standalone / 'server.js').is_file():
        raise RuntimeError('standalone_server_missing')
    shutil.copytree(STAGE / 'web/.next/static', standalone / '.next/static', dirs_exist_ok=True)
    shutil.copytree(STAGE / 'web/public', standalone / 'public', dirs_exist_ok=True)
    children = []
    try:
        for name, command, cwd in [
            ('runtime', [sys.executable, str(Path(__file__).resolve()), '--backend'], STAGE),
            ('web', [str(NODE), str(standalone / 'server.js')], standalone),
        ]:
            with (logs / f'{name}.log').open('ab') as log:
                children.append(subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW))
        print(json.dumps({'url': 'http://127.0.0.1:8100/login?next=/workspace', 'fixture_username': 'preview-user',
            'fixture_password': password, 'pids': [child.pid for child in children], 'changes': len(receipt['changes'])}), flush=True)
        while all(child.poll() is None for child in children):
            threading.Event().wait(1)
    finally:
        for child in children:
            if child.poll() is None:
                subprocess.run(['taskkill.exe', '/PID', str(child.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', action='store_true')
    parser.add_argument('--sync-only', action='store_true')
    args = parser.parse_args()
    if args.backend:
        backend()
    elif args.sync_only:
        print(json.dumps(stage_sources()))
    else:
        start()
