"""LOCAL ONLY: published source + reviewed patches, isolated accounts and data.

Never used by production release transactions. No historical ownership migration,
real model transport, server deployment, or GPU access is included.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
import secrets

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'artifacts/product-launch-20261008'
PUBLISHED = Path('D:/EV12/system-repair-20261008-gates')
STAGE = Path('D:/EV12/product-launch-local-20261008')
WEB = ROOT / 'web/research-agent-workstation'
NODE = Path('D:/下载/node.exe')
PREFIXES = {'web/research-agent-workstation/src': 'web/src', 'src/evomind_runtime': 'runtime/evomind_runtime'}
SUFFIXES = {'.py', '.ts', '.tsx', '.mjs', '.mts', '.css'}
# Reviewed pre-existing personal preferences, needed instead of global settings.
CARRY = {'web/research-agent-workstation/src/app/api/settings/route.ts',
         'web/research-agent-workstation/src/lib/server/user-preferences.ts'}
TEST_CARRY = {'web/research-agent-workstation/src/components/workstation/screens/assistant-presentation.test.mjs',
              'web/research-agent-workstation/src/lib/server/report-assistant-bridge-contract.test.ts',
              'web/research-agent-workstation/src/lib/server/evolution-integrity.test.ts'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def stage_sources():
    baseline = json.loads((EVIDENCE / 'baseline.json').read_text(encoding='utf-8'))
    previous = json.loads((ROOT / 'artifacts/advanced-tools-migration-20261008/source-scope.json').read_text(encoding='utf-8'))
    if sha(PUBLISHED / 'source-receipt.json') != baseline['published_receipt_sha256']:
        raise RuntimeError('published_receipt_changed')
    before = {row['path']: row for row in baseline['files']}
    prior = {row['path']: row for row in previous['changes']}
    changes = []
    # First resolve the entire file whitelist. Do not partially copy on conflict.
    for prefix, target_prefix in PREFIXES.items():
        for path in sorted((ROOT / prefix).rglob('*')):
            if not path.is_file() or path.suffix not in SUFFIXES or '__pycache__' in path.parts:
                continue
            name = path.relative_to(ROOT).as_posix()
            old = before.get(name)
            if old is None and sha(path) == sha(PUBLISHED / target_prefix / path.relative_to(ROOT / prefix)):
                continue  # Existing declaration files were not in the first audit suffix set.
            edited = old is None or sha(path) != old['before']
            if not edited and name not in prior and name not in CARRY | TEST_CARRY:
                continue
            if old and old['before'] != old['published'] and name not in CARRY | TEST_CARRY:
                if name not in prior or prior[name]['sha256'] != old['before']:
                    raise RuntimeError('unreviewed_preexisting_overlap:' + name)
            if old is None and not any(value in name for value in (
                '/task-workspace/', '/api/assistant/files/', '/account-registry.', '/product-account-api.test.',
                '/user_files.py', '/personal_tool_boundary.py')):
                raise RuntimeError('unreviewed_new_source:' + name)
            changes.append({'path': name, 'sha256': sha(path), 'before': old['before'] if old else None,
                'published': old['published'] if old else None,
                'reason': 'reviewed_personal_preferences' if name in CARRY else 'reviewed_test_contract_updated_for_new_navigation' if name in TEST_CARRY else 'prior_approved_patch_and_current_delta' if name in prior else 'current_product_patch',
                'test_only': '.test.' in name,
                'target': target_prefix + '/' + path.relative_to(ROOT / prefix).as_posix()})
    STAGE.mkdir(parents=True, exist_ok=True)
    for published in PREFIXES.values():
        shutil.copytree(PUBLISHED / published, STAGE / published, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    for row in changes:
        target = STAGE / row['target']; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / row['path'], target)
    for name in ('public', 'support', 'scripts', 'test-fixtures'):
        if (PUBLISHED / 'web' / name).is_dir():
            shutil.copytree(PUBLISHED / 'web' / name, STAGE / 'web' / name, dirs_exist_ok=True)
    for name in ('package.json', 'package-lock.json', 'next.config.mjs', 'tsconfig.json', 'postcss.config.mjs', 'tailwind.config.ts', 'next-env.d.ts'):
        source = PUBLISHED / 'web' / name
        if not source.is_file():
            source = WEB / name
        if source.is_file():
            shutil.copy2(source, STAGE / 'web' / name)
    modules = STAGE / 'web/node_modules'
    if not modules.exists():
        subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
            f"New-Item -ItemType Junction -Path '{modules}' -Target '{WEB / 'node_modules'}' | Out-Null"], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if modules.resolve() != (WEB / 'node_modules').resolve():
        raise RuntimeError('dependency_junction_mismatch')
    fingerprint = hashlib.sha256(json.dumps([(r['path'], r['sha256']) for r in changes if not r['test_only']], sort_keys=True).encode()).hexdigest()
    receipt = {'schema': 'evomind.product_launch.local.v1', 'candidate': 'local-product-' + fingerprint[:12],
        'source_digest': fingerprint, 'published_receipt_sha256': baseline['published_receipt_sha256'],
        'changes': changes, 'excluded_unrelated_changes': sum(row['before'] != row['published'] and row['path'] not in {item['path'] for item in changes} for row in before.values()),
        'production_deployed': False, 'stage': str(STAGE)}
    (EVIDENCE / 'source-scope.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return receipt


def backend():
    sys.path[:0] = [str(STAGE / 'runtime'), str(ROOT / 'src')]
    if Path(os.environ.get('EVOMIND_TEST_FIXTURE_ROOT', '')).resolve() != (STAGE / 'data').resolve():
        raise RuntimeError('fixture_root_required')
    from product_launch_provider_fixture import respond
    from evomind_runtime import personal_model_client
    personal_model_client.post_public_json = respond
    real_connect = socket.socket.connect
    def local_only(connection, address):
        if not isinstance(address, tuple) or address[0] not in {'127.0.0.1', '::1'}:
            raise OSError('isolated_fixture_external_network_disabled')
        return real_connect(connection, address)
    socket.socket.connect = local_only
    from evomind_runtime.http_server import ensure_token, make_handler
    from evomind_runtime.runtime import AgentRuntime
    runtime = AgentRuntime(STAGE / 'data')
    server = ThreadingHTTPServer(('127.0.0.1', 8879), make_handler(runtime, ensure_token(runtime.runtime_root)))
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close(); runtime.close()


def environment():
    data = STAGE / 'data'; data.mkdir(parents=True, exist_ok=True)
    account_db = data / 'private-accounts/accounts.sqlite3'
    if not account_db.exists():
        for user in ('alice', 'bob'):
            subprocess.run([sys.executable, str(ROOT / 'scripts/manage_product_accounts.py'), 'create', '--database', str(account_db),
                '--username', user, '--confirm-account-change', '--password-stdin'], input=f'{user}-product-fixture-only\n',
                text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    with sqlite3.connect(data / 'web-fixture.db') as db:
        db.execute('CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at DATETIME NOT NULL)')
    # This local-only signing key persists across this fixture's service restart.
    secret_file = account_db.parent / 'preview-session.key'
    if not secret_file.exists():
        secret_file.write_text(secrets.token_hex(32), encoding='ascii')
    keep = {'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATH', 'PATHEXT', 'TEMP', 'TMP', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'PROGRAMDATA', 'NUMBER_OF_PROCESSORS'}
    env = {key: value for key, value in os.environ.items() if key.upper() in keep}
    env.update({'WORKSTATION_ROOT': str(data), 'WORKSTATION_DATA_DIR': str(data),
        'DATABASE_URL': 'file:' + (data / 'web-fixture.db').as_posix(), 'WORKSTATION_ACCOUNTS_DB': str(account_db),
        'WORKSTATION_SESSION_SECRET': secret_file.read_text(encoding='ascii'), 'WORKSTATION_LOCAL_HTTPS': '0',
        'EVOMIND_TEST_FIXTURE_ROOT': str(data), 'EVOMIND_RUNTIME_PORT': '8879', 'NEXT_TELEMETRY_DISABLED': '1',
        'HOSTNAME': '127.0.0.1', 'PORT': '8102', 'PYTHONUTF8': '1'})
    return env


def start(build):
    for port in (8102, 8879):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(('127.0.0.1', port))
    receipt = stage_sources() if build else json.loads((EVIDENCE / 'source-scope.json').read_text(encoding='utf-8'))
    env = environment(); logs = STAGE / 'logs'; logs.mkdir(exist_ok=True)
    if build:
        with (logs / 'build.log').open('wb') as log:
            subprocess.run([str(NODE), str(WEB / 'node_modules/next/dist/bin/next'), 'build', '--webpack'], cwd=STAGE / 'web', env=env,
                stdout=log, stderr=log, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    standalone = STAGE / 'web/.next/standalone'
    shutil.copytree(STAGE / 'web/.next/static', standalone / '.next/static', dirs_exist_ok=True)
    shutil.copytree(STAGE / 'web/public', standalone / 'public', dirs_exist_ok=True)
    children = []
    for name, command, cwd in [('runtime', [sys.executable, str(Path(__file__).resolve()), '--backend'], STAGE),
        ('web', [str(NODE), str(standalone / 'server.js')], standalone)]:
        with (logs / f'{name}.log').open('ab') as log:
            children.append(subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW))
    record = {'url': 'http://127.0.0.1:8102/', 'candidate': receipt['candidate'], 'pids': [child.pid for child in children],
        'next_build_id': (STAGE / 'web/.next/BUILD_ID').read_text().strip(), 'fixture_users': ['alice', 'bob'], 'production_deployed': False}
    (EVIDENCE / 'preview.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)
    # Persist preview after this launcher exits; explicit local restart can stop
    # only the verified PIDs recorded above. Never discover/kill by port alone.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', action='store_true')
    parser.add_argument('--stage-only', action='store_true')
    parser.add_argument('--skip-build', action='store_true')
    args = parser.parse_args()
    if args.backend: backend()
    elif args.stage_only: print(json.dumps(stage_sources()))
    else: start(not args.skip_build)
