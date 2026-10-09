"""Verify the packaged budget API with isolated data; no HPC or production writes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage-root', type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage_root.resolve()
    stage.relative_to(Path('C:/ProgramData/EvoMind/staging').resolve())
    build = json.loads((stage / 'build-result.json').read_text())
    spec = importlib.util.spec_from_file_location('budget_harness', stage / 'verify_invitation_server_candidate.py')
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    flow = stage / 'budget-flow'
    flow.mkdir(exist_ok=False)
    v.extract(stage / 'web.zip', build['web']['sha256'], flow / 'web', 'operational-overlay-manifest.json')
    v.extract(stage / 'runtime.zip', build['runtime']['sha256'], flow / 'runtime', 'runtime-hotfix-manifest.json')
    (flow / 'workspace').mkdir()
    shutil.copyfile(stage / 'fixture-schema.sqlite', flow / 'workspace/workstation.sqlite')
    root = flow / 'workspace/workspace/runtime'
    root.mkdir(parents=True)
    tenant = 'tenant_' + hashlib.sha256(b'evomind.tenant.v1:invitation_acceptance').hexdigest()[:24]
    policy = {'schema': 'evomind.training_control.v1', 'runtime_root': str(root), 'enabled': True, 'tenants': [tenant], 'owners': ['invitation_acceptance'], 'projects': ['fixture'], 'study_id': 'budget-fixture'}
    v.write_json(flow / 'budget-policy.json', policy)
    secret, password = secrets.token_urlsafe(48), secrets.token_urlsafe(32)
    identity = json.loads((flow / 'web/runtime-build-manifest.json').read_text())['source_tree_sha256']
    environment = v.fixture_environment(flow, secret, password, build['build_id'], identity)
    environment['EVOMIND_AIBUILD_POLICY_FILE'] = str(flow / 'budget-policy.json')
    sys.path.insert(0, str(flow / 'runtime'))
    from evomind_runtime.research_budget import GpuExecutionBudget
    ledger = GpuExecutionBudget(str(root / 'gpu_budget.sqlite3'), 'budget-fixture')
    for index in range(24):
        operation = ledger.reserve(str(index), 1200)
        ledger.settle(operation['id'], 1200, success=True)
    def ledger_rows():
        with sqlite3.connect(ledger.database) as connection:
            return connection.execute('SELECT * FROM gpu_operations ORDER BY id').fetchall()
    original_rows = ledger_rows()
    services = v.FixtureServices(flow, environment, 'C:/Program Files/nodejs/node.exe')
    before, checks = v.production_snapshot(), []
    result = {'schema': 'evomind.budget_acceptance.v1', 'build_id': build['build_id'], 'web_sha256': build['web']['sha256'], 'runtime_sha256': build['runtime']['sha256'], 'fixture_only': True, 'hpc_accessed': False}
    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})
        if not passed:
            raise RuntimeError(name)
    try:
        services.start()
        endpoint = '/api/assistant/execution-budget'
        port = services.web_port
        origin = f'http://127.0.0.1:{port}'
        check('unauthenticated_read_denied', v.request(port, 'GET', endpoint)[0] == 401)
        code, headers, _ = v.request(port, 'POST', '/api/auth/login', body={'username': 'invitation_acceptance', 'password': password}, headers={'Origin': origin})
        check('fixture_login', code == 200)
        cookie = headers['set-cookie'].split(';', 1)[0]
        auth = {'Cookie': cookie, 'Origin': origin}
        _, _, session = v.request(port, 'GET', '/api/session/status', headers=auth)
        auth[v.CSRF] = session['csrf_token']
        code, _, initial = v.request(port, 'GET', endpoint, headers=auth)
        check('owner_reads_exhausted_budget', code == 200 and initial['budget']['engineering_remaining_seconds'] == 0)
        body = {'gpu_limit_seconds': 50000, 'engineering_limit_seconds': 40000, 'expected_revision': 0, 'request_id': 'fixture-budget-once', 'reason': 'Isolated acceptance only', 'confirmed': True}
        check('csrf_required', v.request(port, 'POST', endpoint, body=body, headers={'Cookie': cookie, 'Origin': origin})[0] == 403)
        check('origin_required', v.request(port, 'POST', endpoint, body=body, headers={**auth, 'Origin': 'https://invalid.example'})[0] == 403)
        foreign = {'Cookie': v.COOKIE + '=' + v.signed_principal(secret, 'foreign_fixture', 'tenant_' + 'b' * 24), 'Origin': origin}
        _, _, foreign_session = v.request(port, 'GET', '/api/session/status', headers=foreign)
        foreign[v.CSRF] = foreign_session['csrf_token']
        check('foreign_read_denied', v.request(port, 'GET', endpoint, headers=foreign)[0] == 403)
        check('foreign_write_denied', v.request(port, 'POST', endpoint, body=body, headers=foreign)[0] == 403)
        check('owner_spoof_denied', v.request(port, 'POST', endpoint, body=body, headers={**foreign, 'x-evomind-tenant-id': tenant, 'x-evomind-principal-id': 'invitation_acceptance'})[0] == 403)
        check('explicit_confirmation_required', v.request(port, 'POST', endpoint, body={**body, 'confirmed': False}, headers=auth)[0] == 400)
        check('invalid_allocation_denied', v.request(port, 'POST', endpoint, body={**body, 'engineering_limit_seconds': 50001}, headers=auth)[0] == 400)
        check('below_consumed_denied', v.request(port, 'POST', endpoint, body={**body, 'engineering_limit_seconds': 100}, headers=auth)[0] == 409)
        code, _, saved = v.request(port, 'POST', endpoint, body=body, headers=auth)
        check('amendment_saved', code == 200 and saved['budget']['revision'] == 1 and saved['budget']['engineering_limit_seconds'] == 40000)
        check('audit_recorded', len(saved['history']) == 1 and saved['history'][0]['reason'] == body['reason'])
        code, _, replay = v.request(port, 'POST', endpoint, body=body, headers=auth)
        check('retry_idempotent', code == 200 and replay['replayed'] and len(replay['history']) == 1)
        check('stale_edit_denied', v.request(port, 'POST', endpoint, body={**body, 'request_id': 'fixture-stale-edit'}, headers=auth)[0] == 409)
        check('historical_ledger_unchanged', ledger_rows() == original_rows)
        services.stop(); services.start()
        code, _, restored = v.request(services.web_port, 'GET', endpoint, headers=auth)
        check('restart_persistence', code == 200 and restored['budget']['revision'] == 1 and restored['history'] == saved['history'])
        services.stop()
        operation = ledger.reserve('post-amendment', 60)
        check('real_reservation_uses_saved_limits', operation['engineering_limit_seconds'] == 40000 and operation['revision'] == 1)
        ledger.settle(operation['id'], 0, success=True)
    except Exception as error:
        result['error_class'] = type(error).__name__
        result['failed_check'] = str(error) if isinstance(error, RuntimeError) else 'fixture_exception'
    finally:
        try:
            services.stop()
        except Exception:
            result['cleanup_failed'] = True
        checks.append({'name': 'production_unchanged', 'passed': v.production_snapshot() == before})
        result['checks'] = checks
        result['status'] = 'passed' if len(checks) == 19 and all(c['passed'] for c in checks) and 'error_class' not in result and not result.get('cleanup_failed') else 'failed'
        v.write_json(stage / 'budget-acceptance.json', result)
        print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
