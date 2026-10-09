from __future__ import annotations

import concurrent.futures
import http.client
import json
import sqlite3
import threading
from http.server import ThreadingHTTPServer

import pytest

from evomind_runtime import budget_web
from evomind_runtime.http_server import make_handler
from evomind_runtime.research_budget import GpuExecutionBudget, ResearchBudget
from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.tenant_access import AccessError, Principal


@pytest.fixture
def configured(tmp_path, monkeypatch):
    root = tmp_path / 'runtime'
    root.mkdir()
    principal = Principal('tenant_' + 'a' * 24, 'alice')
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'schema': 'evomind.training_control.v1', 'runtime_root': str(root), 'enabled': True, 'tenants': [principal.tenant_id], 'owners': ['alice'], 'projects': ['project_test'], 'study_id': 'original-study'}))
    monkeypatch.setenv('EVOMIND_AIBUILD_POLICY_FILE', str(policy))
    return root, principal


def payload(**overrides):
    return {'gpu_limit_seconds': 50000, 'engineering_limit_seconds': 40000, 'expected_revision': 0, 'request_id': 'budget-request-001', 'reason': 'Bounded engineering acceptance', 'confirmed': True, **overrides}


def test_status_is_read_only_and_nonowners_cannot_read_or_amend(configured):
    root, owner = configured
    assert budget_web.status(root, owner)['budget']['revision'] == 0
    assert not (root / 'gpu_budget.sqlite3').exists()
    for other in [None, Principal(owner.tenant_id, 'bob'), Principal('tenant_' + 'b' * 24, owner.owner_id)]:
        for fn in [lambda: budget_web.status(root, other), lambda: budget_web.amend(root, other, payload())]:
            with pytest.raises(AccessError) as error:
                fn()
            assert error.value.status == 403
    assert not (root / 'gpu_budget.sqlite3').exists()


def test_amendment_changes_real_admission_without_changing_old_ledger(configured):
    root, owner = configured
    ledger = GpuExecutionBudget(str(root / 'gpu_budget.sqlite3'), 'original-study')
    for index in range(24):
        op = ledger.reserve(str(index), 1200)
        ledger.settle(op['id'], 1200, success=True)
    with pytest.raises(RuntimeError, match='exhausted'):
        ledger.reserve('blocked', 60)
    with sqlite3.connect(ledger.database) as connection:
        before = connection.execute('SELECT * FROM gpu_operations').fetchall()
    result = budget_web.amend(root, owner, payload())
    assert result['budget']['revision'] == 1 and result['no_execution_started']
    assert result['budget']['operations'] == 24
    with sqlite3.connect(ledger.database) as connection:
        assert connection.execute('SELECT * FROM gpu_operations').fetchall() == before
    assert ledger.reserve('now-admitted', 60)['engineering_limit_seconds'] == 40000
    assert ResearchBudget(str(root / 'research.sqlite'), 'official').summary()['gpu_limit_seconds'] == 43200
    assert budget_web.status(root, owner)['history'][0]['owner_id'] == 'alice'


@pytest.mark.parametrize('change', [
    {'confirmed': False}, {'gpu_limit_seconds': True}, {'engineering_limit_seconds': -1},
    {'engineering_limit_seconds': 50001}, {'gpu_limit_seconds': float('nan')},
    {'gpu_limit_seconds': 31536001}, {'reason': ''}, {'study': 'bypass'}, {'expected_revision': True},
])
def test_invalid_mutations_fail_before_creating_ledger(configured, change):
    root, owner = configured
    with pytest.raises(AccessError) as error:
        budget_web.amend(root, owner, payload(**change))
    assert error.value.status == 400
    assert not (root / 'gpu_budget.sqlite3').exists()


def test_idempotency_conflicts_and_concurrent_edits(configured):
    root, owner = configured
    assert not budget_web.amend(root, owner, payload())['replayed']
    assert budget_web.amend(root, owner, payload())['replayed']
    for body, code in [(payload(gpu_limit_seconds=60000), 'budget_request_conflict'),
                       (payload(request_id='new-request-002'), 'budget_revision_conflict')]:
        with pytest.raises(AccessError, match=code):
            budget_web.amend(root, owner, body)
    def save(index):
        try:
            return budget_web.amend(root, owner, payload(expected_revision=1, gpu_limit_seconds=51000 + index, request_id=f'concurrent-request-{index}'))['ok']
        except AccessError as error:
            return error.code
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, [1, 2]))
    assert results.count(True) == 1 and results.count('budget_revision_conflict') == 1
    assert len(budget_web.status(root, owner)['history']) == 2


def test_cannot_lower_below_reserved_or_clear_uncertainty(configured):
    root, owner = configured
    ledger = GpuExecutionBudget(str(root / 'gpu_budget.sqlite3'), 'original-study')
    op = ledger.reserve('uncertain', 1200)
    with pytest.raises(AccessError, match='below_committed'):
        budget_web.amend(root, owner, payload(gpu_limit_seconds=1100, engineering_limit_seconds=1100))
    ledger.settle(op['id'], 1, uncertain=True)
    budget_web.amend(root, owner, payload())
    assert ledger.summary()['reconciliation_required']
    with pytest.raises(RuntimeError, match='reconciliation'):
        ledger.reserve('blocked', 60)


def test_authenticated_http_budget_round_trip(configured, tmp_path):
    root, owner = configured
    runtime = AgentRuntime(workspace_root=tmp_path / 'workspace', runtime_root=root)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(runtime, 'fixture-token'))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def request(method, body=None, principal=owner):
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        headers = {'Authorization': 'Bearer fixture-token', 'Content-Type': 'application/json'}
        if principal:
            headers.update({'X-EvoMind-Access-Scope': 'user.v1', 'X-EvoMind-Tenant-Id': principal.tenant_id, 'X-EvoMind-Principal-Id': principal.owner_id})
        connection.request(method, '/v1/execution-budget', body=json.dumps(body).encode() if body else None, headers=headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result
    try:
        assert request('GET', principal=None)[0] == 403
        assert request('POST', payload(), Principal(owner.tenant_id, 'other'))[0] == 403
        assert request('GET')[1]['budget']['revision'] == 0
        assert request('POST', payload())[0] == 200
        assert request('GET')[1]['budget']['gpu_limit_seconds'] == 50000
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)
