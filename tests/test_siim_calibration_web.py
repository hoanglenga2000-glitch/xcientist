from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from evomind_runtime import siim_calibration_web as web
from evomind_runtime.tools import build_default_registry
from evomind_runtime.policy import PolicyEngine


class Store:
    def __init__(self, policy):
        self.session = {'status': 'running', 'metadata': {
            'tenant_id': policy['tenant_id'], 'owner_principal_id': policy['owner_principal_id'],
            'managed_hpc_identity': deepcopy(policy['managed_hpc_identity'])}}
        self.run = {'status': 'running'}
        self.events = []

    def get_session(self, _):
        return self.session

    def get_assistant_run(self, _):
        return self.run

    def append_event(self, *args):
        self.events.append(args)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    from evomind_runtime import siim_calibration_control as control
    stage = tmp_path / 'stage'
    stage.mkdir()
    (stage / 'service-output').mkdir()
    policy = {'campaign_id': web.CAMPAIGN, 'enabled': True, 'tenant_id': 'tenant',
              'owner_principal_id': 'owner', 'managed_hpc_identity': {
                  'tenant_id': 'tenant', 'owner_principal_id': 'owner', 'job_id': 93207,
                  'allocation_generation': 27, 'profile_instance_id': 'instance'},
              'cases': {}, 'budget_path': str(tmp_path / 'budget.db'),
              'ev_budget_path': str(tmp_path / 'ev-budget.db')}
    policy_path = tmp_path / 'policy.json'
    policy_path.write_text(json.dumps(policy))
    names = ['siim_calibration_suite.py', 'siim_aide_controller.py']
    for name in names:
        (stage / name).write_text('# sealed fixture only')
    config = {'schema': 'evomind.siim_web_bridge.v1', 'policy_sha256': web.sha(policy_path),
              'coordinator_hashes': {name: web.sha(stage / name) for name in names}}
    config_path = tmp_path / 'web.json'
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(web, 'CONFIG', config_path)
    monkeypatch.setattr(web, 'STAGE', stage)
    monkeypatch.setattr(control, 'POLICY_PATH', policy_path)
    store = Store(policy)
    workspace = tmp_path / 'workspace'
    (workspace / 'outputs').mkdir(parents=True)
    ctx = SimpleNamespace(session_id='run_webtest', metadata=deepcopy(store.session['metadata']),
                          store=store, workspace_root=workspace, runtime_root=tmp_path,
                          approval_verified=True, approval_id='approval_fixture', progress=None)
    return ctx, policy, stage


def test_configuration_requires_actual_assistant_run(fixture):
    ctx, policy, _ = fixture
    assert web.configuration(ctx)[1] == policy
    ctx.store.run = None
    with pytest.raises(ValueError, match='requires_assistant_run'):
        web.configuration(ctx)


@pytest.mark.parametrize('key,value', [('tenant_id', 'other'), ('owner_principal_id', 'other')])
def test_owner_scope_cannot_be_spoofed_by_hpc_identity(fixture, key, value):
    ctx, _, _ = fixture
    ctx.metadata[key] = value
    with pytest.raises(ValueError, match='owner_mismatch'):
        web.configuration(ctx)


def test_allocation_and_coordinator_are_pinned(fixture):
    ctx, _, stage = fixture
    ctx.metadata['managed_hpc_identity']['job_id'] += 1
    with pytest.raises(ValueError, match='allocation_changed'):
        web.configuration(ctx)
    diagnostic = web.status_tool({}, ctx)
    assert diagnostic.error == 'siim_web_allocation_changed'
    assert diagnostic.content['current_allocation_health_not_evaluated'] is True
    assert diagnostic.content['automatic_rebind_recommended'] is False
    assert diagnostic.content['general_gpu_budget_tool'] == 'hpc_execution_budget_status'
    ctx.metadata = deepcopy(ctx.store.session['metadata'])
    (stage / 'siim_calibration_suite.py').write_text('# changed')
    with pytest.raises(ValueError, match='coordinator_changed'):
        web.configuration(ctx)


@pytest.mark.parametrize('status', ['paused', 'pausing', 'cancelled', 'completed', 'failed', 'blocked'])
def test_terminal_or_paused_parent_blocks_next_child_call(fixture, status):
    ctx, policy, _ = fixture
    ctx.store.run['status'] = status
    with pytest.raises(ValueError, match='parent_not_active'):
        web.require_parent(ctx.store, ctx.session_id, policy)


def test_parent_pause_flag_blocks_dispatch_before_terminal_state(fixture):
    ctx, policy, _ = fixture
    ctx.store.session['metadata']['user_pause_requested'] = True
    with pytest.raises(ValueError, match='parent_not_active'):
        web.require_parent(ctx.store, ctx.session_id, policy)


def test_run_requires_exact_approval_even_for_full_auto(tmp_path):
    registry = build_default_registry()
    assert not registry.get('hpc_calibration_run').read_only
    assert registry.get('hpc_calibration_status').read_only
    for level in ['workspace-write', 'full-auto']:
        decision = PolicyEngine().evaluate(tool_name='hpc_calibration_run',
            arguments={'campaign': web.CAMPAIGN}, permission_level=level, workspace_root=tmp_path)
        assert decision.requires_approval and not decision.allowed
    assert 'hpc_calibration_run' in {spec.name for spec in registry.specs_for_prompt('SIIM HPC训练')}


def test_unapproved_call_cannot_spawn(fixture, monkeypatch):
    ctx, _, _ = fixture
    ctx.approval_verified = False
    monkeypatch.setattr(web.subprocess, 'Popen', lambda *a, **kw: pytest.fail('unexpected process'))
    result = web.run_tool({'campaign': web.CAMPAIGN}, ctx)
    assert not result.ok and result.error == 'siim_web_exact_approval_required'


def test_duplicate_or_unsettled_dispatch_cannot_spawn(fixture, monkeypatch):
    from evomind_runtime import siim_calibration_budget
    ctx, _, stage = fixture
    monkeypatch.setattr(siim_calibration_budget, 'SiimBudget',
                        lambda *a: SimpleNamespace(summary=lambda: {'settlement_attention': False}))
    monkeypatch.setattr(web.subprocess, 'Popen', lambda *a, **kw: pytest.fail('unexpected process'))
    receipts = stage / 'service-output/web-dispatches'
    receipts.mkdir()
    (receipts / 'run_old.json').write_text(json.dumps({'status': 'running'}))
    result = web.run_tool({'campaign': web.CAMPAIGN}, ctx)
    assert not result.ok and 'requires_reconciliation' in result.error
    (receipts / 'run_old.json').write_text(json.dumps({'status': 'blocked'}))
    (receipts / 'run_webtest.json').write_text(json.dumps({'status': 'blocked'}))
    result = web.run_tool({'campaign': web.CAMPAIGN}, ctx)
    assert not result.ok and 'already_dispatched' in result.error


def test_success_is_frozen_candidates_not_official_score(fixture, monkeypatch):
    from evomind_runtime import siim_calibration_budget, tools
    ctx, _, stage = fixture
    monkeypatch.setattr(siim_calibration_budget, 'SiimBudget',
                        lambda *a: SimpleNamespace(summary=lambda: {'settlement_attention': False}))
    commands = []
    def spawn(argv, **kwargs):
        commands.append(argv)
        assert kwargs['cwd'] == stage
        return SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr(web.subprocess, 'Popen', spawn)
    monkeypatch.setattr(web, 'study_status', lambda *a: {
        'status': 'candidates_frozen_pending_private_grader_review', 'official_score': None,
        'official_submission_count': 0, 'completed_cases': 9})
    monkeypatch.setattr(tools, '_artifact_publish', lambda *a: SimpleNamespace(content={'verified': True}))
    result = web.run_tool({'campaign': web.CAMPAIGN}, ctx)
    assert result.ok and result.content['official_score'] is None
    assert commands[0][-2:] == ['--parent-run', ctx.session_id]
    assert any(event[1] == 'siim.web_dispatch' for event in ctx.store.events)
    receipt = json.loads((ctx.workspace_root / 'outputs/siim-execution-receipt.json').read_text())
    assert receipt['historical_results_preserved'] and receipt['coordinator_exit_code'] == 0


def test_suite_uses_fresh_verification_key_and_native_sessions():
    source = (Path(__file__).resolve().parents[1] / 'scripts/siim_calibration_suite.py').read_text(encoding='utf-8')
    assert "'-schema-v2-'+options.resume_key" in source
    assert "require_parent(parent_store,options.parent_run,policy)" in source
    assert "resource_lease(STAGE/'service-output/suite-leases'" in source
    assert "'/v1/sessions/'" in source and 'create_assistant_run' not in source


def test_budget_status_is_read_only_and_counts_pending_reservations(fixture):
    _, policy, _ = fixture
    for key in ['budget_path', 'ev_budget_path']:
        with sqlite3.connect(policy[key]) as connection:
            connection.execute('CREATE TABLE attempts(reserved REAL,charged REAL,status TEXT)')
            connection.execute("INSERT INTO attempts VALUES(1200,100,'failed')")
    with sqlite3.connect(policy['budget_path']) as connection:
        connection.execute("INSERT INTO attempts VALUES(600,0,'reserved')")
    before = {key: web.sha(policy[key]) for key in ['budget_path', 'ev_budget_path']}
    result = web.budget_readonly(policy)
    assert result['remaining_seconds'] == 85600
    assert result['settlement_attention'] is True
    assert before == {key: web.sha(policy[key]) for key in before}


def test_missing_budget_does_not_create_database(fixture):
    _, policy, _ = fixture
    with pytest.raises(sqlite3.OperationalError):
        web.budget_readonly(policy)
    assert not Path(policy['ev_budget_path']).exists()
