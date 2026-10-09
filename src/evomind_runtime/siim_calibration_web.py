"""Web-owned admission for the registered SIIM study; never fabricates child Runs.

The Assistant owns this long-running tool. The sealed coordinator still uses
native, isolated EvoMind sessions and the original shared budget. Historical
child work remains historical; only subsequent dispatches carry this parent.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from .models import ToolResult, utc_now
from .training_control import identity, read_json
from .execution_progress import project_progress, resource_lease

BASE = Path('C:/ProgramData/EvoMind')
STAGE = BASE / 'staging/siim-mlebench-calibration-20260908/runtime-extension'
CONFIG = BASE / 'config/official-calibration/siim-web-bridge.json'
CAMPAIGN = 'siim_mlebench_calibration_20260908'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_parent(store, parent_id, policy):
    """Fail closed on cancellation, lost parent, identity drift, or paused Run."""
    parent = store.get_session(parent_id)
    run = store.get_assistant_run(parent_id)
    if not parent or not run:
        raise ValueError('siim_web_parent_missing')
    meta = parent.get('metadata') or {}
    if identity(meta)[:2] != (policy['tenant_id'], policy['owner_principal_id']):
        raise ValueError('siim_web_parent_owner_mismatch')
    if meta.get('managed_hpc_identity') != policy['managed_hpc_identity']:
        raise ValueError('siim_web_parent_allocation_changed')
    if (meta.get('user_pause_requested') or parent['status'] in {'paused', 'pausing', 'cancelled'}
            or run['status'] in {'paused', 'pausing', 'cancelled', 'completed', 'failed', 'blocked'}):
        raise ValueError('siim_web_parent_not_active')
    return parent


def configuration(context):
    from .siim_calibration_control import POLICY_PATH
    config = read_json(CONFIG)
    policy = read_json(POLICY_PATH)
    if (config.get('schema') != 'evomind.siim_web_bridge.v1'
            or config.get('policy_sha256') != sha(POLICY_PATH)
            or policy.get('campaign_id') != CAMPAIGN or policy.get('enabled') is not True):
        raise ValueError('siim_web_configuration_changed')
    if identity(context.metadata)[:2] != (policy['tenant_id'], policy['owner_principal_id']):
        raise ValueError('siim_web_owner_mismatch')
    if context.metadata.get('managed_hpc_identity') != policy['managed_hpc_identity']:
        raise ValueError('siim_web_allocation_changed')
    if not context.store.get_assistant_run(context.session_id):
        raise ValueError('siim_web_requires_assistant_run')
    expected = {'siim_calibration_suite.py', 'siim_aide_controller.py'}
    if set(config.get('coordinator_hashes') or {}) != expected:
        raise ValueError('siim_web_coordinator_manifest_invalid')
    for name, digest in config['coordinator_hashes'].items():
        if sha(STAGE / name) != digest:
            raise ValueError('siim_web_coordinator_changed')
    return config, policy


def budget_readonly(policy):
    def rows(path, query):
        with sqlite3.connect(Path(path).as_uri() + '?mode=ro', uri=True) as connection:
            connection.execute('PRAGMA query_only=ON')
            return connection.execute(query).fetchall()
    ev = rows(policy['ev_budget_path'], 'SELECT charged,status FROM attempts')
    current = rows(policy['budget_path'], 'SELECT reserved,charged,status FROM attempts')
    used_ev = sum(row[0] for row in ev)
    used = sum(row[1] for row in current)
    reserved = sum(row[0] for row in current if row[2] == 'reserved')
    return {'total_limit_seconds': 86400, 'ev_charged_seconds': used_ev,
            'siim_charged_seconds': used, 'pending_reserved_seconds': reserved,
            'remaining_seconds': max(0, 86400-used_ev-used-reserved),
            'settlement_attention': any(row[1] not in {'completed', 'failed'} for row in ev)
                or any(row[2] not in {'completed', 'failed'} for row in current),
            'attempt_count': len(current)}


def study_status(context, policy):
    progress_file = STAGE / 'service-output/progress.json'
    progress = read_json(progress_file) if progress_file.is_file() else {}
    child_id = str(progress.get('run_id') or '')
    case = policy['cases'].get(child_id, {})
    child_progress = None
    linked = False
    if case:
        child = context.store.get_session(child_id)
        if child:
            linked = (child.get('metadata') or {}).get('siim_web_parent_run') == context.session_id
            with sqlite3.connect((context.runtime_root / 'runtime.sqlite3').as_uri() + '?mode=ro', uri=True) as connection:
                saved = connection.execute('SELECT payload FROM execution_progress WHERE run_id=?', (child_id,)).fetchone()
            child_progress = project_progress(json.loads(saved[0]) if saved else None, status=child['status'])
    return {
        'schema': 'evomind.siim_web_status.v1', 'campaign': CAMPAIGN,
        'status': progress.get('status', 'not_started'),
        'active_case': case.get('case_id'), 'child_session_id': child_id or None,
        'child_linked_to_this_web_run': linked,
        'completed_cases': int(progress.get('completed_cases') or 0), 'total_cases': 9,
        'child_progress': child_progress,
        'budget': budget_readonly(policy),
        'official_score': None, 'official_submission_count': 0,
        'historical_child_work_not_relabelled': True,
    }


def status_tool(args, context):
    try:
        _, policy = configuration(context)
        return ToolResult('', True, study_status(context, policy), 'Registered SIIM study status; not an official score')
    except ValueError as exc:
        return ToolResult('', False, {
            'scope': 'registered_siim_campaign', 'campaign': CAMPAIGN,
            'general_gpu_budget_tool': 'hpc_execution_budget_status',
            'current_allocation_health_not_evaluated': True,
            'automatic_rebind_recommended': False,
        }, 'Registered SIIM study status unavailable. This does not diagnose current allocation health or the general GPU budget. Keep the old study binding immutable; do not recommend another rebind from this error alone.', error=str(exc))


def run_tool(args, context):
    """One exact-approved, synchronous dispatch with durable web provenance."""
    try:
        config, policy = configuration(context)
        if args.get('campaign') != CAMPAIGN or not context.approval_verified:
            raise ValueError('siim_web_exact_approval_required')
        require_parent(context.store, context.session_id, policy)
        from .siim_calibration_budget import SiimBudget
        budget = SiimBudget(policy['budget_path'], policy['ev_budget_path'], policy.get('resource_limits')).summary()
        if budget['settlement_attention']:
            raise ValueError('siim_web_unsettled_execution')
        receipts = STAGE / 'service-output/web-dispatches'
        receipts.mkdir(exist_ok=True)
        with resource_lease(receipts, CAMPAIGN):
            # A crash is not permission to replace an unconfirmed supervisor.
            for path in receipts.glob('run_*.json'):
                old = read_json(path)
                if old.get('status') == 'running':
                    raise ValueError('siim_web_previous_dispatch_requires_reconciliation')
            from .siim_candidate_contract import dispatch_paths
            operation = str(args.get('operation_id') or '')
            receipt_path, resume_key = dispatch_paths(receipts,context.session_id,operation)
            if receipt_path.exists():
                raise ValueError('siim_web_run_already_dispatched')
            receipt = {
                'schema': 'evomind.siim_web_dispatch.v1', 'parent_run_id': context.session_id,
                'status': 'running', 'started_at': utc_now(), 'policy_sha256': config['policy_sha256'],
                'child_sessions': list(policy['cases']), 'budget_before': budget,
                'historical_results_preserved': True, 'official_submissions': 0,
                'operation_id': operation, 'approval_id': context.approval_id,
            }
            with receipt_path.open('x', encoding='utf-8') as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2)
            context.store.append_event(context.session_id, 'siim.web_dispatch', receipt)
            # This process only coordinates; every fit stays behind the existing
            # training_route -> full identity verification -> HPC execution gate.
            log_path = receipt_path.with_suffix('.log')
            with log_path.open('xb') as log:
                process = subprocess.Popen(
                    [sys.executable, '-X', 'utf8', str(STAGE / 'siim_calibration_suite.py'),
                     '--resume', '--resume-key', resume_key,
                     '--parent-run', context.session_id],
                    cwd=STAGE, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                )
                last_case = None
                while process.poll() is None:
                    snapshot = study_status(context, policy)
                    active = snapshot['child_session_id']
                    if active and active != last_case and snapshot['child_linked_to_this_web_run']:
                        context.store.append_event(context.session_id, 'siim.child_active', {
                            'child_session_id': active, 'case_id': snapshot['active_case'],
                            'child_is_native_session_not_assistant_run': True,
                        })
                        context.store.append_event(context.session_id, 'step_started', {
                            'step_id': 'siim-child-' + active, 'status': 'running',
                            'label': '受管子实验：' + snapshot['active_case'],
                            'detail': '独立会话 ' + active + '；历史结果保留，新执行关联当前网页任务。',
                        })
                        last_case = active
                    if context.progress:
                        # Supervisor liveness does not establish GPU optimizer progress.
                        context.progress(source='managed_adapter', work_kind='executing',
                            phase='siim_calibration_supervision', worker_state='running',
                            completed_units=snapshot['completed_cases'], total_units=9, unit='cases',
                            budget=snapshot['budget'],
                            detail=f"SIIM {snapshot['active_case'] or 'preflight'} · {(snapshot['child_progress'] or {}).get('phase', '等待子任务回执')}；已完成 {snapshot['completed_cases']}/9。暂停将在当前有界步骤结束后生效。")
                    time.sleep(5)
            snapshot = study_status(context, policy)
            ready = process.returncode == 0 and snapshot['status'] == 'candidates_frozen_pending_private_grader_review'
            receipt.update(status='completed' if ready else 'blocked', finished_at=utc_now(),
                           coordinator_exit_code=process.returncode, result=snapshot,
                           log_sha256=sha(log_path))
            temporary = receipt_path.with_suffix('.new')
            temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(receipt_path)
            receipt_name='siim-execution-receipt'+('--'+operation if operation else '')+'.json'
            target = context.workspace_root / 'outputs'/receipt_name
            target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
            from .tools import _artifact_publish
            published = _artifact_publish({'path': 'outputs/'+receipt_name}, context)
            context.store.append_event(context.session_id, 'siim.web_settled', receipt)
            return ToolResult('', ready, {**snapshot, 'receipt': published.content},
                              'SIIM candidates frozen; hidden scoring still pending' if ready else
                              'SIIM stopped with evidence preserved; do not automatically redispatch',
                              error='' if ready else 'siim_web_child_not_complete')
    except (ValueError, RuntimeError) as exc:
        return ToolResult('', False, {}, 'SIIM web admission blocked; no replacement was started', error=str(exc))
