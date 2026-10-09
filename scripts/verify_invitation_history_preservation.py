"""Prove the pinned manual launcher cannot replay preserved historical work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import uuid
from unittest.mock import patch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage-root', type=Path, required=True)
    parser.add_argument('--recovering-run-id', action='append', default=[])
    args = parser.parse_args()
    sys.path.insert(0, str(args.stage_root))
    import invitation_release_transaction as transaction
    root = transaction.ROOT
    stage = transaction.contained(args.stage_root, root / 'staging')
    before = transaction.quiescent_history(root, recovering_ids=args.recovering_run_id)
    launcher_path = root / 'bundle/runtime/run_python_runtime.py'
    fixture = stage / ('history-fixture-' + uuid.uuid4().hex)
    (fixture / 'bundle/runtime').mkdir(parents=True)
    (fixture / 'config').mkdir()
    (fixture / 'config/node-config.json').write_text(json.dumps({'hpc': {'state': 'blocked'}}))
    shutil.copyfile(launcher_path, fixture / 'bundle/runtime/run_python_runtime.py')
    sys.path.insert(0, str(stage / 'runtime'))
    from evomind_runtime.runtime import AgentRuntime
    from evomind_runtime.assistant_runs import AssistantRunService
    from evomind_runtime.models import utc_now
    runtime = AgentRuntime(fixture / 'data')
    for state in ('paused', 'recovering'):
        identifier = 'run_history_' + state
        runtime.create_session(session_id=identifier, objective='isolated preservation fixture',
                               metadata={'user_pause_requested': state == 'paused'})
        runtime.store.update_session(identifier, status=state)
        runtime.store.create_assistant_run({
            'id': identifier, 'session_id': identifier, 'conversation_id': 'fixture',
            'prompt': 'Do not execute this historical fixture.', 'task_root': str(fixture),
            'status': state, 'created_at': utc_now(), 'updated_at': utc_now(),
        })
    runtime.close()
    baseline = transaction.quiescent_history(fixture, recovering_ids=['run_history_recovering'])
    launcher = transaction.load_module(launcher_path, 'verified_manual_launcher')
    checks = []

    def forbidden(*args, **kwargs):
        raise RuntimeError('startup_replay_or_external_execution_detected')

    class IsolatedServer:
        def __init__(self, address, handler):
            assert address == ('127.0.0.1', 0)
        def serve_forever(self, **kwargs):
            checks.append('manual_http_startup_reached')
        def server_close(self):
            checks.append('manual_http_shutdown_reached')

    with patch.object(launcher, 'ThreadingHTTPServer', IsolatedServer), \
            patch.object(AssistantRunService, 'recover_incomplete', forbidden), \
            patch.object(AssistantRunService, 'start', forbidden), \
            patch.object(threading.Thread, 'start', forbidden), \
            patch.object(socket.socket, 'connect', forbidden), \
            patch.object(subprocess, 'Popen', forbidden):
        launcher.serve_manual(fixture / 'data', 0)
    if checks != ['manual_http_startup_reached', 'manual_http_shutdown_reached']:
        raise RuntimeError('manual_startup_not_exercised')
    transaction.assert_idle(fixture, preservation=baseline)
    transaction.assert_idle(root, preservation=before)
    receipt = {
        'schema': 'evomind.history_preservation_acceptance.v1', 'status': 'passed',
        'startup_replayed': False, 'hpc_accessed': False,
        'checks': checks + ['fixture_history_unchanged', 'production_history_unchanged', 'no_workers_or_external_execution'],
        'runtime_sha256': transaction.sha(stage / 'runtime.zip'),
        'candidate_code': transaction.code_tree(stage / 'runtime/evomind_runtime'),
        'harness_sha256': transaction.sha(Path(__file__)),
        'transaction_sha256': transaction.sha(stage / 'invitation_release_transaction.py'),
        'preservation': before,
    }
    transaction.write_json(stage / 'history-preservation-acceptance.json', receipt)
    print(json.dumps({'status': 'passed', 'preserved_runs': before['preserved_runs'],
                      'startup_replayed': False, 'hpc_accessed': False, 'checks': receipt['checks']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
