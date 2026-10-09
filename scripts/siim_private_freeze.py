"""Operator-side freeze builder and one-shot private grader driver for the SIIM calibration.

Runs on the application host under the service account (see
``start_siim_private_freeze.ps1``) so the named DPAPI HPC profile can be used.
It never trains, never submits to Kaggle and never exposes private labels to a
model: it only copies the three admitted frozen submissions into
``<artifact_root>/private-freeze``, writes the freeze receipt, and (in ``grade``
mode, once) invokes the official MLE-bench grader on the HPC container.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('C:/ProgramData/EvoMind')
STAGE = BASE / 'staging/siim-mlebench-calibration-20260908/runtime-extension'
OUTPUT = STAGE / 'service-output'
POLICY_PATH = BASE / 'config/official-calibration/siim-mlebench-20260908.json'
ARMS = ('fixed_baseline', 'evomind', 'aide')

PUBLIC_WORKER = r'''
import hashlib, json, shutil, time
from pathlib import Path

ARTIFACTS = Path(__ARTIFACTS__)
DATA = Path(__DATA__)
REQUESTS = json.loads(__REQUESTS__)
BLOCKED = Path(__BLOCKED__)
out = {'at': time.time(), 'arms': {}}
test_ids = None
test_csv = DATA / 'prepared/public/test.csv'
header = test_csv.read_text(encoding='utf-8').splitlines()
import csv, io
rows = list(csv.DictReader(io.StringIO(test_csv.read_text(encoding='utf-8'))))
test_ids = [row['image_name'] for row in rows]
if len(test_ids) != 4142 or len(set(test_ids)) != 4142:
    raise ValueError('official_test_manifest_invalid')
freeze_dir = ARTIFACTS / 'private-freeze'
freeze_dir.mkdir(exist_ok=True)
selected = {}
for arm in ('fixed_baseline', 'evomind', 'aide'):
    spec = REQUESTS[arm]
    root = Path(spec['hpc_artifact_root'])
    if not root.is_dir() or BLOCKED not in root.parents:
        raise ValueError('candidate_root_rejected:' + arm)
    source = root / 'submission.csv'
    if source.is_symlink() or not source.is_file():
        raise ValueError('candidate_submission_missing:' + arm)
    target = freeze_dir / (arm + '-submission.csv')
    shutil.copyfile(source, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    frame = list(csv.DictReader(io.StringIO(target.read_text(encoding='utf-8'))))
    names = [row['image_name'] for row in frame]
    values = [float(row['target']) for row in frame]
    if names != test_ids or len(values) != 4142:
        raise ValueError('submission_alignment_failed:' + arm)
    if any((v != v) or v < 0 or v > 1 for v in values):
        raise ValueError('submission_probability_invalid:' + arm)
    selected[arm] = {'submission_path': str(target), 'submission_sha256': digest,
                     'candidate_sha256': spec['candidate_sha256'], 'case_id': spec['case_id'],
                     'oof_roc_auc': spec['oof_roc_auc']}
    out['arms'][arm] = {'submission_sha256': digest, 'case_id': spec['case_id'],
                        'oof_roc_auc': spec['oof_roc_auc']}
freeze = {
    'schema': 'evomind.siim_private_score_freeze.v1',
    'competition': 'siim-isic-melanoma-classification',
    'evaluation_variant': 'mlebench_hidden_test_calibration_20260908',
    'all_candidates_frozen_before_test': True,
    'independent_verification_passed': True,
    'modeling_tasks_terminal': True,
    'private_feedback_used': False,
    'selected': selected,
    'frozen_at': time.time(),
}
freeze_path = freeze_dir / 'siim_private_score_freeze.json'
if freeze_path.exists():
    raise ValueError('freeze_already_exists')
temporary = freeze_path.with_suffix('.new')
temporary.write_text(json.dumps(freeze, indent=2, ensure_ascii=False), encoding='utf-8')
temporary.replace(freeze_path)
out['freeze_path'] = str(freeze_path)
out['freeze_sha256'] = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
out['status'] = 'frozen'
print(json.dumps(out, ensure_ascii=True, default=str))
'''


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observer_identity(binding: dict, tenant: str, owner: str, namespace: str) -> tuple[str, dict]:
    token = (BASE / 'data/workspace/runtime/runtime.token').read_text(encoding='utf-8').strip()

    def api(method: str, path: str, body=None, timeout: int = 300):
        conn = http.client.HTTPConnection('127.0.0.1', 8765, timeout=timeout)
        try:
            conn.request(method, path, json.dumps(body).encode() if body is not None else None,
                         {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
            response = conn.getresponse()
            raw = response.read(4 * 1024 * 1024)
            if response.status not in {200, 201}:
                raise ValueError('operator_api_failed:' + str(response.status))
            return json.loads(raw) if raw else {}
        finally:
            conn.close()

    session = 'session_siim_freeze_' + namespace
    api('POST', '/v1/sessions', {
        'session_id': session,
        'objective': 'Read-only SIIM freeze/grading operator; no training, no private labels to a model.',
        'permission_level': 'observe',
        'workspace_root': str(BASE / 'data/acceptance/siim-calibration-20260908/operator'),
        'metadata': {'managed_hpc_identity': binding, 'tenant_id': tenant,
                     'owner_principal_id': owner, 'run_allowed_tool_names': ['hpc_verify']}})
    verified = api('POST', '/v1/sessions/' + session + '/tools',
                   {'tool_name': 'hpc_verify', 'arguments': {},
                    'idempotency_key': 'freeze-identity-' + namespace})
    evidence = (verified.get('result') or {}).get('content') or {}
    from xsci.terminal_tools import hpc_identity_evidence_complete
    if not (verified.get('result') or {}).get('ok') or not hpc_identity_evidence_complete(
            evidence, expected_profile=binding['credential_profile'], expected_job_id=binding['job_id']):
        raise ValueError('freeze_operator_identity_failed')
    return session, evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--mode', choices=['freeze', 'grade'], default='freeze')
    args = parser.parse_args()
    if not args.namespace.replace('-', '').isalnum():
        raise ValueError('invalid_namespace')
    receipt_path = OUTPUT / ('private-freeze-' + args.namespace + '.json')
    if receipt_path.exists():
        raise ValueError('operation_exists')

    policy = json.loads(POLICY_PATH.read_text(encoding='utf-8-sig'))
    binding = policy['managed_hpc_identity']
    admitted = json.loads((OUTPUT / 'case-results-admitted-v2.json').read_text(encoding='utf-8-sig'))
    best = {arm: max((row for row in admitted if row.get('arm') == arm),
                     key=lambda row: (row.get('oof_roc_auc') or 0)) for arm in ARMS
            if any(row.get('arm') == arm for row in admitted)}
    if set(best) != set(ARMS):
        raise ValueError('all_three_arms_required')
    requests = {arm: {'hpc_artifact_root': row['hpc_artifact_root'],
                      'candidate_sha256': row['candidate_sha256'],
                      'case_id': row['case_id'], 'oof_roc_auc': row['oof_roc_auc']}
                for arm, row in best.items()}

    sys.path[:0] = [str(BASE / 'bundle/runtime'),
                    'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src']
    namespace = args.namespace + '-' + str(time.time_ns())[-6:]
    session, evidence = observer_identity(binding, policy['tenant_id'],
                                          policy['owner_principal_id'], namespace)
    from evomind_runtime.tools import _load_bound_hpc_config
    from research_agent_workstation.server.core.gpu_credentials import connect_ssh
    client = connect_ssh(_load_bound_hpc_config(binding['credential_profile'],
                                                binding['job_id'], session), timeout=30)
    try:
        if args.mode == 'freeze':
            program = (PUBLIC_WORKER
                       .replace('__ARTIFACTS__', repr(policy['artifact_root']))
                       .replace('__DATA__', repr('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_official_data/siim-isic-melanoma-classification'))
                       .replace('__REQUESTS__', repr(json.dumps(requests)))
                       .replace('__BLOCKED__', repr('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/siim_calibration_results')))
            _, stdout, stderr = client.exec_command('python3 -c ' + shlex.quote(program), timeout=600)
            raw = stdout.read(2 * 1024 * 1024)
            code = stdout.channel.recv_exit_status()
            result = json.loads(raw) if raw else {'status': 'failed'}
            if code:
                raise ValueError('freeze_worker_failed:' + stderr.read(4096).decode('utf-8', 'replace')[:200])
        else:
            freeze = (Path(policy['artifact_root']) / 'private-freeze'
                      / 'siim_private_score_freeze.json')
            sftp = client.open_sftp()
            remote_grader = str(Path(policy['artifact_root']) / 'private-freeze' / 'siim_private_score_frozen.py')
            with sftp.open(remote_grader, 'w') as handle:
                handle.write(Path(__file__).with_name('siim_private_score_frozen.py').read_text(encoding='utf-8'))
            sftp.close()
            _, digest_out, digest_err = client.exec_command(
                'sha256sum ' + shlex.quote(str(freeze)), timeout=120)
            freeze_digest = digest_out.read(4096).decode('utf-8', 'replace').split()[0] \
                if digest_out.channel.recv_exit_status() == 0 else ''
            if len(freeze_digest) != 64:
                raise ValueError('freeze_digest_unavailable:' + digest_err.read(2048).decode('utf-8', 'replace')[:160])
            command = ('cd ' + shlex.quote(str(Path(policy['artifact_root']) / 'private-freeze'))
                       + ' && python3 ' + shlex.quote(remote_grader)
                       + ' --freeze ' + shlex.quote(str(freeze))
                       + ' --freeze-sha256 ' + shlex.quote(freeze_digest))
            _, stdout, stderr = client.exec_command(command, timeout=1800)
            raw = stdout.read(2 * 1024 * 1024)
            code = stdout.channel.recv_exit_status()
            result = json.loads(raw) if raw else {'status': 'failed'}
            if code:
                raise ValueError('private_grading_failed:' + stderr.read(4096).decode('utf-8', 'replace')[:200])
    finally:
        client.close()

    receipt = {'schema': 'evomind.siim_private_freeze_operator.v1', 'mode': args.mode,
               'namespace': namespace, 'at': datetime.now(timezone.utc).isoformat(),
               'session_id': session, 'hpc_identity_samples': evidence.get('samples_passed'),
               'job_id': binding['job_id'], 'result': result, 'training_started': False,
               'kaggle_submission_made': False}
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=True, default=str))
    return 0


if __name__ == '__main__':
    main()
