"""Application-host read-only evidence projection; no HPC connection or secrets."""
import argparse
import json
from pathlib import Path
import re
import sqlite3
import sys

BASE = Path('C:/ProgramData/EvoMind')
sys.path[:0] = [str(BASE/'bundle/runtime'), 'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    args = p.parse_args()
    if not re.fullmatch('run_[a-f0-9]{32}', args.run):
        raise ValueError('invalid_run')
    policy = json.loads((BASE/'config/official-calibration/siim-mlebench-20260908.json').read_text())
    with sqlite3.connect((BASE/'data/workspace/runtime/runtime.sqlite3').as_uri()+'?mode=ro', uri=True) as c:
        c.row_factory = sqlite3.Row
        c.execute('PRAGMA query_only=ON')
        parent = c.execute('SELECT id,status,model,error_class FROM assistant_runs WHERE id=?', (args.run,)).fetchone()
        if not parent:
            raise ValueError('no_assistant_run')
        result = {'parent': dict(parent), 'children': []}
        def project(run_id):
            row = c.execute('SELECT status,metadata_json FROM sessions WHERE id=?', (run_id,)).fetchone()
            if row is None:
                return None
            meta = json.loads(row['metadata_json'])
            calls = []
            for raw in c.execute('SELECT id,tool_name,status,result_json FROM tool_calls WHERE session_id=? ORDER BY created_at', (run_id,)):
                item = {key: raw[key] for key in ['id','tool_name','status']}
                data = json.loads(raw['result_json'] or '{}')
                content = data.get('content') or {}
                error = str(data.get('error') or '')
                item['error_code'] = error if re.fullmatch('[A-Za-z0-9_:. -]{0,140}', error) else 'see_private_operator_record'
                if raw['tool_name'] == 'hpc_verify' and data.get('ok'):
                    from xsci.terminal_tools import hpc_identity_evidence_complete
                    binding = policy['managed_hpc_identity']
                    item['complete_identity_receipt'] = hpc_identity_evidence_complete(content, expected_profile=binding['credential_profile'], expected_job_id=binding['job_id'])
                if content.get('metrics'):
                    item['metrics'] = {k: content['metrics'].get(k) for k in ['oof_roc_auc','oof_complete','seed','runtime_seconds','candidate_sha256']}
                calls.append(item)
            events = c.execute("SELECT payload_json FROM events WHERE session_id=? AND event_type='model.transport_attempt' ORDER BY seq", (run_id,)).fetchall()
            progress = c.execute('SELECT payload FROM execution_progress WHERE run_id=?', (run_id,)).fetchone()
            from evomind_runtime.execution_progress import project_progress
            return {'run_id': run_id, 'status': row['status'], 'web_parent': meta.get('siim_web_parent_run'),
                    'model_contract': {k:(meta.get('model_execution_contract') or {}).get(k) for k in ['model','wire_protocol','endpoint_sha256']},
                    'calls': calls[-6:], 'model_transport_event_count': len(events),
                    'progress': project_progress(json.loads(progress[0]) if progress else None, status=row['status'])}
        result['parent']['execution'] = project(args.run)
        for run_id in policy['cases']:
            entry = project(run_id)
            if entry:
                result['children'].append(entry)
    from evomind_runtime.siim_calibration_web import budget_readonly
    result['budget'] = budget_readonly(policy)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
