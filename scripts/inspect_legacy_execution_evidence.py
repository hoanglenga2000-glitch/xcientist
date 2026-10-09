"""Structural audit of four known legacy calls; no private result text emitted."""
import json
from pathlib import Path
import sqlite3

IDS=['call_b4c7d7d8095a451a5d0d5878f6671620','call_ad79487f22e6e88e8acc01bc8cdb9eb9',
     'call_5c0891e12bbd853969ef3aae01accb80','call_d2f39b4909fe4403aa821d191c649721']
database=Path('C:/ProgramData/EvoMind/data/workspace/runtime/runtime.sqlite3')
rows=[]
with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as connection:
    connection.row_factory=sqlite3.Row
    connection.execute('PRAGMA query_only=ON')
    for identifier in IDS:
        call=dict(connection.execute('SELECT * FROM tool_calls WHERE id=?',(identifier,)).fetchone())
        result=json.loads(call.get('result_json') or '{}')
        arguments=json.loads(call.get('arguments_json') or '{}')
        session=dict(connection.execute('SELECT * FROM sessions WHERE id=?',(call['session_id'],)).fetchone())
        metadata=json.loads(session['metadata_json'])
        identity=metadata.get('managed_hpc_identity') or {}
        receipt_paths=[]
        if call['tool_name']=='hpc_execute_solution':
            workspace=Path(session['workspace_root'])
            base=workspace/'work/hpc/solutions'
            if base.is_dir():
                receipt_paths=[str(path.relative_to(workspace)) for path in base.rglob('*.json')
                    if path.name in {'execution-result.json','verification.json','result.json','receipt.json','metrics.json'}][:40]
        events=[]
        for raw in connection.execute('SELECT event_type,created_at,payload_json FROM events WHERE session_id=? AND payload_json LIKE ? ORDER BY seq',(call['session_id'],'%'+identifier+'%')):
            payload=json.loads(raw['payload_json'])
            nested=payload.get('result') if isinstance(payload.get('result'),dict) else {}
            events.append({'event_type':raw['event_type'],'created_at':raw['created_at'],'payload_fields':list(payload),
                           'result_fields':list(nested),'ok':nested.get('ok'),'status':payload.get('status')})
        rows.append({'call_id':identifier,'tool':call['tool_name'],'result_fields':list(result),'result_ok':result.get('ok'),
            'result_content_fields':list(result.get('content') or {}) if isinstance(result.get('content'),dict) else [],
            'workspace':session['workspace_root'],'metadata_fields':list(metadata),'job_id':identity.get('job_id'),
            'profile':identity.get('credential_profile'),'argument_fields':list(arguments),
            'requested_job_id':arguments.get('job_id'),'requested_profile':arguments.get('credential_profile'),
            'solution_id':arguments.get('solution_id'),'local_receipt_paths':receipt_paths,'events':events})
print(json.dumps({'scope':'structural_metadata_only','calls':rows},ensure_ascii=True))
