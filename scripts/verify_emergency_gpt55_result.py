"""Read-only acceptance of the exact Chrome-created recovery Run."""
import hashlib,json,sqlite3
from pathlib import Path
ROOT=Path('C:/ProgramData/EvoMind')
RUN='run_a22789294b004a8c96390f0b59035417'
db=ROOT/'data/workspace/runtime/runtime.sqlite3'
backup=ROOT/'backups/emergency-gpt55-20260908-v5/databases/workspace/runtime/runtime.sqlite3'
def connect(path):
    c=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;return c
with connect(db) as c,connect(backup) as old:
    preserved={}
    for table in ['assistant_runs','tool_calls','approvals','sessions']:
        columns=[r[1] for r in old.execute('PRAGMA table_info('+table+')')]
        after={r['id']:dict(r) for r in c.execute('SELECT '+','.join(columns)+' FROM '+table)}
        original=[dict(r) for r in old.execute('SELECT * FROM '+table)]
        preserved[table]={'old_rows':len(original),'unchanged':all(after.get(r['id'])==r for r in original)}
    run=dict(c.execute('SELECT id,status,model,model_provider FROM assistant_runs WHERE id=?',(RUN,)).fetchone())
    meta=json.loads(c.execute('SELECT metadata_json FROM sessions WHERE id=?',(RUN,)).fetchone()[0])
    contract=meta.get('model_execution_contract',{})
    calls=[{'tool':r['tool_name'],'status':r['status'],'ok':json.loads(r['result_json'] or '{}').get('ok')}
        for r in c.execute('SELECT tool_name,status,result_json FROM tool_calls WHERE session_id=?',(RUN,))]
    attempts=[json.loads(r[0]) for r in c.execute("SELECT payload_json FROM events WHERE session_id=? AND event_type='model.transport_attempt'",(RUN,))]
path=ROOT/'data/workspace/runtime/assistant_tasks'/RUN/'outputs/fallback-check.txt'
data=path.read_bytes()
result={'schema':'evomind.emergency_gpt55_acceptance.v1','run':run,'tools':calls,
    'model_contract':{k:contract.get(k) for k in ['schema','model','provider','wire_protocol','reasoning_effort','timeout_seconds','max_request_retries']},
    'model_requests':len(attempts),'model_failures':sum(r.get('status')=='failed' for r in attempts),
    'historical_rows':preserved,'artifact_bytes':len(data),'artifact_sha256':hashlib.sha256(data).hexdigest(),
    'correct_result':b'323' in data and b'EMERGENCY_GPT55_OK' in data,
    'new_gpu_training_started':False,'full_invitation_beta_release_claim':False}
result['passed']=run['status']=='completed' and contract.get('model')=='gpt-5.5' and result['correct_result'] and all(r['ok'] for r in calls) and all(r['unchanged'] for r in preserved.values())
output=ROOT/'staging/emergency-gpt55-20260908/live-acceptance.json'
with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
