"""Export small, source-bound receipts only. Never exports rows/models/predictions."""
import hashlib,json,sqlite3,time
from datetime import datetime,timezone
from pathlib import Path

BASE=Path('C:/ProgramData/EvoMind')
STAGE=BASE/'staging/ev-public-calibration-20260908/runtime-extension'

def main():
    policy_path=BASE/'config/official-calibration/ev-public-20260908.json'
    raw=policy_path.read_bytes();policy=json.loads(raw)
    db=BASE/'data/workspace/runtime/runtime.sqlite3'
    cases=[]
    with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
        for run,case in policy['cases'].items():
            calls=c.execute('SELECT id,tool_name,status,arguments_json,result_json,created_at FROM tool_calls WHERE session_id=? ORDER BY created_at',(run,)).fetchall()
            candidates=[];writes=[]
            for call in calls:
                args=json.loads(call['arguments_json'] or '{}');result=json.loads(call['result_json'] or '{}');content=result.get('content') or {}
                if call['tool_name']=='file_write' and result.get('ok'):
                    body=args.get('content')
                    if isinstance(body,str):writes.append({'call_id':call['id'],'path':args.get('path'),'sha256':hashlib.sha256(body.encode()).hexdigest()})
                if call['tool_name']=='hpc_execute_solution':
                    candidate={'call_id':call['id'],'status':call['status'],'ok':result.get('ok'),
                               'source_path':args.get('script_path'),'attempt_id':content.get('attempt_id'),
                               'metrics':content.get('metrics'),'diagnostic':content.get('diagnostic')}
                    attempt=content.get('attempt_id')
                    if isinstance(attempt,str) and len(attempt)==32 and all(x in '0123456789abcdef' for x in attempt):
                        root=Path(case['workspace_root'])/'outputs/hpc'/('ev-'+attempt)
                        manifest=root/'artifact-manifest.json'
                        if manifest.is_file() and not manifest.is_symlink() and manifest.stat().st_size<256*1024:
                            candidate['artifact_manifest']=json.loads(manifest.read_text())
                            candidate['artifact_manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
                    metric=candidate['metrics'] or {}
                    candidate['matching_model_file_write_calls']=[w['call_id'] for w in writes if w['sha256']==metric.get('candidate_sha256')]
                    candidates.append(candidate)
            events=[json.loads(r[0]) for r in c.execute("SELECT payload_json FROM events WHERE session_id=? AND event_type='model.transport_attempt' ORDER BY seq",(run,))]
            cases.append({'run_id':run,'case_id':case['case_id'],'arm':case['arm'],'seed':case['seed'],
                          'candidates':candidates,'model_attempts':events,'tool_status_counts':{
                              status:sum(row['status']==status for row in calls) for status in {row['status'] for row in calls}}})
    with sqlite3.connect(Path(policy['budget_path']).as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
        attempts=[dict(row) for row in c.execute('SELECT id,run_id,source_sha,reserved,charged,status FROM attempts ORDER BY rowid')]
        clocks=[dict(row) for row in c.execute('SELECT run_id,started FROM cases ORDER BY started')]
    receipt={'schema':'evomind.ev_calibration_evidence_snapshot.v1','at':datetime.now(timezone.utc).isoformat(),
        'competition':'playground-series-s6e9','policy_sha256':hashlib.sha256(raw).hexdigest(),
        'cases':cases,'budget_attempts':attempts,'case_clocks':clocks,'official_submissions_by_suite':0,
        'raw_data_exported':False,'model_weights_exported':False,'full_predictions_exported':False}
    out=STAGE/'service-output'/('evidence-snapshot-'+str(time.time_ns())+'.json')
    with out.open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
                      'bytes':out.stat().st_size,'cases_with_completed_candidates':sum(any(a.get('ok') for a in c['candidates']) for c in cases)}))

if __name__=='__main__':main()
