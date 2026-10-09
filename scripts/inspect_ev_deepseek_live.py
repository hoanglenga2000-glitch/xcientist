"""Read-only, exact-Run acceptance. Exposes no credentials or unrelated task text."""
import argparse,json,sqlite3
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path('C:/ProgramData/EvoMind')


def clean(value):
    if isinstance(value,dict):
        return {k:('[REDACTED]' if k.lower() in {'password','api_key','token','authorization','secret','private_key'} else clean(v)) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    if not args.run.startswith('run_') or len(args.run)!=36:raise ValueError('Exact run ID required')
    db=ROOT/'data/workspace/runtime/runtime.sqlite3'
    with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
        row=c.execute('SELECT id,session_id,status,model,model_provider,error_class FROM assistant_runs WHERE id=?',(args.run,)).fetchone()
        if not row:raise ValueError('Run missing')
        run=dict(row);session_id=run['session_id']
        meta=json.loads(c.execute('SELECT metadata_json FROM sessions WHERE id=?',(session_id,)).fetchone()[0])
        contract=meta.get('model_execution_contract',{})
        attempts=[json.loads(r[0]) for r in c.execute("SELECT payload_json FROM events WHERE session_id=? AND event_type='model.transport_attempt' ORDER BY seq",(session_id,))]
        calls=[{'id':r['id'],'name':r['tool_name'],'status':r['status'],'result':clean(json.loads(r['result_json'] or '{}'))}
               for r in c.execute('SELECT id,tool_name,status,result_json FROM tool_calls WHERE session_id=? ORDER BY created_at',(session_id,))]
    required=['designated_proxy_path_verified','pinned_gateway_host_key_verified','allocation_role_authenticated',
              'expected_host_uuid_match','expected_gpu_uuid_match','expected_gpu_model_and_memory_match','allowed_remote_root_match']
    hpc=[]
    for call in calls:
        if call['name']=='hpc_verify':
            content=call['result'].get('content') or {}
            hpc.append({'ok':call['result'].get('ok'),'status':content.get('status'),
                        'job_id':content.get('job_id'),'samples_passed':content.get('samples_passed'),
                        'job_container_verified':content.get('job_container_verified'),
                        'designated_proxy_path_verified':content.get('designated_proxy_path_verified'),
                        'read_only':content.get('read_only')})
    result={'schema':'evomind.deepseek_live_acceptance.v1','checked_at':datetime.now(timezone.utc).isoformat(),
            'run':run,'contract':{k:contract.get(k) for k in ['schema','model','provider','wire_protocol','timeout_seconds','max_request_retries','endpoint_sha256','route_config_sha256']},
            'model_attempts':attempts,'tool_calls':calls,'hpc_summary':hpc,
            'model_requests_succeeded':sum(x.get('status')=='completed' for x in attempts),
            'model_requests_failed':sum(x.get('status')=='failed' for x in attempts),
            'no_training_tools':all(x['name'] in {'hpc_verify','report_source_readiness'} for x in calls)}
    result['model_cutover_verified']=bool(contract.get('model')=='deepseek-v4-pro' and contract.get('wire_protocol')=='chat_completions_v1'
                                          and result['model_requests_succeeded'] and all(x.get('model')=='deepseek-v4-pro' for x in attempts))
    path=ROOT/'staging/ev-deepseek-v4-pro-20260908'/('live-'+args.run+'.json')
    path.write_text(json.dumps(result,ensure_ascii=True,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'tool_calls','model_attempts'}},ensure_ascii=True))


if __name__=='__main__':main()
