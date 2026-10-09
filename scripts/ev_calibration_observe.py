"""Read-only exact-attempt fit progress, using a fresh managed identity proof."""
import hashlib,http.client,json,shlex,sqlite3,sys,time
from pathlib import Path
from datetime import datetime,timezone
BASE=Path('C:/ProgramData/EvoMind');STAGE=BASE/'staging/ev-public-calibration-20260908/runtime-extension'
sys.path[:0]=[str(BASE/'bundle/runtime'),'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src']


def main():
    policy=json.loads((BASE/'config/official-calibration/ev-public-20260908.json').read_text())
    with sqlite3.connect(Path(policy['budget_path']).as_uri()+'?mode=ro',uri=True) as c:
        c.execute('PRAGMA query_only=ON');rows=c.execute("SELECT id,run_id FROM attempts WHERE status='reserved'").fetchall()
    if len(rows)!=1:raise ValueError('Exactly one running attempt required')
    attempt,run_id=rows[0];case=policy['cases'][run_id]
    root=policy['artifact_root']+'/'+case['case_id']+'/'+attempt
    token=(BASE/'data/workspace/runtime/runtime.token').read_text().strip()
    def api(method,path,body=None):
        c=http.client.HTTPConnection('127.0.0.1',8765,timeout=180)
        try:
            c.request(method,path,body=json.dumps(body).encode() if body is not None else None,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            r=c.getresponse();raw=r.read(2*1024*1024)
            if r.status not in {200,201}:raise ValueError('Observer API failed')
            return json.loads(raw)
        finally:c.close()
    observer='session_ev_fit_observer_'+str(time.time_ns())
    binding=policy['managed_hpc_identity']
    api('POST','/v1/sessions',{'session_id':observer,'objective':'Read-only EV attempt checkpoint progress; no training or mutation.',
                             'permission_level':'observe','workspace_root':str(BASE/'data/acceptance/ev-calibration-20260908/observer'),
                             'metadata':{'managed_hpc_identity':binding,'tenant_id':policy['tenant_id'],'owner_principal_id':policy['owner_principal_id'],'run_allowed_tool_names':['hpc_verify']}})
    verified=api('POST','/v1/sessions/'+observer+'/tools',{'tool_name':'hpc_verify','arguments':{},'idempotency_key':'fresh-observer-identity'})
    evidence=(verified.get('result') or {}).get('content') or {}
    from xsci.terminal_tools import hpc_identity_evidence_complete
    if not hpc_identity_evidence_complete(evidence,expected_profile=binding['credential_profile'],expected_job_id=binding['job_id']):raise ValueError('Observer identity failed')
    from evomind_runtime.tools import _load_bound_hpc_config
    from research_agent_workstation.server.core.gpu_credentials import connect_ssh
    client=connect_ssh(_load_bound_hpc_config(binding['credential_profile'],binding['job_id'],observer),timeout=30)
    try:
        program="""import json,subprocess
from pathlib import Path
p=Path(ROOT_VALUE)
r={'root_exists':p.exists(),'completed_fit_events':[],'checkpoints':[]}
events=p/'fit-events.jsonl'
if events.is_file() and events.stat().st_size<65536:
 for line in events.read_text().splitlines():
  try:
   x=json.loads(line);r['completed_fit_events'].append({k:x[k] for k in ['phase','fold','seed','train_rows','valid_rows','auc','ended'] if k in x})
  except ValueError:pass
if p.exists():r['checkpoints']=[{'name':f.name,'bytes':f.stat().st_size} for f in p.glob('*.joblib') if f.is_file()]
r['final_metrics_present']=(p/'metrics.json').is_file()
q=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
r['gpu_summary']=q.stdout.strip() if q.returncode==0 else None
print(json.dumps(r))
""".replace('ROOT_VALUE',repr(root))
        _,stdout,stderr=client.exec_command('python3 -c '+shlex.quote(program),timeout=30)
        data=stdout.read(131072);status=stdout.channel.recv_exit_status()
        if status:raise ValueError('Progress read failed')
        observed=json.loads(data)
    finally:client.close()
    receipt={'schema':'evomind.ev_attempt_observation.v1','at':datetime.now(timezone.utc).isoformat(),'case_id':case['case_id'],
             'run_id':run_id,'attempt_id':attempt,'observer_session':observer,'hpc_identity_samples':5,'read_only':True,'observed':observed}
    output=STAGE/'service-output'/('observation-'+str(time.time_ns())+'.json');output.write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


if __name__=='__main__':
    try:main()
    except Exception as e:(STAGE/'service-output/observation-error.json').write_text(json.dumps({'status':'failed','error_type':type(e).__name__,'code':str(e) if isinstance(e,ValueError) else 'observer_failed'}));raise SystemExit(2)
