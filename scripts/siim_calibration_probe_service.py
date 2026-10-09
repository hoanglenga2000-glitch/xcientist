"""Service-owner SIIM preflight: fresh managed identity, bounded read-only inventory."""
import hashlib,http.client,json,shlex,sqlite3,sys,time
from pathlib import Path
from datetime import datetime,timezone
BASE=Path('C:/ProgramData/EvoMind')
STAGE=BASE/'staging/siim-mlebench-calibration-20260908'
sys.path[:0]=[str(BASE/'bundle/runtime'),'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src']

def main():
    policy=json.loads((BASE/'config/official-calibration/ev-public-20260908.json').read_text())
    with sqlite3.connect(Path(policy['budget_path']).as_uri()+'?mode=ro',uri=True) as c:
        c.execute('PRAGMA query_only=ON');attempts=c.execute('SELECT charged,status,reserved FROM attempts').fetchall()
        # A read-only inventory can overlap the one already-running EV worker.
        # New training still requires its final settlement and budget snapshot.
        unsettled=any(r[1] not in {'completed','failed'} for r in attempts)
    binding=policy['managed_hpc_identity'];session='session_siim_calibration_environment_'+str(time.time_ns())
    token=(BASE/'data/workspace/runtime/runtime.token').read_text().strip()
    def api(method,path,body):
        conn=http.client.HTTPConnection('127.0.0.1',8765,timeout=180)
        try:
            conn.request(method,path,json.dumps(body).encode(),{'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            response=conn.getresponse();raw=response.read(2*1024*1024)
            if response.status not in {200,201}:raise ValueError('inventory_api_failed')
            return json.loads(raw)
        finally:conn.close()
    api('POST','/v1/sessions',{'session_id':session,'objective':'SIIM MLE-bench read-only data and framework inventory; no private scoring or fitting.',
        'permission_level':'observe','workspace_root':str(BASE/'data/acceptance/siim-calibration-20260908/operator'),
        'metadata':{'managed_hpc_identity':binding,'tenant_id':policy['tenant_id'],'owner_principal_id':policy['owner_principal_id'],
                    'run_allowed_tool_names':['hpc_verify']}})
    checked=api('POST','/v1/sessions/'+session+'/tools',{'tool_name':'hpc_verify','arguments':{},'idempotency_key':'fresh-siim-preflight'})
    result=checked.get('result') or {};evidence=result.get('content') or {}
    from xsci.terminal_tools import hpc_identity_evidence_complete
    if not result.get('ok') or not hpc_identity_evidence_complete(evidence,expected_profile=binding['credential_profile'],expected_job_id=binding['job_id']):
        raise ValueError('siim_identity_not_ready')
    from evomind_runtime.tools import _load_bound_hpc_config
    from research_agent_workstation.server.core.gpu_credentials import connect_ssh
    client=connect_ssh(_load_bound_hpc_config(binding['credential_profile'],binding['job_id'],session),timeout=30)
    program=(STAGE/'siim_calibration_environment_probe.py').read_text()
    try:
        _,stdout,stderr=client.exec_command('python3 -c '+shlex.quote(program),timeout=90)
        raw=stdout.read(262145);code=stdout.channel.recv_exit_status()
        if code or len(raw)>262144:raise ValueError('siim_inventory_failed')
        inventory=json.loads(raw)
    finally:client.close()
    receipt={'schema':'evomind.siim_calibration_inventory.v1','at':datetime.now(timezone.utc).isoformat(),
        'session_id':session,'hpc_identity':evidence,'inventory':inventory,
        'ev_charged_slot_seconds':sum(r[0] for r in attempts),'remaining_total_slot_seconds':86400-sum(r[0] for r in attempts),
        'ev_settlement_pending':unsettled,'conservative_available_slot_seconds':86400-sum(max(r[0],r[2]) if r[1] not in {'completed','failed'} else r[0] for r in attempts),
        'training_started':False,'private_labels_read':False,'gpu_files_written':False}
    output=STAGE/'service-output/environment.json'
    with output.open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({'status':'inventory_complete','inventory':inventory,'remaining_gpu_slot_hours':receipt['remaining_total_slot_seconds']/3600}))

if __name__=='__main__':
    try:main()
    except Exception as e:
        with (STAGE/'service-output/environment-error.json').open('w') as f:json.dump({'status':'blocked','error_type':type(e).__name__,'detail':str(e) if isinstance(e,ValueError) else 'inventory_failed'},f)
        raise SystemExit(2)
