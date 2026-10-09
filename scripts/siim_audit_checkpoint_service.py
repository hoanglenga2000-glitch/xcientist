"""Read-only checkpoint hashing for the single active SIIM attempt, with fresh identity."""
import hashlib,http.client,json,shlex,sqlite3,sys,time
from pathlib import Path
BASE=Path('C:/ProgramData/EvoMind');STAGE=BASE/'staging/siim-mlebench-calibration-20260908/runtime-extension'
sys.path[:0]=[str(BASE/'bundle/runtime'),'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src']

def main():
    policy=json.loads((BASE/'config/official-calibration/siim-mlebench-20260908.json').read_text())
    with sqlite3.connect(Path(policy['budget_path']).as_uri()+'?mode=ro',uri=True) as c:
        c.execute('PRAGMA query_only=ON');rows=c.execute("SELECT id,run_id,source_sha FROM attempts WHERE status='reserved'").fetchall()
    if len(rows)!=1:raise ValueError('single_active_attempt_required')
    attempt,run,source_sha=rows[0];case=policy['cases'][run];root=policy['artifact_root']+'/'+case['case_id']+'/'+attempt
    expected={'arm':case['arm'],'seed':case['seed'],'candidate_sha256':source_sha,'protocol_sha256':policy['protocol_sha256'],'data_manifest_sha256':policy['data_manifest_sha256']}
    binding=policy['managed_hpc_identity'];session='session_siim_checkpoint_audit_'+str(time.time_ns())
    token=(BASE/'data/workspace/runtime/runtime.token').read_text().strip()
    def api(path,body):
        connection=http.client.HTTPConnection('127.0.0.1',8765,timeout=180)
        try:
            connection.request('POST',path,json.dumps(body).encode(),{'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            response=connection.getresponse();raw=response.read(2*1024*1024)
            if response.status not in {200,201}:raise ValueError('checkpoint_audit_api_failed')
            return json.loads(raw)
        finally:connection.close()
    api('/v1/sessions',{'session_id':session,'objective':'Read-only SIIM complete-fold checkpoint hashes; no fitting or private answer access.',
        'permission_level':'observe','workspace_root':str(BASE/'data/acceptance/siim-calibration-20260908/operator'),
        'metadata':{'managed_hpc_identity':binding,'tenant_id':policy['tenant_id'],'owner_principal_id':policy['owner_principal_id'],'run_allowed_tool_names':['hpc_verify']}})
    verification=api('/v1/sessions/'+session+'/tools',{'tool_name':'hpc_verify','arguments':{},'idempotency_key':'fresh-checkpoint-audit-identity'})
    result=verification.get('result') or {};evidence=result.get('content') or {}
    from xsci.terminal_tools import hpc_identity_evidence_complete
    if not result.get('ok') or not hpc_identity_evidence_complete(evidence,expected_profile=binding['credential_profile'],expected_job_id=binding['job_id']):raise ValueError('checkpoint_identity_failed')
    from evomind_runtime.tools import _load_bound_hpc_config
    from research_agent_workstation.server.core.gpu_credentials import connect_ssh
    client=connect_ssh(_load_bound_hpc_config(binding['credential_profile'],binding['job_id'],session),timeout=30)
    source=(STAGE/'siim_checkpoint_contract.py').read_text(encoding='utf-8')
    program=source+'\ncheckpoint=inventory_complete_folds(Path('+repr(root)+'),'+repr(expected)+',Path('+repr(policy['artifact_root'])+'))\n'
    program+='import pandas as pd\ndata=Path('+repr(policy['persistent_data_root'])+')\n'
    program+='if hash_file(data/"manifest.json")!='+repr(policy['data_manifest_sha256'])+':raise ValueError("frozen_data_manifest_changed")\n'
    program+='data_manifest=json.loads((data/"manifest.json").read_text())\n'
    program+='for name in ["train.csv","frozen-folds.csv"]:\n if hash_file(data/name)!=data_manifest["files"][name]["sha256"]:raise ValueError("frozen_data_changed")\n'
    program+='verification,_=reconstruct_partial_oof(checkpoint,'+repr(expected)+',Path('+repr(root)+'),pd.read_csv(data/"train.csv"),pd.read_csv(data/"frozen-folds.csv"))\n'
    program+='checkpoint["partial_oof_verification"]=verification\nprint(json.dumps(checkpoint))\n'
    try:
        _,stdout,stderr=client.exec_command('python3 -c '+shlex.quote(program),timeout=90)
        raw=stdout.read(131073);code=stdout.channel.recv_exit_status()
        if code or len(raw)>131072:raise ValueError('checkpoint_inventory_failed')
        manifest=json.loads(raw)
    finally:client.close()
    receipt={'schema':'evomind.siim_checkpoint_audit_receipt.v1','at_epoch':time.time(),'run_id':run,'attempt_id':attempt,
             'checkpoint_manifest':manifest,'hpc_identity':evidence,'audit_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'models_transferred':False,'new_training_started':False}
    output=STAGE/'service-output'/('checkpoint-audit-'+str(time.time_ns())+'.json')
    with output.open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({'status':'checkpoint_inventory_verified','complete_folds':len(manifest['complete_folds']),'output':str(output)}))

if __name__=='__main__':main()
