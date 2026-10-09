"""Service-owner infrastructure operation using the existing managed data transport."""
from __future__ import annotations
import hashlib,http.client,json,os,sqlite3,sys
from datetime import datetime,timezone
from pathlib import Path

BASE=Path('C:/ProgramData/EvoMind');STAGE=BASE/'staging/ev-public-calibration-20260908'
sys.path[:0]=[str(BASE/'bundle/runtime'),'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src']
SESSION='session_ev_calibration_data_provision_20260908'


def main():
    result_path=STAGE/'service-output/data-provision.json'
    if result_path.exists():raise ValueError('Prior provisioning requires reconciliation')
    token=(BASE/'data/workspace/runtime/runtime.token').read_text().strip()
    cfg=json.loads((BASE/'config/node-config.json').read_text(encoding='utf-8-sig'))
    def request(method,path,body=None):
        c=http.client.HTTPConnection('127.0.0.1',int(cfg['network']['runtime_port']),timeout=180)
        try:
            c.request(method,path,body=json.dumps(body).encode() if body is not None else None,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            r=c.getresponse();raw=r.read(2*1024*1024)
            if r.status not in {200,201}:raise ValueError('Managed API rejected')
            return json.loads(raw)
        finally:c.close()
    with sqlite3.connect((BASE/'data/workspace/runtime/runtime.sqlite3').as_uri()+'?mode=ro',uri=True) as c:
        c.execute('PRAGMA query_only=ON')
        meta=json.loads(c.execute('SELECT metadata_json FROM sessions WHERE id=?',('run_7b210f24e219409b868edee14d4421d5',)).fetchone()[0])
        if c.execute('SELECT 1 FROM sessions WHERE id=?',(SESSION,)).fetchone():raise ValueError('Session exists')
    identity=meta['managed_hpc_identity']
    if identity['job_id']!=93207 or identity['allocation_generation']!=27:raise ValueError('Allocation mismatch')
    work=BASE/'data/acceptance/ev-calibration-20260908/data-provision'
    request('POST','/v1/sessions',{'session_id':SESSION,'objective':'User-authorized official EV data preparation only; no model fitting.',
                                'permission_level':'workspace-write','workspace_root':str(work),
                                'metadata':{'managed_hpc_identity':identity,'tenant_id':identity['tenant_id'],
                                            'owner_principal_id':identity['owner_principal_id'],'run_allowed_tool_names':['hpc_verify'],
                                            'operator_assisted_infrastructure':True}})
    result=request('POST','/v1/sessions/'+SESSION+'/tools',{'tool_name':'hpc_verify','arguments':{},'idempotency_key':'fresh-identity-before-data'})
    evidence=(result.get('result') or {}).get('content') or {}
    from xsci.terminal_tools import hpc_identity_evidence_complete
    if not hpc_identity_evidence_complete(evidence,expected_profile=identity['credential_profile'],expected_job_id=93207):raise ValueError('Fresh identity failed')
    from evomind_runtime.tools import _load_bound_hpc_config
    from evomind_runtime.hpc_runtime_overlay import HpcRuntime
    from research_agent_workstation.server.core.gpu_credentials import connect_ssh
    bound=_load_bound_hpc_config(identity['credential_profile'],93207,SESSION)
    adapter=(STAGE/'ev_hpc_data_adapter.py').read_bytes()
    script=work/'ev-data-adapter.sh';script.parent.mkdir(parents=True,exist_ok=True)
    with script.open('xb') as f:f.write(b"#!/bin/bash\nset -eu\npython3 - <<'EV_DATA_PY'\n"+adapter+b"\nEV_DATA_PY\n")
    runtime=HpcRuntime(run_id=SESSION,local_run_dir=work/'hpc',timeout_seconds=1500,connector=lambda:connect_ssh(bound,timeout=30))
    key=os.environ.pop('EVOMIND_EV_KAGGLE_KEY').encode()
    try:
        prepared=runtime.execute_managed_data_task(task_id='ev-data-provision',receipt_competition='playground-series-s6e9',script_path=script,
            persistent_data_root='/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/competition_data/ev_public_calibration_20260908',
            secret_files={'EVOMIND_SECRET_KAGGLE_API_FILE':key},timeout_seconds=1500)
    finally:key=None
    client=connect_ssh(bound,timeout=30)
    try:
        sftp=client.open_sftp()
        try:secret_clean=len(sftp.listdir(prepared['remote_task_root']+'/.secrets'))==0
        finally:sftp.close()
    finally:client.close()
    receipt={'schema':'evomind.ev_data_provision.operator.v1','at':datetime.now(timezone.utc).isoformat(),
             'status':'completed' if prepared['status']=='completed' and secret_clean else 'failed',
             'session_id':SESSION,'managed_data_result':prepared,'hpc_identity':evidence,
             'secret_cleanup_confirmed':secret_clean,'adapter_sha256':hashlib.sha256(adapter).hexdigest(),
             'operator_assisted_data_preparation':True,'training_started':False,'gpu_training_hours':0}
    with result_path.open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({'status':receipt['status'],'secret_cleanup_confirmed':secret_clean,'receipt':prepared.get('receipt'),'training_started':False}))
    return 0 if receipt['status']=='completed' else 2


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:
        (STAGE/'service-output/data-provision-error.json').write_text(json.dumps({'status':'failed','error_type':type(e).__name__}),encoding='utf-8')
        raise SystemExit(2)
