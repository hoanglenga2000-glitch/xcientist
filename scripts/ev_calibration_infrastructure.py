"""Run as the existing service owner: inspect only, through the managed HPC binding."""
from __future__ import annotations
import http.client,json,os,shlex,sqlite3,sys
from datetime import datetime,timezone
from pathlib import Path

BASE=Path('C:/ProgramData/EvoMind')
STAGE=BASE/'staging/ev-public-calibration-20260908'
RELEASE=Path('C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866')
sys.path[:0]=[str(BASE/'bundle/runtime'),str(RELEASE/'src')]
REFERENCE_RUN='run_7b210f24e219409b868edee14d4421d5'
SESSION='session_ev_calibration_infrastructure_20260908'


def main():
    output=STAGE/'service-output/infrastructure.json'
    if output.exists():raise ValueError('Existing infrastructure receipt requires reconciliation')
    config=json.loads((BASE/'config/node-config.json').read_text(encoding='utf-8-sig'))
    port=int(config['network']['runtime_port'])
    token=(BASE/'data/workspace/runtime/runtime.token').read_text().strip()
    def request(method,path,body=None):
        conn=http.client.HTTPConnection('127.0.0.1',port,timeout=180)
        try:
            conn.request(method,path,body=json.dumps(body).encode() if body is not None else None,
                         headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            response=conn.getresponse();raw=response.read(2*1024*1024)
            if response.status not in {200,201}:raise ValueError('managed_api_rejected')
            return json.loads(raw)
        finally:conn.close()
    with sqlite3.connect((BASE/'data/workspace/runtime/runtime.sqlite3').as_uri()+'?mode=ro',uri=True) as c:
        c.execute('PRAGMA query_only=ON')
        meta=json.loads(c.execute('SELECT metadata_json FROM sessions WHERE id=?',(REFERENCE_RUN,)).fetchone()[0])
        if c.execute('SELECT 1 FROM sessions WHERE id=?',(SESSION,)).fetchone():raise ValueError('Session already exists')
    identity=meta['managed_hpc_identity']
    if identity['job_id']!=93207 or identity['allocation_generation']!=27:raise ValueError('Unexpected allocation')
    body={'session_id':SESSION,'objective':'EV calibration infrastructure read-only inspection, no training.',
          'permission_level':'observe','workspace_root':str(BASE/'data/acceptance/ev-calibration-20260908/operator'),
          'metadata':{'managed_hpc_identity':identity,'tenant_id':identity['tenant_id'],
                      'owner_principal_id':identity['owner_principal_id'],'run_allowed_tool_names':['hpc_verify']}}
    request('POST','/v1/sessions',body)
    checked=request('POST','/v1/sessions/'+SESSION+'/tools',{'tool_name':'hpc_verify','arguments':{},'idempotency_key':'ev-calibration-infrastructure-identity-once'})
    result=checked.get('result') or {};evidence=result.get('content') or {}
    from xsci.terminal_tools import hpc_identity_evidence_complete
    if not result.get('ok') or not hpc_identity_evidence_complete(evidence,expected_profile=identity['credential_profile'],expected_job_id=93207):
        raise ValueError('Fresh managed identity required')
    from evomind_runtime.tools import _load_bound_hpc_config
    from research_agent_workstation.server.core.gpu_credentials import connect_ssh
    bound=_load_bound_hpc_config(identity['credential_profile'],93207,SESSION)
    client=connect_ssh(bound,timeout=30)
    try:
        program="""import importlib.util,importlib.metadata,json,os,platform,sys,ctypes.util
from pathlib import Path
names=['numpy','pandas','scikit-learn','catboost','lightgbm','xgboost','scipy','kaggle']
versions={}
for name in names:
 try: versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError: versions[name]=None
p=Path('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/competition_data/ev_public_calibration_20260908')
print(json.dumps({'python':sys.version.split()[0],'executable':sys.executable,'platform':platform.platform(),'machine':platform.machine(),'cpu_count':os.cpu_count(),'versions':versions,'seccomp_library':ctypes.util.find_library('seccomp'),'campaign_data_root_exists':p.exists(),'expected_data_manifest_exists':(p/'.evomind/data-manifest.json').is_file(),'read_only':True,'training_started':False}))
"""
        stdin,stdout,stderr=client.exec_command('python3 -c '+shlex.quote(program),timeout=30)
        raw=stdout.read(65536);status=stdout.channel.recv_exit_status()
        if status!=0:raise ValueError('Environment inspection failed')
        environment=json.loads(raw)
    finally:client.close()
    receipt={'schema':'evomind.ev_calibration.infrastructure.v1','checked_at':datetime.now(timezone.utc).isoformat(),
             'status':'read_only_inspection_complete','session_id':SESSION,'hpc_identity':evidence,
             'environment':environment,'hpc_data_writes':0,'training_started':False,'gpu_training_hours':0}
    with output.open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({'status':receipt['status'],'environment':environment,'identity_samples':evidence['samples_passed']}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        p=STAGE/'service-output/infrastructure-error.json'
        p.write_text(json.dumps({'status':'failed','error_type':type(exc).__name__,'code':str(exc) if str(exc) in {'Fresh managed identity required','Environment inspection failed','Unexpected allocation'} else 'infrastructure_inspection_failed'}),encoding='utf-8')
        raise SystemExit(2)
