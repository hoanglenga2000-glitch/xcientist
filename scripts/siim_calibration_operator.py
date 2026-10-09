"""Bounded SIIM preparation/reconciliation through the current managed identity."""
import argparse,hashlib,http.client,json,shlex,sqlite3,sys,time
from pathlib import Path
BASE=Path('C:/ProgramData/EvoMind');STAGE=BASE/'staging/siim-mlebench-calibration-20260908'
sys.path[:0]=[str(BASE/'bundle/runtime'),'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src']

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=['reconcile','prepare','isolation','dependency','cancel-stalled','cancel-audit'],required=True);p.add_argument('--operation-id',required=True);args=p.parse_args()
 if not args.operation_id.replace('-','').isalnum():raise ValueError('invalid_operation_id')
 out=STAGE/'service-output'/('operator-'+args.operation_id+'.json')
 if out.exists():raise ValueError('operation_exists')
 policy=json.loads((BASE/'config/official-calibration/ev-public-20260908.json').read_text())
 binding=policy['managed_hpc_identity'];session='session_siim_operator_'+args.operation_id
 token=(BASE/'data/workspace/runtime/runtime.token').read_text().strip()
 def api(path,body):
  conn=http.client.HTTPConnection('127.0.0.1',8765,timeout=180)
  try:
   conn.request('POST',path,json.dumps(body).encode(),{'Authorization':'Bearer '+token,'Content-Type':'application/json'})
   r=conn.getresponse();raw=r.read(2*1024*1024)
   if r.status not in {200,201}:raise ValueError('operator_api_failed')
   return json.loads(raw)
  finally:conn.close()
 api('/v1/sessions',{'session_id':session,'objective':'SIIM '+args.action+'; no model training, no private grading.',
  'permission_level':'observe' if args.action=='reconcile' else 'workspace-write',
  'workspace_root':str(BASE/'data/acceptance/siim-calibration-20260908/operator'),
  'metadata':{'managed_hpc_identity':binding,'tenant_id':policy['tenant_id'],'owner_principal_id':policy['owner_principal_id'],'run_allowed_tool_names':['hpc_verify']}})
 v=api('/v1/sessions/'+session+'/tools',{'tool_name':'hpc_verify','arguments':{},'idempotency_key':'fresh-operator-identity'})
 evidence=(v.get('result') or {}).get('content') or {}
 from xsci.terminal_tools import hpc_identity_evidence_complete
 if not (v.get('result') or {}).get('ok') or not hpc_identity_evidence_complete(evidence,expected_profile=binding['credential_profile'],expected_job_id=binding['job_id']):raise ValueError('operator_hpc_identity_failed')
 from evomind_runtime.tools import _load_bound_hpc_config
 from research_agent_workstation.server.core.gpu_credentials import connect_ssh
 client=connect_ssh(_load_bound_hpc_config(binding['credential_profile'],binding['job_id'],session),timeout=30)
 worker=STAGE/{'reconcile':'siim_calibration_reconcile_worker.py','prepare':'siim_calibration_prepare_data.py','isolation':'siim_calibration_isolation_probe.py','dependency':'siim_dependency_reconcile.py','cancel-stalled':'siim_cancel_stalled_worker.py','cancel-audit':'siim_cancel_group_audit.py'}[args.action]
 source=worker.read_bytes();limit=60 if args.action=='reconcile' else 1800
 if args.action=='isolation':source=(STAGE/'siim_worker_isolation.py').read_bytes()+b'\n'+source
 if args.action=='cancel-stalled':
  wrapped='try:\n'+''.join(' '+line+'\n' for line in source.decode().splitlines())
  wrapped+='except Exception as e:\n import json\n print(json.dumps({"status":"failed","error_type":type(e).__name__,"code":str(e) if isinstance(e,ValueError) else "cancel_failed","signals_sent":0}))\n'
  source=wrapped.encode()
 started=time.time()
 try:
  _,stdout,stderr=client.exec_command('python3 -c '+shlex.quote(source.decode()),timeout=limit)
  raw=stdout.read(512*1024+1);err=stderr.read(65537);code=stdout.channel.recv_exit_status()
  if len(raw)>512*1024:raise ValueError('response_size_exceeded')
  try:result=json.loads(raw)
  except ValueError:result={'status':'failed','detail':'invalid_worker_response'}
  receipt={'schema':'evomind.siim_operator_receipt.v1','action':args.action,'session_id':session,'started':started,'ended':time.time(),
    'exit_code':code,'worker_sha256':hashlib.sha256(source).hexdigest(),'stderr_sha256':hashlib.sha256(err).hexdigest(),
    'result':result,'hpc_identity':evidence,'training_started':False}
 finally:client.close()
 with out.open('x') as f:json.dump(receipt,f,indent=2)
 print(json.dumps({'action':args.action,'exit_code':code,'result':result,'receipt':str(out)}))

if __name__=='__main__':main()
