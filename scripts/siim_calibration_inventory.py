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
    binding=policy['managed_hpc_identity'];session='session_siim_calibration_inventory_'+str(time.time_ns())
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
    program=r'''import csv,hashlib,importlib.metadata,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra');competition='siim-isic-melanoma-classification'
result={'python':sys.version.split()[0],'versions':{},'data':[],'source':{},'caches':{},'read_only':True}
for name in ['torch','torchvision','timm','numpy','pandas','scikit-learn','Pillow','tensorflow','kaggle','aide','omegaconf','openai']:
 try:result['versions'][name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:result['versions'][name]=None
for data in [root/'mlebench_official_data'/competition,root/'mlebench_data'/competition,root/'competition_data'/competition]:
 row={'root':str(data),'exists':data.is_dir(),'files':{},'images':{}}
 if data.is_dir():
  for rel in ['prepared/public/train.csv','prepared/public/test.csv','prepared/public/sample_submission.csv','prepared/private/test.csv','raw/train.csv','raw/tfrecords/train00-2071.tfrec','raw/tfrecords/train06-2071.tfrec']:
   p=data/rel
   if p.is_file():
    item={'bytes':p.stat().st_size}
    if rel.endswith('.csv') and '/private/' not in rel and p.stat().st_size<32*1024*1024:
     item['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
     with p.open() as f:
      reader=csv.DictReader(f);item['columns']=reader.fieldnames;item['rows']=sum(1 for _ in reader)
    row['files'][rel]=item
  for rel in ['prepared/public/jpeg/train','prepared/public/jpeg/test']:
   p=data/rel
   if p.is_dir():row['images'][rel]=sum(x.is_file() and x.suffix=='.jpg' for x in p.iterdir())
 result['data'].append(row)
for name in ['mle-bench','AIDE','aide','aideml']:
 p=root/name
 entry={'exists':p.is_dir()}
 if (p/'.git').exists():
  q=subprocess.run(['git','-C',str(p),'rev-parse','HEAD'],capture_output=True,text=True,timeout=10)
  if q.returncode==0:entry['commit']=q.stdout.strip()
 if name=='mle-bench' and p.is_dir():
  files={}
  for rel in ['mlebench/grade.py','mlebench/grade_helpers.py','mlebench/competitions/'+competition+'/config.yaml','mlebench/competitions/'+competition+'/prepare.py','mlebench/competitions/'+competition+'/grade.py']:
   f=p/rel
   if f.is_file():files[rel]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
  entry['files']=files
 result['source'][name]=entry
for rel in ['mlebench_model_cache/torch/hub/checkpoints','mlebench_model_cache/huggingface','mlebench_model_cache','siim_job90353/runtime']:
 p=root/rel
 if p.is_dir():result['caches'][rel]=[x.name for x in sorted(p.iterdir())][:60]
disk=shutil.disk_usage(root);result['free_disk_bytes']=disk.free
print(json.dumps(result))
'''
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
    output=STAGE/'service-output/inventory.json'
    with output.open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({'status':'inventory_complete','inventory':inventory,'remaining_gpu_slot_hours':receipt['remaining_total_slot_seconds']/3600}))

if __name__=='__main__':
    try:main()
    except Exception as e:
        with (STAGE/'service-output/inventory-error.json').open('w') as f:json.dump({'status':'blocked','error_type':type(e).__name__,'detail':str(e) if isinstance(e,ValueError) else 'inventory_failed'},f)
        raise SystemExit(2)
