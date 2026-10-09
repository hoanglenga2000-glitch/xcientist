"""Read-only exact-attempt fit progress, using a fresh managed identity proof."""
import argparse,hashlib,http.client,json,shlex,sqlite3,sys,time
from pathlib import Path
from datetime import datetime,timezone
BASE=Path('C:/ProgramData/EvoMind');STAGE=BASE/'staging/siim-mlebench-calibration-20260908/runtime-extension'
sys.path[:0]=[str(BASE/'bundle/runtime'),'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--settled-case');args=parser.parse_args()
    policy=json.loads((BASE/'config/official-calibration/siim-mlebench-20260908.json').read_text())
    with sqlite3.connect(Path(policy['budget_path']).as_uri()+'?mode=ro',uri=True) as c:
        c.execute('PRAGMA query_only=ON')
        if args.settled_case:
            if args.settled_case not in policy['cases']:raise ValueError('Requested case is not registered')
            if c.execute("SELECT 1 FROM attempts WHERE status IN ('reserved','uncertain','exceeded') LIMIT 1").fetchone():raise ValueError('All executions must be settled for the terminal audit')
            rows=c.execute("SELECT id,run_id FROM attempts WHERE run_id=? AND status IN ('failed','completed') ORDER BY rowid DESC LIMIT 1",(args.settled_case,)).fetchall()
        else:rows=c.execute("SELECT id,run_id FROM attempts WHERE status='reserved'").fetchall()
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
    observer='session_siim_fit_observer_'+str(time.time_ns())
    binding=policy['managed_hpc_identity']
    api('POST','/v1/sessions',{'session_id':observer,'objective':'Read-only SIIM attempt checkpoint progress; no training or mutation.',
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
        program="""import json,subprocess,os,re
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
progress=p/'fit-progress.json'
if progress.is_file():r['progress']=json.loads(progress.read_text())
opt=p/'optimizer-progress.json'
if opt.is_file():r['optimizer_progress']=json.loads(opt.read_text())
r['final_metrics_present']=(p/'metrics.json').is_file()
r['reload_checks']=[]
final=p/'final.joblib'
final_mtime=final.stat().st_mtime if final.is_file() else None
for key in ['fold-0','fold-1','fold-2','fold-3','fold-4','final']:
 check=p/(key+'-reload.json')
 if check.is_file() and not check.is_symlink() and check.stat().st_size<8192:
  try:
   value=json.loads(check.read_text())
   row={k:value.get(k) for k in ['status','model','max_abs_diff','auc_difference','rows','pid']}
   row.update(mtime=check.stat().st_mtime,after_final_checkpoint=final_mtime is not None and check.stat().st_mtime>=final_mtime)
   r['reload_checks'].append(row)
  except (OSError,ValueError):pass
r['post_fit_reload_passed']=sum(x['status']=='passed' and x['after_final_checkpoint'] and isinstance(x['max_abs_diff'],(int,float)) and x['max_abs_diff']<=1e-6 for x in r['reload_checks'])
r['diagnostic_log_summary']={}
for name in ['candidate-output.log','operator-error.txt','reload-output.log','runtime-stack.txt']:
 log=p/name
 if log.is_file():
  with log.open('rb') as f:
   f.seek(max(0,log.stat().st_size-8192));tail=f.read().decode('utf-8','replace')
  r['diagnostic_log_summary'][name]={'bytes':log.stat().st_size,'permission_error':'PermissionError' in tail,'file_not_found':'FileNotFoundError' in tail,'cuda_error':'CUDA error' in tail,'out_of_memory':'out of memory' in tail,'traceback':'Traceback' in tail,'in_cuda_init':'_lazy_init' in tail,'in_loader':'dataloader.py' in tail,'in_futex':'threading.py' in tail}
  if name=='runtime-stack.txt':r['diagnostic_log_summary'][name]['functions']=re.findall(r'line [0-9]+ in ([A-Za-z_][A-Za-z_0-9]*)',tail)[-24:]
q=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
r['gpu_summary']=q.stdout.strip() if q.returncode==0 else None
r['worker_processes']=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or proc.name==str(os.getpid()):continue
 try:
  cmd=(proc/'cmdline').read_bytes()
  argv=[arg for arg in cmd.split(bytes([0])) if arg]
  if not any(arg.endswith(b'/train_gpu.py') and RUN_VALUE.encode() in arg for arg in argv):continue
  parts=(proc/'stat').read_text().split(') ',1)[1].split()
  r['worker_processes'].append({'pid':int(proc.name),'state':parts[0],'cpu_ticks':int(parts[11])+int(parts[12]),'wait':(proc/'wchan').read_text().strip()})
 except (OSError,ValueError):pass
python='/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/siim_job90353/runtime/abebff4efcc16309f4b48e0a5032b39cab9814193da3b67f61f608fc7814c677/venv/bin/python'
q=subprocess.run([python,'-c','import sys,json;print(json.dumps({"prefix":sys.prefix,"executable":sys.executable}))'],capture_output=True,text=True,timeout=10)
if q.returncode==0:r['runtime_path_check']=json.loads(q.stdout)
print(json.dumps(r))
""".replace('ROOT_VALUE',repr(root)).replace('RUN_VALUE',repr(run_id))
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
