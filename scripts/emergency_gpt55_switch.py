"""Explicit emergency restoration; UI/history/credentials stay unchanged.

Only --apply mutates production. Payload identity, no current managed work,
backups, lock, exact seals and process-stop proof precede replacement.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import ctypes
import hashlib
import http.client
import json
import os
import re
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
import zipfile

ROOT=Path('C:/ProgramData/EvoMind')
OLD_RUN='run_0c57bbde44e94a988843c2f6981f3c65'
LEGACY={'call_b4c7d7d8095a451a5d0d5878f6671620','call_ad79487f22e6e88e8acc01bc8cdb9eb9',
        'call_5c0891e12bbd853969ef3aae01accb80','call_d2f39b4909fe4403aa821d191c649721'}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,code):
    if not value: raise ValueError(code)
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream: json.dump(value,stream,ensure_ascii=True,indent=2)
def files(root):
    return {p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}

def current_work():
    db=ROOT/'data/workspace/runtime/runtime.sqlite3'
    with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        rows=[dict(r) for r in c.execute("SELECT id,status FROM assistant_runs WHERE status IN ('queued','planning','running','verifying','recovering','pausing')")]
        require(all(r['id']==OLD_RUN and r['status']=='recovering' for r in rows),'current_run_prevents_restart')
        running=[dict(r) for r in c.execute("SELECT id,session_id FROM tool_calls WHERE status='running'")]
        require({r['id'] for r in running}<=LEGACY,'current_tool_prevents_restart')
        require(c.execute("SELECT count(*) FROM tool_calls WHERE session_id=? AND status NOT IN ('completed','failed')",(OLD_RUN,)).fetchone()[0]==0,'target_work_unsettled')
        require(c.execute("SELECT count(*) FROM approvals WHERE status='approved' AND tool_call_id IN (SELECT id FROM tool_calls WHERE status='waiting_approval')").fetchone()[0]==0,'approved_work_prevents_restart')
    with sqlite3.connect((ROOT/'data/workspace/runtime/gpu_budget.sqlite3').as_uri()+'?mode=ro',uri=True) as c:
        rows=c.execute('SELECT status,charged FROM gpu_operations').fetchall()
        require(all(r[0] in {'passed','failed'} for r in rows),'gpu_work_prevents_restart')
    return {'legacy_unknown_rows_retained':len(running),'gpu_operations':len(rows),'charged_seconds':sum(r[1] for r in rows)}

@contextmanager
def mutex():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMutexW.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    kernel.ReleaseMutex.argtypes=kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.CreateMutexW(None,False,'Global\\EvoMind-Byoa-V12-Deployment')
    require(handle,'mutex_unavailable')
    answer=kernel.WaitForSingleObject(handle,0)
    try:
        require(answer==0,'mutex_busy_or_abandoned')
        yield
    finally:
        if answer in {0,0x80}: kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)

def service(action,backup):
    p=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
        'C:/SecureInput/Invoke-ServiceAccountAction.ps1','-Action',action,'-TimeoutMinutes','12'],
        capture_output=True,timeout=900,creationflags=subprocess.CREATE_NO_WINDOW)
    write(backup/(action.lower()+'-'+str(time.time_ns())+'.json'),{'exit_code':p.returncode,'output_sha256':hashlib.sha256(p.stdout+p.stderr).hexdigest()})
    if p.returncode:
        raw=(ROOT/'logs/cloud-node-service-action.err.log').read_text(encoding='utf-8-sig',errors='replace')
        codes=sorted(set(re.findall(r'\b[A-Z][A-Z0-9_]*(?:FAILED|MISSING|REJECTED|INVALID|MISMATCH|TIMEOUT|REQUIRED|UNSAFE)\b',raw)))
        safe_lines=[line for line in raw.splitlines() if not re.search(r'(?i)password|passwd|secret|token|credential|api.?key|authorization|bearer|密码|口令|凭据|登录|账号|账户|sk-',line)]
        write(backup/'startup-error-redacted.json',{'codes':codes,'redacted_lines':safe_lines[-30:]})
        print(json.dumps({'phase':'service_failed','codes':codes,'diagnostic':str(backup/'startup-error-redacted.json')}),flush=True)
    require(p.returncode==0,'service_action_failed')
    value=json.loads(p.stdout.decode('utf-8-sig'))
    require(value.get('status')=='completed' and value.get('action')==action,'service_action_unconfirmed')

def stopped():
    import psutil
    require(not [r for r in psutil.net_connections('tcp') if r.status=='LISTEN' and r.laddr.port in {8088,8765,65068}], 'listeners_still_active')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    require(sha(args.manifest)==args.sha256,'manifest_changed')
    m=json.loads(args.manifest.read_text())
    stage=args.manifest.resolve().parent
    stage.relative_to(ROOT/'staging')
    for name,digest in m['payloads'].items():
        require('/' not in name and '\\' not in name and sha(stage/name)==digest,'payload_changed')
    active=ROOT/'bundle/runtime/evomind_runtime'
    baseline=m['baseline_runtime']
    actual=files(active)
    require(set(actual)==set(baseline),'active_runtime_file_set_changed')
    require(all(hashlib.sha256(p.read_text(encoding='utf-8').replace('\r\n','\n').encode()).hexdigest()==baseline[n] for n,p in actual.items()),'active_runtime_source_changed')
    launcher=ROOT/'bundle/scripts/Start-Node.ps1'
    require(sha(launcher)==m['baseline_launcher_sha256'],'active_launcher_changed')
    cfg=ROOT/'config/node-config.json'
    config=json.loads(cfg.read_text(encoding='utf-8-sig'))
    require(config['hpc']['state']!='active','global_hpc_startup_requires_separate_review')
    before=current_work()
    if not args.apply:
        print(json.dumps({'status':'ready_for_emergency_switch','scope':'model_fallback_manual_recovery_old_ui','work':before}));return
    backup=ROOT/'backups/emergency-gpt55-20260908-v5'
    require(not backup.exists(),'prior_switch_requires_reconciliation')
    with mutex():
        require(current_work()==before,'work_changed')
        backup.mkdir(parents=True)
        subprocess.run(['icacls.exe',str(backup),'/inheritance:r','/grant:r','*S-1-5-18:(OI)(CI)F','*S-1-5-32-544:(OI)(CI)F'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        write(backup/'before.json',{'config_sha256':sha(cfg),'work':before,'manual_recovery':True,'web_root':config['web_runtime_root']})
        paths=['bundle/scripts/Start-Node.ps1','bundle/runtime/run_python_runtime.py','state/bundle-integrity.json']
        for relative in paths:
            target=ROOT/relative
            saved=backup/'files'/relative;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,saved)
        shutil.copytree(active,backup/'runtime-before',ignore=shutil.ignore_patterns('__pycache__'))
        stage_runtime=backup/'runtime-candidate';stage_runtime.mkdir()
        with zipfile.ZipFile(stage/'runtime.zip') as z:
            manifest=json.loads(z.read('runtime-hotfix-manifest.json'))
            require(set(z.namelist())=={r['path'] for r in manifest['files']}|{'runtime-hotfix-manifest.json'},'archive_file_set')
            for r in manifest['files']:
                name=r['path'];require(name.startswith('evomind_runtime/') and '..' not in name and '\\' not in name,'archive_path')
                data=z.read(name);require(len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],'archive_hash')
                target=stage_runtime/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            (stage_runtime/'runtime-hotfix-manifest.json').write_bytes(z.read('runtime-hotfix-manifest.json'))
        swapped=False
        try:
            print(json.dumps({'phase':'stopping_for_fallback'}),flush=True)
            service('Stop',backup);stopped()
            require(current_work()==before,'work_changed_during_stop')
            for path in (ROOT/'data').rglob('*'):
                if path.is_file() and path.suffix in {'.sqlite3','.sqlite','.db'}:
                    target=backup/'databases'/path.relative_to(ROOT/'data');target.parent.mkdir(parents=True,exist_ok=True)
                    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(target) as dest: source.backup(dest)
            active.rename(backup/'runtime-original')
            (stage_runtime/'evomind_runtime').rename(active);swapped=True
            shutil.copy2(stage/'Start-Node.ps1',launcher)
            shutil.copy2(stage/'entry.py',ROOT/'bundle/runtime/run_python_runtime.py')
            for folder in (active,launcher,ROOT/'bundle/runtime/run_python_runtime.py'):
                grant=':RX'
                owner_args=['takeown.exe','/F',str(folder),'/A']+(['/R','/D','Y'] if folder.is_dir() else [])
                subprocess.run(owner_args,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                subprocess.run(['icacls.exe',str(folder),'/inheritance:r','/grant:r',str(config['dedicated_user'])+grant,
                    '*S-1-5-18:F','*S-1-5-32-544:F','/T','/C','/Q'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                if folder.is_dir():
                    subprocess.run(['icacls.exe',str(folder),'/grant:r',str(config['dedicated_user'])+':(OI)(CI)RX',
                        '*S-1-5-18:(OI)(CI)F','*S-1-5-32-544:(OI)(CI)F','/Q'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            seal=json.loads((backup/'files/state/bundle-integrity.json').read_text(encoding='utf-8-sig'))
            existing={r['path']:r for r in seal['files']}
            expected=set(existing)
            expected={n for n in expected if not n.startswith('runtime/evomind_runtime/')}
            expected|={'runtime/evomind_runtime/'+n for n in files(active)}
            rows=[]
            for name in sorted(expected,key=str.casefold):
                path=ROOT/'bundle'/name
                if not (name.startswith('runtime/evomind_runtime/') or name in {'scripts/Start-Node.ps1','runtime/run_python_runtime.py','runtime/runtime-hotfix-manifest.json'}):
                    require(sha(path)==existing[name]['sha256'],'unrelated_bundle_file_changed')
                rows.append({'path':name,'size':path.stat().st_size,'sha256':sha(path)})
            seal.update(files=rows,file_count=len(rows),sealed_at_utc=datetime.now(timezone.utc).isoformat())
            temp=ROOT/'state/bundle-integrity.emergency.tmp'
            write(temp,seal);os.replace(temp,ROOT/'state/bundle-integrity.json')
            require(sha(cfg)==json.loads((backup/'before.json').read_text())['config_sha256'],'config_changed')
            print(json.dumps({'phase':'starting_gpt55_manual_recovery'}),flush=True)
            service('Start',backup)
            deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                try:
                    conn=http.client.HTTPConnection('127.0.0.1',8088,timeout=5);conn.request('GET','/api/healthz');response=conn.getresponse();health=json.loads(response.read());conn.close()
                    if response.status==200 and health.get('status')=='ready':break
                except (OSError,ValueError):pass
                time.sleep(1)
            else: raise ValueError('web_health_not_restored')
            require(current_work()==before,'historical_work_changed_after_restart')
            result={'status':'gpt55_activated_pending_live_acceptance','model':'gpt-5.5','web_build':health.get('build_id'),
                'automatic_historical_recovery':False,'historical_rows_changed':False,'research_databases_restored':False,'backup':str(backup)}
            write(backup/'result.json',result);print(json.dumps(result),flush=True)
        except Exception as error:
            try:
                service('Stop',backup);stopped()
                if swapped:
                    active.rename(backup/'failed-runtime');(backup/'runtime-original').rename(active)
                for relative in paths: shutil.copy2(backup/'files'/relative,ROOT/relative)
                rollback='code_restored_service_held_to_prevent_history_replay'
            except Exception: rollback='manual_reconciliation_required'
            write(backup/'failure.json',{'status':'HOLD','error_class':type(error).__name__,'rollback':rollback,'databases_restored':False})
            raise

if __name__=='__main__': main()
