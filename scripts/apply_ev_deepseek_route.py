"""Remote, transactional two-file model-route switch; no database or HPC mutation."""
from __future__ import annotations
import argparse,ctypes,hashlib,http.client,json,os,shutil,sqlite3,subprocess,time
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('C:/ProgramData/EvoMind')
STAGE=ROOT/'staging/ev-deepseek-v4-pro-20260908'
BACKUP=ROOT/'backups/ev-deepseek-v4-pro-20260908'
MANIFEST_SHA='3dbcda801e9f68f7641efe71e7c1c432bd7dd4a054305c7f73c3c1616ad469c0'
LEGACY_RUN='run_0c57bbde44e94a988843c2f6981f3c65'
LEGACY_CALLS={'call_b4c7d7d8095a451a5d0d5878f6671620','call_ad79487f22e6e88e8acc01bc8cdb9eb9',
              'call_5c0891e12bbd853969ef3aae01accb80','call_d2f39b4909fe4403aa821d191c649721'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(value,code):
    if not value:raise ValueError(code)


def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)


def work_guard():
    db=ROOT/'data/workspace/runtime/runtime.sqlite3'
    with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c:
        c.execute('PRAGMA query_only=ON')
        runs=c.execute("SELECT id,status FROM assistant_runs WHERE status IN ('queued','planning','running','verifying','recovering','pausing')").fetchall()
        require(all(r==(LEGACY_RUN,'recovering') for r in runs),'current_run_prevents_restart')
        calls=c.execute("SELECT id,session_id FROM tool_calls WHERE status='running'").fetchall()
        require(all(r[0] in LEGACY_CALLS for r in calls),'current_tool_prevents_restart')
        require(c.execute("SELECT count(*) FROM approvals WHERE status='approved' AND tool_call_id IN (SELECT id FROM tool_calls WHERE status='waiting_approval')").fetchone()[0]==0,'approved_work_prevents_restart')
    gpu=ROOT/'data/workspace/runtime/gpu_budget.sqlite3'
    with sqlite3.connect(gpu.as_uri()+'?mode=ro',uri=True) as c:
        c.execute('PRAGMA query_only=ON')
        rows=c.execute('SELECT status,charged FROM gpu_operations').fetchall()
        require(all(r[0] in {'passed','failed'} for r in rows),'gpu_work_prevents_restart')
    return {'preserved_legacy_runs':len(runs),'preserved_legacy_calls':len(calls),
            'gpu_operations':len(rows),'charged_seconds':sum(r[1] for r in rows)}


@contextmanager
def deployment_lock():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMutexW.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    kernel.ReleaseMutex.argtypes=kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.CreateMutexW(None,False,'Global\\EvoMind-Byoa-V12-Deployment')
    require(handle,'deployment_mutex_unavailable')
    acquired=kernel.WaitForSingleObject(handle,0)
    try:
        require(acquired==0,'deployment_mutex_busy')
        yield
    finally:
        if acquired in {0,0x80}:kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def service(action):
    print(json.dumps({'phase':'service_'+action.lower()}),flush=True)
    p=subprocess.run(['powershell.exe','-NoProfile','-File','C:/SecureInput/Invoke-ServiceAccountAction.ps1',
                      '-Action',action,'-TimeoutMinutes','10'],capture_output=True,timeout=720,
                     creationflags=subprocess.CREATE_NO_WINDOW)
    write(BACKUP/(action.lower()+'-'+str(time.time_ns())+'.json'),{'exit_code':p.returncode,
          'output_sha256':hashlib.sha256(p.stdout+p.stderr).hexdigest()})
    require(p.returncode==0,'service_'+action.lower()+'_failed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    mpath=STAGE/'switch-manifest.json'
    require(sha(mpath)==MANIFEST_SHA,'candidate_manifest_changed')
    m=json.loads(mpath.read_text())
    for name,digest in m['payloads'].items():require(sha(STAGE/name)==digest,'payload_changed')
    launcher=ROOT/'bundle/scripts/Start-Node.ps1';transport=ROOT/'bundle/runtime/evomind_runtime/model_transport.py'
    seal_path=ROOT/'state/bundle-integrity.json';config_path=ROOT/'config/node-config.json'
    config=json.loads(config_path.read_text(encoding='utf-8-sig'))
    require(config['hpc']['state']!='active','global_hpc_startup_requires_review')
    require(sha(launcher)==m['baseline_launcher_sha256'],'launcher_baseline_changed')
    require(sha(transport)==m['baseline_transport_sha256'],'transport_baseline_changed')
    require(sha(seal_path)==m['baseline_seal_sha256'],'bundle_seal_changed')
    route=ROOT/'state/ev-deepseek-route-20260908.json'
    require(not route.exists(),'route_already_installed')
    before=work_guard()
    seal=json.loads(seal_path.read_text(encoding='utf-8-sig'))
    for row in seal['files']:require(sha(ROOT/'bundle'/row['path'])==row['sha256'],'existing_bundle_integrity_failed')
    if not args.apply:
        print(json.dumps({'status':'ready_for_scoped_model_switch','model':m['model'],'work':before,'production_changed':False}));return
    key=Path(config['secrets_root'])/'pezayo_ev_deepseek_v4_pro.xml'
    require(key.is_file(),'service_model_credential_not_installed')
    require(not BACKUP.exists(),'previous_switch_requires_reconciliation')
    with deployment_lock():
        require(work_guard()==before,'work_changed')
        BACKUP.mkdir(parents=True)
        subprocess.run(['icacls.exe',str(BACKUP),'/inheritance:r','/grant:r','SYSTEM:(OI)(CI)F','Administrators:(OI)(CI)F'],check=True,stdout=subprocess.DEVNULL)
        preserved=['bundle/scripts/Start-Node.ps1','bundle/runtime/evomind_runtime/model_transport.py','state/bundle-integrity.json']
        for rel in preserved:
            saved=BACKUP/rel;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,saved)
        config_sha=sha(config_path)
        write(BACKUP/'before.json',{'config_sha256':config_sha,'work':before,'model_before':'gpt-5.5',
                                  'model_after':'deepseek-v4-pro','historical_recovery':False})
        changed=False
        try:
            service('Stop')
            require(work_guard()==before,'work_changed_during_stop')
            import psutil
            require(not [x for x in psutil.net_connections('tcp') if x.status=='LISTEN' and x.laddr.port in {8088,8765,65068}],'managed_listeners_not_stopped')
            for source,target in [('Start-Node.ps1',launcher),('model_transport.py',transport),('route.json',route)]:
                temporary=target.with_name(target.name+'.deepseek-new')
                require(not temporary.exists(),'temporary_path_collision')
                shutil.copyfile(STAGE/source,temporary)
                subprocess.run(['icacls.exe',str(temporary),'/inheritance:r','/grant:r','SYSTEM:F','Administrators:F','EvoMindSvc:R'],check=True,stdout=subprocess.DEVNULL)
                os.replace(temporary,target)
                changed=True
            changed_paths={'scripts/Start-Node.ps1','runtime/evomind_runtime/model_transport.py'}
            for row in seal['files']:
                p=ROOT/'bundle'/row['path']
                if row['path'] in changed_paths:row.update(size=p.stat().st_size,sha256=sha(p))
                else:require(sha(p)==row['sha256'],'unrelated_bundle_file_changed')
            seal['sealed_at_utc']=datetime.now(timezone.utc).isoformat()
            temporary=seal_path.with_name('bundle-integrity.deepseek-new.json')
            write(temporary,seal);os.replace(temporary,seal_path)
            require(sha(config_path)==config_sha,'node_config_changed')
            service('Start')
            require(work_guard()==before,'historical_work_changed')
            conn=http.client.HTTPConnection('127.0.0.1',8088,timeout=10)
            conn.request('GET','/api/healthz');response=conn.getresponse();health=json.loads(response.read());conn.close()
            require(response.status==200 and health.get('status')=='ready','web_health_not_ready')
            result={'status':'activated_pending_fresh_run_validation','model':'deepseek-v4-pro','base_url':'https://api.pezayo.com/v1',
                    'protocol':'chat_completions','backup':str(BACKUP),'hpc_binding_unchanged':True,'old_keys_retained':True,
                    'historical_runs_not_replayed':True,'work_before':before,'work_after':work_guard(),'config_sha256_unchanged':True}
            write(BACKUP/'result.json',result);print(json.dumps(result),flush=True)
        except Exception as error:
            rollback='not_needed'
            if changed:
                try:
                    service('Stop')
                    for rel in preserved:shutil.copy2(BACKUP/rel,ROOT/rel)
                    if route.exists():route.rename(BACKUP/'failed-route.json')
                    service('Start');rollback='prior_code_and_service_restored'
                except Exception:rollback='manual_reconciliation_required'
            write(BACKUP/'failure.json',{'error_type':type(error).__name__,'code':str(error)[:120],'rollback':rollback})
            raise


if __name__=='__main__':main()
