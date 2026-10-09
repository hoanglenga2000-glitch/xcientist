"""Install only the registered EV study extension; preserve existing studies and ledgers."""
import argparse,json,os,shutil,sqlite3,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('C:/ProgramData/EvoMind');STAGE=ROOT/'staging/ev-public-calibration-20260908/runtime-extension'
BACKUP=ROOT/'backups/ev-calibration-runtime-20260908'
sys.path.insert(0,str(ROOT/'staging/ev-deepseek-v4-pro-20260908'))
import apply_ev_deepseek_route as ops
ops.BACKUP=BACKUP
sha,require=ops.sha,ops.require


def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');args=p.parse_args()
    mpath=STAGE/'manifest.json'
    require(sha(mpath)=='7d890001563c2496b4dc49f713b0e9b32565f9dca7c3fe136296dcf2902906c5','payload_manifest_changed')
    m=json.loads(mpath.read_text())
    for name,h in m['files'].items():require(sha(STAGE/name)==h,'payload_changed')
    tools=ROOT/'bundle/runtime/evomind_runtime/tools.py';seal_path=ROOT/'state/bundle-integrity.json'
    require(sha(tools)==m['baseline_tools_sha256'],'active_tools_changed')
    require(sha(tools.with_name('runtime.py'))==m['baseline_runtime_sha256'],'active_runtime_changed')
    require(sha(seal_path)==m['baseline_seal_sha256'],'active_seal_changed')
    for name in ['ev_calibration_control.py','ev_calibration_runner.py']:
        require(not (tools.parent/name).exists(),'extension_already_exists')
    cfg_path=ROOT/'config/node-config.json';cfg=json.loads(cfg_path.read_text(encoding='utf-8-sig'))
    require(cfg['hpc']['state']!='active','global_hpc_startup_requires_review')
    seal=json.loads(seal_path.read_text(encoding='utf-8-sig'))
    for row in seal['files']:require(sha(ROOT/'bundle'/row['path'])==row['sha256'],'existing_bundle_changed')
    policy_path=ROOT/'config/official-calibration/ev-public-20260908.json'
    require(not policy_path.exists(),'policy_already_registered')
    template=json.loads((STAGE/'policy-template.json').read_text())
    db=ROOT/'data/workspace/runtime/runtime.sqlite3'
    with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c:
        c.execute('PRAGMA query_only=ON')
        reference=json.loads(c.execute('SELECT metadata_json FROM sessions WHERE id=?',(template['reference_identity_run'],)).fetchone()[0])
        require(not any(c.execute('SELECT 1 FROM sessions WHERE id=?',(run,)).fetchone() for run in template['cases']),'case_already_exists')
    identity=reference['managed_hpc_identity']
    require(identity['job_id']==93207 and identity['allocation_generation']==27,'allocation_changed')
    policy={**template,'managed_hpc_identity':identity,'tenant_id':identity['tenant_id'],'owner_principal_id':identity['owner_principal_id']}
    before=ops.work_guard()
    if not args.apply:
        print(json.dumps({'status':'ready_for_scoped_extension','cases':6,'old_study_unchanged':True,'work':before}));return
    require(not BACKUP.exists(),'previous_extension_requires_reconciliation')
    with ops.deployment_lock():
        require(ops.work_guard()==before,'work_changed')
        BACKUP.mkdir(parents=True)
        subprocess.run(['icacls.exe',str(BACKUP),'/inheritance:r','/grant:r','SYSTEM:(OI)(CI)F','Administrators:(OI)(CI)F'],check=True,stdout=subprocess.DEVNULL)
        shutil.copy2(tools,BACKUP/'tools.py');shutil.copy2(tools.with_name('runtime.py'),BACKUP/'runtime.py');shutil.copy2(seal_path,BACKUP/'seal.json')
        unchanged={str(p.relative_to(ROOT)):sha(p) for p in [cfg_path,ROOT/'bundle/scripts/Start-Node.ps1',ROOT/'bundle/runtime/evomind_runtime/store.py',ROOT/'bundle/runtime/evomind_runtime/model_transport.py',ROOT/'config/research-control/policy.json']}
        ops.write(BACKUP/'before.json',{'unchanged':unchanged,'work':before})
        changed=False
        try:
            ops.service('Stop');require(ops.work_guard()==before,'work_changed_during_stop')
            for name in ['tools.py','runtime.py','ev_calibration_control.py','ev_calibration_runner.py']:
                target=tools.parent/name;temp=target.with_name(name+'.ev-new')
                require(not temp.exists(),'temporary_collision');shutil.copyfile(STAGE/name,temp)
                subprocess.run(['icacls.exe',str(temp),'/inheritance:r','/grant:r','SYSTEM:F','Administrators:F','EvoMindSvc:R'],check=True,stdout=subprocess.DEVNULL)
                os.replace(temp,target);changed=True
            policy_path.parent.mkdir(exist_ok=True)
            ops.write(policy_path,policy)
            subprocess.run(['icacls.exe',str(policy_path),'/inheritance:r','/grant:r','SYSTEM:F','Administrators:F','EvoMindSvc:R'],check=True,stdout=subprocess.DEVNULL)
            by_path={r['path']:r for r in seal['files']}
            for name in ['tools.py','runtime.py','ev_calibration_control.py','ev_calibration_runner.py']:
                path='runtime/evomind_runtime/'+name;target=ROOT/'bundle'/path
                by_path[path]={'path':path,'size':target.stat().st_size,'sha256':sha(target)}
            for rel,h in unchanged.items():require(sha(ROOT/rel)==h,'unrelated_configuration_changed')
            seal.update(files=sorted(by_path.values(),key=lambda r:r['path'].casefold()),file_count=len(by_path),sealed_at_utc=datetime.now(timezone.utc).isoformat())
            temp=seal_path.with_name('bundle-integrity.ev-new.json');ops.write(temp,seal);os.replace(temp,seal_path)
            ops.service('Start');require(ops.work_guard()==before,'historical_work_changed')
            result={'status':'ev_calibration_extension_activated','cases':6,'policy_sha256':sha(policy_path),'old_study_and_budget_unchanged':True,
                    'hpc_binding_unchanged':True,'model':'deepseek-v4-pro','training_started':False,'backup':str(BACKUP)}
            ops.write(BACKUP/'result.json',result);print(json.dumps(result),flush=True)
        except Exception as e:
            rollback='not_needed'
            if changed:
                try:
                    ops.service('Stop');shutil.copy2(BACKUP/'tools.py',tools);shutil.copy2(BACKUP/'runtime.py',tools.with_name('runtime.py'));shutil.copy2(BACKUP/'seal.json',seal_path)
                    for name in ['ev_calibration_control.py','ev_calibration_runner.py']:
                        path=tools.parent/name
                        if path.exists():path.rename(BACKUP/('failed-'+name))
                    if policy_path.exists():policy_path.rename(BACKUP/'failed-policy.json')
                    ops.service('Start');rollback='prior_runtime_restored'
                except Exception:rollback='manual_reconciliation_required'
            ops.write(BACKUP/'failure.json',{'error_type':type(e).__name__,'code':str(e)[:100],'rollback':rollback})
            raise


if __name__=='__main__':main()
