"""Backed-up SIIM-only runtime activation; old budgets/allocations remain unchanged."""
import json,os,shutil,sqlite3,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
BASE=Path('C:/ProgramData/EvoMind');STAGE=BASE/'staging/siim-mlebench-calibration-20260908/runtime-extension'
BACKUP=BASE/'backups/siim-calibration-runtime-20260909'
sys.path.insert(0,str(BASE/'staging/ev-deepseek-v4-pro-20260908'))
import apply_ev_deepseek_route as ops
ops.BACKUP=BACKUP;sha,require=ops.sha,ops.require
NAMES=['tools.py','runtime.py','siim_calibration_control.py','siim_calibration_budget.py','siim_calibration_runner.py','siim_worker_isolation.py']

def ev_guard():
    with sqlite3.connect((BASE/'data/workspace/runtime/ev_calibration_budget.sqlite3').as_uri()+'?mode=ro',uri=True) as c:
        rows=c.execute('SELECT id,status,charged FROM attempts ORDER BY id').fetchall()
        require(all(r[1] in {'completed','failed'} for r in rows),'ev_execution_unsettled')
        return rows

def main():
    require(sha(STAGE/'manifest.json')=='153df8fe2571876713366c9c17b82dee29928f9ee29b575faee8ccd65f1f93fb','payload_manifest_changed')
    manifest=json.loads((STAGE/'manifest.json').read_text())
    for name,h in manifest['files'].items():require(sha(STAGE/name)==h,'payload_changed')
    parent=BASE/'bundle/runtime/evomind_runtime';seal_path=BASE/'state/bundle-integrity.json'
    for name,key in [('tools.py','baseline_tools_sha256'),('runtime.py','baseline_runtime_sha256')]:require(sha(parent/name)==manifest[key],'runtime_baseline_changed')
    require(sha(seal_path)==manifest['baseline_seal_sha256'],'seal_changed')
    for name in NAMES[2:]:require(not(parent/name).exists(),'siim_extension_already_exists')
    policy_path=BASE/'config/official-calibration/siim-mlebench-20260908.json'
    require(not policy_path.exists(),'siim_policy_already_exists')
    template=json.loads((STAGE/'policy-template.json').read_text())
    with sqlite3.connect((BASE/'data/workspace/runtime/runtime.sqlite3').as_uri()+'?mode=ro',uri=True) as c:
        c.execute('PRAGMA query_only=ON')
        reference=json.loads(c.execute('SELECT metadata_json FROM sessions WHERE id=?',(template['reference_identity_run'],)).fetchone()[0])
        require(not any(c.execute('SELECT 1 FROM sessions WHERE id=?',(r,)).fetchone() for r in template['cases']),'new_case_exists')
        old=json.loads(c.execute('SELECT metadata_json FROM sessions WHERE id=?',('run_da9c926f9d7c50ec99dd179db7e9a2f6',)).fetchone()[0])
        require(old.get('user_pause_requested') is True,'ev_not_paused')
    binding=reference['managed_hpc_identity'];require(binding['job_id']==93207 and binding['allocation_generation']==27,'hpc_binding_changed')
    policy={**template,'managed_hpc_identity':binding,'tenant_id':binding['tenant_id'],'owner_principal_id':binding['owner_principal_id']}
    before=ops.work_guard();ev=ev_guard();require(not BACKUP.exists(),'prior_activation_requires_reconciliation')
    invariants={str(p):sha(p) for p in [BASE/'config/node-config.json',BASE/'config/research-control/policy.json',BASE/'config/official-calibration/ev-public-20260908.json',parent/'ev_calibration_control.py',parent/'ev_calibration_runner.py',parent/'model_transport.py']}
    with ops.deployment_lock():
        BACKUP.mkdir(parents=True);subprocess.run(['icacls.exe',str(BACKUP),'/inheritance:r','/grant:r','SYSTEM:(OI)(CI)F','Administrators:(OI)(CI)F'],check=True,stdout=subprocess.DEVNULL)
        for name in ['tools.py','runtime.py']:shutil.copy2(parent/name,BACKUP/name)
        shutil.copy2(seal_path,BACKUP/'seal.json');ops.write(BACKUP/'before.json',{'work':before,'ev':ev,'invariants':invariants})
        changed=False
        try:
            ops.service('Stop');require(ops.work_guard()==before and ev_guard()==ev,'work_changed_during_stop')
            for name in NAMES:
                target=parent/name;temporary=target.with_name(name+'.siim-new')
                require(not temporary.exists(),'temporary_collision');shutil.copyfile(STAGE/name,temporary)
                subprocess.run(['icacls.exe',str(temporary),'/inheritance:r','/grant:r','SYSTEM:F','Administrators:F','EvoMindSvc:R'],check=True,stdout=subprocess.DEVNULL)
                os.replace(temporary,target);changed=True
            ops.write(policy_path,policy)
            subprocess.run(['icacls.exe',str(policy_path),'/inheritance:r','/grant:r','SYSTEM:F','Administrators:F','EvoMindSvc:R'],check=True,stdout=subprocess.DEVNULL)
            seal=json.loads(seal_path.read_text(encoding='utf-8-sig'));rows={row['path']:row for row in seal['files']}
            for rel,row in rows.items():
                if rel not in ['runtime/evomind_runtime/tools.py','runtime/evomind_runtime/runtime.py']:require(sha(BASE/'bundle'/rel)==row['sha256'],'unrelated_bundle_changed')
            for name in NAMES:
                rel='runtime/evomind_runtime/'+name;path=parent/name;rows[rel]={'path':rel,'size':path.stat().st_size,'sha256':sha(path)}
            seal.update(files=sorted(rows.values(),key=lambda r:r['path'].casefold()),file_count=len(rows),sealed_at_utc=datetime.now(timezone.utc).isoformat())
            temporary=seal_path.with_name('bundle-integrity.siim-new.json');ops.write(temporary,seal);os.replace(temporary,seal_path)
            for path,h in invariants.items():require(sha(path)==h,'unrelated_policy_changed')
            ops.service('Start');require(ops.work_guard()==before and ev_guard()==ev,'work_changed_after_start')
            result={'status':'siim_extension_activated','policy_sha256':sha(policy_path),'cases':9,'model':'deepseek-v4-pro',
                'ev_charged_seconds':sum(r[2] for r in ev),'old_studies_unchanged':True,'official_submissions':0,'training_started':False}
            ops.write(BACKUP/'result.json',result);print(json.dumps(result))
        except Exception as e:
            rollback='not_needed'
            if changed:
                try:
                    ops.service('Stop')
                    for name in ['tools.py','runtime.py']:shutil.copy2(BACKUP/name,parent/name)
                    shutil.copy2(BACKUP/'seal.json',seal_path)
                    for name in NAMES[2:]:
                        if (parent/name).exists():(parent/name).rename(BACKUP/('failed-'+name))
                    if policy_path.exists():policy_path.rename(BACKUP/'failed-policy.json')
                    ops.service('Start');rollback='restored'
                except Exception:rollback='manual_reconciliation_required'
            ops.write(BACKUP/'failure.json',{'error_type':type(e).__name__,'rollback':rollback});raise

if __name__=='__main__':main()
