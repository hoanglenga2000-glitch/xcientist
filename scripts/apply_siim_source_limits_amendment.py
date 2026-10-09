"""Backed-up source-based limit amendment; preserves every case start and charged attempt."""
import json,os,shutil,sqlite3,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
BASE=Path('C:/ProgramData/EvoMind');STAGE=BASE/'staging/siim-mlebench-calibration-20260908/official-limits-amendment'
RUNTIME_STAGE=BASE/'staging/siim-mlebench-calibration-20260908/runtime-extension'
BACKUP=BASE/'backups/siim-official-limits-amendment-20260909'
sys.path[:0]=[str(BASE/'staging/ev-deepseek-v4-pro-20260908'),str(BASE/'bundle/runtime')]
import apply_ev_deepseek_route as ops
ops.BACKUP=BACKUP;sha,require=ops.sha,ops.require
MODULES=['siim_calibration_budget.py','siim_calibration_control.py','siim_calibration_runner.py']

def budget_snapshot():
    snapshot={}
    for name in ['ev_calibration_budget.sqlite3','siim_calibration_budget.sqlite3']:
        with sqlite3.connect((BASE/'data/workspace/runtime'/name).as_uri()+'?mode=ro',uri=True) as c:
            c.execute('PRAGMA query_only=ON')
            rows=c.execute('SELECT * FROM attempts ORDER BY id').fetchall()
            names=[r[1] for r in c.execute('PRAGMA table_info(attempts)')]
            status_index=names.index('status')
            require(all(r[status_index] in {'completed','failed'} for r in rows),'training_or_settlement_in_flight')
            snapshot[name]={'attempts':rows,'cases':c.execute('SELECT run_id,started FROM cases ORDER BY run_id').fetchall()}
    return snapshot

def main():
    require(sha(STAGE/'manifest.json')=='21bdf388dd276362f161d03e105d92e1687e8fb0e6f200764c37024a7a037d93','amendment_payload_changed')
    manifest=json.loads((STAGE/'manifest.json').read_text(encoding='utf-8'))
    for relative,h in manifest['files'].items():require(sha(STAGE/relative)==h,'payload_file_changed')
    parent=BASE/'bundle/runtime/evomind_runtime';policy_path=BASE/'config/official-calibration/siim-mlebench-20260908.json';seal_path=BASE/'state/bundle-integrity.json'
    require(sha(policy_path)==manifest['baseline_policy_sha256'],'baseline_policy_changed')
    require(sha(seal_path)==manifest['baseline_seal_sha256'],'baseline_seal_changed')
    for name in MODULES:require(sha(parent/name)==manifest['baseline_module_hashes'][name],'baseline_module_changed')
    old=json.loads(policy_path.read_text(encoding='utf-8'));new=json.loads((STAGE/'policy.json').read_text(encoding='utf-8'))
    require(new['supersedes_policy_sha256']==sha(policy_path),'policy_parent_changed')
    for key in ['cases','managed_hpc_identity','tenant_id','owner_principal_id','protocol_sha256','data_manifest_sha256','baseline_source_sha256']:
        if key=='cases':
            require(set(new[key])==set(old[key]),'case_set_changed')
            for run in old[key]:
                for field in ['arm','seed','case_id','workspace_root']:require(new[key][run][field]==old[key][run][field],'case_identity_changed')
        else:require(new[key]==old[key],'scientific_or_resource_identity_changed')
    require(new['resource_limits']['global_limit_seconds']==86400,'global_budget_expansion_not_authorized')
    work=ops.work_guard();budget=budget_snapshot()
    db=BASE/'data/workspace/runtime/runtime.sqlite3'
    with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c:
        sessions={run:json.loads(row[0]) for run in old['cases'] if (row:=c.execute('SELECT metadata_json FROM sessions WHERE id=?',(run,)).fetchone())}
    targets={parent/name:STAGE/name for name in MODULES}
    targets[policy_path]=STAGE/'policy.json'
    for name in ['siim_calibration_suite.py','siim_aide_controller.py','resource-amendment.json']:targets[RUNTIME_STAGE/name]=STAGE/name
    for run,row in new['cases'].items():
        for name in ['task.json','data-profile.json']:
            targets[RUNTIME_STAGE/'cases'/row['case_id']/name]=STAGE/'cases'/row['case_id']/name
            local=Path(row['workspace_root'])/'inputs'/name
            if local.exists():
                require(sha(local)==old['cases'][run]['input_hashes']['inputs/'+name],'existing_agent_input_changed')
                targets[local]=STAGE/'cases'/row['case_id']/name
    require(not BACKUP.exists(),'prior_amendment_requires_reconciliation')
    with ops.deployment_lock():
        BACKUP.mkdir(parents=True)
        subprocess.run(['icacls.exe',str(BACKUP),'/inheritance:r','/grant:r','SYSTEM:(OI)(CI)F','Administrators:(OI)(CI)F'],check=True,stdout=subprocess.DEVNULL)
        backup_targets={}
        for index,target in enumerate(targets):
            saved=BACKUP/('file-'+str(index))
            if target.exists():shutil.copy2(target,saved);backup_targets[str(target)]=str(saved)
            else:backup_targets[str(target)]=None
        shutil.copy2(seal_path,BACKUP/'seal.json');ops.write(BACKUP/'before.json',{'targets':backup_targets,'sessions':sessions,'budget':budget,'work':work})
        changed=False
        try:
            ops.service('Stop');require(budget_snapshot()==budget and ops.work_guard()==work,'work_changed_during_stop')
            for target,source in targets.items():
                target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.official-limits-new')
                require(not temporary.exists(),'temporary_collision');shutil.copyfile(source,temporary)
                subprocess.run(['icacls.exe',str(temporary),'/inheritance:r','/grant:r','SYSTEM:F','Administrators:F','EvoMindSvc:R'],check=True,stdout=subprocess.DEVNULL)
                os.replace(temporary,target);changed=True
            from evomind_runtime.store import RuntimeStore
            store=RuntimeStore(db)
            try:
                for run,metadata in sessions.items():
                    require(metadata.get('siim_calibration')=={'policy_sha256':manifest['baseline_policy_sha256']},'existing_case_policy_mismatch')
                    metadata={**metadata,'siim_calibration':{'policy_sha256':manifest['new_policy_sha256']},
                              'resource_limits_amendment':{'parent_policy_sha256':manifest['baseline_policy_sha256'],'case_starts_reset':False}}
                    store.update_session(run,metadata_json=json.dumps(metadata))
                    store.append_event(run,'siim.resource_limits_amended',{'old_policy_sha256':manifest['baseline_policy_sha256'],
                        'new_policy_sha256':manifest['new_policy_sha256'],'global_budget_expanded':False,'charges_and_starts_preserved':True,
                        'authorization':'用户要求统一按照比赛限制自动调节','case_ceiling_seconds':86400,'step_ceiling_seconds':32400})
            finally:store.close()
            seal=json.loads(seal_path.read_text(encoding='utf-8-sig'))
            changed_paths={'runtime/evomind_runtime/'+name for name in MODULES}
            for row in seal['files']:
                path=BASE/'bundle'/row['path']
                if row['path'] in changed_paths:row.update(sha256=sha(path),size=path.stat().st_size)
                else:require(sha(path)==row['sha256'],'unrelated_bundle_changed')
            seal['sealed_at_utc']=datetime.now(timezone.utc).isoformat();temporary=seal_path.with_name('bundle-integrity.official-limits-new.json');ops.write(temporary,seal);os.replace(temporary,seal_path)
            require(budget_snapshot()==budget,'budget_reset_detected')
            ops.service('Start');require(budget_snapshot()==budget and ops.work_guard()==work,'state_changed_after_start')
            result={'status':'source_limits_amendment_activated','policy_sha256':sha(policy_path),
                'case_ceiling_hours':24,'step_ceiling_hours':9,'global_gpu_budget_hours':24,
                'case_starts_and_all_charges_preserved':True,'data_and_model_hyperparameters_unchanged':True,
                'checkpoint_resume_enabled':True,'training_started':False}
            ops.write(BACKUP/'result.json',result);print(json.dumps(result),flush=True)
        except Exception as error:
            if changed:
                ops.service('Stop')
                for path,saved in backup_targets.items():
                    target=Path(path)
                    if saved:shutil.copy2(saved,target)
                    elif target.exists():target.rename(BACKUP/('failed-new-'+target.name))
                shutil.copy2(BACKUP/'seal.json',seal_path)
                from evomind_runtime.store import RuntimeStore
                store=RuntimeStore(db)
                try:
                    for run,metadata in sessions.items():store.update_session(run,metadata_json=json.dumps(metadata))
                finally:store.close()
                ops.service('Start')
            ops.write(BACKUP/'failure.json',{'error_type':type(error).__name__,'rollback':'attempted' if changed else 'not_needed'});raise

if __name__=='__main__':main()
