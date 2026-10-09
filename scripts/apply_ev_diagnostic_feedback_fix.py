"""Transactional EV-only failure feedback update; scorer/protocol remain frozen."""
import json,os,shutil,sqlite3,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('C:/ProgramData/EvoMind')
STAGE=ROOT/'staging/ev-public-calibration-20260908/diagnostic-feedback-fix'
BACKUP=ROOT/'backups/ev-diagnostic-feedback-fix-20260908'
sys.path.insert(0,str(ROOT/'staging/ev-deepseek-v4-pro-20260908'))
import apply_ev_deepseek_route as ops
ops.BACKUP=BACKUP
sha,require=ops.sha,ops.require
EXPECTED={
 'ev_calibration_control.py':'084d5315d252a042f3ef598d2d9456057a5fa5a1b564c877220f0096b982e02e',
 'ev_calibration_diagnostics.py':'9a1c0ae1d83d616eb019bb244d10c0b2c0959b3fae3b924496dfd92446f265e5'}

def guard():
    with sqlite3.connect((ROOT/'data/workspace/runtime/ev_calibration_budget.sqlite3').as_uri()+'?mode=ro',uri=True) as c:
        rows=c.execute('SELECT id,run_id,status,charged FROM attempts ORDER BY id').fetchall()
        clocks=c.execute('SELECT run_id,started FROM cases ORDER BY run_id').fetchall()
        require(all(r[2] in {'completed','failed'} for r in rows),'calibration_execution_in_flight')
    policy=json.loads((ROOT/'config/official-calibration/ev-public-20260908.json').read_text())
    with sqlite3.connect((ROOT/'data/workspace/runtime/runtime.sqlite3').as_uri()+'?mode=ro',uri=True) as c:
        for run,row in policy['cases'].items():
            if row['arm']!='evomind':continue
            found=c.execute('SELECT metadata_json FROM sessions WHERE id=?',(run,)).fetchone()
            require(found is None or json.loads(found[0]).get('user_pause_requested') is True,'ev_case_not_paused')
    return {'attempts':rows,'clocks':clocks}

def main():
    parent=ROOT/'bundle/runtime/evomind_runtime';seal_path=ROOT/'state/bundle-integrity.json'
    require(sha(parent/'ev_calibration_control.py')=='838b44caf1538aad39a48bcff37ad5fe3f27893260337b97cc1ab9cf8c0a41e0','control_baseline_changed')
    require(sha(seal_path)=='6507aee4779591be74c0a9bbe59be3a51f28133de48a3f5e79ef742872ec331a','seal_baseline_changed')
    require(not(parent/'ev_calibration_diagnostics.py').exists(),'diagnostic_extension_exists')
    for name,h in EXPECTED.items():require(sha(STAGE/name)==h,'candidate_hash_changed')
    before=ops.work_guard();budget=guard()
    invariant_paths=[ROOT/'config/official-calibration/ev-public-20260908.json',parent/'ev_calibration_runner.py',parent/'runtime.py',ROOT/'config/research-control/policy.json']
    invariants={str(p):sha(p) for p in invariant_paths}
    require(not BACKUP.exists(),'prior_fix_requires_reconciliation')
    with ops.deployment_lock():
        BACKUP.mkdir(parents=True)
        subprocess.run(['icacls.exe',str(BACKUP),'/inheritance:r','/grant:r','SYSTEM:(OI)(CI)F','Administrators:(OI)(CI)F'],check=True,stdout=subprocess.DEVNULL)
        shutil.copy2(parent/'ev_calibration_control.py',BACKUP/'ev_calibration_control.py');shutil.copy2(seal_path,BACKUP/'seal.json')
        ops.write(BACKUP/'before.json',{'invariants':invariants,'work':before,'budget':budget})
        changed=False
        try:
            ops.service('Stop');require(guard()==budget and ops.work_guard()==before,'work_changed')
            for name in EXPECTED:
                target=parent/name;temporary=target.with_name(name+'.diagnostic-new')
                require(not temporary.exists(),'temporary_collision');shutil.copyfile(STAGE/name,temporary)
                subprocess.run(['icacls.exe',str(temporary),'/inheritance:r','/grant:r','SYSTEM:F','Administrators:F','EvoMindSvc:R'],check=True,stdout=subprocess.DEVNULL)
                os.replace(temporary,target);changed=True
            seal=json.loads(seal_path.read_text(encoding='utf-8-sig'));rows={r['path']:r for r in seal['files']}
            for rel,row in rows.items():
                if rel!='runtime/evomind_runtime/ev_calibration_control.py':require(sha(ROOT/'bundle'/rel)==row['sha256'],'unrelated_bundle_changed')
            for name in EXPECTED:
                path=parent/name;rel='runtime/evomind_runtime/'+name
                rows[rel]={'path':rel,'size':path.stat().st_size,'sha256':sha(path)}
            seal.update(files=sorted(rows.values(),key=lambda r:r['path'].casefold()),file_count=len(rows),sealed_at_utc=datetime.now(timezone.utc).isoformat())
            temporary=seal_path.with_name('bundle-integrity.diagnostic-new.json');ops.write(temporary,seal);os.replace(temporary,seal_path)
            for p,h in invariants.items():require(sha(p)==h,'protocol_or_scorer_changed')
            ops.service('Start');require(guard()==budget and ops.work_guard()==before,'work_changed_after_start')
            result={'status':'ev_failure_feedback_activated','model':'deepseek-v4-pro','scorer_and_protocol_unchanged':True,
                    'case_clocks_and_ledgers_unchanged':True,'modeling_code_modified':False,'raw_error_text_exposed':False,'files':EXPECTED}
            ops.write(BACKUP/'result.json',result);print(json.dumps(result),flush=True)
        except Exception as e:
            rollback='not_needed'
            if changed:
                try:
                    ops.service('Stop');shutil.copy2(BACKUP/'ev_calibration_control.py',parent/'ev_calibration_control.py');shutil.copy2(BACKUP/'seal.json',seal_path)
                    added=parent/'ev_calibration_diagnostics.py'
                    if added.exists():added.rename(BACKUP/'failed-diagnostics.py')
                    ops.service('Start');rollback='restored'
                except Exception:rollback='manual_reconciliation_required'
            ops.write(BACKUP/'failure.json',{'error_type':type(e).__name__,'rollback':rollback});raise

if __name__=='__main__':main()
