"""Scoped, backed-up output-cap correction after a verified pre-fit truncation."""
import json,os,shutil,sqlite3,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('C:/ProgramData/EvoMind');STAGE=ROOT/'staging/ev-public-calibration-20260908/output-budget-fix';BACKUP=ROOT/'backups/ev-output-budget-fix-20260908'
sys.path.insert(0,str(ROOT/'staging/ev-deepseek-v4-pro-20260908'))
import apply_ev_deepseek_route as ops
ops.BACKUP=BACKUP;sha,require=ops.sha,ops.require
def main():
 require(sha(STAGE/'manifest.json')=='a13aaf1da61389cc790b467e9e38516d24161b1ad36dc350b1e89a82bd4ae983','manifest_changed')
 m=json.loads((STAGE/'manifest.json').read_text());target=ROOT/'bundle/runtime/evomind_runtime/runtime.py';seal_path=ROOT/'state/bundle-integrity.json'
 require(sha(target)==m['baseline_runtime_sha256'] and sha(seal_path)==m['baseline_seal_sha256'],'baseline_changed')
 require(sha(STAGE/'runtime.py')==m['candidate_sha256'],'candidate_changed')
 def case_guard():
  with sqlite3.connect((ROOT/'data/workspace/runtime/ev_calibration_budget.sqlite3').as_uri()+'?mode=ro',uri=True) as c:
   rows=c.execute('SELECT status,charged FROM attempts').fetchall()
   require(all(r[0] in {'completed','failed'} for r in rows),'calibration_work_in_flight')
   return rows
 before=ops.work_guard();cases=case_guard()
 require(not BACKUP.exists(),'prior_fix_requires_reconciliation')
 with ops.deployment_lock():
  BACKUP.mkdir(parents=True);subprocess.run(['icacls.exe',str(BACKUP),'/inheritance:r','/grant:r','SYSTEM:(OI)(CI)F','Administrators:(OI)(CI)F'],check=True,stdout=subprocess.DEVNULL)
  shutil.copy2(target,BACKUP/'runtime.py');shutil.copy2(seal_path,BACKUP/'seal.json');ops.write(BACKUP/'before.json',{'work':before,'case_budget':cases})
  changed=False
  try:
   ops.service('Stop');require(case_guard()==cases and ops.work_guard()==before,'work_changed')
   temp=target.with_name('runtime.output-cap-new.py');require(not temp.exists(),'temporary_collision');shutil.copyfile(STAGE/'runtime.py',temp)
   subprocess.run(['icacls.exe',str(temp),'/inheritance:r','/grant:r','SYSTEM:F','Administrators:F','EvoMindSvc:R'],check=True,stdout=subprocess.DEVNULL)
   os.replace(temp,target);changed=True
   seal=json.loads(seal_path.read_text(encoding='utf-8-sig'))
   for row in seal['files']:
    path=ROOT/'bundle'/row['path']
    if row['path']=='runtime/evomind_runtime/runtime.py':row.update(sha256=sha(path),size=path.stat().st_size)
    else:require(sha(path)==row['sha256'],'unrelated_file_changed')
   seal['sealed_at_utc']=datetime.now(timezone.utc).isoformat();temp=seal_path.with_name('bundle-integrity.output-cap-new.json');ops.write(temp,seal);os.replace(temp,seal_path)
   ops.service('Start');require(case_guard()==cases and ops.work_guard()==before,'work_changed_after_start')
   result={'status':'ev_output_budget_8192_activated','model':'deepseek-v4-pro','existing_ledgers_unchanged':True,'other_tasks_default_4096':True}
   ops.write(BACKUP/'result.json',result);print(json.dumps(result),flush=True)
  except Exception as e:
   rollback='not_needed'
   if changed:
    try:ops.service('Stop');shutil.copy2(BACKUP/'runtime.py',target);shutil.copy2(BACKUP/'seal.json',seal_path);ops.service('Start');rollback='restored'
    except Exception:rollback='manual_reconciliation_required'
   ops.write(BACKUP/'failure.json',{'error_type':type(e).__name__,'rollback':rollback});raise
if __name__=='__main__':main()
