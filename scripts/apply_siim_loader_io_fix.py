"""Transactional reuse of verified SIIM Python; clocks/scorer/policy are immutable."""
import json,os,shutil,sqlite3,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
BASE=Path('C:/ProgramData/EvoMind');STAGE=BASE/'staging/siim-mlebench-calibration-20260908/loader-io-fix';BACKUP=BASE/'backups/siim-loader-io-fix-20260909'
sys.path.insert(0,str(BASE/'staging/ev-deepseek-v4-pro-20260908'))
import apply_ev_deepseek_route as ops
ops.BACKUP=BACKUP;sha,require=ops.sha,ops.require
BEFORE={'siim_calibration_control.py':'0ca9c00c93fd09c17d4bc4abfce99d81a800e31e8a2c842bc411b64073f88330'}
AFTER={'siim_calibration_control.py':'a60e85b867f8c42a4a84e933ffe0155756ff4f01f45514c665e1759ee4f6308c','siim_dataloader_runtime.py':'f4d74f6db67a5dd75e8325649f7909a1aae731cb8e5eaa2532fdf7ee08c9ccfa'}

def budget():
    result={}
    for name in ['ev_calibration_budget.sqlite3','siim_calibration_budget.sqlite3']:
        with sqlite3.connect((BASE/'data/workspace/runtime'/name).as_uri()+'?mode=ro',uri=True) as c:
            rows=c.execute('SELECT id,status,charged FROM attempts ORDER BY id').fetchall()
            require(all(r[1] in {'completed','failed'} for r in rows),'work_in_flight')
            result[name]={'attempts':rows,'clocks':c.execute('SELECT run_id,started FROM cases ORDER BY run_id').fetchall()}
    return result

def main():
    parent=BASE/'bundle/runtime/evomind_runtime';seal_path=BASE/'state/bundle-integrity.json'
    for name,h in BEFORE.items():require(sha(parent/name)==h,'active_source_changed')
    for name,h in AFTER.items():require(sha(STAGE/name)==h,'candidate_changed')
    require(sha(seal_path)=='63fd905f3eaf8d637826e55ee499f8320c3cc1112e350f447978fce8d1248423','seal_changed')
    require(not(parent/'siim_dataloader_runtime.py').exists() and not BACKUP.exists(),'prior_fix_requires_reconciliation')
    work=ops.work_guard();counts=budget();policy=BASE/'config/official-calibration/siim-mlebench-20260908.json'
    immutable={str(policy):sha(policy),str(parent/'siim_calibration_runner.py'):sha(parent/'siim_calibration_runner.py')}
    with ops.deployment_lock():
        BACKUP.mkdir(parents=True);ops.write(BACKUP/'before.json',{'work':work,'budget':counts,'immutable':immutable})
        for name in BEFORE:shutil.copy2(parent/name,BACKUP/name)
        shutil.copy2(seal_path,BACKUP/'seal.json');changed=False
        try:
            ops.service('Stop');require(budget()==counts and ops.work_guard()==work,'work_changed')
            for name in AFTER:
                target=parent/name;temporary=target.with_name(name+'.envfix-new');require(not temporary.exists(),'temp_exists');shutil.copyfile(STAGE/name,temporary)
                subprocess.run(['icacls.exe',str(temporary),'/inheritance:r','/grant:r','SYSTEM:F','Administrators:F','EvoMindSvc:R'],check=True,stdout=subprocess.DEVNULL)
                os.replace(temporary,target);changed=True
            seal=json.loads(seal_path.read_text(encoding='utf-8-sig'));rows={r['path']:r for r in seal['files']}
            for rel,r in rows.items():
                if rel not in ['runtime/evomind_runtime/'+n for n in BEFORE]:require(sha(BASE/'bundle'/rel)==r['sha256'],'unrelated_bundle_changed')
            for name,h in AFTER.items():
                rel='runtime/evomind_runtime/'+name;rows[rel]={'path':rel,'size':(parent/name).stat().st_size,'sha256':h}
            seal.update(files=sorted(rows.values(),key=lambda r:r['path'].casefold()),file_count=len(rows),sealed_at_utc=datetime.now(timezone.utc).isoformat())
            temporary=seal_path.with_name('bundle-integrity.siim-envfix.json');ops.write(temporary,seal);os.replace(temporary,seal_path)
            for path,h in immutable.items():require(sha(path)==h,'policy_or_scorer_changed')
            ops.service('Start');require(budget()==counts and ops.work_guard()==work,'state_changed_after_start')
            result={'status':'siim_serial_dataloader_io_activated','budget_reset':False,'protocol_changed':False,'scorer_changed':False,'modeling_code_changed':False}
            ops.write(BACKUP/'result.json',result);print(json.dumps(result))
        except Exception as e:
            if changed:
                ops.service('Stop')
                for name in BEFORE:shutil.copy2(BACKUP/name,parent/name)
                shutil.copy2(BACKUP/'seal.json',seal_path)
                if(parent/'siim_dataloader_runtime.py').exists():(parent/'siim_dataloader_runtime.py').rename(BACKUP/'failed-siim_dataloader_runtime.py')
                ops.service('Start')
            raise

if __name__=='__main__':main()
