"""One-file admission fix using the existing guarded switch/restart implementation."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('C:/ProgramData/EvoMind')
STAGE=ROOT/'staging/ev-deepseek-v4-pro-20260908'
FIX=STAGE/'contract-fix-v2'
BACKUP=ROOT/'backups/ev-deepseek-contract-fix-20260908'
sys.path.insert(0,str(STAGE))
import apply_ev_deepseek_route as ops
ops.BACKUP=BACKUP
sha=ops.sha
require=ops.require


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    manifest=FIX/'manifest.json'
    require(sha(manifest)=='0cd1cc3dd08c42d09c34e869bf5cbf2614b86094caaa679bc28e8ac4d41aab25','manifest_changed')
    m=json.loads(manifest.read_text())
    target=ROOT/'bundle/runtime/evomind_runtime/store.py';seal_path=ROOT/'state/bundle-integrity.json'
    require(sha(target)==m['baseline_store_sha256'],'store_changed')
    require(sha(seal_path)==m['baseline_seal_sha256'],'seal_changed')
    require(sha(FIX/'store.py')==m['candidate_store_sha256'],'candidate_changed')
    seal=json.loads(seal_path.read_text(encoding='utf-8-sig'))
    for row in seal['files']:require(sha(ROOT/'bundle'/row['path'])==row['sha256'],'bundle_integrity_failed')
    config=ROOT/'config/node-config.json';cfg=json.loads(config.read_text(encoding='utf-8-sig'))
    require(cfg['hpc']['state']!='active','global_hpc_startup_requires_review')
    unchanged={str(p.relative_to(ROOT)):sha(p) for p in [config,ROOT/'bundle/scripts/Start-Node.ps1',ROOT/'bundle/runtime/evomind_runtime/model_transport.py',ROOT/'state/ev-deepseek-route-20260908.json']}
    work=ops.work_guard()
    if not args.apply:
        print(json.dumps({'status':'ready_for_contract_fix','work':work,'production_changed':False}));return
    require(not BACKUP.exists(),'prior_fix_requires_reconciliation')
    with ops.deployment_lock():
        require(ops.work_guard()==work,'work_changed')
        BACKUP.mkdir(parents=True)
        subprocess.run(['icacls.exe',str(BACKUP),'/inheritance:r','/grant:r','SYSTEM:(OI)(CI)F','Administrators:(OI)(CI)F'],check=True,stdout=subprocess.DEVNULL)
        shutil.copy2(target,BACKUP/'store.py');shutil.copy2(seal_path,BACKUP/'seal.json')
        ops.write(BACKUP/'before.json',{'unchanged':unchanged,'work':work})
        changed=False
        try:
            ops.service('Stop')
            require(ops.work_guard()==work,'work_changed_during_stop')
            temporary=target.with_name('store.deepseek-v2-new.py')
            require(not temporary.exists(),'temporary_collision')
            shutil.copyfile(FIX/'store.py',temporary)
            subprocess.run(['icacls.exe',str(temporary),'/inheritance:r','/grant:r','SYSTEM:F','Administrators:F','EvoMindSvc:R'],check=True,stdout=subprocess.DEVNULL)
            os.replace(temporary,target);changed=True
            for row in seal['files']:
                p=ROOT/'bundle'/row['path']
                if row['path']=='runtime/evomind_runtime/store.py':row.update(size=p.stat().st_size,sha256=sha(p))
                else:require(sha(p)==row['sha256'],'unrelated_file_changed')
            seal['sealed_at_utc']=datetime.now(timezone.utc).isoformat()
            tmpseal=seal_path.with_name('bundle-integrity.deepseek-v2-new.json');ops.write(tmpseal,seal);os.replace(tmpseal,seal_path)
            for p,digest in unchanged.items():require(sha(ROOT/p)==digest,'unrelated_configuration_changed')
            ops.service('Start')
            require(ops.work_guard()==work,'historical_work_changed')
            result={'status':'contract_admission_fixed_pending_live_validation','model':'deepseek-v4-pro',
                    'strict_guards_retained':True,'hpc_binding_unchanged':True,'historical_work_unchanged':True,'backup':str(BACKUP)}
            ops.write(BACKUP/'result.json',result);print(json.dumps(result),flush=True)
        except Exception as error:
            rollback='not_needed'
            if changed:
                try:
                    ops.service('Stop');shutil.copy2(BACKUP/'store.py',target);shutil.copy2(BACKUP/'seal.json',seal_path);ops.service('Start')
                    rollback='pre_fix_state_restored'
                except Exception:rollback='manual_reconciliation_required'
            ops.write(BACKUP/'failure.json',{'error_type':type(error).__name__,'code':str(error)[:120],'rollback':rollback})
            raise


if __name__=='__main__':main()
