"""User-authorized EV stop for the SIIM MLE-bench cutover; no HPC signals."""
import json, sqlite3, subprocess, sys, time
from pathlib import Path
BASE=Path('C:/ProgramData/EvoMind')
sys.path.insert(0,str(BASE/'bundle/runtime'))
from evomind_runtime.store import RuntimeStore

def main():
    policy=json.loads((BASE/'config/official-calibration/ev-public-20260908.json').read_text())
    db=BASE/'data/workspace/runtime/runtime.sqlite3'
    store=RuntimeStore(db)
    paused=[]
    try:
        for run,row in policy['cases'].items():
            session=store.get_session(run)
            if row['arm']!='evomind' or session is None:continue
            store.set_user_pause_requested(run,True)
            store.append_event(run,'calibration.user_approved_competition_cutover',{
                'reason':'switch_to_mlebench_siim_isic','model_code_modified':False,
                'budget_reset':False,'hpc_signals_sent':0})
            paused.append(run)
    finally:store.close()
    # Only the known EV coordinator is stopped, never the runtime/HPC worker.
    command="$ErrorActionPreference='Stop';Stop-ScheduledTask -TaskName 'EvoMind-EV-Suite-diagnostics-v5-20260908'"
    result=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,timeout=30)
    if result.returncode:raise ValueError('controller_pause_failed')
    receipt={'status':'ev_followup_stopped_for_siim','runs':paused,'budget_reset':False,'hpc_signals_sent':0,'at_epoch':time.time()}
    path=BASE/'staging/ev-public-calibration-20260908/runtime-extension/service-output/stop-for-siim.json'
    with path.open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))

if __name__=='__main__':main()
