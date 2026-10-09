"""Scoped operator pause at tool boundaries; preserves models, clocks and ledger."""
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
            store.append_event(run,'calibration.operator_infrastructure_pause',{
                'reason':'restore_sanitized_failure_feedback','model_code_modified':False,
                'budget_reset':False,'hpc_signals_sent':0})
            paused.append(run)
    finally:store.close()
    # Only the known EV coordinator is stopped, never the runtime/HPC worker.
    command="$ErrorActionPreference='Stop';Stop-ScheduledTask -TaskName 'EvoMind-EV-Suite-outputcap-v4-20260908'"
    result=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,timeout=30)
    if result.returncode:raise ValueError('controller_pause_failed')
    receipt={'status':'operator_pause_requested','runs':paused,'budget_reset':False,'hpc_signals_sent':0,'at_epoch':time.time()}
    path=BASE/'staging/ev-public-calibration-20260908/runtime-extension/service-output/diagnostic-pause.json'
    with path.open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))

if __name__=='__main__':main()
