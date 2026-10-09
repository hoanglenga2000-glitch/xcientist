"""Reuse the guarded application-only patch workflow for completion recovery."""
import prepare_siim_web_bridge as patch

patch.OUT=patch.ROOT/'artifacts/siim-mlebench-calibration-20260908/completion-recovery'
patch.DEST=patch.REMOTE+'/staging/siim-mlebench-calibration-20260908/completion-recovery'
patch.TARGETS['siim_calibration_web.py']='bundle/runtime/evomind_runtime/siim_calibration_web.py'
patch.NEW_MODULES={'siim_candidate_contract.py':'bundle/runtime/evomind_runtime/siim_candidate_contract.py'}
patch.SNAPSHOT_EXTRA={'siim-web-bridge.json':'config/official-calibration/siim-web-bridge.json'}
patch.EXTRA_INSTALLERS={'apply_siim_completion_recovery.py':True}

if __name__=='__main__':
    patch.main()
