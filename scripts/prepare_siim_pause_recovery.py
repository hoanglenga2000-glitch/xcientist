import prepare_siim_web_bridge as patch
patch.OUT=patch.ROOT/'artifacts/siim-mlebench-calibration-20260908/pause-recovery'
patch.DEST=patch.REMOTE+'/staging/siim-mlebench-calibration-20260908/pause-recovery'
patch.TARGETS={name:'staging/siim-mlebench-calibration-20260908/runtime-extension/'+name for name in ['siim_calibration_suite.py','siim_aide_controller.py']}
patch.NEW_MODULES={}
patch.SNAPSHOT_EXTRA={'siim-web-bridge.json':'config/official-calibration/siim-web-bridge.json'}
patch.EXTRA_INSTALLERS={'apply_siim_pause_recovery.py':True}
if __name__=='__main__':patch.main()
