"""Prepare the exact request-limit patch against the live app baseline."""
import prepare_siim_web_bridge as patch
patch.OUT=patch.ROOT/'artifacts/siim-mlebench-calibration-20260908/web-request-limits'
patch.DEST=patch.REMOTE+'/staging/siim-mlebench-calibration-20260908/web-request-limits'
patch.TARGETS={'model_transport.py':'bundle/runtime/evomind_runtime/model_transport.py'}
patch.NEW_MODULES={name:'bundle/runtime/evomind_runtime/'+name for name in ['siim_request_limits.py','model_deadline_worker.py']}
patch.GENERATE_BRIDGE_CONFIG=False
patch.EXTRA_INSTALLERS={'apply_siim_request_limits.py':True}
if __name__=='__main__':patch.main()
