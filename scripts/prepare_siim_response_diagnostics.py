"""One-module diagnostics rollout; no provider/request/model or budget changes."""
import prepare_siim_web_bridge as patch
patch.OUT=patch.ROOT/'artifacts/siim-mlebench-calibration-20260908/response-diagnostics'
patch.DEST=patch.REMOTE+'/staging/siim-mlebench-calibration-20260908/response-diagnostics'
patch.TARGETS={'model_transport.py':'bundle/runtime/evomind_runtime/model_transport.py'}
patch.NEW_MODULES={}
patch.SNAPSHOT_EXTRA={}
patch.GENERATE_BRIDGE_CONFIG=False
patch.EXTRA_INSTALLERS={'apply_siim_response_diagnostics.py':True}
if __name__=='__main__':patch.main()
