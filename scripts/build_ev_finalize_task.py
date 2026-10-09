"""Derive a no-secret, no-network finalization task from the reviewed data transport."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/ev-public-calibration-20260908/finalize-v3'
OUT.mkdir(exist_ok=True)
source=(ROOT/'scripts/ev_calibration_provision.py').read_text(encoding='utf-8')
source=source.replace('session_ev_calibration_data_provision_20260908','session_ev_calibration_data_finalize_v3_20260908')
source=source.replace("'service-output/data-provision.json'","'service-output/data-finalize-v3.json'")
source=source.replace("'service-output/data-provision-error.json'","'service-output/data-finalize-v3-error.json'")
source=source.replace('data/acceptance/ev-calibration-20260908/data-provision','data/acceptance/ev-calibration-20260908/data-finalize-v3')
source=source.replace("'ev_hpc_data_adapter.py'","'ev_hpc_data_finalize.py'")
source=source.replace("'ev-data-provision'","'ev-data-finalize-v3'")
source=source.replace("key=os.environ.pop('EVOMIND_EV_KAGGLE_KEY').encode()","key=None")
source=source.replace("secret_files={'EVOMIND_SECRET_KAGGLE_API_FILE':key},timeout_seconds=1500","secret_files={},timeout_seconds=300")
assert 'data/acceptance/ev-calibration-20260908/data-provision' not in source
(OUT/'ev_calibration_finalize_v3.py').write_text(source,encoding='utf-8')
launcher=(ROOT/'scripts/start_ev_infrastructure_task.ps1').read_text(encoding='utf-8')
launcher=launcher.replace('EvoMind-EV-Infrastructure-20260908','EvoMind-EV-Finalize-v3-20260908').replace('ev_calibration_infrastructure.py','ev_calibration_finalize_v3.py')
launcher=launcher.replace('read_only_hpc_identity_and_package_metadata','finalize_existing_official_data_no_download_no_training')
(OUT/'start_ev_finalize.ps1').write_text(launcher,encoding='utf-8')
print('Finalization task built; no credential or download step included.')
