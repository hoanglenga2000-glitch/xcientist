from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/ev-public-calibration-20260908/environment-repair'
OUT.mkdir(exist_ok=True)
source=(ROOT/'artifacts/ev-public-calibration-20260908/finalize-v3/ev_calibration_finalize_v3.py').read_text(encoding='utf-8')
source=source.replace('session_ev_calibration_data_finalize_v3_20260908','session_ev_calibration_environment_20260908')
source=source.replace('data-finalize-v3','environment-repair').replace('ev_hpc_data_finalize.py','ev_hpc_environment_repair.py')
source=source.replace('timeout_seconds=300','timeout_seconds=900')
(OUT/'ev_calibration_environment.py').write_text(source,encoding='utf-8')
launcher=(ROOT/'scripts/start_ev_infrastructure_task.ps1').read_text(encoding='utf-8')
launcher=launcher.replace('EvoMind-EV-Infrastructure-20260908','EvoMind-EV-Environment-20260908').replace('ev_calibration_infrastructure.py','ev_calibration_environment.py')
launcher=launcher.replace('New-TimeSpan -Minutes 5','New-TimeSpan -Minutes 18').replace('read_only_hpc_identity_and_package_metadata','isolate_existing_dependencies_no_training')
(OUT/'start_ev_environment.ps1').write_text(launcher,encoding='utf-8')
print('Environment repair task prepared; frozen data and baseline parameters remain unchanged.')
