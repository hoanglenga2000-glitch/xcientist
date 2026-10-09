from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/ev-public-calibration-20260908/runtime-extension'
launcher=(ROOT/'scripts/start_ev_infrastructure_task.ps1').read_text(encoding='utf-8')
launcher=launcher.replace("param(","param(")
launcher=launcher.replace("$stage='C:\\ProgramData\\EvoMind\\staging\\ev-public-calibration-20260908'","$stage='C:\\ProgramData\\EvoMind\\staging\\ev-public-calibration-20260908\\runtime-extension'")
launcher=launcher.replace("$taskName='EvoMind-EV-Infrastructure-20260908'","$taskName='EvoMind-EV-Observe-'+[DateTime]::UtcNow.ToString('yyyyMMddHHmmss')")
launcher=launcher.replace("$name='EvoMind-EV-Infrastructure-20260908'","$name='EvoMind-EV-Observe-'+[DateTime]::UtcNow.ToString('yyyyMMddHHmmss')")
launcher=launcher.replace('ev_calibration_infrastructure.py','ev_calibration_observe.py')
launcher=launcher.replace("New-Item -ItemType Directory -Path $output|Out-Null","if(-not(Test-Path -LiteralPath $output)){New-Item -ItemType Directory -Path $output|Out-Null}")
(OUT/'start_ev_observer.ps1').write_text(launcher,encoding='utf-8')
print('Observer task prepared.')
