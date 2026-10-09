$ErrorActionPreference = 'Stop'
$candidateRoot = 'C:/ProgramData/EvoMind/staging/system-completeness-20260907/endurance-candidate-v9'
$candidateManifest = $candidateRoot + '/candidate-manifest.json'
if ((Get-FileHash -LiteralPath $candidateManifest -Algorithm SHA256).Hash.ToLowerInvariant() -ne '398cedf2f7a79004c03dab5d08053f8b57df7cd25edd07c73a8e047c207d87c0') { throw 'candidate_manifest_mismatch' }
$env:PYTHONPATH = $candidateRoot + ';C:/EMQA/sys0907-report-deps-v1/site-packages;C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src'
$env:PYTHONUTF8 = '1'
& 'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv/Scripts/python.exe' -X utf8 ($candidateRoot + '/run_model_endurance_acceptance.py') --model gpt-5.5 --root C:/EMQA/sys0907-gpt55-stability-v9 --candidate-manifest $candidateManifest --gateway-config C:/ProgramData/EvoMind/state/cliproxy.runtime.yaml --protocol responses --tier omit --route configured_upstream --case-interval-seconds 180 --seconds 5400
exit $LASTEXITCODE
