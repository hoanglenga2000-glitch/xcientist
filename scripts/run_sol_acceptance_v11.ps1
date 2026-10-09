param([ValidateSet('preflight','stability')][string]$Mode = 'preflight')
$ErrorActionPreference = 'Stop'
$candidateRoot = 'C:/ProgramData/EvoMind/staging/system-completeness-20260907/endurance-candidate-v11'
$candidateManifest = $candidateRoot + '/candidate-manifest.json'
if ((Get-FileHash -LiteralPath $candidateManifest -Algorithm SHA256).Hash.ToLowerInvariant() -ne '91b6bb8c8d31281222b30794d46a0517f1797e2a90f818c9606c084303616fd7') { throw 'candidate_manifest_mismatch' }
$env:PYTHONPATH = $candidateRoot + ';C:/EMQA/sys0907-report-delivered-v2/site-packages;C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
$testRoot = if ($Mode -eq 'preflight') { 'C:/EMQA/sys0908-sol-report-preflight-v11' } else { 'C:/EMQA/sys0908-sol-stability-v11' }
$env:MPLCONFIGDIR = $testRoot + '/matplotlib-cache'
$arguments = @('-X','utf8',($candidateRoot + '/run_model_endurance_acceptance.py'),'--model','gpt-5.6-sol','--root',$testRoot,'--candidate-manifest',$candidateManifest,'--gateway-config','C:/ProgramData/EvoMind/state/cliproxy.runtime.yaml','--protocol','responses','--tier','omit','--route','configured_upstream')
if ($Mode -eq 'preflight') { $arguments += @('--preflight-cases','3','--report-every','1') }
else { $arguments += @('--case-interval-seconds','180','--seconds','5400') }
& 'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv/Scripts/python.exe' @arguments
exit $LASTEXITCODE
