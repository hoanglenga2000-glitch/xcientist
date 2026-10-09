param([ValidateSet('preflight','stability')][string]$Mode = 'preflight')
$ErrorActionPreference = 'Stop'
$candidateRoot = 'C:/ProgramData/EvoMind/staging/system-completeness-20260907/endurance-candidate-v10'
$candidateManifest = $candidateRoot + '/candidate-manifest.json'
if ((Get-FileHash -LiteralPath $candidateManifest -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'f817265a7b095c7bcbbc1bc9ba4ba80ec108a6fe86b7f8f123ac49267a25f3c0') { throw 'candidate_manifest_mismatch' }
$env:PYTHONPATH = $candidateRoot + ';C:/EMQA/sys0907-report-deps-v1/site-packages;C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src'
$env:PYTHONUTF8 = '1'
$testRoot = if ($Mode -eq 'preflight') { 'C:/EMQA/sys0907-gpt55-report-preflight-v10' } else { 'C:/EMQA/sys0907-gpt55-stability-v10' }
$arguments = @('-X','utf8',($candidateRoot + '/run_model_endurance_acceptance.py'),'--model','gpt-5.5','--root',$testRoot,'--candidate-manifest',$candidateManifest,'--gateway-config','C:/ProgramData/EvoMind/state/cliproxy.runtime.yaml','--protocol','responses','--tier','omit','--route','configured_upstream')
if ($Mode -eq 'preflight') { $arguments += @('--preflight-cases','1','--report-every','1') }
else { $arguments += @('--case-interval-seconds','180','--seconds','5400') }
& 'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv/Scripts/python.exe' @arguments
exit $LASTEXITCODE
