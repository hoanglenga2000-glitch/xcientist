param(
  [ValidateSet('gpt-6-astra','gpt-5.6-sol')][string]$Model = 'gpt-5.6-sol',
  [Parameter(Mandatory=$true)][ValidatePattern('^C:/EMQA/sys0907-[a-z0-9-]+$')][string]$OutputRoot
)
$ErrorActionPreference = 'Stop'
$candidateRoot = 'C:/ProgramData/EvoMind/staging/system-completeness-20260907/endurance-candidate-v4'
$candidateManifest = $candidateRoot + '/candidate-manifest.json'
if ((Get-FileHash -LiteralPath $candidateManifest -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'bd1bd1639c7b9991a8bb43ea91f41ee1029faaf349fb29d98c7104b9200c8828') { throw 'candidate_manifest_mismatch' }
$env:PYTHONPATH = $candidateRoot + ';C:/EMQA/sys0907-report-deps-v1/site-packages;C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src'
$env:PYTHONUTF8 = '1'
& 'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv/Scripts/python.exe' -X utf8 ($candidateRoot + '/run_model_endurance_acceptance.py') --model $Model --root $OutputRoot --candidate-manifest $candidateManifest --gateway-config C:/ProgramData/EvoMind/state/cliproxy.runtime.yaml --protocol responses --tier omit --seconds 5400
exit $LASTEXITCODE
