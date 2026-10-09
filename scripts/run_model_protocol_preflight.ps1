param(
  [ValidateSet('gpt-6-astra','gpt-5.6-sol')][string]$Model = 'gpt-5.6-sol',
  [Parameter(Mandatory=$true)][ValidatePattern('^C:/EMQA/sys0907-[a-z0-9-]+$')][string]$OutputRoot
)
$ErrorActionPreference = 'Stop'
$candidateRoot = 'C:/ProgramData/EvoMind/staging/system-completeness-20260907/endurance-candidate-v3'
$candidateManifest = $candidateRoot + '/candidate-manifest.json'
if ((Get-FileHash -LiteralPath $candidateManifest -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'cbc67f6d42f2fe4dd72c158240a05e230b31b2c25b247407d0b858530d6d1c16') { throw 'candidate_manifest_mismatch' }
$env:PYTHONPATH = $candidateRoot + ';C:/EMQA/sys0907-report-deps-v1/site-packages;C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src'
$env:PYTHONUTF8 = '1'
& 'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv/Scripts/python.exe' -X utf8 ($candidateRoot + '/run_model_endurance_acceptance.py') --model $Model --root $OutputRoot --candidate-manifest $candidateManifest --gateway-config C:/ProgramData/EvoMind/state/cliproxy.runtime.yaml --protocol responses --tier omit --preflight-cases 2
exit $LASTEXITCODE
