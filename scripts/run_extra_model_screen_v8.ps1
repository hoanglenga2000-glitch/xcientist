$ErrorActionPreference = 'Stop'
$candidateRoot = 'C:/ProgramData/EvoMind/staging/system-completeness-20260907/endurance-candidate-v8'
$candidateManifest = $candidateRoot + '/candidate-manifest.json'
if ((Get-FileHash -LiteralPath $candidateManifest -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'b7def2b3cad1877e7ab314fd274d8dfafd577753786a7b8f3d9501070c7a40ef') { throw 'candidate_manifest_mismatch' }
$env:PYTHONPATH = $candidateRoot + ';C:/EMQA/sys0907-report-deps-v1/site-packages;C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src'
$env:PYTHONUTF8 = '1'
$entries = @(
  @{model='gpt-5.6-terra';root='C:/EMQA/sys0907-terra-direct-preflight-v8'},
  @{model='gpt-5.6-luna';root='C:/EMQA/sys0907-luna-direct-preflight-v8'},
  @{model='gpt-5.5';root='C:/EMQA/sys0907-gpt55-direct-preflight-v8'}
)
foreach ($entry in $entries) {
  & 'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv/Scripts/python.exe' -X utf8 ($candidateRoot + '/run_model_endurance_acceptance.py') --model $entry.model --root $entry.root --candidate-manifest $candidateManifest --gateway-config C:/ProgramData/EvoMind/state/cliproxy.runtime.yaml --protocol responses --tier omit --route configured_upstream --preflight-cases 1
  [Console]::Out.WriteLine((@{model=$entry.model;scope='one_synthetic_case_only';exit_code=$LASTEXITCODE;qualified=$false} | ConvertTo-Json -Compress))
  if ($entry.model -ne 'gpt-5.5') { Start-Sleep -Seconds 16 }
}
