$ErrorActionPreference = 'Stop'
$candidate = 'C:\ProgramData\EvoMind\staging\system-completeness-20260907\endurance-candidate-v2'
$expected = 'cbbcd3584aa3457c3952a01d10d9098edc643d90eefa12520594a601c9656a49'
$manifest = Join-Path $candidate 'candidate-manifest.json'
if ((Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant() -cne $expected) { throw 'candidate_manifest_mismatch' }
$python = 'C:\EvoMind\releases\664a636ddd419a66f73cc10820c0a16429784866\.venv\Scripts\python.exe'
$qaRoot = 'C:\EMQA'
$receipt = Join-Path $qaRoot 'sys0907-endurance-launch-v1.json'
if (Test-Path -LiteralPath $receipt) { throw 'launch_receipt_exists_do_not_duplicate' }
New-Item -ItemType Directory -Path $qaRoot -Force | Out-Null
$env:PYTHONPATH = $candidate + ';C:\EvoMind\releases\664a636ddd419a66f73cc10820c0a16429784866\src'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$records = @()
foreach ($entry in @(@{model='gpt-6-astra';name='astra'}, @{model='gpt-5.6-sol';name='sol'})) {
  $output = Join-Path $qaRoot ('sys0907-' + $entry.name + '-v1')
  if (Test-Path -LiteralPath $output) { throw 'endurance_output_exists_do_not_duplicate' }
  $arguments = @('-X','utf8',(Join-Path $candidate 'run_model_endurance_acceptance.py'),'--model',$entry.model,'--seconds','5400','--root',$output,'--gateway-config','C:\ProgramData\EvoMind\state\cliproxy.runtime.yaml','--candidate-manifest',$manifest)
  $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $candidate -WindowStyle Hidden -RedirectStandardOutput ($output + '.stdout.log') -RedirectStandardError ($output + '.stderr.log') -PassThru
  $records += [ordered]@{model=$entry.model;launcher_pid=$process.Id;root=$output;candidate_manifest_sha256=$expected;started_at_utc=[DateTime]::UtcNow.ToString('o');requested_seconds=5400;gpu_actions=0}
  $payload = [ordered]@{schema='evomind.two_model_endurance_launch.v1';production_changed=$false;runs=$records} | ConvertTo-Json -Depth 5
  [IO.File]::WriteAllText($receipt,$payload,[Text.UTF8Encoding]::new($false))
}
Write-Output $payload
