$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Security
$target='C:\ProgramData\EvoMind\staging\ev-public-calibration-20260908\kaggle-key.machine-dpapi.json'
$plain=$null;$bytes=$null
try{
  if(Test-Path -LiteralPath $target){throw 'Envelope exists; reconcile first'}
  $plain=[Console]::ReadLine()
  if([string]::IsNullOrWhiteSpace($plain) -or $plain.Length -lt 20){throw 'Input invalid'}
  $bytes=[Text.Encoding]::UTF8.GetBytes($plain)
  $protected=[Security.Cryptography.ProtectedData]::Protect($bytes,$null,[Security.Cryptography.DataProtectionScope]::LocalMachine)
  $body=@{schema='evomind.ev_kaggle.machine_dpapi_transit.v1';ciphertext=[Convert]::ToBase64String($protected)}|ConvertTo-Json -Compress
  $f=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
  try{$buffer=[Text.Encoding]::UTF8.GetBytes($body);$f.Write($buffer,0,$buffer.Length);$f.Flush($true)}finally{$f.Dispose()}
  & icacls.exe $target /inheritance:r /grant:r 'SYSTEM:F' 'Administrators:F' 'EvoMindSvc:R' *> $null
  if($LASTEXITCODE -ne 0){throw 'ACL failed'}
  Write-Output '{"status":"encrypted_transit_stored","plaintext_written":false}'
}catch{[pscustomobject]@{status='failed';error_type=$_.Exception.GetType().Name}|ConvertTo-Json -Compress;exit 1}
finally{$plain=$null;if($null -ne $bytes){[Array]::Clear($bytes,0,$bytes.Length)}}
