$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Security
$target='C:\ProgramData\EvoMind\staging\ev-deepseek-v4-pro-20260908\model-key.machine-dpapi.json'
$plain=$null
$bytes=$null
try {
  if(Test-Path -LiteralPath $target){throw 'Encrypted envelope already exists'}
  $plain=[Console]::ReadLine()
  if([string]::IsNullOrWhiteSpace($plain) -or $plain.Length -lt 20){throw 'Invalid key input'}
  $bytes=[Text.Encoding]::UTF8.GetBytes($plain)
  $protected=[Security.Cryptography.ProtectedData]::Protect($bytes,$null,[Security.Cryptography.DataProtectionScope]::LocalMachine)
  $envelope=@{schema='evomind.model_key.machine_dpapi_transit.v1';ciphertext=[Convert]::ToBase64String($protected)}|ConvertTo-Json -Compress
  $stream=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
  try{$encoded=[Text.Encoding]::UTF8.GetBytes($envelope);$stream.Write($encoded,0,$encoded.Length);$stream.Flush($true)}finally{$stream.Dispose()}
  & icacls.exe $target /inheritance:r /grant:r 'SYSTEM:F' 'Administrators:F' 'EvoMindSvc:R' *> $null
  if($LASTEXITCODE -ne 0){throw 'Envelope ACL failed'}
  Write-Output '{"status":"encrypted_transit_stored","plaintext_written":false}'
}catch{[pscustomobject]@{status='failed';error_type=$_.Exception.GetType().Name}|ConvertTo-Json -Compress;exit 1}
finally{$plain=$null;if($null -ne $bytes){[Array]::Clear($bytes,0,$bytes.Length)}}
