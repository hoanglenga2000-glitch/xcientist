$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Add-Type -AssemblyName System.Security
$stage='C:\ProgramData\EvoMind\staging\ev-deepseek-v4-pro-20260908'
$taskName='EvoMind-DeepSeek-Key-20260908'
$serviceOutput=Join-Path $stage 'service-output'
$result=Join-Path $serviceOutput 'key-install.json'
if((Test-Path -LiteralPath $result) -or (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue)){throw 'Previous installation requires reconciliation'}
New-Item -ItemType Directory -Path $serviceOutput|Out-Null
& icacls.exe $serviceOutput /inheritance:r /grant:r 'SYSTEM:(OI)(CI)F' 'Administrators:(OI)(CI)F' 'EvoMindSvc:(OI)(CI)M' *> $null
if($LASTEXITCODE -ne 0){throw 'Output ACL failed'}
$stored=Get-Content -Raw -LiteralPath 'C:\ProgramData\EvoMind\bootstrap\service-account.machine-dpapi.json'|ConvertFrom-Json
$account="$env:COMPUTERNAME\EvoMindSvc"
if($stored.schema -ne 'evomind.machine_dpapi_service_credential.v1' -or $stored.username -ine $account){throw 'Service account identity mismatch'}
$bytes=[Security.Cryptography.ProtectedData]::Unprotect([Convert]::FromBase64String($stored.protected_password),$null,[Security.Cryptography.DataProtectionScope]::LocalMachine)
$plain=$null
try{
  $plain=[Text.Encoding]::UTF8.GetString($bytes)
  $action=New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument "-NoLogo -NoProfile -NonInteractive -File `"$stage\install_ev_deepseek_service_key.ps1`"" -WorkingDirectory $stage
  $principal=New-ScheduledTaskPrincipal -UserId $account -LogonType Password -RunLevel Limited
  $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 3) -MultipleInstances IgnoreNew
  $task=New-ScheduledTask -Action $action -Principal $principal -Settings $settings
  Register-ScheduledTask -TaskName $taskName -InputObject $task -User $account -Password $plain|Out-Null
}finally{$plain=$null;[Array]::Clear($bytes,0,$bytes.Length)}
Start-ScheduledTask -TaskName $taskName
$deadline=(Get-Date).AddSeconds(45)
while((Get-Date) -lt $deadline -and -not(Test-Path -LiteralPath $result)){Start-Sleep -Seconds 1}
if(-not(Test-Path -LiteralPath $result)){throw 'Credential installation result pending; do not dispatch twice'}
$receipt=Get-Content -Raw -LiteralPath $result|ConvertFrom-Json
if($receipt.status -ne 'installed'){$receipt|ConvertTo-Json -Compress;exit 1}
$receipt|ConvertTo-Json -Compress
