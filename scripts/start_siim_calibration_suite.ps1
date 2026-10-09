param([switch]$Resume,[ValidatePattern('^[A-Za-z0-9_-]{1,40}$')][string]$ResumeKey='initial')
$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
Add-Type -AssemblyName System.Security
$stage='C:\ProgramData\EvoMind\staging\siim-mlebench-calibration-20260908\runtime-extension';$name=if($Resume){'EvoMind-SIIM-Suite-'+$ResumeKey+'-20260908'}else{'EvoMind-SIIM-Suite-20260908'}
if($Resume -and @(Get-ScheduledTask|Where-Object {$_.TaskName -like 'EvoMind-SIIM-Suite*' -and $_.State -eq 'Running'}).Count -gt 0){throw 'A suite controller is still running'}
if(Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue){throw 'Suite already exists; reconcile first'}
$output=Join-Path $stage 'service-output';if(-not(Test-Path -LiteralPath $output)){New-Item -ItemType Directory -Path $output|Out-Null}
& icacls.exe $output /inheritance:r /grant:r 'SYSTEM:(OI)(CI)F' 'Administrators:(OI)(CI)F' 'EvoMindSvc:(OI)(CI)M' *> $null
if($LASTEXITCODE -ne 0){throw 'Output ACL failed'}
$stored=Get-Content -Raw -LiteralPath 'C:\ProgramData\EvoMind\bootstrap\service-account.machine-dpapi.json'|ConvertFrom-Json
$account="$env:COMPUTERNAME\EvoMindSvc"
if($stored.schema -ne 'evomind.machine_dpapi_service_credential.v1' -or $stored.username -ine $account){throw 'Service identity mismatch'}
$bytes=[Security.Cryptography.ProtectedData]::Unprotect([Convert]::FromBase64String($stored.protected_password),$null,[Security.Cryptography.DataProtectionScope]::LocalMachine)
try{
 $plain=[Text.Encoding]::UTF8.GetString($bytes)
 $extra=if($Resume){' --resume --resume-key '+$ResumeKey}else{''}
 $action=New-ScheduledTaskAction -Execute 'C:\EvoMind\releases\664a636ddd419a66f73cc10820c0a16429784866\.venv\Scripts\python.exe' -Argument "-X utf8 `"$stage\siim_calibration_suite.py`"$extra" -WorkingDirectory $stage
 $principal=New-ScheduledTaskPrincipal -UserId $account -LogonType Password -RunLevel Limited
 # This is the application coordinator lifetime, not a GPU allowance. The
 # source-backed case/step limits and unchanged 24 GPU-hour ledger remain authoritative.
 $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 48) -MultipleInstances IgnoreNew
 Register-ScheduledTask -TaskName $name -InputObject (New-ScheduledTask -Action $action -Principal $principal -Settings $settings) -User $account -Password $plain|Out-Null
}finally{$plain=$null;[Array]::Clear($bytes,0,$bytes.Length)}
Start-ScheduledTask -TaskName $name
[ordered]@{status='suite_dispatched';task=$name;max_gpu_hours=24;max_parallel_runs=1;official_submissions_enabled=$false}|ConvertTo-Json -Compress
