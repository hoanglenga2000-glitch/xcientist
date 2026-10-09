$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
Add-Type -AssemblyName System.Security
$stage='C:\ProgramData\EvoMind\staging\ev-public-calibration-20260908';$name='EvoMind-EV-Data-20260908'
if(Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue){throw 'Task exists; reconcile first'}
$stored=Get-Content -Raw -LiteralPath 'C:\ProgramData\EvoMind\bootstrap\service-account.machine-dpapi.json'|ConvertFrom-Json
$account="$env:COMPUTERNAME\EvoMindSvc"
if($stored.schema -ne 'evomind.machine_dpapi_service_credential.v1' -or $stored.username -ine $account){throw 'Service identity mismatch'}
$bytes=[Security.Cryptography.ProtectedData]::Unprotect([Convert]::FromBase64String($stored.protected_password),$null,[Security.Cryptography.DataProtectionScope]::LocalMachine)
try{
 $plain=[Text.Encoding]::UTF8.GetString($bytes)
 $action=New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument "-NoLogo -NoProfile -NonInteractive -File `"$stage\ev_calibration_service_worker.ps1`" -Mode data" -WorkingDirectory $stage
 $principal=New-ScheduledTaskPrincipal -UserId $account -LogonType Password -RunLevel Limited
 $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 35) -MultipleInstances IgnoreNew
 Register-ScheduledTask -TaskName $name -InputObject (New-ScheduledTask -Action $action -Principal $principal -Settings $settings) -User $account -Password $plain|Out-Null
}finally{$plain=$null;[Array]::Clear($bytes,0,$bytes.Length)}
Start-ScheduledTask -TaskName $name
[ordered]@{status='data_preparation_dispatched';task=$name;training_started=$false;gpu_training_hours=0}|ConvertTo-Json -Compress
