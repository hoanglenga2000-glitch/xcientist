param(
    [ValidateSet('freeze','grade')][string]$Mode='freeze',
    [ValidatePattern('^[A-Za-z0-9-]+$')][string]$Namespace
)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Add-Type -AssemblyName System.Security
$stage='C:\ProgramData\EvoMind\staging\siim-mlebench-calibration-20260908\runtime-extension'
if(-not $Namespace){$Namespace=$Mode+'-'+[DateTime]::UtcNow.ToString('yyyyMMddHHmmss')}
$name='EvoMind-SIIM-PrivateFreeze-'+$Namespace
if(Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue){throw 'Existing task requires reconciliation'}
$output=Join-Path $stage 'service-output'
if(-not(Test-Path -LiteralPath $output)){New-Item -ItemType Directory -Path $output|Out-Null}
& icacls.exe $output /inheritance:r /grant:r 'SYSTEM:(OI)(CI)F' 'Administrators:(OI)(CI)F' 'EvoMindSvc:(OI)(CI)M' *> $null
if($LASTEXITCODE -ne 0){throw 'Output ACL failed'}
$stored=Get-Content -Raw -LiteralPath 'C:\ProgramData\EvoMind\bootstrap\service-account.machine-dpapi.json'|ConvertFrom-Json
$account="$env:COMPUTERNAME\EvoMindSvc"
if($stored.schema -ne 'evomind.machine_dpapi_service_credential.v1' -or $stored.username -ine $account){throw 'Service identity mismatch'}
$bytes=[Security.Cryptography.ProtectedData]::Unprotect([Convert]::FromBase64String($stored.protected_password),$null,[Security.Cryptography.DataProtectionScope]::LocalMachine)
try{
  $plain=[Text.Encoding]::UTF8.GetString($bytes)
  $script=Join-Path $stage 'siim_private_freeze.py'
  $action=New-ScheduledTaskAction -Execute 'C:\EvoMind\releases\664a636ddd419a66f73cc10820c0a16429784866\.venv\Scripts\python.exe' -Argument "-X utf8 `"$script`" --namespace $Namespace --mode $Mode" -WorkingDirectory $stage
  $principal=New-ScheduledTaskPrincipal -UserId $account -LogonType Password -RunLevel Limited
  $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 40) -MultipleInstances IgnoreNew
  Register-ScheduledTask -TaskName $name -InputObject (New-ScheduledTask -Action $action -Principal $principal -Settings $settings) -User $account -Password $plain|Out-Null
}finally{$plain=$null;[Array]::Clear($bytes,0,$bytes.Length)}
Start-ScheduledTask -TaskName $name
[ordered]@{status='dispatched';task=$name;mode=$Mode;namespace=$Namespace;kaggle_submission_made=$false}|ConvertTo-Json -Compress
