[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$TaskName,
    [Parameter(Mandatory = $true)][string]$Manifest,
    [string]$Description = 'EvoMind persistent process',
    [int]$ExecutionTimeLimitDays = 10,
    [switch]$Start
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$wrapper = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'run_persistent_process_from_manifest.ps1')).Path
$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$spec = Get-Content -Raw -LiteralPath $manifestPath -Encoding utf8 | ConvertFrom-Json
if ($spec.schema -ne 'evomind.persistent_process_manifest.v1') {
    throw 'Unexpected persistent-process manifest schema'
}
if ($spec.process_signals_allowed -ne $false) {
    throw 'Persistent-process manifest permits process signals'
}
foreach ($binding in @($spec.artifact_bindings)) {
    $path = [string]$binding.path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Bound artifact is missing: $path"
    }
    if ((Get-Sha256 -Path $path) -ne [string]$binding.sha256) {
        throw "Bound artifact hash drifted: $path"
    }
}

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $existing -and $existing.State -eq 'Running') {
    throw "Scheduled task is already running: $TaskName"
}
$pwsh = (Get-Process -Id $PID).Path
$arguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}" -Manifest "{1}"' -f $wrapper, $manifestPath
$action = New-ScheduledTaskAction -Execute $pwsh -Argument $arguments -WorkingDirectory $projectRoot
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Days $ExecutionTimeLimitDays) `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd `
    -StartWhenAvailable `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Principal $principal `
    -Settings $settings `
    -Description $Description `
    -Force | Out-Null
if ($Start) {
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 2
}
$task = Get-ScheduledTask -TaskName $TaskName
$info = Get-ScheduledTaskInfo -TaskName $TaskName
$registrationDirectory = Join-Path $projectRoot 'workspace\local_gpu\persistent_task_registrations'
New-Item -ItemType Directory -Force -Path $registrationDirectory | Out-Null
$safeName = $TaskName -replace '[^A-Za-z0-9._-]', '_'
$registrationPath = Join-Path $registrationDirectory "$safeName.json"
$payload = [ordered]@{
    schema = 'evomind.persistent_task_registration.v1'
    created_at = (Get-Date -Format o)
    task_name = $TaskName
    task_state = [string]$task.State
    task_path = [string]$task.TaskPath
    task_id = [string]$spec.task_id
    manifest_path = $manifestPath
    manifest_sha256 = Get-Sha256 -Path $manifestPath
    wrapper_path = $wrapper
    wrapper_sha256 = Get-Sha256 -Path $wrapper
    executable = [string]$spec.executable
    last_run_time = $info.LastRunTime.ToString('o')
    last_task_result = $info.LastTaskResult
    interactive_user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    process_signals_sent = 0
}
[System.IO.File]::WriteAllText(
    $registrationPath,
    ($payload | ConvertTo-Json -Depth 20) + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)
$payload | ConvertTo-Json -Depth 20
