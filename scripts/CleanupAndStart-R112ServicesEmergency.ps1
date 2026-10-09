param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{12}$')][string]$RecoveryId,
  [Parameter(Mandatory)][string]$PriorRecoveryTask
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = 'C:\ProgramData\EvoMind'
$mainTask = 'EvoMind-Cloud-Node'
$resultPath = "C:\SecureInput\r112-clean-start-$RecoveryId-result.json"

function Write-AtomicJson([string]$Path, [object]$Value) {
  $temporary = "$Path.$PID.tmp"
  [IO.File]::WriteAllText($temporary, (($Value | ConvertTo-Json -Depth 10) + "`n"), [Text.UTF8Encoding]::new($false))
  Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Owner-Of($Process) {
  [string](($Process | Invoke-CimMethod -MethodName GetOwner).User)
}

function Assert-ResidualIdentity($Process, [int]$Port = 0) {
  $owner = Owner-Of $Process
  if ($owner -ne 'EvoMindSvc') { throw "RESIDUAL_OWNER_REJECTED:$($Process.ProcessId)" }
  $name = [string]$Process.Name
  $command = [string]$Process.CommandLine
  $path = [string]$Process.ExecutablePath
  $valid = if ($Port) {
    switch ($Port) {
      65068 { $name -eq 'cliproxy-7.2.128.exe' -and $path -match 'ProgramData\\EvoMind\\bin\\cliproxy-7\.2\.128\.exe$' }
      8765 { $name -in @('python.exe','pythonw.exe') -and $command -match 'run_python_runtime\.py.+--port\s+8765' }
      8088 { $name -eq 'node.exe' -and $command -match '(?:^|\s)server\.js(?:\s|$)' }
      7890 { $name -in @('python.exe','pythonw.exe') -and $command -match 'hpc_socks_bridge\.py.+--listen-port\s+7890' }
      default { $false }
    }
  } else {
    ($name -eq 'pwsh.exe' -and $command -match 'Watch-TenantEnrollmentQueue\.ps1') -or
    ($name -eq 'powershell.exe' -and $command -match '(?:cloud-node-service-action-wrapper|Invoke-Startup)\.ps1')
  }
  if (-not $valid) { throw "RESIDUAL_IDENTITY_REJECTED:$($Process.ProcessId):$Port" }
  return [ordered]@{ pid=[int]$Process.ProcessId; port=$Port; name=$name; owner=$owner }
}

try {
  foreach ($name in @($mainTask, $PriorRecoveryTask)) {
    $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($task -and [string]$task.State -eq 'Running') { Stop-ScheduledTask -TaskName $name -ErrorAction Stop }
  }
  Start-Sleep -Seconds 2

  $verified = @()
  $pidSet = [Collections.Generic.HashSet[int]]::new()
  $listeners = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in @(65068,8765,8088,7890) })
  foreach ($listener in $listeners) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$listener.OwningProcess)" -ErrorAction Stop
    $verified += Assert-ResidualIdentity $process ([int]$listener.LocalPort)
    [void]$pidSet.Add([int]$listener.OwningProcess)
  }
  $extra = @(Get-CimInstance Win32_Process | Where-Object {
    [string]$_.CommandLine -match 'Watch-TenantEnrollmentQueue\.ps1|cloud-node-service-action-wrapper\.ps1|Invoke-Startup\.ps1'
  })
  foreach ($process in $extra) {
    $verified += Assert-ResidualIdentity $process 0
    [void]$pidSet.Add([int]$process.ProcessId)
  }
  foreach ($pidValue in @($pidSet)) { Stop-Process -Id $pidValue -Force -ErrorAction Stop }

  $deadline = [DateTime]::UtcNow.AddSeconds(30)
  do {
    Start-Sleep -Milliseconds 500
    $remainingListeners = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in @(65068,8765,8088,7890) })
    $remainingProcesses = @(Get-CimInstance Win32_Process | Where-Object {
      [string]$_.CommandLine -match 'Watch-TenantEnrollmentQueue\.ps1|cloud-node-service-action-wrapper\.ps1|Invoke-Startup\.ps1'
    })
  } while (($remainingListeners.Count -or $remainingProcesses.Count) -and [DateTime]::UtcNow -lt $deadline)
  if ($remainingListeners.Count -or $remainingProcesses.Count) { throw 'RESIDUAL_PROCESS_CLEANUP_TIMEOUT' }

  $main = Get-ScheduledTask -TaskName $mainTask -ErrorAction Stop
  if ([string]$main.State -ne 'Ready') { throw 'MAIN_TASK_NOT_READY_AFTER_CLEANUP' }
  if ([string]$main.Actions[0].Arguments -notmatch '-Action Start(?:\s|$)') { throw 'MAIN_TASK_ACTION_NOT_START' }
  Start-ScheduledTask -TaskName $mainTask
  Write-AtomicJson -Path $resultPath -Value ([ordered]@{
    schema = 'evomind.r112_clean_start_recovery.v1'
    recovery_id = $RecoveryId
    status = 'start_requested'
    requested_at_utc = [DateTime]::UtcNow.ToString('o')
    verified_processes_stopped = @($verified)
    other_processes_modified = $false
    remote_worker_modified = $false
  })
  exit 0
} catch {
  $safeError = [string]$_.Exception.Message -replace '[^A-Za-z0-9_.:-]', '_'
  if ($safeError.Length -gt 160) { $safeError = $safeError.Substring(0,160) }
  Write-AtomicJson -Path $resultPath -Value ([ordered]@{
    schema = 'evomind.r112_clean_start_recovery.v1'
    recovery_id = $RecoveryId
    status = 'failed'
    error_class = $_.Exception.GetType().Name
    error_code = $safeError
    failed_at_utc = [DateTime]::UtcNow.ToString('o')
    remote_worker_modified = $false
  })
  exit 1
}
