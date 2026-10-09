param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{12}$')][string]$RecoveryId,
  [Parameter(Mandatory)][string]$PriorRecoveryTask
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = 'C:\ProgramData\EvoMind'
$mainTask = 'EvoMind-Cloud-Node'
$resultPath = "C:\SecureInput\r112-emergency-$RecoveryId-result.json"
$configPath = Join-Path $root 'config\node-config.json'
$backupRoot = Join-Path $root "backups\emergency-r112-$RecoveryId"

function Sha([string]$Path) {
  (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-AtomicJson([string]$Path, [object]$Value) {
  $temporary = "$Path.$PID.tmp"
  [IO.File]::WriteAllText(
    $temporary,
    (($Value | ConvertTo-Json -Depth 12) + "`n"),
    [Text.UTF8Encoding]::new($false)
  )
  Move-Item -LiteralPath $temporary -Destination $Path -Force
}

try {
  foreach ($name in @($mainTask, $PriorRecoveryTask)) {
    $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($task -and [string]$task.State -eq 'Running') {
      Stop-ScheduledTask -TaskName $name -ErrorAction Stop
    }
  }
  $deadline = [DateTime]::UtcNow.AddSeconds(30)
  do {
    $running = @(@($mainTask, $PriorRecoveryTask) | Where-Object {
      $task = Get-ScheduledTask -TaskName $_ -ErrorAction SilentlyContinue
      $task -and [string]$task.State -eq 'Running'
    })
    if (-not $running.Count) { break }
    Start-Sleep -Milliseconds 500
  } while ([DateTime]::UtcNow -lt $deadline)
  if ($running.Count) { throw 'RECOVERY_TASKS_DID_NOT_STOP' }

  if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw 'NODE_CONFIG_MISSING'
  }
  New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
  $preHash = Sha $configPath
  Copy-Item -LiteralPath $configPath -Destination (Join-Path $backupRoot 'node-config.json') -Force
  $config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
  $priorHpcState = [string]$config.hpc.state
  if ($priorHpcState -notin @('active', 'blocked')) { throw 'HPC_STATE_UNEXPECTED' }
  if ($priorHpcState -eq 'active') {
    $config.hpc.state = 'blocked'
    $config.hpc.reason = "emergency_r112_service_recovery_$RecoveryId"
    Write-AtomicJson -Path $configPath -Value $config
  }
  $postHash = Sha $configPath
  Write-AtomicJson -Path (Join-Path $backupRoot 'recovery-manifest.json') -Value ([ordered]@{
    schema = 'evomind.r112_emergency_service_recovery.v1'
    recovery_id = $RecoveryId
    created_at_utc = [DateTime]::UtcNow.ToString('o')
    prior_hpc_state = $priorHpcState
    pre_config_sha256 = $preHash
    post_config_sha256 = $postHash
    profile_modified = $false
    remote_worker_modified = $false
  })

  $main = Get-ScheduledTask -TaskName $mainTask -ErrorAction Stop
  if ([string]$main.State -ne 'Ready') { throw 'MAIN_TASK_NOT_READY' }
  if ([string]$main.Actions[0].Arguments -notmatch '-Action Start(?:\s|$)') {
    throw 'MAIN_TASK_ACTION_NOT_START'
  }
  Start-ScheduledTask -TaskName $mainTask
  Write-AtomicJson -Path $resultPath -Value ([ordered]@{
    schema = 'evomind.r112_emergency_service_recovery_result.v1'
    recovery_id = $RecoveryId
    status = 'start_requested'
    requested_at_utc = [DateTime]::UtcNow.ToString('o')
    prior_hpc_state = $priorHpcState
    current_hpc_state = 'blocked'
    profile_modified = $false
    remote_worker_modified = $false
  })
  exit 0
} catch {
  $safeError = [string]$_.Exception.Message -replace '[^A-Za-z0-9_.:-]', '_'
  if ($safeError.Length -gt 160) { $safeError = $safeError.Substring(0, 160) }
  Write-AtomicJson -Path $resultPath -Value ([ordered]@{
    schema = 'evomind.r112_emergency_service_recovery_result.v1'
    recovery_id = $RecoveryId
    status = 'failed'
    error_class = $_.Exception.GetType().Name
    error_code = $safeError
    failed_at_utc = [DateTime]::UtcNow.ToString('o')
    profile_modified = $false
    remote_worker_modified = $false
  })
  exit 1
}
