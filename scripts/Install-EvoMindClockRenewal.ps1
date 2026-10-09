param(
  [Parameter(Mandatory=$true)][string]$StageRoot,
  [Parameter(Mandatory=$true)][string]$ExpectedWatchdogSha,
  [Parameter(Mandatory=$true)][string]$ExpectedCandidateSha,
  [Parameter(Mandatory=$true)][string]$ExpectedHelperSha,
  [Parameter(Mandatory=$true)][string]$ExpectedBuild
)
$ErrorActionPreference = 'Stop'
$maintenanceRoot = 'C:\ProgramData\EvoMind\maintenance'
$target = Join-Path $maintenanceRoot 'Watch-EvoMindNode.ps1'
$candidate = Join-Path $StageRoot 'Watch-EvoMindNode.candidate.ps1'
$helper = Join-Path $StageRoot 'Renew-EvoMindServiceClock.ps1'
if ((Get-FileHash -LiteralPath $target).Hash -ne $ExpectedWatchdogSha -or
    (Get-FileHash -LiteralPath $candidate).Hash -ne $ExpectedCandidateSha -or
    (Get-FileHash -LiteralPath $helper).Hash -ne $ExpectedHelperSha) { throw 'CLOCK_PATCH_HASH_MISMATCH' }
foreach ($file in @($candidate,$helper)) {
  $parseErrors=$null; $parseTokens=$null
  [System.Management.Automation.Language.Parser]::ParseFile($file,[ref]$parseTokens,[ref]$parseErrors) | Out-Null
  if ($parseErrors.Count) { throw 'CLOCK_PATCH_PARSE_FAILED' }
}
$health = Invoke-RestMethod -Uri 'http://127.0.0.1:8088/api/healthz' -TimeoutSec 10
if ($health.status -ne 'ready' -or $health.build_id -ne $ExpectedBuild) { throw 'CLOCK_CONSUMER_BUILD_NOT_ACTIVE' }
$scheduler = New-Object -ComObject 'Schedule.Service'
$scheduler.Connect()
$watchdogTasks = @()
foreach ($task in $scheduler.GetFolder('\').GetTasks(1)) {
  foreach ($action in $task.Definition.Actions) {
    if ([string]$action.Arguments -like '*Watch-EvoMindNode.ps1*' -and [string]$action.Arguments -like '*-Repair*' -and $task.Enabled) {
      $watchdogTasks += [pscustomobject]@{ name=$task.Name; enabled=[bool]$task.Enabled; next_run=$task.NextRunTime.ToUniversalTime().ToString('o') }
    }
  }
}
if ($watchdogTasks.Count -ne 1) { throw 'RECURRING_WATCHDOG_NOT_UNAMBIGUOUS' }
$guard = New-Object System.Threading.Mutex($false,'Global\EvoMind-Byoa-V12-Deployment')
$held = $false
try {
  $held = $guard.WaitOne(0)
  if (-not $held) { throw 'DEPLOYMENT_IN_PROGRESS' }
  if ((Get-FileHash -LiteralPath $target).Hash -ne $ExpectedWatchdogSha) { throw 'WATCHDOG_CHANGED_UNDER_LOCK' }
  $backupRoot = Join-Path 'C:\ProgramData\EvoMind\backups' ('clock-renewal-' + [guid]::NewGuid().ToString('N'))
  New-Item -ItemType Directory -Path $backupRoot | Out-Null
  Copy-Item -LiteralPath $target -Destination (Join-Path $backupRoot 'Watch-EvoMindNode.ps1')
  Copy-Item -LiteralPath 'C:\ProgramData\EvoMind\state\service-clock-anchor.json' -Destination (Join-Path $backupRoot 'service-clock-anchor.json')
  $helperTarget = Join-Path $maintenanceRoot 'Renew-EvoMindServiceClock.ps1'
  if (Test-Path -LiteralPath $helperTarget) { throw 'RENEWAL_HELPER_ALREADY_PRESENT' }
  Copy-Item -LiteralPath $helper -Destination $helperTarget
  Set-Acl -LiteralPath $helperTarget -AclObject (Get-Acl -LiteralPath $target)
  $temporary = $target + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
  Copy-Item -LiteralPath $candidate -Destination $temporary
  Set-Acl -LiteralPath $temporary -AclObject (Get-Acl -LiteralPath $target)
  [IO.File]::Replace($temporary,$target,(Join-Path $backupRoot 'atomic-watchdog-backup.ps1'))
  $beforePids = @(Get-NetTCPConnection -State Listen -LocalPort 8088,8765 | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique)
  $renewal = & $helperTarget -Apply -MinimumAgeHours 0
  if ($renewal.status -ne 'renewed') { throw 'LIVE_CLOCK_RENEWAL_FAILED' }
  $clockText = & 'C:/Program Files/nodejs/node.exe' (Join-Path $StageRoot 'Verify-EvoMindServiceClock.cjs') $ExpectedBuild
  if ($LASTEXITCODE -ne 0) { throw 'LIVE_CLOCK_CONSUMER_REJECTED' }
  $clock = $clockText | ConvertFrom-Json
  if ($clock.status -ne 'passed' -or $clock.time_source -ne 'managed_anchor') { throw 'LIVE_CLOCK_CONSUMER_REJECTED' }
  $afterPids = @(Get-NetTCPConnection -State Listen -LocalPort 8088,8765 | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique)
  if (($beforePids -join ',') -ne ($afterPids -join ',')) { throw 'CLOCK_RENEWAL_INTERRUPTED_SERVICE' }
  $receipt = [ordered]@{schema='evomind.clock_renewal_installation.v1';status='passed';build_id=$ExpectedBuild;backup=$backupRoot;watchdog_sha256=(Get-FileHash -LiteralPath $target).Hash;helper_sha256=(Get-FileHash -LiteralPath $helperTarget).Hash;renewal=$renewal;clock=$clock;listener_pids_unchanged=$true;listener_pids=$afterPids;scheduled_watchdog=$watchdogTasks}
  [IO.File]::WriteAllText((Join-Path $StageRoot 'clock-renewal-installation.json'),($receipt|ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
  $receipt | ConvertTo-Json -Depth 6 -Compress
} finally { if($held){$guard.ReleaseMutex()};$guard.Dispose() }
