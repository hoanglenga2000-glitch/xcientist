# F4: node watchdog. Verifies every layer of the EvoMind node, writes fresh
# evidence, and (with -Repair) restarts the node through the same scheduled
# task path the boot trigger uses. Skips repair while a deployment holds the
# Global\EvoMind-Byoa-V12-Deployment mutex or while state\node-maintenance.flag
# exists. ASCII only: the node host is a GBK console.
param(
  [switch]$Repair,
  [string]$StateRoot = 'C:\ProgramData\EvoMind\state',
  [string]$LogsRoot = 'C:\ProgramData\EvoMind\logs',
  [string]$TaskName = 'EvoMind-Cloud-Node'
)
$ErrorActionPreference = 'Continue'
$now = (Get-Date).ToUniversalTime()

function Get-ListenerOwnerPid {
  param([int]$Port)
  $conn = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($conn) { return [int]$conn.OwningProcess }
  return 0
}

# Records a repair attempt and returns $true, or returns $false when 3 attempts
# already happened in the last 30 minutes (backoff window).
function Register-RepairAttempt {
  $path = Join-Path $StateRoot 'watchdog-repair-attempts.json'
  $recent = @()
  if (Test-Path -LiteralPath $path) {
    try {
      $doc = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
      foreach ($item in @($doc.attempts_utc)) {
        # PS 7 converts ISO strings to DateTime on read; PS 5.1 keeps strings.
        if ($item -is [DateTime]) { $at = $item.ToUniversalTime() }
        else { $at = [DateTime]::Parse([string]$item, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime() }
        if (($now - $at).TotalMinutes -lt 30) { $recent += $at.ToString('o') }
      }
    } catch { $recent = @() }
  }
  if ($recent.Count -ge 3) { return $false }
  $recent += $now.ToString('o')
  $json = [pscustomobject]@{ schema = 'evomind.watchdog_repair_attempts.v1'; attempts_utc = @($recent) } | ConvertTo-Json
  [IO.File]::WriteAllText($path, $json + "`n", [Text.UTF8Encoding]::new($false))
  return $true
}

$portMap = @(
  [pscustomobject]@{ role = 'caddy_http'; port = 80 },
  [pscustomobject]@{ role = 'caddy_https'; port = 443 },
  [pscustomobject]@{ role = 'web'; port = 8088 },
  [pscustomobject]@{ role = 'runtime'; port = 8765 },
  [pscustomobject]@{ role = 'llm_gateway'; port = 65068 },
  [pscustomobject]@{ role = 'hpc_socks'; port = 7890 }
)
$listeners = [pscustomobject]@{}
$missing = @()
$portByRole = @{}
foreach ($item in $portMap) {
  $ownerPid = Get-ListenerOwnerPid $item.port
  $portByRole[$item.role] = $ownerPid
  $listeners | Add-Member -NotePropertyName $item.role -NotePropertyValue ([pscustomobject]@{ port = $item.port; pid = $ownerPid }) -Force
  if ($ownerPid -le 0) { $missing += $item.role }
}

$healthJson = [pscustomobject]@{ status_code = 0; ok = $false; status = ''; build_id = '' }
try {
  $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8088/api/healthz' -UseBasicParsing -TimeoutSec 15 -ErrorAction Stop
  $payload = $r.Content | ConvertFrom-Json
  $healthJson = [pscustomobject]@{
    status_code = [int]$r.StatusCode
    ok          = [bool]$payload.ok
    status      = [string]$payload.status
    build_id    = [string]$payload.build_id
  }
} catch {
  if ($_.Exception.Response) {
    try { $healthJson.status_code = [int]$_.Exception.Response.StatusCode } catch { }
  }
}

$publicProbe = [pscustomobject]@{ status_code = 0; ok = $false }
try {
  $r2 = Invoke-WebRequest -Uri 'https://evomind.zhjjq.tech/api/healthz' -UseBasicParsing -TimeoutSec 20 -ErrorAction Stop
  $payload2 = $r2.Content | ConvertFrom-Json
  $publicProbe = [pscustomobject]@{
    status_code = [int]$r2.StatusCode
    ok          = ([bool]$payload2.ok -and [string]$payload2.status -eq 'ready')
  }
} catch { }

$edgeService = Get-Service -Name 'EvoMind-PublicEdge' -ErrorAction SilentlyContinue
$edgeStatus = 'missing'
if ($edgeService) { $edgeStatus = [string]$edgeService.Status }

$gatewayOk = $null
$gatewayDetail = ''
$gatewayPath = Join-Path $StateRoot 'llm-gateway-health.json'
if (Test-Path -LiteralPath $gatewayPath) {
  try {
    $gw = Get-Content -LiteralPath $gatewayPath -Raw | ConvertFrom-Json
    $gatewayOk = [bool]$gw.ok
    $gatewayDetail = [string]$gw.detail
  } catch { }
}

$coreOk = $false
if ($healthJson.ok -and $healthJson.status -eq 'ready' -and
    $portByRole['web'] -gt 0 -and $portByRole['runtime'] -gt 0 -and $portByRole['caddy_https'] -gt 0) {
  $coreOk = $true
}

# Renew the trusted QPC sample without interrupting active work.
# The immutable launch anchor and its environment binding stay unchanged.
$anchorPath = Join-Path $StateRoot 'service-clock-anchor.json'
$clockRenewal = $null
try {
  $renewalHelper = Join-Path $PSScriptRoot 'Renew-EvoMindServiceClock.ps1'
  $clockRenewal = & $renewalHelper -AnchorPath $anchorPath -Apply:$Repair
} catch { $clockRenewal = [pscustomobject]@{ status='renewal_rejected'; renewed=$false } }
$anchorOk = $false
$anchorAgeHours = -1
if (Test-Path -LiteralPath $anchorPath) {
  try {
    $anchorDoc = Get-Content -LiteralPath $anchorPath -Raw | ConvertFrom-Json
    $clockSample = $anchorDoc
    if ($anchorDoc.renewal) { $clockSample = $anchorDoc.renewal }
    $anchorCreated = [DateTimeOffset]::Parse([string]$clockSample.created_at_utc)
    $anchorAgeHours = [math]::Round(([DateTimeOffset]::UtcNow - $anchorCreated).TotalHours, 2)
    if ($clockRenewal -and $clockRenewal.status -in @('fresh','renewed','renewal_due') -and $anchorDoc.schema -eq 'evomind.service_clock_anchor.v1' -and $anchorAgeHours -ge 0 -and $anchorAgeHours -lt 20) {
      $anchorOk = $true
    }
  } catch {
    $anchorOk = $false
  }
}

$state = [pscustomobject]@{
  schema           = 'evomind.node_health.v1'
  checked_at_utc   = $now.ToString('o')
  core_ok          = $coreOk
  missing_roles    = $missing
  healthz          = $healthJson
  public_healthz   = $publicProbe
  edge_service     = $edgeStatus
  llm_gateway      = [pscustomobject]@{ ok = $gatewayOk; detail = $gatewayDetail }
  service_clock_anchor = [pscustomobject]@{ ok = $anchorOk; age_hours = $anchorAgeHours; path = $anchorPath }
  repair_requested = [bool]$Repair
  repair_outcome   = 'not_attempted'
  task_name        = $TaskName
}

$listenersDoc = [pscustomobject]@{
  schema           = 'evomind.node_listeners.v1'
  observed_at_utc  = $now.ToString('o')
  listeners        = $listeners
}

$repairOutcome = if ($clockRenewal -and $clockRenewal.renewed) { 'clock_renewed_without_restart' } else { 'not_attempted' }
if ((-not $coreOk) -and $Repair) {
  $maintenanceFlag = Join-Path $StateRoot 'node-maintenance.flag'
  $mutexHeld = $true
  try {
    $mutex = New-Object System.Threading.Mutex($false, 'Global\EvoMind-Byoa-V12-Deployment')
    $watchdogHasLock = $false
    try {
      try { $watchdogHasLock = $mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $watchdogHasLock = $true }
      $mutexHeld = -not $watchdogHasLock
    } finally { if ($watchdogHasLock) { $mutex.ReleaseMutex() }; $mutex.Dispose() }
  } catch { }
  $flagAgeMinutes = -1
  if (Test-Path -LiteralPath $maintenanceFlag) {
    try { $flagAgeMinutes = ((Get-Date) - (Get-Item -LiteralPath $maintenanceFlag).LastWriteTime).TotalMinutes } catch { $flagAgeMinutes = 0 }
  }
  if ($flagAgeMinutes -ge 0 -and $flagAgeMinutes -lt 30) {
    $repairOutcome = 'skipped_maintenance_flag'
  } elseif ($mutexHeld) {
    $repairOutcome = 'skipped_deployment_in_progress'
  } elseif (-not (Register-RepairAttempt)) {
    # Backoff: more than 3 repairs within 30 minutes means restarts are not helping.
    $repairOutcome = 'backoff_alert_only'
    $alertLine = '{0} watchdog backoff: 3 repairs in last 30 min and core still down; manual action required' -f $now.ToString('o')
    Add-Content -LiteralPath (Join-Path $LogsRoot 'watchdog-alerts.log') -Value $alertLine -Encoding ascii
  } else {
    try {
      & 'C:/SecureInput/Invoke-ServiceAccountAction.ps1' -Action Start | Out-Null
      $deadline = (Get-Date).AddSeconds(240)
      $recovered = $false
      while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        try {
          $r3 = Invoke-WebRequest -Uri 'http://127.0.0.1:8088/api/healthz' -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
          $p3 = $r3.Content | ConvertFrom-Json
          if ([bool]$p3.ok -and [string]$p3.status -eq 'ready') { $recovered = $true; break }
        } catch { }
      }
      if ($recovered) { $repairOutcome = 'restart_ok' } else { $repairOutcome = 'restart_timeout' }
    } catch {
      $repairOutcome = 'restart_failed'
    }
  }
}
$state.repair_outcome = $repairOutcome

[IO.File]::WriteAllText((Join-Path $StateRoot 'node-health-last.json'), ($state | ConvertTo-Json -Depth 8) + "`n", [Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText((Join-Path $StateRoot 'node-listeners.json'), ($listenersDoc | ConvertTo-Json -Depth 6) + "`n", [Text.UTF8Encoding]::new($false))

$summary = [pscustomobject]@{
  core_ok        = $coreOk
  anchor_ok      = $anchorOk
  anchor_age_hours = $anchorAgeHours
  missing_roles  = ($missing -join ',')
  healthz        = $healthJson.status
  public_status  = $publicProbe.status_code
  edge_service   = $edgeStatus
  gateway_ok     = $gatewayOk
  repair_outcome = $repairOutcome
}
$summary | ConvertTo-Json -Compress
if ($coreOk -and $anchorOk) { exit 0 } else { exit 1 }
