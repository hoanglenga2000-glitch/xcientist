# F2: bounded functional probe for the local cliproxy llm-gateway.
# No paid model calls and no secrets: verifies listener, HTTP layer and that the
# region-blocked OAuth refresh loop is gone. Writes state\llm-gateway-health.json.
param(
  [string]$StateRoot = 'C:\ProgramData\EvoMind\state',
  [string]$LogsRoot = 'C:\ProgramData\EvoMind\logs'
)
$ErrorActionPreference = 'Continue'
$now = (Get-Date).ToUniversalTime()

$listener = Get-NetTCPConnection -State Listen -LocalPort 65068 -ErrorAction SilentlyContinue | Select-Object -First 1
$portListening = [bool]$listener
$ownerPid = 0
$processName = ''
if ($listener) {
  $ownerPid = [int]$listener.OwningProcess
  $proc = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
  if ($proc) { $processName = [string]$proc.ProcessName }
}

$httpStatus = 0
try {
  $resp = Invoke-WebRequest -Uri 'http://127.0.0.1:65068/v1/models' -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
  $httpStatus = [int]$resp.StatusCode
} catch {
  if ($_.Exception.Response) {
    try { $httpStatus = [int]$_.Exception.Response.StatusCode } catch { $httpStatus = 0 }
  }
}
$httpLayerOk = @(200, 401, 403) -contains $httpStatus

$logPath = Join-Path $LogsRoot 'llm-gateway.out.log'
$refreshFailures = 0
$logTailLines = 0
if (Test-Path -LiteralPath $logPath) {
  $lines = @(Get-Content -LiteralPath $logPath -Tail 600 -ErrorAction SilentlyContinue)
  $logTailLines = $lines.Count
  foreach ($line in $lines) {
    if ($line -match 'Token refresh attempt \d+ failed') {
      if ($line -match '^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]') {
        $ts = [datetime]::ParseExact($Matches[1], 'yyyy-MM-dd HH:mm:ss', [Globalization.CultureInfo]::InvariantCulture)
        $ageMinutes = ($now.ToLocalTime() - $ts).TotalMinutes
        if ($ageMinutes -le 20) { $refreshFailures++ }
      }
    }
  }
}

$ok = ($portListening -and $httpLayerOk -and ($refreshFailures -eq 0))
if (-not $portListening) { $detail = 'gateway_port_not_listening' }
elseif ($refreshFailures -gt 0) { $detail = 'oauth_refresh_loop' }
elseif (-not $httpLayerOk) { $detail = 'http_layer_unreachable' }
else { $detail = 'ok' }

$payload = [pscustomobject]@{
  schema                 = 'evomind.llm_gateway_health.v1'
  ok                     = $ok
  detail                 = $detail
  checked_at_utc         = $now.ToString('o')
  evidence               = [pscustomobject]@{
    port_65068_listening       = $portListening
    owner_pid                  = $ownerPid
    process_name               = $processName
    http_status                = $httpStatus
    http_layer_ok              = $httpLayerOk
    refresh_failures_last_20m  = $refreshFailures
    log_tail_lines             = $logTailLines
    log_path                   = $logPath
  }
}

$target = Join-Path $StateRoot 'llm-gateway-health.json'
[IO.File]::WriteAllText($target, ($payload | ConvertTo-Json -Depth 6) + "`n", [Text.UTF8Encoding]::new($false))
$payload | ConvertTo-Json -Depth 6 -Compress
if ($ok) { exit 0 } else { exit 1 }
