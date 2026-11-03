# F2: bounded functional probe for the local cliproxy llm-gateway, plus (2026-11-03)
# the runtime's authenticated per-provider model probe (/v1/health/models). A 401/403
# from a model upstream is a failure; the runtime caches the paid probe for 15 minutes.
# No secret is printed. Writes state\llm-gateway-health.json.
param(
  [string]$StateRoot = 'C:\ProgramData\EvoMind\state',
  [string]$LogsRoot = 'C:\ProgramData\EvoMind\logs',
  [string]$RuntimeTokenPath = 'C:\ProgramData\EvoMind\data\workspace\runtime\runtime.token'
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
# The local gateway answers 401 without a client key: that only proves its HTTP layer is up.
$httpLayerOk = @(200, 401, 403) -contains $httpStatus

$modelsOk = $false
$modelsDetail = 'probe_unavailable'
$modelProviders = @()
try {
  $runtimeToken = (Get-Content -LiteralPath $RuntimeTokenPath -Raw -ErrorAction Stop).Trim()
  $probe = $null
  try {
    $probe = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/v1/health/models' -Headers @{ Authorization = "Bearer $runtimeToken" } -TimeoutSec 120 -ErrorAction Stop
  } catch {
    if ($_.ErrorDetails -and $_.ErrorDetails.Message) { $probe = $_.ErrorDetails.Message | ConvertFrom-Json }
  }
  $runtimeToken = $null
  if ($probe) {
    $modelsOk = [bool]$probe.ok
    $modelProviders = @($probe.providers | ForEach-Object { [pscustomobject]@{ provider = $_.provider; model = $_.model; ok = [bool]$_.ok; state = $_.state; identity_ok = $_.identity_ok } })
    $failed = @($modelProviders | Where-Object { -not $_.ok } | ForEach-Object { "$($_.provider):$($_.state)" })
    $modelsDetail = if ($modelsOk) { 'ok' } elseif ($failed.Count) { 'model_provider_failed ' + ($failed -join ',') } else { 'no_model_provider' }
  }
} catch { $modelsDetail = 'runtime_token_unreadable' }

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

$ok = ($portListening -and $httpLayerOk -and ($refreshFailures -eq 0) -and $modelsOk)
if (-not $portListening) { $detail = 'gateway_port_not_listening' }
elseif ($refreshFailures -gt 0) { $detail = 'oauth_refresh_loop' }
elseif (-not $httpLayerOk) { $detail = 'http_layer_unreachable' }
elseif (-not $modelsOk) { $detail = $modelsDetail }
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
    model_providers            = $modelProviders
  }
}

$target = Join-Path $StateRoot 'llm-gateway-health.json'
[IO.File]::WriteAllText($target, ($payload | ConvertTo-Json -Depth 6) + "`n", [Text.UTF8Encoding]::new($false))
$payload | ConvertTo-Json -Depth 6 -Compress
if ($ok) { exit 0 } else { exit 1 }
