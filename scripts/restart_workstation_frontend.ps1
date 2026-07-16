param(
  [int]$Port = 8088,
  [string]$ProjectDir = "",
  [string]$PythonExecutable = "",
  [string]$DatabaseUrl = "",
  [ValidateSet("development", "production")]
  [string]$Mode = "development"
)

$ErrorActionPreference = "Stop"

if (-not $ProjectDir) {
  $workspaceRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
  $ProjectDir = Join-Path $workspaceRoot "web\research-agent-workstation"
}

$project = Resolve-Path $ProjectDir
$workspaceRoot = Resolve-Path (Join-Path $project.Path "..\..")
$next = Join-Path $project ".next"
$processIds = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess -Unique

foreach ($processId in $processIds) {
  Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
}

$resolvedNext = Resolve-Path $next -ErrorAction SilentlyContinue
if ($Mode -eq "development" -and $resolvedNext -and $resolvedNext.Path.StartsWith($project.Path)) {
  Remove-Item -LiteralPath $resolvedNext.Path -Recurse -Force -ErrorAction SilentlyContinue
}

$buildIdPath = Join-Path $next "BUILD_ID"
$buildId = $null
if ($Mode -eq "production") {
  if (-not (Test-Path -LiteralPath $buildIdPath -PathType Leaf)) {
    throw "Production mode requires a completed Next.js build at $buildIdPath"
  }
  $buildId = (Get-Content -LiteralPath $buildIdPath -Raw).Trim()
  if ([string]::IsNullOrWhiteSpace($buildId)) {
    throw "Production Next.js BUILD_ID is empty."
  }
}

# Child processes started by Start-Process inherit this PowerShell environment.
$env:NODE_ENV = if ($Mode -eq "production") { "production" } else { "development" }
$env:WORKSTATION_ROOT = $workspaceRoot.Path

$selectedPython = $PythonExecutable
if (-not $selectedPython) {
  $selectedPython = $env:WORKSTATION_PYTHON
}
if (-not $selectedPython) {
  $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
  if ($pythonCommand) {
    $selectedPython = $pythonCommand.Source
  }
}
if ($selectedPython) {
  if (Test-Path -LiteralPath $selectedPython -PathType Leaf) {
    $env:WORKSTATION_PYTHON = (Resolve-Path -LiteralPath $selectedPython).Path
  } else {
    $pythonCommand = Get-Command $selectedPython -ErrorAction Stop
    $env:WORKSTATION_PYTHON = $pythonCommand.Source
  }
}

$databasePath = Join-Path $project.Path "prisma\workstation.db"
$env:DATABASE_URL = if ($DatabaseUrl) {
  $DatabaseUrl
} else {
  "file:$($databasePath.Replace('\', '/'))"
}
$prismaPush = Join-Path $project.Path "scripts\prisma-db-push.mjs"
if ($env:DATABASE_URL.StartsWith("file:", [System.StringComparison]::OrdinalIgnoreCase)) {
  $previousErrorActionPreference = $ErrorActionPreference
  $prismaOutput = @()
  $prismaExitCode = $null
  try {
    # Windows PowerShell 5.1 promotes native stderr to ErrorRecord objects when
    # ErrorActionPreference is Stop. Prisma writes progress to stderr on success.
    $ErrorActionPreference = "Continue"
    $global:LASTEXITCODE = 0
    $prismaOutput = @(& node.exe $prismaPush "--skip-generate" 2>&1)
    $prismaExitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousErrorActionPreference
  }
  if ($prismaExitCode -ne 0) {
    $diagnostic = @($prismaOutput | Select-Object -Last 20 | ForEach-Object { [string]$_ }) -join [Environment]::NewLine
    throw "Prisma database initialization failed with exit code $prismaExitCode.$([Environment]::NewLine)$diagnostic"
  }
}

$nextCli = Join-Path $project.Path "node_modules\next\dist\bin\next"
if (-not (Test-Path -LiteralPath $nextCli)) {
  throw "Next.js CLI not found. Run npm ci in $($project.Path) first."
}

$nextCommand = if ($Mode -eq "production") { "start" } else { "dev" }
$server = Start-Process -FilePath "node.exe" -ArgumentList @(
  "node_modules\next\dist\bin\next",
  $nextCommand,
  "--hostname",
  "127.0.0.1",
  "--port",
  [string]$Port
) -WorkingDirectory $project.Path -WindowStyle Hidden -PassThru

# Fresh .next removal above forces a cold compile on first request, so allow
# more headroom than a warm restart before declaring the server unhealthy.
$deadline = (Get-Date).AddSeconds(90)
$homeOk = $false
while ((Get-Date) -lt $deadline) {
  try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
      $homeOk = $true
      break
    }
  } catch {
    Start-Sleep -Milliseconds 800
  }
}

if (-not $homeOk) {
  if ($server -and -not $server.HasExited) {
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
  }
  throw "Frontend did not become ready on http://127.0.0.1:$Port"
}

$listenerIds = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess -Unique)
if ($listenerIds.Count -ne 1) {
  throw "Expected exactly one frontend listener on port $Port, found $($listenerIds.Count)."
}
$listener = Get-CimInstance Win32_Process -Filter "ProcessId = $($listenerIds[0])" -ErrorAction Stop
$listenerCommand = [string]$listener.CommandLine
if ($Mode -eq "production" -and $listenerCommand -notmatch '(?i)next(?:\\|/)dist(?:\\|/)bin(?:\\|/)next[^\r\n]*\sstart(?:\s|$)') {
  throw "Frontend listener is not the expected Next.js production server."
}
if ($Mode -eq "development" -and $listenerCommand -notmatch '(?i)next(?:\\|/)dist(?:\\|/)bin(?:\\|/)next[^\r\n]*\sdev(?:\s|$)') {
  throw "Frontend listener is not the expected Next.js development server."
}

$html = (Invoke-WebRequest -Uri "http://127.0.0.1:$Port" -UseBasicParsing -TimeoutSec 10).Content
$cssHref = [regex]::Match($html, 'href="([^"]*\.css[^"]*)"').Groups[1].Value
if (-not $cssHref) {
  throw "No CSS bundle was referenced by the frontend HTML."
}

$css = Invoke-WebRequest -Uri "http://127.0.0.1:$Port$cssHref" -UseBasicParsing -TimeoutSec 10
if ($css.StatusCode -ne 200 -or -not $css.Content.Contains("--tw-border-spacing-x")) {
  throw "CSS health check failed for $cssHref"
}

$summary = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/workstation-summary" -Method Get -TimeoutSec 20

[pscustomobject]@{
  ok = $true
  url = "http://127.0.0.1:$Port"
  mode = $Mode
  next_command = $nextCommand
  build_id = $buildId
  process_id = $server.Id
  listener_pid = $listenerIds[0]
  command_verified = $true
  css_href = $cssHref
  css_length = $css.Content.Length
  latest_experiment = $summary.runtime.latest_experiment_dir
  agent_trace_count = $summary.runtime.agent_trace.Count
  event_count = $summary.runtime.event_log.Count
} | ConvertTo-Json -Depth 4
