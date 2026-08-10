param(
  [string]$Port = "8088",
  [string]$ProjectDir = "",
  [string]$PythonExecutable = "",
  [string]$DatabaseUrl = "",
  [ValidateSet("development", "production")]
  [string]$Mode = "development"
)

$ErrorActionPreference = "Stop"

$workspaceRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not $ProjectDir) {
  $ProjectDir = Join-Path $workspaceRoot.Path "web\research-agent-workstation"
}
$project = Resolve-Path $ProjectDir
$next = Join-Path $project.Path ".next"

if ($Port -notmatch "^\d+$" -or [int]$Port -lt 1 -or [int]$Port -gt 65535) {
  throw "Port must be an integer from 1 to 65535."
}

if (-not $PythonExecutable) {
  $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
  if (-not $pythonCommand) {
    throw "Python executable was not provided and python is not on PATH."
  }
  $PythonExecutable = $pythonCommand.Source
}
$PythonExecutable = (Resolve-Path $PythonExecutable).Path

$env:WORKSTATION_ROOT = $workspaceRoot.Path
$env:WORKSTATION_PYTHON = $PythonExecutable
$env:WORKSTATION_HOST = "127.0.0.1"
$env:WORKSTATION_PORT = $Port
$env:NODE_ENV = $Mode
$env:DATABASE_URL = if ($DatabaseUrl) { $DatabaseUrl } else { "file:./prisma/workstation.db" }

$processIds = Get-NetTCPConnection -LocalPort ([int]$Port) -State Listen -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess -Unique
foreach ($processId in $processIds) {
  Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
}

if ($Mode -eq "development") {
  $resolvedNext = Resolve-Path $next -ErrorAction SilentlyContinue
  if ($resolvedNext -and $resolvedNext.Path.StartsWith($project.Path, [System.StringComparison]::OrdinalIgnoreCase)) {
    Remove-Item -LiteralPath $resolvedNext.Path -Recurse -Force -ErrorAction SilentlyContinue
  }
} elseif (-not (Test-Path (Join-Path $next "BUILD_ID"))) {
  throw "Production build is missing. Run npm run build before starting production mode."
}

$nodeCommand = Get-Command node.exe -ErrorAction Stop
$nextBin = Join-Path $project.Path "node_modules\next\dist\bin\next"
if (-not (Test-Path $nextBin)) {
  throw "Next.js CLI was not found at $nextBin"
}

$prismaDatabase = Join-Path $project.Path "prisma\workstation.db"
$prismaPush = Join-Path $project.Path "scripts\prisma-db-push.mjs"
if (-not (Test-Path $prismaPush)) {
  throw "Prisma database initializer was not found at $prismaPush"
}
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$prismaOutput = @(& node.exe $prismaPush "--skip-generate" 2>&1)
$prismaExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($prismaExitCode -ne 0) {
  throw "Prisma database initialization failed: $($prismaOutput -join [Environment]::NewLine)"
}
if ($env:DATABASE_URL -eq "file:./prisma/workstation.db" -and -not (Test-Path $prismaDatabase)) {
  throw "Prisma database initialization completed without creating $prismaDatabase"
}

$nextCommand = if ($Mode -eq "production") { "start" } else { "dev" }
$arguments = @(
  "node_modules\next\dist\bin\next",
  $nextCommand,
  "--hostname",
  "127.0.0.1",
  "--port",
  $Port
)
$server = Start-Process -FilePath $nodeCommand.Source -ArgumentList $arguments -WorkingDirectory $project.Path -WindowStyle Hidden -PassThru

$deadline = (Get-Date).AddSeconds(90)
$homeOk = $false
while ((Get-Date) -lt $deadline) {
  if ($server.HasExited) {
    throw "Frontend process exited before becoming ready (exit code $($server.ExitCode))."
  }
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
  Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
  throw "Frontend did not become ready on http://127.0.0.1:$Port"
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
  url = "http://127.0.0.1:$Port/?page=assistant"
  mode = $Mode
  pid = $server.Id
  python = $PythonExecutable
  database_url_configured = [bool]$env:DATABASE_URL
  css_href = $cssHref
  css_length = $css.Content.Length
  latest_experiment = $summary.runtime.latest_experiment_dir
  agent_trace_count = $summary.runtime.agent_trace.Count
  event_count = $summary.runtime.event_log.Count
} | ConvertTo-Json -Depth 4
