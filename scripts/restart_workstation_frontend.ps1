param(
  [int]$Port = 8088,
  [string]$ProjectDir = ""
)

$ErrorActionPreference = "Stop"

if (-not $ProjectDir) {
  $workspaceRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
  $ProjectDir = Join-Path $workspaceRoot "web\research-agent-workstation"
}

$project = Resolve-Path $ProjectDir
$next = Join-Path $project ".next"
$processIds = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess -Unique

foreach ($processId in $processIds) {
  Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
}

$resolvedNext = Resolve-Path $next -ErrorAction SilentlyContinue
if ($resolvedNext -and $resolvedNext.Path.StartsWith($project.Path)) {
  Remove-Item -LiteralPath $resolvedNext.Path -Recurse -Force -ErrorAction SilentlyContinue
}

# Force NODE_ENV=development for the dev server. If the launching shell has
# NODE_ENV=production (common after a build/CI step), `next dev` will refuse to
# wire the PostCSS/Tailwind loader and every page returns HTTP 500 with
# "Module parse failed: Unexpected character '@'" on globals.css @tailwind.
# Child processes started by Start-Process inherit this PowerShell env.
$env:NODE_ENV = "development"

Start-Process -FilePath "cmd.exe" -ArgumentList "/c npm run dev" -WorkingDirectory $project.Path -WindowStyle Hidden

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
  url = "http://127.0.0.1:$Port"
  css_href = $cssHref
  css_length = $css.Content.Length
  latest_experiment = $summary.runtime.latest_experiment_dir
  agent_trace_count = $summary.runtime.agent_trace.Count
  event_count = $summary.runtime.event_log.Count
} | ConvertTo-Json -Depth 4
