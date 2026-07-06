# build_and_restart.ps1 - Safe Next.js build → restart flow
# After npm run build, ALWAYS run this instead of just npm run dev.
# Prevents "Cannot find module './XXXX.js'" stale-module errors.
param(
  [int]$Port = 8088
)

$ErrorActionPreference = "Stop"

$workspaceRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ProjectDir = Join-Path $workspaceRoot "web\research-agent-workstation"

Write-Host "=== Build Next.js ===" -ForegroundColor Cyan
Push-Location $ProjectDir
try {
  npm run build
  if ($LASTEXITCODE -ne 0) {
    throw "Build failed with exit code $LASTEXITCODE"
  }
} finally {
  Pop-Location
}

Write-Host "=== Build OK. Now restarting dev server on port $Port ===" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "restart_workstation_frontend.ps1") -Port $Port

Write-Host "=== Done. Frontend live at http://127.0.0.1:$Port ===" -ForegroundColor Green
