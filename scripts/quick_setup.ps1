# Quick setup for AI Research Workstation.
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\quick_setup.ps1
param(
  [switch]$SkipBuild,
  [switch]$SkipNpmInstall,
  [switch]$SkipVerify
)

$ErrorActionPreference = "Stop"
try {
  [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
  $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {
  # Best effort for legacy Windows PowerShell.
}
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$Web = Join-Path $Root "web\research-agent-workstation"
$ShimDir = Join-Path $env:USERPROFILE ".xsci\bin"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

function Step([string]$Text) {
  Write-Host ""
  Write-Host ">>> $Text" -ForegroundColor Cyan
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "AI Research Workstation - Quick Setup" -ForegroundColor Cyan
Write-Host "Root: $Root" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Step "Step 1/5: Python environment"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
  Write-Host "  [FAIL] Python not found. Install Python 3.10+ from https://python.org" -ForegroundColor Red
  exit 1
}
Write-Host "  [OK] $(& $python.Source --version) at $($python.Source)" -ForegroundColor Green
& $python.Source -m pip install -e $Root --quiet
if ($LASTEXITCODE -ne 0) {
  Write-Host "  [FAIL] pip install -e . failed" -ForegroundColor Red
  exit 1
}
Write-Host "  [OK] Python dependencies installed" -ForegroundColor Green

Step "Step 2/5: Node.js frontend"
$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) {
  Write-Host "  [FAIL] Node.js not found. Install Node 18+ from https://nodejs.org" -ForegroundColor Red
  exit 1
}
Write-Host "  [OK] Node $(& $node.Source --version)" -ForegroundColor Green

if (-not $SkipNpmInstall -and -not (Test-Path (Join-Path $Web "node_modules"))) {
  Push-Location $Web
  try {
    npm install
    if ($LASTEXITCODE -ne 0) {
      throw "npm install failed with exit code $LASTEXITCODE"
    }
    Write-Host "  [OK] npm install" -ForegroundColor Green
  } finally {
    Pop-Location
  }
} else {
  Write-Host "  [OK] node_modules exists or npm install was skipped" -ForegroundColor Green
}

if (-not $SkipBuild) {
  Step "Step 3/5: Build frontend"
  Push-Location $Web
  try {
    npm run build
    if ($LASTEXITCODE -ne 0) {
      throw "npm run build failed with exit code $LASTEXITCODE"
    }
    Write-Host "  [OK] npm run build" -ForegroundColor Green
  } finally {
    Pop-Location
  }
}

Step "Step 4/5: Install CLI commands"
& (Join-Path $Root "scripts\install_autokaggle_cli.ps1") -PrependShimPath
if ($LASTEXITCODE -ne 0) {
  Write-Host "  [FAIL] CLI wrapper installation failed" -ForegroundColor Red
  exit 1
}
Write-Host "  [OK] CLI wrappers installed in %USERPROFILE%\.xsci\bin" -ForegroundColor Green

Step "Step 5/5: Verify installation"
if (-not (Test-Path (Join-Path $Root ".env"))) {
  Copy-Item (Join-Path $Root ".env.example") (Join-Path $Root ".env") -ErrorAction SilentlyContinue
  Write-Host "  [OK] Created .env from .env.example" -ForegroundColor Green
} else {
  Write-Host "  [OK] .env already exists" -ForegroundColor Green
}

& $python.Source -c "import xsci; print('xsci OK')" | Out-Null
if ($LASTEXITCODE -ne 0) {
  Write-Host "  [FAIL] xsci import failed" -ForegroundColor Red
  exit 1
}
$compileTargets = @(
  (Join-Path $Root "src\xsci\kaggle.py"),
  (Join-Path $Root "src\xsci\config.py"),
  (Join-Path $Root "src\xsci\kaggle_session.py")
)
& $python.Source -m py_compile @compileTargets
if ($LASTEXITCODE -ne 0) {
  Write-Host "  [FAIL] Python compile check failed" -ForegroundColor Red
  exit 1
}
Write-Host "  [OK] Python compile check" -ForegroundColor Green

if (-not $SkipVerify) {
  & $python.Source (Join-Path $Root "scripts\verify_new_user_release_readiness.py") --write-report
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Setup complete" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Next steps:"
Write-Host "  1. Configure an LLM key if needed:"
Write-Host "     evomind setup"
Write-Host "  2. Start the workstation:"
Write-Host "     powershell -File scripts\start_verified_workstation.ps1 restart"
Write-Host "  3. Open:"
Write-Host "     http://127.0.0.1:8088/?page=control"
Write-Host "  4. Check:"
Write-Host "     evomind ready"
Write-Host ""
Write-Host "Training and official Kaggle submission stay behind workstation gates."
Write-Host "Full guide: docs\EvoMind_New_User_Final_Setup_Guide_20260707.md"
Write-Host "============================================================" -ForegroundColor Green
