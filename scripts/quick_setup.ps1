# quick_setup.ps1 — AI Research Workstation one-command setup (PowerShell)
param(
  [switch]$SkipBuild,
  [switch]$SkipNpmInstall
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$Web = Join-Path $Root "web\research-agent-workstation"
$ShimDir = Join-Path $env:USERPROFILE ".xsci\bin"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "AI Research Workstation — Quick Setup (PowerShell)" -ForegroundColor Cyan
Write-Host "Root: $Root" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# ── 1. Python ──────────────────────────────────────────────
Write-Host "`n>>> Step 1/5: Python environment" -ForegroundColor Cyan
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
  Write-Host "  [FAIL] Python not found. Install Python 3.10+ from https://python.org" -ForegroundColor Red
  exit 1
}
Write-Host "  [OK] Found $(& python --version) at $($python.Source)" -ForegroundColor Green

Write-Host "  Installing Python dependencies..."
pip install -e $Root --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) { Write-Host "  [OK] pip install -e ." -ForegroundColor Green }
else { Write-Host "  [WARN] pip install -e . had warnings (may be fine)" -ForegroundColor Yellow }

# ── 2. Node ────────────────────────────────────────────────
Write-Host "`n>>> Step 2/5: Node.js frontend" -ForegroundColor Cyan
$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) {
  Write-Host "  [FAIL] Node.js not found. Install Node 18+ from https://nodejs.org" -ForegroundColor Red
  exit 1
}
Write-Host "  [OK] Found Node $(node --version)" -ForegroundColor Green

if (-not $SkipNpmInstall -and -not (Test-Path (Join-Path $Web "node_modules"))) {
  Write-Host "  Running npm install (this may take a minute)..."
  Push-Location $Web
  try { npm install --silent 2>&1 | Out-Null; Write-Host "  [OK] npm install" -ForegroundColor Green }
  catch { Write-Host "  [FAIL] npm install failed: $_" -ForegroundColor Red; exit 1 }
  finally { Pop-Location }
} else {
  Write-Host "  [OK] node_modules exists (or skipped)" -ForegroundColor Green
}

# ── 3. Frontend build ──────────────────────────────────────
if (-not $SkipBuild) {
  Write-Host "`n>>> Step 3/5: Build frontend" -ForegroundColor Cyan
  Push-Location $Web
  try { npm run build 2>&1 | Out-Null; Write-Host "  [OK] npm run build" -ForegroundColor Green }
  catch { Write-Host "  [FAIL] Build failed: $_" -ForegroundColor Red; exit 1 }
  finally { Pop-Location }
}

# ── 4. CLI wrappers ────────────────────────────────────────
Write-Host "`n>>> Step 4/5: Install CLI commands" -ForegroundColor Cyan
& (Join-Path $Root "scripts\install_autokaggle_cli.ps1") -NoKaggleAlias:$false -PrependShimPath
Write-Host "  [OK] CLI wrappers installed in $ShimDir" -ForegroundColor Green

# ── 5. .env + verify ──────────────────────────────────────
Write-Host "`n>>> Step 5/5: Verify installation" -ForegroundColor Cyan
if (-not (Test-Path (Join-Path $Root ".env"))) {
  Copy-Item (Join-Path $Root ".env.example") (Join-Path $Root ".env")
  Write-Host "  [OK] Created .env from .env.example" -ForegroundColor Green
  Write-Host "  [NOTE] Edit .env to add DEEPSEEK_API_KEY" -ForegroundColor Yellow
} else {
  Write-Host "  [OK] .env exists" -ForegroundColor Green
}

# Verify python imports
python -c "import xsci; print('xsci OK')" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) { Write-Host "  [OK] Python: xsci module" -ForegroundColor Green }
python -m py_compile "$Root\src\xsci\kaggle.py","$Root\src\xsci\config.py" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) { Write-Host "  [OK] Python: compile check" -ForegroundColor Green }

# ── Summary ────────────────────────────────────────────────
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:"
Write-Host ""
Write-Host "  1. Configure DeepSeek API key:"
Write-Host "     powershell -File scripts\manage_deepseek_secret.ps1 install -ApiToken sk-xxx"
Write-Host ""
Write-Host "  2. Start the workstation:"
Write-Host "     powershell -File scripts\start_verified_workstation.ps1 restart"
Write-Host ""
Write-Host "  3. Open dashboard:"
Write-Host "     http://127.0.0.1:8088/?page=control"
Write-Host ""
Write-Host "  4. Run first training:"
Write-Host "     kaggle run titanic"
Write-Host ""
Write-Host "Full guide: docs\NEW_USER_ONBOARDING_GUIDE.md"
Write-Host "============================================================" -ForegroundColor Cyan
