# install.ps1 — AI Research Workstation ONE-COMMAND installer
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
# What it does: checks deps → installs Python/Node deps → builds frontend →
#               installs CLI → configures secrets → verifies → prints next steps.
param(
  [string]$DeepSeekApiKey = "",
  [string]$KaggleApiToken = "",
  [switch]$SkipBuild,
  [switch]$SkipNpmInstall,
  [switch]$SkipSecretPrompt,
  [switch]$SkipVerify
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSCommandPath
$Web = Join-Path $Root "web\research-agent-workstation"
$Green = "Green"; $Yellow = "Yellow"; $Cyan = "Cyan"; $Red = "Red"

Write-Host "`n================================================================" -ForegroundColor $Cyan
Write-Host "  AI Research Workstation — One-Command Installer" -ForegroundColor $Cyan
Write-Host "  XCIENTIST / Kaggle Research Agent" -ForegroundColor $Cyan
Write-Host "================================================================`n" -ForegroundColor $Cyan

# ── 0. Prerequisites ──────────────────────────────────────────
Write-Host ">>> Checking prerequisites..." -ForegroundColor $Cyan

# Check Python
$python = (Get-Command python -ErrorAction SilentlyContinue) ?? (Get-Command python3 -ErrorAction SilentlyContinue)
if (-not $python) {
  Write-Host "  [MISS] Python not found. Please install Python 3.10+ from https://python.org" -ForegroundColor $Red
  exit 1
}
$pyVer = & $python.Source --version 2>&1
Write-Host "  [OK] $pyVer ($($python.Source))" -ForegroundColor $Green

# Check Node
$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) {
  Write-Host "  [MISS] Node.js not found. Install Node 18+ from https://nodejs.org" -ForegroundColor $Red
  exit 1
}
Write-Host "  [OK] Node $(node --version)" -ForegroundColor $Green

# Check Git
$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) {
  Write-Host "  [WARN] Git not found. Some features may be limited." -ForegroundColor $Yellow
} else {
  Write-Host "  [OK] Git $(git --version)" -ForegroundColor $Green
}

# ── 1. Python deps ────────────────────────────────────────────
Write-Host "`n>>> Step 1/5: Python dependencies" -ForegroundColor $Cyan

pip install -e $Root --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
  Write-Host "  [OK] pip install -e . (editable install)" -ForegroundColor $Green
} else {
  Write-Host "  [WARN] pip install -e . had warnings, trying pip install -r requirements.txt" -ForegroundColor $Yellow
  pip install -r (Join-Path $Root "requirements.txt") --quiet
  pip install -e $Root --no-deps --quiet
  Write-Host "  [OK] Fallback install complete" -ForegroundColor $Green
}

# Verify Python import
& python -c "import xsci; print('xsci', xsci.__file__)" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
  Write-Host "  [OK] xsci Python module" -ForegroundColor $Green
} else {
  Write-Host "  [FAIL] xsci module failed to import. Check the error above." -ForegroundColor $Red
  exit 1
}

# Also verify key XSCI submodules compile
& python -m py_compile src/xsci/kaggle.py,src/xsci/config.py,src/xsci/kaggle_session.py 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
  Write-Host "  [OK] Core XSCI modules compile" -ForegroundColor $Green
}

# ── 2. Node/npm ───────────────────────────────────────────────
if (-not $SkipNpmInstall) {
  Write-Host "`n>>> Step 2/5: Frontend (npm install)" -ForegroundColor $Cyan
  if (Test-Path (Join-Path $Web "node_modules")) {
    Write-Host "  [OK] node_modules exists (skipping npm install)" -ForegroundColor $Green
  } else {
    Write-Host "  Running npm install (1-2 minutes)..."
    Push-Location $Web
    try {
      npm install 2>&1 | Out-Null
      Write-Host "  [OK] npm install" -ForegroundColor $Green
    } catch {
      Write-Host "  [FAIL] npm install failed: $_" -ForegroundColor $Red
      exit 1
    } finally { Pop-Location }
  }
}

# ── 3. Frontend build ─────────────────────────────────────────
if (-not $SkipBuild) {
  Write-Host "`n>>> Step 3/5: Build frontend" -ForegroundColor $Cyan
  Push-Location $Web
  try {
    npm run build 2>&1 | Out-Null
    Write-Host "  [OK] npm run build" -ForegroundColor $Green
  } catch {
    Write-Host "  [FAIL] Build failed: $_" -ForegroundColor $Red
    exit 1
  } finally { Pop-Location }
}

# ── 4. CLI wrappers ───────────────────────────────────────────
Write-Host "`n>>> Step 4/5: Install CLI commands" -ForegroundColor $Cyan
& (Join-Path $Root "scripts\install_autokaggle_cli.ps1") -NoKaggleAlias:$false -PrependShimPath 2>&1 | Out-Null

# Create bash wrappers for Git Bash compatibility
$shimDir = Join-Path $env:USERPROFILE ".xsci\bin"
New-Item -ItemType Directory -Force -Path $shimDir | Out-Null
@"
#!/usr/bin/env bash
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
exec python -X utf8 -m xsci.kaggle "$@"
"@ | Set-Content -Encoding ASCII (Join-Path $shimDir "kaggle") -Force
@"
#!/usr/bin/env bash
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
exec python -X utf8 -m xsci.kaggle official "$@"
"@ | Set-Content -Encoding ASCII (Join-Path $shimDir "kaggle-official") -Force
@"
#!/usr/bin/env bash
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
exec python -X utf8 -m xsci.kaggle "$@"
"@ | Set-Content -Encoding ASCII (Join-Path $shimDir "autokaggle") -Force

Write-Host "  [OK] CLI wrappers installed ($shimDir)" -ForegroundColor $Green

# Refresh PATH in this session
$env:Path = "$shimDir;$env:Path"

# ── 5. Secrets ────────────────────────────────────────────────
if (-not $SkipSecretPrompt) {
  Write-Host "`n>>> Step 5/5: Configuration" -ForegroundColor $Cyan

  # .env file
  if (-not (Test-Path (Join-Path $Root ".env"))) {
    Copy-Item (Join-Path $Root ".env.example") (Join-Path $Root ".env") -ErrorAction SilentlyContinue
    Write-Host "  [OK] Created .env file" -ForegroundColor $Green
  } else {
    Write-Host "  [OK] .env exists" -ForegroundColor $Green
  }

  # DeepSeek API key
  if ($DeepSeekApiKey) {
    & (Join-Path $Root "scripts\manage_deepseek_secret.ps1") install -ApiToken $DeepSeekApiKey 2>&1 | Out-Null
    Write-Host "  [OK] DeepSeek API key saved (Windows DPAPI)" -ForegroundColor $Green
  } else {
    Write-Host "  [ ] DeepSeek API key: not provided. Run after install:"
    Write-Host "      powershell -File scripts\manage_deepseek_secret.ps1 install -ApiToken sk-xxx" -ForegroundColor $Yellow
  }

  # Kaggle token
  if ($KaggleApiToken) {
    & (Join-Path $Root "scripts\manage_kaggle_secret.ps1") install-token -ApiToken $KaggleApiToken 2>&1 | Out-Null
    Write-Host "  [OK] Kaggle API token saved (Windows DPAPI)" -ForegroundColor $Green
  } else {
    Write-Host "  [ ] Kaggle token: not provided. Optional — only needed for Kaggle downloads." -ForegroundColor $Yellow
  }
}

# ── Verify ────────────────────────────────────────────────────
if (-not $SkipVerify) {
  & python -c "
from xsci.kaggle import _has_llm, is_onboarded
print(f'llm_configured={_has_llm()}')
print(f'onboarded={is_onboarded()}')
" 2>&1 | ForEach-Object { Write-Host "  [verify] $_" -ForegroundColor $Cyan }
}

# ── Done ──────────────────────────────────────────────────────
Write-Host "`n================================================================" -ForegroundColor $Green
Write-Host "  Installation complete!" -ForegroundColor $Green
Write-Host "================================================================" -ForegroundColor $Green
Write-Host ""
Write-Host "  NEXT STEPS:"
Write-Host ""
Write-Host "  1. Start the workstation:"
Write-Host "     powershell -File scripts/start_verified_workstation.ps1 restart"
Write-Host "     (or: cd web/research-agent-workstation && npm run dev)"
Write-Host ""
Write-Host "  2. Open the dashboard:"
Write-Host "     http://127.0.0.1:8088/?page=control"
Write-Host ""
Write-Host "  3. Try the CLI agent:"
Write-Host "     kaggle"
Write-Host "     kaggle ready"
Write-Host "     kaggle competitions titanic"
Write-Host ""
Write-Host "  4. Run your first training:"
Write-Host "     kaggle run titanic"
Write-Host ""
Write-Host "  Full guide: docs/NEW_USER_ONBOARDING_GUIDE.md"
Write-Host "================================================================" -ForegroundColor $Green
