# AI Research Workstation one-command installer.
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
#
# This script installs local dependencies, CLI wrappers, optional DPAPI secrets,
# and runs a lightweight release check. It does not start training and does not
# print secret values.
param(
  [string]$DeepSeekApiKey = "",
  [string]$KaggleApiToken = "",
  [switch]$SkipBuild,
  [switch]$SkipNpmInstall,
  [switch]$SkipSecretPrompt,
  [switch]$SkipVerify
)

$ErrorActionPreference = "Stop"
try {
  [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
  $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {
  # Best effort for legacy Windows PowerShell.
}
$Root = Split-Path -Parent $PSCommandPath
$Web = Join-Path $Root "web\research-agent-workstation"
$ShimDir = if ($env:XSCI_SHIM_DIR) {
  $env:XSCI_SHIM_DIR
} else {
  Join-Path $env:USERPROFILE ".xsci\bin"
}
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

function Write-Step([string]$Text) {
  Write-Host ""
  Write-Host ">>> $Text" -ForegroundColor Cyan
}

function Require-Command([string]$Name, [string]$InstallHint) {
  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if (-not $cmd) {
    Write-Host "  [FAIL] $Name not found. $InstallHint" -ForegroundColor Red
    exit 1
  }
  return $cmd
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  AI Research Workstation - One-Command Installer" -ForegroundColor Cyan
Write-Host "  EvoMind / XCIENTIST Research Agent" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

Write-Step "Checking prerequisites"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
  $python = Get-Command python3 -ErrorAction SilentlyContinue
}
if (-not $python) {
  Write-Host "  [FAIL] Python not found. Install Python 3.10+ from https://python.org" -ForegroundColor Red
  exit 1
}
Write-Host "  [OK] $(& $python.Source --version) ($($python.Source))" -ForegroundColor Green

$node = Require-Command "node" "Install Node.js 18+ from https://nodejs.org"
Write-Host "  [OK] Node $(& $node.Source --version)" -ForegroundColor Green

$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) {
  Write-Host "  [OK] $(& $git.Source --version)" -ForegroundColor Green
} else {
  Write-Host "  [WARN] Git not found. Clone/update features may be limited." -ForegroundColor Yellow
}

Write-Step "Step 1/5: Python dependencies"
& $python.Source -m pip install -e $Root --quiet
if ($LASTEXITCODE -ne 0) {
  Write-Host "  [WARN] editable install failed; trying requirements fallback" -ForegroundColor Yellow
  & $python.Source -m pip install -r (Join-Path $Root "requirements.txt") --quiet
  & $python.Source -m pip install -e $Root --no-deps --quiet
}

& $python.Source -c "import xsci; print('xsci import ok')" | Out-Null
if ($LASTEXITCODE -ne 0) {
  Write-Host "  [FAIL] xsci import failed." -ForegroundColor Red
  exit 1
}
Write-Host "  [OK] xsci Python package" -ForegroundColor Green

$compileTargets = @(
  (Join-Path $Root "src\xsci\kaggle.py"),
  (Join-Path $Root "src\xsci\config.py"),
  (Join-Path $Root "src\xsci\kaggle_session.py")
)
& $python.Source -m py_compile @compileTargets
if ($LASTEXITCODE -ne 0) {
  Write-Host "  [FAIL] core Python modules failed to compile." -ForegroundColor Red
  exit 1
}
Write-Host "  [OK] core Python modules compile" -ForegroundColor Green

if (-not $SkipNpmInstall) {
  Write-Step "Step 2/5: Frontend dependencies"
  if (Test-Path (Join-Path $Web "node_modules")) {
    Write-Host "  [OK] node_modules exists; skipping npm install" -ForegroundColor Green
  } else {
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
  }
}

if (-not $SkipBuild) {
  Write-Step "Step 3/5: Build frontend"
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

Write-Step "Step 4/5: Install CLI commands"
& (Join-Path $Root "scripts\install_autokaggle_cli.ps1") -PrependShimPath
if ($LASTEXITCODE -ne 0) {
  Write-Host "  [FAIL] CLI wrapper installation failed." -ForegroundColor Red
  exit 1
}

# Git Bash compatible wrappers. CMD wrappers are created by install_autokaggle_cli.ps1.
New-Item -ItemType Directory -Force -Path $ShimDir | Out-Null
@"
#!/usr/bin/env bash
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
exec python -X utf8 -m xsci.kaggle "`$@"
"@ | Set-Content -Encoding ASCII (Join-Path $ShimDir "evomind") -Force
@"
#!/usr/bin/env bash
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
exec python -X utf8 -m xsci.kaggle official "`$@"
"@ | Set-Content -Encoding ASCII (Join-Path $ShimDir "kaggle-official") -Force
@"
#!/usr/bin/env bash
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
exec python -X utf8 -m xsci.kaggle "`$@"
"@ | Set-Content -Encoding ASCII (Join-Path $ShimDir "autokaggle") -Force
Remove-Item -LiteralPath (Join-Path $ShimDir "kaggle") -Force -ErrorAction SilentlyContinue

$env:Path = "$ShimDir;$env:Path"
Write-Host "  [OK] CLI wrappers installed in %USERPROFILE%\.xsci\bin" -ForegroundColor Green

if (-not $SkipSecretPrompt) {
  Write-Step "Step 5/5: Optional configuration"
  if (-not (Test-Path (Join-Path $Root ".env"))) {
    Copy-Item (Join-Path $Root ".env.example") (Join-Path $Root ".env") -ErrorAction SilentlyContinue
    Write-Host "  [OK] Created .env from .env.example" -ForegroundColor Green
  } else {
    Write-Host "  [OK] .env already exists" -ForegroundColor Green
  }

  if ($DeepSeekApiKey) {
    & (Join-Path $Root "scripts\manage_deepseek_secret.ps1") install-key -ApiKey $DeepSeekApiKey | Out-Null
    Write-Host "  [OK] DeepSeek key saved with Windows DPAPI" -ForegroundColor Green
  } else {
    Write-Host "  [INFO] DeepSeek key not provided. Configure later with:" -ForegroundColor Yellow
    Write-Host "         powershell -File scripts\manage_deepseek_secret.ps1 install-key -ApiKey sk-xxx"
  }

  if ($KaggleApiToken) {
    & (Join-Path $Root "scripts\manage_kaggle_secret.ps1") install-token -ApiToken $KaggleApiToken | Out-Null
    Write-Host "  [OK] Kaggle token saved with Windows DPAPI" -ForegroundColor Green
  } else {
    Write-Host "  [INFO] Kaggle token not provided. It is only required for downloads/submissions." -ForegroundColor Yellow
  }
}

if (-not $SkipVerify) {
  Write-Step "Release readiness smoke"
  & $python.Source (Join-Path $Root "scripts\verify_new_user_release_readiness.py") --write-report
  if ($LASTEXITCODE -ne 0) {
    Write-Host "  [WARN] New-user release smoke reported issues. See reports/NEW_USER_RELEASE_READINESS.md" -ForegroundColor Yellow
  }
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Installation complete" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "Next steps:"
Write-Host "  1. Start the workstation:"
Write-Host "     powershell -File scripts\start_verified_workstation.ps1 restart"
Write-Host "  2. Open:"
Write-Host "     http://127.0.0.1:8088/?page=control"
Write-Host "  3. Check terminal agent:"
Write-Host "     evomind ready"
Write-Host "     evomind"
Write-Host ""
Write-Host "Training, GPU jobs, and official Kaggle submission remain gate-controlled."
Write-Host "Full guide: docs\EvoMind_New_User_Final_Setup_Guide_20260707.md"
Write-Host "================================================================" -ForegroundColor Green
