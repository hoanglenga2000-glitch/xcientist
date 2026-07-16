# AI Research Workstation one-command installer.
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
#
# This script installs local dependencies, CLI wrappers, optional DPAPI secrets,
# and runs a lightweight release check. It does not start training and does not
# print secret values.
param(
  [System.Security.SecureString]$DeepSeekApiKey,
  [System.Security.SecureString]$KaggleApiToken,
  [switch]$SkipBuild,
  [switch]$SkipNpmInstall,
  [switch]$SkipSecretPrompt,
  [switch]$SkipVerify,
  [switch]$NoPathPrepend
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

if (-not $SkipNpmInstall -or -not $SkipBuild) {
  Write-Step "Initialize local workstation database"
  $webEnvPath = Join-Path $Web ".env"
  if (-not (Test-Path $webEnvPath)) {
    $normalizedRoot = $Root.Replace("\", "/").Replace('"', '\"')
    @(
      'DATABASE_URL="file:./workstation.db"'
      ('WORKSTATION_ROOT="{0}"' -f $normalizedRoot)
      'NEXT_TELEMETRY_DISABLED=1'
      ''
    ) | Set-Content -LiteralPath $webEnvPath -Encoding UTF8
    Write-Host "  [OK] created non-secret web .env" -ForegroundColor Green
  }
  $databaseUrl = $env:DATABASE_URL
  if (-not $databaseUrl) {
    $databaseUrlLine = Get-Content -LiteralPath $webEnvPath |
      Where-Object { $_ -match '^\s*DATABASE_URL\s*=' } |
      Select-Object -First 1
    if ($databaseUrlLine) {
      $databaseUrl = (($databaseUrlLine -split '=', 2)[1]).Trim().Trim('"').Trim("'")
    }
  }
  if (-not $databaseUrl) {
    throw "DATABASE_URL is missing from web .env"
  }
  if ($databaseUrl -eq "file:./prisma/workstation.db") {
    $databaseUrl = "file:./workstation.db"
    Write-Host "  [INFO] normalized legacy SQLite path" -ForegroundColor Yellow
  }
  if (-not $databaseUrl.StartsWith("file:", [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Host "  [INFO] external database configured; automatic schema push skipped" -ForegroundColor Yellow
  } else {
    $env:DATABASE_URL = $databaseUrl
    $previousRustLog = $env:RUST_LOG
    if (-not $previousRustLog) {
      $env:RUST_LOG = "info"
    }
    Push-Location $Web
    try {
      npm run db:push -- --skip-generate
      if ($LASTEXITCODE -ne 0) {
        throw "npm run db:push failed with exit code $LASTEXITCODE"
      }
      Write-Host "  [OK] local SQLite schema is ready" -ForegroundColor Green
    } finally {
      Pop-Location
      if ($previousRustLog) {
        $env:RUST_LOG = $previousRustLog
      } else {
        Remove-Item Env:RUST_LOG -ErrorAction SilentlyContinue
      }
    }
  }
}

if (-not $SkipNpmInstall -or -not $SkipBuild) {
  Write-Step "Generate Prisma client"
  if (-not (Test-Path (Join-Path $Web "node_modules"))) {
    throw "Frontend dependencies are missing; run without -SkipNpmInstall or install them first."
  }
  Push-Location $Web
  try {
    npm run db:generate
    if ($LASTEXITCODE -ne 0) {
      throw "npm run db:generate failed with exit code $LASTEXITCODE"
    }
    Write-Host "  [OK] Prisma client generated" -ForegroundColor Green
  } finally {
    Pop-Location
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
$cliInstallArguments = if ($NoPathPrepend) {
  @{ NoPathPrepend = $true }
} else {
  @{ PrependShimPath = $true }
}
& (Join-Path $Root "scripts\install_autokaggle_cli.ps1") @cliInstallArguments
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

$workstationRootPointer = Join-Path $env:USERPROFILE ".xsci\workstation-root.txt"
[System.IO.Directory]::CreateDirectory((Split-Path -Parent $workstationRootPointer)) | Out-Null
[System.IO.File]::WriteAllText(
  $workstationRootPointer,
  [System.IO.Path]::GetFullPath($Root),
  [System.Text.UTF8Encoding]::new($false)
)
Write-Host "  [OK] Workstation source root registered" -ForegroundColor Green

if (-not $SkipSecretPrompt) {
  Write-Step "Step 5/5: Optional configuration"
  if (Test-Path (Join-Path $Root ".env")) {
    Write-Host "  [WARN] Legacy root .env detected. EvoMind does not load it; migrate secrets with 'evomind setup'." -ForegroundColor Yellow
  }

  if ($null -ne $DeepSeekApiKey) {
    $managerOutput = & (Join-Path $Root "scripts\manage_deepseek_secret.ps1") install-key -SecureApiKey $DeepSeekApiKey
    if ($LASTEXITCODE -ne 0) { throw "DeepSeek DPAPI credential installation failed." }
    $managerStatus = $managerOutput | ConvertFrom-Json
    if ($managerStatus.status -ne "configured") { throw "DeepSeek DPAPI credential verification failed." }
    Write-Host "  [OK] DeepSeek key saved with Windows DPAPI" -ForegroundColor Green
  } else {
    Write-Host "  [INFO] DeepSeek key not provided. Configure later with the hidden prompt:" -ForegroundColor Yellow
    Write-Host "         powershell -File scripts\manage_deepseek_secret.ps1 install-key"
  }

  if ($null -ne $KaggleApiToken) {
    $managerOutput = & (Join-Path $Root "scripts\manage_kaggle_secret.ps1") install-token -SecureApiToken $KaggleApiToken
    if ($LASTEXITCODE -ne 0) { throw "Kaggle DPAPI credential installation failed." }
    $managerStatus = $managerOutput | ConvertFrom-Json
    if ($managerStatus.status -ne "configured") { throw "Kaggle DPAPI credential verification failed." }
    Write-Host "  [OK] Kaggle token saved with Windows DPAPI" -ForegroundColor Green
  } else {
    Write-Host "  [INFO] Kaggle token not provided. It is only required for downloads/submissions." -ForegroundColor Yellow
  }
}

if (-not $SkipVerify) {
  Write-Step "Release readiness smoke"
  & $python.Source (Join-Path $Root "scripts\verify_new_user_release_readiness.py") --write-report
  if ($LASTEXITCODE -ne 0) {
    throw "New-user release smoke failed. See reports/NEW_USER_RELEASE_READINESS.md"
  }
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Installation complete" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "Next steps:"
Write-Host "  1. Configure at least one protected LLM provider:"
Write-Host "     evomind setup"
Write-Host "  2. Start the verified workstation:"
Write-Host "     powershell -File scripts\start_verified_workstation.ps1 restart"
Write-Host "  3. Open:"
Write-Host "     http://127.0.0.1:8088/?page=control"
Write-Host "  4. Check terminal agent:"
Write-Host "     evomind ready"
Write-Host "     evomind"
Write-Host "  UI-only before LLM setup: evomind dashboard start (provider remains Not Configured)."
Write-Host ""
Write-Host "Training, GPU jobs, and official Kaggle submission remain gate-controlled."
Write-Host "Full guide: docs\EvoMind_New_User_Final_Setup_Guide_20260707.md"
Write-Host "================================================================" -ForegroundColor Green
