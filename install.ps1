# Local-first Research Workstation installer. All Python packages stay inside
# the installation .venv; credentials stay in Windows DPAPI and are never
# written to .env or logs.
param(
  [string]$PythonExecutable = "",
  [string]$DeepSeekApiKey = "",
  [string]$KaggleApiToken = "",
  [string]$DataDir = "",
  [string]$LogsDir = "",
  [string]$BackupsDir = "",
  [string]$ProfilesDir = "",
  [string]$SecretsDir = "",
  [int]$Port = 8088,
  [switch]$OfflineOnly,
  [switch]$SkipBuild,
  [switch]$SkipNpmInstall,
  [switch]$SkipSecretPrompt,
  [switch]$SkipGatewayStart,
  [switch]$SkipMutableReconciliation,
  [switch]$SkipVerify,
  [switch]$InstallUserShims
)

$ErrorActionPreference = "Stop"
try {
  [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
  $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {}

$Root = [IO.Path]::GetFullPath((Split-Path -Parent $PSCommandPath))
$Web = Join-Path $Root "web\research-agent-workstation"
$Standalone = Join-Path $Root "app\server.js"
$BundleMode = Test-Path -LiteralPath $Standalone -PathType Leaf
$LocalAppDataBase = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData) }
$AppDataBase = if ($env:APPDATA) { $env:APPDATA } else { [Environment]::GetFolderPath([Environment+SpecialFolder]::ApplicationData) }
$ManagedLocalRoot = Join-Path $LocalAppDataBase "EvoMind"
$DataDir = if ([string]::IsNullOrWhiteSpace($DataDir)) { if ($BundleMode) { Join-Path $ManagedLocalRoot "data" } else { Join-Path $Root "user-data" } } else { [IO.Path]::GetFullPath($DataDir) }
$LogsDir = if ([string]::IsNullOrWhiteSpace($LogsDir)) { if ($BundleMode) { Join-Path $ManagedLocalRoot "logs" } else { Join-Path $DataDir "logs" } } else { [IO.Path]::GetFullPath($LogsDir) }
$BackupsDir = if ([string]::IsNullOrWhiteSpace($BackupsDir)) { if ($BundleMode) { Join-Path $ManagedLocalRoot "backups" } else { Join-Path $DataDir "backups" } } else { [IO.Path]::GetFullPath($BackupsDir) }
$ProfilesDir = if ([string]::IsNullOrWhiteSpace($ProfilesDir)) { Join-Path $AppDataBase "EvoMind\profiles" } else { [IO.Path]::GetFullPath($ProfilesDir) }
$SecretsDir = if ([string]::IsNullOrWhiteSpace($SecretsDir)) { Join-Path $AppDataBase "EvoMind\secrets" } else { [IO.Path]::GetFullPath($SecretsDir) }
$StatePath = Join-Path $DataDir "install-state.json"
$Transaction = $null
if ($BundleMode) {
  $transactionPath = Join-Path $ManagedLocalRoot "app\transaction.active.json"
  $markerPath = Join-Path $ManagedLocalRoot ".install-marker.json"
  if (-not (Test-Path -LiteralPath $transactionPath -PathType Leaf) -or -not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
    throw "SIGNED_BOOTSTRAP_REQUIRED: install the Windows bundle through @evomind-ai/cli; direct/offline bundle execution is disabled."
  }
  $Transaction = Get-Content -LiteralPath $transactionPath -Raw | ConvertFrom-Json
  $installMarker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
  $canonicalRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\')
  $canonicalDestination = [IO.Path]::GetFullPath([string]$Transaction.destination).TrimEnd('\')
  if ($Transaction.schema -ne "evomind.release_transaction.v1" -or $Transaction.phase -ne "version_staged" -or
      $canonicalRoot -ne $canonicalDestination -or $Transaction.install_id -ne $installMarker.install_id -or
      [IO.Path]::GetFullPath($DataDir) -ne [IO.Path]::GetFullPath([string]$installMarker.paths.data) -or
      [IO.Path]::GetFullPath($LogsDir) -ne [IO.Path]::GetFullPath([string]$installMarker.paths.logs) -or
      [IO.Path]::GetFullPath($BackupsDir) -ne [IO.Path]::GetFullPath([string]$installMarker.paths.backups)) {
    throw "SIGNED_BOOTSTRAP_REQUIRED: active release transaction identity or canonical data layout mismatch."
  }
}
$FinalVenv = if ($BundleMode) { Join-Path $DataDir ("runtime\python-env\" + [string]$Transaction.target_bundle_sha256) } else { Join-Path $Root ".venv" }
if ($BundleMode) {
  $transactionId = [string]$env:EVOMIND_RELEASE_TRANSACTION_ID
  if ($transactionId -notmatch '^[0-9a-fA-F-]{36}$' -or $transactionId -ne [string]$Transaction.transaction_id) {
    throw "SIGNED_BOOTSTRAP_REQUIRED: Python environment transaction identity mismatch."
  }
  $Venv = Join-Path $DataDir ("runtime\python-env\.stage-" + [string]$Transaction.target_bundle_sha256 + "-" + $transactionId)
  if (Test-Path -LiteralPath $Venv) { throw "Transaction-scoped Python staging environment already exists." }
} else {
  $Venv = $FinalVenv
}
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$PublishedVenvPython = Join-Path $FinalVenv "Scripts\python.exe"
$WheelDir = Join-Path $Root "runtime\wheels"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPYCACHEPREFIX = Join-Path $DataDir "tmp\pycache"
$env:WORKSTATION_PORT = [string]$Port
$env:WORKSTATION_HOST = "127.0.0.1"
$env:OPENAI_BASE_URL = "http://127.0.0.1:65068/v1"
$env:WORKSTATION_LOCAL_FALLBACK = "1"

function Write-Step([string]$Text) {
  Write-Host ""
  Write-Host ">>> $Text" -ForegroundColor Cyan
}

function Get-BootstrapPython {
  $candidates = @()
  if ($PythonExecutable) { $candidates += $PythonExecutable }
  if ($env:WORKSTATION_PYTHON) { $candidates += $env:WORKSTATION_PYTHON }
  $candidates += @(
    (Join-Path $Root "runtime\python\python.exe"),
    (Join-Path $Root "runtime\python.exe"),
    $VenvPython,
    "python.exe",
    "python",
    "py.exe"
  )
  foreach ($candidate in $candidates) {
    if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { return (Resolve-Path -LiteralPath $candidate).Path }
    $resolved = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($resolved -and $resolved.Source) { return $resolved.Source }
  }
  throw "Python 3.10+ was not found (WORKSTATION_PYTHON, bundled runtime, .venv, PATH)."
}

function Invoke-Checked([string]$Executable, [string[]]$Arguments, [string]$Label) {
  & $Executable @Arguments
  if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}

function Write-JsonAtomic([string]$Path, [object]$Value) {
  $parent = Split-Path -Parent $Path
  New-Item -ItemType Directory -Force -Path $parent | Out-Null
  $temp = "$Path.$PID.tmp"
  [IO.File]::WriteAllText($temp, (($Value | ConvertTo-Json -Depth 8) + "`n"), [Text.UTF8Encoding]::new($false))
  Move-Item -LiteralPath $temp -Destination $Path -Force
}

function Set-ManagedEnvValues([string]$Path, [System.Collections.IDictionary]$Values) {
  $existing = if (Test-Path -LiteralPath $Path) { @(Get-Content -LiteralPath $Path) } else { @() }
  $managed = @{}
  foreach ($key in $Values.Keys) { $managed[[string]$key] = $true }
  $preserved = @($existing | Where-Object {
    $line = [string]$_
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=') {
      return -not $managed.ContainsKey($Matches[1])
    }
    return $true
  })
  while ($preserved.Count -gt 0 -and [string]::IsNullOrWhiteSpace([string]$preserved[-1])) {
    $preserved = if ($preserved.Count -eq 1) { @() } else { @($preserved[0..($preserved.Count - 2)]) }
  }
  $lines = @($preserved)
  if ($lines.Count -gt 0) { $lines += "" }
  $lines += "# Managed by install.ps1; secrets are stored outside this file."
  foreach ($key in $Values.Keys) { $lines += "${key}=$($Values[$key])" }
  $temporary = "$Path.$PID.tmp"
  [IO.File]::WriteAllLines($temporary, $lines, [Text.UTF8Encoding]::new($false))
  Move-Item -LiteralPath $temporary -Destination $Path -Force
}

Write-Host ""
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "  Research Workstation - Local-First Idempotent Installer" -ForegroundColor Cyan
Write-Host "===============================================================" -ForegroundColor Cyan

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
foreach ($relative in @("workspace", "prisma", "tmp")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $DataDir $relative) | Out-Null
}
foreach ($directory in @($LogsDir, $BackupsDir, $ProfilesDir, $SecretsDir)) {
  New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

Write-Step "1/7 Discovering Python and creating an isolated environment"
$bootstrapPython = Get-BootstrapPython
$bootstrapVersion = & $bootstrapPython --version 2>&1
if ($LASTEXITCODE -ne 0) { throw "Bootstrap Python failed to execute." }
if (-not (Test-Path -LiteralPath $VenvPython)) {
  if ((Split-Path -Leaf $bootstrapPython).ToLowerInvariant() -eq "py.exe") {
    Invoke-Checked $bootstrapPython @("-3", "-m", "venv", $Venv) "venv creation"
  } else {
    Invoke-Checked $bootstrapPython @("-m", "venv", $Venv) "venv creation"
  }
}
if (-not (Test-Path -LiteralPath $VenvPython)) { throw "Isolated .venv Python was not created." }
$env:WORKSTATION_PYTHON = $VenvPython
Write-Host "  [OK] $bootstrapVersion -> $VenvPython" -ForegroundColor Green

Write-Step "2/7 Installing Python dependencies"
$wheelFiles = if (Test-Path -LiteralPath $WheelDir) { @(Get-ChildItem -LiteralPath $WheelDir -Filter *.whl -File) } else { @() }
if ($wheelFiles.Count -gt 0) {
  $requirements = Join-Path $Root "requirements.txt"
  if (Test-Path -LiteralPath $requirements) {
    Invoke-Checked $VenvPython @("-m", "pip", "install", "--no-index", "--find-links", $WheelDir, "-r", $requirements) "offline dependency installation"
  }
  $projectWheel = $wheelFiles | Where-Object { $_.Name -match '^(xcientist|research[_-]workstation)-' } | Select-Object -First 1
  if ($projectWheel) {
    Invoke-Checked $VenvPython @("-m", "pip", "install", "--no-index", "--find-links", $WheelDir, $projectWheel.FullName) "offline project installation"
  } elseif (Test-Path -LiteralPath (Join-Path $Root "pyproject.toml")) {
    Invoke-Checked $VenvPython @("-m", "pip", "install", "-e", $Root, "--no-deps", "--no-build-isolation") "local project installation"
  }
  $installMode = "offline_wheels"
} else {
  if ($OfflineOnly) { throw "OfflineOnly was requested, but runtime\wheels contains no wheel set." }
  if (-not (Test-Path -LiteralPath (Join-Path $Root "pyproject.toml"))) { throw "pyproject.toml is missing and no offline wheel is available." }
  Invoke-Checked $VenvPython @("-m", "pip", "install", "-e", $Root, "--no-build-isolation") "online project installation"
  $installMode = "online_fallback"
}
Invoke-Checked $VenvPython @("-c", "import xsci, research_os; print('python runtime ready')") "Python import smoke"

Write-Step "3/7 Preparing frontend runtime"
if ($BundleMode) {
  $node = Join-Path $Root "runtime\node\node.exe"
  if (-not (Test-Path -LiteralPath $node)) {
    $nodeCommand = Get-Command node -ErrorAction SilentlyContinue
    if (-not $nodeCommand) { throw "Standalone bundle requires runtime\node\node.exe or Node.js on PATH." }
  }
  Write-Host "  [OK] standalone application detected" -ForegroundColor Green
} elseif (Test-Path -LiteralPath $Web) {
  $node = Get-Command node -ErrorAction Stop
  $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
  if (-not $npm) { $npm = Get-Command npm -ErrorAction Stop }
  if (-not $SkipNpmInstall) {
    Push-Location $Web
    try { Invoke-Checked $npm.Source @("ci", "--no-audit", "--no-fund") "npm ci" } finally { Pop-Location }
  }
  Push-Location $Web
  try { Invoke-Checked $npm.Source @("run", "db:generate") "Prisma client generation" } finally { Pop-Location }
  if (-not $SkipBuild) {
    Push-Location $Web
    try {
      Invoke-Checked $npm.Source @("run", "build") "production build"
    } finally { Pop-Location }
  }
} else {
  throw "Neither standalone app/server.js nor source web application was found."
}

Write-Step "4/7 Applying additive database migrations with backup"
$database = if ($BundleMode) { Join-Path $DataDir "prisma\workstation.db" } else { Join-Path $Web "prisma\workstation.db" }
$migrationRoot = if ($BundleMode) { Join-Path $Root "app\prisma\migrations" } else { Join-Path $Web "prisma\migrations" }
$env:DATABASE_URL = "file:$($database.Replace('\','/'))"
Invoke-Checked $VenvPython @((Join-Path $Root "scripts\release_db_migrate.py"), "--database", $database, "--migrations", $migrationRoot, "--backup-dir", (Join-Path $BackupsDir "database")) "database migration"
if (-not $SkipMutableReconciliation) {
  Invoke-Checked $VenvPython @((Join-Path $Root "scripts\reconcile_action_log_mirror.py"), "--database", $database, "--runtime-root", (Join-Path $DataDir "workspace\runtime")) "action audit mirror reconciliation"
}

Write-Step "5/7 Configuring loopback gateway and deterministic fallback"
$rootEnv = if ($BundleMode) { Join-Path $DataDir "config\runtime.env" } else { Join-Path $Root ".env" }
$env:WORKSTATION_ROOT = if ($BundleMode) { $DataDir } else { $Root }
$env:WORKSTATION_DATA_DIR = $DataDir
$env:WORKSTATION_LOGS_DIR = $LogsDir
$env:WORKSTATION_BACKUPS_DIR = $BackupsDir
$env:EVOMIND_PROFILES_DIR = $ProfilesDir
$env:EVOMIND_SECRETS_DIR = $SecretsDir
Set-ManagedEnvValues $rootEnv ([ordered]@{
  OPENAI_BASE_URL = "http://127.0.0.1:65068/v1"
  WORKSTATION_LOCAL_FALLBACK = "1"
  WORKSTATION_HOST = "127.0.0.1"
  WORKSTATION_PORT = [string]$Port
  WORKSTATION_ROOT = $env:WORKSTATION_ROOT.Replace('\','/')
  WORKSTATION_DATA_DIR = $DataDir.Replace('\','/')
  WORKSTATION_LOGS_DIR = $LogsDir.Replace('\','/')
  WORKSTATION_BACKUPS_DIR = $BackupsDir.Replace('\','/')
  EVOMIND_PROFILES_DIR = $ProfilesDir.Replace('\','/')
  EVOMIND_SECRETS_DIR = $SecretsDir.Replace('\','/')
  WORKSTATION_PYTHON = $PublishedVenvPython.Replace('\','/')
  DATABASE_URL = "file:$($database.Replace('\','/'))"
})
if ($SkipGatewayStart) {
  $gatewayCommand = "skipped_transactional_install"
  Write-Host "  [OK] gateway lifecycle deferred until committed start" -ForegroundColor Green
} else {
  $gatewayCommand = "start"
  $gatewayOutput = & $VenvPython (Join-Path $Root "scripts\manage_local_gateway.py") $gatewayCommand --output (Join-Path $DataDir "gateway-status.json") 2>&1
  if ($LASTEXITCODE -ne 0) { throw "Local gateway discovery failed unexpectedly: $($gatewayOutput -join ' ')" }
  Write-Host "  [OK] gateway discovery complete ($gatewayCommand); degraded deterministic fallback is accepted" -ForegroundColor Green
}

if (-not $SkipSecretPrompt) {
  if ($DeepSeekApiKey) {
    & (Join-Path $Root "scripts\manage_deepseek_secret.ps1") install-key -ApiKey $DeepSeekApiKey | Out-Null
  }
  if ($KaggleApiToken) {
    & (Join-Path $Root "scripts\manage_kaggle_secret.ps1") install-token -ApiToken $KaggleApiToken | Out-Null
  }
}

Write-Step "6/7 Registering managed release state"
$releaseManifest = Join-Path $Root "release-manifest.json"
if ((Test-Path -LiteralPath $releaseManifest) -and -not $BundleMode) {
  Invoke-Checked $VenvPython @(
    (Join-Path $Root "scripts\workstation_lifecycle.py"), "init",
    "--root", $Root, "--package", $Root,
    "--data-root", $DataDir, "--backups-root", $BackupsDir
  ) "release registration"
  $layout = "standalone_release"
} elseif ($BundleMode) {
  # The npm bootstrapper owns current.json and the fsynced transaction journal.
  # Keep the content-addressed version directory immutable after extraction.
  $layout = "standalone_release"
} else {
  $layout = "source_tree"
}
$state = [ordered]@{
  format_version = 1
  status = "installed"
  layout = $layout
  root = $Root
  data_dir = $DataDir
  logs_dir = $LogsDir
  backups_dir = $BackupsDir
  profiles_dir = $ProfilesDir
  secrets_dir = $SecretsDir
  python = $PublishedVenvPython
  python_install_mode = $installMode
  database = $database
  host = "127.0.0.1"
  port = $Port
  gateway_base_url = "http://127.0.0.1:65068/v1"
  local_fallback = $true
  installed_at = (Get-Date).ToString("s")
}
Write-JsonAtomic $StatePath $state

if ($InstallUserShims -and (Test-Path -LiteralPath (Join-Path $Root "scripts\install_autokaggle_cli.ps1"))) {
  & (Join-Path $Root "scripts\install_autokaggle_cli.ps1") -PrependShimPath
  if ($LASTEXITCODE -ne 0) { throw "CLI shim installation failed." }
}

Write-Step "7/7 Verifying installation"
Invoke-Checked $VenvPython @("-m", "py_compile", (Join-Path $Root "scripts\manage_workstation_dashboard.py"), (Join-Path $Root "scripts\manage_local_gateway.py"), (Join-Path $Root "scripts\release_db_migrate.py")) "lifecycle compile smoke"
if (-not $SkipVerify -and -not $BundleMode -and (Test-Path -LiteralPath (Join-Path $Root "scripts\verify_new_user_release_readiness.py"))) {
  & $VenvPython (Join-Path $Root "scripts\verify_new_user_release_readiness.py") --write-report
  if ($LASTEXITCODE -ne 0) { Write-Host "  [WARN] optional resource gates remain; the local core installation is intact" -ForegroundColor Yellow }
}

if ($BundleMode) {
  if (Test-Path -LiteralPath $FinalVenv) { throw "Immutable Python environment destination already exists before publish." }
  Move-Item -LiteralPath $Venv -Destination $FinalVenv
  if (-not (Test-Path -LiteralPath $PublishedVenvPython -PathType Leaf)) {
    throw "Published Python environment is incomplete."
  }
}

Write-Host ""
Write-Host "Installation complete." -ForegroundColor Green
Write-Host "Start:  powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1"
Write-Host "Status: powershell -NoProfile -ExecutionPolicy Bypass -File .\status.ps1"
Write-Host "Open:   http://127.0.0.1:$Port/?page=assistant"
Write-Host "Data:   $DataDir"
Write-Output ([ordered]@{
  ok = $true
  status = "installed"
  version = $(if (Test-Path -LiteralPath $releaseManifest) { (Get-Content -LiteralPath $releaseManifest -Raw | ConvertFrom-Json).version } else { "source" })
  data_dir = $DataDir
  logs_dir = $LogsDir
  backups_dir = $BackupsDir
  profiles_dir = $ProfilesDir
  secrets_dir = $SecretsDir
  database = $database
  database_backup_policy = "additive_prebackup"
} | ConvertTo-Json -Depth 5 -Compress)
