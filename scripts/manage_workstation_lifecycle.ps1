param(
  [ValidateSet("status", "migrate", "upgrade", "rollback", "uninstall")]
  [string]$Command = "status",
  [string]$InstallRoot = "",
  [string]$PackagePath = "",
  [string]$BackupPath = "",
  [string]$HostName = "127.0.0.1",
  [int]$Port = 8088,
  [switch]$NoRestart,
  [switch]$PurgeUserData,
  [switch]$Confirm
)

$ErrorActionPreference = "Stop"
try {
  [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
  $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {}

$ScriptRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$Root = if ($InstallRoot) { [IO.Path]::GetFullPath($InstallRoot) } else { $ScriptRoot }
$BundleMode = Test-Path -LiteralPath (Join-Path $Root "app\server.js") -PathType Leaf
$LocalAppDataBase = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData) }
$ManagedLocalRoot = Join-Path $LocalAppDataBase "EvoMind"
$DataRoot = if ($BundleMode) { Join-Path $ManagedLocalRoot "data" } elseif ($env:WORKSTATION_DATA_DIR) { [IO.Path]::GetFullPath($env:WORKSTATION_DATA_DIR) } else { $Root }
$BackupsRoot = if ($BundleMode) { Join-Path $ManagedLocalRoot "backups" } elseif ($env:WORKSTATION_BACKUPS_DIR) { [IO.Path]::GetFullPath($env:WORKSTATION_BACKUPS_DIR) } else { Join-Path $DataRoot "backups" }
$env:WORKSTATION_DATA_DIR = $DataRoot
$env:WORKSTATION_BACKUPS_DIR = $BackupsRoot

function Get-WorkstationPython {
  $candidates = @()
  if ($env:WORKSTATION_PYTHON) { $candidates += $env:WORKSTATION_PYTHON }
  $candidates += @(
    (Join-Path $Root "runtime\python\python.exe"),
    (Join-Path $Root "runtime\python.exe"),
    (Join-Path $Root ".venv\Scripts\python.exe"),
    "python.exe",
    "python"
  )
  foreach ($candidate in $candidates) {
    if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { return (Resolve-Path -LiteralPath $candidate).Path }
    $resolved = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($resolved -and $resolved.Source) { return $resolved.Source }
  }
  throw "WORKSTATION_LIFECYCLE_FAILED: Python was not found."
}

function Get-ExternalPython {
  foreach ($candidate in @("python.exe", "python")) {
    $resolved = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($resolved -and $resolved.Source) {
      $full = [IO.Path]::GetFullPath($resolved.Source)
      if (-not $full.StartsWith(([IO.Path]::GetFullPath($Root) + [IO.Path]::DirectorySeparatorChar), [StringComparison]::OrdinalIgnoreCase)) {
        return $full
      }
    }
  }
  return $null
}

function Invoke-JsonProcess([string]$Executable, [string[]]$Arguments, [switch]$AllowFailure) {
  $output = & $Executable @Arguments 2>&1
  $exitCode = $LASTEXITCODE
  $text = ($output -join "`n")
  if ($exitCode -ne 0 -and -not $AllowFailure) {
    throw "Command failed ($exitCode): $Executable $($Arguments -join ' ')`n$text"
  }
  return [ordered]@{ exit_code = $exitCode; output = $text; ok = ($exitCode -eq 0) }
}

$python = Get-WorkstationPython
$manager = Join-Path $Root "scripts\manage_workstation_dashboard.py"
$gateway = Join-Path $Root "scripts\manage_local_gateway.py"
$migrator = Join-Path $Root "scripts\release_db_migrate.py"
$migrationRoot = if ($BundleMode) { Join-Path $Root "app\prisma\migrations" } else { Join-Path $Root "web\research-agent-workstation\prisma\migrations" }
$helper = Join-Path $Root "scripts\workstation_lifecycle.py"
$env:WORKSTATION_HOST = $HostName
$env:WORKSTATION_PORT = [string]$Port
$env:WORKSTATION_PYTHON = $python

if ($Command -eq "status") {
  if ($BundleMode) {
    $transactionPath = Join-Path $ManagedLocalRoot "app\transaction.active.json"
    $currentPath = Join-Path $ManagedLocalRoot "app\current.json"
    $binding = if (Test-Path -LiteralPath $transactionPath -PathType Leaf) {
      Get-Content -LiteralPath $transactionPath -Raw | ConvertFrom-Json
    } elseif (Test-Path -LiteralPath $currentPath -PathType Leaf) {
      Get-Content -LiteralPath $currentPath -Raw | ConvertFrom-Json
    } else { $null }
    $boundDirectory = if ($binding -and $binding.destination) {
      [IO.Path]::GetFullPath([string]$binding.destination)
    } elseif ($binding -and $binding.directory) {
      [IO.Path]::GetFullPath((Join-Path (Join-Path $ManagedLocalRoot "app\versions") ([string]$binding.directory)))
    } else { "" }
    $rootMatches = $boundDirectory -and ($boundDirectory.TrimEnd('\') -eq ([IO.Path]::GetFullPath($Root)).TrimEnd('\'))
    $installPayload = if ($rootMatches) {
      [ordered]@{ status = "installed"; version = [string]$(if ($binding.target_version) { $binding.target_version } else { $binding.version }); immutable_version_dir = $true }
    } else {
      [ordered]@{ status = "failed"; message = "content-addressed install binding mismatch" }
    }
    $install = [ordered]@{ exit_code = $(if ($rootMatches) { 0 } else { 1 }); output = ($installPayload | ConvertTo-Json -Compress); ok = [bool]$rootMatches }
  } else {
    $install = Invoke-JsonProcess $python @($helper, "status", "--root", $Root, "--data-root", $DataRoot, "--backups-root", $BackupsRoot) -AllowFailure
  }
  $dashboard = Invoke-JsonProcess $python @($manager, "status", "--host", $HostName, "--port", [string]$Port) -AllowFailure
  $gatewayStatus = Invoke-JsonProcess $python @($gateway, "status") -AllowFailure
  [ordered]@{ status = "ok"; root = $Root; python = $python; install = $install; dashboard = $dashboard; gateway = $gatewayStatus } | ConvertTo-Json -Depth 8
  exit 0
}

if ($Command -eq "migrate") {
  $database = if ($env:DATABASE_URL -and $env:DATABASE_URL.StartsWith("file:")) {
    $env:DATABASE_URL.Substring(5)
  } elseif (Test-Path -LiteralPath (Join-Path $Root "app\server.js")) {
    Join-Path $DataRoot "prisma\workstation.db"
  } else {
    Join-Path $Root "web\research-agent-workstation\prisma\workstation.db"
  }
  $backupDir = Join-Path $BackupsRoot "database"
  $result = Invoke-JsonProcess $python @($migrator, "--database", $database, "--migrations", $migrationRoot, "--backup-dir", $backupDir)
  $result.output
  exit 0
}

if ($Command -eq "uninstall") {
  if ($BundleMode) {
    throw "SIGNED_BOOTSTRAP_REQUIRED: installed bundle removal and --purge-data must run through @evomind-ai/cli."
  }
  if (-not $Confirm) { throw "Uninstall requires -Confirm. User data is preserved unless -PurgeUserData is set." }
  $stop = Invoke-JsonProcess $python @($manager, "stop", "--host", $HostName, "--port", [string]$Port, "--timeout", "30")
  $externalPython = Get-ExternalPython
  if ($externalPython) { $python = $externalPython }
  $arguments = @($helper, "uninstall", "--root", $Root, "--data-root", $DataRoot, "--backups-root", $BackupsRoot)
  if ($PurgeUserData) { $arguments += "--purge-user-data" }
  $result = Invoke-JsonProcess $python $arguments
  $result.output
  exit 0
}

if ($Command -eq "upgrade") {
  throw "SIGNED_BOOTSTRAP_REQUIRED: direct PackagePath upgrades were removed; use evomind upgrade --manifest <signed-manifest>."
}

if ($Command -eq "rollback") {
  if ($BundleMode) {
    throw "SIGNED_BOOTSTRAP_REQUIRED: installed bundle rollback must run through @evomind-ai/cli."
  }
  $stop = Invoke-JsonProcess $python @($manager, "stop", "--host", $HostName, "--port", [string]$Port, "--timeout", "30")
  $arguments = @($helper, "rollback", "--root", $Root, "--data-root", $DataRoot, "--backups-root", $BackupsRoot)
  if ($BackupPath) { $arguments += @("--backup", (Resolve-Path -LiteralPath $BackupPath).Path) }
  $rollback = Invoke-JsonProcess $python $arguments
  $python = Get-WorkstationPython
  $migrator = Join-Path $Root "scripts\release_db_migrate.py"
  $migration = Invoke-JsonProcess $python @($migrator, "--database", (Join-Path $DataRoot "prisma\workstation.db"), "--migrations", $migrationRoot, "--backup-dir", (Join-Path $BackupsRoot "database"))
  $restart = $null
  if (-not $NoRestart) {
    $restart = Invoke-JsonProcess $python @((Join-Path $Root "scripts\manage_workstation_dashboard.py"), "start", "--host", $HostName, "--port", [string]$Port, "--timeout", "90")
  }
  [ordered]@{ status = "rolled_back"; rollback = $rollback; migration = $migration; restart = $restart } | ConvertTo-Json -Depth 8
  exit 0
}
