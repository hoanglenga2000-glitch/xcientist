param(
    [switch]$User,
    [switch]$InstallKaggleAlias,
    [switch]$PrependShimPath,
    [switch]$NoKaggleAlias,
    [switch]$NoPathPrepend
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "Installing AutoKaggle terminal agent from $repoRoot"

$pipArgs = @("install", "-e", ".")
if ($User) {
    $pipArgs = @("install", "--user", "-e", ".")
}

$shouldInstallAlias = -not $NoKaggleAlias -or $InstallKaggleAlias
$shouldPrependPath = -not $NoPathPrepend -or $PrependShimPath

python -m pip @pipArgs

if ($shouldInstallAlias) {
    $userProfile = [Environment]::GetFolderPath("UserProfile")
    $shimDir = Join-Path $userProfile ".xsci\bin"
    New-Item -ItemType Directory -Force -Path $shimDir | Out-Null

    @"
@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python -X utf8 -m xsci.kaggle %*
"@ | Set-Content -Encoding ASCII (Join-Path $shimDir "autokaggle.cmd")

    @"
@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python -X utf8 -m xsci.kaggle %*
"@ | Set-Content -Encoding ASCII (Join-Path $shimDir "kaggle.cmd")

    @"
@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python -X utf8 -m xsci.kaggle official %*
"@ | Set-Content -Encoding ASCII (Join-Path $shimDir "kaggle-official.cmd")

    Write-Host ""
    Write-Host "Created command shims in $shimDir"

    if ($shouldPrependPath) {
        $currentUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
        $parts = @()
        if ($currentUserPath) {
            $parts = $currentUserPath -split ";" | Where-Object { $_ -and ($_ -ne $shimDir) }
        }
        $newUserPath = (@($shimDir) + $parts) -join ";"
        [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
        $currentProcessParts = $env:Path -split ";" | Where-Object { $_ -and ($_ -ne $shimDir) }
        $env:Path = (@($shimDir) + $currentProcessParts) -join ";"
        Write-Host "Prepended $shimDir to the user PATH. New terminals can run 'kaggle' directly."
    } else {
        Write-Host "Kaggle alias created, but PATH was not changed because -NoPathPrepend was set."
    }
} else {
    Write-Host ""
    Write-Host "Skipped kaggle alias because -NoKaggleAlias was set."
}

Write-Host ""
Write-Host "Installed console commands:"
foreach ($cmd in @("autokaggle", "kaggle", "kaggle-official", "xsci")) {
    $found = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($found) {
        Write-Host "  OK  $cmd -> $($found.Source)"
    } else {
        Write-Host "  MISS $cmd (check that Python Scripts is on PATH)"
    }
}

Write-Host ""
Write-Host "Next:"
Write-Host "  kaggle --help"
Write-Host "  kaggle setup"
Write-Host "  kaggle"
Write-Host "  kaggle dashboard start"
Write-Host "  http://127.0.0.1:8088/?page=control"
Write-Host "  kaggle official competitions list"
