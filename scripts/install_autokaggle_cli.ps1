param(
    [switch]$User,
    [switch]$InstallLegacyKaggleAlias,
    [switch]$PrependShimPath,
    [switch]$NoKaggleAlias,
    [switch]$NoPathPrepend
)

$ErrorActionPreference = "Stop"
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {
    # Best effort for legacy Windows PowerShell.
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "Installing EvoMind terminal agent from the current repository."

$pipArgs = @("install", "-e", ".", "--quiet")
if ($User) {
    $pipArgs = @("install", "--user", "-e", ".", "--quiet")
}

$shouldInstallAlias = -not $NoKaggleAlias
$shouldInstallLegacyKaggleAlias = [bool]$InstallLegacyKaggleAlias
$shouldPrependPath = -not $NoPathPrepend -or $PrependShimPath

python -m pip @pipArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "  FAIL pip install -e . failed." -ForegroundColor Red
    exit 1
}

if ($shouldInstallAlias) {
    $userProfile = [Environment]::GetFolderPath("UserProfile")
    $shimDir = if ($env:XSCI_SHIM_DIR) {
        $env:XSCI_SHIM_DIR
    } else {
        Join-Path $userProfile ".xsci\bin"
    }
    New-Item -ItemType Directory -Force -Path $shimDir | Out-Null

    @"
@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python -X utf8 -m xsci.kaggle %*
"@ | Set-Content -Encoding ASCII (Join-Path $shimDir "evomind.cmd")

    @"
@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python -X utf8 -m xsci.kaggle %*
"@ | Set-Content -Encoding ASCII (Join-Path $shimDir "autokaggle.cmd")

    $legacyKaggleShim = Join-Path $shimDir "kaggle.cmd"
    if ($shouldInstallLegacyKaggleAlias) {
        @"
@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python -X utf8 -m xsci.kaggle %*
"@ | Set-Content -Encoding ASCII $legacyKaggleShim
    } elseif (Test-Path $legacyKaggleShim) {
        Remove-Item -LiteralPath $legacyKaggleShim -Force
        Write-Host "Removed legacy kaggle.cmd product alias. Use 'evomind' for the research agent."
    }

    @"
@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python -X utf8 -m xsci.kaggle official %*
"@ | Set-Content -Encoding ASCII (Join-Path $shimDir "kaggle-official.cmd")

    Write-Host ""
    Write-Host "Created command shims in %USERPROFILE%\.xsci\bin"

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
        Write-Host "Prepended .xsci\bin to the user PATH. New terminals can run 'evomind' directly."
    } else {
        Write-Host "EvoMind aliases created, but PATH was not changed because -NoPathPrepend was set."
    }
} else {
    Write-Host ""
    Write-Host "Skipped EvoMind aliases because -NoKaggleAlias was set."
}

Write-Host ""
Write-Host "Installed console commands:"
foreach ($cmd in @("evomind", "autokaggle", "kaggle-official", "xsci")) {
    $found = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($found) {
        Write-Host "  OK  $cmd"
    } else {
        Write-Host "  MISS $cmd (check that Python Scripts is on PATH)"
    }
}

Write-Host ""
Write-Host "Next:"
Write-Host "  evomind --help"
Write-Host "  evomind setup"
Write-Host "  evomind"
Write-Host "  evomind dashboard start"
Write-Host "  http://127.0.0.1:8088/?page=control"
Write-Host "  evomind official competitions list"
Write-Host ""
Write-Host "Note: use 'evomind' as the product command. Use 'kaggle-official' or 'evomind official ...' for the official Kaggle CLI."
