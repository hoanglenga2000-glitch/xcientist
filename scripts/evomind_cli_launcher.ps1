$ErrorActionPreference = "Stop"

try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {
    # Best effort for legacy Windows PowerShell.
}

$commandArgs = @($args)
$configPath = Join-Path $PSScriptRoot "evomind-launcher.json"
if (Test-Path -LiteralPath $configPath) {
    $config = Get-Content -Raw -LiteralPath $configPath -Encoding UTF8 | ConvertFrom-Json
    $repoRoot = [System.IO.Path]::GetFullPath([string]$config.repo_root)
    $fallbackPython = [string]$config.python_executable
} else {
    $repoRoot = Split-Path -Parent $PSScriptRoot
    $fallbackPython = ""
}

if (-not [System.IO.Path]::IsPathRooted($repoRoot) -or
    -not (Test-Path -LiteralPath (Join-Path $repoRoot "pyproject.toml") -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $repoRoot "src\xsci\kaggle.py") -PathType Leaf)) {
    Write-Error "EVOMIND_LAUNCHER_FAILED: configured repository root is invalid: $repoRoot"
    exit 1
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($uv -and $uv.Source) {
    & $uv.Source run --project $repoRoot --directory $repoRoot python -X utf8 -m xsci.kaggle @commandArgs
    exit $LASTEXITCODE
}

if (-not $fallbackPython -or -not (Test-Path -LiteralPath $fallbackPython -PathType Leaf)) {
    Write-Error "EVOMIND_LAUNCHER_FAILED: uv is unavailable and the fallback Python is missing."
    exit 1
}
$sourceRoot = Join-Path $repoRoot "src"
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$sourceRoot;$env:PYTHONPATH" } else { $sourceRoot }
& $fallbackPython -X utf8 -m xsci.kaggle @commandArgs
exit $LASTEXITCODE
