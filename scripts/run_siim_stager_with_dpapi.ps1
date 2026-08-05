[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$StagerArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = 'D:\tools\hermes\hermes-agent\venv\Scripts\python.exe'
$stager = 'D:\EvoMind-MLE22-Local\scripts\stage_siim_public_from_kaggle.py'
$credentialPath = Join-Path $env:APPDATA 'ResearchAgentWorkstation\kaggle_api_token.xml'

foreach ($path in @($python, $stager, $credentialPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required SIIM stager dependency is missing: $path"
    }
}
if (-not $root.StartsWith('D:\桌面\codex\科研港科技', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Unexpected EvoMind project root'
}

$credential = Import-Clixml -LiteralPath $credentialPath
$secret = $credential.GetNetworkCredential().Password
if ([string]::IsNullOrWhiteSpace($secret)) {
    throw 'The DPAPI-protected Kaggle credential is empty'
}

$previousToken = $env:KAGGLE_API_TOKEN
$previousUsername = $env:KAGGLE_USERNAME
$previousKey = $env:KAGGLE_KEY
$previousUtf8 = $env:PYTHONUTF8
$exitCode = 1
try {
    if (
        $credential.UserName -eq '__KAGGLE_API_TOKEN__' -or
        $secret -match '^KGAT_[A-Za-z0-9_-]{16,}$'
    ) {
        $env:KAGGLE_API_TOKEN = $secret
        Remove-Item Env:KAGGLE_USERNAME, Env:KAGGLE_KEY -ErrorAction SilentlyContinue
    }
    else {
        Remove-Item Env:KAGGLE_API_TOKEN -ErrorAction SilentlyContinue
        $env:KAGGLE_USERNAME = $credential.UserName
        $env:KAGGLE_KEY = $secret
    }
    $env:PYTHONUTF8 = '1'
    & $python $stager @StagerArguments
    $exitCode = $LASTEXITCODE
}
finally {
    $secret = $null
    foreach ($name in @('KAGGLE_API_TOKEN', 'KAGGLE_USERNAME', 'KAGGLE_KEY', 'PYTHONUTF8')) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
    if ($null -ne $previousToken) { $env:KAGGLE_API_TOKEN = $previousToken }
    if ($null -ne $previousUsername) { $env:KAGGLE_USERNAME = $previousUsername }
    if ($null -ne $previousKey) { $env:KAGGLE_KEY = $previousKey }
    if ($null -ne $previousUtf8) { $env:PYTHONUTF8 = $previousUtf8 }
}
exit $exitCode
