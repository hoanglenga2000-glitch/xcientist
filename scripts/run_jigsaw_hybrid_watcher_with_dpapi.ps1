[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$WatcherArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $projectRoot 'workspace\release-venv\Scripts\python.exe'
$watcher = Join-Path $projectRoot 'scripts\watch_jigsaw_hybrid_multiseed_completion.py'
$stateDirectory = Join-Path $env:APPDATA 'ResearchAgentWorkstation'
$credentialPath = Join-Path $stateDirectory 'hpc_ssh_credential.xml'
$metadataPath = Join-Path $stateDirectory 'hpc_ssh_metadata.json'

foreach ($path in @($python, $watcher, $credentialPath, $metadataPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required watcher dependency is missing: $path"
    }
}

$credential = Import-Clixml -LiteralPath $credentialPath
$metadata = Get-Content -LiteralPath $metadataPath -Raw -Encoding utf8 | ConvertFrom-Json
if ([string]$metadata.remote_workspace -ne '/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra') {
    throw 'HPC credential metadata escaped the dedicated remote root.'
}

$oldValues = @{}
foreach ($name in @(
    'GPU_SSH_HOST',
    'GPU_SSH_PORT',
    'GPU_SSH_USER',
    'GPU_SSH_PASSWORD',
    'GPU_SSH_SOCKS_HOST',
    'GPU_SSH_SOCKS_PORT',
    'PYTHONPATH'
)) {
    $oldValues[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

try {
    $env:GPU_SSH_HOST = [string]$metadata.host
    $env:GPU_SSH_PORT = [string]$metadata.port
    $env:GPU_SSH_USER = [string]$credential.UserName
    $env:GPU_SSH_PASSWORD = $credential.GetNetworkCredential().Password
    $env:GPU_SSH_SOCKS_HOST = [string]$metadata.socks_host
    $env:GPU_SSH_SOCKS_PORT = [string]$metadata.socks_port
    $env:PYTHONPATH = "$projectRoot;$projectRoot\src"
    & $python $watcher @WatcherArgs
    exit $LASTEXITCODE
} finally {
    foreach ($name in $oldValues.Keys) {
        $value = $oldValues[$name]
        if ($null -eq $value) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($name, $value, 'Process')
        }
    }
    $credential = $null
}
