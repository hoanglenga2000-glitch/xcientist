[CmdletBinding()]
param(
    [string]$OutputDir = "artifacts/browser-e2e",
    [int]$Port = 18198
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ArtifactsRoot = [IO.Path]::GetFullPath((Join-Path $Root "artifacts"))
$Output = [IO.Path]::GetFullPath((Join-Path $Root $OutputDir))
$ArtifactsPrefix = $ArtifactsRoot.TrimEnd([char]'\', [char]'/') + [IO.Path]::DirectorySeparatorChar
if (-not $Output.StartsWith($ArtifactsPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Browser E2E output must stay under $ArtifactsRoot"
}

New-Item -ItemType Directory -Force -Path $Output | Out-Null
$RuntimeRoot = Join-Path $Output "runtime-中文 空格"
if (Test-Path -LiteralPath $RuntimeRoot) {
    $ResolvedRuntime = [IO.Path]::GetFullPath($RuntimeRoot)
    $OutputPrefix = $Output.TrimEnd([char]'\', [char]'/') + [IO.Path]::DirectorySeparatorChar
    if (-not $ResolvedRuntime.StartsWith($OutputPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe browser E2E cleanup target: $ResolvedRuntime"
    }
    Remove-Item -LiteralPath $ResolvedRuntime -Recurse -Force
}
New-Item -ItemType Directory -Force -Path (Join-Path $RuntimeRoot "prisma") | Out-Null

$PreviousEnvironment = @{}
foreach ($Name in @(
    "WORKSTATION_ROOT", "WORKSTATION_DATA_DIR", "DATABASE_URL",
    "WORKSTATION_DISABLE_AGENT_EXECUTION", "OPENAI_API_KEY", "KAGGLE_USERNAME", "KAGGLE_KEY",
    "WORKSTATION_CDP_PORT", "WORKSTATION_CONTROL_AUDIT_CDP_PORT", "WORKSTATION_STATEFUL_CDP_PORT"
)) {
    $PreviousEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, "Process")
}

$Steps = [ordered]@{}
$Failure = $null
try {
    $env:WORKSTATION_ROOT = $RuntimeRoot
    $env:WORKSTATION_DATA_DIR = $RuntimeRoot
    $env:DATABASE_URL = "file:$((Join-Path $RuntimeRoot 'prisma\workstation.db').Replace('\','/'))"
    $env:WORKSTATION_DISABLE_AGENT_EXECUTION = "1"
    $env:OPENAI_API_KEY = ""
    $env:KAGGLE_USERNAME = ""
    $env:KAGGLE_KEY = ""
    $env:WORKSTATION_CDP_PORT = [string]($Port + 1000)
    $env:WORKSTATION_CONTROL_AUDIT_CDP_PORT = [string]($Port + 1001)
    $env:WORKSTATION_STATEFUL_CDP_PORT = [string]($Port + 1002)

    Push-Location (Join-Path $Root "web\research-agent-workstation")
    try {
        npx prisma migrate deploy
        $Steps.migrate = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($Steps.migrate -ne 0) { throw "Clean browser database migration failed." }

    python (Join-Path $Root "scripts\manage_workstation_dashboard.py") start --host 127.0.0.1 --port $Port --timeout 120
    $Steps.start = $LASTEXITCODE
    if ($Steps.start -ne 0) { throw "Browser E2E workstation start failed." }

    $BaseUrl = "http://127.0.0.1:$Port"
    python (Join-Path $Root "scripts\verify_workstation_runtime_navigation.py") --base-url $BaseUrl --timeout 30
    $Steps.navigation = $LASTEXITCODE
    node (Join-Path $Root "scripts\verify_workstation_click_smoke.mjs") --base-url $BaseUrl
    $Steps.click = $LASTEXITCODE
    node (Join-Path $Root "scripts\verify_workstation_interactive_controls.mjs") --base-url $BaseUrl
    $Steps.interactive = $LASTEXITCODE
    node (Join-Path $Root "scripts\verify_workstation_stateful_interactions.mjs") --base-url $BaseUrl
    $Steps.stateful = $LASTEXITCODE

    foreach ($Name in @("navigation", "click", "interactive", "stateful")) {
        if ($Steps[$Name] -ne 0) { throw "Browser E2E step failed: $Name" }
    }
} catch {
    $Failure = $_.Exception.Message
} finally {
    python (Join-Path $Root "scripts\manage_workstation_dashboard.py") stop --host 127.0.0.1 --port $Port --timeout 60
    $Steps.stop = $LASTEXITCODE
    if ($Steps.stop -ne 0 -and -not $Failure) { $Failure = "Managed workstation stop failed." }

    foreach ($Name in $PreviousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($Name, $PreviousEnvironment[$Name], "Process")
    }
}

$PortOpen = $null -ne (Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if ($PortOpen -and -not $Failure) { $Failure = "Browser E2E port remains open: $Port" }

$Result = [ordered]@{
    schema = "research_workstation.browser_release_e2e.v1"
    ok = -not [bool]$Failure
    port = $Port
    steps = $Steps
    port_released = -not $PortOpen
    failure = $Failure
}
$Result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Output "result.json") -Encoding utf8
$Result | ConvertTo-Json -Depth 5

if (Test-Path -LiteralPath $RuntimeRoot) {
    $ResolvedRuntime = [IO.Path]::GetFullPath($RuntimeRoot)
    $OutputPrefix = $Output.TrimEnd([char]'\', [char]'/') + [IO.Path]::DirectorySeparatorChar
    if (-not $ResolvedRuntime.StartsWith($OutputPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe final browser E2E cleanup target: $ResolvedRuntime"
    }
    Remove-Item -LiteralPath $ResolvedRuntime -Recurse -Force
}

if ($Failure) { throw $Failure }
