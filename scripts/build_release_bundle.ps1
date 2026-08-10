[CmdletBinding()]
param(
    [string]$OutputRoot = "",
    [switch]$ReplaceExisting,
    [switch]$SkipInstall,
    [switch]$SkipBuild,
    [switch]$SkipAudit,
    [switch]$SkipWheelhouse,
    [string]$WheelhouseSource = ""
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
Set-StrictMode -Version Latest

$SentinelName = ".evomind-release-root.json"
$SentinelSchema = "evomind.release.output-root.v1"
$SentinelOwner = "scripts/build_release_bundle.ps1"
$BuilderMetadataSchema = "evomind.release.builder-metadata.v1"

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$ArgumentList,
        [Parameter(Mandatory)][string]$Description
    )
    & $FilePath @ArgumentList
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Description failed with exit code $exitCode"
    }
}

function Invoke-NativeText {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$ArgumentList,
        [Parameter(Mandatory)][string]$Description
    )
    $output = @(& $FilePath @ArgumentList)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Description failed with exit code $exitCode"
    }
    return (($output | Out-String).Trim())
}

function Get-ContainedPath {
    param(
        [Parameter(Mandatory)][string]$Parent,
        [Parameter(Mandatory)][string]$Candidate,
        [switch]$AllowEqual
    )
    $parentFull = [IO.Path]::GetFullPath($Parent)
    $candidateFull = [IO.Path]::GetFullPath($Candidate)
    $relative = [IO.Path]::GetRelativePath($parentFull, $candidateFull)
    $segments = @($relative -split '[\\/]')
    if (
        [IO.Path]::IsPathRooted($relative) -or
        $relative -eq ".." -or
        ($segments.Count -gt 0 -and $segments[0] -eq "..") -or
        (-not $AllowEqual -and $relative -eq ".")
    ) {
        throw "Path escapes the allowed root '$parentFull': $candidateFull"
    }
    return $candidateFull
}

function Test-IsReparsePoint {
    param([Parameter(Mandatory)][string]$Path)
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if ($null -eq $item) { return $false }
    return (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)
}

function Assert-NoReparsePath {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Boundary
    )
    $full = [IO.Path]::GetFullPath($Path)
    $boundaryFull = [IO.Path]::GetFullPath($Boundary)
    [void](Get-ContainedPath -Parent $boundaryFull -Candidate $full -AllowEqual)
    $cursor = $full
    while ($true) {
        if (Test-IsReparsePoint -Path $cursor) {
            throw "Reparse points and junctions are forbidden in release paths: $cursor"
        }
        if ($cursor.Equals($boundaryFull, [StringComparison]::OrdinalIgnoreCase)) { break }
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $cursor) {
            throw "Release path did not reach containment boundary: $full"
        }
        $cursor = $parent
    }
}

function Assert-NoReparseTree {
    param([Parameter(Mandatory)][string]$Path)
    if (Test-IsReparsePoint -Path $Path) {
        throw "Reparse-point release tree is forbidden: $Path"
    }
    $reparse = Get-ChildItem -LiteralPath $Path -Recurse -Force -ErrorAction Stop |
        Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 } |
        Select-Object -First 1
    if ($null -ne $reparse) {
        throw "Reparse points are forbidden in release trees: $($reparse.FullName)"
    }
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory)][object]$Value,
        [Parameter(Mandatory)][string]$Path,
        [int]$Depth = 10
    )
    $Value | ConvertTo-Json -Depth $Depth | Set-Content -LiteralPath $Path -Encoding utf8
}

function Write-OutputSentinel {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$TransactionId,
        [Parameter(Mandatory)][string]$Target,
        [Parameter(Mandatory)][string]$State,
        [Parameter(Mandatory)][long]$SourceDateEpoch
    )
    Write-JsonFile -Path (Join-Path $Path $SentinelName) -Value ([ordered]@{
        schema = $SentinelSchema
        owner = $SentinelOwner
        transaction_id = $TransactionId
        target = [IO.Path]::GetFullPath($Target)
        state = $State
        source_date_epoch = $SourceDateEpoch
    })
}

function Read-OwnedOutputSentinel {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$ExpectedTarget
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Release output target is not a directory: $Path"
    }
    $sentinelPath = Join-Path $Path $SentinelName
    if (-not (Test-Path -LiteralPath $sentinelPath -PathType Leaf) -or (Test-IsReparsePoint -Path $sentinelPath)) {
        throw "Existing release output lacks a regular builder-owned sentinel: $sentinelPath"
    }
    $payload = Get-Content -LiteralPath $sentinelPath -Raw | ConvertFrom-Json
    if ($payload.schema -ne $SentinelSchema -or $payload.owner -ne $SentinelOwner) {
        throw "Existing release output sentinel is not owned by this builder: $sentinelPath"
    }
    $sentinelTarget = [IO.Path]::GetFullPath([string]$payload.target)
    $expectedFull = [IO.Path]::GetFullPath($ExpectedTarget)
    if (-not $sentinelTarget.Equals($expectedFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Existing release output sentinel target mismatch: $sentinelTarget"
    }
    if ([string]::IsNullOrWhiteSpace([string]$payload.transaction_id)) {
        throw "Existing release output sentinel has no transaction id"
    }
    return $payload
}

function Remove-OwnedStaging {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$TransactionId,
        [Parameter(Mandatory)][string]$Boundary
    )
    if (-not (Test-Path -LiteralPath $Path)) { return }
    Assert-NoReparsePath -Path $Path -Boundary $Boundary
    $payload = Read-OwnedOutputSentinel -Path $Path -ExpectedTarget $script:OutputRoot
    if ($payload.transaction_id -ne $TransactionId -or $payload.state -notin @("staging", "validated")) {
        throw "Refusing to clean staging not owned by transaction $TransactionId"
    }
    Remove-Item -LiteralPath $Path -Recurse -Force
}

function Copy-Tree {
    param(
        [Parameter(Mandatory)][string]$Source,
        [Parameter(Mandatory)][string]$Destination
    )
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) { return }
    Assert-NoReparseTree -Path $Source
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Get-ChildItem -LiteralPath $Source -Force | Copy-Item -Destination $Destination -Recurse -Force
}

function Copy-TrackedFile {
    param(
        [Parameter(Mandatory)][string]$Relative,
        [Parameter(Mandatory)][string]$SourceRoot,
        [Parameter(Mandatory)][string]$DestinationRoot
    )
    $source = Join-Path $SourceRoot $Relative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Tracked release source is missing: $Relative"
    }
    if (Test-IsReparsePoint -Path $source) {
        throw "Tracked release source is a reparse point: $Relative"
    }
    $destination = Join-Path $DestinationRoot $Relative
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
}

function Get-FileSetDigest {
    param(
        [Parameter(Mandatory)][string]$BasePath,
        [Parameter(Mandatory)][string[]]$RelativePaths
    )
    $normalized = @($RelativePaths | ForEach-Object { $_.Replace('\', '/') })
    [Array]::Sort($normalized, [StringComparer]::Ordinal)
    $aggregate = [Security.Cryptography.IncrementalHash]::CreateHash([Security.Cryptography.HashAlgorithmName]::SHA256)
    try {
        foreach ($relative in $normalized) {
            if ([string]::IsNullOrWhiteSpace($relative)) { continue }
            $path = Join-Path $BasePath $relative
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "Source digest input is missing: $relative"
            }
            if (Test-IsReparsePoint -Path $path) {
                throw "Source digest input is a reparse point: $relative"
            }
            $fileHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
            $size = (Get-Item -LiteralPath $path).Length
            $record = "$relative`0$fileHash`0$size`n"
            $aggregate.AppendData([Text.Encoding]::UTF8.GetBytes($record))
        }
        return [Convert]::ToHexString($aggregate.GetHashAndReset()).ToLowerInvariant()
    } finally {
        $aggregate.Dispose()
    }
}

function Get-TreeDigest {
    param([Parameter(Mandatory)][string]$Path)
    Assert-NoReparseTree -Path $Path
    $base = [IO.Path]::GetFullPath($Path)
    $relative = @(Get-ChildItem -LiteralPath $base -Recurse -File -Force | ForEach-Object {
        [IO.Path]::GetRelativePath($base, $_.FullName).Replace('\', '/')
    })
    return Get-FileSetDigest -BasePath $base -RelativePaths $relative
}

function Resolve-ApplicationPath {
    param([Parameter(Mandatory)][string]$Name)
    $commands = @(Get-Command $Name -CommandType Application -ErrorAction Stop)
    if ($commands.Count -eq 0) {
        throw "Required executable was not found: $Name"
    }
    $command = $commands | Select-Object -First 1
    $source = [string]$command.Source
    if ([string]::IsNullOrWhiteSpace($source)) {
        throw "Required executable has no Source path: $Name"
    }
    return $source
}

if ($SkipBuild) {
    throw "SkipBuild is disabled for release publishing; a fresh, receipt-bound Next build is required"
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Web = Join-Path $Root "web\research-agent-workstation"
$ArtifactsRoot = [IO.Path]::GetFullPath((Join-Path $Root "artifacts"))
if (-not (Test-Path -LiteralPath $ArtifactsRoot)) {
    New-Item -ItemType Directory -Path $ArtifactsRoot | Out-Null
}
if (-not (Test-Path -LiteralPath $ArtifactsRoot -PathType Container)) {
    throw "Artifacts root is not a directory: $ArtifactsRoot"
}
Assert-NoReparsePath -Path $ArtifactsRoot -Boundary $Root

if (-not $OutputRoot) {
    $OutputRoot = Join-Path $ArtifactsRoot "release-dist"
}
$OutputRoot = Get-ContainedPath -Parent $ArtifactsRoot -Candidate $OutputRoot
$OutputParent = Split-Path -Parent $OutputRoot
if (-not (Test-Path -LiteralPath $OutputParent)) {
    [IO.Directory]::CreateDirectory($OutputParent) | Out-Null
}
Assert-NoReparsePath -Path $OutputParent -Boundary $ArtifactsRoot

$Git = Resolve-ApplicationPath "git"
$Node = Resolve-ApplicationPath "node"
$Npm = Resolve-ApplicationPath "npm"
$PythonCommand = Resolve-ApplicationPath "python"
$Uv = Resolve-ApplicationPath "uv"

$ToolchainContractPath = Join-Path $Root "configs\release\toolchain.json"
$ReleaseContractPath = Join-Path $Root "configs\release\release-contract.json"
$UvLockPath = Join-Path $Root "uv.lock"
$RequiredTracked = @(
    "configs/release/toolchain.json",
    "configs/release/release-contract.json",
    "pyproject.toml",
    "uv.lock",
    "web/research-agent-workstation/package.json",
    "web/research-agent-workstation/package-lock.json",
    "scripts/build_release_bundle.ps1",
    "scripts/release_finalize.py",
    "scripts/release_db_migrate.py",
    "scripts/verify_release_bundle.py",
    "scripts/verify_openai_gateway.py",
    "scripts/apply_workstation_migrations.py",
    "scripts/manage_local_gateway.py",
    "scripts/reconcile_action_log_mirror.py",
    "scripts/manage_workstation_dashboard.py",
    "scripts/workstation_lifecycle.py",
    "scripts/start_verified_workstation.ps1",
    "install.ps1", "start.ps1", "stop.ps1", "status.ps1",
    "migrate.ps1", "upgrade.ps1", "rollback.ps1", "uninstall.ps1"
)
foreach ($relative in $RequiredTracked) {
    [void](Invoke-NativeText -FilePath $Git -ArgumentList @(
        "-C", $Root, "ls-files", "--error-unmatch", "--", $relative
    ) -Description "git tracked-source check for $relative")
}

$TrackedText = Invoke-NativeText -FilePath $Git -ArgumentList @(
    "-C", $Root, "-c", "core.quotepath=false", "ls-files"
) -Description "git tracked source enumeration"
$TrackedFiles = @($TrackedText -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
$SourceStatus = Invoke-NativeText -FilePath $Git -ArgumentList @(
    "-C", $Root, "status", "--porcelain=v1", "--untracked-files=all"
) -Description "clean release source check"
if (-not [string]::IsNullOrWhiteSpace($SourceStatus)) {
    throw "Release builds require a clean source tree: $SourceStatus"
}
$TrackedFileSet = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($relative in $TrackedFiles) { [void]$TrackedFileSet.Add($relative.Replace('\', '/')) }
$SourceDigest = Get-FileSetDigest -BasePath $Root -RelativePaths $TrackedFiles

$Commit = (Invoke-NativeText -FilePath $Git -ArgumentList @("-C", $Root, "rev-parse", "HEAD") -Description "git commit lookup").Trim()
$EpochText = (Invoke-NativeText -FilePath $Git -ArgumentList @(
    "-C", $Root, "show", "-s", "--format=%ct", "HEAD"
) -Description "git source epoch lookup").Trim()
$Epoch = [int64]$EpochText

$Toolchain = Get-Content -LiteralPath $ToolchainContractPath -Raw | ConvertFrom-Json
if ($Toolchain.schema -ne "evomind.release.toolchain.v1") {
    throw "Unsupported release toolchain contract schema: $($Toolchain.schema)"
}
$NodeVersion = (Invoke-NativeText -FilePath $Node -ArgumentList @("--version") -Description "node version check").Trim()
if ($NodeVersion.StartsWith("v", [StringComparison]::OrdinalIgnoreCase)) { $NodeVersion = $NodeVersion.Substring(1) }
$NpmVersion = (Invoke-NativeText -FilePath $Npm -ArgumentList @("--version") -Description "npm version check").Trim()
$PythonInfoText = Invoke-NativeText -FilePath $PythonCommand -ArgumentList @(
    "-c",
    "import json,platform,sys,sysconfig; print(json.dumps({'executable':sys.executable,'base_executable':getattr(sys,'_base_executable',sys.executable),'version':platform.python_version(),'implementation':platform.python_implementation(),'abi':'cp%d%d'%sys.version_info[:2],'platform':'win_amd64' if sys.maxsize > 2**32 and sys.platform == 'win32' else sysconfig.get_platform()}))"
) -Description "python runtime identity check"
$PythonInfo = $PythonInfoText | ConvertFrom-Json
$BuildPython = [IO.Path]::GetFullPath([string]$PythonInfo.executable)
$BaseExecutable = [string]$PythonInfo.base_executable
if ([string]::IsNullOrWhiteSpace($BaseExecutable)) { $BaseExecutable = $BuildPython }
$HostPython = [IO.Path]::GetFullPath($BaseExecutable)
$HostPythonVersion = [string]$PythonInfo.version
$UvVersionText = (Invoke-NativeText -FilePath $Uv -ArgumentList @("--version") -Description "uv version check").Trim()
$UvVersion = @($UvVersionText -split '\s+')[1]

$ActualToolchain = [ordered]@{
    node = $NodeVersion
    npm = $NpmVersion
    python = $HostPythonVersion
    python_implementation = [string]$PythonInfo.implementation
    python_abi = [string]$PythonInfo.abi
    python_platform = [string]$PythonInfo.platform
    uv = $UvVersion
}
$ExpectedToolchain = [ordered]@{
    node = [string]$Toolchain.node.version
    npm = [string]$Toolchain.npm.version
    python = [string]$Toolchain.python.version
    python_implementation = [string]$Toolchain.python.implementation
    python_abi = [string]$Toolchain.python.abi
    python_platform = [string]$Toolchain.python.platform
    uv = [string]$Toolchain.uv.version
}
foreach ($name in $ExpectedToolchain.Keys) {
    if ($ActualToolchain[$name] -cne $ExpectedToolchain[$name]) {
        throw "Release toolchain mismatch for ${name}: expected '$($ExpectedToolchain[$name])', got '$($ActualToolchain[$name])'"
    }
}
[void](Invoke-NativeText -FilePath $BuildPython -ArgumentList @("-m", "pip", "--version") -Description "pip availability check")

$Package = Get-Content -LiteralPath (Join-Path $Web "package.json") -Raw | ConvertFrom-Json
$Version = [string]$Package.version
if ([string]$Package.packageManager -cne "npm@$NpmVersion") {
    throw "package.json packageManager must exactly match the release npm: npm@$NpmVersion"
}

$ExistingOutput = Test-Path -LiteralPath $OutputRoot
if ($ExistingOutput) {
    Assert-NoReparsePath -Path $OutputRoot -Boundary $ArtifactsRoot
    [void](Read-OwnedOutputSentinel -Path $OutputRoot -ExpectedTarget $OutputRoot)
    if (-not $ReplaceExisting) {
        throw "Existing builder-owned output requires the explicit -ReplaceExisting switch: $OutputRoot"
    }
} elseif ($ReplaceExisting) {
    throw "-ReplaceExisting was supplied but the release output does not exist: $OutputRoot"
}

$TransactionId = [guid]::NewGuid().ToString("N")
$StagingRoot = Join-Path $OutputParent ".evomind-release-staging-$TransactionId"
if (Test-Path -LiteralPath $StagingRoot) {
    throw "Random release staging path already exists: $StagingRoot"
}
New-Item -ItemType Directory -Path $StagingRoot | Out-Null
Write-OutputSentinel -Path $StagingRoot -TransactionId $TransactionId -Target $OutputRoot -State "staging" -SourceDateEpoch $Epoch
Assert-NoReparsePath -Path $StagingRoot -Boundary $ArtifactsRoot

$Bundle = Join-Path $StagingRoot "EvoMind"
$Zip = Join-Path $StagingRoot "EvoMind-win-x64-$Version.zip"
$Result = Join-Path $StagingRoot "build-result.json"
$Metadata = Join-Path $Bundle "metadata"
$Runtime = Join-Path $Bundle "runtime"
$Python = Join-Path $Runtime "python"
$WheelDir = Join-Path $Runtime "wheels"
$PipReport = Join-Path $StagingRoot ".pip-wheelhouse-validation.json"
$Published = $false
$RollbackRoot = $null

try {
    New-Item -ItemType Directory -Path $Bundle | Out-Null
    New-Item -ItemType Directory -Force -Path $Metadata | Out-Null

    $PreviousSourceDateEpoch = $env:SOURCE_DATE_EPOCH
    $PreviousNextTelemetry = $env:NEXT_TELEMETRY_DISABLED
    try {
        $env:SOURCE_DATE_EPOCH = [string]$Epoch
        $env:NEXT_TELEMETRY_DISABLED = "1"
        Push-Location $Web
        try {
            if (-not $SkipInstall) {
                Invoke-NativeCommand -FilePath $Npm -ArgumentList @(
                    "ci", "--registry=https://registry.npmjs.org"
                ) -Description "npm ci"
            }
            if (-not $SkipAudit) {
                Invoke-NativeCommand -FilePath $Npm -ArgumentList @("run", "audit:prod") -Description "npm production audit"
                Invoke-NativeCommand -FilePath $Npm -ArgumentList @("run", "audit:all") -Description "npm full audit"
            }
            Invoke-NativeCommand -FilePath $Npm -ArgumentList @("run", "db:generate") -Description "npm Prisma generation"
            Invoke-NativeCommand -FilePath $Npm -ArgumentList @("run", "typecheck") -Description "npm typecheck"
            Invoke-NativeCommand -FilePath $Npm -ArgumentList @("run", "lint") -Description "npm lint"
            Invoke-NativeCommand -FilePath $Npm -ArgumentList @("run", "build") -Description "npm Next build"
        } finally {
            Pop-Location
        }
    } finally {
        $env:SOURCE_DATE_EPOCH = $PreviousSourceDateEpoch
        $env:NEXT_TELEMETRY_DISABLED = $PreviousNextTelemetry
    }
    foreach ($requiredBuildPath in @(
        (Join-Path $Web ".next\BUILD_ID"),
        (Join-Path $Web ".next\standalone\server.js"),
        (Join-Path $Web ".next\static")
    )) {
        if (-not (Test-Path -LiteralPath $requiredBuildPath)) {
            throw "Fresh Next build output is incomplete: $requiredBuildPath"
        }
    }

    $App = Join-Path $Bundle "app"
    Copy-Tree -Source (Join-Path $Web ".next\standalone") -Destination $App
    Copy-Tree -Source (Join-Path $Web ".next\static") -Destination (Join-Path $App ".next\static")
    Copy-Tree -Source (Join-Path $Web "public") -Destination (Join-Path $App "public")
    foreach ($relative in $TrackedFiles | Where-Object { $_ -like "web/research-agent-workstation/prisma/*" }) {
        $prismaRelative = $relative.Substring("web/research-agent-workstation/".Length)
        Copy-TrackedFile -Relative $relative -SourceRoot $Root -DestinationRoot (Join-Path $App "..\_repo_copy")
        $temporaryCopy = Join-Path (Join-Path $App "..\_repo_copy") $relative
        $prismaDestination = Join-Path $App $prismaRelative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $prismaDestination) | Out-Null
        Move-Item -LiteralPath $temporaryCopy -Destination $prismaDestination -Force
    }
    $temporaryRepoCopy = Join-Path $Bundle "_repo_copy"
    if (Test-Path -LiteralPath $temporaryRepoCopy) {
        Remove-Item -LiteralPath $temporaryRepoCopy -Recurse -Force
    }
    $CopiedEnv = Join-Path $App ".env"
    if (Test-Path -LiteralPath $CopiedEnv) {
        throw "Standalone build unexpectedly contains a forbidden .env file"
    }

    if (-not (Test-Path -LiteralPath $HostPython -PathType Leaf)) {
        throw "Python runtime path is invalid: $HostPython"
    }
    $HostPythonRoot = Split-Path -Parent $HostPython
    New-Item -ItemType Directory -Force -Path $Python | Out-Null
    Get-ChildItem -LiteralPath $HostPythonRoot -File | Where-Object {
        $_.Name -in @("python.exe", "pythonw.exe", "python3.dll", "python312.dll", "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt")
    } | Copy-Item -Destination $Python -Force
    Copy-Tree -Source (Join-Path $HostPythonRoot "DLLs") -Destination (Join-Path $Python "DLLs")
    $HostLib = Join-Path $HostPythonRoot "Lib"
    Assert-NoReparseTree -Path $HostLib
    Get-ChildItem -LiteralPath $HostLib -Recurse -File | Where-Object {
        $_.FullName -notmatch '[\\/]site-packages[\\/]' -and $_.FullName -notmatch '[\\/]__pycache__[\\/]'
    } | ForEach-Object {
        $relative = [IO.Path]::GetRelativePath($HostLib, $_.FullName)
        $destination = Join-Path (Join-Path $Python "Lib") $relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $destination -Force
    }
    foreach ($relative in $TrackedFiles | Where-Object {
        $_ -match '^src/(xsci|research_os|research_agent_workstation|evomind_runtime)/'
    }) {
        Copy-TrackedFile -Relative $relative -SourceRoot $Root -DestinationRoot $Python
    }
    foreach ($name in @("pyproject.toml", "uv.lock", "requirements.txt", "requirements-dev.txt")) {
        if ($TrackedFileSet.Contains($name)) {
            Copy-TrackedFile -Relative $name -SourceRoot $Root -DestinationRoot $Python
        }
    }

    New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null
    $LockedRequirements = Join-Path $Metadata "python-locked-requirements.txt"
    Invoke-NativeCommand -FilePath $Uv -ArgumentList @(
        "export", "--frozen", "--no-dev", "--no-emit-project", "--no-annotate", "--no-header",
        "--format", "requirements.txt", "--python", $HostPythonVersion, "--output-file", $LockedRequirements
    ) -Description "uv locked dependency export"
    if (-not (Test-Path -LiteralPath $LockedRequirements -PathType Leaf) -or
        (Get-Item -LiteralPath $LockedRequirements).Length -eq 0) {
        throw "uv produced an empty locked requirements export"
    }

    if ($SkipWheelhouse) {
        if ([string]::IsNullOrWhiteSpace($WheelhouseSource)) {
            throw "SkipWheelhouse requires -WheelhouseSource"
        }
        $ResolvedWheelhouse = [IO.Path]::GetFullPath($WheelhouseSource)
        if (-not (Test-Path -LiteralPath $ResolvedWheelhouse -PathType Container)) {
            throw "WheelhouseSource does not exist: $ResolvedWheelhouse"
        }
        Assert-NoReparseTree -Path $ResolvedWheelhouse
        foreach ($wheel in Get-ChildItem -LiteralPath $ResolvedWheelhouse -Filter "*.whl" -File) {
            if (-not $wheel.Name.StartsWith("xcientist-$Version-", [StringComparison]::OrdinalIgnoreCase)) {
                Copy-Item -LiteralPath $wheel.FullName -Destination (Join-Path $WheelDir $wheel.Name)
            }
        }
    } else {
        $PreviousPipIndex = $env:PIP_INDEX_URL
        try {
            $env:PIP_INDEX_URL = "https://pypi.org/simple"
            Invoke-NativeCommand -FilePath $BuildPython -ArgumentList @(
                "-m", "pip", "download", "--disable-pip-version-check", "--dest", $WheelDir,
                "--only-binary=:all:", "--platform", "win_amd64", "--implementation", "cp",
                "--python-version", "312", "--abi", "cp312", "--require-hashes",
                "--index-url", "https://pypi.org/simple", "-r", $LockedRequirements
            ) -Description "hash-locked Python wheelhouse download"
        } finally {
            $env:PIP_INDEX_URL = $PreviousPipIndex
        }
    }

    $PreviousSourceDateEpoch = $env:SOURCE_DATE_EPOCH
    try {
        $env:SOURCE_DATE_EPOCH = [string]$Epoch
        Invoke-NativeCommand -FilePath $BuildPython -ArgumentList @(
            "-m", "pip", "wheel", "--disable-pip-version-check", "--no-deps", "--no-cache-dir",
            "--wheel-dir", $WheelDir, $Root
        ) -Description "first-party project wheel build"
    } finally {
        $env:SOURCE_DATE_EPOCH = $PreviousSourceDateEpoch
    }
    $ProjectWheels = @(Get-ChildItem -LiteralPath $WheelDir -Filter "xcientist-$Version-*.whl" -File)
    if ($ProjectWheels.Count -ne 1) {
        throw "Exactly one first-party xcientist $Version wheel is required; found $($ProjectWheels.Count)"
    }
    Invoke-NativeCommand -FilePath $BuildPython -ArgumentList @(
        "-m", "pip", "install", "--disable-pip-version-check", "--dry-run", "--ignore-installed",
        "--no-index", "--find-links", $WheelDir, "--only-binary=:all:", "--require-hashes",
        "--report", $PipReport, "-r", $LockedRequirements
    ) -Description "offline wheelhouse exact-set validation"
    if (-not (Test-Path -LiteralPath $PipReport -PathType Leaf)) {
        throw "pip did not produce the wheelhouse validation report"
    }

    $NodeDir = Join-Path $Runtime "node"
    New-Item -ItemType Directory -Force -Path $NodeDir | Out-Null
    Copy-Item -LiteralPath $Node -Destination (Join-Path $NodeDir "node.exe") -Force
    $NodeLicense = Join-Path (Split-Path -Parent $Node) "LICENSE"
    if (Test-Path -LiteralPath $NodeLicense -PathType Leaf) {
        Copy-Item -LiteralPath $NodeLicense -Destination (Join-Path $NodeDir "LICENSE") -Force
    }
    $BundledNodeVersion = (Invoke-NativeText -FilePath (Join-Path $NodeDir "node.exe") -ArgumentList @(
        "--version"
    ) -Description "bundled node version check").Trim()
    if ($BundledNodeVersion.StartsWith("v", [StringComparison]::OrdinalIgnoreCase)) {
        $BundledNodeVersion = $BundledNodeVersion.Substring(1)
    }
    if ($BundledNodeVersion -cne $NodeVersion) {
        throw "Bundled node version drifted after copy: $BundledNodeVersion"
    }

    $allowedExtensions = @(".py", ".ps1", ".mjs", ".js", ".cmd", ".bat")
    foreach ($relative in $TrackedFiles | Where-Object { $_ -like "scripts/*" }) {
        if ($allowedExtensions.Contains([IO.Path]::GetExtension($relative).ToLowerInvariant())) {
            Copy-TrackedFile -Relative $relative -SourceRoot $Root -DestinationRoot $Bundle
        }
    }
    foreach ($relative in $TrackedFiles | Where-Object { $_ -like "configs/*" }) {
        Copy-TrackedFile -Relative $relative -SourceRoot $Root -DestinationRoot (Join-Path $Bundle "config")
    }
    foreach ($relative in @(".env.example", "README.md", "docs/NEW_USER_ONBOARDING_GUIDE.md")) {
        if ($TrackedFileSet.Contains($relative)) {
            Copy-TrackedFile -Relative $relative -SourceRoot $Root -DestinationRoot $Bundle
        }
    }
    foreach ($entrypoint in @(
        "install.ps1", "start.ps1", "stop.ps1", "status.ps1",
        "migrate.ps1", "upgrade.ps1", "rollback.ps1", "uninstall.ps1"
    )) {
        Copy-TrackedFile -Relative $entrypoint -SourceRoot $Root -DestinationRoot $Bundle
    }

    # Next/sharp may materialize optional wasm-only packages on Windows during build.
    # They are not part of the locked runtime tree and make `npm ls --all` fail as
    # extraneous. Remove only the observed optional extras before SBOM/tree capture.
    foreach ($relativeExtraneous in @("@emnapi\runtime", "@img\sharp-wasm32")) {
        $extraneousPath = Join-Path (Join-Path $Web "node_modules") $relativeExtraneous
        if (Test-Path -LiteralPath $extraneousPath) {
            Remove-Item -LiteralPath $extraneousPath -Recurse -Force
        }
    }

    Push-Location $Web
    try {
        $SbomText = Invoke-NativeText -FilePath $Npm -ArgumentList @(
            "sbom", "--sbom-format=cyclonedx", "--registry=https://registry.npmjs.org"
        ) -Description "npm CycloneDX SBOM generation"
        [void]($SbomText | ConvertFrom-Json)
        $SbomText | Set-Content -LiteralPath (Join-Path $Metadata "node-sbom.cdx.json") -Encoding utf8

        $DependencyTreeText = Invoke-NativeText -FilePath $Npm -ArgumentList @(
            "ls", "--all", "--json"
        ) -Description "npm dependency tree validation"
        $DependencyTree = $DependencyTreeText | ConvertFrom-Json
        $ProblemsProperty = $DependencyTree.PSObject.Properties["problems"]
        $Problems = @()
        if ($null -ne $ProblemsProperty) {
            $Problems = @($ProblemsProperty.Value) | Where-Object {
                -not [string]::IsNullOrWhiteSpace([string]$_)
            }
        }
        if ($Problems.Count -gt 0) {
            throw "npm ls reported dependency problems: $($Problems -join '; ')"
        }
        $DependencyTreeText | Set-Content -LiteralPath (Join-Path $Metadata "node-dependency-tree.json") -Encoding utf8
    } finally {
        Pop-Location
    }

    Copy-Item -LiteralPath (Join-Path $Web "package-lock.json") -Destination (Join-Path $Metadata "package-lock.json") -Force
    Copy-Item -LiteralPath $ToolchainContractPath -Destination (Join-Path $Metadata "toolchain.json") -Force
    Copy-Item -LiteralPath $ReleaseContractPath -Destination (Join-Path $Metadata "release-contract.json") -Force

    $BuildId = (Get-Content -LiteralPath (Join-Path $App ".next\BUILD_ID") -Raw).Trim()
    if ([string]::IsNullOrWhiteSpace($BuildId)) { throw "Staged Next BUILD_ID is empty" }
    $NextTreeDigest = Get-TreeDigest -Path (Join-Path $App ".next")
    $ToolchainContractHash = (Get-FileHash -LiteralPath $ToolchainContractPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $ReleaseContractHash = (Get-FileHash -LiteralPath $ReleaseContractPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $PackageLockHash = (Get-FileHash -LiteralPath (Join-Path $Web "package-lock.json") -Algorithm SHA256).Hash.ToLowerInvariant()
    $UvLockHash = (Get-FileHash -LiteralPath $UvLockPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $LockedRequirementsHash = (Get-FileHash -LiteralPath $LockedRequirements -Algorithm SHA256).Hash.ToLowerInvariant()
    $BuilderMetadataPath = Join-Path $Metadata "builder-metadata.json"
    Write-JsonFile -Path $BuilderMetadataPath -Depth 12 -Value ([ordered]@{
        schema = $BuilderMetadataSchema
        builder = $SentinelOwner
        git_commit = $Commit
        source_date_epoch = $Epoch
        source_digest = $SourceDigest
        source_dirty = $false
        build_id = $BuildId
        next_tree_sha256 = $NextTreeDigest
        toolchain_contract_sha256 = $ToolchainContractHash
        release_contract_sha256 = $ReleaseContractHash
        package_lock_sha256 = $PackageLockHash
        uv_lock_sha256 = $UvLockHash
        locked_requirements_sha256 = $LockedRequirementsHash
        toolchain = [ordered]@{
            node = $NodeVersion
            npm = $NpmVersion
            python = $HostPythonVersion
            python_implementation = [string]$PythonInfo.implementation
            python_abi = [string]$PythonInfo.abi
            python_platform = [string]$PythonInfo.platform
            uv = $UvVersion
            node_binary_sha256 = (Get-FileHash -LiteralPath (Join-Path $NodeDir "node.exe") -Algorithm SHA256).Hash.ToLowerInvariant()
            python_binary_sha256 = (Get-FileHash -LiteralPath (Join-Path $Python "python.exe") -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        commands = @(
            "npm ci", "npm run audit:prod", "npm run audit:all", "npm run db:generate",
            "npm run typecheck", "npm run lint", "npm run build", "uv export --frozen",
            "python -m pip download --require-hashes", "python -m pip wheel --no-deps",
            "python -m pip install --dry-run --require-hashes", "npm sbom", "npm ls --all --json"
        )
        options = [ordered]@{
            skip_install = [bool]$SkipInstall
            skip_audit = [bool]$SkipAudit
            reused_wheelhouse = [bool]$SkipWheelhouse
        }
    })

    Invoke-NativeCommand -FilePath $BuildPython -ArgumentList @(
        (Join-Path $Root "scripts\release_finalize.py"),
        "--bundle", $Bundle,
        "--zip", $Zip,
        "--version", $Version,
        "--project-name", "xcientist",
        "--commit", $Commit,
        "--source-date-epoch", [string]$Epoch,
        "--node-version", $BundledNodeVersion,
        "--python-version", "$HostPythonVersion/cp312/win_amd64/bundled",
        "--builder-metadata", $BuilderMetadataPath,
        "--uv-lock", $UvLockPath,
        "--locked-requirements", $LockedRequirements,
        "--pip-report", $PipReport,
        "--wheelhouse", $WheelDir,
        "--max-mib", "500",
        "--result", $Result
    ) -Description "release finalization"

    $StageResult = Get-Content -LiteralPath $Result -Raw | ConvertFrom-Json
    if (-not $StageResult.ok) { throw "Release finalizer did not report success" }
    if ((Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLowerInvariant() -cne [string]$StageResult.zip_sha256) {
        throw "Release ZIP hash changed after finalization"
    }
    if ((Get-FileHash -LiteralPath (Join-Path $Bundle "release-manifest.json") -Algorithm SHA256).Hash.ToLowerInvariant() -cne
        [string]$StageResult.manifest_sha256) {
        throw "Release manifest hash changed after finalization"
    }
    Assert-NoReparseTree -Path $StagingRoot
    Write-OutputSentinel -Path $StagingRoot -TransactionId $TransactionId -Target $OutputRoot -State "validated" -SourceDateEpoch $Epoch

    Assert-NoReparsePath -Path $OutputParent -Boundary $ArtifactsRoot
    Assert-NoReparsePath -Path $StagingRoot -Boundary $ArtifactsRoot
    if ($ExistingOutput) {
        Assert-NoReparsePath -Path $OutputRoot -Boundary $ArtifactsRoot
        [void](Read-OwnedOutputSentinel -Path $OutputRoot -ExpectedTarget $OutputRoot)
        $rollbackSuffix = [DateTimeOffset]::UtcNow.ToString("yyyyMMddTHHmmssZ")
        $RollbackRoot = Join-Path $OutputParent ".evomind-release-rollback-$rollbackSuffix-$TransactionId"
        if (Test-Path -LiteralPath $RollbackRoot) { throw "Rollback path collision: $RollbackRoot" }
        [IO.Directory]::Move($OutputRoot, $RollbackRoot)
        try {
            [IO.Directory]::Move($StagingRoot, $OutputRoot)
            $Published = $true
        } catch {
            $publishFailure = $_
            if (Test-Path -LiteralPath $OutputRoot) {
                throw "Atomic release switch failed and target path is unexpectedly occupied: $publishFailure"
            }
            [IO.Directory]::Move($RollbackRoot, $OutputRoot)
            $RollbackRoot = $null
            throw $publishFailure
        }
    } else {
        [IO.Directory]::Move($StagingRoot, $OutputRoot)
        $Published = $true
    }

    $PublishedUtc = [DateTimeOffset]::UtcNow.ToString("o")
    $PublishedResultPath = Join-Path $OutputRoot "build-result.json"
    $PublishedResult = Get-Content -LiteralPath $PublishedResultPath -Raw | ConvertFrom-Json
    $PublishedResult.bundle = Join-Path $OutputRoot "EvoMind"
    $PublishedResult.zip = Join-Path $OutputRoot "EvoMind-win-x64-$Version.zip"
    $PublishedResult | Add-Member -NotePropertyName published_utc -NotePropertyValue $PublishedUtc -Force
    $PublishedResult | Add-Member -NotePropertyName rollback -NotePropertyValue $RollbackRoot -Force
    Write-JsonFile -Path $PublishedResultPath -Value $PublishedResult -Depth 12
    $PublishedSentinel = Read-OwnedOutputSentinel -Path $OutputRoot -ExpectedTarget $OutputRoot
    $PublishedSentinel.state = "published"
    $PublishedSentinel | Add-Member -NotePropertyName published_utc -NotePropertyValue $PublishedUtc -Force
    $PublishedSentinel | Add-Member -NotePropertyName rollback -NotePropertyValue $RollbackRoot -Force
    Write-JsonFile -Path (Join-Path $OutputRoot $SentinelName) -Value $PublishedSentinel -Depth 10
    Get-Content -LiteralPath $PublishedResultPath -Raw
} catch {
    $failure = $_
    if (-not $Published -and (Test-Path -LiteralPath $StagingRoot)) {
        try {
            Remove-OwnedStaging -Path $StagingRoot -TransactionId $TransactionId -Boundary $ArtifactsRoot
        } catch {
            Write-Warning "Failed to clean owned staging '$StagingRoot': $_"
        }
    }
    throw $failure
}
