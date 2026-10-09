param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [Parameter(Mandatory)][string]$RemoteRoot,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedScriptSha256,
  [switch]$PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = 'C:\ProgramData\EvoMind'
$expectedRemote = [IO.Path]::GetFullPath("C:\SecureInput\seal-release-scope-$TransactionId")
if ([IO.Path]::GetFullPath($RemoteRoot) -cne $expectedRemote) { throw 'SEAL_SCOPE_REMOTE_ROOT_REJECTED' }

$target = Join-Path $root 'bundle\scripts\Seal-Release.ps1'
$candidate = Join-Path $expectedRemote 'Seal-Release.candidate.ps1'
$resultPath = Join-Path $expectedRemote "transaction-$TransactionId-result.json"
$errorPath = Join-Path $expectedRemote "transaction-$TransactionId-error.json"

function Get-Sha256([string]$Path) {
  (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-Utf8Json([string]$Path, [object]$Value) {
  [IO.File]::WriteAllText($Path, (($Value | ConvertTo-Json -Depth 8) + "`n"), [Text.UTF8Encoding]::new($false))
}

function Assert-PowerShellAst([string]$Path) {
  $tokens = $null
  $errors = $null
  [Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors) | Out-Null
  if (@($errors).Count) { throw 'SEAL_SCOPE_CANDIDATE_AST_REJECTED' }
}

function Read-BindingHash {
  $binding = Join-Path $root 'byoa\tenants\tenant_ba0ef9d3767f2fb385b856e0\hpc-binding.json'
  if (-not (Test-Path -LiteralPath $binding -PathType Leaf)) { throw 'SEAL_SCOPE_BINDING_MISSING' }
  Get-Sha256 $binding
}

function Read-LoopbackHealth {
  $response = Invoke-WebRequest 'http://127.0.0.1:8088/api/healthz' -UseBasicParsing -TimeoutSec 15
  $payload = $response.Content | ConvertFrom-Json
  if ([int]$response.StatusCode -ne 200 -or -not [bool]$payload.ok -or [string]$payload.status -ne 'ready' -or @($payload.failures).Count) {
    throw 'SEAL_SCOPE_LOOPBACK_HEALTH_REJECTED'
  }
  [ordered]@{ status_code = [int]$response.StatusCode; status = [string]$payload.status; build_id = [string]$payload.build_id }
}

$oldBlock = @'
$immutableRoots = @('C:\ProgramData\EvoMind\bin','C:\ProgramData\EvoMind\pki','C:\ProgramData\EvoMind\trust','C:\ProgramData\EvoMind\public-edge','C:\ProgramData\EvoMind\web-overlays') | Where-Object { Test-Path -LiteralPath $_ -PathType Container }
foreach ($root in $immutableRoots) {
  Set-SealedTreeAcl -Path $root -RuntimeRights 'RX' -FailureCode "IMMUTABLE_RUNTIME_ROOT_ACL_FAILED:$root"
}
'@

$newBlock = @'
$immutableRoots = @('C:\ProgramData\EvoMind\bin','C:\ProgramData\EvoMind\pki','C:\ProgramData\EvoMind\trust','C:\ProgramData\EvoMind\public-edge') | Where-Object { Test-Path -LiteralPath $_ -PathType Container }
foreach ($root in $immutableRoots) {
  Set-SealedTreeAcl -Path $root -RuntimeRights 'RX' -FailureCode "IMMUTABLE_RUNTIME_ROOT_ACL_FAILED:$root"
}

# Web overlays are append-only and each newly installed overlay already receives
# an inheritable, immutable ACL before publication. Re-seal only the parent root
# and validate the active overlay identity; never recurse through historical
# overlays during a deployment.
$webOverlayRoot = 'C:\ProgramData\EvoMind\web-overlays'
if (Test-Path -LiteralPath $webOverlayRoot -PathType Container) {
  Invoke-AclNative -FilePath 'icacls.exe' -ArgumentList @($webOverlayRoot,'/inheritance:r','/grant:r',"${account}:(OI)(CI)RX","${system}:(OI)(CI)F","${administrators}:(OI)(CI)F") -FailureCode 'WEB_OVERLAY_ROOT_ACL_FAILED'
  $activeWebRoot = Resolve-ConfigPath ([string]$config.web_runtime_root)
  Assert-PathUnderRoot -Path $activeWebRoot -Root $webOverlayRoot | Out-Null
  if ([IO.Path]::GetFullPath($activeWebRoot).TrimEnd('\') -ieq [IO.Path]::GetFullPath($webOverlayRoot).TrimEnd('\')) { throw 'WEB_OVERLAY_ACTIVE_ROOT_REJECTED' }
  $activeOverlayManifestPath = Join-Path $activeWebRoot 'operational-overlay-manifest.json'
  $activeRuntimeManifestPath = Join-Path $activeWebRoot 'runtime-build-manifest.json'
  foreach ($manifestPath in @($activeOverlayManifestPath,$activeRuntimeManifestPath)) {
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "WEB_OVERLAY_ACTIVE_MANIFEST_MISSING:$manifestPath" }
  }
  $activeOverlayManifest = Get-Content -LiteralPath $activeOverlayManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
  $activeRuntimeManifest = Get-Content -LiteralPath $activeRuntimeManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
  if ([string]$activeOverlayManifest.schema -ne 'evomind.web_operational_overlay.v1' -or [string]$activeOverlayManifest.overlay_id -ne [string]$activeRuntimeManifest.build_id) { throw 'WEB_OVERLAY_ACTIVE_IDENTITY_REJECTED' }
  $webOverlayScopeEvidence = [ordered]@{ schema='evomind.web_overlay_acl_scope.v1'; sealed_at_utc=[DateTime]::UtcNow.ToString('o'); parent_root=$webOverlayRoot; active_root=$activeWebRoot; recursive_parent_scan=$false }
  Write-JsonAtomic -Path 'C:\ProgramData\EvoMind\state\web-overlay-acl-scope.json' -Value $webOverlayScopeEvidence
}
'@

if ((Test-Path -LiteralPath $resultPath) -or (Test-Path -LiteralPath $errorPath)) { throw 'SEAL_SCOPE_RESULT_EXISTS' }

$mutex = [Threading.Mutex]::new($false, 'Global\EvoMind-Byoa-V12-Deployment')
$owned = $false
$backup = $null
$mutated = $false
try {
  $owned = $mutex.WaitOne([TimeSpan]::FromSeconds(10))
  if (-not $owned) { throw 'SEAL_SCOPE_DEPLOYMENT_LOCKED' }
  foreach ($path in @($expectedRemote, $target)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "SEAL_SCOPE_INPUT_MISSING:$path" }
  }
  if ((Get-Sha256 $target) -ne $ExpectedScriptSha256) { throw 'SEAL_SCOPE_SOURCE_SHA_REJECTED' }

  $beforeHealth = Read-LoopbackHealth
  $beforeBinding = Read-BindingHash
  $source = Get-Content -LiteralPath $target -Raw -Encoding UTF8
  if ($source.Contains($oldBlock)) {
    $patched = $source.Replace($oldBlock, $newBlock)
  } elseif ($source.Contains('Resolve-WebRuntimeIdentity -Config $config | Out-Null')) {
    $identityOld = '  Resolve-WebRuntimeIdentity -Config $config | Out-Null'
    $identityNew = @'
  $activeOverlayManifestPath = Join-Path $activeWebRoot 'operational-overlay-manifest.json'
  $activeRuntimeManifestPath = Join-Path $activeWebRoot 'runtime-build-manifest.json'
  foreach ($manifestPath in @($activeOverlayManifestPath,$activeRuntimeManifestPath)) {
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "WEB_OVERLAY_ACTIVE_MANIFEST_MISSING:$manifestPath" }
  }
  $activeOverlayManifest = Get-Content -LiteralPath $activeOverlayManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
  $activeRuntimeManifest = Get-Content -LiteralPath $activeRuntimeManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
  if ([string]$activeOverlayManifest.schema -ne 'evomind.web_operational_overlay.v1' -or [string]$activeOverlayManifest.overlay_id -ne [string]$activeRuntimeManifest.build_id) { throw 'WEB_OVERLAY_ACTIVE_IDENTITY_REJECTED' }
'@
    $patched = $source.Replace($identityOld, $identityNew)
  } else {
    throw 'SEAL_SCOPE_SOURCE_CONTRACT_MISSING'
  }
  if ($patched.Contains("'C:\ProgramData\EvoMind\web-overlays') | Where-Object")) { throw 'SEAL_SCOPE_PARENT_RECURSION_REMAINS' }
  if (-not $patched.Contains('recursive_parent_scan=$false')) { throw 'SEAL_SCOPE_MARKER_MISSING' }
  [IO.File]::WriteAllText($candidate, $patched, [Text.UTF8Encoding]::new($false))
  Assert-PowerShellAst $candidate
  if ($PreflightOnly) {
    [ordered]@{ status='preflight_verified'; transaction_id=$TransactionId; candidate_sha256=Get-Sha256 $candidate; recursive_web_overlay_parent_scan=$false; service_restarted=$false; hpc_accessed=$false; gpu_touched=$false; secrets_printed=$false } | ConvertTo-Json -Compress
    return
  }

  $backupRoot = Join-Path $root ('backups\pre-seal-release-scope-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
  New-Item -ItemType Directory -Path $backupRoot | Out-Null
  $backup = Join-Path $backupRoot 'Seal-Release.ps1'
  Copy-Item -LiteralPath $target -Destination $backup
  Move-Item -LiteralPath $candidate -Destination $target -Force
  $mutated = $true
  Assert-PowerShellAst $target

  $afterHealth = Read-LoopbackHealth
  $afterBinding = Read-BindingHash
  if ($beforeBinding -ne $afterBinding) { throw 'SEAL_SCOPE_HPC_BINDING_CHANGED' }
  $installedSha = Get-Sha256 $target
  $payload = [ordered]@{
    schema = 'evomind.seal_release_scope_hotfix.v1'
    status = 'installed_and_verified'
    transaction_id = $TransactionId
    old_sha256 = $ExpectedScriptSha256
    new_sha256 = $installedSha
    backup_path = $backup
    before_health = $beforeHealth
    after_health = $afterHealth
    recursive_web_overlay_parent_scan = $false
    active_overlay_identity_required = $true
    hpc_binding_unchanged = $true
    service_restarted = $false
    secrets_printed = $false
    hpc_accessed = $false
    gpu_touched = $false
  }
  Write-Utf8Json $resultPath $payload
  $payload | ConvertTo-Json -Depth 8 -Compress
} catch {
  $failure = $_
  $rollback = 'not_required'
  if ($mutated -and $backup -and (Test-Path -LiteralPath $backup -PathType Leaf)) {
    try {
      Copy-Item -LiteralPath $backup -Destination $target -Force
      Assert-PowerShellAst $target
      $rollback = 'restored'
    } catch {
      $rollback = 'failed'
    }
  }
  Write-Utf8Json $errorPath ([ordered]@{ schema='evomind.seal_release_scope_hotfix_error.v1'; status='failed'; transaction_id=$TransactionId; message=[string]$failure.Exception.Message; rollback=$rollback; secrets_printed=$false; hpc_accessed=$false; gpu_touched=$false })
  throw
} finally {
  if (Test-Path -LiteralPath $candidate) { Remove-Item -LiteralPath $candidate -Force -ErrorAction SilentlyContinue }
  if ($owned) { try { [void]$mutex.ReleaseMutex() } catch {} }
  $mutex.Dispose()
}
