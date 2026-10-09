param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [Parameter(Mandatory)][string]$RemoteRoot,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$CandidateSha256,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedOldSha256,
  [Parameter(Mandatory)][string]$ExpectedBuildId,
  [Parameter(Mandatory)][ValidateRange(1,2147483647)][int]$ExpectedJobId,
  [Parameter(Mandatory)][ValidateRange(1,2147483647)][int]$ExpectedGeneration,
  [Parameter(Mandatory)][ValidatePattern('^tenant_[a-f0-9]{24}$')][string]$ExpectedTenantId
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = 'C:\ProgramData\EvoMind'
$expectedRemote = "C:\SecureInput\assistant-runs-hotfix-$TransactionId"
if ([IO.Path]::GetFullPath($RemoteRoot) -cne $expectedRemote) { throw 'ASSISTANT_RUNS_REMOTE_ROOT_REJECTED' }
$candidate = Join-Path $expectedRemote 'assistant_runs.py'
$resultPath = Join-Path $expectedRemote "transaction-$TransactionId-result.json"
$errorPath = Join-Path $expectedRemote "transaction-$TransactionId-error.json"
$action = 'C:\SecureInput\Invoke-ServiceAccountAction.ps1'
$seal = Join-Path $root 'bundle\scripts\Seal-Release.ps1'
$target = Join-Path $root 'bundle\runtime\evomind_runtime\assistant_runs.py'

function Sha([string]$Path) {
  (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-Utf8Json([string]$Path, [object]$Value) {
  [IO.File]::WriteAllText(
    $Path,
    (($Value | ConvertTo-Json -Depth 12) + "`n"),
    [Text.UTF8Encoding]::new($false)
  )
}

function Tree([string]$Path) {
  $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
  $rows = @(
    Get-ChildItem -LiteralPath $full -Recurse -File -Force |
      Where-Object { $_.FullName -notmatch '[\\/]__pycache__[\\/]' -and $_.Extension -notin @('.pyc', '.pyo') } |
      Sort-Object FullName |
      ForEach-Object { "$($_.FullName.Substring($full.Length).TrimStart('\').Replace('\','/'))|$([long]$_.Length)|$(Sha $_.FullName)" }
  )
  $hash = [Security.Cryptography.SHA256]::Create()
  try {
    $digest = ([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes(($rows -join "`n"))))).Replace('-', '').ToLowerInvariant()
  } finally {
    $hash.Dispose()
  }
  [ordered]@{ file_count = $rows.Count; sha256 = $digest }
}

function Read-Runtime {
  $config = Get-Content -LiteralPath "$root\config\node-config.json" -Raw -Encoding UTF8 | ConvertFrom-Json
  $runtime = Get-Content -LiteralPath (Join-Path ([string]$config.web_runtime_root) 'runtime-build-manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
  [ordered]@{ config = $config; runtime = $runtime }
}

function Assert-Managed([string]$BuildId) {
  $task = Get-ScheduledTask -TaskPath '\' -TaskName 'EvoMind-Cloud-Node' -ErrorAction Stop
  $info = Get-ScheduledTaskInfo -TaskPath '\' -TaskName 'EvoMind-Cloud-Node' -ErrorAction Stop
  if ([string]$task.State -ne 'Ready' -or [int64]$info.LastTaskResult -ne 0 -or [string]$task.Actions[0].Arguments -notmatch '-Action Start(?:\s|$)') {
    throw 'ASSISTANT_RUNS_MANAGED_TASK_NOT_READY'
  }
  $state = Get-Content -LiteralPath "$root\state\node-processes.json" -Raw -Encoding UTF8 | ConvertFrom-Json
  foreach ($role in @('tenant_enrollment_queue', 'llm_gateway', 'python_runtime', 'web')) {
    $record = @($state.records | Where-Object { [string]$_.role -eq $role })
    if ($record.Count -ne 1 -or -not (Get-Process -Id ([int]$record[0].pid) -ErrorAction SilentlyContinue)) {
      throw "ASSISTANT_RUNS_ROLE_INVALID:$role"
    }
  }
  $current = Read-Runtime
  if ([string]$current.runtime.build_id -ne $BuildId) { throw 'ASSISTANT_RUNS_BUILD_MISMATCH' }
  [ordered]@{ build_id = [string]$current.runtime.build_id; roles = 4; task_ready = $true; config = $current.config }
}

function Read-HpcState {
  $current = Read-Runtime
  $config = $current.config
  $bindingPath = Join-Path $root "byoa\tenants\$ExpectedTenantId\hpc-binding.json"
  $binding = Get-Content -LiteralPath $bindingPath -Raw -Encoding UTF8 | ConvertFrom-Json
  $expectedProfile = "${ExpectedTenantId}_job${ExpectedJobId}_g${ExpectedGeneration}"
  if (
    [string]$binding.schema -ne 'evomind.tenant_hpc_binding.v1' -or
    [string]$binding.tenant_id -ne $ExpectedTenantId -or
    [string]$binding.state -ne 'active' -or
    [int]$binding.job_id -ne $ExpectedJobId -or
    [int]$binding.allocation_generation -ne $ExpectedGeneration -or
    [string]$binding.credential_profile -ne $expectedProfile
  ) { throw 'ASSISTANT_RUNS_BINDING_REJECTED' }
  $profileInfo = Get-CimInstance Win32_UserProfile -ErrorAction SilentlyContinue |
    Where-Object { [IO.Path]::GetFileName(([string]$_.LocalPath).TrimEnd('\')) -ieq [string]$config.dedicated_user } |
    Select-Object -First 1
  $serviceHome = if ($profileInfo) { [string]$profileInfo.LocalPath } else { "C:\Users\$([string]$config.dedicated_user)" }
  $profileRoot = Join-Path $serviceHome "AppData\Roaming\ResearchAgentWorkstation\profiles\$expectedProfile"
  if (-not (Test-Path -LiteralPath $profileRoot -PathType Container)) { throw 'ASSISTANT_RUNS_PROFILE_MISSING' }
  [ordered]@{
    binding_sha256 = Sha $bindingPath
    byoa_tree = Tree (Join-Path $root 'byoa')
    profile_root = $profileRoot
    profile_tree = Tree $profileRoot
    job_id = [int]$binding.job_id
    generation = [int]$binding.allocation_generation
    state = [string]$binding.state
  }
}

function Assert-HpcSame([object]$Before, [object]$After) {
  if (
    $Before.binding_sha256 -ne $After.binding_sha256 -or
    $Before.byoa_tree.sha256 -ne $After.byoa_tree.sha256 -or
    $Before.byoa_tree.file_count -ne $After.byoa_tree.file_count -or
    $Before.profile_root -ne $After.profile_root -or
    $Before.profile_tree.sha256 -ne $After.profile_tree.sha256 -or
    $Before.profile_tree.file_count -ne $After.profile_tree.file_count -or
    $Before.job_id -ne $After.job_id -or
    $Before.generation -ne $After.generation -or
    $Before.state -ne $After.state
  ) { throw 'ASSISTANT_RUNS_HPC_STATE_CHANGED' }
}

function Wait-Exact([string]$Uri) {
  $samples = @()
  $consecutive = 0
  $deadline = [DateTime]::UtcNow.AddSeconds(180)
  do {
    $row = [ordered]@{ status_code = 0; build_id = $null; exact = $false; elapsed_ms = $null }
    $clock = [Diagnostics.Stopwatch]::StartNew()
    try {
      $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 12
      $payload = $response.Content | ConvertFrom-Json
      $row.status_code = [int]$response.StatusCode
      $row.build_id = [string]$payload.build_id
      $row.exact = $row.status_code -eq 200 -and [bool]$payload.ok -and [string]$payload.status -eq 'ready' -and $row.build_id -eq $ExpectedBuildId -and @($payload.failures).Count -eq 0
    } catch {
    } finally {
      $clock.Stop()
      $row.elapsed_ms = [math]::Round($clock.Elapsed.TotalMilliseconds, 3)
    }
    if ($row.exact) { $consecutive++ } else { $consecutive = 0 }
    $samples += [pscustomobject]$row
    if ($consecutive -ge 3) { return [ordered]@{ status = 'ready'; samples = @($samples | Select-Object -Last 3) } }
    Start-Sleep -Seconds 2
  } while ([DateTime]::UtcNow -lt $deadline)
  throw "ASSISTANT_RUNS_EXACT_HEALTH_TIMEOUT:$Uri"
}

if ((Test-Path -LiteralPath $resultPath) -or (Test-Path -LiteralPath $errorPath)) { throw 'ASSISTANT_RUNS_RESULT_EXISTS' }
$mutex = [Threading.Mutex]::new($false, 'Global\EvoMind-Byoa-V12-Deployment')
$owned = $false
$backup = $null
$mutated = $false
$serviceStopped = $false
$beforeHpc = $null
$rollbackError = $null
try {
  $owned = $mutex.WaitOne([TimeSpan]::FromSeconds(10))
  if (-not $owned) { throw 'ASSISTANT_RUNS_DEPLOYMENT_LOCKED' }
  foreach ($path in @($candidate, $action, $seal, $target)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "ASSISTANT_RUNS_INPUT_MISSING:$path" }
  }
  if ((Sha $candidate) -ne $CandidateSha256) { throw 'ASSISTANT_RUNS_CANDIDATE_HASH_MISMATCH' }
  if ((Sha $target) -ne $ExpectedOldSha256) { throw 'ASSISTANT_RUNS_OLD_HASH_REJECTED' }
  $beforeRuntime = Assert-Managed $ExpectedBuildId
  $beforeHpc = Read-HpcState
  $stop = @(& $action -Action Stop -TimeoutMinutes 12) | Out-String | ConvertFrom-Json
  if ([string]$stop.status -ne 'completed' -or [string]$stop.action -ne 'Stop') { throw 'ASSISTANT_RUNS_STOP_FAILED' }
  $serviceStopped = $true
  foreach ($port in 65068, 8765, 8088, 7890) {
    if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) { throw "ASSISTANT_RUNS_LISTENER_REMAINS:$port" }
  }
  $backup = Join-Path "$root\backups" ('pre-assistant-runs-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
  New-Item -ItemType Directory -Path $backup -Force | Out-Null
  Copy-Item -LiteralPath $target -Destination (Join-Path $backup 'assistant_runs.py')
  $mutated = $true
  Move-Item -LiteralPath $target -Destination (Join-Path $backup 'assistant_runs.moved.py')
  Copy-Item -LiteralPath $candidate -Destination $target
  if ((Sha $target) -ne $CandidateSha256) { throw 'ASSISTANT_RUNS_INSTALLED_HASH_REJECTED' }
  & $seal | Out-Null
  Assert-HpcSame $beforeHpc (Read-HpcState)
  $start = @(& $action -Action Start -TimeoutMinutes 12) | Out-String | ConvertFrom-Json
  if ([string]$start.status -ne 'completed' -or [string]$start.action -ne 'Start') { throw 'ASSISTANT_RUNS_START_FAILED' }
  $serviceStopped = $false
  $loopback = Wait-Exact 'http://127.0.0.1:8088/api/healthz'
  $public = Wait-Exact 'https://evomind.zhjjq.tech/api/healthz'
  $afterRuntime = Assert-Managed $ExpectedBuildId
  $afterHpc = Read-HpcState
  Assert-HpcSame $beforeHpc $afterHpc
  if ((Sha $target) -ne $CandidateSha256) { throw 'ASSISTANT_RUNS_POSTSTART_HASH_REJECTED' }
  $release = [string]$afterRuntime.config.release_root
  $python = Join-Path $release '.venv\Scripts\python.exe'
  $bundleRuntime = Join-Path $root 'bundle\runtime'
  $oldPythonPath = $env:PYTHONPATH
  try {
    $env:PYTHONPATH = $bundleRuntime + ';' + (Join-Path $release 'src')
    $identity = & $python -c "import hashlib,json,evomind_runtime.assistant_runs as a;print(json.dumps({'path':a.__file__,'sha256':hashlib.sha256(open(a.__file__,'rb').read()).hexdigest()}))"
    if ($LASTEXITCODE -ne 0) { throw 'ASSISTANT_RUNS_IMPORT_FAILED' }
  } finally {
    $env:PYTHONPATH = $oldPythonPath
  }
  $importIdentity = $identity | Out-String | ConvertFrom-Json
  if ([string]$importIdentity.sha256 -ne $CandidateSha256) { throw 'ASSISTANT_RUNS_IMPORT_HASH_REJECTED' }
  $payload = [ordered]@{
    schema = 'evomind.assistant_runs_hotfix_deployment_result.v1'
    status = 'installed_and_verified'
    transaction_id = $TransactionId
    build_id = $ExpectedBuildId
    backup_root = $backup
    old_sha256 = $ExpectedOldSha256
    assistant_runs_sha256 = $CandidateSha256
    import_identity = $importIdentity
    loopback = $loopback
    public = $public
    job_id = $afterHpc.job_id
    allocation_generation = $afterHpc.generation
    binding_state = $afterHpc.state
    hpc_binding_unchanged = $true
    profile_unchanged = $true
    byoa_tree_unchanged = $true
    hpc_accessed = $false
    gpu_touched = $false
    training_started = $false
    kaggle_submissions = 0
    secrets_printed = $false
  }
  Write-Utf8Json $resultPath $payload
  $payload | ConvertTo-Json -Depth 12 -Compress
} catch {
  $failure = $_
  $rollback = 'not_required'
  if ($backup -and $mutated) {
    try {
      try {
        $stopRollback = @(& $action -Action Stop -TimeoutMinutes 12) | Out-String | ConvertFrom-Json
        if ([string]$stopRollback.status -ne 'completed') { throw 'ASSISTANT_RUNS_ROLLBACK_STOP_FAILED' }
        $serviceStopped = $true
      } catch {
        $listeners = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in @(65068, 8765, 8088, 7890) })
        if ($listeners.Count) { throw }
        $serviceStopped = $true
      }
      if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Force }
      Copy-Item -LiteralPath (Join-Path $backup 'assistant_runs.py') -Destination $target
      & $seal | Out-Null
      Assert-HpcSame $beforeHpc (Read-HpcState)
      $restart = @(& $action -Action Start -TimeoutMinutes 12) | Out-String | ConvertFrom-Json
      if ([string]$restart.status -ne 'completed') { throw 'ASSISTANT_RUNS_ROLLBACK_START_FAILED' }
      $serviceStopped = $false
      Wait-Exact 'http://127.0.0.1:8088/api/healthz' | Out-Null
      Assert-HpcSame $beforeHpc (Read-HpcState)
      $rollback = 'restored_and_verified'
    } catch {
      $rollback = 'failed'
      $rollbackError = [string]$_.Exception.Message
    }
  } elseif ($serviceStopped) {
    try {
      $restartOnly = @(& $action -Action Start -TimeoutMinutes 12) | Out-String | ConvertFrom-Json
      if ([string]$restartOnly.status -eq 'completed') { $serviceStopped = $false; $rollback = 'start_only_recovered' }
    } catch {
    }
  }
  Write-Utf8Json $errorPath ([ordered]@{
    schema = 'evomind.assistant_runs_hotfix_deployment_error.v1'
    status = 'failed'
    transaction_id = $TransactionId
    build_id = $ExpectedBuildId
    message = [string]$failure.Exception.Message
    rollback = $rollback
    rollback_error = $rollbackError
    secrets_printed = $false
    hpc_accessed = $false
    gpu_touched = $false
    training_started = $false
    kaggle_submissions = 0
  })
  throw
} finally {
  if ($owned) { try { [void]$mutex.ReleaseMutex() } catch {} }
  $mutex.Dispose()
}
