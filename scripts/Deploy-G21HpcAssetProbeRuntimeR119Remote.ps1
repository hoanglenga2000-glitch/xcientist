param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [Parameter(Mandatory)][string]$RemoteRoot,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$ApprovalSha256
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = 'C:\ProgramData\EvoMind'
$expectedRemote = [IO.Path]::GetFullPath("C:\SecureInput\r119-hpc-asset-probe-runtime-$TransactionId")
if ([IO.Path]::GetFullPath($RemoteRoot) -cne $expectedRemote) { throw 'R119_REMOTE_ROOT_REJECTED' }
$inputRoot = Join-Path $expectedRemote 'envelope-expanded'
$approvalPath = Join-Path $inputRoot 'approval.json'
$archive = Join-Path $inputRoot 'runtime.zip'
$sourceManifestPath = Join-Path $inputRoot 'source-manifest.json'
$buildResultPath = Join-Path $inputRoot 'build-result.json'
$receiptPath = Join-Path $inputRoot 'test-receipt.json'
$newSpecPath = Join-Path $inputRoot 'new-spec.json'
$policyEvidencePath = Join-Path $inputRoot 'policy-evidence.json'
$stage = Join-Path $expectedRemote 'runtime-stage'
$resultPath = Join-Path $expectedRemote "transaction-$TransactionId-result.json"
$errorPath = Join-Path $expectedRemote "transaction-$TransactionId-error.json"
$action = 'C:\SecureInput\Invoke-ServiceAccountAction.ps1'
$seal = Join-Path $root 'bundle\scripts\Seal-Release.ps1'
$bundleRuntime = Join-Path $root 'bundle\runtime'
$target = Join-Path $bundleRuntime 'evomind_runtime'
$fixedRun = 'run_7b1efb878afb40f396db431e91f093a5'
$fixedAllocation = 'G21'
$fixedWebBuild = 'overlay-controlled-secret-binding-r112-standalone-f38c40f2dbdc'
$baselineZipSha = '6e0c5c67a9cb69c68f5f2f88777e41a0db86c4340cfb2b0d04b981e84259908c'
$baselineTreeSha = 'b9d60506be3d01e4bf8dab9111915cc532acb74c2aa8632adc6a8d15312c2592'
$newSpecRawSha = 'da8c3fbe8b14107dfadf96ad906b4cf982bc714ebf6420225a1e3722cd41a752'
$newSpecCanonicalSha = 'fbb9939dff28c355c2798bf32fce104845337064f42f9c280f2b6906cccf4036'
$currentBoardSha = '6b908173083889ac21f8afea52528b4b5e35d2ff862a880a6065d65b169f443c'
$policyEvidenceRawSha = '47df860192502afcaacd7b1b6bc7fc4d86110c2e665278c8b7b3ea421893b7a7'
$policyEvidenceCanonicalSha = 'f81701f13aeebef146659e091cacfc4e39ae18df345ee6b58be2ad5ef40829f0'
$migrationId = 'g21-conditional-strong-baseline-v2'

function Sha([string]$Path){
  $stream=[IO.File]::OpenRead($Path);$hash=[Security.Cryptography.SHA256]::Create()
  try{([BitConverter]::ToString($hash.ComputeHash($stream))).Replace('-','').ToLowerInvariant()}finally{$stream.Dispose();$hash.Dispose()}
}

function Safe-Error([object]$Value) {
  $text=[string]$Value
  $text=$text-replace'(?i)(?:Bearer|Basic)\s+\S+','[credential redacted]'
  $text=$text-replace'(?i)\b(?:password|passwd|pwd|credential|token|secret|api[_ -]?key|authorization|bearer|密码|口令|凭据|账号|登录)\s*[:=]\s*\S+','[credential redacted]'
  $text=$text-replace'[\r\n]+','_'
  if($text.Length-gt500){$text=$text.Substring(0,500)}
  $text
}

function Write-AtomicJson([string]$Path, [object]$Value) {
  $temporary = "$Path.$PID.tmp"
  [IO.File]::WriteAllText($temporary, (($Value | ConvertTo-Json -Depth 20) + "`n"), [Text.UTF8Encoding]::new($false))
  Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Read-Object([string]$Path, [string]$Code) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$Code`:MISSING" }
  try { $value = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json }
  catch { throw "$Code`:INVALID_JSON" }
  if ($null -eq $value -or $value -is [array]) { throw "$Code`:OBJECT_REQUIRED" }
  $value
}

function Assert-NoReparse([string]$Path) {
  $rootItem = Get-Item -LiteralPath $Path -Force
  if ($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "R119_REPARSE_REJECTED:$Path" }
  $bad = @(Get-ChildItem -LiteralPath $Path -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })
  if ($bad.Count) { throw "R119_REPARSE_REJECTED:$($bad[0].FullName)" }
}

function Tree([string]$Path) {
  $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
  $files = @(Get-ChildItem -LiteralPath $full -Recurse -File -Force | Where-Object { $_.Name -notmatch '\.(pyc|pyo)$' -and $_.FullName -notmatch '[\\/]__pycache__[\\/]' } | Sort-Object FullName)
  $rows = @($files | ForEach-Object { "$($_.FullName.Substring($full.Length).TrimStart('\').Replace('\','/'))|$([long]$_.Length)|$(Sha $_.FullName)" })
  $hash = [Security.Cryptography.SHA256]::Create()
  try { $digest = ([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes(($rows -join "`n"))))).Replace('-','').ToLowerInvariant() }
  finally { $hash.Dispose() }
  [ordered]@{ file_count=$files.Count; sha256=$digest }
}

function Other-RuntimeTree {
  $full = [IO.Path]::GetFullPath($bundleRuntime).TrimEnd('\')
  $rows = @(Get-ChildItem -LiteralPath $full -Recurse -File -Force | ForEach-Object {
    $relative = $_.FullName.Substring($full.Length).TrimStart('\').Replace('\','/')
    if ($relative -notmatch '^(?:evomind_runtime(?:/|\.)|__pycache__/)' -and $relative -notmatch '\.(pyc|pyo)$') {
      "$relative|$([long]$_.Length)|$(Sha $_.FullName)"
    }
  } | Where-Object { $_ } | Sort-Object)
  $hash = [Security.Cryptography.SHA256]::Create()
  try { $digest = ([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes(($rows -join "`n"))))).Replace('-','').ToLowerInvariant() }
  finally { $hash.Dispose() }
  [ordered]@{ file_count=$rows.Count; sha256=$digest }
}

function Read-Runtime {
  $config = Read-Object (Join-Path $root 'config\node-config.json') 'R119_NODE_CONFIG'
  $runtimeManifestPath = Join-Path ([string]$config.web_runtime_root) 'runtime-build-manifest.json'
  $runtime = Read-Object $runtimeManifestPath 'R119_WEB_RUNTIME_MANIFEST'
  [ordered]@{ config=$config; runtime=$runtime; runtime_manifest_path=$runtimeManifestPath }
}

function Get-RuntimeHeaders {
  $current = Read-Runtime
  $tokenPath = Join-Path ([string]$current.config.data_root) 'workspace\runtime\runtime.token'
  if (-not (Test-Path -LiteralPath $tokenPath -PathType Leaf)) { throw 'R119_RUNTIME_TOKEN_MISSING' }
  $token = (Get-Content -LiteralPath $tokenPath -Raw).Trim()
  if (-not $token) { throw 'R119_RUNTIME_TOKEN_EMPTY' }
  @{ Authorization = 'Bearer ' + $token }
}

function Assert-Managed([string]$ExpectedBuild) {
  $statePath = Join-Path $root 'state\node-processes.json'
  $state = Read-Object $statePath 'R119_NODE_PROCESS_STATE'
  $roles = [ordered]@{}
  foreach ($role in @('tenant_enrollment_queue','llm_gateway','python_runtime','web')) {
    $records = @($state.records | Where-Object { [string]$_.role -ceq $role })
    if ($records.Count -ne 1) { throw "R119_ROLE_COUNT_REJECTED:$role" }
    $pidValue = [int]$records[0].pid
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction Stop
    $owner = $process | Invoke-CimMethod -MethodName GetOwner
    if ([string]$owner.User -cne 'EvoMindSvc') { throw "R119_ROLE_OWNER_REJECTED:$role" }
    $roles[$role] = [ordered]@{ pid=$pidValue; owner=[string]$owner.User; name=[string]$process.Name }
  }
  if(@($roles.Values|ForEach-Object{[int]$_.pid}|Select-Object -Unique).Count-ne4){throw 'R119_ROLE_PID_DUPLICATE'}
  $listeners = [ordered]@{}
  foreach ($port in @(65068,8765,8088,7890)) {
    $rows = @(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue)
    if ($rows.Count -ne 1) { throw "R119_LISTENER_COUNT_REJECTED:$port" }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$rows[0].OwningProcess)" -ErrorAction Stop
    $owner = $process | Invoke-CimMethod -MethodName GetOwner
    if ([string]$owner.User -cne 'EvoMindSvc') { throw "R119_LISTENER_OWNER_REJECTED:$port" }
    $identity=switch($port){
      65068{if([string]$process.Name-ceq'cliproxy-7.2.128.exe'-and[string]$process.ExecutablePath-match'ProgramData\\EvoMind\\bin\\cliproxy-7\.2\.128\.exe$'){'llm_gateway'}else{$null}}
      8765{if([string]$process.Name-in@('python.exe','pythonw.exe')-and[string]$process.CommandLine-match'run_python_runtime\.py.+--port\s+8765'){'python_runtime_child'}else{$null}}
      8088{if([string]$process.Name-ceq'node.exe'-and[string]$process.CommandLine-match'(?:^|\s)server\.js(?:\s|$)'){'web'}else{$null}}
      7890{if([string]$process.Name-in@('python.exe','pythonw.exe')-and[string]$process.CommandLine-match'hpc_socks_bridge\.py.+--listen-port\s+7890'){'managed_hpc_bridge'}else{$null}}
    }
    if(-not$identity){throw "R119_LISTENER_IDENTITY_REJECTED:$port"}
    $listeners[[string]$port] = [ordered]@{ pid=[int]$rows[0].OwningProcess; parent_pid=[int]$process.ParentProcessId; owner=[string]$owner.User; name=[string]$process.Name; identity=$identity }
  }
  if(@($listeners.Values|ForEach-Object{[int]$_.pid}|Select-Object -Unique).Count-ne4){throw 'R119_LISTENER_PID_DUPLICATE'}
  foreach($binding in @(
    [pscustomobject]@{role='llm_gateway';port='65068'},
    [pscustomobject]@{role='web';port='8088'}
  )){if([int]$roles[$binding.role].pid-ne[int]$listeners[$binding.port].pid){throw "R119_ROLE_LISTENER_PID_MISMATCH:$($binding.role)"}}
  if([int]$listeners['8765'].parent_pid-ne[int]$roles['python_runtime'].pid){throw 'R119_PYTHON_LISTENER_PARENT_REJECTED'}
  $current = Read-Runtime
  if ([string]$current.runtime.build_id -cne $ExpectedBuild) { throw 'R119_WEB_BUILD_MISMATCH' }
  [ordered]@{ build_id=[string]$current.runtime.build_id; roles=$roles; listeners=$listeners; state_sha256=Sha $statePath }
}

function Assert-FreshManaged([object]$Before,[object]$After){
  foreach($role in @('tenant_enrollment_queue','llm_gateway','python_runtime','web')){
    if([int]$Before.roles[$role].pid-eq[int]$After.roles[$role].pid){throw "R119_ROLE_PID_NOT_FRESH:$role"}
  }
  foreach($port in @('65068','8765','8088','7890')){
    if([int]$Before.listeners[$port].pid-eq[int]$After.listeners[$port].pid){throw "R119_LISTENER_PID_NOT_FRESH:$port"}
  }
}

function Wait-Exact([string]$Uri, [string]$Build) {
  $samples = @(); $consecutive = 0; $deadline = [DateTime]::UtcNow.AddSeconds(180)
  do {
    $row = [ordered]@{ status_code=0; build_id=$null; exact=$false; elapsed_ms=$null }
    $timer = [Diagnostics.Stopwatch]::StartNew()
    try {
      $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 12
      $payload = $response.Content | ConvertFrom-Json
      $row.status_code = [int]$response.StatusCode
      $row.build_id = [string]$payload.build_id
      $row.exact = ($row.status_code -eq 200 -and [bool]$payload.ok -and [string]$payload.status -ceq 'ready' -and $row.build_id -ceq $Build -and @($payload.failures).Count -eq 0)
    } catch {} finally { $timer.Stop(); $row.elapsed_ms = [math]::Round($timer.Elapsed.TotalMilliseconds,3) }
    if ($row.exact) { $consecutive++ } else { $consecutive = 0 }
    $samples += [pscustomobject]$row
    if ($consecutive -ge 3) { return [ordered]@{ status='ready'; samples=@($samples | Select-Object -Last 3) } }
    Start-Sleep -Seconds 2
  } while ([DateTime]::UtcNow -lt $deadline)
  throw "R119_EXACT_HEALTH_TIMEOUT:$Uri"
}

function Read-RunGate {
  $snapshot = Invoke-RestMethod -Uri "http://127.0.0.1:8765/v1/runs/$fixedRun" -Headers (Get-RuntimeHeaders) -TimeoutSec 20
  $pending = @($snapshot.approvals | Where-Object { [string]$_.status -ceq 'pending' }).Count
  $active = @($snapshot.active_tool_calls).Count
  if (
    [string]$snapshot.id -cne $fixedRun -or
    $snapshot.terminal -isnot [bool] -or -not [bool]$snapshot.terminal -or
    [string]$snapshot.status -notin @('blocked','failed','cancelled','completed') -or
    $active -ne 0 -or $pending -ne 0
  ) { throw 'R119_TARGET_RUN_NOT_TERMINAL_QUIESCENT' }
  $lastEventSeq=if($snapshot.PSObject.Properties.Name -contains 'last_event_seq'){[int64]$snapshot.last_event_seq}else{[int64]0}
  [ordered]@{ run_id=$fixedRun; status=[string]$snapshot.status; terminal=$true; effective_active_count=0; pending_approval_count=0; last_event_seq=$lastEventSeq }
}

function Read-HpcState {
  $current = Read-Runtime
  $tenantRoot = Join-Path $root 'byoa\tenants\tenant_ba0ef9d3767f2fb385b856e0'
  $bindingPath = Join-Path $tenantRoot 'hpc-binding.json'
  $binding = Read-Object $bindingPath 'R119_HPC_BINDING'
  if (
    [string]$binding.schema -cne 'evomind.tenant_hpc_binding.v1' -or
    [string]$binding.tenant_id -cne 'tenant_ba0ef9d3767f2fb385b856e0' -or
    [int]$binding.job_id -lt 1 -or [int]$binding.allocation_generation -lt 1 -or
    [string]$binding.credential_profile -cne "tenant_ba0ef9d3767f2fb385b856e0_job$([int]$binding.job_id)_g$([int]$binding.allocation_generation)"
  ) { throw 'R119_HPC_BINDING_REJECTED' }
  $profileInfo = Get-CimInstance Win32_UserProfile -ErrorAction SilentlyContinue | Where-Object { [IO.Path]::GetFileName(([string]$_.LocalPath).TrimEnd('\')) -ieq [string]$current.config.dedicated_user } | Select-Object -First 1
  $serviceHome = if ($profileInfo) { [string]$profileInfo.LocalPath } else { "C:\Users\$([string]$current.config.dedicated_user)" }
  $profileRoot = Join-Path $serviceHome "AppData\Roaming\ResearchAgentWorkstation\profiles\$([string]$binding.credential_profile)"
  if (-not (Test-Path -LiteralPath $profileRoot -PathType Container)) { throw 'R119_HPC_PROFILE_MISSING' }
  [ordered]@{
    binding_sha256=Sha $bindingPath; byoa_tree=Tree (Join-Path $root 'byoa');
    profile_root=$profileRoot; profile_tree=Tree $profileRoot;
    job_id=[int]$binding.job_id; allocation_generation=[int]$binding.allocation_generation;
    binding_state=[string]$binding.state; credential_profile=[string]$binding.credential_profile
  }
}

function Assert-HpcSame([object]$Before, [object]$After) {
  if (
    $Before.binding_sha256 -cne $After.binding_sha256 -or
    $Before.byoa_tree.sha256 -cne $After.byoa_tree.sha256 -or $Before.byoa_tree.file_count -ne $After.byoa_tree.file_count -or
    $Before.profile_root -cne $After.profile_root -or $Before.profile_tree.sha256 -cne $After.profile_tree.sha256 -or $Before.profile_tree.file_count -ne $After.profile_tree.file_count -or
    $Before.job_id -ne $After.job_id -or $Before.allocation_generation -ne $After.allocation_generation -or
    $Before.binding_state -cne $After.binding_state -or $Before.credential_profile -cne $After.credential_profile
  ) { throw 'R119_HPC_LOCAL_STATE_CHANGED' }
}

function Invoke-ManagedAction([ValidateSet('Stop','Start')][string]$Name) {
  if (-not (Test-Path -LiteralPath $action -PathType Leaf)) { throw 'R119_SERVICE_ACTION_MISSING' }
  $result = @(& $action -Action $Name -TimeoutMinutes 12) | Out-String | ConvertFrom-Json
  if ([string]$result.status -cne 'completed' -or [string]$result.action -cne $Name) { throw "R119_MANAGED_$($Name.ToUpperInvariant())_FAILED" }
  $result
}

function Assert-Stopped {
  $listeners = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in @(65068,8765,8088,7890) })
  if ($listeners.Count) { throw 'R119_POST_STOP_LISTENERS_REMAIN' }
}

function Read-ApprovalAndInputs {
  foreach($path in @($approvalPath,$archive,$sourceManifestPath,$buildResultPath,$receiptPath,$newSpecPath,$policyEvidencePath,$action,$seal,$target)){
    if(-not(Test-Path -LiteralPath $path)){throw "R119_INPUT_MISSING:$path"}
  }
  if((Sha $approvalPath)-cne$ApprovalSha256){throw 'R119_APPROVAL_HASH_REJECTED'}
  $approval=Read-Object $approvalPath 'R119_APPROVAL'
  if([string]$approval.schema-cne'evomind.g21_hpc_asset_probe_runtime_deployment_approval.r119.v1'-or[string]$approval.status-cne'approved_for_single_runtime_transaction'-or
    $approval.one_time-isnot[bool]-or-not[bool]$approval.one_time-or[string]$approval.deployment_id-cne$TransactionId-or[string]$approval.run_id-cne$fixedRun-or[string]$approval.allocation-cne$fixedAllocation-or[string]$approval.web_build_id-cne$fixedWebBuild-or
    [string]$approval.preflight.current_runtime_tree_sha256-cne$baselineTreeSha-or[int]$approval.preflight.current_runtime_file_count-ne32-or
    [string]$approval.baseline.zip_sha256-cne$baselineZipSha-or[string]$approval.baseline.runtime_tree_sha256-cne$baselineTreeSha-or[int]$approval.baseline.runtime_file_count-ne32-or
    [int]$approval.candidate.runtime_file_count-ne32-or[int]$approval.candidate.changed_file_count-ne1-or[int]$approval.candidate.unchanged_file_count-ne31-or
    [string]$approval.production_invariants.goal_spec_sha256-cne$newSpecCanonicalSha-or[string]$approval.production_invariants.goal_board_sha256-cne$currentBoardSha-or
    -not[bool]$approval.production_invariants.goal_mutation_forbidden-or-not[bool]$approval.production_invariants.database_content_mutation_forbidden-or-not[bool]$approval.production_invariants.hpc_access_forbidden-or
    [string]$approval.deployer.remote_runner_filename-cne[IO.Path]::GetFileName($PSCommandPath)-or(Sha $PSCommandPath)-cne[string]$approval.deployer.remote_runner_sha256-or[string]$approval.deployer.service_action_path-cne$action-or
    -not[bool]$approval.boundaries.runtime_only-or-not[bool]$approval.boundaries.web_unchanged-or-not[bool]$approval.boundaries.hpc_access_forbidden-or-not[bool]$approval.boundaries.gpu_touch_forbidden-or-not[bool]$approval.boundaries.weather_forbidden-or-not[bool]$approval.boundaries.training_forbidden_during_deployment){throw 'R119_APPROVAL_CONTRACT_REJECTED'}
  try{$created=[DateTimeOffset]::Parse([string]$approval.created_at);$expires=[DateTimeOffset]::Parse([string]$approval.expires_at)}catch{throw 'R119_APPROVAL_TIME_REJECTED'}
  $now=[DateTimeOffset]::UtcNow;if($expires-le$created-or$now-lt$created.AddMinutes(-5)-or$now-ge$expires-or($expires-$created).TotalMinutes-gt60){throw 'R119_APPROVAL_EXPIRED_OR_TIME_REJECTED'}
  $fixedFiles=@(
    [pscustomobject]@{path=$archive;expected=[string]$approval.candidate.sha256},
    [pscustomobject]@{path=$sourceManifestPath;expected=[string]$approval.candidate.source_manifest_sha256},
    [pscustomobject]@{path=$buildResultPath;expected=[string]$approval.candidate.build_result_sha256},
    [pscustomobject]@{path=$receiptPath;expected=[string]$approval.verification.test_receipt_sha256},
    [pscustomobject]@{path=$newSpecPath;expected=$newSpecRawSha},[pscustomobject]@{path=$policyEvidencePath;expected=$policyEvidenceRawSha}
  )
  foreach($item in $fixedFiles){if((Sha $item.path)-cne$item.expected){throw "R119_STAGED_HASH_REJECTED:$($item.path)"}}
  $receipt=Read-Object $receiptPath 'R119_TEST_RECEIPT';$build=Read-Object $buildResultPath 'R119_BUILD_RESULT'
  if([string]$receipt.schema-cne'evomind.g21_hpc_asset_probe_runtime_test_receipt.r119.v1'-or[string]$receipt.status-cne'verified'-or[string]$receipt.candidate_zip_sha256-cne[string]$approval.candidate.sha256-or[string]$receipt.source_manifest_sha256-cne[string]$approval.candidate.source_manifest_sha256-or[string]$receipt.build_result_sha256-cne[string]$approval.candidate.build_result_sha256-or
    @($receipt.verification_rounds).Count-ne2-or[bool]$receipt.production_deployed-or[bool]$receipt.hpc_accessed-or[bool]$receipt.gpu_touched-or[int]$receipt.remote_writes-ne0-or
    [string]$build.schema-cne'evomind.g21_hpc_asset_probe_runtime.r119.v1'-or[string]$build.status-cne'built_and_verified'-or[string]$build.zip_sha256-cne[string]$approval.candidate.sha256-or[string]$build.source_manifest_sha256-cne[string]$approval.candidate.source_manifest_sha256-or[string]$build.candidate_runtime_tree_sha256-cne[string]$approval.candidate.installed_runtime_tree_sha256-or[bool]$build.production_deployed-or[bool]$build.hpc_accessed-or[bool]$build.gpu_touched-or[int]$build.remote_writes-ne0){throw 'R119_RELEASE_RECEIPT_REJECTED'}
  [ordered]@{approval=$approval;receipt=$receipt}
}

function Read-Candidate([object]$Approval) {
  if (Test-Path -LiteralPath $stage) { throw 'R119_STAGE_EXISTS' }
  Expand-Archive -LiteralPath $archive -DestinationPath $stage
  Assert-NoReparse $stage
  $manifestPath=Join-Path $stage 'hpc-asset-probe-runtime-manifest.json'
  if((Test-Path -LiteralPath (Join-Path $stage 'runtime-hotfix-manifest.json'))-or(Test-Path -LiteralPath (Join-Path $stage 'goal-runtime-overlay-manifest.json'))-or(Test-Path -LiteralPath (Join-Path $stage 'conditional-policy-runtime-manifest.json'))-or(Test-Path -LiteralPath (Join-Path $stage 'open-polymer-legacy-adoption-runtime-manifest.json'))){throw 'R119_STALE_RUNTIME_MANIFEST_REJECTED'}
  $manifest=Read-Object $manifestPath 'R119_POLICY_MANIFEST'
  if((Sha $sourceManifestPath)-cne(Sha $manifestPath)){throw 'R119_MANIFEST_BINDING_REJECTED'}
  if (
    [string]$manifest.schema-cne'evomind.g21_hpc_asset_probe_runtime.r119.v1'-or[string]$manifest.run_id-cne$fixedRun-or[string]$manifest.allocation-cne$fixedAllocation-or
    [string]$manifest.target-cne'bundle/runtime/evomind_runtime'-or[string]$manifest.baseline_zip_sha256-cne$baselineZipSha-or[string]$manifest.baseline_runtime_tree_sha256-cne$baselineTreeSha-or
    [int]$manifest.baseline_runtime_file_count-ne32-or[int]$manifest.candidate_runtime_file_count-ne32-or[int]$manifest.changed_file_count-ne1-or[int]$manifest.unchanged_file_count-ne31-or
    [string]$manifest.candidate_runtime_tree_sha256-cne[string]$Approval.candidate.installed_runtime_tree_sha256-or
    [string]$manifest.new_goal_spec_file_sha256-cne$newSpecRawSha-or[string]$manifest.new_goal_spec_canonical_sha256-cne$newSpecCanonicalSha-or
    [string]$manifest.policy_evidence_file_sha256-cne$policyEvidenceRawSha-or[string]$manifest.policy_evidence_canonical_sha256-cne$policyEvidenceCanonicalSha-or
    [bool]$manifest.production_deployable-or[bool]$manifest.production_deployed-or[bool]$manifest.hpc_accessed-or[bool]$manifest.gpu_touched-or[int]$manifest.remote_writes-ne0
  ){throw 'R119_POLICY_MANIFEST_CONTRACT_REJECTED'}
  $declared=@{}
  foreach($entry in @($manifest.files)){
    $relative=[string]$entry.path
    if($relative-cnotmatch'^evomind_runtime/[A-Za-z0-9_./-]+\.py$'-or$declared.ContainsKey($relative)){throw 'R119_MANIFEST_ENTRY_REJECTED'}
    $declared[$relative]=$entry
  }
  $actual=@(Get-ChildItem -LiteralPath $stage -Recurse -File -Force|ForEach-Object{$_.FullName.Substring($stage.Length+1).Replace('\','/')}|Where-Object{$_-ne'hpc-asset-probe-runtime-manifest.json'})
  if($declared.Count-ne32-or$actual.Count-ne32-or@($actual|Where-Object{-not$declared.ContainsKey($_)}).Count){throw 'R119_CANDIDATE_FILESET_REJECTED'}
  foreach($relative in $declared.Keys){$path=Join-Path $stage $relative.Replace('/','\');$entry=$declared[$relative];if(-not(Test-Path -LiteralPath $path -PathType Leaf)-or[long](Get-Item -LiteralPath $path).Length-ne[long]$entry.bytes-or(Sha $path)-cne[string]$entry.sha256){throw "R119_CANDIDATE_HASH_REJECTED:$relative"}}
  # Keep the verified candidate in C:\SecureInput until every read-only
  # production gate has passed and the managed service is stopped.  A
  # nonterminal Run must leave the production bundle and database untouched.
  $candidate=Join-Path $stage 'evomind_runtime'
  Assert-NoReparse $candidate
  $tree=Tree $candidate
  if($tree.file_count-ne32-or$tree.sha256-cne[string]$Approval.candidate.installed_runtime_tree_sha256){throw 'R119_CANDIDATE_TREE_REJECTED'}
  [ordered]@{path=$candidate;manifest=$manifest;tree=$tree}
}

function Backup-Database([string]$BackupRoot, [object]$Config) {
  $database = Join-Path ([string]$Config.data_root) 'workspace\runtime\runtime.sqlite3'
  if (-not (Test-Path -LiteralPath $database -PathType Leaf)) { throw 'R119_RUNTIME_DATABASE_MISSING' }
  $destination=Join-Path $BackupRoot 'database';if(Test-Path -LiteralPath $destination){throw 'R119_DATABASE_BACKUP_PATH_EXISTS'};New-Item -ItemType Directory -Path $destination|Out-Null
  $files=@()
  foreach($suffix in @('','-wal','-shm')){
    $source="$database$suffix";$exists=Test-Path -LiteralPath $source -PathType Leaf
    $row=[ordered]@{suffix=$suffix;existed=$exists;bytes=0;sha256=''}
    if($exists){$targetName=if($suffix){"runtime.sqlite3$suffix"}else{'runtime.sqlite3'};Copy-Item -LiteralPath $source -Destination (Join-Path $destination $targetName);$row.bytes=[long](Get-Item -LiteralPath $source).Length;$row.sha256=Sha $source}
    $files+=[pscustomobject]$row
  }
  [ordered]@{path=$database;files=$files}
}

function Restore-Database([string]$BackupRoot,[object]$DatabaseBackup){
  foreach($suffix in @('','-wal','-shm')){$current="$($DatabaseBackup.path)$suffix";if(Test-Path -LiteralPath $current){Remove-Item -LiteralPath $current -Force}}
  foreach($row in @($DatabaseBackup.files)){if([bool]$row.existed){$name=if([string]$row.suffix){"runtime.sqlite3$([string]$row.suffix)"}else{'runtime.sqlite3'};$source=Join-Path (Join-Path $BackupRoot 'database') $name;Copy-Item -LiteralPath $source -Destination "$($DatabaseBackup.path)$([string]$row.suffix)";if((Sha "$($DatabaseBackup.path)$([string]$row.suffix)")-cne[string]$row.sha256){throw 'R119_DATABASE_ROLLBACK_HASH_REJECTED'}}}
}

function Read-RawSessionEvents([int64]$AfterSeq,[int64]$UpperSeq,[object]$Headers){
  if($AfterSeq-lt0-or$UpperSeq-lt$AfterSeq){throw 'R119_GOAL_EVENT_WINDOW_REJECTED'}
  $cursor=$AfterSeq;$pages=0;$events=@()
  while($cursor-lt$UpperSeq){
    $pages++;if($pages-gt100){throw 'R119_GOAL_EVENT_PAGINATION_LIMIT'}
    $page=Invoke-RestMethod -Uri "http://127.0.0.1:8765/v1/sessions/$fixedRun/events?after=$cursor" -Headers $Headers -TimeoutSec 20
    $rows=@($page.events|Where-Object{[int64]$_.seq-le$UpperSeq})
    if(-not$rows.Count){throw 'R119_GOAL_EVENT_PAGE_GAP'}
    $last=$cursor
    foreach($event in $rows){
      $seq=[int64]$event.seq
      if($seq-le$last-or$seq-gt$UpperSeq){throw 'R119_GOAL_EVENT_SEQUENCE_REJECTED'}
      $events+=$event;$last=$seq
    }
    if($last-le$cursor){throw 'R119_GOAL_EVENT_CURSOR_STALLED'}
    $cursor=$last
  }
  [ordered]@{after_seq=$AfterSeq;upper_seq=$UpperSeq;last_seq=$cursor;page_count=$pages;event_count=$events.Count;events=$events}
}

function Read-GoalGate {
  $headers=Get-RuntimeHeaders
  $goal=Invoke-RestMethod -Uri 'http://127.0.0.1:8765/v1/goals/goal_g21_five_competition' -Headers $headers -TimeoutSec 20
  if([string]$goal.id-cne'goal_g21_five_competition'-or[string]$goal.run_id-cne$fixedRun-or[string]$goal.allocation-cne$fixedAllocation-or[string]$goal.status-cne'blocked'-or[string]$goal.spec_sha256-cne$newSpecCanonicalSha-or[string]$goal.board_sha256-cne$currentBoardSha){throw 'R119_GOAL_GATE_REJECTED'}
  $rows=@($goal.board.competitions)
  $completed=@($rows|Where-Object{[string]$_.goal_status-ceq'VERIFIED_COMPLETE'})
  if($rows.Count-ne5-or$completed.Count-ne1-or[string]$completed[0].competition-cne'open_polymer'){throw 'R119_GOAL_BOARD_REJECTED'}
  $list=Invoke-RestMethod -Uri 'http://127.0.0.1:8765/v1/goals' -Headers $headers -TimeoutSec 20
  if(@($list.goals).Count-ne1-or[string]$list.goals[0].id-cne[string]$goal.id){throw 'R119_GOAL_COUNT_REJECTED'}
  $runGoal=Invoke-RestMethod -Uri "http://127.0.0.1:8765/v1/runs/$fixedRun/goal" -Headers $headers -TimeoutSec 20
  $session=Invoke-RestMethod -Uri "http://127.0.0.1:8765/v1/sessions/$fixedRun" -Headers $headers -TimeoutSec 20
  $metadata=$session.metadata
  if([string]$runGoal.id-cne[string]$goal.id-or[string]$runGoal.board_sha256-cne$currentBoardSha-or[string]$metadata.goal_id-cne[string]$goal.id-or[string]$metadata.goal_spec_sha256-cne$newSpecCanonicalSha-or[string]$metadata.goal_board_sha256-cne$currentBoardSha-or[string]$metadata.goal_human_baseline_sha256-cne[string]$goal.human_baseline_sha256){throw 'R119_GOAL_METADATA_REJECTED'}
  [ordered]@{goal=$goal;goal_record_count=1;session_goal_metadata=$metadata}
}

function Read-ToolSchema {
  $tools=Invoke-RestMethod -Uri 'http://127.0.0.1:8765/v1/tools' -Headers (Get-RuntimeHeaders) -TimeoutSec 20
  $rows=@($tools.tools|Where-Object{[string]$_.name-ceq'hpc_asset_probe'})
  if($rows.Count-ne1-or-not[bool]$rows[0].read_only-or[string]$rows[0].capability-cne'hpc.read'){throw 'R119_HPC_ASSET_PROBE_TOOL_REJECTED'}
  $schema=$rows[0].input_schema
  if([bool]$schema.additionalProperties-or@($schema.required).Count-ne0-or@($schema.properties.PSObject.Properties).Count-ne1-or@($schema.properties.scope.enum).Count-ne1-or[string]$schema.properties.scope.enum[0]-cne'cure_mindgames'){throw 'R119_HPC_ASSET_PROBE_SCHEMA_REJECTED'}
  [ordered]@{name='hpc_asset_probe';read_only=$true;schema_verified=$true}
}

if((Test-Path -LiteralPath $resultPath)-or(Test-Path -LiteralPath $errorPath)){throw 'R119_RESULT_EXISTS'}
$mutex=[Threading.Mutex]::new($false,'Global\EvoMind-Byoa-V12-Deployment')
$owned=$false;$backup=$null;$candidate=$null;$bundleCandidate=$null;$bundleCandidateIntroduced=$false;$targetBackedUp=$false;$targetSwapped=$false;$serviceStopped=$false;$beforeHpc=$null;$databaseBackup=$null;$rollbackError=$null
try{
  $owned=$mutex.WaitOne([TimeSpan]::FromSeconds(10));if(-not$owned){throw 'R119_DEPLOYMENT_LOCKED'}
  $bound=Read-ApprovalAndInputs;$approval=$bound.approval
  $candidate=Read-Candidate $approval
  $preLoopback=Wait-Exact 'http://127.0.0.1:8088/api/healthz' $fixedWebBuild
  $prePublic=Wait-Exact 'https://evomind.zhjjq.tech/api/healthz' $fixedWebBuild
  $managedBefore=Assert-Managed $fixedWebBuild
  $runBefore=Read-RunGate
  $goalBefore=Read-GoalGate
  $currentRuntime=Tree $target
  if($currentRuntime.file_count-ne32-or$currentRuntime.sha256-cne$baselineTreeSha){throw 'R119_CURRENT_RUNTIME_NOT_R118'}
  $webRoot=[string](Read-Runtime).config.web_runtime_root;$webBefore=Tree $webRoot
  $beforeHpc=Read-HpcState;$otherBefore=Other-RuntimeTree
  $stop=Invoke-ManagedAction Stop;$serviceStopped=$true;Assert-Stopped
  $backup=Join-Path "$root\backups" ('pre-r119-hpc-asset-probe-runtime-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')+'-'+$TransactionId)
  if(Test-Path -LiteralPath $backup){throw 'R119_BACKUP_PATH_EXISTS'}
  New-Item -ItemType Directory -Path (Join-Path $backup 'original')|Out-Null
  $databaseBackup=Backup-Database $backup (Read-Runtime).config
  Copy-Item -LiteralPath $approvalPath -Destination (Join-Path $backup 'approval.json')
  Write-AtomicJson (Join-Path $backup 'backup-manifest.json') ([ordered]@{schema='evomind.g21_hpc_asset_probe_runtime_backup.r119.v1';transaction_id=$TransactionId;approval_sha256=$ApprovalSha256;web_build_id=$fixedWebBuild;old_runtime_tree=$currentRuntime;new_runtime_tree=$candidate.tree;other_runtime_tree=$otherBefore;database=$databaseBackup;hpc=$beforeHpc})
  $bundleCandidate=Join-Path $bundleRuntime "evomind_runtime.$TransactionId.stage"
  if(Test-Path -LiteralPath $bundleCandidate){throw 'R119_BUNDLE_CANDIDATE_EXISTS'}
  Move-Item -LiteralPath $candidate.path -Destination $bundleCandidate
  $bundleCandidateIntroduced=$true
  $candidate.path=$bundleCandidate
  Move-Item -LiteralPath $target -Destination (Join-Path $backup 'original\evomind_runtime')
  $targetBackedUp=$true
  Move-Item -LiteralPath $candidate.path -Destination $target
  $targetSwapped=$true
  $installedTree=Tree $target
  if($installedTree.file_count-ne32-or$installedTree.sha256-cne[string]$approval.candidate.installed_runtime_tree_sha256){throw 'R119_INSTALLED_RUNTIME_TREE_REJECTED'}
  $otherAfter=Other-RuntimeTree
  if($otherAfter.file_count-ne$otherBefore.file_count-or$otherAfter.sha256-cne$otherBefore.sha256){throw 'R119_OTHER_RUNTIME_CHANGED'}
  & $seal|Out-Null
  Assert-HpcSame $beforeHpc (Read-HpcState)
  $start=Invoke-ManagedAction Start;$serviceStopped=$false
  Wait-Exact 'http://127.0.0.1:8088/api/healthz' $fixedWebBuild|Out-Null
  Assert-Managed $fixedWebBuild|Out-Null
  $goalAfterStart=Read-GoalGate
  $toolAfterStart=Read-ToolSchema
  $restartStop=Invoke-ManagedAction Stop;$serviceStopped=$true;Assert-Stopped
  $restartStart=Invoke-ManagedAction Start;$serviceStopped=$false
  $loopback=Wait-Exact 'http://127.0.0.1:8088/api/healthz' $fixedWebBuild
  $public=Wait-Exact 'https://evomind.zhjjq.tech/api/healthz' $fixedWebBuild
  $managedAfter=Assert-Managed $fixedWebBuild
  Assert-FreshManaged $managedBefore $managedAfter
  $runAfter=Read-RunGate
  if($runAfter.status-cne$runBefore.status-or[int64]$runAfter.last_event_seq-ne[int64]$runBefore.last_event_seq){throw 'R119_RUN_LIFECYCLE_CHANGED'}
  $goalAfter=Read-GoalGate
  $toolAfter=Read-ToolSchema
  if([string]$goalAfter.goal.spec_sha256-cne[string]$goalBefore.goal.spec_sha256-or[string]$goalAfter.goal.board_sha256-cne[string]$goalBefore.goal.board_sha256-or[string]$goalAfter.goal.human_baseline_sha256-cne[string]$goalBefore.goal.human_baseline_sha256-or[string]$goalAfter.goal.updated_at-cne[string]$goalBefore.goal.updated_at){throw 'R119_GOAL_CHANGED'}
  $afterHpc=Read-HpcState;Assert-HpcSame $beforeHpc $afterHpc
  $webAfter=Tree $webRoot
  if($webAfter.file_count-ne$webBefore.file_count-or$webAfter.sha256-cne$webBefore.sha256){throw 'R119_WEB_TREE_CHANGED'}
  $finalTree=Tree $target
  if($finalTree.sha256-cne$installedTree.sha256-or$finalTree.file_count-ne32){throw 'R119_FINAL_RUNTIME_TREE_REJECTED'}
  $payload=[ordered]@{
    schema='evomind.g21_hpc_asset_probe_runtime_deployment_result.r119.v1';status='installed_and_verified';transaction_id=$TransactionId;approval_sha256=$ApprovalSha256;
    web_build_id=$fixedWebBuild;web_tree_before=$webBefore;web_tree_after=$webAfter;web_tree_unchanged=$true;pre_loopback=$preLoopback;pre_public=$prePublic;loopback=$loopback;public=$public;
    managed_before=$managedBefore;managed_after=$managedAfter;fresh_role_pids=$true;fresh_listener_pids=$true;run_before=$runBefore;run_after=$runAfter;
    old_runtime_tree=$currentRuntime;installed_runtime_tree=$finalTree;backup_root=$backup;database_backup=$databaseBackup;
    goal_before=$goalBefore.goal;goal=$goalAfter.goal;session_goal_metadata=$goalAfter.session_goal_metadata;goal_record_count=[int]$goalBefore.goal_record_count;goal_unchanged=$true;database_content_unchanged=$true;database_mutation_attempted=$false;database_raw_hash_compared_after_restart=$false;hpc_asset_probe_schema_verified=$true;hpc_asset_probe_read_only=$true;tool_after_start=$toolAfterStart;tool_after_restart=$toolAfter;
    hpc_binding_unchanged=$true;profile_unchanged=$true;byoa_unchanged=$true;hpc_accessed=$false;gpu_touched=$false;hpc_remote_writes=0;direct_signals_sent=0;other_processes_modified=$false;
    production_writes_performed=$true;production_write_category_count=4;production_write_categories=@('runtime_backup','database_backup','runtime_atomic_swap','release_reseal');managed_service_actions=@('Stop','Start','Stop','Start');web_replaced=$false;training_started=$false;goal_put_calls=0;hpc_tool_calls=0
  }
  Write-AtomicJson $resultPath $payload;$payload|ConvertTo-Json -Depth 20 -Compress
}catch{
  $failure=$_;$rollback='not_required'
  if($targetBackedUp){
    try{
      if(-not$serviceStopped){Invoke-ManagedAction Stop|Out-Null;$serviceStopped=$true;Assert-Stopped}
      if($targetBackedUp-and(Test-Path -LiteralPath $target)){Remove-Item -LiteralPath $target -Recurse -Force}
      if(-not(Test-Path -LiteralPath (Join-Path $backup 'original\evomind_runtime') -PathType Container)){throw 'R119_RUNTIME_BACKUP_MISSING'}
      Move-Item -LiteralPath (Join-Path $backup 'original\evomind_runtime') -Destination $target
      Restore-Database $backup $databaseBackup
      & $seal|Out-Null
      Assert-HpcSame $beforeHpc (Read-HpcState)
      Invoke-ManagedAction Start|Out-Null;$serviceStopped=$false
      $rollbackLoopback=Wait-Exact 'http://127.0.0.1:8088/api/healthz' $fixedWebBuild
      $rollbackPublic=Wait-Exact 'https://evomind.zhjjq.tech/api/healthz' $fixedWebBuild
      $restored=Tree $target
      if($restored.file_count-ne32-or$restored.sha256-cne$baselineTreeSha){throw 'R119_ROLLBACK_RUNTIME_REJECTED'}
      $rollbackWeb=Tree $webRoot;if($rollbackWeb.file_count-ne$webBefore.file_count-or$rollbackWeb.sha256-cne$webBefore.sha256){throw 'R119_ROLLBACK_WEB_TREE_CHANGED'}
      $rollbackManaged=Assert-Managed $fixedWebBuild;Assert-FreshManaged $managedBefore $rollbackManaged;Assert-HpcSame $beforeHpc (Read-HpcState)
      $rollback='restored_and_verified'
    }catch{$rollback='failed';$rollbackError=Safe-Error $_.Exception.Message}
  }elseif($serviceStopped){
    try{Invoke-ManagedAction Start|Out-Null;$serviceStopped=$false;Wait-Exact 'http://127.0.0.1:8088/api/healthz' $fixedWebBuild|Out-Null;$rollback='start_only_recovered'}catch{$rollback='failed';$rollbackError=Safe-Error $_.Exception.Message}
  }
  $safeMessage=Safe-Error $failure.Exception.Message
  Write-AtomicJson $errorPath ([ordered]@{schema='evomind.g21_hpc_asset_probe_runtime_deployment_error.r119.v1';status='failed';transaction_id=$TransactionId;approval_sha256=$ApprovalSha256;message=$safeMessage;rollback=$rollback;rollback_error=$rollbackError;hpc_accessed=$false;gpu_touched=$false;hpc_remote_writes=0;direct_signals_sent=0;other_processes_modified=$false;production_writes_performed=[bool]($backup-or$bundleCandidateIntroduced-or$targetBackedUp-or$targetSwapped);target_backed_up=$targetBackedUp;target_swapped=$targetSwapped;managed_service_actions_performed=[bool]($serviceStopped-or$backup)})
  throw
}finally{
  if($bundleCandidateIntroduced-and$bundleCandidate-and(Test-Path -LiteralPath $bundleCandidate)){Remove-Item -LiteralPath $bundleCandidate -Recurse -Force -ErrorAction SilentlyContinue}
  if($owned){try{[void]$mutex.ReleaseMutex()}catch{}};$mutex.Dispose()
}

