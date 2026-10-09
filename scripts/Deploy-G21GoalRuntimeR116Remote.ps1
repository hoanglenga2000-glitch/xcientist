param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [Parameter(Mandatory)][string]$RemoteRoot,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$ApprovalSha256
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = 'C:\ProgramData\EvoMind'
$expectedRemote = [IO.Path]::GetFullPath("C:\SecureInput\r116-goal-runtime-$TransactionId")
if ([IO.Path]::GetFullPath($RemoteRoot) -cne $expectedRemote) { throw 'R116_REMOTE_ROOT_REJECTED' }
$approvalPath = Join-Path $expectedRemote 'approval.json'
$archive = Join-Path $expectedRemote 'runtime.zip'
$sourceManifestPath = Join-Path $expectedRemote 'source-manifest.json'
$buildResultPath = Join-Path $expectedRemote 'build-result.json'
$goalSpecPath = Join-Path $expectedRemote 'goal-spec.json'
$humanPath = Join-Path $expectedRemote 'human-baseline.json'
$boardPath = Join-Path $expectedRemote 'goal-board.json'
$bootstrapPath = Join-Path $expectedRemote 'bootstrap.json'
$rejectionPath = Join-Path $expectedRemote 'rejected-candidate.json'
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
$r115ZipSha = 'fc65d7134f9f53fa6eda21ee2fcd804a330d9184a83d5206a6b7469a217d89ba'
$r115TreeSha = 'b92cc96dae5ac1c1ed3641999eec56eda782b4959ac8709a0c0584ac1438955a'
$goalSpecFileSha = '380a3b2c037c8067a4618717cb7aaf8847d323d4be401a88d64009468132848f'
$humanFileSha = '5b6ed3d912ad1cd4655c96f832629cf9603a4b0ad58ab6e27c01861fadd69c5f'
$rejectedCandidateSha = '37929a1c9baf682e4f946450f834f10fe3b7a78323084ae96948bd0e92f8de8d'
$rejectionEvidenceSha = '8900c32d41a47561715dc215ec2e560e338c2d164d6fa410554b7a2b83b5fb83'

function Sha([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }

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
  if ($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "R116_REPARSE_REJECTED:$Path" }
  $bad = @(Get-ChildItem -LiteralPath $Path -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })
  if ($bad.Count) { throw "R116_REPARSE_REJECTED:$($bad[0].FullName)" }
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
  $config = Read-Object (Join-Path $root 'config\node-config.json') 'R116_NODE_CONFIG'
  $runtimeManifestPath = Join-Path ([string]$config.web_runtime_root) 'runtime-build-manifest.json'
  $runtime = Read-Object $runtimeManifestPath 'R116_WEB_RUNTIME_MANIFEST'
  [ordered]@{ config=$config; runtime=$runtime; runtime_manifest_path=$runtimeManifestPath }
}

function Get-RuntimeHeaders {
  $current = Read-Runtime
  $tokenPath = Join-Path ([string]$current.config.data_root) 'workspace\runtime\runtime.token'
  if (-not (Test-Path -LiteralPath $tokenPath -PathType Leaf)) { throw 'R116_RUNTIME_TOKEN_MISSING' }
  $token = (Get-Content -LiteralPath $tokenPath -Raw).Trim()
  if (-not $token) { throw 'R116_RUNTIME_TOKEN_EMPTY' }
  @{ Authorization = 'Bearer ' + $token }
}

function Assert-Managed([string]$ExpectedBuild) {
  $statePath = Join-Path $root 'state\node-processes.json'
  $state = Read-Object $statePath 'R116_NODE_PROCESS_STATE'
  $roles = [ordered]@{}
  foreach ($role in @('tenant_enrollment_queue','llm_gateway','python_runtime','web')) {
    $records = @($state.records | Where-Object { [string]$_.role -ceq $role })
    if ($records.Count -ne 1) { throw "R116_ROLE_COUNT_REJECTED:$role" }
    $pidValue = [int]$records[0].pid
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction Stop
    $owner = $process | Invoke-CimMethod -MethodName GetOwner
    if ([string]$owner.User -cne 'EvoMindSvc') { throw "R116_ROLE_OWNER_REJECTED:$role" }
    $roles[$role] = [ordered]@{ pid=$pidValue; owner=[string]$owner.User; name=[string]$process.Name }
  }
  if(@($roles.Values|ForEach-Object{[int]$_.pid}|Select-Object -Unique).Count-ne4){throw 'R116_ROLE_PID_DUPLICATE'}
  $listeners = [ordered]@{}
  foreach ($port in @(65068,8765,8088,7890)) {
    $rows = @(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue)
    if ($rows.Count -ne 1) { throw "R116_LISTENER_COUNT_REJECTED:$port" }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$rows[0].OwningProcess)" -ErrorAction Stop
    $owner = $process | Invoke-CimMethod -MethodName GetOwner
    if ([string]$owner.User -cne 'EvoMindSvc') { throw "R116_LISTENER_OWNER_REJECTED:$port" }
    $identity=switch($port){
      65068{if([string]$process.Name-ceq'cliproxy-7.2.128.exe'-and[string]$process.ExecutablePath-match'ProgramData\\EvoMind\\bin\\cliproxy-7\.2\.128\.exe$'){'llm_gateway'}else{$null}}
      8765{if([string]$process.Name-in@('python.exe','pythonw.exe')-and[string]$process.CommandLine-match'run_python_runtime\.py.+--port\s+8765'){'python_runtime_child'}else{$null}}
      8088{if([string]$process.Name-ceq'node.exe'-and[string]$process.CommandLine-match'(?:^|\s)server\.js(?:\s|$)'){'web'}else{$null}}
      7890{if([string]$process.Name-in@('python.exe','pythonw.exe')-and[string]$process.CommandLine-match'hpc_socks_bridge\.py.+--listen-port\s+7890'){'managed_hpc_bridge'}else{$null}}
    }
    if(-not$identity){throw "R116_LISTENER_IDENTITY_REJECTED:$port"}
    $listeners[[string]$port] = [ordered]@{ pid=[int]$rows[0].OwningProcess; parent_pid=[int]$process.ParentProcessId; owner=[string]$owner.User; name=[string]$process.Name; identity=$identity }
  }
  if(@($listeners.Values|ForEach-Object{[int]$_.pid}|Select-Object -Unique).Count-ne4){throw 'R116_LISTENER_PID_DUPLICATE'}
  foreach($binding in @(
    [pscustomobject]@{role='llm_gateway';port='65068'},
    [pscustomobject]@{role='web';port='8088'}
  )){if([int]$roles[$binding.role].pid-ne[int]$listeners[$binding.port].pid){throw "R116_ROLE_LISTENER_PID_MISMATCH:$($binding.role)"}}
  if([int]$listeners['8765'].parent_pid-ne[int]$roles['python_runtime'].pid){throw 'R116_PYTHON_LISTENER_PARENT_REJECTED'}
  $current = Read-Runtime
  if ([string]$current.runtime.build_id -cne $ExpectedBuild) { throw 'R116_WEB_BUILD_MISMATCH' }
  [ordered]@{ build_id=[string]$current.runtime.build_id; roles=$roles; listeners=$listeners; state_sha256=Sha $statePath }
}

function Assert-FreshManaged([object]$Before,[object]$After){
  foreach($role in @('tenant_enrollment_queue','llm_gateway','python_runtime','web')){
    if([int]$Before.roles[$role].pid-eq[int]$After.roles[$role].pid){throw "R116_ROLE_PID_NOT_FRESH:$role"}
  }
  foreach($port in @('65068','8765','8088','7890')){
    if([int]$Before.listeners[$port].pid-eq[int]$After.listeners[$port].pid){throw "R116_LISTENER_PID_NOT_FRESH:$port"}
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
  throw "R116_EXACT_HEALTH_TIMEOUT:$Uri"
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
  ) { throw 'R116_TARGET_RUN_NOT_TERMINAL_QUIESCENT' }
  $lastEventSeq=if($snapshot.PSObject.Properties.Name -contains 'last_event_seq'){[int64]$snapshot.last_event_seq}else{[int64]0}
  [ordered]@{ run_id=$fixedRun; status=[string]$snapshot.status; terminal=$true; effective_active_count=0; pending_approval_count=0; last_event_seq=$lastEventSeq }
}

function Read-HpcState {
  $current = Read-Runtime
  $tenantRoot = Join-Path $root 'byoa\tenants\tenant_ba0ef9d3767f2fb385b856e0'
  $bindingPath = Join-Path $tenantRoot 'hpc-binding.json'
  $binding = Read-Object $bindingPath 'R116_HPC_BINDING'
  if (
    [string]$binding.schema -cne 'evomind.tenant_hpc_binding.v1' -or
    [string]$binding.tenant_id -cne 'tenant_ba0ef9d3767f2fb385b856e0' -or
    [int]$binding.job_id -lt 1 -or [int]$binding.allocation_generation -lt 1 -or
    [string]$binding.credential_profile -cne "tenant_ba0ef9d3767f2fb385b856e0_job$([int]$binding.job_id)_g$([int]$binding.allocation_generation)"
  ) { throw 'R116_HPC_BINDING_REJECTED' }
  $profileInfo = Get-CimInstance Win32_UserProfile -ErrorAction SilentlyContinue | Where-Object { [IO.Path]::GetFileName(([string]$_.LocalPath).TrimEnd('\')) -ieq [string]$current.config.dedicated_user } | Select-Object -First 1
  $serviceHome = if ($profileInfo) { [string]$profileInfo.LocalPath } else { "C:\Users\$([string]$current.config.dedicated_user)" }
  $profileRoot = Join-Path $serviceHome "AppData\Roaming\ResearchAgentWorkstation\profiles\$([string]$binding.credential_profile)"
  if (-not (Test-Path -LiteralPath $profileRoot -PathType Container)) { throw 'R116_HPC_PROFILE_MISSING' }
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
  ) { throw 'R116_HPC_LOCAL_STATE_CHANGED' }
}

function Invoke-ManagedAction([ValidateSet('Stop','Start')][string]$Name) {
  if (-not (Test-Path -LiteralPath $action -PathType Leaf)) { throw 'R116_SERVICE_ACTION_MISSING' }
  $result = @(& $action -Action $Name -TimeoutMinutes 12) | Out-String | ConvertFrom-Json
  if ([string]$result.status -cne 'completed' -or [string]$result.action -cne $Name) { throw "R116_MANAGED_$($Name.ToUpperInvariant())_FAILED" }
  $result
}

function Assert-Stopped {
  $listeners = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in @(65068,8765,8088,7890) })
  if ($listeners.Count) { throw 'R116_POST_STOP_LISTENERS_REMAIN' }
}

function Read-ApprovalAndInputs {
  foreach ($path in @($approvalPath,$archive,$sourceManifestPath,$buildResultPath,$goalSpecPath,$humanPath,$boardPath,$bootstrapPath,$rejectionPath,$action,$seal,$target)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "R116_INPUT_MISSING:$path" }
  }
  if ((Sha $approvalPath) -cne $ApprovalSha256) { throw 'R116_APPROVAL_HASH_REJECTED' }
  $approval = Read-Object $approvalPath 'R116_APPROVAL'
  if (
    [string]$approval.schema -cne 'evomind.g21_goal_runtime_deployment_approval.v1' -or [string]$approval.status -cne 'approved' -or
    $approval.single_use -isnot [bool] -or -not [bool]$approval.single_use -or [string]$approval.deployment_id -cne $TransactionId -or
    [string]$approval.scope.mode -cne 'runtime_only' -or [string]$approval.scope.run_id -cne $fixedRun -or [string]$approval.scope.allocation -cne $fixedAllocation -or
    [string]$approval.scope.expected_web_build_id -cne $fixedWebBuild -or [string]$approval.scope.expected_current_runtime_tree_sha256 -cne $r115TreeSha -or
    [string]$approval.baseline.zip_sha256 -cne $r115ZipSha -or [string]$approval.baseline.runtime_tree_sha256 -cne $r115TreeSha -or [int]$approval.baseline.runtime_file_count -ne 30 -or
    [string]$approval.candidate.release -cne 'r116-g21-goal-r115-base' -or [string]$approval.candidate.zip_sha256 -ceq $rejectedCandidateSha -or [int]$approval.candidate.runtime_file_count -ne 32 -or [int]$approval.candidate.changed_file_count -ne 7 -or
    [string]$approval.bootstrap.goal_spec_file_sha256 -cne $goalSpecFileSha -or [string]$approval.bootstrap.human_baseline_file_sha256 -cne $humanFileSha -or
    [string]$approval.bootstrap.goal_record_status -cne 'blocked' -or
    [string]$approval.bootstrap.artifact_filename -cne 'bootstrap.json' -or
    [string]$approval.deployer.remote_runner_filename -cne [IO.Path]::GetFileName($PSCommandPath) -or (Sha $PSCommandPath) -cne [string]$approval.deployer.remote_runner_sha256 -or
    [string]$approval.deployer.service_action_path -cne $action -or
    [string]$approval.verification.candidate_verifier_filename -cne 'verify_g21_goal_runtime_r116_candidate.py' -or [string]$approval.verification.candidate_verifier_sha256 -cnotmatch '^[a-f0-9]{64}$' -or
    [string]$approval.rejected_candidate.zip_sha256 -cne $rejectedCandidateSha -or [string]$approval.rejected_candidate.evidence_sha256 -cne $rejectionEvidenceSha -or [string]$approval.rejected_candidate.status -cne 'NOT_DEPLOYABLE' -or
    -not [bool]$approval.prohibited.web_replacement -or -not [bool]$approval.prohibited.direct_scheduled_task_start -or -not [bool]$approval.prohibited.process_kill -or -not [bool]$approval.prohibited.direct_sqlite_write -or
    -not [bool]$approval.prohibited.hpc_access -or -not [bool]$approval.prohibited.chrome -or -not [bool]$approval.prohibited.training
  ) { throw 'R116_APPROVAL_CONTRACT_REJECTED' }
  try { $issued=[DateTimeOffset]::Parse([string]$approval.issued_at_utc);$expires=[DateTimeOffset]::Parse([string]$approval.expires_at_utc) }
  catch { throw 'R116_APPROVAL_TIME_REJECTED' }
  $now=[DateTimeOffset]::UtcNow
  if ($expires -le $issued -or $now -lt $issued.AddMinutes(-5) -or $now -ge $expires -or ($expires-$issued).TotalHours -gt 24) { throw 'R116_APPROVAL_EXPIRED_OR_TIME_REJECTED' }
  $fixedFiles = @(
    [pscustomobject]@{path=$archive;name='runtime.zip';expected=[string]$approval.candidate.zip_sha256},
    [pscustomobject]@{path=$sourceManifestPath;name='source-manifest.json';expected=[string]$approval.candidate.source_manifest_sha256},
    [pscustomobject]@{path=$buildResultPath;name='build-result.json';expected=[string]$approval.candidate.build_result_sha256},
    [pscustomobject]@{path=$goalSpecPath;name='goal-spec.json';expected=[string]$approval.bootstrap.goal_spec_file_sha256},
    [pscustomobject]@{path=$humanPath;name='human-baseline.json';expected=[string]$approval.bootstrap.human_baseline_file_sha256},
    [pscustomobject]@{path=$boardPath;name='goal-board.json';expected=[string]$approval.bootstrap.goal_board_file_sha256},
    [pscustomobject]@{path=$bootstrapPath;name='bootstrap.json';expected=[string]$approval.bootstrap.artifact_sha256},
    [pscustomobject]@{path=$rejectionPath;name='rejected-candidate.json';expected=[string]$approval.rejected_candidate.evidence_sha256}
  )
  foreach ($item in $fixedFiles) { if ((Sha $item.path) -cne $item.expected) { throw "R116_STAGED_HASH_REJECTED:$($item.name)" } }
  $receipts=@($approval.test_receipts)
  if($receipts.Count-lt1){throw 'R116_TEST_RECEIPT_REQUIRED'}
  for($index=0;$index-lt$receipts.Count;$index++){
    $expectedName="test-receipt-$($index+1).json"
    if([string]$receipts[$index].filename-cne$expectedName-or[string]$receipts[$index].schema-cne'evomind.g21_goal_runtime_test_receipt.v1'){throw 'R116_TEST_RECEIPT_BINDING_REJECTED'}
    $receiptPath=Join-Path $expectedRemote $expectedName
    if(-not(Test-Path -LiteralPath $receiptPath -PathType Leaf)-or(Sha $receiptPath)-cne[string]$receipts[$index].sha256){throw 'R116_TEST_RECEIPT_HASH_REJECTED'}
    $receipt=Read-Object $receiptPath 'R116_TEST_RECEIPT'
    if([string]$receipt.status-cne'passed'-or[string]$receipt.run_id-cne$fixedRun-or[string]$receipt.candidate_zip_sha256-cne[string]$approval.candidate.zip_sha256-or[string]$receipt.bootstrap_artifact_sha256-cne[string]$approval.bootstrap.artifact_sha256-or[string]$receipt.verifier.filename-cne[string]$approval.verification.candidate_verifier_filename-or[string]$receipt.verifier.sha256-cne[string]$approval.verification.candidate_verifier_sha256-or[bool]$receipt.checks.hpc_accessed-or[bool]$receipt.checks.gpu_touched-or[int]$receipt.checks.remote_writes-ne0-or[int]$receipt.test_artifacts.targeted_goal_builder_junit.tests-lt55-or[int]$receipt.test_artifacts.targeted_goal_builder_junit.failures-ne0-or[int]$receipt.test_artifacts.targeted_goal_builder_junit.errors-ne0-or[int]$receipt.test_artifacts.runtime_regression_junit.tests-lt112-or[int]$receipt.test_artifacts.runtime_regression_junit.failures-ne0-or[int]$receipt.test_artifacts.runtime_regression_junit.errors-ne0){throw 'R116_TEST_RECEIPT_CONTRACT_REJECTED'}
  }
  $bootstrap=Read-Object $bootstrapPath 'R116_BOOTSTRAP_ARTIFACT'
  if([string]$bootstrap.schema-cne'evomind.g21_goal_runtime_bootstrap.v1'-or[string]$bootstrap.run_id-cne$fixedRun-or[string]$bootstrap.allocation-cne$fixedAllocation-or[string]$bootstrap.goal_record_status-cne'blocked'-or[string]$bootstrap.goal_spec_file_sha256-cne$goalSpecFileSha-or[string]$bootstrap.human_baseline_evidence_file_sha256-cne$humanFileSha-or[string]$bootstrap.goal_board_file_sha256-cne[string]$approval.bootstrap.goal_board_file_sha256-or[string]$bootstrap.goal_board_sha256-cne[string]$approval.bootstrap.goal_board_sha256-or[bool]$bootstrap.production_deployable){throw 'R116_BOOTSTRAP_ARTIFACT_REJECTED'}
  $rejection=Read-Object $rejectionPath 'R116_REJECTION_EVIDENCE'
  if([string]$rejection.schema-cne'evomind.g21_goal_runtime_candidate_rejection.v1'-or[string]$rejection.status-cne'NOT_DEPLOYABLE'-or-not@($rejection.candidate_files|Where-Object{[string]$_.sha256-ceq$rejectedCandidateSha}).Count-or[bool]$rejection.production_deployed){throw 'R116_REJECTION_EVIDENCE_REJECTED'}
  $buildResult=Read-Object $buildResultPath 'R116_BUILD_RESULT'
  if([string]$buildResult.schema-cne'evomind.g21_goal_runtime_overlay.clean.v1'-or[string]$buildResult.status-cne'built_and_verified'-or[string]$buildResult.zip_sha256-cne[string]$approval.candidate.zip_sha256-or[string]$buildResult.manifest_sha256-cne[string]$approval.candidate.source_manifest_sha256-or[string]$buildResult.candidate_runtime_tree_sha256-cne[string]$approval.candidate.installed_runtime_tree_sha256-or[string]$buildResult.bootstrap_artifact_sha256-cne[string]$approval.bootstrap.artifact_sha256-or[bool]$buildResult.production_deployable-or[bool]$buildResult.production_deployed-or[bool]$buildResult.hpc_accessed-or[bool]$buildResult.gpu_touched-or[int]$buildResult.remote_writes-ne0){throw 'R116_BUILD_RESULT_REJECTED'}
  [ordered]@{approval=$approval;receipts=$receipts}
}

function Read-Candidate([object]$Approval) {
  if((Sha $archive)-ceq$rejectedCandidateSha){throw 'REJECTED_CANDIDATE_ZIP_SHA256'}
  if (Test-Path -LiteralPath $stage) { throw 'R116_STAGE_EXISTS' }
  Expand-Archive -LiteralPath $archive -DestinationPath $stage
  Assert-NoReparse $stage
  $manifestPath = Join-Path $stage 'goal-runtime-overlay-manifest.json'
  if (Test-Path -LiteralPath (Join-Path $stage 'runtime-hotfix-manifest.json')) { throw 'R116_STALE_RUNTIME_MANIFEST_REJECTED' }
  $manifest = Read-Object $manifestPath 'R116_GOAL_MANIFEST'
  if ((Sha $manifestPath) -cne [string]$Approval.candidate.embedded_manifest_sha256 -or (Sha $sourceManifestPath) -cne (Sha $manifestPath)) { throw 'R116_MANIFEST_BINDING_REJECTED' }
  if (
    [string]$manifest.schema -cne 'evomind.g21_goal_runtime_overlay.clean.v1' -or [string]$manifest.run_id -cne $fixedRun -or [string]$manifest.allocation -cne $fixedAllocation -or
    [string]$manifest.target -cne 'bundle/runtime/evomind_runtime' -or [string]$manifest.baseline_zip_sha256 -cne $r115ZipSha -or
    [int]$manifest.baseline_runtime_file_count -ne 30 -or [int]$manifest.candidate_runtime_file_count -ne 32 -or [int]$manifest.file_count -ne 32 -or [int]$manifest.changed_file_count -ne 7 -or
    [string]$manifest.candidate_runtime_tree_sha256 -cne [string]$Approval.candidate.installed_runtime_tree_sha256 -or [string]$manifest.bootstrap_artifact_sha256 -cne [string]$Approval.bootstrap.artifact_sha256 -or
    [bool]$manifest.production_deployable -or [bool]$manifest.production_deployed -or -not[bool]$manifest.reviewed_goal_only -or
    [bool]$manifest.hpc_accessed -or [bool]$manifest.gpu_touched -or [int]$manifest.remote_writes -ne 0 -or
    [string]$manifest.goal_spec_sha256 -cne $goalSpecFileSha -or [string]$manifest.human_baseline_evidence_sha256 -cne $humanFileSha
  ) { throw 'R116_GOAL_MANIFEST_CONTRACT_REJECTED' }
  if ($manifest.PSObject.Properties.Name -contains 'goal_board_file_sha256' -and [string]$manifest.goal_board_file_sha256 -cne [string]$Approval.bootstrap.goal_board_file_sha256) { throw 'R116_GOAL_BOARD_FILE_BINDING_REJECTED' }
  if ($manifest.PSObject.Properties.Name -contains 'goal_board_sha256' -and [string]$manifest.goal_board_sha256 -cne [string]$Approval.bootstrap.goal_board_sha256) { throw 'R116_GOAL_BOARD_BINDING_REJECTED' }
  $declared=@{}
  foreach($entry in @($manifest.files)){
    $relative=[string]$entry.path
    if($relative-cnotmatch'^evomind_runtime/[A-Za-z0-9_./-]+\.py$'-or$declared.ContainsKey($relative)){throw 'R116_MANIFEST_ENTRY_REJECTED'}
    $declared[$relative]=$entry
  }
  $actual=@(Get-ChildItem -LiteralPath $stage -Recurse -File -Force|ForEach-Object{$_.FullName.Substring($stage.Length+1).Replace('\','/')}|Where-Object{$_-ne'goal-runtime-overlay-manifest.json'})
  if($declared.Count-ne32-or$actual.Count-ne32-or@($actual|Where-Object{-not$declared.ContainsKey($_)}).Count){throw 'R116_CANDIDATE_FILESET_REJECTED'}
  foreach($relative in $declared.Keys){$path=Join-Path $stage $relative.Replace('/','\');$entry=$declared[$relative];if(-not(Test-Path -LiteralPath $path -PathType Leaf)-or[long](Get-Item -LiteralPath $path).Length-ne[long]$entry.bytes-or(Sha $path)-cne[string]$entry.sha256){throw "R116_CANDIDATE_HASH_REJECTED:$relative"}}
  # Keep the verified candidate in C:\SecureInput until every read-only
  # production gate has passed and the managed service is stopped.  A
  # nonterminal Run must leave the production bundle and database untouched.
  $candidate=Join-Path $stage 'evomind_runtime'
  Assert-NoReparse $candidate
  $tree=Tree $candidate
  if($tree.file_count-ne32-or$tree.sha256-cne[string]$Approval.candidate.installed_runtime_tree_sha256){throw 'R116_CANDIDATE_TREE_REJECTED'}
  [ordered]@{path=$candidate;manifest=$manifest;tree=$tree}
}

function Backup-Database([string]$BackupRoot, [object]$Config) {
  $database = Join-Path ([string]$Config.data_root) 'workspace\runtime\runtime.sqlite3'
  if (-not (Test-Path -LiteralPath $database -PathType Leaf)) { throw 'R116_RUNTIME_DATABASE_MISSING' }
  $destination=Join-Path $BackupRoot 'database';if(Test-Path -LiteralPath $destination){throw 'R116_DATABASE_BACKUP_PATH_EXISTS'};New-Item -ItemType Directory -Path $destination|Out-Null
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
  foreach($row in @($DatabaseBackup.files)){if([bool]$row.existed){$name=if([string]$row.suffix){"runtime.sqlite3$([string]$row.suffix)"}else{'runtime.sqlite3'};$source=Join-Path (Join-Path $BackupRoot 'database') $name;Copy-Item -LiteralPath $source -Destination "$($DatabaseBackup.path)$([string]$row.suffix)";if((Sha "$($DatabaseBackup.path)$([string]$row.suffix)")-cne[string]$row.sha256){throw 'R116_DATABASE_ROLLBACK_HASH_REJECTED'}}}
}

function Read-RawSessionEvents([int64]$AfterSeq,[int64]$UpperSeq,[object]$Headers){
  if($AfterSeq-lt0-or$UpperSeq-lt$AfterSeq){throw 'R116_GOAL_EVENT_WINDOW_REJECTED'}
  $cursor=$AfterSeq;$pages=0;$events=@()
  while($cursor-lt$UpperSeq){
    $pages++;if($pages-gt100){throw 'R116_GOAL_EVENT_PAGINATION_LIMIT'}
    $page=Invoke-RestMethod -Uri "http://127.0.0.1:8765/v1/sessions/$fixedRun/events?after=$cursor" -Headers $Headers -TimeoutSec 20
    $rows=@($page.events|Where-Object{[int64]$_.seq-le$UpperSeq})
    if(-not$rows.Count){throw 'R116_GOAL_EVENT_PAGE_GAP'}
    $last=$cursor
    foreach($event in $rows){
      $seq=[int64]$event.seq
      if($seq-le$last-or$seq-gt$UpperSeq){throw 'R116_GOAL_EVENT_SEQUENCE_REJECTED'}
      $events+=$event;$last=$seq
    }
    if($last-le$cursor){throw 'R116_GOAL_EVENT_CURSOR_STALLED'}
    $cursor=$last
  }
  [ordered]@{after_seq=$AfterSeq;upper_seq=$UpperSeq;last_seq=$cursor;page_count=$pages;event_count=$events.Count;events=$events}
}

function Invoke-GoalBootstrap([object]$Approval,[int64]$EventAfterSeq) {
  $headers=Get-RuntimeHeaders;$jsonHeaders=$headers+@{'Content-Type'='application/json'}
  $spec=Read-Object $goalSpecPath 'R116_GOAL_SPEC';$human=Read-Object $humanPath 'R116_HUMAN_BASELINE';$board=Read-Object $boardPath 'R116_GOAL_BOARD'
  if([string]$spec.human_baseline_evidence_sha256-cne$humanFileSha-or[string]$human.classification-cne'HUMAN_BASELINE_UNDEFINED'-or[bool]$human.completion.training_authorized_by_this_artifact-or[string]$board.goal_record_status-cne'blocked'){throw 'R116_BOOTSTRAP_EVIDENCE_REJECTED'}
  $body=[ordered]@{run_id=$fixedRun;spec=$spec;board=$board;status='blocked'}|ConvertTo-Json -Depth 30 -Compress
  $post=$null;$created=$false
  try{$response=Invoke-WebRequest -Uri 'http://127.0.0.1:8765/v1/goals' -Method Post -Headers $jsonHeaders -Body $body -UseBasicParsing -TimeoutSec 30;$post=$response.Content|ConvertFrom-Json;$created=([int]$response.StatusCode-eq201)}catch{
    try{$post=Invoke-RestMethod -Uri 'http://127.0.0.1:8765/v1/goals/goal_g21_five_competition' -Headers $headers -TimeoutSec 20}catch{throw 'R116_GOAL_POST_AND_RECOVERY_GET_FAILED'}
  }
  $goal=Invoke-RestMethod -Uri 'http://127.0.0.1:8765/v1/goals/goal_g21_five_competition' -Headers $headers -TimeoutSec 20
  if([string]$goal.id-cne'goal_g21_five_competition'-or[string]$goal.run_id-cne$fixedRun-or[string]$goal.status-cne'blocked'-or[string]$goal.spec_sha256-cne[string]$Approval.bootstrap.goal_spec_sha256-or[string]$goal.board_sha256-cne[string]$Approval.bootstrap.goal_board_sha256-or[string]$goal.human_baseline_sha256-cne$humanFileSha){throw 'R116_GOAL_IDENTITY_REJECTED'}
  $list=Invoke-RestMethod -Uri 'http://127.0.0.1:8765/v1/goals' -Headers $headers -TimeoutSec 20
  if(@($list.goals).Count-ne1-or[string]$list.goals[0].id-cne[string]$goal.id){throw 'R116_GOAL_LIST_REJECTED'}
  $runGoal=Invoke-RestMethod -Uri "http://127.0.0.1:8765/v1/runs/$fixedRun/goal" -Headers $headers -TimeoutSec 20
  if([string]$runGoal.id-cne[string]$goal.id-or[string]$runGoal.board_sha256-cne[string]$goal.board_sha256){throw 'R116_RUN_GOAL_REJECTED'}
  $runSnapshot=Invoke-RestMethod -Uri "http://127.0.0.1:8765/v1/runs/$fixedRun" -Headers $headers -TimeoutSec 20
  if([string]$runSnapshot.goal.id-cne[string]$goal.id-or[string]$runSnapshot.goal.spec_sha256-cne[string]$goal.spec_sha256-or[string]$runSnapshot.goal.board_sha256-cne[string]$goal.board_sha256){throw 'R116_RUN_GOAL_METADATA_PROJECTION_REJECTED'}
  $session=Invoke-RestMethod -Uri "http://127.0.0.1:8765/v1/sessions/$fixedRun" -Headers $headers -TimeoutSec 20
  $metadata=$session.metadata
  if([string]$metadata.goal_id-cne[string]$goal.id-or[string]$metadata.goal_key-cne[string]$goal.goal_key-or[string]$metadata.goal_spec_sha256-cne[string]$goal.spec_sha256-or[string]$metadata.goal_board_sha256-cne[string]$goal.board_sha256-or[string]$metadata.goal_human_baseline_sha256-cne[string]$goal.human_baseline_sha256){throw 'R116_SESSION_GOAL_METADATA_REJECTED'}
  $putBody=[ordered]@{board=$board;status='blocked'}|ConvertTo-Json -Depth 30 -Compress
  $updated=Invoke-RestMethod -Uri 'http://127.0.0.1:8765/v1/goals/goal_g21_five_competition' -Method Put -Headers $jsonHeaders -Body $putBody -TimeoutSec 30
  if([string]$updated.board_sha256-cne[string]$goal.board_sha256-or[string]$updated.status-cne'blocked'){throw 'R116_GOAL_PUT_REJECTED'}
  $eventUpperSnapshot=Invoke-RestMethod -Uri "http://127.0.0.1:8765/v1/runs/$fixedRun" -Headers $headers -TimeoutSec 20
  $eventUpper=[int64]$eventUpperSnapshot.last_event_seq
  $eventWindow=Read-RawSessionEvents $EventAfterSeq $eventUpper $headers
  $createdEvents=@($eventWindow.events|Where-Object{[string]$_.event_type-ceq'goal.created'})
  $updatedCount=@($eventWindow.events|Where-Object{[string]$_.event_type-ceq'goal.updated'}).Count
  if($createdEvents.Count-ne1-or$updatedCount-ne0){throw 'R116_GOAL_CREATED_EVENT_COUNT_REJECTED'}
  $createdEvent=$createdEvents[0]
  if([int64]$createdEvent.seq-le$EventAfterSeq-or[string]$createdEvent.payload.goal_id-cne[string]$goal.id-or[string]$createdEvent.payload.goal_key-cne[string]$goal.goal_key-or[string]$createdEvent.payload.spec_sha256-cne[string]$goal.spec_sha256-or[string]$createdEvent.payload.board_sha256-cne[string]$goal.board_sha256){throw 'R116_GOAL_CREATED_EVENT_BINDING_REJECTED'}
  [ordered]@{goal=$goal;created=$created;goal_created_event_count=$createdEvents.Count;goal_record_count=@($list.goals).Count;event_window=[ordered]@{after_seq=$eventWindow.after_seq;upper_seq=$eventWindow.upper_seq;page_count=$eventWindow.page_count;event_count=$eventWindow.event_count;goal_created_seq=[int64]$createdEvent.seq};session_goal_metadata=[ordered]@{goal_id=[string]$metadata.goal_id;goal_key=[string]$metadata.goal_key;goal_spec_sha256=[string]$metadata.goal_spec_sha256;goal_board_sha256=[string]$metadata.goal_board_sha256;goal_human_baseline_sha256=[string]$metadata.goal_human_baseline_sha256}}
}

if((Test-Path -LiteralPath $resultPath)-or(Test-Path -LiteralPath $errorPath)){throw 'R116_RESULT_EXISTS'}
$mutex=[Threading.Mutex]::new($false,'Global\EvoMind-Byoa-V12-Deployment')
$owned=$false;$backup=$null;$candidate=$null;$bundleCandidate=$null;$bundleCandidateIntroduced=$false;$targetBackedUp=$false;$targetSwapped=$false;$serviceStopped=$false;$beforeHpc=$null;$databaseBackup=$null;$rollbackError=$null
try{
  $owned=$mutex.WaitOne([TimeSpan]::FromSeconds(10));if(-not$owned){throw 'R116_DEPLOYMENT_LOCKED'}
  $bound=Read-ApprovalAndInputs;$approval=$bound.approval
  $candidate=Read-Candidate $approval
  $preLoopback=Wait-Exact 'http://127.0.0.1:8088/api/healthz' $fixedWebBuild
  $prePublic=Wait-Exact 'https://evomind.zhjjq.tech/api/healthz' $fixedWebBuild
  $managedBefore=Assert-Managed $fixedWebBuild
  $runBefore=Read-RunGate
  $currentRuntime=Tree $target
  if($currentRuntime.file_count-ne30-or$currentRuntime.sha256-cne$r115TreeSha){throw 'R116_CURRENT_RUNTIME_NOT_R115'}
  $webRoot=[string](Read-Runtime).config.web_runtime_root;$webBefore=Tree $webRoot
  $beforeHpc=Read-HpcState;$otherBefore=Other-RuntimeTree
  $stop=Invoke-ManagedAction Stop;$serviceStopped=$true;Assert-Stopped
  $backup=Join-Path "$root\backups" ('pre-r116-goal-runtime-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')+'-'+$TransactionId)
  if(Test-Path -LiteralPath $backup){throw 'R116_BACKUP_PATH_EXISTS'}
  New-Item -ItemType Directory -Path (Join-Path $backup 'original')|Out-Null
  $databaseBackup=Backup-Database $backup (Read-Runtime).config
  Copy-Item -LiteralPath $approvalPath -Destination (Join-Path $backup 'approval.json')
  Write-AtomicJson (Join-Path $backup 'backup-manifest.json') ([ordered]@{schema='evomind.g21_goal_runtime_backup.v1';transaction_id=$TransactionId;approval_sha256=$ApprovalSha256;web_build_id=$fixedWebBuild;old_runtime_tree=$currentRuntime;new_runtime_tree=$candidate.tree;other_runtime_tree=$otherBefore;database=$databaseBackup;hpc=$beforeHpc})
  $bundleCandidate=Join-Path $bundleRuntime "evomind_runtime.$TransactionId.stage"
  if(Test-Path -LiteralPath $bundleCandidate){throw 'R116_BUNDLE_CANDIDATE_EXISTS'}
  Move-Item -LiteralPath $candidate.path -Destination $bundleCandidate
  $bundleCandidateIntroduced=$true
  $candidate.path=$bundleCandidate
  Move-Item -LiteralPath $target -Destination (Join-Path $backup 'original\evomind_runtime')
  $targetBackedUp=$true
  Move-Item -LiteralPath $candidate.path -Destination $target
  $targetSwapped=$true
  $installedTree=Tree $target
  if($installedTree.file_count-ne32-or$installedTree.sha256-cne[string]$approval.candidate.installed_runtime_tree_sha256){throw 'R116_INSTALLED_RUNTIME_TREE_REJECTED'}
  $otherAfter=Other-RuntimeTree
  if($otherAfter.file_count-ne$otherBefore.file_count-or$otherAfter.sha256-cne$otherBefore.sha256){throw 'R116_OTHER_RUNTIME_CHANGED'}
  & $seal|Out-Null
  Assert-HpcSame $beforeHpc (Read-HpcState)
  $start=Invoke-ManagedAction Start;$serviceStopped=$false
  Wait-Exact 'http://127.0.0.1:8088/api/healthz' $fixedWebBuild|Out-Null
  Assert-Managed $fixedWebBuild|Out-Null
  $goalBootstrap=Invoke-GoalBootstrap $approval ([int64]$runBefore.last_event_seq)
  $restartStop=Invoke-ManagedAction Stop;$serviceStopped=$true;Assert-Stopped
  $restartStart=Invoke-ManagedAction Start;$serviceStopped=$false
  $loopback=Wait-Exact 'http://127.0.0.1:8088/api/healthz' $fixedWebBuild
  $public=Wait-Exact 'https://evomind.zhjjq.tech/api/healthz' $fixedWebBuild
  $managedAfter=Assert-Managed $fixedWebBuild
  Assert-FreshManaged $managedBefore $managedAfter
  $runAfter=Read-RunGate
  if($runAfter.status-cne$runBefore.status){throw 'R116_RUN_LIFECYCLE_CHANGED'}
  $goalAfter=Invoke-GoalBootstrap $approval ([int64]$runBefore.last_event_seq)
  if([bool]$goalAfter.created-or[int]$goalAfter.goal_created_event_count-ne1-or[int]$goalAfter.goal_record_count-ne1-or[int64]$goalAfter.event_window.goal_created_seq-ne[int64]$goalBootstrap.event_window.goal_created_seq){throw 'R116_GOAL_RESTART_REUSE_REJECTED'}
  $afterHpc=Read-HpcState;Assert-HpcSame $beforeHpc $afterHpc
  $webAfter=Tree $webRoot
  if($webAfter.file_count-ne$webBefore.file_count-or$webAfter.sha256-cne$webBefore.sha256){throw 'R116_WEB_TREE_CHANGED'}
  $finalTree=Tree $target
  if($finalTree.sha256-cne$installedTree.sha256-or$finalTree.file_count-ne32){throw 'R116_FINAL_RUNTIME_TREE_REJECTED'}
  $payload=[ordered]@{
    schema='evomind.g21_goal_runtime_deployment_result.v1';status='installed_and_verified';transaction_id=$TransactionId;approval_sha256=$ApprovalSha256;
    web_build_id=$fixedWebBuild;web_tree_before=$webBefore;web_tree_after=$webAfter;web_tree_unchanged=$true;pre_loopback=$preLoopback;pre_public=$prePublic;loopback=$loopback;public=$public;
    managed_before=$managedBefore;managed_after=$managedAfter;fresh_role_pids=$true;fresh_listener_pids=$true;run_before=$runBefore;run_after=$runAfter;
    old_runtime_tree=$currentRuntime;installed_runtime_tree=$finalTree;backup_root=$backup;database_backup=$databaseBackup;
    goal=$goalAfter.goal;session_goal_metadata=$goalAfter.session_goal_metadata;goal_event_window=$goalAfter.event_window;goal_created_event_count=[int]$goalAfter.goal_created_event_count;goal_record_count=[int]$goalAfter.goal_record_count;
    hpc_binding_unchanged=$true;profile_unchanged=$true;byoa_unchanged=$true;hpc_accessed=$false;gpu_touched=$false;hpc_remote_writes=0;direct_signals_sent=0;other_processes_modified=$false;
    production_writes_performed=$true;production_write_category_count=5;production_write_categories=@('runtime_backup','database_backup','runtime_atomic_swap','release_reseal','goal_api_persistence');managed_service_actions=@('Stop','Start','Stop','Start');web_replaced=$false;training_started=$false
  }
  Write-AtomicJson $resultPath $payload;$payload|ConvertTo-Json -Depth 20 -Compress
}catch{
  $failure=$_;$rollback='not_required'
  if($targetBackedUp){
    try{
      if(-not$serviceStopped){Invoke-ManagedAction Stop|Out-Null;$serviceStopped=$true;Assert-Stopped}
      if($targetBackedUp-and(Test-Path -LiteralPath $target)){Remove-Item -LiteralPath $target -Recurse -Force}
      if(-not(Test-Path -LiteralPath (Join-Path $backup 'original\evomind_runtime') -PathType Container)){throw 'R116_RUNTIME_BACKUP_MISSING'}
      Move-Item -LiteralPath (Join-Path $backup 'original\evomind_runtime') -Destination $target
      Restore-Database $backup $databaseBackup
      & $seal|Out-Null
      Assert-HpcSame $beforeHpc (Read-HpcState)
      Invoke-ManagedAction Start|Out-Null;$serviceStopped=$false
      $rollbackLoopback=Wait-Exact 'http://127.0.0.1:8088/api/healthz' $fixedWebBuild
      $rollbackPublic=Wait-Exact 'https://evomind.zhjjq.tech/api/healthz' $fixedWebBuild
      $restored=Tree $target
      if($restored.file_count-ne30-or$restored.sha256-cne$r115TreeSha){throw 'R116_ROLLBACK_RUNTIME_REJECTED'}
      $rollbackWeb=Tree $webRoot;if($rollbackWeb.file_count-ne$webBefore.file_count-or$rollbackWeb.sha256-cne$webBefore.sha256){throw 'R116_ROLLBACK_WEB_TREE_CHANGED'}
      $rollbackManaged=Assert-Managed $fixedWebBuild;Assert-FreshManaged $managedBefore $rollbackManaged;Assert-HpcSame $beforeHpc (Read-HpcState)
      $rollback='restored_and_verified'
    }catch{$rollback='failed';$rollbackError=Safe-Error $_.Exception.Message}
  }elseif($serviceStopped){
    try{Invoke-ManagedAction Start|Out-Null;$serviceStopped=$false;Wait-Exact 'http://127.0.0.1:8088/api/healthz' $fixedWebBuild|Out-Null;$rollback='start_only_recovered'}catch{$rollback='failed';$rollbackError=Safe-Error $_.Exception.Message}
  }
  $safeMessage=Safe-Error $failure.Exception.Message
  Write-AtomicJson $errorPath ([ordered]@{schema='evomind.g21_goal_runtime_deployment_error.v1';status='failed';transaction_id=$TransactionId;approval_sha256=$ApprovalSha256;message=$safeMessage;rollback=$rollback;rollback_error=$rollbackError;hpc_accessed=$false;gpu_touched=$false;hpc_remote_writes=0;direct_signals_sent=0;other_processes_modified=$false;production_writes_performed=[bool]($backup-or$bundleCandidateIntroduced-or$targetBackedUp-or$targetSwapped);target_backed_up=$targetBackedUp;target_swapped=$targetSwapped;managed_service_actions_performed=[bool]($serviceStopped-or$backup)})
  throw
}finally{
  if($bundleCandidateIntroduced-and$bundleCandidate-and(Test-Path -LiteralPath $bundleCandidate)){Remove-Item -LiteralPath $bundleCandidate -Recurse -Force -ErrorAction SilentlyContinue}
  if($owned){try{[void]$mutex.ReleaseMutex()}catch{}};$mutex.Dispose()
}
