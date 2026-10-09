param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [Parameter(Mandatory)][string]$RemoteRoot,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$RunnerSha256
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = 'C:\ProgramData\EvoMind'
$expectedRemote = [IO.Path]::GetFullPath("C:\SecureInput\r120-probe-$TransactionId")
if ([IO.Path]::GetFullPath($RemoteRoot) -cne $expectedRemote) { throw 'R120_PROBE_REMOTE_ROOT_REJECTED' }
$resultPath = Join-Path $expectedRemote "probe-$TransactionId-result.json"
$errorPath = Join-Path $expectedRemote "probe-$TransactionId-error.json"
$fixedRun = 'run_7b1efb878afb40f396db431e91f093a5'
$fixedAllocation = 'G21'
$fixedGoal = 'goal_g21_five_competition'
$fixedWebBuild = 'overlay-controlled-secret-binding-r112-standalone-f38c40f2dbdc'
$expectedRuntimeTree = 'a9595ab4ed9a86786ca65f4456893d64741b449ec84cf8c9ddb42e9bf4110dc7'
$expectedGoalSpec = 'fbb9939dff28c355c2798bf32fce104845337064f42f9c280f2b6906cccf4036'
$expectedGoalBoard = '6b908173083889ac21f8afea52528b4b5e35d2ff862a880a6065d65b169f443c'
$schema = 'evomind.g21_terminal_preserving_asset_probe_execution.r120.v1'
$toolName = 'hpc_asset_probe'
$scope = 'cure_mindgames'
$toolCalls = 0
$probeAttempted = $false

function Sha([string]$Path) {
  $stream = [IO.File]::OpenRead($Path)
  $hash = [Security.Cryptography.SHA256]::Create()
  try { ([BitConverter]::ToString($hash.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
  finally { $stream.Dispose(); $hash.Dispose() }
}

function Write-AtomicJson([string]$Path, [object]$Value) {
  $temporary = "$Path.$PID.tmp"
  [IO.File]::WriteAllText($temporary, (($Value | ConvertTo-Json -Depth 40) + "`n"), [Text.UTF8Encoding]::new($false))
  Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Tree([string]$Path) {
  $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
  $files = @(Get-ChildItem -LiteralPath $full -Recurse -File -Force | Where-Object {
    $_.Name -notmatch '\.(pyc|pyo)$' -and $_.FullName -notmatch '[\\/]__pycache__[\\/]'
  } | Sort-Object FullName)
  $rows = @($files | ForEach-Object {
    "$($_.FullName.Substring($full.Length).TrimStart('\').Replace('\','/'))|$([long]$_.Length)|$(Sha $_.FullName)"
  })
  $hash = [Security.Cryptography.SHA256]::Create()
  try { $digest = ([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes(($rows -join "`n"))))).Replace('-','').ToLowerInvariant() }
  finally { $hash.Dispose() }
  [ordered]@{ file_count=$files.Count; sha256=$digest }
}

function Read-Object([string]$Path, [string]$Code) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$Code`:MISSING" }
  try { $value = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json }
  catch { throw "$Code`:INVALID_JSON" }
  if ($null -eq $value -or $value -is [array]) { throw "$Code`:OBJECT_REQUIRED" }
  $value
}

function Safe-Error([object]$Value) {
  $text = [string]$Value
  $text = $text -replace '(?i)(?:Bearer|Basic)\s+\S+', '[credential redacted]'
  $text = $text -replace '(?i)\b(?:password|passwd|pwd|credential|token|secret|api[_ -]?key|authorization|bearer|密码|口令|凭据|账号|登录)\s*[:=]\s*\S+', '[credential redacted]'
  $text = $text -replace '(?i)[A-Z]:\\[^\r\n]{1,400}', '[local path redacted]'
  $text = $text -replace '[\r\n]+', '_'
  if ($text.Length -gt 500) { $text = $text.Substring(0,500) }
  $text
}

function Read-Runtime {
  $config = Read-Object (Join-Path $root 'config\node-config.json') 'R120_NODE_CONFIG'
  $manifest = Read-Object (Join-Path ([string]$config.web_runtime_root) 'runtime-build-manifest.json') 'R120_WEB_RUNTIME_MANIFEST'
  [ordered]@{ config=$config; manifest=$manifest }
}

function Runtime-Headers {
  $current = Read-Runtime
  $tokenPath = Join-Path ([string]$current.config.data_root) 'workspace\runtime\runtime.token'
  if (-not (Test-Path -LiteralPath $tokenPath -PathType Leaf)) { throw 'R120_RUNTIME_TOKEN_MISSING' }
  $token = (Get-Content -LiteralPath $tokenPath -Raw -Encoding ASCII).Trim()
  if ($token -cnotmatch '^[A-Za-z0-9_-]{32,128}$') { throw 'R120_RUNTIME_TOKEN_INVALID' }
  @{ Authorization=('Bearer ' + $token) }
}

function Get-Api([string]$Path) {
  Invoke-RestMethod -Method Get -Uri ('http://127.0.0.1:8765' + $Path) -Headers (Runtime-Headers) -TimeoutSec 30
}

function Post-Api([string]$Path, [object]$Body) {
  $headers = Runtime-Headers
  $headers['Content-Type'] = 'application/json'
  Invoke-RestMethod -Method Post -Uri ('http://127.0.0.1:8765' + $Path) -Headers $headers -Body ($Body | ConvertTo-Json -Depth 12 -Compress) -TimeoutSec 180
}

function Wait-Exact([string]$Uri) {
  $samples = @()
  foreach ($number in 1..3) {
    $timer = [Diagnostics.Stopwatch]::StartNew()
    try {
      $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 15
      $payload = $response.Content | ConvertFrom-Json
      $exact = ([int]$response.StatusCode -eq 200 -and [bool]$payload.ok -and [string]$payload.status -ceq 'ready' -and [string]$payload.build_id -ceq $fixedWebBuild -and @($payload.failures).Count -eq 0)
    } finally { $timer.Stop() }
    if (-not $exact) { throw "R120_HEALTH_NOT_EXACT:$Uri" }
    $samples += [pscustomobject]@{ sample=$number; status_code=200; build_id=$fixedWebBuild; exact=$true; elapsed_ms=[math]::Round($timer.Elapsed.TotalMilliseconds,3) }
    if ($number -lt 3) { Start-Sleep -Milliseconds 250 }
  }
  [ordered]@{ status='ready'; samples=$samples }
}

function Assert-Managed {
  $statePath = Join-Path $root 'state\node-processes.json'
  $state = Read-Object $statePath 'R120_NODE_PROCESS_STATE'
  $roles = [ordered]@{}
  foreach ($role in @('tenant_enrollment_queue','llm_gateway','python_runtime','web')) {
    $records = @($state.records | Where-Object { [string]$_.role -ceq $role })
    if ($records.Count -ne 1) { throw "R120_ROLE_COUNT_REJECTED:$role" }
    $pidValue = [int]$records[0].pid
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction Stop
    $owner = $process | Invoke-CimMethod -MethodName GetOwner
    if ([string]$owner.User -cne 'EvoMindSvc') { throw "R120_ROLE_OWNER_REJECTED:$role" }
    $roles[$role] = [ordered]@{ pid=$pidValue; owner='EvoMindSvc'; name=[string]$process.Name }
  }
  if (@($roles.Values | ForEach-Object { [int]$_.pid } | Select-Object -Unique).Count -ne 4) { throw 'R120_ROLE_PID_DUPLICATE' }
  $listeners = [ordered]@{}
  foreach ($port in @(65068,8765,8088,7890)) {
    $rows = @(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue)
    if ($rows.Count -ne 1) { throw "R120_LISTENER_COUNT_REJECTED:$port" }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$rows[0].OwningProcess)" -ErrorAction Stop
    $owner = $process | Invoke-CimMethod -MethodName GetOwner
    if ([string]$owner.User -cne 'EvoMindSvc') { throw "R120_LISTENER_OWNER_REJECTED:$port" }
    $listeners[[string]$port] = [ordered]@{ pid=[int]$rows[0].OwningProcess; owner='EvoMindSvc'; name=[string]$process.Name }
  }
  $current = Read-Runtime
  if ([string]$current.manifest.build_id -cne $fixedWebBuild) { throw 'R120_WEB_BUILD_MISMATCH' }
  [ordered]@{ roles=$roles; listeners=$listeners; state_sha256=Sha $statePath; web_build_id=$fixedWebBuild }
}

function Run-Projection([object]$Snapshot) {
  $pending = @($Snapshot.approvals | Where-Object { [string]$_.status -ceq 'pending' }).Count
  $active = @($Snapshot.active_tool_calls).Count
  [ordered]@{
    run_id=[string]$Snapshot.id
    status=[string]$Snapshot.status
    terminal=[bool]$Snapshot.terminal
    effective_active_count=$active
    pending_approval_count=$pending
    error_class=[string]$Snapshot.error_class
    error_message=[string]$Snapshot.error_message
    completed_at=[string]$Snapshot.completed_at
    last_event_seq=if ($Snapshot.PSObject.Properties.Name -contains 'last_event_seq') { [int64]$Snapshot.last_event_seq } else { [int64]0 }
  }
}

function Read-RunGate {
  $snapshot = Get-Api "/v1/runs/$fixedRun"
  $projection = Run-Projection $snapshot
  $session = Get-Api "/v1/sessions/$fixedRun"
  if (
    $projection.run_id -cne $fixedRun -or $projection.status -cne 'cancelled' -or -not $projection.terminal -or
    $projection.effective_active_count -ne 0 -or $projection.pending_approval_count -ne 0 -or
    [string]$session.id -cne $fixedRun -or [string]$session.status -cne 'cancelled'
  ) { throw 'R120_RUN_NOT_CANCELLED_QUIESCENT' }
  [ordered]@{ run=$projection; session_status='cancelled' }
}

function Read-GoalGate {
  $goal = Get-Api "/v1/goals/$fixedGoal"
  $list = Get-Api '/v1/goals'
  $runGoal = Get-Api "/v1/runs/$fixedRun/goal"
  if (
    [string]$goal.id -cne $fixedGoal -or [string]$goal.run_id -cne $fixedRun -or [string]$goal.allocation -cne $fixedAllocation -or
    [string]$goal.status -cne 'blocked' -or [string]$goal.spec_sha256 -cne $expectedGoalSpec -or [string]$goal.board_sha256 -cne $expectedGoalBoard -or
    @($list.goals).Count -ne 1 -or [string]$list.goals[0].id -cne $fixedGoal -or [string]$runGoal.board_sha256 -cne $expectedGoalBoard
  ) { throw 'R120_GOAL_GATE_REJECTED' }
  $rows = @($goal.board.competitions)
  $complete = @($rows | Where-Object { [string]$_.goal_status -ceq 'VERIFIED_COMPLETE' })
  if ($rows.Count -ne 5 -or $complete.Count -ne 1 -or [string]$complete[0].competition -cne 'open_polymer') { throw 'R120_GOAL_BOARD_REJECTED' }
  [ordered]@{ id=$fixedGoal; status='blocked'; spec_sha256=$expectedGoalSpec; board_sha256=$expectedGoalBoard; complete_count=1; target_count=5 }
}

function Read-ToolGate {
  $tools = Get-Api '/v1/tools'
  $rows = @($tools.tools | Where-Object { [string]$_.name -ceq $toolName })
  if ($rows.Count -ne 1 -or -not [bool]$rows[0].read_only -or [string]$rows[0].capability -cne 'hpc.read') { throw 'R120_TOOL_GATE_REJECTED' }
  $input = $rows[0].input_schema
  if (
    [bool]$input.additionalProperties -or @($input.required).Count -ne 0 -or @($input.properties.PSObject.Properties).Count -ne 1 -or
    @($input.properties.scope.enum).Count -ne 1 -or [string]$input.properties.scope.enum[0] -cne $scope
  ) { throw 'R120_TOOL_SCHEMA_REJECTED' }
  [ordered]@{ name=$toolName; capability='hpc.read'; read_only=$true; schema_verified=$true }
}

function Assert-LifecyclePreserved([object]$Before, [object]$After) {
  foreach ($field in @('run_id','status','terminal','effective_active_count','pending_approval_count','error_class','error_message','completed_at')) {
    if ([string]$Before.run.$field -cne [string]$After.run.$field) { throw "R120_LIFECYCLE_FIELD_CHANGED:$field" }
  }
  if ([string]$Before.session_status -cne [string]$After.session_status) { throw 'R120_SESSION_STATUS_CHANGED' }
}

if (-not (Test-Path -LiteralPath $PSCommandPath -PathType Leaf) -or (Sha $PSCommandPath) -cne $RunnerSha256) { throw 'R120_PROBE_RUNNER_HASH_REJECTED' }
if ((Test-Path -LiteralPath $resultPath) -or (Test-Path -LiteralPath $errorPath)) { throw 'R120_PROBE_RESULT_EXISTS' }

try {
  $loopback = Wait-Exact 'http://127.0.0.1:8088/api/healthz'
  $public = Wait-Exact 'https://evomind.zhjjq.tech/api/healthz'
  $managedBefore = Assert-Managed
  $runtimeTreeBefore = Tree (Join-Path $root 'bundle\runtime\evomind_runtime')
  if ($runtimeTreeBefore.file_count -ne 32 -or $runtimeTreeBefore.sha256 -cne $expectedRuntimeTree) { throw 'R120_RUNTIME_TREE_REJECTED' }
  $runBefore = Read-RunGate
  $goalBefore = Read-GoalGate
  $tool = Read-ToolGate
  $idempotencyKey = 'g21-r120-hpc-asset-probe-' + [guid]::NewGuid().ToString('N')
  $probeAttempted = $true
  $toolCalls = 1
  $outcome = Post-Api "/v1/sessions/$fixedRun/tools" ([ordered]@{
    tool_name=$toolName
    arguments=[ordered]@{ scope=$scope }
    idempotency_key=$idempotencyKey
  })
  if ([string]$outcome.status -notin @('completed','failed')) { throw 'R120_PROBE_NOT_SETTLED' }
  $result = $outcome.result
  $callId = [string]$result.tool_call_id
  if ($callId -cnotmatch '^call_[A-Za-z0-9_-]{8,128}$') { throw 'R120_PROBE_CALL_ID_REJECTED' }
  $runAfter = Read-RunGate
  Assert-LifecyclePreserved $runBefore $runAfter
  $eventRows = (Get-Api "/v1/runs/$fixedRun/events?after=$($runBefore.run.last_event_seq)").events
  # The public run-events projection exposes stable aliases (tool_started and
  # tool_completed) and intentionally strips the raw tool-call arguments.
  # Accept the projection, while retaining compatibility with an older raw
  # event response if a managed runtime returns one.
  $started = @($eventRows | Where-Object {
    (([string]$_.event_type -ceq 'tool_started' -or [string]$_.event_type -ceq 'tool.started') -and
      (([string]$_.payload.tool_call_id -ceq $callId) -or ([string]$_.payload.id -ceq $callId)) -and
      (([string]$_.payload.tool -ceq $toolName) -or ([string]$_.payload.tool_name -ceq $toolName)))
  })
  $settledType = if ([string]$outcome.status -ceq 'completed') { 'tool_completed' } else { 'tool_failed' }
  $settledAliases = if ([string]$outcome.status -ceq 'completed') { @('tool_completed','tool.completed') } else { @('tool_completed','tool.failed') }
  $settled = @($eventRows | Where-Object {
    ([string]$_.event_type -in $settledAliases) -and [string]$_.payload.tool_call_id -ceq $callId
  })
  $approvals = @($eventRows | Where-Object { [string]$_.event_type -ceq 'approval.requested' })
  if ($started.Count -ne 1 -or $settled.Count -ne 1 -or $approvals.Count -ne 0) { throw 'R120_PROBE_EVENT_CLOSURE_REJECTED' }
  $managedAfter = Assert-Managed
  $runtimeTreeAfter = Tree (Join-Path $root 'bundle\runtime\evomind_runtime')
  if ($runtimeTreeAfter.sha256 -cne $runtimeTreeBefore.sha256 -or $managedAfter.state_sha256 -cne $managedBefore.state_sha256) { throw 'R120_LOCAL_RUNTIME_OR_PROCESS_STATE_CHANGED' }
  $payload = [ordered]@{
    schema=$schema
    status='completed'
    run_id=$fixedRun
    allocation=$fixedAllocation
    preserve_terminal=$true
    run_before=$runBefore.run
    run_after=$runAfter.run
    goal=$goalBefore
    tool=$tool
    outcome=$outcome
    event_closure=[ordered]@{ call_id=$callId; started=1; settled=1; settled_event=$settledType; approvals_requested=0 }
    runtime_tree=$runtimeTreeAfter
    managed_before=$managedBefore
    managed_after=$managedAfter
    loopback=$loopback
    public=$public
    direct_tool_calls=1
    run_action_calls=0
    cancel_calls=0
    retries=0
    hpc_remote_writes=[int]($result.content.remote_writes)
    signals_sent=[int]($result.content.signals_sent)
    other_processes_modified=[bool]($result.content.other_processes_modified)
    training_started=[bool]($result.content.training_started)
    worker_control=[int]($result.content.worker_control)
    secrets_printed=$false
    production_staging_writes=1
    production_staging_cleanup_policy='retain_for_audit_no_automatic_delete'
  }
  Write-AtomicJson $resultPath $payload
  [ordered]@{schema=$schema;status='completed';transaction_id=$TransactionId;result_filename=[IO.Path]::GetFileName($resultPath);preserve_terminal=$true}|ConvertTo-Json -Compress
  exit 0
} catch {
  $payload = [ordered]@{
    schema=$schema
    status='failed_closed'
    run_id=$fixedRun
    allocation=$fixedAllocation
    error=(Safe-Error $_.Exception.Message)
    preserve_terminal='unverified'
    probe_attempted=$probeAttempted
    direct_tool_calls=$toolCalls
    run_action_calls=0
    cancel_calls=0
    retries=0
    hpc_remote_writes=0
    signals_sent=0
    other_processes_modified=$false
    training_started=$false
    worker_control=0
    secrets_printed=$false
    production_staging_writes=1
    production_staging_cleanup_policy='retain_for_audit_no_automatic_delete'
  }
  if (-not (Test-Path -LiteralPath $errorPath)) { Write-AtomicJson $errorPath $payload }
  [ordered]@{schema=$schema;status='failed_closed';transaction_id=$TransactionId;error_filename=[IO.Path]::GetFileName($errorPath)}|ConvertTo-Json -Compress
  exit 1
}
