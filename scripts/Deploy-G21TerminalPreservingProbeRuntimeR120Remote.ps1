param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [Parameter(Mandatory)][string]$RemoteRoot,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$ApprovalSha256
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = 'C:\ProgramData\EvoMind'
$expectedRemote = [IO.Path]::GetFullPath("C:\SecureInput\r120-terminal-preserving-probe-runtime-$TransactionId")
if ([IO.Path]::GetFullPath($RemoteRoot) -cne $expectedRemote) { throw 'R120_REMOTE_ROOT_REJECTED' }
$inputRoot = Join-Path $expectedRemote 'envelope-expanded'
$approvalPath = Join-Path $inputRoot 'approval.json'
$archive = Join-Path $inputRoot 'runtime.zip'
$sourceManifestPath = Join-Path $inputRoot 'source-manifest.json'
$buildResultPath = Join-Path $inputRoot 'build-result.json'
$receiptPath = Join-Path $inputRoot 'test-receipt.json'
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
$baselineZipSha = '9368c7962632fb976e7a8ba8ede0d3108989e96c00c4dd75bca154008d438206'
$baselineTreeSha = '906868d22a841f591fb84c7a505b75acee3c8df58f179e83d3d4585601d03975'
$baselineManifestSha = '12f80d02d6fd87e413c57bc606d09a27f90c3edac3b4b1d7462826374666b9ac'
$candidateTreeSha = '5998ea21823c121c23a421bb145d8e2266bb79ab59eccc91a16122c3626b61ce'
$goalSpecSha = 'fbb9939dff28c355c2798bf32fce104845337064f42f9c280f2b6906cccf4036'
$goalBoardSha = '6b908173083889ac21f8afea52528b4b5e35d2ff862a880a6065d65b169f443c'

function Sha([string]$Path) {
  $stream=[IO.File]::OpenRead($Path);$hash=[Security.Cryptography.SHA256]::Create()
  try { ([BitConverter]::ToString($hash.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
  finally { $stream.Dispose();$hash.Dispose() }
}

function Safe-Error([object]$Value) {
  $text=[string]$Value
  $text=$text-replace'(?i)(?:Bearer|Basic)\s+\S+','[credential redacted]'
  $text=$text-replace'(?i)\b(?:password|passwd|pwd|credential|token|secret|api[_ -]?key|authorization|bearer|密码|口令|凭据|账号|登录)\s*[:=]\s*\S+','[credential redacted]'
  $text=$text-replace'[\r\n]+','_'
  if($text.Length-gt500){$text=$text.Substring(0,500)}
  $text
}

function Write-AtomicJson([string]$Path,[object]$Value){
  $temporary="$Path.$PID.tmp"
  [IO.File]::WriteAllText($temporary,(($Value|ConvertTo-Json -Depth 30)+"`n"),[Text.UTF8Encoding]::new($false))
  Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Read-Object([string]$Path,[string]$Code){
  if(-not(Test-Path -LiteralPath $Path -PathType Leaf)){throw "$Code`:MISSING"}
  try{$value=Get-Content -LiteralPath $Path -Raw -Encoding UTF8|ConvertFrom-Json}catch{throw "$Code`:INVALID_JSON"}
  if($null-eq$value-or$value-is[array]){throw "$Code`:OBJECT_REQUIRED"}
  $value
}

function Assert-NoReparse([string]$Path){
  $rootItem=Get-Item -LiteralPath $Path -Force
  if($rootItem.Attributes-band[IO.FileAttributes]::ReparsePoint){throw 'R120_REPARSE_REJECTED'}
  $bad=@(Get-ChildItem -LiteralPath $Path -Recurse -Force|Where-Object{$_.Attributes-band[IO.FileAttributes]::ReparsePoint})
  if($bad.Count){throw 'R120_REPARSE_REJECTED'}
}

function Tree([string]$Path){
  $full=[IO.Path]::GetFullPath($Path).TrimEnd('\')
  $files=@(Get-ChildItem -LiteralPath $full -Recurse -File -Force|Where-Object{$_.Name-notmatch'\.(pyc|pyo)$'-and$_.FullName-notmatch'[\\/]__pycache__[\\/]'}|Sort-Object FullName)
  $rows=@($files|ForEach-Object{"$($_.FullName.Substring($full.Length).TrimStart('\').Replace('\','/'))|$([long]$_.Length)|$(Sha $_.FullName)"})
  $hash=[Security.Cryptography.SHA256]::Create()
  try{$digest=([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes(($rows-join"`n"))))).Replace('-','').ToLowerInvariant()}finally{$hash.Dispose()}
  [ordered]@{file_count=$files.Count;sha256=$digest}
}

function Read-Runtime {
  $config=Read-Object (Join-Path $root 'config\node-config.json') 'R120_NODE_CONFIG'
  $manifest=Read-Object (Join-Path ([string]$config.web_runtime_root) 'runtime-build-manifest.json') 'R120_WEB_RUNTIME_MANIFEST'
  [ordered]@{config=$config;manifest=$manifest}
}

function Headers {
  $current=Read-Runtime
  $tokenPath=Join-Path ([string]$current.config.data_root) 'workspace\runtime\runtime.token'
  if(-not(Test-Path -LiteralPath $tokenPath -PathType Leaf)){throw 'R120_RUNTIME_TOKEN_MISSING'}
  $token=(Get-Content -LiteralPath $tokenPath -Raw -Encoding ASCII).Trim()
  if($token-cnotmatch'^[A-Za-z0-9_-]{32,128}$'){throw 'R120_RUNTIME_TOKEN_INVALID'}
  @{Authorization='Bearer '+$token}
}

function Wait-Exact([string]$Uri){
  $samples=@();$consecutive=0;$deadline=[DateTime]::UtcNow.AddSeconds(180)
  do{
    $row=[ordered]@{status_code=0;build_id=$null;exact=$false;elapsed_ms=$null};$timer=[Diagnostics.Stopwatch]::StartNew()
    try{$response=Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 12;$payload=$response.Content|ConvertFrom-Json;$row.status_code=[int]$response.StatusCode;$row.build_id=[string]$payload.build_id;$row.exact=($row.status_code-eq200-and[bool]$payload.ok-and[string]$payload.status-ceq'ready'-and$row.build_id-ceq$fixedWebBuild-and@($payload.failures).Count-eq0)}catch{}finally{$timer.Stop();$row.elapsed_ms=[math]::Round($timer.Elapsed.TotalMilliseconds,3)}
    if($row.exact){$consecutive++}else{$consecutive=0};$samples+=[pscustomobject]$row
    if($consecutive-ge3){return [ordered]@{status='ready';samples=@($samples|Select-Object -Last 3)}}
    Start-Sleep -Seconds 2
  }while([DateTime]::UtcNow-lt$deadline)
  throw "R120_HEALTH_TIMEOUT:$Uri"
}

function Assert-Managed {
  $statePath=Join-Path $root 'state\node-processes.json';$state=Read-Object $statePath 'R120_NODE_PROCESS_STATE';$roles=[ordered]@{}
  foreach($role in @('tenant_enrollment_queue','llm_gateway','python_runtime','web')){
    $records=@($state.records|Where-Object{[string]$_.role-ceq$role});if($records.Count-ne1){throw "R120_ROLE_COUNT:$role"}
    $pidValue=[int]$records[0].pid;$process=Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction Stop;$owner=$process|Invoke-CimMethod -MethodName GetOwner
    if([string]$owner.User-cne'EvoMindSvc'){throw "R120_ROLE_OWNER:$role"};$roles[$role]=[ordered]@{pid=$pidValue;owner='EvoMindSvc';name=[string]$process.Name}
  }
  $listeners=[ordered]@{};foreach($port in @(65068,8765,8088,7890)){$rows=@(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue);if($rows.Count-ne1){throw "R120_LISTENER_COUNT:$port"};$process=Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$rows[0].OwningProcess)";$owner=$process|Invoke-CimMethod -MethodName GetOwner;if([string]$owner.User-cne'EvoMindSvc'){throw "R120_LISTENER_OWNER:$port"};$listeners[[string]$port]=[ordered]@{pid=[int]$rows[0].OwningProcess;owner='EvoMindSvc';name=[string]$process.Name}}
  if([string](Read-Runtime).manifest.build_id-cne$fixedWebBuild){throw 'R120_WEB_BUILD_MISMATCH'}
  [ordered]@{roles=$roles;listeners=$listeners;state_sha256=Sha $statePath}
}

function Assert-Fresh([object]$Before,[object]$After){
  foreach($role in @('tenant_enrollment_queue','llm_gateway','python_runtime','web')){if([int]$Before.roles[$role].pid-eq[int]$After.roles[$role].pid){throw "R120_ROLE_NOT_FRESH:$role"}}
  foreach($port in @('65068','8765','8088','7890')){if([int]$Before.listeners[$port].pid-eq[int]$After.listeners[$port].pid){throw "R120_LISTENER_NOT_FRESH:$port"}}
}

function Read-RunGate {
  $snapshot=Invoke-RestMethod -Uri "http://127.0.0.1:8765/v1/runs/$fixedRun" -Headers (Headers) -TimeoutSec 20
  $pending=@($snapshot.approvals|Where-Object{[string]$_.status-ceq'pending'}).Count;$active=@($snapshot.active_tool_calls).Count
  if([string]$snapshot.id-cne$fixedRun-or[string]$snapshot.status-cne'cancelled'-or-not[bool]$snapshot.terminal-or$active-ne0-or$pending-ne0){throw 'R120_RUN_GATE_REJECTED'}
  [ordered]@{run_id=$fixedRun;status='cancelled';terminal=$true;effective_active_count=0;pending_approval_count=0;last_event_seq=if($snapshot.PSObject.Properties.Name-contains'last_event_seq'){[int64]$snapshot.last_event_seq}else{[int64]0}}
}

function Read-GoalGate {
  $goal=Invoke-RestMethod -Uri 'http://127.0.0.1:8765/v1/goals/goal_g21_five_competition' -Headers (Headers) -TimeoutSec 20
  if([string]$goal.run_id-cne$fixedRun-or[string]$goal.allocation-cne$fixedAllocation-or[string]$goal.status-cne'blocked'-or[string]$goal.spec_sha256-cne$goalSpecSha-or[string]$goal.board_sha256-cne$goalBoardSha){throw 'R120_GOAL_GATE_REJECTED'}
  $rows=@($goal.board.competitions);$complete=@($rows|Where-Object{[string]$_.goal_status-ceq'VERIFIED_COMPLETE'});if($rows.Count-ne5-or$complete.Count-ne1-or[string]$complete[0].competition-cne'open_polymer'){throw 'R120_GOAL_BOARD_REJECTED'}
  [ordered]@{id=[string]$goal.id;status=[string]$goal.status;spec_sha256=[string]$goal.spec_sha256;board_sha256=[string]$goal.board_sha256;human_baseline_sha256=[string]$goal.human_baseline_sha256;updated_at=[string]$goal.updated_at}
}

function Read-ToolSchema {
  $tools=Invoke-RestMethod -Uri 'http://127.0.0.1:8765/v1/tools' -Headers (Headers) -TimeoutSec 20;$rows=@($tools.tools|Where-Object{[string]$_.name-ceq'hpc_asset_probe'})
  if($rows.Count-ne1-or-not[bool]$rows[0].read_only-or[string]$rows[0].capability-cne'hpc.read'){throw 'R120_TOOL_SCHEMA_REJECTED'}
  [ordered]@{name='hpc_asset_probe';read_only=$true;capability='hpc.read'}
}

function Read-LocalBoundary {
  $tenantRoot=Join-Path $root 'byoa\tenants\tenant_ba0ef9d3767f2fb385b856e0';$binding=Join-Path $tenantRoot 'hpc-binding.json'
  if(-not(Test-Path -LiteralPath $binding -PathType Leaf)){throw 'R120_HPC_BINDING_MISSING'}
  [ordered]@{binding_sha256=Sha $binding;byoa_tree=Tree (Join-Path $root 'byoa');other_runtime=Tree $bundleRuntime}
}

function Assert-BoundarySame([object]$Before,[object]$After){if($Before.binding_sha256-cne$After.binding_sha256-or$Before.byoa_tree.sha256-cne$After.byoa_tree.sha256){throw 'R120_LOCAL_HPC_BOUNDARY_CHANGED'}}

function Invoke-Managed([ValidateSet('Stop','Start')][string]$Name){if(-not(Test-Path -LiteralPath $action -PathType Leaf)){throw 'R120_SERVICE_ACTION_MISSING'};$result=@(& $action -Action $Name -TimeoutMinutes 12)|Out-String|ConvertFrom-Json;if([string]$result.status-cne'completed'-or[string]$result.action-cne$Name){throw "R120_MANAGED_$($Name.ToUpperInvariant())_FAILED"};$result}
function Assert-Stopped {if(@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue|Where-Object{$_.LocalPort-in@(65068,8765,8088,7890)}).Count){throw 'R120_LISTENERS_REMAIN'}}

function Backup-Database([string]$BackupRoot,[object]$Config){
  $database=Join-Path ([string]$Config.data_root) 'workspace\runtime\runtime.sqlite3';if(-not(Test-Path -LiteralPath $database -PathType Leaf)){throw 'R120_DATABASE_MISSING'}
  $destination=Join-Path $BackupRoot 'database';New-Item -ItemType Directory -Path $destination|Out-Null;$rows=@()
  foreach($suffix in @('','-wal','-shm')){$source="$database$suffix";$exists=Test-Path -LiteralPath $source -PathType Leaf;$row=[ordered]@{suffix=$suffix;existed=$exists;bytes=0;sha256=''};if($exists){$name=if($suffix){"runtime.sqlite3$suffix"}else{'runtime.sqlite3'};Copy-Item -LiteralPath $source -Destination (Join-Path $destination $name);$row.bytes=[long](Get-Item -LiteralPath $source).Length;$row.sha256=Sha $source};$rows+=[pscustomobject]$row}
  [ordered]@{path=$database;files=$rows}
}

function Restore-Database([string]$BackupRoot,[object]$Backup){foreach($suffix in @('','-wal','-shm')){$path="$($Backup.path)$suffix";if(Test-Path -LiteralPath $path){Remove-Item -LiteralPath $path -Force}};foreach($row in @($Backup.files)){if([bool]$row.existed){$name=if([string]$row.suffix){"runtime.sqlite3$([string]$row.suffix)"}else{'runtime.sqlite3'};Copy-Item -LiteralPath (Join-Path (Join-Path $BackupRoot 'database') $name) -Destination "$($Backup.path)$([string]$row.suffix)"}}}

function Read-Inputs {
  foreach($path in @($approvalPath,$archive,$sourceManifestPath,$buildResultPath,$receiptPath,$action,$seal,$target)){if(-not(Test-Path -LiteralPath $path)){throw 'R120_INPUT_MISSING'}}
  if((Sha $approvalPath)-cne$ApprovalSha256){throw 'R120_APPROVAL_HASH_REJECTED'}
  $approval=Read-Object $approvalPath 'R120_APPROVAL'
  if([string]$approval.schema-cne'evomind.g21_terminal_preserving_probe_runtime_deployment_approval.r120.v1'-or[string]$approval.status-cne'approved_for_single_runtime_transaction'-or-not[bool]$approval.one_time-or[string]$approval.deployment_id-cne$TransactionId-or[string]$approval.run_id-cne$fixedRun-or[string]$approval.allocation-cne$fixedAllocation-or[string]$approval.preflight.current_runtime_tree_sha256-cne$baselineTreeSha-or[string]$approval.baseline.zip_sha256-cne$baselineZipSha-or[string]$approval.baseline.manifest_sha256-cne$baselineManifestSha-or[string]$approval.candidate.installed_runtime_tree_sha256-cne$candidateTreeSha-or[string]$approval.candidate.changed_files[0]-cne'evomind_runtime/http_server.py'-or[int]$approval.candidate.changed_file_count-ne1-or[int]$approval.candidate.unchanged_file_count-ne31-or[string]$approval.deployer.remote_runner_filename-cne[IO.Path]::GetFileName($PSCommandPath)-or(Sha $PSCommandPath)-cne[string]$approval.deployer.remote_runner_sha256){throw 'R120_APPROVAL_CONTRACT_REJECTED'}
  try{$created=[DateTimeOffset]::Parse([string]$approval.created_at);$expires=[DateTimeOffset]::Parse([string]$approval.expires_at)}catch{throw 'R120_APPROVAL_TIME_REJECTED'};$now=[DateTimeOffset]::UtcNow;if($expires-le$created-or$now-lt$created.AddMinutes(-5)-or$now-ge$expires-or($expires-$created).TotalMinutes-gt60){throw 'R120_APPROVAL_EXPIRED'}
  foreach($row in @([pscustomobject]@{path=$archive;sha=[string]$approval.candidate.sha256},[pscustomobject]@{path=$sourceManifestPath;sha=[string]$approval.candidate.source_manifest_sha256},[pscustomobject]@{path=$buildResultPath;sha=[string]$approval.candidate.build_result_sha256},[pscustomobject]@{path=$receiptPath;sha=[string]$approval.verification.test_receipt_sha256})){if((Sha $row.path)-cne$row.sha){throw 'R120_STAGED_HASH_REJECTED'}}
  [ordered]@{approval=$approval}
}

function Read-Candidate([object]$Approval){
  if(Test-Path -LiteralPath $stage){throw 'R120_STAGE_EXISTS'};Expand-Archive -LiteralPath $archive -DestinationPath $stage;Assert-NoReparse $stage
  $manifestPath=Join-Path $stage 'terminal-preserving-probe-runtime-manifest.json';$manifest=Read-Object $manifestPath 'R120_MANIFEST'
  if((Sha $manifestPath)-cne(Sha $sourceManifestPath)-or[string]$manifest.schema-cne'evomind.g21_terminal_preserving_probe_runtime.r120.v1'-or[string]$manifest.baseline_zip_sha256-cne$baselineZipSha-or[string]$manifest.baseline_runtime_tree_sha256-cne$baselineTreeSha-or[string]$manifest.baseline_manifest_sha256-cne$baselineManifestSha-or[string]$manifest.candidate_runtime_tree_sha256-cne[string]$Approval.candidate.installed_runtime_tree_sha256-or[string]$manifest.changed_files[0]-cne'evomind_runtime/http_server.py'-or[int]$manifest.changed_file_count-ne1-or[int]$manifest.unchanged_file_count-ne31-or[bool]$manifest.production_deployable-or[bool]$manifest.production_deployed){throw 'R120_MANIFEST_CONTRACT_REJECTED'}
  $declared=@{};foreach($entry in @($manifest.files)){$relative=[string]$entry.path;if($relative-cnotmatch'^evomind_runtime/[A-Za-z0-9_]+\.py$'-or$declared.ContainsKey($relative)){throw 'R120_MANIFEST_ENTRY_REJECTED'};$declared[$relative]=$entry}
  $actual=@(Get-ChildItem -LiteralPath $stage -Recurse -File -Force|ForEach-Object{$_.FullName.Substring($stage.Length+1).Replace('\','/')}|Where-Object{$_-ne'terminal-preserving-probe-runtime-manifest.json'})
  if($declared.Count-ne32-or$actual.Count-ne32-or@($actual|Where-Object{-not$declared.ContainsKey($_)}).Count){throw 'R120_CANDIDATE_FILESET_REJECTED'}
  foreach($relative in $declared.Keys){$path=Join-Path $stage $relative.Replace('/','\');$entry=$declared[$relative];if([long](Get-Item -LiteralPath $path).Length-ne[long]$entry.bytes-or(Sha $path)-cne[string]$entry.sha256){throw 'R120_CANDIDATE_HASH_REJECTED'}}
  $candidate=Join-Path $stage 'evomind_runtime';$tree=Tree $candidate;if($tree.file_count-ne32-or$tree.sha256-cne[string]$Approval.candidate.installed_runtime_tree_sha256){throw 'R120_CANDIDATE_TREE_REJECTED'}
  [ordered]@{path=$candidate;tree=$tree;manifest=$manifest}
}

if((Test-Path -LiteralPath $resultPath)-or(Test-Path -LiteralPath $errorPath)){throw 'R120_RESULT_EXISTS'}
$mutex=[Threading.Mutex]::new($false,'Global\EvoMind-Byoa-V12-Deployment');$owned=$false;$backup=$null;$candidate=$null;$bundleCandidate=$null;$targetBackedUp=$false;$serviceStopped=$false;$databaseBackup=$null;$boundaryBefore=$null;$webBefore=$null;$managedBefore=$null;$rollbackError=$null
try{
  $owned=$mutex.WaitOne([TimeSpan]::FromSeconds(10));if(-not$owned){throw 'R120_DEPLOYMENT_LOCKED'}
  $approval=(Read-Inputs).approval;$candidate=Read-Candidate $approval
  $preLoopback=Wait-Exact 'http://127.0.0.1:8088/api/healthz';$prePublic=Wait-Exact 'https://evomind.zhjjq.tech/api/healthz';$managedBefore=Assert-Managed;$runBefore=Read-RunGate;$goalBefore=Read-GoalGate;$toolBefore=Read-ToolSchema
  $currentRuntime=Tree $target;if($currentRuntime.file_count-ne32-or$currentRuntime.sha256-cne$baselineTreeSha){throw 'R120_CURRENT_RUNTIME_NOT_R119'}
  $webRoot=[string](Read-Runtime).config.web_runtime_root;$webBefore=Tree $webRoot;$boundaryBefore=Read-LocalBoundary
  Invoke-Managed Stop|Out-Null;$serviceStopped=$true;Assert-Stopped
  $backup=Join-Path "$root\backups" ('pre-r120-terminal-preserving-probe-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')+'-'+$TransactionId);New-Item -ItemType Directory -Path (Join-Path $backup 'original')|Out-Null
  $databaseBackup=Backup-Database $backup (Read-Runtime).config;Copy-Item -LiteralPath $approvalPath -Destination (Join-Path $backup 'approval.json')
  Write-AtomicJson (Join-Path $backup 'backup-manifest.json') ([ordered]@{schema='evomind.g21_terminal_preserving_probe_runtime_backup.r120.v1';transaction_id=$TransactionId;approval_sha256=$ApprovalSha256;old_runtime_tree=$currentRuntime;new_runtime_tree=$candidate.tree;database=$databaseBackup;local_boundary=$boundaryBefore})
  $bundleCandidate=Join-Path $bundleRuntime "evomind_runtime.$TransactionId.stage";Move-Item -LiteralPath $candidate.path -Destination $bundleCandidate
  Move-Item -LiteralPath $target -Destination (Join-Path $backup 'original\evomind_runtime');$targetBackedUp=$true;Move-Item -LiteralPath $bundleCandidate -Destination $target
  $installed=Tree $target;if($installed.file_count-ne32-or$installed.sha256-cne[string]$approval.candidate.installed_runtime_tree_sha256){throw 'R120_INSTALLED_TREE_REJECTED'}
  & $seal|Out-Null;Assert-BoundarySame $boundaryBefore (Read-LocalBoundary);Invoke-Managed Start|Out-Null;$serviceStopped=$false
  Wait-Exact 'http://127.0.0.1:8088/api/healthz'|Out-Null;Assert-Managed|Out-Null;Read-ToolSchema|Out-Null
  Invoke-Managed Stop|Out-Null;$serviceStopped=$true;Assert-Stopped;Invoke-Managed Start|Out-Null;$serviceStopped=$false
  $loopback=Wait-Exact 'http://127.0.0.1:8088/api/healthz';$public=Wait-Exact 'https://evomind.zhjjq.tech/api/healthz';$managedAfter=Assert-Managed;Assert-Fresh $managedBefore $managedAfter
  $runAfter=Read-RunGate;if($runAfter.status-cne$runBefore.status-or$runAfter.last_event_seq-ne$runBefore.last_event_seq){throw 'R120_RUN_CHANGED'};$goalAfter=Read-GoalGate;if(($goalAfter|ConvertTo-Json -Compress)-cne($goalBefore|ConvertTo-Json -Compress)){throw 'R120_GOAL_CHANGED'};$toolAfter=Read-ToolSchema
  Assert-BoundarySame $boundaryBefore (Read-LocalBoundary);$webAfter=Tree $webRoot;if($webAfter.sha256-cne$webBefore.sha256-or$webAfter.file_count-ne$webBefore.file_count){throw 'R120_WEB_CHANGED'};$finalTree=Tree $target;if($finalTree.sha256-cne$installed.sha256){throw 'R120_FINAL_TREE_REJECTED'}
  $payload=[ordered]@{schema='evomind.g21_terminal_preserving_probe_runtime_deployment_result.r120.v1';status='installed_and_verified';transaction_id=$TransactionId;approval_sha256=$ApprovalSha256;web_build_id=$fixedWebBuild;pre_loopback=$preLoopback;pre_public=$prePublic;loopback=$loopback;public=$public;managed_before=$managedBefore;managed_after=$managedAfter;fresh_role_pids=$true;fresh_listener_pids=$true;run_before=$runBefore;run_after=$runAfter;goal_before=$goalBefore;goal=$goalAfter;goal_unchanged=$true;tool_before=$toolBefore;tool_after=$toolAfter;old_runtime_tree=$currentRuntime;installed_runtime_tree=$finalTree;backup_root=$backup;database_backup=$databaseBackup;web_tree_unchanged=$true;hpc_binding_unchanged=$true;byoa_unchanged=$true;profile_unchanged=$true;hpc_accessed=$false;gpu_touched=$false;hpc_remote_writes=0;direct_signals_sent=0;other_processes_modified=$false;training_started=$false;hpc_tool_calls=0;goal_put_calls=0;managed_service_actions=@('Stop','Start','Stop','Start')}
  Write-AtomicJson $resultPath $payload;$payload|ConvertTo-Json -Depth 25 -Compress
}catch{
  $failure=$_;$rollback='not_required'
  if($targetBackedUp){try{if(-not$serviceStopped){Invoke-Managed Stop|Out-Null;$serviceStopped=$true;Assert-Stopped};if(Test-Path -LiteralPath $target){Remove-Item -LiteralPath $target -Recurse -Force};Move-Item -LiteralPath (Join-Path $backup 'original\evomind_runtime') -Destination $target;Restore-Database $backup $databaseBackup;& $seal|Out-Null;Assert-BoundarySame $boundaryBefore (Read-LocalBoundary);Invoke-Managed Start|Out-Null;$serviceStopped=$false;$rollbackLoopback=Wait-Exact 'http://127.0.0.1:8088/api/healthz';$rollbackPublic=Wait-Exact 'https://evomind.zhjjq.tech/api/healthz';$restored=Tree $target;if($restored.sha256-cne$baselineTreeSha){throw 'R120_ROLLBACK_TREE_REJECTED'};$rollback='restored_and_verified'}catch{$rollback='failed';$rollbackError=Safe-Error $_.Exception.Message}}
  elseif($serviceStopped){try{Invoke-Managed Start|Out-Null;$serviceStopped=$false;Wait-Exact 'http://127.0.0.1:8088/api/healthz'|Out-Null;$rollback='start_only_recovered'}catch{$rollback='failed';$rollbackError=Safe-Error $_.Exception.Message}}
  Write-AtomicJson $errorPath ([ordered]@{schema='evomind.g21_terminal_preserving_probe_runtime_deployment_error.r120.v1';status='failed';transaction_id=$TransactionId;approval_sha256=$ApprovalSha256;message=Safe-Error $failure.Exception.Message;rollback=$rollback;rollback_error=$rollbackError;hpc_accessed=$false;gpu_touched=$false;hpc_remote_writes=0;direct_signals_sent=0;other_processes_modified=$false})
  throw
}finally{if($bundleCandidate-and(Test-Path -LiteralPath $bundleCandidate)){Remove-Item -LiteralPath $bundleCandidate -Recurse -Force -ErrorAction SilentlyContinue};if($owned){try{[void]$mutex.ReleaseMutex()}catch{}};$mutex.Dispose()}
