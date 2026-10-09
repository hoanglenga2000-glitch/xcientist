param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [Parameter(Mandatory)][string]$RemoteRoot,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$RuntimeSha256,
  [Parameter(Mandatory)][string]$ExpectedBuildId,
  [ValidatePattern('^(?:|run_[a-f0-9]{32})$')][string]$ExpectedRunId=''
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$root='C:\ProgramData\EvoMind'
$expectedRemote=[IO.Path]::GetFullPath("C:\SecureInput\runtime-package-$TransactionId")
if([IO.Path]::GetFullPath($RemoteRoot)-cne$expectedRemote){throw 'RUNTIME_PACKAGE_REMOTE_ROOT_REJECTED'}
$archive=Join-Path $expectedRemote 'runtime.zip'
$stage=Join-Path $expectedRemote 'runtime-stage'
$resultPath=Join-Path $expectedRemote "transaction-$TransactionId-result.json"
$errorPath=Join-Path $expectedRemote "transaction-$TransactionId-error.json"
$action='C:\SecureInput\Invoke-ServiceAccountAction.ps1'
$seal=Join-Path $root 'bundle\scripts\Seal-Release.ps1'
$bundleRuntime=Join-Path $root 'bundle\runtime'
$target=Join-Path $bundleRuntime 'evomind_runtime'

function Sha([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
function Tree([string]$Path){
  $full=[IO.Path]::GetFullPath($Path).TrimEnd('\')
  $rows=@(Get-ChildItem -LiteralPath $full -Recurse -File -Force|Where-Object{$_.Name-notmatch'\.(pyc|pyo)$'-and$_.FullName-notmatch'[\\/]__pycache__[\\/]'}|Sort-Object FullName|ForEach-Object{"$($_.FullName.Substring($full.Length).TrimStart('\').Replace('\','/'))|$([long]$_.Length)|$(Sha $_.FullName)"})
  $hash=[Security.Cryptography.SHA256]::Create()
  try{$digest=([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes(($rows-join"`n"))))).Replace('-','').ToLowerInvariant()}finally{$hash.Dispose()}
  [ordered]@{file_count=$rows.Count;sha256=$digest}
}
function Other-RuntimeTree {
  $full=[IO.Path]::GetFullPath($bundleRuntime).TrimEnd('\')
  $rows=@(Get-ChildItem -LiteralPath $full -Recurse -File -Force|ForEach-Object{
    $relative=$_.FullName.Substring($full.Length).TrimStart('\').Replace('\','/')
    if($relative-notmatch'^(?:evomind_runtime(?:/|\.)|__pycache__/)' -and $relative-notmatch'\.(pyc|pyo)$'){"$relative|$([long]$_.Length)|$(Sha $_.FullName)"}
  }|Where-Object{$_}|Sort-Object)
  $hash=[Security.Cryptography.SHA256]::Create()
  try{$digest=([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes(($rows-join"`n"))))).Replace('-','').ToLowerInvariant()}finally{$hash.Dispose()}
  [ordered]@{file_count=$rows.Count;sha256=$digest}
}
function Read-Runtime {
  $config=Get-Content -LiteralPath "$root\config\node-config.json" -Raw -Encoding UTF8|ConvertFrom-Json
  $runtime=Get-Content -LiteralPath (Join-Path ([string]$config.web_runtime_root) 'runtime-build-manifest.json') -Raw -Encoding UTF8|ConvertFrom-Json
  [ordered]@{config=$config;runtime=$runtime}
}
function Assert-Managed([string]$ExpectedBuild){
  $task=$null;$info=$null
  foreach($attempt in 1..10){try{$task=Get-ScheduledTask -TaskPath '\' -TaskName 'EvoMind-Cloud-Node' -ErrorAction Stop;$info=Get-ScheduledTaskInfo -TaskPath '\' -TaskName 'EvoMind-Cloud-Node' -ErrorAction Stop;break}catch{if($attempt-eq10){throw};Start-Sleep -Milliseconds 500}}
  if([string]$task.State-ne'Ready'-or[int64]$info.LastTaskResult-ne0-or[string]$task.Actions[0].Arguments-notmatch'-Action Start(?:\s|$)'){throw 'RUNTIME_PACKAGE_MANAGED_TASK_NOT_READY'}
  $state=Get-Content -LiteralPath "$root\state\node-processes.json" -Raw -Encoding UTF8|ConvertFrom-Json
  foreach($role in @('tenant_enrollment_queue','llm_gateway','python_runtime','web')){
    $record=@($state.records|Where-Object{[string]$_.role-eq$role})
    if($record.Count-ne1-or-not(Get-Process -Id ([int]$record[0].pid) -ErrorAction SilentlyContinue)){throw "RUNTIME_PACKAGE_ROLE_INVALID:$role"}
    $owner=Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$record[0].pid)"|Invoke-CimMethod -MethodName GetOwner
    if([string]$owner.User-ne'EvoMindSvc'){throw "RUNTIME_PACKAGE_OWNER_INVALID:$role"}
  }
  $current=Read-Runtime;if([string]$current.runtime.build_id-ne$ExpectedBuild){throw 'RUNTIME_PACKAGE_BUILD_MISMATCH'}
  [ordered]@{build_id=[string]$current.runtime.build_id;roles=4;task_ready=$true}
}
function Read-HpcState {
  $current=Read-Runtime;$config=$current.config
  $tenantRoot=Join-Path $root 'byoa\tenants\tenant_ba0ef9d3767f2fb385b856e0';$bindingPath=Join-Path $tenantRoot 'hpc-binding.json'
  $binding=Get-Content -LiteralPath $bindingPath -Raw -Encoding UTF8|ConvertFrom-Json
  if([string]$binding.schema-ne'evomind.tenant_hpc_binding.v1'-or[string]$binding.tenant_id-ne'tenant_ba0ef9d3767f2fb385b856e0'-or[string]$binding.state-notin@('provisioning','active','frozen','retired')-or[int]$binding.job_id-lt1-or[int]$binding.allocation_generation-lt1-or[string]$binding.credential_profile-ne("tenant_ba0ef9d3767f2fb385b856e0_job$([int]$binding.job_id)_g$([int]$binding.allocation_generation)")){throw 'RUNTIME_PACKAGE_HPC_BINDING_REJECTED'}
  $profileInfo=Get-CimInstance Win32_UserProfile -ErrorAction SilentlyContinue|Where-Object{[IO.Path]::GetFileName(([string]$_.LocalPath).TrimEnd('\'))-ieq[string]$config.dedicated_user}|Select-Object -First 1
  $serviceHome=if($profileInfo){[string]$profileInfo.LocalPath}else{"C:\Users\$([string]$config.dedicated_user)"}
  $profileRoot=Join-Path $serviceHome "AppData\Roaming\ResearchAgentWorkstation\profiles\$([string]$binding.credential_profile)"
  if(-not(Test-Path -LiteralPath $profileRoot -PathType Container)){throw 'RUNTIME_PACKAGE_HPC_PROFILE_MISSING'}
  [ordered]@{binding_sha256=Sha $bindingPath;byoa_tree=Tree (Join-Path $root 'byoa');profile_root=$profileRoot;profile_tree=Tree $profileRoot;job_id=[int]$binding.job_id;generation=[int]$binding.allocation_generation;state=[string]$binding.state;credential_profile=[string]$binding.credential_profile}
}
function Assert-HpcSame([object]$Before,[object]$After,[bool]$IncludeByoaTree){
  if($Before.binding_sha256-ne$After.binding_sha256-or$Before.profile_root-ne$After.profile_root-or$Before.profile_tree.sha256-ne$After.profile_tree.sha256-or$Before.profile_tree.file_count-ne$After.profile_tree.file_count-or$Before.job_id-ne$After.job_id-or$Before.generation-ne$After.generation-or$Before.state-ne$After.state-or$Before.credential_profile-ne$After.credential_profile){throw 'RUNTIME_PACKAGE_HPC_CHANGED'}
  if($IncludeByoaTree-and($Before.byoa_tree.sha256-ne$After.byoa_tree.sha256-or$Before.byoa_tree.file_count-ne$After.byoa_tree.file_count)){throw 'RUNTIME_PACKAGE_BYOA_CHANGED'}
}
function Wait-Exact([string]$Uri,[string]$Build){
  $samples=@();$consecutive=0;$deadline=[DateTime]::UtcNow.AddSeconds(180)
  do{$row=[ordered]@{status_code=0;build_id=$null;exact=$false;elapsed_ms=$null};$sw=[Diagnostics.Stopwatch]::StartNew();try{$response=Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 12;$payload=$response.Content|ConvertFrom-Json;$row.status_code=[int]$response.StatusCode;$row.build_id=[string]$payload.build_id;$row.exact=($row.status_code-eq200-and[bool]$payload.ok-and[string]$payload.status-eq'ready'-and$row.build_id-eq$Build-and@($payload.failures).Count-eq0)}catch{}finally{$sw.Stop();$row.elapsed_ms=[math]::Round($sw.Elapsed.TotalMilliseconds,3)};if($row.exact){$consecutive++}else{$consecutive=0};$samples+=[pscustomobject]$row;if($consecutive-ge3){return [ordered]@{status='ready';samples=@($samples|Select-Object -Last 3)}};Start-Sleep -Seconds 2}while([DateTime]::UtcNow-lt$deadline)
  throw "RUNTIME_PACKAGE_EXACT_HEALTH_TIMEOUT:$Uri"
}
function Stop-VerifiedResidualListeners {
  $listeners=@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue|Where-Object{$_.LocalPort-in@(65068,8765,8088,7890)})
  foreach($listener in $listeners){
    $port=[int]$listener.LocalPort;$pidValue=[int]$listener.OwningProcess
    $process=Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction Stop
    $owner=$process|Invoke-CimMethod -MethodName GetOwner
    if([string]$owner.User-ne'EvoMindSvc'){throw "RUNTIME_PACKAGE_RESIDUAL_OWNER_REJECTED:$port"}
    $name=[string]$process.Name;$command=[string]$process.CommandLine;$path=[string]$process.ExecutablePath
    $valid=switch($port){
      65068{$name-eq'cliproxy-7.2.128.exe'-and$path-match'ProgramData\\EvoMind\\bin\\cliproxy-7\.2\.128\.exe$'}
      8765{$name-in@('python.exe','pythonw.exe')-and$command-match'run_python_runtime\.py.+--port\s+8765'}
      8088{$name-eq'node.exe'-and$command-match'(?:^|\s)server\.js(?:\s|$)'}
      7890{$name-in@('python.exe','pythonw.exe')-and$command-match'hpc_socks_bridge\.py.+--listen-port\s+7890'}
      default{$false}
    }
    if(-not$valid){throw "RUNTIME_PACKAGE_RESIDUAL_IDENTITY_REJECTED:$port"}
  }
  foreach($listener in $listeners){Stop-Process -Id ([int]$listener.OwningProcess) -Force -ErrorAction Stop}
  $deadline=[DateTime]::UtcNow.AddSeconds(20)
  do{Start-Sleep -Milliseconds 250;$remaining=@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue|Where-Object{$_.LocalPort-in@(65068,8765,8088,7890)})}while($remaining.Count-and[DateTime]::UtcNow-lt$deadline)
  if($remaining.Count){throw 'RUNTIME_PACKAGE_RESIDUAL_LISTENER_REMAINS'}
  $script:residualListenerStops+=$listeners.Count
}
function Stop-ManagedNode {
  try{$result=@(& $action -Action Stop -TimeoutMinutes 12)|Out-String|ConvertFrom-Json}
  catch{
    if([string]$_.Exception.Message-notmatch'POST_STOP_LISTENER_REMAINS'){throw}
    Stop-VerifiedResidualListeners
    return [ordered]@{status='completed';action='Stop';residual_listener_recovery=$true}
  }
  if([string]$result.status-ne'completed'-or[string]$result.action-ne'Stop'){throw 'RUNTIME_PACKAGE_STOP_FAILED'}
  $result
}
function Read-Hotfix {
  $manifest=Get-Content -LiteralPath (Join-Path $stage 'runtime-hotfix-manifest.json') -Raw -Encoding UTF8|ConvertFrom-Json
  if([string]$manifest.schema-ne'evomind.super_agent_runtime_hotfix.v1'-or[int]$manifest.file_count-notin@(14,15,17,30,32)-or[string]$manifest.target-ne'bundle/runtime/evomind_runtime'){throw 'RUNTIME_PACKAGE_MANIFEST_REJECTED'}
  $declared=@($manifest.files.path);$actual=@(Get-ChildItem -LiteralPath $stage -Recurse -File -Force|ForEach-Object{$_.FullName.Substring($stage.Length+1).Replace('\','/')})
  if($actual.Count-ne([int]$manifest.file_count+1)-or@($actual|Where-Object{$_-ne'runtime-hotfix-manifest.json'-and$declared-notcontains$_}).Count){throw 'RUNTIME_PACKAGE_FILESET_REJECTED'}
  foreach($entry in $manifest.files){$path=Join-Path $stage ([string]$entry.path).Replace('/','\');if(-not(Test-Path -LiteralPath $path -PathType Leaf)-or[long](Get-Item -LiteralPath $path).Length-ne[long]$entry.bytes-or(Sha $path)-ne[string]$entry.sha256){throw "RUNTIME_PACKAGE_HASH_REJECTED:$($entry.path)"}}
  $manifest
}
function Get-RuntimeHeaders {
  $current=Read-Runtime
  $tokenPath=Join-Path ([string]$current.config.data_root) 'workspace\runtime\runtime.token'
  if(-not(Test-Path -LiteralPath $tokenPath -PathType Leaf)){throw 'RUNTIME_PACKAGE_RUNTIME_TOKEN_MISSING'}
  @{Authorization='Bearer '+(Get-Content -LiteralPath $tokenPath -Raw).Trim()}
}
function Assert-TargetRunQuiescent {
  if(-not$ExpectedRunId){return $null}
  $snapshot=Invoke-RestMethod -Uri "http://127.0.0.1:8765/v1/runs/$ExpectedRunId" -Headers (Get-RuntimeHeaders) -TimeoutSec 20
  $pendingApprovalCount=@($snapshot.approvals|Where-Object{[string]$_.status-eq'pending'}).Count
  if([string]$snapshot.id-ne$ExpectedRunId-or[string]$snapshot.status-notin@('blocked','failed','cancelled')-or@($snapshot.active_tool_calls).Count-ne0-or$pendingApprovalCount-ne0){throw 'RUNTIME_PACKAGE_TARGET_RUN_NOT_QUIESCENT'}
  [ordered]@{run_id=$ExpectedRunId;status=[string]$snapshot.status;active_tool_count=0;pending_approval_count=0;artifact_count=@($snapshot.artifacts).Count}
}

if((Test-Path $resultPath)-or(Test-Path $errorPath)){throw 'RUNTIME_PACKAGE_RESULT_EXISTS'}
$mutex=[Threading.Mutex]::new($false,'Global\EvoMind-Byoa-V12-Deployment')
$owned=$false;$backup=$null;$mutated=$false;$serviceStopped=$false;$beforeHpc=$null;$rollbackError=$null;$script:residualListenerStops=0
try{
  $owned=$mutex.WaitOne([TimeSpan]::FromSeconds(10));if(-not$owned){throw 'RUNTIME_PACKAGE_DEPLOYMENT_LOCKED'}
  foreach($path in @($archive,$action,$seal,$target)){if(-not(Test-Path -LiteralPath $path)){throw "RUNTIME_PACKAGE_INPUT_MISSING:$path"}}
  if((Sha $archive)-ne$RuntimeSha256){throw 'RUNTIME_PACKAGE_ARCHIVE_SHA_MISMATCH'}
  if(Test-Path -LiteralPath $stage){throw 'RUNTIME_PACKAGE_STAGE_EXISTS'}
  Expand-Archive -LiteralPath $archive -DestinationPath $stage
  $manifest=Read-Hotfix
  $candidate=Join-Path $bundleRuntime ("evomind_runtime.$TransactionId.stage")
  if(Test-Path -LiteralPath $candidate){throw 'RUNTIME_PACKAGE_CANDIDATE_EXISTS'}
  Copy-Item -LiteralPath (Join-Path $stage 'evomind_runtime') -Destination $candidate -Recurse
  foreach($entry in $manifest.files){$path=Join-Path $candidate ([IO.Path]::GetFileName([string]$entry.path));if((Sha $path)-ne[string]$entry.sha256){throw 'RUNTIME_PACKAGE_CANDIDATE_HASH_REJECTED'}}
  $newTree=Tree $candidate
  $beforeRuntime=Assert-Managed $ExpectedBuildId
  $targetRunPreflight=Assert-TargetRunQuiescent
  $stop=Stop-ManagedNode
  $serviceStopped=$true
  foreach($port in 65068,8765,8088,7890){if(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue){throw "RUNTIME_PACKAGE_LISTENER_REMAINS:$port"}}
  $beforeHpc=Read-HpcState;$otherBefore=Other-RuntimeTree;$oldTree=Tree $target
  $backup=Join-Path "$root\backups" ('pre-runtime-package-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
  New-Item -ItemType Directory -Path (Join-Path $backup 'original') -Force|Out-Null
  [IO.File]::WriteAllText((Join-Path $backup 'backup-manifest.json'),(( [ordered]@{schema='evomind.runtime_package_backup.v1';transaction_id=$TransactionId;build_id=$ExpectedBuildId;runtime_sha256=$RuntimeSha256;source_manifest_sha256=[string]$manifest.source_manifest_sha256;old_tree=$oldTree;new_tree=$newTree;other_runtime_tree=$otherBefore;hpc=$beforeHpc}|ConvertTo-Json -Depth 8)+"`n"),[Text.UTF8Encoding]::new($false))
  $mutated=$true
  Move-Item -LiteralPath $target -Destination (Join-Path $backup 'original\evomind_runtime')
  Move-Item -LiteralPath $candidate -Destination $target
  $installedTree=Tree $target
  if($installedTree.sha256-ne$newTree.sha256-or$installedTree.file_count-ne$newTree.file_count){throw 'RUNTIME_PACKAGE_INSTALLED_TREE_REJECTED'}
  $otherAfter=Other-RuntimeTree
  if($otherAfter.sha256-ne$otherBefore.sha256-or$otherAfter.file_count-ne$otherBefore.file_count){throw 'RUNTIME_PACKAGE_OTHER_RUNTIME_CHANGED'}
  & $seal|Out-Null;Assert-HpcSame $beforeHpc (Read-HpcState) $true
  $start=@(& $action -Action Start -TimeoutMinutes 12)|Out-String|ConvertFrom-Json
  if([string]$start.status-ne'completed'-or[string]$start.action-ne'Start'){throw 'RUNTIME_PACKAGE_START_FAILED'}
  $serviceStopped=$false
  $loopback=Wait-Exact 'http://127.0.0.1:8088/api/healthz' $ExpectedBuildId;$public=Wait-Exact 'https://evomind.zhjjq.tech/api/healthz' $ExpectedBuildId
  $afterRuntime=Assert-Managed $ExpectedBuildId;$afterHpc=Read-HpcState;Assert-HpcSame $beforeHpc $afterHpc $false
  $requiredEntries=[ordered]@{}
  foreach($name in @('assistant_runs.py','competition_data.py','store.py','super_agent_runtime.py','tools.py')){
    $entry=@($manifest.files|Where-Object{[string]$_.path-eq"evomind_runtime/$name"})
    if($entry.Count-ne1-or(Sha (Join-Path $target $name))-ne[string]$entry[0].sha256){throw "RUNTIME_PACKAGE_FILE_SHA_REJECTED:$name"}
    $requiredEntries[$name]=[string]$entry[0].sha256
  }
  $targetRunAfter=Assert-TargetRunQuiescent
  if(($targetRunPreflight|ConvertTo-Json -Compress)-cne($targetRunAfter|ConvertTo-Json -Compress)){throw 'RUNTIME_PACKAGE_TARGET_RUN_CHANGED'}
  $payload=[ordered]@{schema='evomind.runtime_package_deployment_result.v1';status='installed_and_verified';transaction_id=$TransactionId;build_id=$ExpectedBuildId;runtime_sha256=$RuntimeSha256;source_manifest_sha256=[string]$manifest.source_manifest_sha256;backup_root=$backup;old_runtime_tree=$oldTree;installed_runtime_tree=$installedTree;assistant_runs_sha256=$requiredEntries['assistant_runs.py'];competition_data_sha256=$requiredEntries['competition_data.py'];store_sha256=$requiredEntries['store.py'];super_agent_runtime_sha256=$requiredEntries['super_agent_runtime.py'];tools_sha256=$requiredEntries['tools.py'];target_run_preflight=$targetRunPreflight;target_run_after=$targetRunAfter;target_run_unchanged=$true;loopback=$loopback;public=$public;runtime=$afterRuntime;hpc_binding_unchanged=$true;byoa_stop_window_unchanged=$true;profile_unchanged=$true;job_id=$afterHpc.job_id;allocation_generation=$afterHpc.generation;binding_state=$afterHpc.state;residual_listener_stops=[int]$script:residualListenerStops;secrets_printed=$false;hpc_accessed=$false;gpu_touched=$false}
  [IO.File]::WriteAllText($resultPath,(($payload|ConvertTo-Json -Depth 10)+"`n"),[Text.UTF8Encoding]::new($false));$payload|ConvertTo-Json -Depth 10 -Compress
}catch{
  $failure=$_;$rollback='not_required'
  if($backup-and$mutated){try{try{$rs=Stop-ManagedNode;$serviceStopped=$true}catch{$listeners=@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue|Where-Object{$_.LocalPort-in@(65068,8765,8088,7890)});if($listeners.Count){throw};$serviceStopped=$true};if(Test-Path -LiteralPath $target){Remove-Item -LiteralPath $target -Recurse -Force};Move-Item -LiteralPath (Join-Path $backup 'original\evomind_runtime') -Destination $target;& $seal|Out-Null;Assert-HpcSame $beforeHpc (Read-HpcState) $true;$restart=@(& $action -Action Start -TimeoutMinutes 12)|Out-String|ConvertFrom-Json;if([string]$restart.status-ne'completed'){throw 'RUNTIME_PACKAGE_ROLLBACK_START_FAILED'};$serviceStopped=$false;Wait-Exact 'http://127.0.0.1:8088/api/healthz' $ExpectedBuildId|Out-Null;Assert-HpcSame $beforeHpc (Read-HpcState) $false;$rollback='restored_and_verified'}catch{$rollback='failed';$rollbackError=[string]$_.Exception.Message}}
  elseif($serviceStopped){try{$restart=@(& $action -Action Start -TimeoutMinutes 12)|Out-String|ConvertFrom-Json;if([string]$restart.status-eq'completed'){$serviceStopped=$false;$rollback='start_only_recovered'}}catch{}}
  $errorPayload=[ordered]@{schema='evomind.runtime_package_deployment_error.v1';status='failed';transaction_id=$TransactionId;build_id=$ExpectedBuildId;message=[string]$failure.Exception.Message;rollback=$rollback;rollback_error=if($rollbackError){$rollbackError}else{$null};secrets_printed=$false;hpc_accessed=$false;gpu_touched=$false}
  [IO.File]::WriteAllText($errorPath,(($errorPayload|ConvertTo-Json -Depth 6)+"`n"),[Text.UTF8Encoding]::new($false));throw
}finally{if($owned){try{[void]$mutex.ReleaseMutex()}catch{}};$mutex.Dispose()}
