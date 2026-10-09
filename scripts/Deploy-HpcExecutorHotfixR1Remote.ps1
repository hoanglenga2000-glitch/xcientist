param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [Parameter(Mandatory)][string]$RemoteRoot,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$HotfixSha256,
  [Parameter(Mandatory)][string]$ExpectedBuildId
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$root='C:\ProgramData\EvoMind'
$expectedRemote="C:\SecureInput\hpc-executor-hotfix-$TransactionId"
if([IO.Path]::GetFullPath($RemoteRoot)-cne$expectedRemote){throw 'HPC_EXECUTOR_HOTFIX_REMOTE_ROOT_REJECTED'}
$archive=Join-Path $expectedRemote 'hotfix.zip'
$stage=Join-Path $expectedRemote 'hotfix-stage'
$resultPath=Join-Path $expectedRemote "transaction-$TransactionId-result.json"
$errorPath=Join-Path $expectedRemote "transaction-$TransactionId-error.json"
$action='C:\SecureInput\Invoke-ServiceAccountAction.ps1'
$seal=Join-Path $root 'bundle\scripts\Seal-Release.ps1'
$toolsRelative='evomind_runtime/tools.py'
$hpcRelative='research_os/hpc_runtime.py'
$toolsTarget=Join-Path $root 'bundle\runtime\evomind_runtime\tools.py'
$expectedOldTools='35afc02f899eb6f2a01d046e74f9ce5bc00c76ccc8c042305b49deaafae1726b'
$expectedOldHpc='f99c0fc52d97be222df4d923227905a4ea74dbdb76dd5dc6af22e3f320b116ce'

function Sha([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
function Write-Utf8([string]$Path,[object]$Value){[IO.File]::WriteAllText($Path,(($Value|ConvertTo-Json -Depth 12)+"`n"),[Text.UTF8Encoding]::new($false))}
function Tree-Except([string]$Path,[string[]]$ExcludedNames){
  $full=[IO.Path]::GetFullPath($Path).TrimEnd('\')
  $rows=@(Get-ChildItem -LiteralPath $full -Recurse -File -Force|ForEach-Object{
    $relative=$_.FullName.Substring($full.Length).TrimStart('\').Replace('\','/')
    if($ExcludedNames-notcontains$relative-and$relative-notmatch'(^|/)__pycache__/'-and$relative-notmatch'\.(pyc|pyo)$'){"$relative|$([long]$_.Length)|$(Sha $_.FullName)"}
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
  if([string]$task.State-ne'Ready'-or[int64]$info.LastTaskResult-ne0-or[string]$task.Actions[0].Arguments-notmatch'-Action Start(?:\s|$)'){throw 'HPC_EXECUTOR_HOTFIX_MANAGED_TASK_NOT_READY'}
  $state=Get-Content -LiteralPath "$root\state\node-processes.json" -Raw -Encoding UTF8|ConvertFrom-Json
  foreach($role in @('tenant_enrollment_queue','llm_gateway','python_runtime','web')){
    $record=@($state.records|Where-Object{[string]$_.role-eq$role})
    if($record.Count-ne1-or-not(Get-Process -Id ([int]$record[0].pid) -ErrorAction SilentlyContinue)){throw "HPC_EXECUTOR_HOTFIX_ROLE_INVALID:$role"}
    $owner=Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$record[0].pid)"|Invoke-CimMethod -MethodName GetOwner
    if([string]$owner.User-ne'EvoMindSvc'){throw "HPC_EXECUTOR_HOTFIX_OWNER_INVALID:$role"}
  }
  $current=Read-Runtime;if([string]$current.runtime.build_id-ne$ExpectedBuild){throw 'HPC_EXECUTOR_HOTFIX_BUILD_MISMATCH'}
  [ordered]@{build_id=[string]$current.runtime.build_id;roles=4;task_ready=$true;state=$state}
}
function Resolve-ActiveRelease([object]$Managed){
  $runtimeRecord=@($Managed.state.records|Where-Object{[string]$_.role-eq'python_runtime'})[0]
  $process=Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$runtimeRecord.pid)" -ErrorAction Stop
  $chain=@()
  foreach($index in 0..4){
    if(-not$process){break}
    $chain+=$process
    if([string]$process.ExecutablePath-match'\\.venv\\Scripts\\pythonw?\.exe$'){break}
    if([int]$process.ParentProcessId-le0){break}
    $process=Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$process.ParentProcessId)" -ErrorAction SilentlyContinue
  }
  $venvProcess=@($chain|Where-Object{[string]$_.ExecutablePath-match'\\.venv\\Scripts\\pythonw?\.exe$'}|Select-Object -First 1)
  if($venvProcess.Count-ne1){throw 'HPC_EXECUTOR_HOTFIX_ACTIVE_RELEASE_PROCESS_NOT_FOUND'}
  $scripts=Split-Path -Parent ([string]$venvProcess[0].ExecutablePath)
  $venv=Split-Path -Parent $scripts
  $release=[IO.Path]::GetFullPath((Split-Path -Parent $venv)).TrimEnd('\')
  if(-not$release.StartsWith('C:\EvoMind\releases\',[StringComparison]::OrdinalIgnoreCase)){throw 'HPC_EXECUTOR_HOTFIX_RELEASE_ROOT_REJECTED'}
  $python=Join-Path $release '.venv\Scripts\python.exe'
  $targets=@(
    [ordered]@{name='site';path=Join-Path $release '.venv\Lib\site-packages\research_os\hpc_runtime.py';package_root=Join-Path $release '.venv\Lib\site-packages\research_os'}
  )
  $trackedSources=@(
    [ordered]@{name='src';path=Join-Path $release 'src\research_os\hpc_runtime.py'},
    [ordered]@{name='build';path=Join-Path $release 'build\lib\research_os\hpc_runtime.py'}
  )
  foreach($target in $targets){if(-not(Test-Path -LiteralPath $target.path -PathType Leaf)){throw "HPC_EXECUTOR_HOTFIX_RELEASE_TARGET_MISSING:$($target.name)"}}
  foreach($target in $trackedSources){if(-not(Test-Path -LiteralPath $target.path -PathType Leaf)){throw "HPC_EXECUTOR_HOTFIX_TRACKED_SOURCE_MISSING:$($target.name)"}}
  if(-not(Test-Path -LiteralPath $python -PathType Leaf)){throw 'HPC_EXECUTOR_HOTFIX_RELEASE_PYTHON_MISSING'}
  [ordered]@{root=$release;python=$python;targets=$targets;tracked_sources=$trackedSources;process_chain=@($chain|ForEach-Object{[ordered]@{pid=[int]$_.ProcessId;parent_pid=[int]$_.ParentProcessId;executable=[string]$_.ExecutablePath}})}
}
function Read-HpcState {
  $current=Read-Runtime;$config=$current.config
  $tenantRoot=Join-Path $root 'byoa\tenants\tenant_ba0ef9d3767f2fb385b856e0';$bindingPath=Join-Path $tenantRoot 'hpc-binding.json'
  $binding=Get-Content -LiteralPath $bindingPath -Raw -Encoding UTF8|ConvertFrom-Json
  if([string]$binding.schema-ne'evomind.tenant_hpc_binding.v1'-or[string]$binding.tenant_id-ne'tenant_ba0ef9d3767f2fb385b856e0'-or[string]$binding.state-ne'active'-or[int]$binding.job_id-ne91699-or[int]$binding.allocation_generation-ne15-or[string]$binding.credential_profile-ne'tenant_ba0ef9d3767f2fb385b856e0_job91699_g15'){throw 'HPC_EXECUTOR_HOTFIX_BINDING_REJECTED'}
  $profileInfo=Get-CimInstance Win32_UserProfile -ErrorAction SilentlyContinue|Where-Object{[IO.Path]::GetFileName(([string]$_.LocalPath).TrimEnd('\'))-ieq[string]$config.dedicated_user}|Select-Object -First 1
  $serviceHome=if($profileInfo){[string]$profileInfo.LocalPath}else{"C:\Users\$([string]$config.dedicated_user)"}
  $profileRoot=Join-Path $serviceHome "AppData\Roaming\ResearchAgentWorkstation\profiles\$([string]$binding.credential_profile)"
  if(-not(Test-Path -LiteralPath $profileRoot -PathType Container)){throw 'HPC_EXECUTOR_HOTFIX_PROFILE_MISSING'}
  [ordered]@{binding_sha256=Sha $bindingPath;byoa_tree=Tree-Except (Join-Path $root 'byoa') @();profile_root=$profileRoot;profile_tree=Tree-Except $profileRoot @();job_id=[int]$binding.job_id;generation=[int]$binding.allocation_generation;state=[string]$binding.state;credential_profile=[string]$binding.credential_profile}
}
function Assert-HpcSame([object]$Before,[object]$After){
  if($Before.binding_sha256-ne$After.binding_sha256-or$Before.byoa_tree.sha256-ne$After.byoa_tree.sha256-or$Before.byoa_tree.file_count-ne$After.byoa_tree.file_count-or$Before.profile_root-ne$After.profile_root-or$Before.profile_tree.sha256-ne$After.profile_tree.sha256-or$Before.profile_tree.file_count-ne$After.profile_tree.file_count-or$Before.job_id-ne$After.job_id-or$Before.generation-ne$After.generation-or$Before.state-ne$After.state-or$Before.credential_profile-ne$After.credential_profile){throw 'HPC_EXECUTOR_HOTFIX_HPC_STATE_CHANGED'}
}
function Wait-Exact([string]$Uri,[string]$Build){
  $samples=@();$consecutive=0;$deadline=[DateTime]::UtcNow.AddSeconds(180)
  do{$row=[ordered]@{status_code=0;build_id=$null;exact=$false;elapsed_ms=$null};$sw=[Diagnostics.Stopwatch]::StartNew();try{$response=Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 12;$payload=$response.Content|ConvertFrom-Json;$row.status_code=[int]$response.StatusCode;$row.build_id=[string]$payload.build_id;$row.exact=($row.status_code-eq200-and[bool]$payload.ok-and[string]$payload.status-eq'ready'-and$row.build_id-eq$Build-and@($payload.failures).Count-eq0)}catch{}finally{$sw.Stop();$row.elapsed_ms=[math]::Round($sw.Elapsed.TotalMilliseconds,3)};if($row.exact){$consecutive++}else{$consecutive=0};$samples+=[pscustomobject]$row;if($consecutive-ge3){return [ordered]@{status='ready';samples=@($samples|Select-Object -Last 3)}};Start-Sleep -Seconds 2}while([DateTime]::UtcNow-lt$deadline)
  throw "HPC_EXECUTOR_HOTFIX_EXACT_HEALTH_TIMEOUT:$Uri"
}
function Read-Hotfix {
  $manifestPath=Join-Path $stage 'hpc-executor-hotfix-manifest.json'
  $manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8|ConvertFrom-Json
  if([string]$manifest.schema-ne'evomind.super_agent_hpc_executor_hotfix.v1'-or[string]$manifest.build_id-ne$ExpectedBuildId-or[int]$manifest.file_count-ne2-or@($manifest.files).Count-ne2){throw 'HPC_EXECUTOR_HOTFIX_MANIFEST_REJECTED'}
  $declared=@($manifest.files.path);$required=@($toolsRelative,$hpcRelative)
  if(@($declared|Where-Object{$required-notcontains$_}).Count-or@($required|Where-Object{$declared-notcontains$_}).Count){throw 'HPC_EXECUTOR_HOTFIX_DECLARED_FILESET_REJECTED'}
  $actual=@(Get-ChildItem -LiteralPath $stage -Recurse -File -Force|ForEach-Object{$_.FullName.Substring($stage.Length+1).Replace('\','/')})
  if($actual.Count-ne3-or@($actual|Where-Object{$_-ne'hpc-executor-hotfix-manifest.json'-and$declared-notcontains$_}).Count){throw 'HPC_EXECUTOR_HOTFIX_ACTUAL_FILESET_REJECTED'}
  foreach($entry in $manifest.files){$path=Join-Path $stage ([string]$entry.path).Replace('/','\');if(-not(Test-Path -LiteralPath $path -PathType Leaf)-or[long](Get-Item -LiteralPath $path).Length-ne[long]$entry.bytes-or(Sha $path)-ne[string]$entry.sha256){throw "HPC_EXECUTOR_HOTFIX_HASH_REJECTED:$($entry.path)"}}
  $manifest
}

if((Test-Path $resultPath)-or(Test-Path $errorPath)){throw 'HPC_EXECUTOR_HOTFIX_RESULT_EXISTS'}
$mutex=[Threading.Mutex]::new($false,'Global\EvoMind-Byoa-V12-Deployment')
$owned=$false;$backup=$null;$mutated=$false;$serviceStopped=$false;$beforeHpc=$null;$releaseInfo=$null;$rollbackError=$null
try{
  $owned=$mutex.WaitOne([TimeSpan]::FromSeconds(10));if(-not$owned){throw 'HPC_EXECUTOR_HOTFIX_DEPLOYMENT_LOCKED'}
  foreach($path in @($archive,$action,$seal,$toolsTarget)){if(-not(Test-Path -LiteralPath $path)){throw "HPC_EXECUTOR_HOTFIX_INPUT_MISSING:$path"}}
  if((Sha $archive)-ne$HotfixSha256){throw 'HPC_EXECUTOR_HOTFIX_ARCHIVE_SHA_MISMATCH'}
  if(Test-Path -LiteralPath $stage){throw 'HPC_EXECUTOR_HOTFIX_STAGE_EXISTS'}
  Expand-Archive -LiteralPath $archive -DestinationPath $stage
  $manifest=Read-Hotfix
  $toolsEntry=@($manifest.files|Where-Object{$_.path-eq$toolsRelative})[0]
  $hpcEntry=@($manifest.files|Where-Object{$_.path-eq$hpcRelative})[0]
  $beforeRuntime=Assert-Managed $ExpectedBuildId
  $releaseInfo=Resolve-ActiveRelease $beforeRuntime
  if((Sha $toolsTarget)-ne$expectedOldTools){throw 'HPC_EXECUTOR_HOTFIX_OLD_TOOLS_SHA_REJECTED'}
  foreach($target in $releaseInfo.targets){if((Sha $target.path)-ne$expectedOldHpc){throw "HPC_EXECUTOR_HOTFIX_OLD_HPC_SHA_REJECTED:$($target.name)"}}
  foreach($target in $releaseInfo.tracked_sources){if((Sha $target.path)-ne$expectedOldHpc){throw "HPC_EXECUTOR_HOTFIX_TRACKED_SOURCE_SHA_REJECTED:$($target.name)"}}
  $stop=@(& $action -Action Stop -TimeoutMinutes 12)|Out-String|ConvertFrom-Json
  if([string]$stop.status-ne'completed'-or[string]$stop.action-ne'Stop'){throw 'HPC_EXECUTOR_HOTFIX_STOP_FAILED'}
  $serviceStopped=$true
  foreach($port in 65068,8765,8088,7890){if(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue){throw "HPC_EXECUTOR_HOTFIX_LISTENER_REMAINS:$port"}}
  $beforeHpc=Read-HpcState
  $treesBefore=[ordered]@{bundle=Tree-Except (Split-Path -Parent $toolsTarget) @('tools.py')}
  foreach($target in $releaseInfo.targets){$treesBefore[$target.name]=Tree-Except $target.package_root @('hpc_runtime.py')}
  $backup=Join-Path "$root\backups" ('pre-hpc-executor-hotfix-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
  New-Item -ItemType Directory -Path (Join-Path $backup 'original') -Force|Out-Null
  Copy-Item -LiteralPath $toolsTarget -Destination (Join-Path $backup 'original\bundle-tools.py')
  foreach($target in $releaseInfo.targets){Copy-Item -LiteralPath $target.path -Destination (Join-Path $backup "original\$($target.name)-hpc_runtime.py")}
  Write-Utf8 (Join-Path $backup 'backup-manifest.json') ([ordered]@{schema='evomind.hpc_executor_hotfix_backup.v1';transaction_id=$TransactionId;build_id=$ExpectedBuildId;hotfix_sha256=$HotfixSha256;source_manifest_sha256=[string]$manifest.source_manifest_sha256;old_tools_sha256=$expectedOldTools;old_hpc_sha256=$expectedOldHpc;new_tools_sha256=[string]$toolsEntry.sha256;new_hpc_sha256=[string]$hpcEntry.sha256;release_root=$releaseInfo.root;other_trees=$treesBefore;hpc=$beforeHpc})
  $mutated=$true
  Move-Item -LiteralPath $toolsTarget -Destination (Join-Path $backup 'original\moved-bundle-tools.py')
  Copy-Item -LiteralPath (Join-Path $stage 'evomind_runtime\tools.py') -Destination $toolsTarget
  foreach($target in $releaseInfo.targets){
    Move-Item -LiteralPath $target.path -Destination (Join-Path $backup "original\moved-$($target.name)-hpc_runtime.py")
    Copy-Item -LiteralPath (Join-Path $stage 'research_os\hpc_runtime.py') -Destination $target.path
  }
  if((Sha $toolsTarget)-ne[string]$toolsEntry.sha256){throw 'HPC_EXECUTOR_HOTFIX_INSTALLED_TOOLS_SHA_REJECTED'}
  foreach($target in $releaseInfo.targets){if((Sha $target.path)-ne[string]$hpcEntry.sha256){throw "HPC_EXECUTOR_HOTFIX_INSTALLED_HPC_SHA_REJECTED:$($target.name)"}}
  $treesAfter=[ordered]@{bundle=Tree-Except (Split-Path -Parent $toolsTarget) @('tools.py')}
  foreach($target in $releaseInfo.targets){$treesAfter[$target.name]=Tree-Except $target.package_root @('hpc_runtime.py')}
  foreach($name in $treesBefore.Keys){if($treesBefore[$name].sha256-ne$treesAfter[$name].sha256-or$treesBefore[$name].file_count-ne$treesAfter[$name].file_count){throw "HPC_EXECUTOR_HOTFIX_OTHER_TREE_CHANGED:$name"}}
  & $seal|Out-Null;Assert-HpcSame $beforeHpc (Read-HpcState)
  $start=@(& $action -Action Start -TimeoutMinutes 12)|Out-String|ConvertFrom-Json
  if([string]$start.status-ne'completed'-or[string]$start.action-ne'Start'){throw 'HPC_EXECUTOR_HOTFIX_START_FAILED'}
  $serviceStopped=$false
  $loopback=Wait-Exact 'http://127.0.0.1:8088/api/healthz' $ExpectedBuildId;$public=Wait-Exact 'https://evomind.zhjjq.tech/api/healthz' $ExpectedBuildId
  $afterRuntime=Assert-Managed $ExpectedBuildId;$afterHpc=Read-HpcState;Assert-HpcSame $beforeHpc $afterHpc
  if((Sha $toolsTarget)-ne[string]$toolsEntry.sha256){throw 'HPC_EXECUTOR_HOTFIX_POSTSTART_TOOLS_SHA_REJECTED'}
  foreach($target in $releaseInfo.targets){if((Sha $target.path)-ne[string]$hpcEntry.sha256){throw "HPC_EXECUTOR_HOTFIX_POSTSTART_HPC_SHA_REJECTED:$($target.name)"}}
  foreach($target in $releaseInfo.tracked_sources){if((Sha $target.path)-ne$expectedOldHpc){throw "HPC_EXECUTOR_HOTFIX_POSTSTART_TRACKED_SOURCE_CHANGED:$($target.name)"}}
  $importOutput=& $releaseInfo.python -c "import hashlib,json,research_os.hpc_runtime as h;p=h.__file__;print(json.dumps({'path':p,'sha256':hashlib.sha256(open(p,'rb').read()).hexdigest()}))"
  if($LASTEXITCODE-ne0){throw 'HPC_EXECUTOR_HOTFIX_IMPORT_FAILED'}
  $importIdentity=$importOutput|Out-String|ConvertFrom-Json
  if([string]$importIdentity.sha256-ne[string]$hpcEntry.sha256-or[IO.Path]::GetFullPath([string]$importIdentity.path)-cne[IO.Path]::GetFullPath([string]$releaseInfo.targets[0].path)){throw 'HPC_EXECUTOR_HOTFIX_IMPORT_IDENTITY_REJECTED'}
  $runtimeSummary=[ordered]@{build_id=[string]$afterRuntime.build_id;roles=[int]$afterRuntime.roles;task_ready=[bool]$afterRuntime.task_ready}
  $payload=[ordered]@{schema='evomind.hpc_executor_hotfix_deployment_result.v1';status='installed_and_verified';transaction_id=$TransactionId;build_id=$ExpectedBuildId;hotfix_sha256=$HotfixSha256;source_manifest_sha256=[string]$manifest.source_manifest_sha256;backup_root=$backup;release_root=$releaseInfo.root;old_tools_sha256=$expectedOldTools;old_hpc_sha256=$expectedOldHpc;tools_sha256=[string]$toolsEntry.sha256;hpc_runtime_sha256=[string]$hpcEntry.sha256;updated_target_count=2;tracked_release_source_unchanged=$true;import_identity=$importIdentity;loopback=$loopback;public=$public;runtime=$runtimeSummary;hpc_binding_unchanged=$true;byoa_tree_unchanged=$true;profile_unchanged=$true;job_id=$afterHpc.job_id;allocation_generation=$afterHpc.generation;binding_state=$afterHpc.state;secrets_printed=$false;hpc_accessed=$false;gpu_touched=$false;training_started=$false;kaggle_submissions=0}
  Write-Utf8 $resultPath $payload;$payload|ConvertTo-Json -Depth 12 -Compress
}catch{
  $failure=$_;$rollback='not_required'
  if($backup-and$mutated){try{
    try{$rs=@(& $action -Action Stop -TimeoutMinutes 12)|Out-String|ConvertFrom-Json;if([string]$rs.status-ne'completed'){throw 'HPC_EXECUTOR_HOTFIX_ROLLBACK_STOP_FAILED'};$serviceStopped=$true}catch{$listeners=@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue|Where-Object{$_.LocalPort-in@(65068,8765,8088,7890)});if($listeners.Count){throw};$serviceStopped=$true}
    if(Test-Path -LiteralPath $toolsTarget){Remove-Item -LiteralPath $toolsTarget -Force};Copy-Item -LiteralPath (Join-Path $backup 'original\bundle-tools.py') -Destination $toolsTarget
    foreach($target in $releaseInfo.targets){if(Test-Path -LiteralPath $target.path){Remove-Item -LiteralPath $target.path -Force};Copy-Item -LiteralPath (Join-Path $backup "original\$($target.name)-hpc_runtime.py") -Destination $target.path}
    & $seal|Out-Null;Assert-HpcSame $beforeHpc (Read-HpcState)
    $restart=@(& $action -Action Start -TimeoutMinutes 12)|Out-String|ConvertFrom-Json;if([string]$restart.status-ne'completed'){throw 'HPC_EXECUTOR_HOTFIX_ROLLBACK_START_FAILED'};$serviceStopped=$false
    Wait-Exact 'http://127.0.0.1:8088/api/healthz' $ExpectedBuildId|Out-Null;Assert-HpcSame $beforeHpc (Read-HpcState);$rollback='restored_and_verified'
  }catch{$rollback='failed';$rollbackError=[string]$_.Exception.Message}}
  elseif($serviceStopped){try{$restart=@(& $action -Action Start -TimeoutMinutes 12)|Out-String|ConvertFrom-Json;if([string]$restart.status-eq'completed'){$serviceStopped=$false;$rollback='start_only_recovered'}}catch{}}
  $errorPayload=[ordered]@{schema='evomind.hpc_executor_hotfix_deployment_error.v1';status='failed';transaction_id=$TransactionId;build_id=$ExpectedBuildId;message=[string]$failure.Exception.Message;rollback=$rollback;rollback_error=if($rollbackError){$rollbackError}else{$null};secrets_printed=$false;hpc_accessed=$false;gpu_touched=$false;training_started=$false;kaggle_submissions=0}
  Write-Utf8 $errorPath $errorPayload;throw
}finally{if($owned){try{[void]$mutex.ReleaseMutex()}catch{}};$mutex.Dispose()}
