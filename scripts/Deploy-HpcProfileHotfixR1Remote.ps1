param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [Parameter(Mandatory)][string]$RemoteRoot,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$HotfixSha256,
  [Parameter(Mandatory)][string]$ExpectedBuildId
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$root='C:\ProgramData\EvoMind'
$expectedRemote="C:\SecureInput\hpc-profile-hotfix-$TransactionId"
if([IO.Path]::GetFullPath($RemoteRoot)-cne$expectedRemote){throw 'HPC_PROFILE_HOTFIX_REMOTE_ROOT_REJECTED'}
$archive=Join-Path $expectedRemote 'hotfix.zip'
$stage=Join-Path $expectedRemote 'hotfix-stage'
$resultPath=Join-Path $expectedRemote "transaction-$TransactionId-result.json"
$errorPath=Join-Path $expectedRemote "transaction-$TransactionId-error.json"
$action='C:\SecureInput\Invoke-ServiceAccountAction.ps1'
$seal=Join-Path $root 'bundle\scripts\Seal-Release.ps1'
$bundleRuntime=Join-Path $root 'bundle\runtime'
$gpuRelative='research_agent_workstation/server/core/gpu_credentials.py'
$terminalRelative='xsci/terminal_tools.py'
$gpuTarget=Join-Path $bundleRuntime $gpuRelative.Replace('/','\')
$terminalTarget=Join-Path $bundleRuntime $terminalRelative.Replace('/','\')
$expectedOldGpu='a8d19ff2ceedcb5b5b4d716bb23380cef2ab307bdaaa10402ab691679e8bf804'
$expectedOldTerminal='eec0457e427b271cf714fa2d229a18eed973d45be24d786692887a3419f4e9e2'

function Sha([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
function Write-Utf8([string]$Path,[object]$Value){[IO.File]::WriteAllText($Path,(($Value|ConvertTo-Json -Depth 12)+"`n"),[Text.UTF8Encoding]::new($false))}
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
    if($relative-ne$gpuRelative-and$relative-ne$terminalRelative-and$relative-notmatch'[\\/]__pycache__[\\/]'-and$relative-notmatch'\.(pyc|pyo)$'){"$relative|$([long]$_.Length)|$(Sha $_.FullName)"}
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
  if([string]$task.State-ne'Ready'-or[int64]$info.LastTaskResult-ne0-or[string]$task.Actions[0].Arguments-notmatch'-Action Start(?:\s|$)'){throw 'HPC_PROFILE_HOTFIX_MANAGED_TASK_NOT_READY'}
  $state=Get-Content -LiteralPath "$root\state\node-processes.json" -Raw -Encoding UTF8|ConvertFrom-Json
  foreach($role in @('tenant_enrollment_queue','llm_gateway','python_runtime','web')){
    $record=@($state.records|Where-Object{[string]$_.role-eq$role})
    if($record.Count-ne1-or-not(Get-Process -Id ([int]$record[0].pid) -ErrorAction SilentlyContinue)){throw "HPC_PROFILE_HOTFIX_ROLE_INVALID:$role"}
    $owner=Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$record[0].pid)"|Invoke-CimMethod -MethodName GetOwner
    if([string]$owner.User-ne'EvoMindSvc'){throw "HPC_PROFILE_HOTFIX_OWNER_INVALID:$role"}
  }
  $current=Read-Runtime;if([string]$current.runtime.build_id-ne$ExpectedBuild){throw 'HPC_PROFILE_HOTFIX_BUILD_MISMATCH'}
  [ordered]@{build_id=[string]$current.runtime.build_id;roles=4;task_ready=$true}
}
function Read-HpcState {
  $current=Read-Runtime;$config=$current.config
  $tenantRoot=Join-Path $root 'byoa\tenants\tenant_ba0ef9d3767f2fb385b856e0';$bindingPath=Join-Path $tenantRoot 'hpc-binding.json'
  $binding=Get-Content -LiteralPath $bindingPath -Raw -Encoding UTF8|ConvertFrom-Json
  if([string]$binding.schema-ne'evomind.tenant_hpc_binding.v1'-or[string]$binding.tenant_id-ne'tenant_ba0ef9d3767f2fb385b856e0'-or[string]$binding.state-ne'active'-or[int]$binding.job_id-ne91699-or[int]$binding.allocation_generation-ne15-or[string]$binding.credential_profile-ne'tenant_ba0ef9d3767f2fb385b856e0_job91699_g15'){throw 'HPC_PROFILE_HOTFIX_BINDING_REJECTED'}
  $profileInfo=Get-CimInstance Win32_UserProfile -ErrorAction SilentlyContinue|Where-Object{[IO.Path]::GetFileName(([string]$_.LocalPath).TrimEnd('\'))-ieq[string]$config.dedicated_user}|Select-Object -First 1
  $serviceHome=if($profileInfo){[string]$profileInfo.LocalPath}else{"C:\Users\$([string]$config.dedicated_user)"}
  $profileRoot=Join-Path $serviceHome "AppData\Roaming\ResearchAgentWorkstation\profiles\$([string]$binding.credential_profile)"
  if(-not(Test-Path -LiteralPath $profileRoot -PathType Container)){throw 'HPC_PROFILE_HOTFIX_PROFILE_MISSING'}
  [ordered]@{binding_sha256=Sha $bindingPath;byoa_tree=Tree (Join-Path $root 'byoa');profile_root=$profileRoot;profile_tree=Tree $profileRoot;job_id=[int]$binding.job_id;generation=[int]$binding.allocation_generation;state=[string]$binding.state;credential_profile=[string]$binding.credential_profile}
}
function Assert-HpcSame([object]$Before,[object]$After){
  if($Before.binding_sha256-ne$After.binding_sha256-or$Before.byoa_tree.sha256-ne$After.byoa_tree.sha256-or$Before.byoa_tree.file_count-ne$After.byoa_tree.file_count-or$Before.profile_root-ne$After.profile_root-or$Before.profile_tree.sha256-ne$After.profile_tree.sha256-or$Before.profile_tree.file_count-ne$After.profile_tree.file_count-or$Before.job_id-ne$After.job_id-or$Before.generation-ne$After.generation-or$Before.state-ne$After.state-or$Before.credential_profile-ne$After.credential_profile){throw 'HPC_PROFILE_HOTFIX_HPC_STATE_CHANGED'}
}
function Wait-Exact([string]$Uri,[string]$Build){
  $samples=@();$consecutive=0;$deadline=[DateTime]::UtcNow.AddSeconds(180)
  do{$row=[ordered]@{status_code=0;build_id=$null;exact=$false;elapsed_ms=$null};$sw=[Diagnostics.Stopwatch]::StartNew();try{$response=Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 12;$payload=$response.Content|ConvertFrom-Json;$row.status_code=[int]$response.StatusCode;$row.build_id=[string]$payload.build_id;$row.exact=($row.status_code-eq200-and[bool]$payload.ok-and[string]$payload.status-eq'ready'-and$row.build_id-eq$Build-and@($payload.failures).Count-eq0)}catch{}finally{$sw.Stop();$row.elapsed_ms=[math]::Round($sw.Elapsed.TotalMilliseconds,3)};if($row.exact){$consecutive++}else{$consecutive=0};$samples+=[pscustomobject]$row;if($consecutive-ge3){return [ordered]@{status='ready';samples=@($samples|Select-Object -Last 3)}};Start-Sleep -Seconds 2}while([DateTime]::UtcNow-lt$deadline)
  throw "HPC_PROFILE_HOTFIX_EXACT_HEALTH_TIMEOUT:$Uri"
}
function Read-Hotfix {
  $manifestPath=Join-Path $stage 'hpc-profile-hotfix-manifest.json'
  $manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8|ConvertFrom-Json
  if([string]$manifest.schema-ne'evomind.super_agent_hpc_profile_hotfix.v1'-or[string]$manifest.build_id-ne$ExpectedBuildId-or[int]$manifest.file_count-ne2-or@($manifest.files).Count-ne2){throw 'HPC_PROFILE_HOTFIX_MANIFEST_REJECTED'}
  $declared=@($manifest.files.path);$required=@($gpuRelative,$terminalRelative)
  if(@($declared|Where-Object{$required-notcontains$_}).Count-or@($required|Where-Object{$declared-notcontains$_}).Count){throw 'HPC_PROFILE_HOTFIX_DECLARED_FILESET_REJECTED'}
  $actual=@(Get-ChildItem -LiteralPath $stage -Recurse -File -Force|ForEach-Object{$_.FullName.Substring($stage.Length+1).Replace('\','/')})
  if($actual.Count-ne3-or@($actual|Where-Object{$_-ne'hpc-profile-hotfix-manifest.json'-and$declared-notcontains$_}).Count){throw 'HPC_PROFILE_HOTFIX_ACTUAL_FILESET_REJECTED'}
  foreach($entry in $manifest.files){$path=Join-Path $stage ([string]$entry.path).Replace('/','\');if(-not(Test-Path -LiteralPath $path -PathType Leaf)-or[long](Get-Item -LiteralPath $path).Length-ne[long]$entry.bytes-or(Sha $path)-ne[string]$entry.sha256){throw "HPC_PROFILE_HOTFIX_HASH_REJECTED:$($entry.path)"}}
  $manifest
}

if((Test-Path $resultPath)-or(Test-Path $errorPath)){throw 'HPC_PROFILE_HOTFIX_RESULT_EXISTS'}
$mutex=[Threading.Mutex]::new($false,'Global\EvoMind-Byoa-V12-Deployment')
$owned=$false;$backup=$null;$mutated=$false;$serviceStopped=$false;$beforeHpc=$null;$rollbackError=$null
try{
  $owned=$mutex.WaitOne([TimeSpan]::FromSeconds(10));if(-not$owned){throw 'HPC_PROFILE_HOTFIX_DEPLOYMENT_LOCKED'}
  foreach($path in @($archive,$action,$seal,$gpuTarget,$terminalTarget)){if(-not(Test-Path -LiteralPath $path)){throw "HPC_PROFILE_HOTFIX_INPUT_MISSING:$path"}}
  if((Sha $archive)-ne$HotfixSha256){throw 'HPC_PROFILE_HOTFIX_ARCHIVE_SHA_MISMATCH'}
  if((Sha $gpuTarget)-ne$expectedOldGpu-or(Sha $terminalTarget)-ne$expectedOldTerminal){throw 'HPC_PROFILE_HOTFIX_OLD_SHA_REJECTED'}
  if(Test-Path -LiteralPath $stage){throw 'HPC_PROFILE_HOTFIX_STAGE_EXISTS'}
  Expand-Archive -LiteralPath $archive -DestinationPath $stage
  $manifest=Read-Hotfix
  $gpuEntry=@($manifest.files|Where-Object{$_.path-eq$gpuRelative})[0]
  $terminalEntry=@($manifest.files|Where-Object{$_.path-eq$terminalRelative})[0]
  $beforeRuntime=Assert-Managed $ExpectedBuildId
  $stop=@(& $action -Action Stop -TimeoutMinutes 12)|Out-String|ConvertFrom-Json
  if([string]$stop.status-ne'completed'-or[string]$stop.action-ne'Stop'){throw 'HPC_PROFILE_HOTFIX_STOP_FAILED'}
  $serviceStopped=$true
  foreach($port in 65068,8765,8088,7890){if(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue){throw "HPC_PROFILE_HOTFIX_LISTENER_REMAINS:$port"}}
  $beforeHpc=Read-HpcState;$otherBefore=Other-RuntimeTree
  $backup=Join-Path "$root\backups" ('pre-hpc-profile-hotfix-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
  New-Item -ItemType Directory -Path (Join-Path $backup 'research_agent_workstation\server\core') -Force|Out-Null
  New-Item -ItemType Directory -Path (Join-Path $backup 'xsci') -Force|Out-Null
  New-Item -ItemType Directory -Path (Join-Path $backup 'original\research_agent_workstation\server\core') -Force|Out-Null
  New-Item -ItemType Directory -Path (Join-Path $backup 'original\xsci') -Force|Out-Null
  Copy-Item -LiteralPath $gpuTarget -Destination (Join-Path $backup $gpuRelative.Replace('/','\'))
  Copy-Item -LiteralPath $terminalTarget -Destination (Join-Path $backup $terminalRelative.Replace('/','\'))
  Write-Utf8 (Join-Path $backup 'backup-manifest.json') ([ordered]@{schema='evomind.hpc_profile_hotfix_backup.v1';transaction_id=$TransactionId;build_id=$ExpectedBuildId;hotfix_sha256=$HotfixSha256;source_manifest_sha256=[string]$manifest.source_manifest_sha256;old_gpu_sha256=$expectedOldGpu;old_terminal_sha256=$expectedOldTerminal;new_gpu_sha256=[string]$gpuEntry.sha256;new_terminal_sha256=[string]$terminalEntry.sha256;other_runtime_tree=$otherBefore;hpc=$beforeHpc})
  $mutated=$true
  Move-Item -LiteralPath $gpuTarget -Destination (Join-Path $backup ('original\'+$gpuRelative.Replace('/','\')))
  Move-Item -LiteralPath (Join-Path $stage $gpuRelative.Replace('/','\')) -Destination $gpuTarget
  Move-Item -LiteralPath $terminalTarget -Destination (Join-Path $backup ('original\'+$terminalRelative.Replace('/','\')))
  Move-Item -LiteralPath (Join-Path $stage $terminalRelative.Replace('/','\')) -Destination $terminalTarget
  if((Sha $gpuTarget)-ne[string]$gpuEntry.sha256-or(Sha $terminalTarget)-ne[string]$terminalEntry.sha256){throw 'HPC_PROFILE_HOTFIX_INSTALLED_SHA_REJECTED'}
  $otherAfter=Other-RuntimeTree
  if($otherAfter.sha256-ne$otherBefore.sha256-or$otherAfter.file_count-ne$otherBefore.file_count){throw 'HPC_PROFILE_HOTFIX_OTHER_RUNTIME_CHANGED'}
  & $seal|Out-Null;Assert-HpcSame $beforeHpc (Read-HpcState)
  $start=@(& $action -Action Start -TimeoutMinutes 12)|Out-String|ConvertFrom-Json
  if([string]$start.status-ne'completed'-or[string]$start.action-ne'Start'){throw 'HPC_PROFILE_HOTFIX_START_FAILED'}
  $serviceStopped=$false
  $loopback=Wait-Exact 'http://127.0.0.1:8088/api/healthz' $ExpectedBuildId;$public=Wait-Exact 'https://evomind.zhjjq.tech/api/healthz' $ExpectedBuildId
  $afterRuntime=Assert-Managed $ExpectedBuildId;$afterHpc=Read-HpcState;Assert-HpcSame $beforeHpc $afterHpc
  if((Sha $gpuTarget)-ne[string]$gpuEntry.sha256-or(Sha $terminalTarget)-ne[string]$terminalEntry.sha256){throw 'HPC_PROFILE_HOTFIX_POSTSTART_SHA_REJECTED'}
  $payload=[ordered]@{schema='evomind.hpc_profile_hotfix_deployment_result.v1';status='installed_and_verified';transaction_id=$TransactionId;build_id=$ExpectedBuildId;hotfix_sha256=$HotfixSha256;source_manifest_sha256=[string]$manifest.source_manifest_sha256;backup_root=$backup;old_gpu_sha256=$expectedOldGpu;old_terminal_sha256=$expectedOldTerminal;gpu_sha256=[string]$gpuEntry.sha256;terminal_sha256=[string]$terminalEntry.sha256;loopback=$loopback;public=$public;runtime=$afterRuntime;hpc_binding_unchanged=$true;byoa_tree_unchanged=$true;profile_unchanged=$true;job_id=$afterHpc.job_id;allocation_generation=$afterHpc.generation;binding_state=$afterHpc.state;secrets_printed=$false;hpc_accessed=$false;gpu_touched=$false;training_started=$false;kaggle_submissions=0}
  Write-Utf8 $resultPath $payload;$payload|ConvertTo-Json -Depth 12 -Compress
}catch{
  $failure=$_;$rollback='not_required'
  if($backup-and$mutated){try{try{$rs=@(& $action -Action Stop -TimeoutMinutes 12)|Out-String|ConvertFrom-Json;if([string]$rs.status-ne'completed'){throw 'HPC_PROFILE_HOTFIX_ROLLBACK_STOP_FAILED'};$serviceStopped=$true}catch{$listeners=@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue|Where-Object{$_.LocalPort-in@(65068,8765,8088,7890)});if($listeners.Count){throw};$serviceStopped=$true};$oldGpu=Join-Path $backup ('original\'+$gpuRelative.Replace('/','\'));$oldTerminal=Join-Path $backup ('original\'+$terminalRelative.Replace('/','\'));if(Test-Path -LiteralPath $gpuTarget){Move-Item -LiteralPath $gpuTarget -Destination (Join-Path $stage 'rollback-new-gpu_credentials.py') -Force};if(Test-Path -LiteralPath $terminalTarget){Move-Item -LiteralPath $terminalTarget -Destination (Join-Path $stage 'rollback-new-terminal_tools.py') -Force};if(Test-Path -LiteralPath $oldGpu){Move-Item -LiteralPath $oldGpu -Destination $gpuTarget}else{Copy-Item -LiteralPath (Join-Path $backup $gpuRelative.Replace('/','\')) -Destination $gpuTarget};if(Test-Path -LiteralPath $oldTerminal){Move-Item -LiteralPath $oldTerminal -Destination $terminalTarget}else{Copy-Item -LiteralPath (Join-Path $backup $terminalRelative.Replace('/','\')) -Destination $terminalTarget};& $seal|Out-Null;Assert-HpcSame $beforeHpc (Read-HpcState);$restart=@(& $action -Action Start -TimeoutMinutes 12)|Out-String|ConvertFrom-Json;if([string]$restart.status-ne'completed'){throw 'HPC_PROFILE_HOTFIX_ROLLBACK_START_FAILED'};$serviceStopped=$false;Wait-Exact 'http://127.0.0.1:8088/api/healthz' $ExpectedBuildId|Out-Null;Assert-HpcSame $beforeHpc (Read-HpcState);$rollback='restored_and_verified'}catch{$rollback='failed';$rollbackError=[string]$_.Exception.Message}}
  elseif($serviceStopped){try{$restart=@(& $action -Action Start -TimeoutMinutes 12)|Out-String|ConvertFrom-Json;if([string]$restart.status-eq'completed'){$serviceStopped=$false;$rollback='start_only_recovered'}}catch{}}
  $errorPayload=[ordered]@{schema='evomind.hpc_profile_hotfix_deployment_error.v1';status='failed';transaction_id=$TransactionId;build_id=$ExpectedBuildId;message=[string]$failure.Exception.Message;rollback=$rollback;rollback_error=if($rollbackError){$rollbackError}else{$null};secrets_printed=$false;hpc_accessed=$false;gpu_touched=$false;training_started=$false;kaggle_submissions=0}
  Write-Utf8 $errorPath $errorPayload;throw
}finally{if($owned){try{[void]$mutex.ReleaseMutex()}catch{}};$mutex.Dispose()}
