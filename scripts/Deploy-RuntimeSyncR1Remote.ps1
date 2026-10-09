param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [Parameter(Mandatory)][string]$RemoteRoot,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$WebSha256,
  [Parameter(Mandatory)][string]$ExpectedBuildId,
  [switch]$PreflightOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

$root='C:\ProgramData\EvoMind'
$expectedRemote=[IO.Path]::GetFullPath("C:\SecureInput\runtime-sync-$TransactionId")
if([IO.Path]::GetFullPath($RemoteRoot)-cne$expectedRemote){throw 'RUNTIME_SYNC_REMOTE_ROOT_REJECTED'}
$archive=Join-Path $expectedRemote 'web.zip'
$adminHashFile=Join-Path $expectedRemote 'admin.scrypt'
$resultPath=Join-Path $expectedRemote "transaction-$TransactionId-result.json"
$errorPath=Join-Path $expectedRemote "transaction-$TransactionId-error.json"
$extract=Join-Path $expectedRemote 'extracted'
$action='C:\SecureInput\Invoke-ServiceAccountAction.ps1'
$installWeb=Join-Path $root 'bundle\scripts\Install-WebOverlay.ps1'
$fixedInstaller=Join-Path $expectedRemote 'Install-WebOverlay-r41-fixed.ps1'
$seal=Join-Path $root 'bundle\scripts\Seal-Release.ps1'
$bundleRuntime=Join-Path $root 'bundle\runtime'

function Sha([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
function Assert-NoReparse([string]$Path){
  $rootItem=Get-Item -LiteralPath $Path -Force
  if(($rootItem.Attributes-band[IO.FileAttributes]::ReparsePoint)-ne0){throw "RUNTIME_SYNC_REPARSE_REJECTED:$Path"}
  $bad=@(Get-ChildItem -LiteralPath $Path -Recurse -Force|Where-Object{($_.Attributes-band[IO.FileAttributes]::ReparsePoint)-ne0})
  if($bad.Count){throw "RUNTIME_SYNC_REPARSE_REJECTED:$($bad[0].FullName)"}
}
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
    if($relative-notmatch'^(evomind_runtime|xsci|__pycache__)/'-and$relative-notmatch'\.(pyc|pyo)$'){"$relative|$([long]$_.Length)|$(Sha $_.FullName)"}
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
  if([string]$task.State-ne'Ready'-or[int64]$info.LastTaskResult-ne0-or[string]$task.Actions[0].Arguments-notmatch'-Action Start(?:\s|$)'){throw 'RUNTIME_SYNC_MANAGED_TASK_NOT_READY'}
  $state=Get-Content -LiteralPath "$root\state\node-processes.json" -Raw -Encoding UTF8|ConvertFrom-Json
  foreach($role in @('tenant_enrollment_queue','llm_gateway','python_runtime','web')){
    $record=@($state.records|Where-Object{[string]$_.role-eq$role})
    if($record.Count-ne1-or-not(Get-Process -Id ([int]$record[0].pid) -ErrorAction SilentlyContinue)){throw "RUNTIME_SYNC_ROLE_INVALID:$role"}
    $owner=Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$record[0].pid)"|Invoke-CimMethod -MethodName GetOwner
    if([string]$owner.User-ne'EvoMindSvc'){throw "RUNTIME_SYNC_OWNER_INVALID:$role"}
  }
  $current=Read-Runtime
  if(-not [string]::IsNullOrWhiteSpace($ExpectedBuild) -and [string]$current.runtime.build_id-ne$ExpectedBuild){throw 'RUNTIME_SYNC_BUILD_MISMATCH'}
  [ordered]@{build_id=[string]$current.runtime.build_id;roles=4;task_ready=$true}
}
function Read-HpcState {
  $current=Read-Runtime;$config=$current.config
  $tenantRoot=Join-Path $root 'byoa\tenants\tenant_ba0ef9d3767f2fb385b856e0'
  $bindingPath=Join-Path $tenantRoot 'hpc-binding.json'
  $binding=Get-Content -LiteralPath $bindingPath -Raw -Encoding UTF8|ConvertFrom-Json
  if([string]$binding.schema-ne'evomind.tenant_hpc_binding.v1'-or[string]$binding.tenant_id-ne'tenant_ba0ef9d3767f2fb385b856e0'-or[string]$binding.state-notin@('provisioning','active','frozen','retired')-or[int]$binding.job_id-lt1-or[int]$binding.allocation_generation-lt1-or[string]$binding.credential_profile-ne("tenant_ba0ef9d3767f2fb385b856e0_job$([int]$binding.job_id)_g$([int]$binding.allocation_generation)")){throw 'RUNTIME_SYNC_HPC_BINDING_REJECTED'}
  $profileInfo=Get-CimInstance Win32_UserProfile -ErrorAction SilentlyContinue|Where-Object{[IO.Path]::GetFileName(([string]$_.LocalPath).TrimEnd('\'))-ieq[string]$config.dedicated_user}|Select-Object -First 1
  $serviceHome=if($profileInfo){[string]$profileInfo.LocalPath}else{"C:\Users\$([string]$config.dedicated_user)"}
  $profileRoot=Join-Path $serviceHome "AppData\Roaming\ResearchAgentWorkstation\profiles\$([string]$binding.credential_profile)"
  if(-not(Test-Path -LiteralPath $profileRoot -PathType Container)){throw 'RUNTIME_SYNC_HPC_PROFILE_MISSING'}
  [ordered]@{binding_sha256=Sha $bindingPath;byoa_tree=Tree (Join-Path $root 'byoa');profile_root=$profileRoot;profile_tree=Tree $profileRoot;job_id=[int]$binding.job_id;generation=[int]$binding.allocation_generation;state=[string]$binding.state;credential_profile=[string]$binding.credential_profile}
}
function Assert-HpcSame([object]$Before,[object]$After,[bool]$IncludeByoaTree){
  if($Before.binding_sha256-ne$After.binding_sha256-or$Before.profile_root-ne$After.profile_root-or$Before.profile_tree.sha256-ne$After.profile_tree.sha256-or$Before.profile_tree.file_count-ne$After.profile_tree.file_count-or$Before.job_id-ne$After.job_id-or$Before.generation-ne$After.generation-or$Before.state-ne$After.state-or$Before.credential_profile-ne$After.credential_profile){throw 'RUNTIME_SYNC_HPC_CHANGED'}
  if($IncludeByoaTree-and($Before.byoa_tree.sha256-ne$After.byoa_tree.sha256-or$Before.byoa_tree.file_count-ne$After.byoa_tree.file_count)){throw 'RUNTIME_SYNC_BYOA_CHANGED'}
}
function Wait-Exact([string]$Uri,[string]$Build){
  $samples=@();$consecutive=0;$deadline=[DateTime]::UtcNow.AddSeconds(180)
  do{
    $row=[ordered]@{status_code=0;build_id=$null;exact=$false;elapsed_ms=$null};$sw=[Diagnostics.Stopwatch]::StartNew()
    try{$response=Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 12;$payload=$response.Content|ConvertFrom-Json;$row.status_code=[int]$response.StatusCode;$row.build_id=[string]$payload.build_id;$row.exact=($row.status_code-eq200-and[bool]$payload.ok-and[string]$payload.status-eq'ready'-and$row.build_id-eq$Build-and@($payload.failures).Count-eq0)}catch{}finally{$sw.Stop();$row.elapsed_ms=[math]::Round($sw.Elapsed.TotalMilliseconds,3)}
    if($row.exact){$consecutive++}else{$consecutive=0};$samples+=[pscustomobject]$row
    if($consecutive-ge3){return [ordered]@{status='ready';samples=@($samples|Select-Object -Last 3)}}
    Start-Sleep -Seconds 2
  }while([DateTime]::UtcNow-lt$deadline)
  throw "RUNTIME_SYNC_EXACT_HEALTH_TIMEOUT:$Uri"
}
function Restore-Runtime([string]$Backup,[bool]$HadXsci){
  foreach($name in @('evomind_runtime','xsci')){$target=Join-Path $bundleRuntime $name;if(Test-Path -LiteralPath $target){Remove-Item -LiteralPath $target -Recurse -Force}}
  Move-Item -LiteralPath (Join-Path $Backup 'original\evomind_runtime') -Destination (Join-Path $bundleRuntime 'evomind_runtime')
  if($HadXsci){Move-Item -LiteralPath (Join-Path $Backup 'original\xsci') -Destination (Join-Path $bundleRuntime 'xsci')}
}
function Restore-Config([string]$Backup){
  $candidate="$root\config\node-config.json.$PID.restore"
  Copy-Item -LiteralPath $Backup -Destination $candidate -Force
  Move-Item -LiteralPath $candidate -Destination "$root\config\node-config.json" -Force
}
function Write-FixedInstaller {
  if(Test-Path -LiteralPath $fixedInstaller){throw 'RUNTIME_SYNC_FIXED_INSTALLER_EXISTS'}
  $installerText=Get-Content -LiteralPath $installWeb -Raw -Encoding UTF8
  $libraryOld=". (Join-Path `$PSScriptRoot 'lib\Runtime.ps1')"
  $libraryNew=". '$root\bundle\scripts\lib\Runtime.ps1'"
  if(-not $installerText.Contains($libraryOld)){throw 'RUNTIME_SYNC_INSTALLER_LIBRARY_CONTRACT_MISSING'}
  $installerText=$installerText.Replace($libraryOld,$libraryNew)
  $cleanupOld="if (Test-Path -LiteralPath `$candidate -PathType Container) {`n    Assert-PathUnderRoot -Path `$candidate -Root `$root | Out-Null`n    Remove-Item -LiteralPath `$candidate -Recurse -Force`n  }"
  $cleanupNew = @'
if (Test-Path -LiteralPath $candidate -PathType Container) {
    Assert-PathUnderRoot -Path $candidate -Root $root | Out-Null
    for ($attempt = 1; $attempt -le 5; $attempt++) {
      try {
        & takeown.exe /F $candidate /A /R /D Y 1>$null 2>$null
        & icacls.exe $candidate /inheritance:r /grant:r 'SYSTEM:(OI)(CI)F' "${administrators}:(OI)(CI)F" /T /C 1>$null 2>$null
        Remove-Item -LiteralPath $candidate -Recurse -Force -ErrorAction Stop
        break
      } catch {
        if ($attempt -eq 5) { throw }
        Start-Sleep -Milliseconds 1000
      }
    }
  }
'@
  if(-not $installerText.Contains($cleanupOld)){throw 'RUNTIME_SYNC_INSTALLER_CLEANUP_CONTRACT_MISSING'}
  $installerText=$installerText.Replace($cleanupOld,$cleanupNew)
  [IO.File]::WriteAllText($fixedInstaller,$installerText,[Text.UTF8Encoding]::new($false))
  $tokens=$null;$errors=$null
  [Management.Automation.Language.Parser]::ParseFile($fixedInstaller,[ref]$tokens,[ref]$errors)|Out-Null
  if(@($errors).Count){throw 'RUNTIME_SYNC_FIXED_INSTALLER_AST_REJECTED'}
  if(-not $installerText.Contains('for ($attempt = 1; $attempt -le 5; $attempt++)')){throw 'RUNTIME_SYNC_FIXED_INSTALLER_RETRY_MISSING'}
}

if((Test-Path $resultPath)-or(Test-Path $errorPath)){throw 'RUNTIME_SYNC_RESULT_EXISTS'}
$mutex=[Threading.Mutex]::new($false,'Global\EvoMind-Byoa-V12-Deployment')
$owned=$false;$backup=$null;$mutated=$false;$runtimeMutated=$false;$serviceStopped=$false;$beforeHpc=$null;$hadXsci=$false;$rollbackError=$null;$newOverlay=$null;$oldBuildId=$null
try{
  $owned=$mutex.WaitOne([TimeSpan]::FromSeconds(10));if(-not$owned){throw 'RUNTIME_SYNC_DEPLOYMENT_LOCKED'}
  foreach($path in @($archive,$adminHashFile,$action,$installWeb,$seal,$bundleRuntime)){if(-not(Test-Path -LiteralPath $path)){throw "RUNTIME_SYNC_INPUT_MISSING:$path"}}
  if((Sha $archive)-ne$WebSha256){throw 'RUNTIME_SYNC_ARCHIVE_SHA_MISMATCH'}
  $adminHash=(Get-Content -LiteralPath $adminHashFile -Raw).Trim()
  if($adminHash-notmatch'^scrypt\$16384\$8\$1\$[a-f0-9]{32}\$[a-f0-9]{64}$'){throw 'RUNTIME_SYNC_ADMIN_HASH_INVALID'}
  $adminHash=$null
  if($PreflightOnly){
    Write-FixedInstaller
    [ordered]@{status='preflight_verified';transaction_id=$TransactionId;fixed_installer_ast=$true;cleanup_retry_count=5;service_stopped=$false;hpc_accessed=$false;gpu_touched=$false;secrets_printed=$false}|ConvertTo-Json -Compress
    return
  }
  if(Test-Path -LiteralPath $extract){throw 'RUNTIME_SYNC_EXTRACT_EXISTS'}
  [IO.Compression.ZipFile]::ExtractToDirectory($archive,$extract)
  Assert-NoReparse $extract
  $overlayManifest=Get-Content -LiteralPath (Join-Path $extract 'operational-overlay-manifest.json') -Raw -Encoding UTF8|ConvertFrom-Json
  if([string]$overlayManifest.schema-ne'evomind.web_operational_overlay.v1'-or[string]$overlayManifest.overlay_id-ne$ExpectedBuildId){throw 'RUNTIME_SYNC_OVERLAY_IDENTITY_REJECTED'}
  $runtimeEntries=@($overlayManifest.files|Where-Object{[string]$_.path-like'support/python-runtime/*'})
  foreach($required in @('support/python-runtime/evomind_runtime/assistant_runs.py','support/python-runtime/evomind_runtime/competition_data.py','support/python-runtime/evomind_runtime/run_secrets.py','support/python-runtime/evomind_runtime/http_server.py','support/python-runtime/evomind_runtime/runtime.py','support/python-runtime/evomind_runtime/tools.py','support/python-runtime/xsci/terminal_tools.py')){if(@($runtimeEntries|Where-Object{[string]$_.path-eq$required}).Count-ne1){throw "RUNTIME_SYNC_REQUIRED_FILE_MISSING:$required"}}
  $sourceRuntime=Join-Path $extract 'support\python-runtime'
  $actual=@(Get-ChildItem -LiteralPath $sourceRuntime -Recurse -File -Force|Where-Object{$_.Name-notmatch'\.(pyc|pyo)$'-and$_.FullName-notmatch'[\\/]__pycache__[\\/]'}|ForEach-Object{'support/python-runtime/'+$_.FullName.Substring($sourceRuntime.Length+1).Replace('\','/')})
  if($actual.Count-ne$runtimeEntries.Count-or@($actual|Where-Object{$runtimeEntries.path-notcontains$_}).Count){throw 'RUNTIME_SYNC_SOURCE_FILESET_REJECTED'}
  foreach($entry in $runtimeEntries){$relative=([string]$entry.path).Substring('support/python-runtime/'.Length);$path=Join-Path $sourceRuntime $relative.Replace('/','\');if(-not(Test-Path -LiteralPath $path -PathType Leaf)-or[long](Get-Item -LiteralPath $path).Length-ne[long]$entry.bytes-or(Sha $path)-ne[string]$entry.sha256){throw "RUNTIME_SYNC_SOURCE_HASH_REJECTED:$relative"}}
  foreach($package in @('evomind_runtime','xsci')){if(-not(Test-Path -LiteralPath (Join-Path $sourceRuntime $package) -PathType Container)){throw "RUNTIME_SYNC_PACKAGE_MISSING:$package"}}
  $sourceTrees=[ordered]@{evomind_runtime=Tree (Join-Path $sourceRuntime 'evomind_runtime');xsci=Tree (Join-Path $sourceRuntime 'xsci')}
  Write-FixedInstaller
  # A fresh deployment may legitimately replace an older verified build.  The
  # initial gate therefore validates that the current process is managed and
  # healthy without requiring it to already equal the incoming build.  The
  # post-start gate below remains exact and still rejects any drift.
  $beforeRuntime=Assert-Managed ''
  $oldBuildId=[string]$beforeRuntime.build_id
  $targetOverlay=Join-Path "$root\web-overlays" $ExpectedBuildId
  if(Test-Path -LiteralPath $targetOverlay){throw 'RUNTIME_SYNC_TARGET_OVERLAY_EXISTS'}
  $stop=@(& $action -Action Stop -TimeoutMinutes 12)|Out-String|ConvertFrom-Json
  if([string]$stop.status-ne'completed'-or[string]$stop.action-ne'Stop'){throw 'RUNTIME_SYNC_STOP_FAILED'}
  $serviceStopped=$true
  foreach($port in 65068,8765,8088,7890){if(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue){throw "RUNTIME_SYNC_LISTENER_REMAINS:$port"}}
  $beforeHpc=Read-HpcState
  $otherBefore=Other-RuntimeTree
  $backup=Join-Path "$root\backups" ('pre-runtime-sync-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
  New-Item -ItemType Directory -Path (Join-Path $backup 'original') -Force|Out-Null
  Copy-Item -LiteralPath "$root\config\node-config.json" -Destination (Join-Path $backup 'node-config.json')
  $hadXsci=Test-Path -LiteralPath (Join-Path $bundleRuntime 'xsci') -PathType Container
  $oldTrees=[ordered]@{evomind_runtime=Tree (Join-Path $bundleRuntime 'evomind_runtime');xsci=if($hadXsci){Tree (Join-Path $bundleRuntime 'xsci')}else{$null}}
  [IO.File]::WriteAllText((Join-Path $backup 'backup-manifest.json'),(( [ordered]@{schema='evomind.runtime_sync_backup.v2';transaction_id=$TransactionId;old_build_id=$oldBuildId;new_build_id=$ExpectedBuildId;web_sha256=$WebSha256;had_xsci=$hadXsci;old_trees=$oldTrees;new_trees=$sourceTrees;other_runtime_tree=$otherBefore;hpc=$beforeHpc}|ConvertTo-Json -Depth 8)+"`n"),[Text.UTF8Encoding]::new($false))
  $mutated=$true
  $install=@(& $fixedInstaller -ArchivePath $archive -ExpectedArchiveSha256 $WebSha256 -AdminPasswordHashFile $adminHashFile)|Out-String|ConvertFrom-Json
  if([string]$install.status-ne'installed'-or[string]$install.overlay_id-ne$ExpectedBuildId){throw 'RUNTIME_SYNC_WEB_INSTALL_FAILED'}
  $newOverlay=[string]$install.root
  Move-Item -LiteralPath (Join-Path $bundleRuntime 'evomind_runtime') -Destination (Join-Path $backup 'original\evomind_runtime')
  $runtimeMutated=$true
  if($hadXsci){Move-Item -LiteralPath (Join-Path $bundleRuntime 'xsci') -Destination (Join-Path $backup 'original\xsci')}
  Move-Item -LiteralPath (Join-Path $sourceRuntime 'evomind_runtime') -Destination (Join-Path $bundleRuntime 'evomind_runtime')
  Move-Item -LiteralPath (Join-Path $sourceRuntime 'xsci') -Destination (Join-Path $bundleRuntime 'xsci')
  Assert-NoReparse (Join-Path $bundleRuntime 'evomind_runtime');Assert-NoReparse (Join-Path $bundleRuntime 'xsci')
  $installedTrees=[ordered]@{evomind_runtime=Tree (Join-Path $bundleRuntime 'evomind_runtime');xsci=Tree (Join-Path $bundleRuntime 'xsci')}
  if($installedTrees.evomind_runtime.sha256-ne$sourceTrees.evomind_runtime.sha256-or$installedTrees.evomind_runtime.file_count-ne$sourceTrees.evomind_runtime.file_count-or$installedTrees.xsci.sha256-ne$sourceTrees.xsci.sha256-or$installedTrees.xsci.file_count-ne$sourceTrees.xsci.file_count){throw 'RUNTIME_SYNC_INSTALLED_TREE_REJECTED'}
  $otherAfter=Other-RuntimeTree
  if($otherAfter.sha256-ne$otherBefore.sha256-or$otherAfter.file_count-ne$otherBefore.file_count){throw 'RUNTIME_SYNC_OTHER_RUNTIME_CHANGED'}
  & $seal|Out-Null
  Assert-HpcSame $beforeHpc (Read-HpcState) $true
  $start=@(& $action -Action Start -TimeoutMinutes 12)|Out-String|ConvertFrom-Json
  if([string]$start.status-ne'completed'-or[string]$start.action-ne'Start'){throw 'RUNTIME_SYNC_START_FAILED'}
  $serviceStopped=$false
  $loopback=Wait-Exact 'http://127.0.0.1:8088/api/healthz' $ExpectedBuildId
  $public=Wait-Exact 'https://evomind.zhjjq.tech/api/healthz' $ExpectedBuildId
  $afterRuntime=Assert-Managed $ExpectedBuildId
  $afterHpc=Read-HpcState;Assert-HpcSame $beforeHpc $afterHpc $false
  $payload=[ordered]@{schema='evomind.runtime_sync_deployment_result.v2';status='installed_and_verified';transaction_id=$TransactionId;old_build_id=$oldBuildId;build_id=$ExpectedBuildId;web_sha256=$WebSha256;backup_root=$backup;web_overlay_root=$newOverlay;old_runtime_trees=$oldTrees;installed_runtime_trees=$installedTrees;loopback=$loopback;public=$public;runtime=$afterRuntime;hpc_binding_unchanged=$true;byoa_stop_window_unchanged=$true;profile_unchanged=$true;job_id=$afterHpc.job_id;allocation_generation=$afterHpc.generation;binding_state=$afterHpc.state;secrets_printed=$false;hpc_accessed=$false;gpu_touched=$false}
  [IO.File]::WriteAllText($resultPath,(($payload|ConvertTo-Json -Depth 10)+"`n"),[Text.UTF8Encoding]::new($false));$payload|ConvertTo-Json -Depth 10 -Compress
}catch{
  $failure=$_;$rollback='not_required'
  if($backup-and$mutated){
    try{
      try{$rs=@(& $action -Action Stop -TimeoutMinutes 12)|Out-String|ConvertFrom-Json;if([string]$rs.status-ne'completed'){throw 'RUNTIME_SYNC_ROLLBACK_STOP_FAILED'};$serviceStopped=$true}catch{$listeners=@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue|Where-Object{$_.LocalPort-in@(65068,8765,8088,7890)});if($listeners.Count){throw};$serviceStopped=$true}
      Restore-Config (Join-Path $backup 'node-config.json')
      if($newOverlay-and(Test-Path -LiteralPath $newOverlay -PathType Container)){Remove-Item -LiteralPath $newOverlay -Recurse -Force}
      if($runtimeMutated){Restore-Runtime $backup $hadXsci}
      & $seal|Out-Null
      Assert-HpcSame $beforeHpc (Read-HpcState) $true
      $restart=@(& $action -Action Start -TimeoutMinutes 12)|Out-String|ConvertFrom-Json
      if([string]$restart.status-ne'completed'){throw 'RUNTIME_SYNC_ROLLBACK_START_FAILED'}
      $serviceStopped=$false
      Wait-Exact 'http://127.0.0.1:8088/api/healthz' $oldBuildId|Out-Null
      Assert-HpcSame $beforeHpc (Read-HpcState) $false
      $rollback='restored_and_verified'
    }catch{$rollback='failed';$rollbackError=[string]$_.Exception.Message}
  }elseif($serviceStopped){try{$restart=@(& $action -Action Start -TimeoutMinutes 12)|Out-String|ConvertFrom-Json;if([string]$restart.status-eq'completed'){$serviceStopped=$false;$rollback='start_only_recovered'}}catch{}}
  $errorPayload=[ordered]@{schema='evomind.runtime_sync_deployment_error.v1';status='failed';transaction_id=$TransactionId;build_id=$ExpectedBuildId;message=[string]$failure.Exception.Message;rollback=$rollback;rollback_error=if($rollbackError){$rollbackError}else{$null};secrets_printed=$false;hpc_accessed=$false;gpu_touched=$false}
  [IO.File]::WriteAllText($errorPath,(($errorPayload|ConvertTo-Json -Depth 6)+"`n"),[Text.UTF8Encoding]::new($false));throw
}finally{
  if(Test-Path -LiteralPath $fixedInstaller){Remove-Item -LiteralPath $fixedInstaller -Force -ErrorAction SilentlyContinue}
  if($owned){try{[void]$mutex.ReleaseMutex()}catch{}};$mutex.Dispose()
}
