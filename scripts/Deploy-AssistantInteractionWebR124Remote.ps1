param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [Parameter(Mandatory)][string]$RemoteRoot,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$WebSha256,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$SourceManifestSha256,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$BuildResultSha256,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$RemoteRunnerSha256,
  [Parameter(Mandatory)][ValidatePattern('^overlay-assistant-interaction-r124-[a-f0-9]{12}$')][string]$NewBuildId,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedRuntimeTreeSha256,
  [switch]$PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

$root='C:\ProgramData\EvoMind'
$sourceBaselineBuildId='overlay-controlled-secret-binding-r112-standalone-f38c40f2dbdc'
$oldBuildId='overlay-assistant-interaction-r122-1393046c9de7'
$fixedRun='run_6bc61ff2ca9145cb9ce3ac29a479f4f2'
$fixedTenant='tenant_ba0ef9d3767f2fb385b856e0'
$fixedJob=93112
$fixedGeneration=25
$expectedRemote=[IO.Path]::GetFullPath("C:\SecureInput\assistant-interaction-r124-$TransactionId")
if([IO.Path]::GetFullPath($RemoteRoot)-cne$expectedRemote){throw 'R124_REMOTE_ROOT_REJECTED'}

$archive=Join-Path $expectedRemote 'web.zip'
$sourceManifestPath=Join-Path $expectedRemote 'source-manifest.json'
$buildResultPath=Join-Path $expectedRemote 'build-result.json'
$runnerPath=Join-Path $expectedRemote 'Deploy-AssistantInteractionWebR124Remote.ps1'
$adminHashPath=Join-Path $expectedRemote 'admin.scrypt'
$extract=Join-Path $expectedRemote 'candidate-extract'
$resultPath=Join-Path $expectedRemote "transaction-$TransactionId-result.json"
$errorPath=Join-Path $expectedRemote "transaction-$TransactionId-error.json"
$action='C:\SecureInput\Invoke-ServiceAccountAction.ps1'
$installWeb=Join-Path $root 'bundle\scripts\Install-WebOverlay.ps1'
$seal=Join-Path $root 'bundle\scripts\Seal-Release.ps1'
$runtimeTarget=Join-Path $root 'bundle\runtime\evomind_runtime'

function Get-Sha256([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
function Get-TextSha256([string]$Value){$h=[Security.Cryptography.SHA256]::Create();try{([BitConverter]::ToString($h.ComputeHash([Text.Encoding]::UTF8.GetBytes($Value)))).Replace('-','').ToLowerInvariant()}finally{$h.Dispose()}}
function Safe-Error([object]$Value){$text=[string]$Value;$text=$text-replace'(?i)(?:Bearer|Basic)\s+\S+','[credential redacted]';$text=$text-replace'(?i)\b(?:password|passwd|pwd|credential|token|secret|api[_ -]?key|authorization|bearer|密码|口令|凭据|账号|登录)\s*[:=]\s*\S+','[credential redacted]';$text=$text-replace'[\r\n]+','_';if($text.Length-gt500){$text=$text.Substring(0,500)};$text}
function Write-Json([string]$Path,[object]$Value){$tmp="$Path.$PID.tmp";[IO.File]::WriteAllText($tmp,(($Value|ConvertTo-Json -Depth 40)+"`n"),[Text.UTF8Encoding]::new($false));Move-Item -LiteralPath $tmp -Destination $Path -Force}
function Read-Object([string]$Path,[string]$Code){if(-not(Test-Path -LiteralPath $Path -PathType Leaf)){throw "${Code}_MISSING"};try{$v=Get-Content -LiteralPath $Path -Raw -Encoding UTF8|ConvertFrom-Json}catch{throw "${Code}_INVALID_JSON"};if($null-eq$v-or$v-is[array]){throw "${Code}_OBJECT_REQUIRED"};$v}
function Assert-Relative([string]$Path){if(-not$Path-or[IO.Path]::IsPathRooted($Path)-or(($Path-split'/')-contains'..')-or$Path.Contains('\')){throw "R124_PATH_UNSAFE:$Path"}}
function Get-RegularFiles([string]$Path,[string[]]$Exclude=@()){$full=[IO.Path]::GetFullPath($Path).TrimEnd('\');$skip=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase);foreach($x in $Exclude){[void]$skip.Add($x)};@(Get-ChildItem -LiteralPath $full -Recurse -File -Force|Where-Object{$_.Name-notmatch'\.(pyc|pyo)$'-and$_.FullName-notmatch'[\\/]__pycache__[\\/]'}|ForEach-Object{$r=$_.FullName.Substring($full.Length+1).Replace('\','/');if(-not$skip.Contains($r)){[ordered]@{path=$r;bytes=[long]$_.Length;sha256=Get-Sha256 $_.FullName}}}|Where-Object{$null-ne$_}|Sort-Object path)}
function Get-Map([object[]]$Entries){$m=@{};foreach($e in @($Entries)){$p=[string]$e.path;Assert-Relative $p;if($m.ContainsKey($p)){throw "R124_DUPLICATE_PATH:$p"};$m[$p]=$e};$m}
function Get-Tree([string]$Path){$rows=Get-RegularFiles $Path;$lines=[Collections.Generic.List[string]]::new();foreach($row in $rows){$lines.Add("$($row.path)|$($row.bytes)|$($row.sha256)")};$lines.Sort([StringComparer]::Ordinal);$text=[string]::Join("`n",$lines);[ordered]@{file_count=$rows.Count;sha256=Get-TextSha256 $text;ordinal_sort=$true}}
function Verify-Manifest([string]$Path,[object]$Manifest){$expected=Get-Map @($Manifest.files);if($expected.Count-ne[int]$Manifest.file_count){throw 'R124_MANIFEST_COUNT_REJECTED'};$actual=Get-Map @(Get-RegularFiles $Path @('operational-overlay-manifest.json'));if($actual.Count-ne$expected.Count){throw 'R124_MANIFEST_FILESET_REJECTED'};foreach($p in $expected.Keys){if(-not$actual.ContainsKey($p)-or[long]$actual[$p].bytes-ne[long]$expected[$p].bytes-or[string]$actual[$p].sha256-cne[string]$expected[$p].sha256){throw "R124_MANIFEST_HASH_REJECTED:$p"}};[ordered]@{file_count=$actual.Count;mismatch=0}}
function Test-Zip([string]$Path){$z=[IO.Compression.ZipFile]::OpenRead($Path);try{$seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase);$unsafe=0;$duplicate=0;$symlink=0;$regular=0;$total=[int64]0;foreach($e in $z.Entries){$n=[string]$e.FullName;if(-not$seen.Add($n)){$duplicate++};$normalizedName=$n.Replace('\','/');if([IO.Path]::IsPathRooted($normalizedName)-or(($normalizedName-split'/')-contains'..')-or$normalizedName.StartsWith('/')-or$normalizedName-match'^[A-Za-z]:/'){$unsafe++};if(((($e.ExternalAttributes -shr 16) -band 0xF000) -eq 0xA000)){$symlink++};if(-not$n.EndsWith('/')){$regular++;$total+=[int64]$e.Length}};[ordered]@{unsafe_entries=$unsafe;duplicate_entries=$duplicate;symlink_entries=$symlink;regular_files=$regular;uncompressed_bytes=$total}}finally{$z.Dispose()}}
function Read-RuntimeInfo{$c=Read-Object (Join-Path $root 'config\node-config.json') 'R124_NODE_CONFIG';$m=Read-Object (Join-Path ([string]$c.web_runtime_root) 'runtime-build-manifest.json') 'R124_WEB_RUNTIME_MANIFEST';[ordered]@{config=$c;manifest=$m}}
function Get-Headers{$c=(Read-RuntimeInfo).config;$p=Join-Path ([string]$c.data_root) 'workspace\runtime\runtime.token';if(-not(Test-Path -LiteralPath $p -PathType Leaf)){throw 'R124_RUNTIME_TOKEN_MISSING'};$t=(Get-Content -LiteralPath $p -Raw -Encoding ASCII).Trim();if($t-cnotmatch'^[A-Za-z0-9_-]{32,128}$'){throw 'R124_RUNTIME_TOKEN_INVALID'};@{Authorization='Bearer '+$t}}
function Invoke-Api([string]$Path){Invoke-RestMethod -Method Get -Uri ('http://127.0.0.1:8765'+$Path)-Headers (Get-Headers)-TimeoutSec 30}
function Get-Health([string]$Uri,[string]$Build){$samples=@();foreach($n in 1..3){$r=Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 15;$p=$r.Content|ConvertFrom-Json;if([int]$r.StatusCode-ne200-or-not[bool]$p.ok-or[string]$p.status-cne'ready'-or[string]$p.build_id-cne$Build-or@($p.failures).Count){throw 'R124_HEALTH_REJECTED'};$samples+=[ordered]@{sample=$n;status_code=200;build_id=$Build;exact=$true};if($n-lt3){Start-Sleep -Milliseconds 250}};[ordered]@{status='ready';samples=$samples}}
function Get-ManagedState{$task=Get-ScheduledTask -TaskPath '\' -TaskName 'EvoMind-Cloud-Node' -ErrorAction Stop;$info=Get-ScheduledTaskInfo -TaskPath '\' -TaskName 'EvoMind-Cloud-Node' -ErrorAction Stop;if([string]$task.State-cne'Ready'-or[int64]$info.LastTaskResult-ne0-or[string]$task.Actions[0].Arguments-cnotmatch'-Action Start(?:\s|$)'){throw 'R124_MANAGED_TASK_REJECTED'};$s=Read-Object (Join-Path $root 'state\node-processes.json') 'R124_NODE_STATE';$roles=[ordered]@{};foreach($role in @('tenant_enrollment_queue','llm_gateway','python_runtime','web')){$r=@($s.records|Where-Object{[string]$_.role-ceq$role});if($r.Count-ne1){throw "R124_ROLE_COUNT:$role"};$rolePid=[int]$r[0].pid;$proc=Get-CimInstance Win32_Process -Filter "ProcessId=$rolePid";if($null-eq$proc){throw "R124_ROLE_PROCESS:$role"};$owner=$proc|Invoke-CimMethod -MethodName GetOwner;if([string]$owner.User-cne'EvoMindSvc'){throw "R124_ROLE_OWNER:$role"};$roles[$role]=[ordered]@{pid=$rolePid;owner='EvoMindSvc';name=[string]$proc.Name}};$listeners=[ordered]@{};foreach($port in 65068,8765,8088,7890){$rows=@(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue);if($rows.Count-ne1){throw "R124_LISTENER_COUNT:$port"};$proc=Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$rows[0].OwningProcess)";$owner=$proc|Invoke-CimMethod -MethodName GetOwner;if([string]$owner.User-cne'EvoMindSvc'){throw "R124_LISTENER_OWNER:$port"};$listeners[[string]$port]=[ordered]@{pid=[int]$rows[0].OwningProcess;owner='EvoMindSvc';name=[string]$proc.Name}};[ordered]@{roles=$roles;listeners=$listeners;state_sha256=Get-Sha256 (Join-Path $root 'state\node-processes.json')}}
function Assert-Fresh([object]$Before,[object]$After){foreach($role in @('tenant_enrollment_queue','llm_gateway','python_runtime','web')){if([int]$Before.roles[$role].pid-eq[int]$After.roles[$role].pid){throw "R124_ROLE_NOT_FRESH:$role"}};foreach($port in @('65068','8765','8088','7890')){if([int]$Before.listeners[$port].pid-eq[int]$After.listeners[$port].pid){throw "R124_LISTENER_NOT_FRESH:$port"}}}
function Get-RunGate{$s=Invoke-Api "/v1/runs/$fixedRun";$active=@($s.active_tool_calls).Count;$pending=@($s.approvals|Where-Object{[string]$_.status-ceq'pending'}).Count;if([string]$s.id-cne$fixedRun-or-not[bool]$s.terminal-or$active-ne0-or$pending-ne0){throw 'R124_RUN_GATE_REJECTED'};[ordered]@{run_id=$fixedRun;status=[string]$s.status;terminal=$true;effective_active_count=0;pending_approval_count=0;last_event_seq=if($s.PSObject.Properties.Name-contains'last_event_seq'){[int64]$s.last_event_seq}else{[int64]0}}}
function Get-Boundary{$info=Read-RuntimeInfo;$binding=Join-Path $root "byoa\tenants\$fixedTenant\hpc-binding.json";$b=Read-Object $binding 'R124_HPC_BINDING';if([string]$b.schema-cne'evomind.tenant_hpc_binding.v1'-or[string]$b.tenant_id-cne$fixedTenant-or[string]$b.state-cne'active'-or[int]$b.job_id-ne$fixedJob-or[int]$b.allocation_generation-ne$fixedGeneration-or[string]$b.credential_profile-cne("${fixedTenant}_job${fixedJob}_g${fixedGeneration}")){throw 'R124_G25_BINDING_REJECTED'};$profileInfo=Get-CimInstance Win32_UserProfile -ErrorAction SilentlyContinue|Where-Object{[IO.Path]::GetFileName(([string]$_.LocalPath).TrimEnd('\'))-ieq[string]$info.config.dedicated_user}|Select-Object -First 1;$profileHome=if($profileInfo){[string]$profileInfo.LocalPath}else{"C:\Users\$([string]$info.config.dedicated_user)"};$profilePath=Join-Path $profileHome "AppData\Roaming\ResearchAgentWorkstation\profiles\$([string]$b.credential_profile)";if(-not(Test-Path -LiteralPath $profilePath -PathType Container)){throw 'R124_G25_PROFILE_MISSING'};[ordered]@{binding_sha256=Get-Sha256 $binding;byoa_tree=Get-Tree (Join-Path $root 'byoa');profile_root=$profilePath;profile_tree=Get-Tree $profilePath;job_id=$fixedJob;allocation_generation=$fixedGeneration;state='active';credential_profile=[string]$b.credential_profile}}
function Assert-BoundarySame([object]$A,[object]$B){if($A.binding_sha256-cne$B.binding_sha256-or$A.byoa_tree.file_count-ne$B.byoa_tree.file_count-or$A.byoa_tree.sha256-cne$B.byoa_tree.sha256-or$A.profile_root-cne$B.profile_root-or$A.profile_tree.file_count-ne$B.profile_tree.file_count-or$A.profile_tree.sha256-cne$B.profile_tree.sha256-or$A.job_id-ne$B.job_id-or$A.allocation_generation-ne$B.allocation_generation-or$A.state-cne$B.state-or$A.credential_profile-cne$B.credential_profile){throw 'R124_HPC_BYOA_BOUNDARY_CHANGED'}}
function Invoke-Managed([ValidateSet('Stop','Start')][string]$Name){if(-not(Test-Path -LiteralPath $action -PathType Leaf)){throw 'R124_SERVICE_ACTION_MISSING'};$r=@(& $action -Action $Name -TimeoutMinutes 12)|Out-String|ConvertFrom-Json;if([string]$r.status-cne'completed'-or[string]$r.action-cne$Name){throw "R124_MANAGED_$($Name.ToUpperInvariant())_FAILED"};$r}
function Assert-Stopped{if(@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue|Where-Object{$_.LocalPort-in@(65068,8765,8088,7890)}).Count){throw 'R124_STOP_LISTENERS_REMAIN'}}
function Restore-Config([string]$Backup){$candidate=Join-Path $root "config\node-config.json.$PID.restore";Copy-Item -LiteralPath $Backup -Destination $candidate -Force;Move-Item -LiteralPath $candidate -Destination (Join-Path $root 'config\node-config.json') -Force}

function Assert-Candidate {
  foreach($x in @([pscustomobject]@{path=$archive;sha=$WebSha256},[pscustomobject]@{path=$sourceManifestPath;sha=$SourceManifestSha256},[pscustomobject]@{path=$buildResultPath;sha=$BuildResultSha256},[pscustomobject]@{path=$runnerPath;sha=$RemoteRunnerSha256})){if(-not(Test-Path -LiteralPath $x.path -PathType Leaf)-or(Get-Sha256 $x.path)-cne$x.sha){throw "R124_INPUT_HASH_REJECTED:$($x.path)"}}
  if((Get-Sha256 $PSCommandPath)-cne$RemoteRunnerSha256){throw 'R124_REMOTE_RUNNER_SELF_HASH_REJECTED'}
  $source=Read-Object $sourceManifestPath 'R124_SOURCE_MANIFEST';$build=Read-Object $buildResultPath 'R124_BUILD_RESULT'
  if([string]$source.schema-cne'evomind.assistant_interaction_web_source_set.r124.v1'-or-not[bool]$source.frozen-or[int]$source.production_allowlist_count-ne3-or[bool]$source.test_in_production-or[string]$source.baseline.build_id-cne$sourceBaselineBuildId-or[string]$source.baseline.zip_sha256-cne'2284252cf7ae98a1d3db08e9117d863d839e55174d428df2e7e00dc1a7d8eaa2'-or[string]$source.baseline.source_manifest_sha256-cne'782333e5552c26ad31be24503c05eed9844af79362df333e3d3641a612e48da1'-or[string]$source.expected_runtime_tree_sha256-cne$ExpectedRuntimeTreeSha256){throw 'R124_SOURCE_MANIFEST_REJECTED'}
  if([string]$build.schema-cne'evomind.assistant_interaction_web_release.r124.v1'-or[string]$build.status-cne'built_and_verified'-or[string]$build.candidate_id-cne$NewBuildId-or[string]$build.zip_sha256-cne$WebSha256-or[string]$build.source_manifest_sha256-cne$SourceManifestSha256-or[int]$build.production_allowlist_count-ne3-or[bool]$build.test_in_production-or-not[bool]($build.targeted_tests.assistant_activity-ceq'passed')-or-not[bool]($build.targeted_tests.assistant_super_agent_contract-ceq'passed')-or-not[bool]($build.targeted_tests.typecheck-ceq'passed')-or-not[bool]($build.targeted_tests.next_build-ceq'passed')-or@($build.verification_rounds).Count-ne2-or[bool]$build.production_deployed){throw 'R124_BUILD_RESULT_REJECTED'}
  $expected=[ordered]@{'src/components/workstation/screens/AssistantScreen.tsx'='47b0afa601dfe14a3d48a57080fdffe7879b49712020ddc5f04b6c8bc9c0eeab';'src/components/workstation/screens/assistant-activity.mjs'='841f3b66b33ffac8a9e223d312d1352adfb8c7d0e055f205e8811baccca1bff9';'src/components/workstation/screens/assistant-activity.d.mts'='38a822860b30db088e5b7757f095438f288078aee606413574c55198e40a60c9'}
  $files=Get-Map @($source.production_files);if($files.Count-ne3){throw 'R124_PRODUCTION_ALLOWLIST_REJECTED'};foreach($p in $expected.Keys){if(-not$files.ContainsKey($p)-or[string]$files[$p].sha256-cne[string]$expected[$p]){throw "R124_PRODUCTION_FILE_REJECTED:$p"}}
  foreach($test in @($source.test_inputs)){if([bool]$test.included_in_production){throw 'R124_TEST_INPUT_PRODUCTION_REJECTED'}}
  $safety=Test-Zip $archive;if($safety.unsafe_entries-or$safety.duplicate_entries-or$safety.symlink_entries-or$safety.regular_files-gt40000-or$safety.uncompressed_bytes-gt3GB){throw 'R124_ARCHIVE_SAFETY_REJECTED'}
  if(Test-Path -LiteralPath $extract){throw 'R124_EXTRACT_EXISTS'};New-Item -ItemType Directory -Path $extract|Out-Null;[IO.Compression.ZipFile]::ExtractToDirectory($archive,$extract)
  $op=Read-Object (Join-Path $extract 'operational-overlay-manifest.json') 'R124_OPERATIONAL_MANIFEST';$runtime=Read-Object (Join-Path $extract 'runtime-build-manifest.json') 'R124_RUNTIME_MANIFEST'
  if([string]$op.schema-cne'evomind.web_operational_overlay.v1'-or[string]$op.overlay_id-cne$NewBuildId-or[string]$op.base_commit-cne'664a636ddd419a66f73cc10820c0a16429784866'-or-not[bool]$op.source_dirty-or[string]$runtime.build_id-cne$NewBuildId-or[string]$runtime.source_tree_sha256-cne[string]$source.source_tree_sha256-or(Get-Sha256 (Join-Path $extract 'release-source-manifest.json'))-cne$SourceManifestSha256){throw 'R124_ARCHIVE_IDENTITY_REJECTED'}
  $changed=Get-Map @($op.changed_source_files);if($changed.Count-ne3){throw 'R124_CHANGED_SOURCE_COUNT_REJECTED'};foreach($p in $expected.Keys){if(-not$changed.ContainsKey($p)-or[string]$changed[$p].sha256-cne[string]$expected[$p]){throw "R124_CHANGED_SOURCE_REJECTED:$p"}}
  $packagedTests=@(Get-ChildItem -LiteralPath $extract -Recurse -File -Force | Where-Object {
    $_.Name -match '(?i)\.test\.(?:ts|tsx|mts|cts|js|mjs)$'
  })
  if($packagedTests.Count){throw 'R124_TEST_FILE_PACKAGED'}
  $closure=Verify-Manifest $extract $op
  if(-not(Test-Path -LiteralPath (Join-Path $extract 'server.js') -PathType Leaf)-or-not(Test-Path -LiteralPath (Join-Path $extract 'node_modules\next\package.json') -PathType Leaf)-or(Test-Path -LiteralPath (Join-Path $extract '.next\standalone'))){throw 'R124_RELEASE_LAYOUT_REJECTED'}
  [ordered]@{source=$source;build=$build;operational=$op;runtime=$runtime;archive_safety=$safety;closure=$closure;tree=Get-Tree $extract}
}

if(-not$PreflightOnly-and((Test-Path -LiteralPath $resultPath)-or(Test-Path -LiteralPath $errorPath))){throw 'R124_RESULT_EXISTS'}
$mutex=[Threading.Mutex]::new($false,'Global\EvoMind-Byoa-V12-Deployment');$owned=$false;$backup=$null;$newOverlay=$null;$mutated=$false;$serviceStopped=$false;$beforeBoundary=$null;$candidate=$null
try{
  $owned=$mutex.WaitOne([TimeSpan]::FromSeconds(10));if(-not$owned){throw 'R124_DEPLOYMENT_LOCKED'}
  foreach($path in @($expectedRemote,$adminHashPath,$action,$installWeb,$seal,$runtimeTarget)){if(-not(Test-Path -LiteralPath $path)){throw "R124_REQUIRED_INPUT_MISSING:$path"}}
  $admin=(Get-Content -LiteralPath $adminHashPath -Raw).Trim();if($admin-cnotmatch'^scrypt\$16384\$8\$1\$[a-f0-9]{32}\$[a-f0-9]{64}$'){throw 'R124_ADMIN_HASH_REJECTED'};$admin=$null
  $candidate=Assert-Candidate
  $preLoopback=Get-Health 'http://127.0.0.1:8088/api/healthz' $oldBuildId;$prePublic=Get-Health 'https://evomind.zhjjq.tech/api/healthz' $oldBuildId
  $managedBefore=Get-ManagedState;$runBefore=Get-RunGate;$runtimeBefore=Get-Tree $runtimeTarget;if($runtimeBefore.file_count-ne32-or[string]$runtimeBefore.sha256-cne$ExpectedRuntimeTreeSha256){throw 'R124_RUNTIME_BASELINE_REJECTED'}
  $beforeInfo=Read-RuntimeInfo;if([string]$beforeInfo.manifest.build_id-cne$oldBuildId){throw 'R124_CURRENT_WEB_BUILD_REJECTED'};$oldWebRoot=[string]$beforeInfo.config.web_runtime_root;$oldWebTree=Get-Tree $oldWebRoot;$beforeBoundary=Get-Boundary
  if($PreflightOnly){[ordered]@{schema='evomind.assistant_interaction_web_deployment_preflight.r124.v1';status='preflight_verified';transaction_id=$TransactionId;old_build_id=$oldBuildId;new_build_id=$NewBuildId;run=$runBefore;binding=[ordered]@{job_id=$fixedJob;allocation_generation=$fixedGeneration;state='active';credential_profile=$beforeBoundary.credential_profile};runtime_tree=$runtimeBefore;candidate_tree=$candidate.tree;managed_service_actions=@();service_restarted=$false;hpc_accessed=$false;gpu_touched=$false;training_started=$false}|ConvertTo-Json -Depth 20 -Compress;return}
  Invoke-Managed Stop|Out-Null;$serviceStopped=$true;Assert-Stopped
  Assert-BoundarySame $beforeBoundary (Get-Boundary)
  $backup=Join-Path $root ('backups\pre-r124-assistant-interaction-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')+'-'+$TransactionId);New-Item -ItemType Directory -Path $backup|Out-Null
  Copy-Item -LiteralPath (Join-Path $root 'config\node-config.json') -Destination (Join-Path $backup 'node-config.json')
  Write-Json (Join-Path $backup 'backup-manifest.json') ([ordered]@{schema='evomind.assistant_interaction_web_backup.r124.v1';transaction_id=$TransactionId;old_build_id=$oldBuildId;new_build_id=$NewBuildId;web_sha256=$WebSha256;run=$runBefore;runtime_tree=$runtimeBefore;boundary=$beforeBoundary;old_web_tree=$oldWebTree})
  $install=@(& $installWeb -ArchivePath $archive -ExpectedArchiveSha256 $WebSha256 -AdminPasswordHashFile $adminHashPath)|Out-String|ConvertFrom-Json
  if([string]$install.status-cne'installed'-or[string]$install.overlay_id-cne$NewBuildId){throw 'R124_WEB_INSTALL_REJECTED'};$mutated=$true;$newOverlay=[string]$install.root
  & $seal|Out-Null
  Assert-BoundarySame $beforeBoundary (Get-Boundary);$runtimeDuring=Get-Tree $runtimeTarget;if($runtimeDuring.file_count-ne$runtimeBefore.file_count-or[string]$runtimeDuring.sha256-cne[string]$runtimeBefore.sha256){throw 'R124_RUNTIME_CHANGED_DURING_INSTALL'}
  Invoke-Managed Start|Out-Null;$serviceStopped=$false
  $loopback=Get-Health 'http://127.0.0.1:8088/api/healthz' $NewBuildId;$public=Get-Health 'https://evomind.zhjjq.tech/api/healthz' $NewBuildId
  $managedAfter=Get-ManagedState;Assert-Fresh $managedBefore $managedAfter
  $runAfter=Get-RunGate;if($runAfter.status-cne$runBefore.status-or$runAfter.last_event_seq-ne$runBefore.last_event_seq){throw 'R124_RUN_CHANGED'}
  $runtimeAfter=Get-Tree $runtimeTarget;if($runtimeAfter.file_count-ne$runtimeBefore.file_count-or[string]$runtimeAfter.sha256-cne[string]$runtimeBefore.sha256){throw 'R124_RUNTIME_CHANGED'}
  $afterBoundary=Get-Boundary;Assert-BoundarySame $beforeBoundary $afterBoundary
  $oldWebAfter=Get-Tree $oldWebRoot;if($oldWebAfter.file_count-ne$oldWebTree.file_count-or[string]$oldWebAfter.sha256-cne[string]$oldWebTree.sha256){throw 'R124_OLD_WEB_CHANGED'}
  $newWebTree=Get-Tree $newOverlay;if($newWebTree.file_count-ne$candidate.tree.file_count-or[string]$newWebTree.sha256-cne[string]$candidate.tree.sha256){throw 'R124_INSTALLED_WEB_TREE_REJECTED'}
  $payload=[ordered]@{schema='evomind.assistant_interaction_web_deployment_result.r124.v1';status='installed_and_verified';transaction_id=$TransactionId;old_build_id=$oldBuildId;new_build_id=$NewBuildId;web_sha256=$WebSha256;source_manifest_sha256=$SourceManifestSha256;build_result_sha256=$BuildResultSha256;backup_root=$backup;loopback=$loopback;public=$public;managed_before=$managedBefore;managed_after=$managedAfter;fresh_roles=$true;fresh_listeners=$true;run_before=$runBefore;run_after=$runAfter;run_unchanged=$true;runtime_before=$runtimeBefore;runtime_after=$runtimeAfter;runtime_unchanged=$true;g25_before=$beforeBoundary;g25_after=$afterBoundary;hpc_binding_unchanged=$true;byoa_unchanged=$true;profile_unchanged=$true;old_web_tree_unchanged=$true;installed_web_tree=$newWebTree;production_web_writes=$true;hpc_remote_writes=0;signals_sent=0;other_processes_modified=$false;hpc_accessed=$false;gpu_touched=$false;training_started=$false;managed_service_actions=@('Stop','Start')}
  Write-Json $resultPath $payload;$payload|ConvertTo-Json -Depth 40 -Compress
}catch{
  $failure=$_
  if($PreflightOnly){throw}
  $rollback='not_required';$rollbackError=$null
  if($mutated-and$backup){
    try{
      if(-not$serviceStopped){Invoke-Managed Stop|Out-Null;$serviceStopped=$true;Assert-Stopped}
      Restore-Config (Join-Path $backup 'node-config.json')
      if($newOverlay-and(Test-Path -LiteralPath $newOverlay -PathType Container)){$resolved=[IO.Path]::GetFullPath($newOverlay);$overlayRoot=[IO.Path]::GetFullPath((Join-Path $root 'web-overlays')).TrimEnd('\')+'\';if(-not$resolved.StartsWith($overlayRoot,[StringComparison]::OrdinalIgnoreCase)){throw 'R124_ROLLBACK_PATH_REJECTED'};Remove-Item -LiteralPath $resolved -Recurse -Force}
      & $seal|Out-Null;Assert-BoundarySame $beforeBoundary (Get-Boundary);$restoredRuntime=Get-Tree $runtimeTarget;if([string]$restoredRuntime.sha256-cne$ExpectedRuntimeTreeSha256){throw 'R124_ROLLBACK_RUNTIME_REJECTED'}
      Invoke-Managed Start|Out-Null;$serviceStopped=$false;Get-Health 'http://127.0.0.1:8088/api/healthz' $oldBuildId|Out-Null;Get-Health 'https://evomind.zhjjq.tech/api/healthz' $oldBuildId|Out-Null;$rollback='restored_and_verified'
    }catch{$rollback='failed';$rollbackError=Safe-Error $_.Exception.Message}
  }elseif($serviceStopped){try{Invoke-Managed Start|Out-Null;$serviceStopped=$false;Get-Health 'http://127.0.0.1:8088/api/healthz' $oldBuildId|Out-Null;$rollback='start_only_recovered'}catch{$rollback='failed';$rollbackError=Safe-Error $_.Exception.Message}}
  Write-Json $errorPath ([ordered]@{schema='evomind.assistant_interaction_web_deployment_error.r124.v1';status='failed';transaction_id=$TransactionId;old_build_id=$oldBuildId;new_build_id=$NewBuildId;message=Safe-Error $failure.Exception.Message;rollback=$rollback;rollback_error=$rollbackError;hpc_remote_writes=0;signals_sent=0;other_processes_modified=$false;hpc_accessed=$false;gpu_touched=$false;training_started=$false})
  throw
}finally{
  if(Test-Path -LiteralPath $extract){Remove-Item -LiteralPath $extract -Recurse -Force -ErrorAction SilentlyContinue}
  if($owned){try{[void]$mutex.ReleaseMutex()}catch{}};$mutex.Dispose()
}
