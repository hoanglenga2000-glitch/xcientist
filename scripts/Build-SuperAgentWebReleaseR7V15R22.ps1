param(
  [Parameter(Mandatory)][string]$OverlayRoot,
  [Parameter(Mandatory)][string]$OutputZip,
  [Parameter(Mandatory)][string]$ExpectedBuildId,
  [Parameter(Mandatory)][string]$ExpectedSourceManifestPath,
  [Parameter(Mandatory)][string]$TestResultPath,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedCompetitionDataSha,
  [string]$BaselineDatabase='D:\桌面\codex\科研港科技\web\research-agent-workstation\prisma\workstation.db',
  [string]$VerificationBasePath='D:\EV12'
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Sha([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
function Free-Port{$listener=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0);try{$listener.Start();[int]$listener.LocalEndpoint.Port}finally{$listener.Stop()}}
function Assert-Overlay([string]$Root){
  $manifest=Get-Content -LiteralPath (Join-Path $Root 'operational-overlay-manifest.json') -Raw -Encoding UTF8|ConvertFrom-Json
  if([string]$manifest.schema-ne'evomind.web_operational_overlay.v1'-or[string]$manifest.overlay_id-ne$ExpectedBuildId){throw 'SUPER_AGENT_OVERLAY_MANIFEST_REJECTED'}
  $source=Get-Content -LiteralPath ([IO.Path]::GetFullPath($ExpectedSourceManifestPath)) -Raw -Encoding UTF8|ConvertFrom-Json
  if([string]$source.schema-ne'evomind.super_agent_source_set.v1'-or-not[bool]$source.frozen-or[string]$manifest.source_identity_sha256-ne[string]$source.source_tree_sha256-or[string]$manifest.source_manifest_sha256-ne(Sha ([IO.Path]::GetFullPath($ExpectedSourceManifestPath)))){throw 'SUPER_AGENT_OVERLAY_SOURCE_IDENTITY_REJECTED'}
  $expected=@($manifest.files|ForEach-Object{[string]$_.path})
  if($expected.Count-ne[int]$manifest.file_count-or@($expected|Select-Object -Unique).Count-ne$expected.Count){throw 'SUPER_AGENT_OVERLAY_FILESET_INVALID'}
  foreach($required in @(
    'server.js','runtime-build-manifest.json','release-source-manifest.json','support/transition_tenant_hpc_profile.py',
    'support/python-runtime/evomind_runtime/assistant_runs.py','support/python-runtime/evomind_runtime/competition_data.py',
    'support/python-runtime/evomind_runtime/run_secrets.py','support/python-runtime/evomind_runtime/tools.py',
    'support/python-runtime/evomind_runtime/agent_kernel_v2.py','support/python-runtime/evomind_runtime/capabilities.py',
    'support/python-runtime/evomind_runtime/connectors.py','support/python-runtime/evomind_runtime/credential_leases.py',
    'support/python-runtime/evomind_runtime/directory_broker.py','support/python-runtime/evomind_runtime/ecosystem.py',
    'support/python-runtime/evomind_runtime/evidence_memory.py','support/python-runtime/evomind_runtime/recovery.py',
    'support/python-runtime/evomind_runtime/remote_connectors.py','support/python-runtime/evomind_runtime/super_agent_runtime.py',
    'support/python-runtime/evomind_runtime/super_agent_store.py','support/python-runtime/evomind_runtime/super_agent_tools.py',
    'support/python-runtime/evomind_runtime/tool_package_runtime.py','support/python-runtime/xsci/terminal_tools.py',
    'support/python-vendor/kaggle/__init__.py'
  )){if($expected-notcontains$required){throw "SUPER_AGENT_OVERLAY_REQUIRED_FILE_MISSING:$required"}}
  foreach($entry in $manifest.files){$relative=[string]$entry.path;if([IO.Path]::IsPathRooted($relative)-or(($relative-split'/')-contains'..')){throw "SUPER_AGENT_OVERLAY_PATH_UNSAFE:$relative"};$path=Join-Path $Root $relative.Replace('/','\');if(-not(Test-Path -LiteralPath $path -PathType Leaf)-or[long](Get-Item -LiteralPath $path).Length-ne[long]$entry.bytes-or(Sha $path)-ne[string]$entry.sha256){throw "SUPER_AGENT_OVERLAY_HASH_REJECTED:$relative"}}
  $actual=@(Get-ChildItem -LiteralPath $Root -Recurse -File -Force|ForEach-Object{$_.FullName.Substring($Root.Length+1).Replace('\','/')})
  $extras=@($actual|Where-Object{$_-ne'operational-overlay-manifest.json'-and$expected-notcontains$_})
  if($actual.Count-ne($expected.Count+1)-or$extras.Count){throw 'SUPER_AGENT_OVERLAY_EXTRAS_REJECTED'}
  $runtime=Get-Content -LiteralPath (Join-Path $Root 'runtime-build-manifest.json') -Raw|ConvertFrom-Json
  if([string]$runtime.build_id-ne$ExpectedBuildId-or[string]$runtime.source_tree_sha256-ne[string]$source.source_tree_sha256){throw 'SUPER_AGENT_RUNTIME_BUILD_IDENTITY_REJECTED'}
  $competition=Join-Path $Root 'support\python-runtime\evomind_runtime\competition_data.py'
  if((Sha $competition)-ne$ExpectedCompetitionDataSha){throw 'SUPER_AGENT_COMPETITION_DATA_DRIFT'}
  [ordered]@{manifest=$manifest;runtime=$runtime;files=$expected.Count;actual=$actual.Count;extras=0}
}
function Assert-CompiledAssistant([string]$Root){
  $server=Join-Path $Root '.next\server'
  $patterns=@(
    'app\api\assistant\runs\route.js',
    'app\api\assistant\runs\[runId]\route.js',
    'app\api\assistant\runs\[runId]\events\route.js',
    'app\api\assistant\runs\[runId]\actions\route.js',
    'app\api\assistant\run-secrets\intent\route.js',
    'app\api\assistant\run-secrets\envelope\route.js',
    'app\api\assistant\uploads\route.js',
    'app\api\assistant\uploads\[uploadId]\complete\route.js',
    'app\api\assistant\artifacts\[artifactId]\route.js',
    'app\api\assistant\approvals\[approvalId]\route.js',
    'app\api\hpc\byoa\binding\route.js',
    'app\api\hpc\byoa\public-key\route.js',
    'app\api\hpc\byoa\enrollment-identity\route.js',
    'app\api\hpc\byoa\enrollment-envelope\route.js'
    'app\api\super-agent\status\route.js'
  )
  foreach($relative in $patterns){if(-not(Test-Path -LiteralPath (Join-Path $server $relative) -PathType Leaf)){throw "SUPER_AGENT_COMPILED_ROUTE_MISSING:$relative"}}
  $compiled=@(Get-ChildItem -LiteralPath (Join-Path $Root '.next') -Recurse -File -Filter '*.js')
  $markers=@('data-ui-assistant-upload-list','data-ui-assistant-upload-progress','data-ui-assistant-controlled-credentials','upload.onprogress','artifact_published','Served model observed','持久任务 Agent','受控下载凭据','tenant-hpc-quick-enrollment','AIMSLAB 60 秒快速换绑','RSA-OAEP-256+A256GCM','super-agent-v1-panel')
  $corpus=($compiled|ForEach-Object{Get-Content -LiteralPath $_.FullName -Raw -ErrorAction SilentlyContinue})-join"`n"
  foreach($marker in $markers){if(-not$corpus.Contains($marker)){throw "SUPER_AGENT_COMPILED_MARKER_MISSING:$marker"}}
  [ordered]@{routes=$patterns.Count;markers=$markers.Count}
}
function Invoke-ProcessSmoke([string]$Root,[object]$Runtime,[int]$Round){
  $fixture=Join-Path ([IO.Path]::GetFullPath($VerificationBasePath)) ('web-smoke-'+[guid]::NewGuid().ToString('N'))
  $runtimePort=Free-Port;do{$webPort=Free-Port}while($webPort-eq$runtimePort)
  $runtimeProcess=$null;$webProcess=$null
  $environment=@('PYTHONPATH','DATABASE_URL','WORKSTATION_ROOT','WORKSTATION_DATA_DIR','EVOMIND_RUNTIME_BUILD_MANIFEST','EVOMIND_SOURCE_TREE_SHA256','EVOMIND_BUILD_COMMIT_HASH','EVOMIND_FRONTEND_VERSION','EVOMIND_BACKEND_VERSION','EVOMIND_RUNTIME_PORT','HOSTNAME','PORT')
  $saved=@{};foreach($name in $environment){$saved[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
  try{
    New-Item -ItemType Directory -Path $fixture -Force|Out-Null
    $database=Join-Path $fixture 'workstation.db';Copy-Item -LiteralPath ([IO.Path]::GetFullPath($BaselineDatabase)) -Destination $database
    $env:PYTHONPATH=(Join-Path $Root 'support\python-runtime')
    $env:WORKSTATION_ROOT=$fixture;$env:WORKSTATION_DATA_DIR=$fixture;$env:EVOMIND_RUNTIME_PORT=[string]$runtimePort
    $env:EVOMIND_RUNTIME_BUILD_MANIFEST=Join-Path $Root 'runtime-build-manifest.json';$env:EVOMIND_SOURCE_TREE_SHA256=[string]$Runtime.source_tree_sha256;$env:EVOMIND_BUILD_COMMIT_HASH=[string]$Runtime.commit_hash;$env:EVOMIND_FRONTEND_VERSION=[string]$Runtime.frontend_version;$env:EVOMIND_BACKEND_VERSION=[string]$Runtime.backend_version
    $env:DATABASE_URL='file:'+$database.Replace('\','/');$env:HOSTNAME='127.0.0.1';$env:PORT=[string]$webPort
    $runtimeProcess=Start-Process -FilePath (Get-Command python.exe).Source -ArgumentList @('-m','evomind_runtime.http_server','--workspace',$fixture,'--host','127.0.0.1','--port',[string]$runtimePort) -WorkingDirectory $fixture -WindowStyle Hidden -PassThru
    $tokenPath=Join-Path $fixture 'workspace\runtime\runtime.token';$deadline=[DateTime]::UtcNow.AddSeconds(30);do{Start-Sleep -Milliseconds 200}while(-not(Test-Path -LiteralPath $tokenPath)-and[DateTime]::UtcNow-lt$deadline-and-not$runtimeProcess.HasExited)
    if(-not(Test-Path -LiteralPath $tokenPath)){throw "SUPER_AGENT_RUNTIME_TOKEN_MISSING_ROUND_$Round"}
    $headers=@{Authorization=('Bearer '+(Get-Content -LiteralPath $tokenPath -Raw).Trim())}
    $tools=(Invoke-RestMethod "http://127.0.0.1:$runtimePort/v1/tools" -Headers $headers -TimeoutSec 10).tools
    foreach($tool in @('training_route','hpc_verify','hpc_execute_solution','competition_data_prepare','competition_data_status','attachment_read','artifact_publish','super_agent_status','capability_discover','directory_read','directory_write_atomic','transfer_fetch','job_execute','tool_synthesize','tool_promote')){if(@($tools|Where-Object{[string]$_.name-eq$tool}).Count-ne1){throw "SUPER_AGENT_RUNTIME_TOOL_MISSING:$tool"}}
    $superAgent=Invoke-RestMethod "http://127.0.0.1:$runtimePort/v1/super-agent/status" -Headers $headers -TimeoutSec 10
    if([string]$superAgent.schema-ne'evomind.super_agent_runtime.v1'-or[string]$superAgent.mode-ne'shadow'-or-not[bool]$superAgent.shadow-or[bool]$superAgent.enabled-or-not[bool]$superAgent.migration_applied-or[int]$superAgent.directory_count-lt1){throw "SUPER_AGENT_V1_STATUS_REJECTED_ROUND_$Round"}
    $session=Invoke-RestMethod "http://127.0.0.1:$runtimePort/v1/sessions" -Method Post -Headers ($headers+@{'Content-Type'='application/json'}) -Body (([ordered]@{objective='neutral route smoke';workspace_root=$fixture}|ConvertTo-Json -Compress)) -TimeoutSec 10
    $csv=Join-Path $fixture 'complex.csv';$writer=[IO.StreamWriter]::new($csv,$false,[Text.UTF8Encoding]::new($false));try{$writer.WriteLine((@((0..19|ForEach-Object{"f$_"}))+'target')-join',');$row=(@((0..19|ForEach-Object{'0.1'}))+'1')-join',';foreach($i in 1..20000){$writer.WriteLine($row)}}finally{$writer.Dispose()}
    $routeBody=[ordered]@{tool_name='training_route';arguments=[ordered]@{path=$csv;task_description='读取数据并自动选择合适算力完成分类训练';target_column='target'}}|ConvertTo-Json -Depth 6 -Compress
    $route=Invoke-RestMethod "http://127.0.0.1:$runtimePort/v1/sessions/$($session.id)/tools" -Method Post -Headers ($headers+@{'Content-Type'='application/json'}) -Body $routeBody -TimeoutSec 30
    $content=$route.result.content;if([string]$content.decision-ne'hpc_gpu'-or@($content.reasons|Where-Object{[string]$_.code-eq'row_threshold'}).Count-ne1-or[bool]$content.local_gpu_allowed){throw "SUPER_AGENT_RUNTIME_ROUTE_REJECTED_ROUND_$Round"}
    $webProcess=Start-Process -FilePath (Get-Command node.exe).Source -ArgumentList @('server.js') -WorkingDirectory $Root -WindowStyle Hidden -PassThru
    $deadline=[DateTime]::UtcNow.AddSeconds(60);$response=$null;$payload=$null;do{Start-Sleep -Milliseconds 250;try{$response=Invoke-WebRequest "http://127.0.0.1:$webPort/api/healthz" -UseBasicParsing -TimeoutSec 8;$payload=$response.Content|ConvertFrom-Json}catch{$response=$null;$payload=$null}}while($null-eq$response-and[DateTime]::UtcNow-lt$deadline-and-not$webProcess.HasExited)
    if($null-eq$response-or[int]$response.StatusCode-ne200-or-not[bool]$payload.ok-or[string]$payload.status-ne'ready'-or[string]$payload.build_id-ne$ExpectedBuildId-or@($payload.failures).Count){throw "SUPER_AGENT_WEB_HEALTH_REJECTED_ROUND_$Round"}
    [ordered]@{round=$Round;status='passed';web_http=200;build_id=[string]$payload.build_id;runtime_tool_count=@($tools).Count;autonomous_route=[string]$content.decision;route_reason='row_threshold';web_pid=[int]$webProcess.Id;runtime_pid=[int]$runtimeProcess.Id;network_accessed=$false;gpu_touched=$false}
  }finally{
    foreach($process in @($webProcess,$runtimeProcess)){if($process-and-not$process.HasExited){Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue;try{$process.WaitForExit()}catch{}}}
    foreach($name in $environment){if($null-eq$saved[$name]){Remove-Item "Env:$name" -ErrorAction SilentlyContinue}else{[Environment]::SetEnvironmentVariable($name,[string]$saved[$name],'Process')}}
    if(Test-Path -LiteralPath $fixture){Remove-Item -LiteralPath $fixture -Recurse -Force}
  }
}

$root=[IO.Path]::GetFullPath($OverlayRoot).TrimEnd('\');$zip=[IO.Path]::GetFullPath($OutputZip);$resultPath=[IO.Path]::GetFullPath($TestResultPath)
foreach($path in @($root,[IO.Path]::GetFullPath($BaselineDatabase),[IO.Path]::GetFullPath($ExpectedSourceManifestPath))){if(-not(Test-Path -LiteralPath $path)){throw "SUPER_AGENT_WEB_INPUT_MISSING:$path"}}
foreach($path in @($zip,$resultPath)){if(Test-Path -LiteralPath $path){throw 'SUPER_AGENT_WEB_OUTPUT_EXISTS'}}
$check=Assert-Overlay $root;$compiled=Assert-CompiledAssistant $root
[IO.Compression.ZipFile]::CreateFromDirectory($root,$zip,[IO.Compression.CompressionLevel]::Optimal,$false)
$rounds=@();$targets=@()
try{
  foreach($round in 1..2){$target=Join-Path ([IO.Path]::GetFullPath($VerificationBasePath)) ('web-'+[guid]::NewGuid().ToString('N'));$targets+=$target;New-Item -ItemType Directory -Path $target|Out-Null;[IO.Compression.ZipFile]::ExtractToDirectory($zip,$target);if(@($target.ToCharArray()|Where-Object{[int]$_-gt127}).Count){throw "SUPER_AGENT_WEB_ASCII_PATH_REJECTED_ROUND_$round"};$identity=Assert-Overlay $target;$compileCheck=Assert-CompiledAssistant $target;$smoke=Invoke-ProcessSmoke $target $identity.runtime $round;$rounds+=[ordered]@{round=$round;status='passed';path_is_ascii=$true;files=$identity.files;actual=$identity.actual;extras=0;compiled=$compileCheck;process_smoke=$smoke}}
  $result=[ordered]@{schema='evomind.super_agent_web_release.v1';status='built_and_final_zip_verified';completed_at_utc=[DateTime]::UtcNow.ToString('o');path=$zip;bytes=[long](Get-Item -LiteralPath $zip).Length;sha256=Sha $zip;build_id=$ExpectedBuildId;source_tree_sha256=[string]$check.runtime.source_tree_sha256;source_manifest_sha256=Sha ([IO.Path]::GetFullPath($ExpectedSourceManifestPath));file_count=$check.files;compiled=$compiled;verification_rounds=$rounds;production_deployed=$false;gpu_touched=$false};[IO.File]::WriteAllText($resultPath,(($result|ConvertTo-Json -Depth 10)+"`n"),[Text.UTF8Encoding]::new($false));$result|ConvertTo-Json -Depth 10 -Compress
}catch{Remove-Item -LiteralPath $zip,$resultPath -Force -ErrorAction SilentlyContinue;throw}finally{foreach($target in $targets){if(Test-Path -LiteralPath $target){Remove-Item -LiteralPath $target -Recurse -Force}}}
