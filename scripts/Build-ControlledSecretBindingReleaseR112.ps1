param(
  [Parameter(Mandatory)][ValidateSet('DryRun','Build')][string]$Mode,
  [string]$BaselineWebRoot = 'D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\web-overlay-controlled-secret-attach-r94',
  [string]$BaselineRuntimeZip = 'D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\evomind-runtime-r111-weather-skip-timeout-gated-20260822.zip',
  [string]$SourceRoot = 'D:\桌面\codex\科研港科技',
  [string]$VerifyRoot = 'D:\EV12',
  [string]$OutputRoot = 'D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts',
  [string]$BaselineDatabase = 'D:\桌面\codex\科研港科技\web\research-agent-workstation\prisma\workstation.db',
  [string]$SmokePython = 'C:\codex-python\python.exe'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

$oldEvidenceZip = 'D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\overlay-controlled-secret-binding-r112.zip'
$oldEvidenceSha256 = '10e59a178ac4594d5157914f379af08883eae8c347fc2df605c9d8de110c2ff8'
$expectedRuntimeZipSha256 = 'ff35e6235c9fa425f26a3b826a36a936f519a3969ac9becc66d204ca06ced235'

function Sha([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Text-Sha([string]$Value) {
  $hash = [Security.Cryptography.SHA256]::Create()
  try { ([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($Value)))).Replace('-','').ToLowerInvariant() }
  finally { $hash.Dispose() }
}
function Write-Json([string]$Path, $Value) {
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
  [IO.File]::WriteAllText($Path,(($Value | ConvertTo-Json -Depth 20)+"`n"),[Text.UTF8Encoding]::new($false))
}
function Read-Json([string]$Path) { Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json }
function Assert-Relative([string]$Relative) {
  if (-not $Relative -or [IO.Path]::IsPathRooted($Relative) -or (($Relative -split '/') -contains '..') -or $Relative.Contains('\')) { throw "R112_PATH_UNSAFE:$Relative" }
}
function Copy-DirectoryContents([string]$From, [string]$To) {
  if (-not (Test-Path -LiteralPath $From -PathType Container)) { throw "R112_COPY_SOURCE_MISSING:$From" }
  New-Item -ItemType Directory -Force -Path $To | Out-Null
  Get-ChildItem -LiteralPath $From -Force | Copy-Item -Destination $To -Recurse -Force -ProgressAction SilentlyContinue
}
function Regular-Files([string]$Root, [string[]]$Exclude = @()) {
  $full = [IO.Path]::GetFullPath($Root).TrimEnd('\')
  $excluded = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
  foreach ($value in $Exclude) { [void]$excluded.Add($value) }
  @(Get-ChildItem -LiteralPath $full -Recurse -File -Force | Where-Object {
    $_.Name -notmatch '\.(pyc|pyo)$' -and $_.FullName -notmatch '[\\/]__pycache__[\\/]'
  } | ForEach-Object {
    $relative = $_.FullName.Substring($full.Length + 1).Replace('\','/')
    if (-not $excluded.Contains($relative)) { [ordered]@{ path=$relative; bytes=[long]$_.Length; sha256=(Sha $_.FullName) } }
  } | Where-Object { $null -ne $_ } | Sort-Object path)
}
function File-Map($Entries) {
  $map = @{}
  foreach ($entry in @($Entries)) {
    $path = [string]$entry.path; Assert-Relative $path
    if ($map.ContainsKey($path)) { throw "R112_DUPLICATE_MANIFEST_PATH:$path" }
    $map[$path] = $entry
  }
  $map
}
function Verify-RegularManifest([string]$Root, $Manifest, [string]$ManifestName = 'operational-overlay-manifest.json') {
  $expected = File-Map $Manifest.files
  if ($expected.Count -ne [int]$Manifest.file_count) { throw 'R112_MANIFEST_COUNT_REJECTED' }
  $actualRows = Regular-Files $Root @($ManifestName)
  $actual = File-Map $actualRows
  $missing = @($expected.Keys | Where-Object { -not $actual.ContainsKey($_) })
  $extra = @($actual.Keys | Where-Object { -not $expected.ContainsKey($_) })
  $mismatch = @($expected.Keys | Where-Object {
    $actual.ContainsKey($_) -and ([long]$actual[$_].bytes -ne [long]$expected[$_].bytes -or [string]$actual[$_].sha256 -ne [string]$expected[$_].sha256)
  })
  [ordered]@{ regular_files=$actual.Count; manifest_files=$expected.Count; missing=$missing; file_extra=$extra; mismatch=$mismatch }
}
function Verify-Patch([string]$Path, [string]$ExpectedSha, [int64]$ExpectedBytes, [string]$Relative) {
  Assert-Relative $Relative
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "R112_PATCH_MISSING:$Relative" }
  $item = Get-Item -LiteralPath $Path
  if ([int64]$item.Length -ne $ExpectedBytes -or (Sha $Path) -ne $ExpectedSha) { throw "R112_PATCH_HASH_MISMATCH:$Relative" }
  [ordered]@{ path=$Relative; bytes=[int64]$item.Length; sha256=(Sha $Path) }
}
function Read-RuntimeBaseline([string]$ZipPath) {
  if ((Sha $ZipPath) -ne $expectedRuntimeZipSha256) { throw 'R112_RUNTIME_ZIP_SHA_REJECTED' }
  $zip = [IO.Compression.ZipFile]::OpenRead($ZipPath)
  try {
    $manifestEntry = $zip.GetEntry('runtime-hotfix-manifest.json')
    if (-not $manifestEntry) { throw 'R112_RUNTIME_MANIFEST_MISSING' }
    $reader = [IO.StreamReader]::new($manifestEntry.Open())
    try { $manifest = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
    if ([string]$manifest.schema -ne 'evomind.super_agent_runtime_hotfix.v1' -or [int]$manifest.file_count -ne 30 -or @($manifest.files).Count -ne 30) { throw 'R112_RUNTIME_MANIFEST_REJECTED' }
    $mismatch = @()
    foreach ($entry in @($manifest.files)) {
      $name = [string]$entry.path; Assert-Relative $name
      $item = $zip.GetEntry($name)
      if (-not $item) { $mismatch += $name; continue }
      $stream = $item.Open(); $memory = [IO.MemoryStream]::new()
      try { $stream.CopyTo($memory); $bytes=$memory.ToArray() } finally { $memory.Dispose(); $stream.Dispose() }
      $hash=[Security.Cryptography.SHA256]::HashData($bytes);$digest=([BitConverter]::ToString($hash)).Replace('-','').ToLowerInvariant()
      if ([long]$bytes.Length -ne [long]$entry.bytes -or $digest -ne [string]$entry.sha256) { $mismatch += $name }
    }
    if ($mismatch.Count) { throw "R112_RUNTIME_BASELINE_MISMATCH:$($mismatch[0])" }
    $map = File-Map $manifest.files
    foreach ($expected in @(
      @{path='evomind_runtime/competition_data.py';sha='b87e0229e935e2f9123dcebe3ee4f902141fa01f82475b8ec7ebcbe72338f5af'},
      @{path='evomind_runtime/store.py';sha='2313d0e181338061bf2f27a20e06695dc8696ebd2b0f1e92aa87942315087292'},
      @{path='evomind_runtime/tools.py';sha='51cc1a53d79576e8fc910b343183b9102e65025785bbd6d9fa1f24add59560bb'}
    )) { if (-not $map.ContainsKey($expected.path) -or [string]$map[$expected.path].sha256 -ne $expected.sha) { throw "R112_RUNTIME_RETAINED_HASH_REJECTED:$($expected.path)" } }
    [ordered]@{ manifest=$manifest; entries=@($manifest.files); file_count=30; zip_sha256=(Sha $ZipPath); mismatch_count=0 }
  } finally { $zip.Dispose() }
}
function Test-BuildCacheRelative([string]$Relative) {
  $normalized = $Relative.Replace('\','/').TrimStart('/')
  return (
    $normalized -match '^\.next/(cache|dev)(/|$)' -or
    $normalized -match '(^|/)node_modules/\.cache(/|$)' -or
    $normalized -match '(^|/)(\.webpack-cache|webpack-cache)(/|$)' -or
    $normalized -match '^build-data(/|$)'
  )
}
function Root-Layout([string]$Root) {
  $full = [IO.Path]::GetFullPath($Root).TrimEnd('\')
  $cache = @(Get-ChildItem -LiteralPath $full -Recurse -File -Force | Where-Object {
    $relative = $_.FullName.Substring($full.Length + 1).Replace('\','/')
    Test-BuildCacheRelative $relative
  })
  $samples = @($cache | Select-Object -First 12 | ForEach-Object { $_.FullName.Substring($full.Length + 1).Replace('\','/') })
  [ordered]@{ root_server=(Test-Path -LiteralPath (Join-Path $Root 'server.js') -PathType Leaf); root_next=(Test-Path -LiteralPath (Join-Path $Root 'node_modules\next\package.json') -PathType Leaf); nested_standalone=(Test-Path -LiteralPath (Join-Path $Root '.next\standalone')); cache_files=$cache.Count; cache_samples=$samples }
}
function Remove-BuildCacheDirectories([string]$Root) {
  $full = [IO.Path]::GetFullPath($Root).TrimEnd('\')
  $prefix = $full + '\'
  $targets = @(Get-ChildItem -LiteralPath $full -Recurse -Directory -Force | Where-Object {
    $relative = $_.FullName.Substring($full.Length + 1).Replace('\','/')
    Test-BuildCacheRelative $relative
  } | Sort-Object { $_.FullName.Length } -Descending)
  foreach ($target in $targets) {
    $resolved = [IO.Path]::GetFullPath($target.FullName)
    if (-not $resolved.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw "R112_CACHE_PATH_ESCAPE:$resolved" }
    Remove-Item -LiteralPath $resolved -Recurse -Force
  }
  [ordered]@{ removed_directories=$targets.Count }
}
function Assert-ReleaseRoot([string]$Root, [string]$BuildId) {
  $layout = Root-Layout $Root
  if (-not $layout.root_server -or -not $layout.root_next -or $layout.nested_standalone -or $layout.cache_files -ne 0) {
    throw "R112_RELEASE_ROOT_CLOSURE_REJECTED:root_server=$($layout.root_server):root_next=$($layout.root_next):nested_standalone=$($layout.nested_standalone):cache_files=$($layout.cache_files):cache_samples=$($layout.cache_samples -join ',')"
  }
  foreach ($required in @('.next\static','public','src','support\python-runtime\evomind_runtime','support\python-runtime\xsci')) { if (-not (Test-Path -LiteralPath (Join-Path $Root $required) -PathType Container)) { throw "R112_RELEASE_REQUIRED_DIR_MISSING:$required" } }
  $runtime = Read-Json (Join-Path $Root 'runtime-build-manifest.json');$manifest = Read-Json (Join-Path $Root 'operational-overlay-manifest.json')
  if ([string]$runtime.build_id -ne $BuildId -or [string]$manifest.overlay_id -ne $BuildId) { throw 'R112_RELEASE_BUILD_ID_REJECTED' }
  $files = Verify-RegularManifest $Root $manifest
  if ($files.missing.Count -or $files.file_extra.Count -or $files.mismatch.Count) { throw 'R112_RELEASE_MANIFEST_FILES_REJECTED' }
  [ordered]@{ layout=$layout; files=$files; runtime=$runtime; manifest=$manifest }
}
function Test-ZipSafety([string]$ZipPath) {
  $zip = [IO.Compression.ZipFile]::OpenRead($ZipPath)
  try {
    $seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase);$unsafe=0;$duplicates=0;$symlinks=0;$regular=0;$directories=0
    foreach ($entry in $zip.Entries) {
      $name=[string]$entry.FullName;if(-not$seen.Add($name)){$duplicates++};if([IO.Path]::IsPathRooted($name)-or(($name-split'/')-contains'..')-or$name.Contains('\')){$unsafe++}
      if((($entry.ExternalAttributes-shr16)-band0xF000)-eq0xA000){$symlinks++};if($name.EndsWith('/')){$directories++}else{$regular++}
    }
    [ordered]@{unsafe_entries=$unsafe;duplicate_entries=$duplicates;symlink_entries=$symlinks;regular_files=$regular;directory_entries=$directories}
  } finally { $zip.Dispose() }
}
function Free-Port { $listener=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0);try{$listener.Start();[int]$listener.LocalEndpoint.Port}finally{$listener.Stop()} }
function Wait-Json([string]$Uri,[int]$TimeoutSeconds=60,[hashtable]$Headers=@{}) {
  $deadline=[DateTime]::UtcNow.AddSeconds($TimeoutSeconds);$last='none'
  do {
    try {
      $response=Invoke-WebRequest -Uri $Uri -Headers $Headers -UseBasicParsing -TimeoutSec 8 -SkipHttpErrorCheck -ErrorAction Stop
      if([int]$response.StatusCode -gt 0){return $response}
    } catch {
      $message=[string]$_.Exception.Message
      if($message.Length -gt 120){$message=$message.Substring(0,120)}
      $last="$($_.Exception.GetType().Name):$message"
    }
    Start-Sleep -Milliseconds 250
  } while([DateTime]::UtcNow-lt$deadline)
  throw "R112_HTTP_TIMEOUT:$last"
}
function Start-ReleaseSmoke([string]$ReleaseRoot,[string]$BuildId,[string]$SourceIdentity,[int]$Round,[string]$VerifyBase,[string]$Database,[string]$PythonExecutable) {
  $fixture=Join-Path $VerifyBase ("r112-smoke-$Round-"+[guid]::NewGuid().ToString('N'));$runtimePort=Free-Port;do{$webPort=Free-Port}while($webPort-eq$runtimePort)
  $runtimeProcess=$null;$webProcess=$null;$runtimeOut=Join-Path $fixture 'runtime.out.log';$runtimeErr=Join-Path $fixture 'runtime.err.log';$webOut=Join-Path $fixture 'web.out.log';$webErr=Join-Path $fixture 'web.err.log'
  $names=@('PYTHONPATH','DATABASE_URL','WORKSTATION_ROOT','WORKSTATION_DATA_DIR','EVOMIND_RUNTIME_BUILD_MANIFEST','EVOMIND_SOURCE_TREE_SHA256','EVOMIND_BUILD_COMMIT_HASH','EVOMIND_FRONTEND_VERSION','EVOMIND_BACKEND_VERSION','EVOMIND_RUNTIME_PORT','HOSTNAME','PORT','NODE_ENV','NEXT_TELEMETRY_DISABLED','WORKSTATION_SESSION_SECRET');$saved=@{};foreach($name in $names){$saved[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
  try{
    New-Item -ItemType Directory -Force -Path $fixture|Out-Null;$databaseCopy=Join-Path $fixture 'workstation.db';Copy-Item -LiteralPath $Database -Destination $databaseCopy -Force
    $env:PYTHONPATH=(Join-Path $ReleaseRoot 'support\python-runtime')+';'+(Join-Path $ReleaseRoot 'support\python-vendor');$env:DATABASE_URL='file:'+$databaseCopy.Replace('\','/');$env:WORKSTATION_ROOT=$fixture;$env:WORKSTATION_DATA_DIR=$fixture;$env:EVOMIND_RUNTIME_PORT=[string]$runtimePort
    $env:EVOMIND_RUNTIME_BUILD_MANIFEST=Join-Path $ReleaseRoot 'runtime-build-manifest.json';$env:EVOMIND_SOURCE_TREE_SHA256=$SourceIdentity;$env:EVOMIND_BUILD_COMMIT_HASH='664a636ddd419a66f73cc10820c0a16429784866';$env:EVOMIND_FRONTEND_VERSION='0.3.0';$env:EVOMIND_BACKEND_VERSION='0.3.0';$env:HOSTNAME='127.0.0.1';$env:PORT=[string]$webPort;$env:NODE_ENV='production';$env:NEXT_TELEMETRY_DISABLED='1';$env:WORKSTATION_SESSION_SECRET=[Convert]::ToBase64String([Security.Cryptography.RandomNumberGenerator]::GetBytes(48))
    $python=[IO.Path]::GetFullPath($PythonExecutable)
    if(-not(Test-Path -LiteralPath $python -PathType Leaf)){throw "R112_SMOKE_PYTHON_MISSING:$python"}
    $node=(Get-Command node.exe -ErrorAction Stop).Source
    $runtimeProcess=Start-Process -FilePath $python -ArgumentList @('-m','evomind_runtime.http_server','--workspace',$fixture,'--host','127.0.0.1','--port',[string]$runtimePort) -WorkingDirectory $fixture -WindowStyle Hidden -PassThru -RedirectStandardOutput $runtimeOut -RedirectStandardError $runtimeErr
    $tokenPath=Join-Path $fixture 'workspace\runtime\runtime.token';$deadline=[DateTime]::UtcNow.AddSeconds(40);do{Start-Sleep -Milliseconds 200}while(-not(Test-Path -LiteralPath $tokenPath)-and[DateTime]::UtcNow-lt$deadline-and-not$runtimeProcess.HasExited)
    $runtimeProcess.Refresh()
    if(-not(Test-Path -LiteralPath $tokenPath)-or$runtimeProcess.HasExited){
      $stderrBytes=if(Test-Path -LiteralPath $runtimeErr){[IO.File]::ReadAllBytes($runtimeErr)}else{[byte[]]@()}
      $stderrSha=if($stderrBytes.Length){([BitConverter]::ToString([Security.Cryptography.SHA256]::HashData($stderrBytes))).Replace('-','').ToLowerInvariant()}else{'none'}
      $exitCode=if($runtimeProcess.HasExited){$runtimeProcess.ExitCode}else{-1}
      throw "R112_RUNTIME_START_FAILED:${Round}:token=$([bool](Test-Path -LiteralPath $tokenPath)):exited=$($runtimeProcess.HasExited):exit_code=${exitCode}:stderr_length=$($stderrBytes.Length):stderr_sha256=$stderrSha"
    };$headers=@{Authorization='Bearer '+(Get-Content -LiteralPath $tokenPath -Raw).Trim()};$runtimeHealth=Wait-Json "http://127.0.0.1:$runtimePort/v1/health" 30 $headers;if([int]$runtimeHealth.StatusCode-ne200){throw "R112_RUNTIME_HEALTH_REJECTED:$Round"}
    $webProcess=Start-Process -FilePath $node -ArgumentList @('server.js') -WorkingDirectory $ReleaseRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $webOut -RedirectStandardError $webErr;$health=Wait-Json "http://127.0.0.1:$webPort/api/healthz" 60;$healthPayload=$health.Content|ConvertFrom-Json
    if([int]$health.StatusCode-ne200-or-not[bool]$healthPayload.ok-or[string]$healthPayload.status-ne'ready'-or[string]$healthPayload.build_id-ne$BuildId-or@($healthPayload.failures).Count){throw "R112_WEB_HEALTH_REJECTED:$Round"}
    $login=Wait-Json "http://127.0.0.1:$webPort/login" 30;if([int]$login.StatusCode-ne200-or-not([string]$login.Content).Contains('DeepEvo')){throw "R112_LOGIN_SMOKE_REJECTED:$Round"};$assistant=Invoke-WebRequest "http://127.0.0.1:$webPort/api/assistant/runs" -UseBasicParsing -TimeoutSec 15 -SkipHttpErrorCheck;if([int]$assistant.StatusCode-notin@(401,403)){throw "R112_ASSISTANT_API_GUARD_REJECTED:$Round"}
    if($webProcess.HasExited-or$runtimeProcess.HasExited){throw "R112_PROCESS_EXITED_DURING_SMOKE:$Round"};Start-Sleep -Milliseconds 300;$stderr=((Get-Content -LiteralPath $runtimeErr -Raw -ErrorAction SilentlyContinue)+"`n"+(Get-Content -LiteralPath $webErr -Raw -ErrorAction SilentlyContinue));$bad=@(@('MODULE_NOT_FOUND','Cannot find module','PrismaClientInitializationError','Cannot find package.*prisma','Cannot find.*sharp','Could not load.*sharp')|Where-Object{$stderr-match$_});if($bad.Count){throw "R112_STARTUP_STDERR_REJECTED:$($bad[0])"}
    [ordered]@{round=$Round;status='passed';runtime_health=200;web_health=200;login_page=200;assistant_api_guard=[int]$assistant.StatusCode;build_id=[string]$healthPayload.build_id;web_alive=$true;runtime_alive=$true;missing_module_errors=0;prisma_errors=0;sharp_errors=0}
  }finally{foreach($process in @($webProcess,$runtimeProcess)){if($process-and-not$process.HasExited){Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue;try{$process.WaitForExit()}catch{}}};foreach($name in $names){if($null-eq$saved[$name]){Remove-Item "Env:$name" -ErrorAction SilentlyContinue}else{[Environment]::SetEnvironmentVariable($name,[string]$saved[$name],'Process')}};Remove-Item -LiteralPath $fixture -Recurse -Force -ErrorAction SilentlyContinue}
}

$baseline=[IO.Path]::GetFullPath($BaselineWebRoot).TrimEnd('\');$runtimeZip=[IO.Path]::GetFullPath($BaselineRuntimeZip);$source=[IO.Path]::GetFullPath($SourceRoot).TrimEnd('\');$verify=[IO.Path]::GetFullPath($VerifyRoot).TrimEnd('\');$output=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
foreach($path in @($baseline,$runtimeZip,$source,$verify,$output,$BaselineDatabase,$SmokePython,$oldEvidenceZip)){if(-not(Test-Path -LiteralPath $path)){throw "R112_INPUT_MISSING:$path"}};if((Sha $oldEvidenceZip)-ne$oldEvidenceSha256){throw 'R112_OLD_EVIDENCE_SHA_REJECTED'}
$overlay=Read-Json (Join-Path $baseline 'operational-overlay-manifest.json');$runtimeManifestBaseline=Read-Json (Join-Path $baseline 'runtime-build-manifest.json');if([string]$overlay.schema-ne'evomind.web_operational_overlay.v1'-or[string]$runtimeManifestBaseline.schema-ne'evomind.runtime_build.v1'){throw 'R112_BASELINE_SCHEMA_REJECTED'};$runtimeInfo=Read-RuntimeBaseline $runtimeZip
$webPatch=@(
  [ordered]@{relative='src/components/workstation/screens/AssistantControlledCredentials.tsx';source=Join-Path $source 'web\research-agent-workstation\src\components\workstation\screens\AssistantControlledCredentials.tsx';bytes=13600;sha256='496617aafd56cdf6eea89fe6fbb807320f8a1ea926737552a0e2cfdf5b8e36ec'},
  [ordered]@{relative='src/components/workstation/screens/AssistantScreen.tsx';source=Join-Path $source 'web\research-agent-workstation\src\components\workstation\screens\AssistantScreen.tsx';bytes=90498;sha256='ac48f5dc170606b4ee0907a19e5c9757c246461ce9f926cdb8375a1be0eebac2'},
  [ordered]@{relative='src/components/workstation/screens/controlled-secret-binding.ts';source=Join-Path $source 'web\research-agent-workstation\src\components\workstation\screens\controlled-secret-binding.ts';bytes=1557;sha256='9486e1941bb9e073018fababffb42795e5e6ab1c34fd61a7304906fc9402dca1'}
)
$pythonPatch=@(
  [ordered]@{relative='evomind_runtime/assistant_runs.py';source=Join-Path $source 'src\evomind_runtime\assistant_runs.py';bytes=100750;sha256='bf52a8fa66a5b15bf06594cfc9714cd5597610048902b2facfc1180c9555a72b'},
  [ordered]@{relative='evomind_runtime/run_secrets.py';source=Join-Path $source 'src\evomind_runtime\run_secrets.py';bytes=25480;sha256='0f4062dffd0828b90be6405b8b68dbe89389165e7e925438badaf49acee84c81'}
)
$verifiedWebPatch=@($webPatch|ForEach-Object{Verify-Patch $_.source $_.sha256 $_.bytes $_.relative});$verifiedPythonPatch=@($pythonPatch|ForEach-Object{Verify-Patch $_.source $_.sha256 $_.bytes $_.relative});$baselineFiles=Verify-RegularManifest $baseline $overlay;$planLayout=Root-Layout (Join-Path $baseline '.next\standalone')
$seed=([string]$overlay.source_identity_sha256)+"`n"+$runtimeInfo.zip_sha256+"`n"+(($verifiedWebPatch+$verifiedPythonPatch|ForEach-Object{"$($_.path)|$($_.bytes)|$($_.sha256)"})-join"`n");$sourceIdentity=Text-Sha $seed;$buildId="overlay-controlled-secret-binding-r112-standalone-$($sourceIdentity.Substring(0,12))"
$dry=[ordered]@{schema='evomind.controlled_secret_binding_release_dry_run.v2';status='passed';build_id=$buildId;root_server=[bool]$planLayout.root_server;root_next=[bool]$planLayout.root_next;nested_standalone=$false;cache_files=0;missing=$baselineFiles.missing.Count;file_extra=$baselineFiles.file_extra.Count;mismatch=$baselineFiles.mismatch.Count;allowlist_count=5;web_allowlist=$verifiedWebPatch;python_allowlist=$verifiedPythonPatch;r111_runtime_count=$runtimeInfo.file_count;r111_runtime_mismatch=$runtimeInfo.mismatch_count;old_evidence_sha256=$oldEvidenceSha256;old_evidence_reused=$false;hpc_accessed=$false;gpu_touched=$false;production_deployed=$false}
if(-not$dry.root_server-or-not$dry.root_next-or$dry.missing-or$dry.file_extra-or$dry.mismatch-or$dry.r111_runtime_mismatch){$dry.status='failed';$dry.failure='R112_DRY_RUN_REJECTED'};$dry|ConvertTo-Json -Depth 12;if($Mode-eq'DryRun'){if($dry.status-ne'passed'){throw$dry.failure};return};if($dry.status-ne'passed'){throw$dry.failure}

$buildRoot=Join-Path $verify ('r112-build-'+[guid]::NewGuid().ToString('N'));$runtimeRoot=Join-Path $verify ('r112-runtime-'+[guid]::NewGuid().ToString('N'));$releaseRoot=Join-Path $verify ('r112-release-'+[guid]::NewGuid().ToString('N'))
$zipPath=Join-Path $output 'overlay-controlled-secret-binding-r112-standalone-v5.zip';$resultPath=Join-Path $output 'overlay-controlled-secret-binding-r112-standalone-v5-build-result.json';$sourceManifestOutput=Join-Path $output 'overlay-controlled-secret-binding-r112-standalone-v7-source-manifest.json';foreach($path in @($zipPath,$resultPath,$sourceManifestOutput)){if(Test-Path -LiteralPath $path){throw "R112_OUTPUT_EXISTS:$path"}}
New-Item -ItemType Directory -Force -Path $buildRoot,$runtimeRoot,$releaseRoot|Out-Null;$buildNodeModules=Join-Path $buildRoot 'node_modules';$junctionCreated=$false
try{
  foreach($name in @('src','public','support','package.json','package-lock.json','next.config.mjs','postcss.config.mjs','tailwind.config.ts','tsconfig.json')){$origin=Join-Path $baseline $name;if(Test-Path -LiteralPath $origin -PathType Container){Copy-DirectoryContents $origin (Join-Path $buildRoot $name)}elseif(Test-Path -LiteralPath $origin -PathType Leaf){Copy-Item -LiteralPath $origin -Destination (Join-Path $buildRoot $name) -Force}else{throw "R112_BUILD_INPUT_MISSING:$name"}}
  foreach($patch in $webPatch){$destination=Join-Path $buildRoot $patch.relative.Replace('/','\');New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination)|Out-Null;Copy-Item -LiteralPath $patch.source -Destination $destination -Force}
  [IO.Compression.ZipFile]::ExtractToDirectory($runtimeZip,$runtimeRoot);$runtimeTarget=Join-Path $buildRoot 'support\python-runtime\evomind_runtime';Remove-Item -LiteralPath $runtimeTarget -Recurse -Force;Copy-DirectoryContents (Join-Path $runtimeRoot 'evomind_runtime') $runtimeTarget;foreach($patch in $pythonPatch){Copy-Item -LiteralPath $patch.source -Destination (Join-Path $runtimeTarget ([IO.Path]::GetFileName($patch.relative))) -Force}
  $verifiedNodeModules=Join-Path $source 'web\research-agent-workstation\node_modules';if(-not(Test-Path -LiteralPath (Join-Path $verifiedNodeModules 'next\dist\bin\next')-PathType Leaf)){throw 'R112_BUILD_TOOLCHAIN_MISSING'};New-Item -ItemType Junction -Path $buildNodeModules -Target $verifiedNodeModules|Out-Null;$junctionCreated=$true
  $buildData=Join-Path $buildRoot 'build-data';New-Item -ItemType Directory -Force -Path $buildData|Out-Null;$envNames=@('DATABASE_URL','WORKSTATION_ROOT','WORKSTATION_DATA_DIR','NEXT_TELEMETRY_DISABLED','NODE_PATH');$saved=@{};foreach($name in $envNames){$saved[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
  try{$env:DATABASE_URL='file:'+(Join-Path $buildData 'workstation.db').Replace('\','/');$env:WORKSTATION_ROOT=$buildData;$env:WORKSTATION_DATA_DIR=$buildData;$env:NEXT_TELEMETRY_DISABLED='1';$env:NODE_PATH=$verifiedNodeModules;Push-Location $buildRoot;try{& (Get-Command node.exe -ErrorAction Stop).Source (Join-Path $verifiedNodeModules 'next\dist\bin\next') build --webpack;if($LASTEXITCODE-ne0){throw 'R112_WEB_BUILD_FAILED'}}finally{Pop-Location}}finally{foreach($name in $envNames){if($null-eq$saved[$name]){Remove-Item "Env:$name" -ErrorAction SilentlyContinue}else{[Environment]::SetEnvironmentVariable($name,[string]$saved[$name],'Process')}}}
  if($junctionCreated){[IO.Directory]::Delete($buildNodeModules);$junctionCreated=$false};Remove-Item -LiteralPath $buildData -Recurse -Force -ErrorAction SilentlyContinue
  $standaloneRoot = Join-Path $buildRoot '.next\standalone'
  Copy-DirectoryContents $standaloneRoot $releaseRoot
  Copy-DirectoryContents (Join-Path $buildRoot '.next\static') (Join-Path $releaseRoot '.next\static')
  Copy-DirectoryContents (Join-Path $buildRoot 'public') (Join-Path $releaseRoot 'public')
  Copy-DirectoryContents (Join-Path $buildRoot 'support') (Join-Path $releaseRoot 'support')
  Copy-DirectoryContents (Join-Path $buildRoot 'src') (Join-Path $releaseRoot 'src')
  $cacheCleanup = Remove-BuildCacheDirectories $releaseRoot
  if(Test-Path -LiteralPath (Join-Path $releaseRoot '.next\standalone')){throw 'R112_NESTED_STANDALONE_REMAINS'}
  $sourceManifest=[ordered]@{schema='evomind.controlled_secret_binding_source_set.v1';created_at_utc=[DateTime]::UtcNow.ToString('o');frozen=$true;source_tree_sha256=$sourceIdentity;baseline_web_overlay_id=[string]$overlay.overlay_id;baseline_web_source_identity_sha256=[string]$overlay.source_identity_sha256;baseline_runtime_zip_sha256=$runtimeInfo.zip_sha256;allowlist_count=5;files=($verifiedWebPatch+$verifiedPythonPatch);r111_runtime_files=$runtimeInfo.entries;old_unverified_evidence=[ordered]@{sha256=$oldEvidenceSha256;reused=$false}}
  Write-Json (Join-Path $releaseRoot 'release-source-manifest.json') $sourceManifest
  $runtimeManifest=[ordered]@{schema='evomind.runtime_build.v1';commit_hash=[string]$runtimeManifestBaseline.commit_hash;source_dirty=$true;source_tree_sha256=$sourceIdentity;build_id=$buildId;build_time=[DateTime]::UtcNow.ToString('o');backend_version='0.3.0';frontend_version='0.3.0';database_schema_version=[string]$runtimeManifestBaseline.database_schema_version;database_schema_sha256=[string]$runtimeManifestBaseline.database_schema_sha256};Write-Json (Join-Path $releaseRoot 'runtime-build-manifest.json') $runtimeManifest
  $regular=Regular-Files $releaseRoot @('operational-overlay-manifest.json');$operational=[ordered]@{schema='evomind.web_operational_overlay.v1';overlay_id=$buildId;created_at_utc=[DateTime]::UtcNow.ToString('o');base_commit=[string]$overlay.base_commit;source_head=(&git -C $source rev-parse HEAD).Trim();source_dirty=$true;entrypoint='server.js';source_identity_sha256=$sourceIdentity;source_manifest_sha256=(Sha (Join-Path $releaseRoot 'release-source-manifest.json'));changed_source_files=($verifiedWebPatch+$verifiedPythonPatch);file_count=$regular.Count;files=$regular};Write-Json (Join-Path $releaseRoot 'operational-overlay-manifest.json') $operational
  $final=Assert-ReleaseRoot $releaseRoot $buildId;[IO.Compression.ZipFile]::CreateFromDirectory($releaseRoot,$zipPath,[IO.Compression.CompressionLevel]::Optimal,$false);$zipSafety=Test-ZipSafety $zipPath;if($zipSafety.unsafe_entries-or$zipSafety.duplicate_entries-or$zipSafety.symlink_entries){throw 'R112_ZIP_SAFETY_REJECTED'}
  $rounds=@();$targets=@();try{foreach($round in 1..2){$target=Join-Path $verify ("r112-verify-$round-"+[guid]::NewGuid().ToString('N'));$targets+=$target;New-Item -ItemType Directory -Path $target|Out-Null;[IO.Compression.ZipFile]::ExtractToDirectory($zipPath,$target);$identity=Assert-ReleaseRoot $target $buildId;$runtimeFiles=@(Get-ChildItem -LiteralPath (Join-Path $target 'support\python-runtime\evomind_runtime') -File -Filter '*.py');if($runtimeFiles.Count-ne30){throw "R112_PYTHON_FILE_COUNT_REJECTED:$round"};& ([IO.Path]::GetFullPath($SmokePython)) -m py_compile @($runtimeFiles.FullName);if($LASTEXITCODE-ne0){throw "R112_PYCOMPILE_REJECTED:$round"};Get-ChildItem -LiteralPath $target -Recurse -Directory -Filter '__pycache__'|Remove-Item -Recurse -Force;$smoke=Start-ReleaseSmoke $target $buildId $sourceIdentity $round $verify ([IO.Path]::GetFullPath($BaselineDatabase)) ([IO.Path]::GetFullPath($SmokePython));$rounds+=[ordered]@{round=$round;status='passed';unsafe_entries=$zipSafety.unsafe_entries;duplicate_entries=$zipSafety.duplicate_entries;symlink_entries=$zipSafety.symlink_entries;manifest_files=$identity.files.manifest_files;actual_regular_files=$identity.files.regular_files;missing=0;file_extra=0;mismatch=0;python_files=30;py_compile='passed';startup=$smoke}}}finally{foreach($target in $targets){Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue}}
  Write-Json $sourceManifestOutput $sourceManifest
  $payload=[ordered]@{schema='evomind.controlled_secret_binding_release.v2';status='built_and_verified';candidate_id=$buildId;zip_path=$zipPath;zip_bytes=[long]((Get-Item -LiteralPath $zipPath).Length);zip_sha256=(Sha $zipPath);source_manifest_path=$sourceManifestOutput;source_manifest_sha256=(Sha $sourceManifestOutput);layout=$final.layout;cache_cleanup=$cacheCleanup;zip_safety=$zipSafety;changed_source_files=($verifiedWebPatch+$verifiedPythonPatch);r111_runtime_count=30;verification_rounds=$rounds;production_deployed=$false;chrome=0;hpc_accessed=$false;competition_tools=0;worker_control=0;training=0;remote_writes=0};Write-Json $resultPath $payload;$payload|ConvertTo-Json -Depth 20 -Compress
}finally{if($junctionCreated-and(Test-Path -LiteralPath $buildNodeModules)){[IO.Directory]::Delete($buildNodeModules)};foreach($path in @($buildRoot,$runtimeRoot,$releaseRoot)){Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction SilentlyContinue}}
