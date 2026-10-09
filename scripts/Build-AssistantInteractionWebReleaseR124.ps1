param(
  [Parameter(Mandatory)][ValidateSet('DryRun','Build')][string]$Mode,
  [string]$BaselineZip = 'D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\overlay-controlled-secret-binding-r112-standalone-v5.zip',
  [string]$SourceRoot = '',
  [string]$VerifyRoot = 'D:\EV12',
  [string]$OutputRoot = 'D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts',
  [string]$BaselineDatabase = '',
  [string]$SmokePython = 'C:\codex-python\python.exe'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.Net.Http

$expectedBaselineZipSha256 = '2284252cf7ae98a1d3db08e9117d863d839e55174d428df2e7e00dc1a7d8eaa2'
$expectedBaselineBuildId = 'overlay-controlled-secret-binding-r112-standalone-f38c40f2dbdc'
$expectedBaselineSourceManifestSha256 = '782333e5552c26ad31be24503c05eed9844af79362df333e3d3641a612e48da1'
$expectedBaselineRuntimeTreeSha256 = '4bb0803db0e0fed319e45f4ac39152382fdcb762650cb2009d1fcd2b0e9d17fc'

function Get-Sha256([string]$Path) {
  (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-TextSha256([string]$Value) {
  $hash = [Security.Cryptography.SHA256]::Create()
  try {
    ([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($Value)))).Replace('-','').ToLowerInvariant()
  } finally {
    $hash.Dispose()
  }
}

function New-SmokeSessionSecret {
  $rng=[Security.Cryptography.RandomNumberGenerator]::Create()
  try {
    $bytes=New-Object byte[] 48
    $rng.GetBytes($bytes)
    [Convert]::ToBase64String($bytes)
  } finally {
    $rng.Dispose()
  }
}

function Write-Utf8Json([string]$Path,[object]$Value) {
  $parent = Split-Path -Parent $Path
  if ($parent -and -not (Test-Path -LiteralPath $parent -PathType Container)) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
  }
  [IO.File]::WriteAllText($Path,(($Value | ConvertTo-Json -Depth 30) + "`n"),[Text.UTF8Encoding]::new($false))
}

function Read-Json([string]$Path) {
  Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Assert-RelativePath([string]$Relative) {
  if (-not $Relative -or [IO.Path]::IsPathRooted($Relative) -or (($Relative -split '/') -contains '..') -or $Relative.Contains('\')) {
    throw "R124_PATH_UNSAFE:$Relative"
  }
}

function Assert-ChildPath([string]$Path,[string]$Parent,[string]$Code) {
  $resolved = [IO.Path]::GetFullPath($Path).TrimEnd('\')
  $root = [IO.Path]::GetFullPath($Parent).TrimEnd('\') + '\'
  if (-not $resolved.StartsWith($root,[StringComparison]::OrdinalIgnoreCase)) { throw $Code }
  $resolved
}

function Copy-DirectoryContents([string]$From,[string]$To) {
  if (-not (Test-Path -LiteralPath $From -PathType Container)) { throw "R124_COPY_SOURCE_MISSING:$From" }
  New-Item -ItemType Directory -Path $To -Force | Out-Null
  # Keep the formal builder compatible with Windows PowerShell 5.1, where
  # Copy-Item does not expose the ProgressAction common parameter.
  Get-ChildItem -LiteralPath $From -Force | Copy-Item -Destination $To -Recurse -Force
}

function Get-RegularFiles([string]$Root,[string[]]$Exclude = @()) {
  $full = [IO.Path]::GetFullPath($Root).TrimEnd('\')
  $excluded = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
  foreach ($value in $Exclude) { [void]$excluded.Add($value) }
  @(Get-ChildItem -LiteralPath $full -Recurse -File -Force | Where-Object {
    $_.Name -notmatch '\.(pyc|pyo)$' -and $_.FullName -notmatch '[\\/]__pycache__[\\/]'
  } | ForEach-Object {
    $relative = $_.FullName.Substring($full.Length + 1).Replace('\','/')
    if (-not $excluded.Contains($relative)) {
      [ordered]@{ path=$relative; bytes=[long]$_.Length; sha256=(Get-Sha256 $_.FullName) }
    }
  } | Where-Object { $null -ne $_ } | Sort-Object path)
}

function Get-FileMap([object[]]$Entries) {
  $map = @{}
  foreach ($entry in @($Entries)) {
    $path = [string]$entry.path
    Assert-RelativePath $path
    if ($map.ContainsKey($path)) { throw "R124_DUPLICATE_PATH:$path" }
    $map[$path] = $entry
  }
  $map
}

function Get-Tree([string]$Root) {
  $rows = Get-RegularFiles $Root
  $text = @($rows | ForEach-Object { "$($_.path)|$($_.bytes)|$($_.sha256)" }) -join "`n"
  [ordered]@{ file_count=$rows.Count; sha256=(Get-TextSha256 $text); files=$rows }
}

function Test-ZipSafety([string]$ZipPath) {
  $zip = [IO.Compression.ZipFile]::OpenRead($ZipPath)
  try {
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $unsafe=0;$duplicates=0;$symlinks=0;$regular=0;$directories=0;$uncompressed=[int64]0
    foreach ($entry in $zip.Entries) {
      $name=[string]$entry.FullName
      if (-not $seen.Add($name)) { $duplicates++ }
      # Zip entries produced by .NET may use backslashes on Windows. Normalize
      # before applying traversal/root checks so platform separators are not
      # mistaken for unsafe names.
      $normalizedName = $name.Replace('\','/')
      if ([IO.Path]::IsPathRooted($normalizedName) -or (($normalizedName -split '/') -contains '..') -or $normalizedName.StartsWith('/') -or $normalizedName -match '^[A-Za-z]:/') { $unsafe++ }
      if (((($entry.ExternalAttributes -shr 16) -band 0xF000) -eq 0xA000)) { $symlinks++ }
      if ($name.EndsWith('/')) { $directories++ } else { $regular++;$uncompressed += [int64]$entry.Length }
    }
    [ordered]@{ unsafe_entries=$unsafe; duplicate_entries=$duplicates; symlink_entries=$symlinks; regular_files=$regular; directory_entries=$directories; uncompressed_bytes=$uncompressed }
  } finally {
    $zip.Dispose()
  }
}

function Expand-SafeZip([string]$ZipPath,[string]$Destination,[string]$Code) {
  $safety = Test-ZipSafety $ZipPath
  if ($safety.unsafe_entries -or $safety.duplicate_entries -or $safety.symlink_entries -or $safety.regular_files -gt 40000 -or $safety.uncompressed_bytes -gt 3GB) {
    throw "${Code}_SAFETY_REJECTED"
  }
  New-Item -ItemType Directory -Path $Destination -Force | Out-Null
  [IO.Compression.ZipFile]::ExtractToDirectory($ZipPath,$Destination)
  $safety
}

function New-SafeZip([string]$Root,[string]$ZipPath) {
  $full=[IO.Path]::GetFullPath($Root).TrimEnd('\')
  $seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
  $zip=[IO.Compression.ZipFile]::Open($ZipPath,[IO.Compression.ZipArchiveMode]::Create)
  try {
    foreach($file in @(Get-ChildItem -LiteralPath $full -Recurse -File -Force | Sort-Object FullName)) {
      if(($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw "R124_ZIP_REPARSE_POINT:$($file.FullName)"}
      $relative=$file.FullName.Substring($full.Length+1).Replace('\','/')
      Assert-RelativePath $relative
      if(-not $seen.Add($relative)){throw "R124_ZIP_DUPLICATE_PATH:$relative"}
      $entry=$zip.CreateEntry($relative,[IO.Compression.CompressionLevel]::Optimal)
      # Fixed timestamps keep the candidate byte stream reproducible while
      # avoiding filesystem metadata in the release identity.
      $entry.LastWriteTime=[DateTimeOffset]::new(1980,1,1,0,0,0,[TimeSpan]::Zero)
      $input=[IO.File]::OpenRead($file.FullName);$output=$entry.Open()
      try{$input.CopyTo($output)}finally{$output.Dispose();$input.Dispose()}
    }
  } finally {
    $zip.Dispose()
  }
  [ordered]@{regular_files=$seen.Count;directory_entries=0;normalized_paths=$true}
}

function Verify-RegularManifest([string]$Root,[object]$Manifest) {
  $expected = Get-FileMap @($Manifest.files)
  if ($expected.Count -ne [int]$Manifest.file_count) { throw 'R124_MANIFEST_COUNT_REJECTED' }
  $actual = Get-FileMap @(Get-RegularFiles $Root @('operational-overlay-manifest.json'))
  $missing = @($expected.Keys | Where-Object { -not $actual.ContainsKey($_) })
  $extra = @($actual.Keys | Where-Object { -not $expected.ContainsKey($_) })
  $mismatch = @($expected.Keys | Where-Object {
    $actual.ContainsKey($_) -and ([long]$actual[$_].bytes -ne [long]$expected[$_].bytes -or [string]$actual[$_].sha256 -cne [string]$expected[$_].sha256)
  })
  [ordered]@{ manifest_files=$expected.Count; actual_regular_files=$actual.Count; missing=$missing; file_extra=$extra; mismatch=$mismatch }
}

function Test-BuildCacheRelative([string]$Relative) {
  $normalized = $Relative.Replace('\','/').TrimStart('/')
  $normalized -match '^\.next/(cache|dev)(/|$)' -or
  $normalized -match '(^|/)node_modules/\.cache(/|$)' -or
  $normalized -match '(^|/)(\.webpack-cache|webpack-cache)(/|$)' -or
  $normalized -match '^build-data(/|$)'
}

function Remove-BuildCaches([string]$Root) {
  $full=[IO.Path]::GetFullPath($Root).TrimEnd('\');$prefix=$full+'\'
  $targets=@(Get-ChildItem -LiteralPath $full -Recurse -Directory -Force | Where-Object {
    Test-BuildCacheRelative $_.FullName.Substring($full.Length+1).Replace('\','/')
  } | Sort-Object {$_.FullName.Length} -Descending)
  foreach($target in $targets){
    $resolved=[IO.Path]::GetFullPath($target.FullName)
    if(-not $resolved.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){throw 'R124_CACHE_PATH_ESCAPE'}
    Remove-Item -LiteralPath $resolved -Recurse -Force
  }
  [ordered]@{removed_directories=$targets.Count}
}

function Get-ReleaseLayout([string]$Root) {
  $full=[IO.Path]::GetFullPath($Root).TrimEnd('\')
  $cache=@(Get-ChildItem -LiteralPath $full -Recurse -File -Force | Where-Object {
    Test-BuildCacheRelative $_.FullName.Substring($full.Length+1).Replace('\','/')
  })
  [ordered]@{
    root_server=(Test-Path -LiteralPath (Join-Path $Root 'server.js') -PathType Leaf)
    root_next=(Test-Path -LiteralPath (Join-Path $Root 'node_modules\next\package.json') -PathType Leaf)
    nested_standalone=(Test-Path -LiteralPath (Join-Path $Root '.next\standalone'))
    cache_files=$cache.Count
  }
}

function Assert-ReleaseRoot([string]$Root,[string]$BuildId) {
  $layout=Get-ReleaseLayout $Root
  if(-not$layout.root_server-or-not$layout.root_next-or$layout.nested_standalone-or$layout.cache_files){throw 'R124_RELEASE_LAYOUT_REJECTED'}
  foreach($required in @('.next\static','public','src','support\python-runtime\evomind_runtime')){
    if(-not(Test-Path -LiteralPath (Join-Path $Root $required)-PathType Container)){throw "R124_RELEASE_DIR_MISSING:$required"}
  }
  $runtime=Read-Json (Join-Path $Root 'runtime-build-manifest.json')
  $manifest=Read-Json (Join-Path $Root 'operational-overlay-manifest.json')
  if([string]$runtime.build_id-cne$BuildId-or[string]$manifest.overlay_id-cne$BuildId){throw 'R124_BUILD_ID_REJECTED'}
  $closure=Verify-RegularManifest $Root $manifest
  if($closure.missing.Count-or$closure.file_extra.Count-or$closure.mismatch.Count){throw 'R124_OPERATIONAL_CLOSURE_REJECTED'}
  [ordered]@{layout=$layout;runtime=$runtime;manifest=$manifest;closure=$closure}
}

function Remove-TestFiles([string]$Root) {
  $removed = 0
  foreach ($file in @(Get-ChildItem -LiteralPath $Root -Recurse -File -Force | Where-Object {
    $_.Name -match '(?i)\.(?:test|spec)\.(?:ts|tsx|mts|cts|js|mjs)$' -or
    $_.FullName -match '(?i)[\\/]__tests__[\\/]'
  })) {
    Remove-Item -LiteralPath $file.FullName -Force
    $removed++
  }
  $removed
}

function Assert-SourceScope([string]$CandidateRoot,[string]$BaselineRoot,[object[]]$Patches,[string]$ForbiddenTestRelative,[switch]$Production) {
  $baselineSrc=Get-FileMap @(Get-RegularFiles (Join-Path $BaselineRoot 'src'))
  $candidateSrc=Get-FileMap @(Get-RegularFiles (Join-Path $CandidateRoot 'src'))
  $allowed=@{};foreach($patch in $Patches){
    $relative=[string]$patch.relative
    if($relative.StartsWith('src/')){$relative=$relative.Substring(4)}
    $allowed[$relative]=[string]$patch.sha256
  }
  $mismatch=@()
  foreach($path in $baselineSrc.Keys){
    if($Production -and $path -match '(?i)\.test\.(?:ts|tsx|mts|cts|js|mjs)$'){continue}
    if($allowed.ContainsKey($path)){continue}
    if(-not$candidateSrc.ContainsKey($path)-or[string]$candidateSrc[$path].sha256-cne[string]$baselineSrc[$path].sha256){$mismatch+=$path}
  }
  foreach($path in $candidateSrc.Keys){
    if($Production -and $path -match '(?i)\.test\.(?:ts|tsx|mts|cts|js|mjs)$'){$mismatch+=$path;continue}
    if(-not$baselineSrc.ContainsKey($path)-and-not$allowed.ContainsKey($path)){$mismatch+=$path}
  }
  foreach($path in $allowed.Keys){if(-not$candidateSrc.ContainsKey($path)-or[string]$candidateSrc[$path].sha256-cne[string]$allowed[$path]){$mismatch+=$path}}
  $forbiddenRelative=$ForbiddenTestRelative
  if($forbiddenRelative.StartsWith('src/')){$forbiddenRelative=$forbiddenRelative.Substring(4)}
  if(Test-Path -LiteralPath (Join-Path (Join-Path $CandidateRoot 'src') $forbiddenRelative.Replace('/','\'))){$mismatch+=$ForbiddenTestRelative}
  foreach($tree in @('public','support')){
    $a=Get-Tree (Join-Path $BaselineRoot $tree);$b=Get-Tree (Join-Path $CandidateRoot $tree)
    if($a.file_count-ne$b.file_count-or[string]$a.sha256-cne[string]$b.sha256){$mismatch+="$tree/**"}
  }
  # Build configuration (Next/Tailwind/TypeScript) is not part of the frozen
  # standalone root; it is supplied as a separate, hash-bound build input.
  if($mismatch.Count){throw "R124_SOURCE_SCOPE_REJECTED:$($mismatch[0])"}
  [ordered]@{baseline_src_files=$baselineSrc.Count;candidate_src_files=$candidateSrc.Count;production_allowlist=$allowed.Count;unchanged_source_files=($baselineSrc.Count-1);added_source_files=2;test_in_production=$false;mismatch=0}
}

function Get-FreePort {
  $listener=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0)
  try{$listener.Start();[int]$listener.LocalEndpoint.Port}finally{$listener.Stop()}
}

function Invoke-HttpResponse([string]$Uri,[hashtable]$Headers=@{},[int]$TimeoutSec=8) {
  try {
    $response=Invoke-WebRequest -Uri $Uri -Headers $Headers -UseBasicParsing -TimeoutSec $TimeoutSec -ErrorAction Stop
    [pscustomobject]@{StatusCode=[int]$response.StatusCode;Content=[string]$response.Content}
  } catch {
    $status=0;$content=''
    if($_.Exception.Response){
      try{$status=[int]$_.Exception.Response.StatusCode}catch{}
      try{
        $stream=$_.Exception.Response.GetResponseStream()
        if($stream){$reader=[IO.StreamReader]::new($stream);try{$content=$reader.ReadToEnd()}finally{$reader.Dispose();$stream.Dispose()}}
      }catch{}
    }
    if($status){return [pscustomobject]@{StatusCode=$status;Content=$content}}
    throw
  }
}

function Wait-Web([string]$Uri,[int]$Seconds=60,[hashtable]$Headers=@{}) {
  $deadline=[DateTime]::UtcNow.AddSeconds($Seconds)
  do{
    try{return Invoke-HttpResponse $Uri $Headers 8}catch{}
    Start-Sleep -Milliseconds 250
  }while([DateTime]::UtcNow-lt$deadline)
  throw 'R124_SMOKE_HTTP_TIMEOUT'
}

function Invoke-ReleaseSmoke([string]$ReleaseRoot,[string]$BuildId,[string]$SourceIdentity,[int]$Round,[string]$VerifyBase,[string]$Database,[string]$PythonExecutable) {
  $fixture=Join-Path $VerifyBase ("r124-smoke-$Round-"+[guid]::NewGuid().ToString('N'))
  $runtimePort=Get-FreePort;do{$webPort=Get-FreePort}while($webPort-eq$runtimePort)
  $runtimeProcess=$null;$webProcess=$null
  $runtimeOut=Join-Path $fixture 'runtime.out.log';$runtimeErr=Join-Path $fixture 'runtime.err.log';$webOut=Join-Path $fixture 'web.out.log';$webErr=Join-Path $fixture 'web.err.log'
  $names=@('PYTHONPATH','DATABASE_URL','WORKSTATION_ROOT','WORKSTATION_DATA_DIR','EVOMIND_RUNTIME_BUILD_MANIFEST','EVOMIND_SOURCE_TREE_SHA256','EVOMIND_BUILD_COMMIT_HASH','EVOMIND_FRONTEND_VERSION','EVOMIND_BACKEND_VERSION','EVOMIND_RUNTIME_PORT','HOSTNAME','PORT','NODE_ENV','NEXT_TELEMETRY_DISABLED','WORKSTATION_SESSION_SECRET')
  $saved=@{};foreach($name in $names){$saved[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
  try{
    New-Item -ItemType Directory -Path $fixture -Force|Out-Null
    $databaseCopy=Join-Path $fixture 'workstation.db';Copy-Item -LiteralPath $Database -Destination $databaseCopy
    $env:PYTHONPATH=(Join-Path $ReleaseRoot 'support\python-runtime')+';'+(Join-Path $ReleaseRoot 'support\python-vendor')
    $env:DATABASE_URL='file:'+$databaseCopy.Replace('\','/');$env:WORKSTATION_ROOT=$fixture;$env:WORKSTATION_DATA_DIR=$fixture;$env:EVOMIND_RUNTIME_PORT=[string]$runtimePort
    $env:EVOMIND_RUNTIME_BUILD_MANIFEST=Join-Path $ReleaseRoot 'runtime-build-manifest.json';$env:EVOMIND_SOURCE_TREE_SHA256=$SourceIdentity;$env:EVOMIND_BUILD_COMMIT_HASH='664a636ddd419a66f73cc10820c0a16429784866';$env:EVOMIND_FRONTEND_VERSION='0.3.0';$env:EVOMIND_BACKEND_VERSION='0.3.0'
    $env:HOSTNAME='127.0.0.1';$env:PORT=[string]$webPort;$env:NODE_ENV='production';$env:NEXT_TELEMETRY_DISABLED='1';$env:WORKSTATION_SESSION_SECRET=New-SmokeSessionSecret
    $python=[IO.Path]::GetFullPath($PythonExecutable);$node=(Get-Command node.exe -ErrorAction Stop).Source
    $runtimeProcess=Start-Process -FilePath $python -ArgumentList @('-m','evomind_runtime.http_server','--workspace',$fixture,'--host','127.0.0.1','--port',[string]$runtimePort) -WorkingDirectory $fixture -WindowStyle Hidden -PassThru -RedirectStandardOutput $runtimeOut -RedirectStandardError $runtimeErr
    $token=Join-Path $fixture 'workspace\runtime\runtime.token';$deadline=[DateTime]::UtcNow.AddSeconds(40);do{Start-Sleep -Milliseconds 200}while(-not(Test-Path -LiteralPath $token)-and[DateTime]::UtcNow-lt$deadline-and-not$runtimeProcess.HasExited)
    if(-not(Test-Path -LiteralPath $token)-or$runtimeProcess.HasExited){throw "R124_RUNTIME_START_REJECTED:$Round"}
    $headers=@{Authorization='Bearer '+(Get-Content -LiteralPath $token -Raw).Trim()}
    $rh=Wait-Web "http://127.0.0.1:$runtimePort/v1/health" 30 $headers;if([int]$rh.StatusCode-ne200){throw "R124_RUNTIME_HEALTH_REJECTED:$Round"}
    $webProcess=Start-Process -FilePath $node -ArgumentList @('server.js') -WorkingDirectory $ReleaseRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $webOut -RedirectStandardError $webErr
    $health=Wait-Web "http://127.0.0.1:$webPort/api/healthz" 60;$payload=$health.Content|ConvertFrom-Json
    if([int]$health.StatusCode-ne200-or-not[bool]$payload.ok-or[string]$payload.status-cne'ready'-or[string]$payload.build_id-cne$BuildId-or@($payload.failures).Count){throw "R124_WEB_HEALTH_REJECTED:$Round"}
    $login=Wait-Web "http://127.0.0.1:$webPort/login" 30
    $assistant=Invoke-HttpResponse "http://127.0.0.1:$webPort/api/assistant/runs" @{} 15
    if([int]$login.StatusCode-ne200-or-not([string]$login.Content).Contains('DeepEvo')-or[int]$assistant.StatusCode-notin@(401,403)){throw "R124_WEB_SMOKE_REJECTED:$Round"}
    if($runtimeProcess.HasExited-or$webProcess.HasExited){throw "R124_PROCESS_EXITED:$Round"}
    $stderr=((Get-Content -LiteralPath $runtimeErr -Raw -ErrorAction SilentlyContinue)+"`n"+(Get-Content -LiteralPath $webErr -Raw -ErrorAction SilentlyContinue))
    if($stderr-match'MODULE_NOT_FOUND|Cannot find module|PrismaClientInitializationError|Cannot find package.*prisma|Cannot find.*sharp'){throw "R124_STDERR_REJECTED:$Round"}
    [ordered]@{round=$Round;status='passed';runtime_health=200;web_health=200;login=200;assistant_guard=[int]$assistant.StatusCode;build_id=$BuildId;web_alive=$true;runtime_alive=$true}
  } finally {
    foreach($process in @($webProcess,$runtimeProcess)){if($process-and-not$process.HasExited){Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue;try{$process.WaitForExit()}catch{}}}
    foreach($name in $names){if($null-eq$saved[$name]){Remove-Item "Env:$name" -ErrorAction SilentlyContinue}else{[Environment]::SetEnvironmentVariable($name,[string]$saved[$name],'Process')}}
    Remove-Item -LiteralPath $fixture -Recurse -Force -ErrorAction SilentlyContinue
  }
}

$source=if($SourceRoot){[IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')}else{[IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot)).TrimEnd('\')}
$web=Join-Path $source 'web\research-agent-workstation'
$baselineZip=[IO.Path]::GetFullPath($BaselineZip)
$verify=[IO.Path]::GetFullPath($VerifyRoot).TrimEnd('\')
$output=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$BaselineDatabase=if($BaselineDatabase){[IO.Path]::GetFullPath($BaselineDatabase)}else{Join-Path $web 'prisma\workstation.db'}
$testRelative='src/components/workstation/screens/assistant-activity.test.ts'
$patches=@(
  [ordered]@{relative='src/components/workstation/screens/AssistantScreen.tsx';source=Join-Path $web 'src\components\workstation\screens\AssistantScreen.tsx';bytes=93570;sha256='47b0afa601dfe14a3d48a57080fdffe7879b49712020ddc5f04b6c8bc9c0eeab'},
  [ordered]@{relative='src/components/workstation/screens/assistant-activity.mjs';source=Join-Path $web 'src\components\workstation\screens\assistant-activity.mjs';bytes=1173;sha256='841f3b66b33ffac8a9e223d312d1352adfb8c7d0e055f205e8811baccca1bff9'},
  [ordered]@{relative='src/components/workstation/screens/assistant-activity.d.mts';source=Join-Path $web 'src\components\workstation\screens\assistant-activity.d.mts';bytes=530;sha256='38a822860b30db088e5b7757f095438f288078aee606413574c55198e40a60c9'}
)
$testInputs=@(
  [ordered]@{relative=$testRelative;source=Join-Path $web 'src\components\workstation\screens\assistant-activity.test.ts';bytes=1578;sha256='5669a121104beafc96470b89c3eabe8109943a380209aaf7324c4c3c0e5b00d2';included_in_production=$false},
  [ordered]@{relative='src/lib/server/assistant-super-agent-contract.test.ts';source=Join-Path $web 'src\lib\server\assistant-super-agent-contract.test.ts';bytes=9535;sha256='6505e12ffaf2d7291c5265b7d0568e36ffc060f30e6abc7536d88bf4722fbc3a';included_in_production=$false}
)
$buildInputs=@(
  [ordered]@{relative='package-lock.json';source=Join-Path $web 'package-lock.json';bytes=174852;sha256='d859abd91d5e4283adca07bf73b02d1df18ae52d7369358e2a88113f35c664ad';included_in_production=$false},
  [ordered]@{relative='next.config.mjs';source=Join-Path $web 'next.config.mjs';bytes=2307;sha256='ebbcda321b4393074ceae3dfa2fbf9349b4c2ffae737c210cec1824f91b0d665';included_in_production=$false},
  [ordered]@{relative='postcss.config.mjs';source=Join-Path $web 'postcss.config.mjs';bytes=103;sha256='2147573f4053310e348ad9d722882426c884376f42a354daea89bcd663b33b25';included_in_production=$false},
  [ordered]@{relative='tailwind.config.ts';source=Join-Path $web 'tailwind.config.ts';bytes=6026;sha256='7e532fb12f0e0c64ba63571df5e6dce15a9cf0d8f21bf11eac3d15035f56dfc9';included_in_production=$false},
  [ordered]@{relative='tsconfig.json';source=Join-Path $web 'tsconfig.json';bytes=743;sha256='c43c5534852c5c47f60af502d3e5067471940d9862f7e2eaa8948be5a76192e3';included_in_production=$false}
)

foreach($path in @($baselineZip,$source,$web,$verify,$output,$BaselineDatabase,$SmokePython)){if(-not(Test-Path -LiteralPath $path)){throw "R124_INPUT_MISSING:$path"}}
if((Get-Sha256 $baselineZip)-cne$expectedBaselineZipSha256){throw 'R124_BASELINE_ZIP_HASH_REJECTED'}
foreach($entry in @($patches)+@($testInputs)){
  if(-not(Test-Path -LiteralPath $entry.source -PathType Leaf)-or[long](Get-Item -LiteralPath $entry.source).Length-ne[long]$entry.bytes-or(Get-Sha256 $entry.source)-cne[string]$entry.sha256){throw "R124_PATCH_HASH_REJECTED:$($entry.relative)"}
}
foreach($entry in @($buildInputs)){
  if(-not(Test-Path -LiteralPath $entry.source -PathType Leaf)-or[long](Get-Item -LiteralPath $entry.source).Length-ne[long]$entry.bytes-or(Get-Sha256 $entry.source)-cne[string]$entry.sha256){throw "R124_BUILD_INPUT_HASH_REJECTED:$($entry.relative)"}
}
$baselineSafety=Test-ZipSafety $baselineZip
if($baselineSafety.unsafe_entries-or$baselineSafety.duplicate_entries-or$baselineSafety.symlink_entries){throw 'R124_BASELINE_ZIP_SAFETY_REJECTED'}
$seed=$expectedBaselineSourceManifestSha256+"`n"+(@($patches+$buildInputs|ForEach-Object{"$($_.relative)|$($_.bytes)|$($_.sha256)"})-join"`n")
$sourceIdentity=Get-TextSha256 $seed
$buildId="overlay-assistant-interaction-r124-$($sourceIdentity.Substring(0,12))"
$dry=[ordered]@{schema='evomind.assistant_interaction_web_release_dry_run.r124.v1';status='passed';build_id=$buildId;baseline=[ordered]@{build_id=$expectedBaselineBuildId;zip_sha256=$expectedBaselineZipSha256;source_manifest_sha256=$expectedBaselineSourceManifestSha256};production_allowlist_count=3;production_allowlist=$patches;build_input_count=5;build_inputs=$buildInputs;test_inputs=$testInputs;test_in_production=$false;expected_runtime_tree_sha256=$expectedBaselineRuntimeTreeSha256;hpc_accessed=$false;gpu_touched=$false;production_deployed=$false}
$dry|ConvertTo-Json -Depth 15
if($Mode-eq'DryRun'){return}

$baselineRoot=Assert-ChildPath (Join-Path $verify ('r124-baseline-'+[guid]::NewGuid().ToString('N'))) $verify 'R124_BASELINE_STAGE_REJECTED'
$buildRoot=Assert-ChildPath (Join-Path $verify ('r124-build-'+[guid]::NewGuid().ToString('N'))) $verify 'R124_BUILD_STAGE_REJECTED'
$releaseRoot=Assert-ChildPath (Join-Path $verify ('r124-release-'+[guid]::NewGuid().ToString('N'))) $verify 'R124_RELEASE_STAGE_REJECTED'
$zipPath=Join-Path $output 'web-overlay-assistant-interaction-r124-v1.zip'
$sourceManifestPath=Join-Path $output 'web-overlay-assistant-interaction-r124-v1-source-manifest.json'
$buildResultPath=Join-Path $output 'web-overlay-assistant-interaction-r124-v1-build-result.json'
foreach($path in @($zipPath,$sourceManifestPath,$buildResultPath)){if(Test-Path -LiteralPath $path){throw "R124_OUTPUT_EXISTS:$path"}}
$junction=$false;$buildNodeModules=Join-Path $buildRoot 'node_modules'
try{
  $expanded=Expand-SafeZip $baselineZip $baselineRoot 'R124_BASELINE'
  $baselineOperational=Read-Json (Join-Path $baselineRoot 'operational-overlay-manifest.json')
  $baselineRuntime=Read-Json (Join-Path $baselineRoot 'runtime-build-manifest.json')
  if([string]$baselineOperational.overlay_id-cne$expectedBaselineBuildId-or[string]$baselineRuntime.build_id-cne$expectedBaselineBuildId-or(Get-Sha256 (Join-Path $baselineRoot 'release-source-manifest.json'))-cne$expectedBaselineSourceManifestSha256){throw 'R124_BASELINE_IDENTITY_REJECTED'}
  $baselineClosure=Verify-RegularManifest $baselineRoot $baselineOperational
  if($baselineClosure.missing.Count-or$baselineClosure.file_extra.Count-or$baselineClosure.mismatch.Count){throw 'R124_BASELINE_CLOSURE_REJECTED'}
  foreach($name in @('src','public','support','package.json')){
    $origin=Join-Path $baselineRoot $name
    if(Test-Path -LiteralPath $origin -PathType Container){Copy-DirectoryContents $origin (Join-Path $buildRoot $name)}elseif(Test-Path -LiteralPath $origin -PathType Leaf){New-Item -ItemType Directory -Path $buildRoot -Force|Out-Null;Copy-Item -LiteralPath $origin -Destination (Join-Path $buildRoot $name)}else{throw "R124_BUILD_INPUT_MISSING:$name"}
  }
  foreach($patch in $patches){$target=Join-Path $buildRoot $patch.relative.Replace('/','\');New-Item -ItemType Directory -Path (Split-Path -Parent $target)-Force|Out-Null;Copy-Item -LiteralPath $patch.source -Destination $target -Force}
  # The frozen R112 standalone root omits its lockfile and build configuration.
  # Add these only as local Next build inputs; none is copied into the release
  # root except the standalone package.json emitted by Next.
  Copy-Item -LiteralPath (Join-Path $web 'package-lock.json') -Destination (Join-Path $buildRoot 'package-lock.json') -Force
  foreach($name in @('next.config.mjs','postcss.config.mjs','tailwind.config.ts','tsconfig.json')){
    Copy-Item -LiteralPath (Join-Path $web $name) -Destination (Join-Path $buildRoot $name) -Force
  }
  $forbiddenTest=Join-Path $buildRoot $testRelative.Replace('/','\');if(Test-Path -LiteralPath $forbiddenTest){Remove-Item -LiteralPath $forbiddenTest -Force}
  $buildScope=Assert-SourceScope $buildRoot $baselineRoot $patches $testRelative
  $verifiedNodeModules=Join-Path $web 'node_modules'
  foreach($required in @('next\dist\bin\next','typescript\bin\tsc')){if(-not(Test-Path -LiteralPath (Join-Path $verifiedNodeModules $required)-PathType Leaf)){throw "R124_TOOLCHAIN_MISSING:$required"}}
  $node=(Get-Command node.exe -ErrorAction Stop).Source
  Push-Location $web
  try{
    & $node --test --experimental-strip-types --disable-warning=MODULE_TYPELESS_PACKAGE_JSON @($testInputs.source)
    if($LASTEXITCODE-ne0){throw 'R124_TARGETED_TEST_REJECTED'}
  }finally{Pop-Location}
  New-Item -ItemType Junction -Path $buildNodeModules -Target $verifiedNodeModules|Out-Null;$junction=$true
  Push-Location $buildRoot
  try{
    & $node (Join-Path $verifiedNodeModules 'typescript\bin\tsc') --noEmit --incremental false
    if($LASTEXITCODE-ne0){throw 'R124_TYPECHECK_REJECTED'}
    $buildData=Join-Path $buildRoot 'build-data';New-Item -ItemType Directory -Path $buildData -Force|Out-Null
    $envNames=@('DATABASE_URL','WORKSTATION_ROOT','WORKSTATION_DATA_DIR','NEXT_TELEMETRY_DISABLED','NODE_PATH');$saved=@{};foreach($name in $envNames){$saved[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
    try{
      $env:DATABASE_URL='file:'+(Join-Path $buildData 'workstation.db').Replace('\','/');$env:WORKSTATION_ROOT=$buildData;$env:WORKSTATION_DATA_DIR=$buildData;$env:NEXT_TELEMETRY_DISABLED='1';$env:NODE_PATH=$verifiedNodeModules
      & $node (Join-Path $verifiedNodeModules 'next\dist\bin\next') build --webpack
      if($LASTEXITCODE-ne0){throw 'R124_WEB_BUILD_REJECTED'}
    }finally{foreach($name in $envNames){if($null-eq$saved[$name]){Remove-Item "Env:$name" -ErrorAction SilentlyContinue}else{[Environment]::SetEnvironmentVariable($name,[string]$saved[$name],'Process')}}}
  }finally{Pop-Location}
  if($junction){[IO.Directory]::Delete($buildNodeModules);$junction=$false}
  Remove-Item -LiteralPath (Join-Path $buildRoot 'build-data') -Recurse -Force -ErrorAction SilentlyContinue
  $removedTestFiles=Remove-TestFiles (Join-Path $buildRoot 'src')
  Copy-DirectoryContents (Join-Path $buildRoot '.next\standalone') $releaseRoot
  Copy-DirectoryContents (Join-Path $buildRoot '.next\static') (Join-Path $releaseRoot '.next\static')
  Copy-DirectoryContents (Join-Path $buildRoot 'public') (Join-Path $releaseRoot 'public')
  Copy-DirectoryContents (Join-Path $buildRoot 'support') (Join-Path $releaseRoot 'support')
  Copy-DirectoryContents (Join-Path $buildRoot 'src') (Join-Path $releaseRoot 'src')
  $cacheCleanup=Remove-BuildCaches $releaseRoot
  $removedTestFiles += Remove-TestFiles $releaseRoot
  $releaseScope=Assert-SourceScope $releaseRoot $baselineRoot $patches $testRelative -Production
  $sourceManifest=[ordered]@{schema='evomind.assistant_interaction_web_source_set.r124.v1';created_at_utc=[DateTime]::UtcNow.ToString('o');frozen=$true;source_tree_sha256=$sourceIdentity;baseline=[ordered]@{build_id=$expectedBaselineBuildId;zip_sha256=$expectedBaselineZipSha256;source_manifest_sha256=$expectedBaselineSourceManifestSha256;operational_file_count=[int]$baselineOperational.file_count};production_allowlist_count=3;production_files=@($patches|ForEach-Object{[ordered]@{path=[string]$_.relative;bytes=[long]$_.bytes;sha256=[string]$_.sha256}});build_input_count=5;build_inputs=@($buildInputs|ForEach-Object{[ordered]@{path=[string]$_.relative;bytes=[long]$_.bytes;sha256=[string]$_.sha256;included_in_production=$false}});test_inputs=@($testInputs|ForEach-Object{[ordered]@{path=[string]$_.relative;bytes=[long]$_.bytes;sha256=[string]$_.sha256;included_in_production=$false}});test_in_production=$false;expected_runtime_tree_sha256=$expectedBaselineRuntimeTreeSha256;production_deployed=$false;hpc_accessed=$false;gpu_touched=$false}
  Write-Utf8Json (Join-Path $releaseRoot 'release-source-manifest.json') $sourceManifest
  $runtimeManifest=[ordered]@{schema='evomind.runtime_build.v1';commit_hash=[string]$baselineRuntime.commit_hash;source_dirty=$true;source_tree_sha256=$sourceIdentity;build_id=$buildId;build_time=[DateTime]::UtcNow.ToString('o');backend_version=[string]$baselineRuntime.backend_version;frontend_version=[string]$baselineRuntime.frontend_version;database_schema_version=[string]$baselineRuntime.database_schema_version;database_schema_sha256=[string]$baselineRuntime.database_schema_sha256}
  Write-Utf8Json (Join-Path $releaseRoot 'runtime-build-manifest.json') $runtimeManifest
  $regular=Get-RegularFiles $releaseRoot @('operational-overlay-manifest.json')
  $operational=[ordered]@{schema='evomind.web_operational_overlay.v1';overlay_id=$buildId;created_at_utc=[DateTime]::UtcNow.ToString('o');base_commit=[string]$baselineOperational.base_commit;source_head=(& git -C $source rev-parse HEAD).Trim();source_dirty=$true;entrypoint='server.js';source_identity_sha256=$sourceIdentity;source_manifest_sha256=(Get-Sha256 (Join-Path $releaseRoot 'release-source-manifest.json'));changed_source_files=@($patches|ForEach-Object{[ordered]@{path=[string]$_.relative;bytes=[long]$_.bytes;sha256=[string]$_.sha256}});file_count=$regular.Count;files=$regular;production_deployed=$false;hpc_accessed=$false;gpu_touched=$false}
  Write-Utf8Json (Join-Path $releaseRoot 'operational-overlay-manifest.json') $operational
  $final=Assert-ReleaseRoot $releaseRoot $buildId
  $zipBuild=New-SafeZip $releaseRoot $zipPath
  $zipSafety=Test-ZipSafety $zipPath;if($zipSafety.unsafe_entries-or$zipSafety.duplicate_entries-or$zipSafety.symlink_entries){throw 'R124_CANDIDATE_ZIP_SAFETY_REJECTED'}
  $rounds=@();$targets=@()
  try{
    foreach($round in 1..2){
      $target=Assert-ChildPath (Join-Path $verify ("r124-verify-$round-"+[guid]::NewGuid().ToString('N'))) $verify 'R124_VERIFY_STAGE_REJECTED';$targets+=$target
      $roundSafety=Expand-SafeZip $zipPath $target "R124_VERIFY_$round"
      $identity=Assert-ReleaseRoot $target $buildId;$scope=Assert-SourceScope $target $baselineRoot $patches $testRelative -Production
      $runtimeFiles=@(Get-ChildItem -LiteralPath (Join-Path $target 'support\python-runtime\evomind_runtime') -File -Filter '*.py')
      & ([IO.Path]::GetFullPath($SmokePython)) -m py_compile @($runtimeFiles.FullName);if($LASTEXITCODE-ne0){throw "R124_PYCOMPILE_REJECTED:$round"}
      Get-ChildItem -LiteralPath $target -Recurse -Directory -Filter '__pycache__'|Remove-Item -Recurse -Force
      $smoke=Invoke-ReleaseSmoke $target $buildId $sourceIdentity $round $verify ([IO.Path]::GetFullPath($BaselineDatabase)) ([IO.Path]::GetFullPath($SmokePython))
      $rounds+=[ordered]@{round=$round;status='passed';zip_safety=$roundSafety;manifest=$identity.closure;source_scope=$scope;python_files=$runtimeFiles.Count;py_compile='passed';startup=$smoke}
    }
  }finally{foreach($target in $targets){Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue}}
  Copy-Item -LiteralPath (Join-Path $releaseRoot 'release-source-manifest.json') -Destination $sourceManifestPath
  $result=[ordered]@{schema='evomind.assistant_interaction_web_release.r124.v1';status='built_and_verified';candidate_id=$buildId;zip_path=$zipPath;zip_bytes=[long](Get-Item -LiteralPath $zipPath).Length;zip_sha256=(Get-Sha256 $zipPath);source_manifest_path=$sourceManifestPath;source_manifest_sha256=(Get-Sha256 $sourceManifestPath);baseline=[ordered]@{build_id=$expectedBaselineBuildId;zip_sha256=$expectedBaselineZipSha256;source_manifest_sha256=$expectedBaselineSourceManifestSha256};production_allowlist_count=3;production_files=$sourceManifest.production_files;build_input_count=5;build_inputs=$sourceManifest.build_inputs;test_inputs=$sourceManifest.test_inputs;test_in_production=$false;removed_test_files=$removedTestFiles;targeted_tests=[ordered]@{assistant_activity='passed';assistant_super_agent_contract='passed';typecheck='passed';next_build='passed'};baseline_closure=$baselineClosure;build_source_scope=$buildScope;release_source_scope=$releaseScope;layout=$final.layout;cache_cleanup=$cacheCleanup;zip_safety=$zipSafety;verification_rounds=$rounds;expected_runtime_tree_sha256=$expectedBaselineRuntimeTreeSha256;production_deployed=$false;hpc_accessed=$false;gpu_touched=$false;training_started=$false}
  Write-Utf8Json $buildResultPath $result
  $result|ConvertTo-Json -Depth 30 -Compress
}finally{
  if($junction-and(Test-Path -LiteralPath $buildNodeModules)){[IO.Directory]::Delete($buildNodeModules)}
  foreach($path in @($baselineRoot,$buildRoot,$releaseRoot)){Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction SilentlyContinue}
}
