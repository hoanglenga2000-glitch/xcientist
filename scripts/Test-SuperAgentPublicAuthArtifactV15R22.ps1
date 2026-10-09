param(
  [Parameter(Mandatory)][string]$OverlayZip,
  [Parameter(Mandatory)][string]$ExpectedBuildId,
  [Parameter(Mandatory)][string]$ResultPath,
  [string]$BaselineDatabase='D:\桌面\codex\科研港科技\web\research-agent-workstation\prisma\workstation.db',
  [string]$VerificationBasePath='D:\EV12'
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Free-Port {
  $listener=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0)
  try {$listener.Start(); [int]$listener.LocalEndpoint.Port} finally {$listener.Stop()}
}

function Send-Http {
  param(
    [Net.Http.HttpClient]$Client,
    [string]$Method,
    [string]$Url,
    [string]$HostHeader,
    [string]$Origin,
    [byte[]]$Bytes,
    [string]$ContentType,
    [string]$Cookie,
    [string]$ChunkSha256,
    [string]$CsrfToken,
    [string]$BearerToken
  )
  $request=[Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::new($Method),$Url)
  $request.Headers.Host=$HostHeader
  if($Origin){$request.Headers.TryAddWithoutValidation('Origin',$Origin)|Out-Null}
  $request.Headers.TryAddWithoutValidation('Sec-Fetch-Site','same-origin')|Out-Null
  if($Cookie){$request.Headers.TryAddWithoutValidation('Cookie',$Cookie)|Out-Null}
  if($ChunkSha256){$request.Headers.TryAddWithoutValidation('X-Chunk-SHA256',$ChunkSha256)|Out-Null}
  if($CsrfToken){$request.Headers.TryAddWithoutValidation('X-EvoMind-CSRF',$CsrfToken)|Out-Null}
  if($BearerToken){$request.Headers.TryAddWithoutValidation('Authorization',"Bearer $BearerToken")|Out-Null}
  if($null -ne $Bytes){
    $content=[Net.Http.ByteArrayContent]::new([byte[]]$Bytes)
    if($ContentType){$content.Headers.ContentType=[Net.Http.Headers.MediaTypeHeaderValue]::new($ContentType)}
    $request.Content=$content
  }
  try {
    $response=$Client.SendAsync($request).GetAwaiter().GetResult()
    $body=$response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    $setCookie=''
    try {$setCookie=($response.Headers.GetValues('Set-Cookie') -join '; ')} catch {}
    [pscustomobject]@{status=[int]$response.StatusCode;body=$body;location=[string]$response.Headers.Location;set_cookie=$setCookie}
  } finally {$request.Dispose()}
}

function Json([string]$Text){if([string]::IsNullOrWhiteSpace($Text)){return $null};$Text|ConvertFrom-Json}
function ShaBytes([byte[]]$Bytes){$hash=[Security.Cryptography.SHA256]::Create();try{([BitConverter]::ToString($hash.ComputeHash($Bytes))).Replace('-','').ToLowerInvariant()}finally{$hash.Dispose()}}

$zip=[IO.Path]::GetFullPath($OverlayZip)
$result=[IO.Path]::GetFullPath($ResultPath)
if(-not(Test-Path -LiteralPath $zip -PathType Leaf)){throw "AUTH_ARTIFACT_ZIP_MISSING:$zip"}
if(Test-Path -LiteralPath $result){throw "AUTH_ARTIFACT_RESULT_EXISTS:$result"}
if(-not(Test-Path -LiteralPath ([IO.Path]::GetFullPath($BaselineDatabase)) -PathType Leaf)){throw 'AUTH_ARTIFACT_BASELINE_DB_MISSING'}
$fixture=Join-Path ([IO.Path]::GetFullPath($VerificationBasePath)) ('auth-'+[guid]::NewGuid().ToString('N'))
$extract=Join-Path $fixture 'overlay'
$runtimeProcess=$null;$webProcess=$null;$client=$null
$saved=@{}
$environment=@('PYTHONPATH','DATABASE_URL','WORKSTATION_ROOT','WORKSTATION_DATA_DIR','EVOMIND_RUNTIME_BUILD_MANIFEST','EVOMIND_SOURCE_TREE_SHA256','EVOMIND_BUILD_COMMIT_HASH','EVOMIND_FRONTEND_VERSION','EVOMIND_BACKEND_VERSION','EVOMIND_RUNTIME_PORT','HOSTNAME','PORT','WORKSTATION_PUBLIC_ORIGIN','WORKSTATION_SESSION_SECRET','WORKSTATION_ADMIN_USERNAME','WORKSTATION_ADMIN_PASSWORD_SCRYPT','WORKSTATION_BOOTSTRAP_TOKEN_HASH','WORKSTATION_LOCAL_HTTPS')
foreach($name in $environment){$saved[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
try {
  New-Item -ItemType Directory -Path $fixture -Force|Out-Null
  [IO.Compression.ZipFile]::ExtractToDirectory($zip,$extract)
  $runtimePort=Free-Port; do {$webPort=Free-Port} while($webPort -eq $runtimePort)
  $database=Join-Path $fixture 'workstation.db'; Copy-Item -LiteralPath ([IO.Path]::GetFullPath($BaselineDatabase)) -Destination $database
  $manifest=Get-Content -LiteralPath (Join-Path $extract 'runtime-build-manifest.json') -Raw|ConvertFrom-Json
  if([string]$manifest.build_id -ne $ExpectedBuildId){throw 'AUTH_ARTIFACT_BUILD_ID_MISMATCH'}
  $env:PYTHONPATH=Join-Path $extract 'support\python-runtime'
  $env:WORKSTATION_ROOT=$fixture; $env:WORKSTATION_DATA_DIR=$fixture; $env:EVOMIND_RUNTIME_PORT=[string]$runtimePort
  $env:EVOMIND_RUNTIME_BUILD_MANIFEST=Join-Path $extract 'runtime-build-manifest.json'
  $env:EVOMIND_SOURCE_TREE_SHA256=[string]$manifest.source_tree_sha256
  $env:EVOMIND_BUILD_COMMIT_HASH=[string]$manifest.commit_hash
  $env:EVOMIND_FRONTEND_VERSION=[string]$manifest.frontend_version
  $env:EVOMIND_BACKEND_VERSION=[string]$manifest.backend_version
  $env:DATABASE_URL='file:'+$database.Replace('\','/')
  $env:HOSTNAME='127.0.0.1'; $env:PORT=[string]$webPort
  $env:WORKSTATION_PUBLIC_ORIGIN='https://evomind.zhjjq.tech'
  $env:WORKSTATION_SESSION_SECRET='fixture-session-secret-7e8c4fd8f5f04834bde3f7c5d0a2c1e9'
  $env:WORKSTATION_ADMIN_USERNAME='admin'
  $env:WORKSTATION_ADMIN_PASSWORD_SCRYPT='scrypt$16384$8$1$01010101010101010101010101010101$1c965e1bf8a174eebb5bdb229fa10295ff3d0ccfcf4aa9984615b2de65a06c9c'
  $bootstrapToken='fixture-bootstrap-token-v15r22'
  $env:WORKSTATION_BOOTSTRAP_TOKEN_HASH=(node -e "console.log(require('crypto').createHash('sha256').update('fixture-bootstrap-token-v15r22').digest('hex'))").Trim()
  $env:WORKSTATION_LOCAL_HTTPS='0'
  $runtimeProcess=Start-Process -FilePath (Get-Command python.exe).Source -ArgumentList @('-m','evomind_runtime.http_server','--workspace',$fixture,'--host','127.0.0.1','--port',[string]$runtimePort) -WorkingDirectory $fixture -WindowStyle Hidden -PassThru
  $tokenPath=Join-Path $fixture 'workspace\runtime\runtime.token';$deadline=[DateTime]::UtcNow.AddSeconds(30)
  do {Start-Sleep -Milliseconds 200} while(-not(Test-Path -LiteralPath $tokenPath)-and[DateTime]::UtcNow-lt$deadline-and-not$runtimeProcess.HasExited)
  if(-not(Test-Path -LiteralPath $tokenPath)){throw 'AUTH_ARTIFACT_RUNTIME_TOKEN_MISSING'}
  $webProcess=Start-Process -FilePath (Get-Command node.exe).Source -ArgumentList @('server.js') -WorkingDirectory $extract -WindowStyle Hidden -PassThru
  $handler=[Net.Http.HttpClientHandler]::new();$handler.AllowAutoRedirect=$false
  $client=[Net.Http.HttpClient]::new($handler);$client.Timeout=[TimeSpan]::FromSeconds(15)
  $deadline=[DateTime]::UtcNow.AddSeconds(60);$health=$null
  do {Start-Sleep -Milliseconds 250;try{$health=Send-Http $client 'GET' "http://127.0.0.1:$webPort/api/healthz" "127.0.0.1:$webPort" '' $null '' '' ''}catch{$health=$null}} while(($null -eq $health -or $health.status -ne 200)-and[DateTime]::UtcNow-lt$deadline-and-not$webProcess.HasExited)
  if($null -eq $health -or $health.status -ne 200){throw 'AUTH_ARTIFACT_WEB_HEALTH_FAILED'}
  $publicLogin=Send-Http $client 'GET' "http://127.0.0.1:$webPort/login" 'evomind.zhjjq.tech' '' $null '' '' ''
  if($publicLogin.status -ne 200 -or $publicLogin.body -notmatch '登录 EvoMind'){throw 'AUTH_ARTIFACT_PUBLIC_LOGIN_PAGE_FAILED'}
  $publicHome=Send-Http $client 'GET' "http://127.0.0.1:$webPort/" 'evomind.zhjjq.tech' '' $null '' '' ''
  if($publicHome.status -ne 307 -or $publicHome.location -notmatch '^https://evomind\.zhjjq\.tech/login\?next=(?:%2F|/)$'){
    $sample=($publicHome.body -replace '\s+',' ');if($sample.Length-gt240){$sample=$sample.Substring(0,240)}
    throw "AUTH_ARTIFACT_PUBLIC_REDIRECT_FAILED:$($publicHome.status):$($publicHome.location):$sample"
  }
  $wrongBody=[Text.Encoding]::UTF8.GetBytes((ConvertTo-Json @{username='admin';password='fixture-wrong-password';next='/'} -Compress))
  $wrong=Send-Http $client 'POST' "http://127.0.0.1:$webPort/api/auth/login" 'evomind.zhjjq.tech' 'https://evomind.zhjjq.tech' $wrongBody 'application/json' '' ''
  if($wrong.status -ne 401){throw 'AUTH_ARTIFACT_WRONG_PASSWORD_STATUS_FAILED'}
  $publicBootstrap=Send-Http $client 'GET' "http://127.0.0.1:$webPort/api/session/bootstrap" 'evomind.zhjjq.tech' '' $null '' '' ''
  if($publicBootstrap.status -ne 404){throw 'AUTH_ARTIFACT_PUBLIC_BOOTSTRAP_LEAK'}
  $loopBody=[Text.Encoding]::UTF8.GetBytes((ConvertTo-Json @{token=$bootstrapToken} -Compress))
  $loop=Send-Http $client 'POST' "http://127.0.0.1:$webPort/api/session/bootstrap" "127.0.0.1:$webPort" "http://127.0.0.1:$webPort" $loopBody 'application/json' '' ''
  $loopJson=Json $loop.body
  if($loop.status -ne 200 -or -not[bool]$loopJson.ok -or [string]::IsNullOrWhiteSpace([string]$loopJson.csrf_token) -or [string]::IsNullOrWhiteSpace($loop.set_cookie)){throw 'AUTH_ARTIFACT_LOOPBACK_BOOTSTRAP_FAILED'}
  $cookie=($loop.set_cookie -split ';',2)[0];$csrf=[string]$loopJson.csrf_token
  $data=New-Object byte[] ((16*1024*1024)+4096);[Array]::Fill($data,[byte]0x5A);$data[0]=0x45;$data[$data.Length-1]=0x21;$dataSha=ShaBytes $data
  $createBody=[Text.Encoding]::UTF8.GetBytes((ConvertTo-Json @{name='artifact-multichunk-smoke.bin';total_bytes=$data.Length;media_type='application/octet-stream';sha256=$dataSha} -Compress))
  $create=Send-Http $client 'POST' "http://127.0.0.1:$webPort/api/assistant/uploads" "127.0.0.1:$webPort" "http://127.0.0.1:$webPort" $createBody 'application/json' $cookie '' $csrf
  $createJson=Json $create.body
  if($create.status -ne 201 -or [string]::IsNullOrWhiteSpace([string]$createJson.id)){throw 'AUTH_ARTIFACT_UPLOAD_CREATE_FAILED'}
  $chunkSize=[int]$createJson.chunk_size;$chunkCount=[int][math]::Ceiling($data.Length/[double]$chunkSize)
  if($chunkCount-lt2){throw 'AUTH_ARTIFACT_MULTICHUNK_CONTRACT_MISSING'}
  foreach($index in 0..($chunkCount-1)){
    $offset=$index*$chunkSize;$length=[math]::Min($chunkSize,$data.Length-$offset);$chunkBytes=New-Object byte[] $length;[Buffer]::BlockCopy($data,$offset,$chunkBytes,0,$length);$chunkSha=ShaBytes $chunkBytes
    $chunk=Send-Http $client 'PUT' "http://127.0.0.1:$webPort/api/assistant/uploads/$([uri]::EscapeDataString([string]$createJson.id))/chunks/$index" "127.0.0.1:$webPort" "http://127.0.0.1:$webPort" $chunkBytes 'application/octet-stream' $cookie $chunkSha $csrf
    if($chunk.status -ne 200){
      $runtimeToken=(Get-Content -LiteralPath $tokenPath -Raw).Trim()
      $direct=Send-Http $client 'PUT' "http://127.0.0.1:$runtimePort/v1/uploads/$([uri]::EscapeDataString([string]$createJson.id))/chunks/$index" "127.0.0.1:$runtimePort" '' $chunkBytes 'application/octet-stream' '' $chunkSha '' $runtimeToken
      $runtimeToken=$null;$sample=($chunk.body-replace'\s+',' ');if($sample.Length-gt200){$sample=$sample.Substring(0,200)}
      throw "AUTH_ARTIFACT_BINARY_CHUNK_FAILED:${index}:$($chunk.status):chunk_size=$($chunkSize):chunk_bytes=$($chunkBytes.Length):total=$($data.Length):direct=$($direct.status):$sample"
    }
  }
  $completeBody=[Text.Encoding]::UTF8.GetBytes('{}')
  $complete=Send-Http $client 'POST' "http://127.0.0.1:$webPort/api/assistant/uploads/$([uri]::EscapeDataString([string]$createJson.id))/complete" "127.0.0.1:$webPort" "http://127.0.0.1:$webPort" $completeBody 'application/json' $cookie '' $csrf
  $completeJson=Json $complete.body
  if($complete.status -ne 200 -or [string]$completeJson.attachment.sha256 -ne $dataSha){throw 'AUTH_ARTIFACT_UPLOAD_COMPLETE_FAILED'}
  $payload=[ordered]@{schema='evomind.super_agent_public_auth_artifact_smoke.v1';status='passed';build_id=$ExpectedBuildId;public_login_page=200;public_home_redirect=307;wrong_password=401;public_bootstrap=404;loopback_bootstrap=200;binary_upload_chunks=$chunkCount;binary_upload_complete=200;artifact_bytes=$data.Length;artifact_sha256=$dataSha;secrets_printed=$false;network_accessed=$false;gpu_touched=$false}
  $parent=Split-Path -Parent $result;if($parent){New-Item -ItemType Directory -Path $parent -Force|Out-Null};[IO.File]::WriteAllText($result,(($payload|ConvertTo-Json -Depth 8)+"`n"),[Text.UTF8Encoding]::new($false));$payload|ConvertTo-Json -Depth 8 -Compress
} finally {
  foreach($process in @($webProcess,$runtimeProcess)){if($process-and-not$process.HasExited){Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue;try{$process.WaitForExit()}catch{}}}
  if($client){$client.Dispose()}
  foreach($name in $environment){if($null -eq $saved[$name]){Remove-Item "Env:$name" -ErrorAction SilentlyContinue}else{[Environment]::SetEnvironmentVariable($name,[string]$saved[$name],'Process')}}
  if(Test-Path -LiteralPath $fixture){Remove-Item -LiteralPath $fixture -Recurse -Force}
}
