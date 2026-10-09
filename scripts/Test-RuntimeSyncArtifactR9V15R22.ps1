param(
  [string]$ExtractedRoot='D:\EV12\runtime-sync-preflight-r9',
  [string]$ResultPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-v15r22-r9-runtime-sync-preflight-20260817.json'
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

function Free-Port{$listener=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0);try{$listener.Start();[int]$listener.LocalEndpoint.Port}finally{$listener.Stop()}}
function Sha([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
function Tree([string]$Path){
  $full=[IO.Path]::GetFullPath($Path).TrimEnd('\')
  $rows=@(Get-ChildItem -LiteralPath $full -Recurse -File -Force|Where-Object{$_.Name-notmatch'\.(pyc|pyo)$'-and$_.FullName-notmatch'[\\/]__pycache__[\\/]'}|Sort-Object FullName|ForEach-Object{"$($_.FullName.Substring($full.Length).TrimStart('\').Replace('\','/'))|$([long]$_.Length)|$(Sha $_.FullName)"})
  $hash=[Security.Cryptography.SHA256]::Create()
  try{$digest=([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes(($rows-join"`n"))))).Replace('-','').ToLowerInvariant()}finally{$hash.Dispose()}
  [ordered]@{file_count=$rows.Count;sha256=$digest}
}

$root=[IO.Path]::GetFullPath($ExtractedRoot).TrimEnd('\')
$result=[IO.Path]::GetFullPath($ResultPath)
if(-not(Test-Path -LiteralPath $root -PathType Container)){throw 'RUNTIME_SYNC_PREFLIGHT_ROOT_MISSING'}
if(Test-Path -LiteralPath $result){throw 'RUNTIME_SYNC_PREFLIGHT_RESULT_EXISTS'}
$runtimeRoot=Join-Path $root 'support\python-runtime'
$runtimeTree=Tree (Join-Path $runtimeRoot 'evomind_runtime')
$xsciTree=Tree (Join-Path $runtimeRoot 'xsci')
if($runtimeTree.file_count -ne 14 -or $runtimeTree.sha256 -ne '8b635306d330d272a1719fd30a66baf616e9b91069f35229b306cc1cac526b30'){throw 'RUNTIME_SYNC_PREFLIGHT_RUNTIME_TREE_REJECTED'}
if($xsciTree.file_count -ne 54 -or $xsciTree.sha256 -ne '9ab47525dcff3eb9a3e77e7e7e973fd031837415a34357399ebdbb4a0fb40715'){throw 'RUNTIME_SYNC_PREFLIGHT_XSCI_TREE_REJECTED'}

$fixture=Join-Path 'D:\EV12' ('runtime-sync-metadata-'+[guid]::NewGuid().ToString('N'))
$fixture=[IO.Path]::GetFullPath($fixture)
if(-not$fixture.StartsWith('D:\EV12\',[StringComparison]::OrdinalIgnoreCase)){throw 'RUNTIME_SYNC_PREFLIGHT_FIXTURE_PATH_REJECTED'}
$process=$null
$environment=@('PYTHONPATH','WORKSTATION_ROOT','WORKSTATION_DATA_DIR','EVOMIND_RUNTIME_PORT')
$saved=@{};foreach($name in $environment){$saved[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
try{
  New-Item -ItemType Directory -Path $fixture -Force|Out-Null
  $port=Free-Port
  $env:PYTHONPATH=$runtimeRoot;$env:WORKSTATION_ROOT=$fixture;$env:WORKSTATION_DATA_DIR=$fixture;$env:EVOMIND_RUNTIME_PORT=[string]$port
  $process=Start-Process -FilePath (Get-Command python.exe).Source -ArgumentList @('-m','evomind_runtime.http_server','--workspace',$fixture,'--host','127.0.0.1','--port',[string]$port) -WorkingDirectory $fixture -WindowStyle Hidden -PassThru
  $tokenPath=Join-Path $fixture 'workspace\runtime\runtime.token';$deadline=[DateTime]::UtcNow.AddSeconds(30)
  do{Start-Sleep -Milliseconds 200}while(-not(Test-Path -LiteralPath $tokenPath)-and[DateTime]::UtcNow-lt$deadline-and-not$process.HasExited)
  if(-not(Test-Path -LiteralPath $tokenPath -PathType Leaf)){throw 'RUNTIME_SYNC_PREFLIGHT_TOKEN_MISSING'}
  $headers=@{Authorization=('Bearer '+(Get-Content -LiteralPath $tokenPath -Raw).Trim());'Content-Type'='application/json'}
  $identity=[ordered]@{tenant_id='tenant_0123456789abcdef01234567';owner_principal_id='admin';job_id=91699;credential_profile='tenant_0123456789abcdef01234567_job91699_g15';allocation_generation=15;profile_instance_id='8d06ef65-129f-4845-a327-79580f3bc6ef';allocation_binding_id='aimslab-job91699-20260817T092437'}
  $sessionId='session_runtime_sync_'+[guid]::NewGuid().ToString('N')
  $created=Invoke-RestMethod "http://127.0.0.1:$port/v1/sessions" -Method Post -Headers $headers -Body (([ordered]@{session_id=$sessionId;objective='metadata roundtrip';workspace_root=$fixture;metadata=[ordered]@{managed_hpc_identity=$identity}}|ConvertTo-Json -Depth 6 -Compress)) -TimeoutSec 10
  $loaded=Invoke-RestMethod "http://127.0.0.1:$port/v1/sessions/$sessionId" -Headers @{Authorization=$headers.Authorization} -TimeoutSec 10
  $actual=$loaded.metadata.managed_hpc_identity
  foreach($name in $identity.Keys){if([string]$actual.$name-ne[string]$identity[$name]){throw "RUNTIME_SYNC_PREFLIGHT_METADATA_MISMATCH:$name"}}
  $payload=[ordered]@{schema='evomind.runtime_sync_preflight.v1';status='passed';runtime_tree=$runtimeTree;xsci_tree=$xsciTree;metadata_roundtrip='passed';session_id=$sessionId;external_network_accessed=$false;gpu_touched=$false;completed_at_utc=[DateTime]::UtcNow.ToString('o')}
  $parent=Split-Path -Parent $result;if($parent){New-Item -ItemType Directory -Path $parent -Force|Out-Null}
  [IO.File]::WriteAllText($result,(($payload|ConvertTo-Json -Depth 8)+"`n"),[Text.UTF8Encoding]::new($false))
  $payload|ConvertTo-Json -Depth 8 -Compress
}finally{
  if($process-and-not$process.HasExited){Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue;try{$process.WaitForExit()}catch{}}
  foreach($name in $environment){if($null-eq$saved[$name]){Remove-Item "Env:$name" -ErrorAction SilentlyContinue}else{[Environment]::SetEnvironmentVariable($name,[string]$saved[$name],'Process')}}
  if(Test-Path -LiteralPath $fixture){$resolved=[IO.Path]::GetFullPath($fixture);if(-not$resolved.StartsWith('D:\EV12\runtime-sync-metadata-',[StringComparison]::OrdinalIgnoreCase)){throw 'RUNTIME_SYNC_PREFLIGHT_CLEANUP_REJECTED'};Remove-Item -LiteralPath $resolved -Recurse -Force}
}
