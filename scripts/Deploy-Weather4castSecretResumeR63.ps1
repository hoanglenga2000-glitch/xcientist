param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [string]$OutputPath='D:/AI-Outputs/EvoMind-Cloud-Deploy/artifacts/weather4cast-secret-resume-r67-production-20260826.json'
)

Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$hostAlias='evomind-shanghai'
$remote="C:\SecureInput\capability-web-$TransactionId"
$oldBuild='overlay-weather4cast-secret-resume-r64-a46876190425'
$newBuild='overlay-weather4cast-secret-resume-r67-0896a68d234b'
$webSha='998159c68b1b3c56e4b18929201887f495547dcd8d45f87c6cdc1566353c9460'
$adminSha='18bb9c74b5b0759f71f6fbd2cbb5f3f4b746e92396519369099673867f59ef0d'
$runnerSha='a7d4c00faafd09a41af1286bf9471a66ea4dc0a79f9b474e659272d9cda7337c'
$web='D:/AI-Outputs/EvoMind-Cloud-Deploy/artifacts/web-overlay-20260826-weather4cast-secret-resume-r67.zip'
$admin='D:/AI-Outputs/EvoMind-Cloud-Deploy/artifacts/evomind-admin-password.scrypt'
$runner='D:/AI-Outputs/EvoMind-Cloud-Deploy/artifacts/Deploy-CapabilityWebR2Remote.ps1'
$output=[IO.Path]::GetFullPath($OutputPath)
$errorOutput=[IO.Path]::ChangeExtension($output,'.error.json')

function Sha([string]$Path){
  (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

foreach($path in @($output,$errorOutput)){
  if(Test-Path -LiteralPath $path){throw "R63_OUTPUT_EXISTS:$path"}
}
foreach($item in @(
  [pscustomobject]@{path=$web;sha=$webSha},
  [pscustomobject]@{path=$admin;sha=$adminSha},
  [pscustomobject]@{path=$runner;sha=$runnerSha}
)){
  if(-not(Test-Path -LiteralPath $item.path -PathType Leaf)){throw "R63_LOCAL_INPUT_MISSING:$($item.path)"}
  if((Sha $item.path)-ne$item.sha){throw "R63_LOCAL_HASH_REJECTED:$($item.path)"}
}

$create="if(Test-Path -LiteralPath '$remote'){throw 'R63_REMOTE_EXISTS'};New-Item -ItemType Directory -Path '$remote'|Out-Null"
$create|ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if($LASTEXITCODE-ne0){throw 'R63_REMOTE_CREATE_FAILED'}
foreach($item in @(
  [pscustomobject]@{path=$web;name='overlay-weather4cast-secret-resume-r64-a46876190425'},
  [pscustomobject]@{path=$admin;name='overlay-weather4cast-secret-resume-r64-a46876190425'},
  [pscustomobject]@{path=$runner;name='overlay-weather4cast-secret-resume-r64-a46876190425'}
)){
  scp $item.path "${hostAlias}:$remote\$($item.name)"
  if($LASTEXITCODE-ne0){throw "R63_UPLOAD_FAILED:$($item.name)"}
}

$verify="`$files=@([pscustomobject]@{name='overlay-weather4cast-secret-resume-r64-a46876190425';sha='overlay-weather4cast-secret-resume-r64-a46876190425'},[pscustomobject]@{name='overlay-weather4cast-secret-resume-r64-a46876190425';sha='overlay-weather4cast-secret-resume-r64-a46876190425'},[pscustomobject]@{name='overlay-weather4cast-secret-resume-r64-a46876190425';sha='overlay-weather4cast-secret-resume-r64-a46876190425'});foreach(`$i in `$files){`$p=Join-Path '$remote' `$i.name;if((Get-FileHash -LiteralPath `$p -Algorithm SHA256).Hash.ToLowerInvariant()-ne`$i.sha){throw 'R63_REMOTE_HASH_REJECTED'}};[ordered]@{status='overlay-weather4cast-secret-resume-r64-a46876190425';transaction_id='overlay-weather4cast-secret-resume-r64-a46876190425'}|ConvertTo-Json -Compress"
$verify|ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if($LASTEXITCODE-ne0){throw 'R63_REMOTE_VERIFY_FAILED'}

ssh $hostAlias "powershell.exe -NoProfile -NonInteractive -File $remote\Deploy-CapabilityWebR2Remote.ps1 -TransactionId $TransactionId -RemoteRoot $remote -WebSha256 $webSha -OldBuildId $oldBuild -NewBuildId $newBuild"
if($LASTEXITCODE-ne0){
  scp "${hostAlias}:$remote\transaction-$TransactionId-error.json" $errorOutput 2>$null
  throw 'R63_TRANSACTION_FAILED'
}
scp "${hostAlias}:$remote\transaction-$TransactionId-result.json" $output
if($LASTEXITCODE-ne0-or-not(Test-Path -LiteralPath $output -PathType Leaf)){throw 'R63_RESULT_DOWNLOAD_FAILED'}

$result=Get-Content -LiteralPath $output -Raw -Encoding UTF8|ConvertFrom-Json
if(
  [string]$result.status-ne'installed_and_verified'-or
  [string]$result.transaction_id-ne$TransactionId-or
  [string]$result.old_build_id-ne$oldBuild-or
  [string]$result.new_build_id-ne$newBuild-or
  [string]$result.web_sha256-ne$webSha-or
  @($result.loopback.samples).Count-ne3-or
  @($result.loopback.samples|Where-Object{-not[bool]$_.exact}).Count-or
  @($result.public.samples).Count-ne3-or
  @($result.public.samples|Where-Object{-not[bool]$_.exact}).Count-or
  -not[bool]$result.hpc_binding_unchanged-or
  -not[bool]$result.byoa_stop_window_unchanged-or
  -not[bool]$result.profile_unchanged-or
  [bool]$result.hpc_accessed-or
  [bool]$result.gpu_touched-or
  [bool]$result.secrets_printed
){throw 'R63_RESULT_REJECTED'}

[ordered]@{
  status='overlay-weather4cast-secret-resume-r64-a46876190425'
  transaction_id=$TransactionId
  build_id=$newBuild
  result_path=$output
  result_sha256=Sha $output
  loopback_samples=3
  public_samples=3
  hpc_accessed=$false
  gpu_touched=$false
}|ConvertTo-Json -Compress




