param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [string]$OutputPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-v15r22-r9-runtime-sync-production-20260817.json'
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$hostAlias='evomind-shanghai'
$remote="C:\SecureInput\runtime-sync-$TransactionId"
$build='overlay-super-agent-byoa-v15r22-r9-92abb43074a5'
$webSha='80ad1bd7ad6a0ef64fd81e791a5e7f3265f779545b30a8bc6189a6b6975f8cff'
$runnerSha='0cf07f998a65a7f771ee01f322cc0331527f6ad1b754725fab782b4f4c2c957f'
$expectedRuntimeSha='8b635306d330d272a1719fd30a66baf616e9b91069f35229b306cc1cac526b30'
# Windows PowerShell 5.1 sorts the 54-file xsci tree differently from PowerShell 7.
# The remote transaction also verifies every ZIP-manifest entry before comparing
# the extracted and installed trees, so this value is the authoritative remote
# tree digest rather than the local preflight digest.
$expectedXsciSha='47c02fe37d0bf54b722b083223b68feeb30a1d5fed3decc15c93d957b92131ef'
$web='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\web-overlay-super-agent-byoa-v15r22-r9-20260817.zip'
$runner=Join-Path $PSScriptRoot 'Deploy-RuntimeSyncR1Remote.ps1'
$output=[IO.Path]::GetFullPath($OutputPath)
$errorOutput=[IO.Path]::ChangeExtension($output,'.error.json')

foreach($path in @($output,$errorOutput)){if(Test-Path -LiteralPath $path){throw "SUPER_AGENT_RUNTIME_SYNC_OUTPUT_EXISTS:$path"}}
foreach($item in @(
  [pscustomobject]@{path=$web;sha=$webSha},
  [pscustomobject]@{path=$runner;sha=$runnerSha}
)){
  if(-not(Test-Path -LiteralPath $item.path -PathType Leaf)){throw "SUPER_AGENT_RUNTIME_SYNC_LOCAL_INPUT_MISSING:$($item.path)"}
  if((Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash.ToLowerInvariant()-ne$item.sha){throw "SUPER_AGENT_RUNTIME_SYNC_LOCAL_HASH_REJECTED:$($item.path)"}
}

$create="if(Test-Path -LiteralPath '$remote'){throw 'SUPER_AGENT_RUNTIME_SYNC_REMOTE_EXISTS'};New-Item -ItemType Directory -Path '$remote'|Out-Null"
$create|ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if($LASTEXITCODE-ne0){throw 'SUPER_AGENT_RUNTIME_SYNC_REMOTE_CREATE_FAILED'}
foreach($item in @(
  [pscustomobject]@{path=$web;name='web.zip'},
  [pscustomobject]@{path=$runner;name='Deploy-RuntimeSyncR1Remote.ps1'}
)){
  scp $item.path "${hostAlias}:$remote\$($item.name)"
  if($LASTEXITCODE-ne0){throw "SUPER_AGENT_RUNTIME_SYNC_UPLOAD_FAILED:$($item.name)"}
}
$verify="`$files=@([pscustomobject]@{name='web.zip';sha='$webSha'},[pscustomobject]@{name='Deploy-RuntimeSyncR1Remote.ps1';sha='$runnerSha'});foreach(`$i in `$files){`$p=Join-Path '$remote' `$i.name;if((Get-FileHash -LiteralPath `$p -Algorithm SHA256).Hash.ToLowerInvariant()-ne`$i.sha){throw 'SUPER_AGENT_RUNTIME_SYNC_REMOTE_HASH_REJECTED'}};`$tokens=`$null;`$errors=`$null;[Management.Automation.Language.Parser]::ParseFile((Join-Path '$remote' 'Deploy-RuntimeSyncR1Remote.ps1'),[ref]`$tokens,[ref]`$errors)|Out-Null;if(@(`$errors).Count){throw 'SUPER_AGENT_RUNTIME_SYNC_REMOTE_AST_REJECTED'};[ordered]@{status='staged_verified';transaction_id='$TransactionId'}|ConvertTo-Json -Compress"
$staged=$verify|ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if($LASTEXITCODE-ne0){throw 'SUPER_AGENT_RUNTIME_SYNC_REMOTE_VERIFY_FAILED'}
$staged
ssh $hostAlias "powershell.exe -NoProfile -NonInteractive -File $remote\Deploy-RuntimeSyncR1Remote.ps1 -TransactionId $TransactionId -RemoteRoot $remote -WebSha256 $webSha -ExpectedBuildId $build"
if($LASTEXITCODE-ne0){
  scp "${hostAlias}:$remote\transaction-$TransactionId-error.json" $errorOutput 2>$null
  throw 'SUPER_AGENT_RUNTIME_SYNC_TRANSACTION_FAILED'
}
scp "${hostAlias}:$remote\transaction-$TransactionId-result.json" $output
if($LASTEXITCODE-ne0-or-not(Test-Path -LiteralPath $output -PathType Leaf)){throw 'SUPER_AGENT_RUNTIME_SYNC_RESULT_DOWNLOAD_FAILED'}
$result=Get-Content -LiteralPath $output -Raw -Encoding UTF8|ConvertFrom-Json
if(
  [string]$result.status-ne'installed_and_verified'-or
  [string]$result.transaction_id-ne$TransactionId-or
  [string]$result.build_id-ne$build-or
  [string]$result.web_sha256-ne$webSha-or
  [int]$result.installed_runtime_trees.evomind_runtime.file_count -ne 14 -or
  [string]$result.installed_runtime_trees.evomind_runtime.sha256-ne$expectedRuntimeSha-or
  [int]$result.installed_runtime_trees.xsci.file_count -ne 54 -or
  [string]$result.installed_runtime_trees.xsci.sha256-ne$expectedXsciSha-or
  -not[bool]$result.hpc_binding_unchanged-or
  -not[bool]$result.byoa_stop_window_unchanged-or
  -not[bool]$result.profile_unchanged-or
  [bool]$result.hpc_accessed-or
  [bool]$result.gpu_touched
){throw 'SUPER_AGENT_RUNTIME_SYNC_RESULT_REJECTED'}
[ordered]@{status='installed_and_verified';transaction_id=$TransactionId;build_id=$build;result_path=$output;result_sha256=(Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()}|ConvertTo-Json -Compress
