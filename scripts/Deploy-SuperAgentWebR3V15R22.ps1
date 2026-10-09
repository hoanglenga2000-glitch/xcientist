param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [string]$OutputPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-v15r22-r3-production-deployment-20260816.json'
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$hostAlias='evomind-shanghai'
$remote="C:\SecureInput\capability-web-$TransactionId"
$old='overlay-super-agent-g13-v15r22-r2-60aa07e43b89'
$new='overlay-super-agent-g13-v15r22-r3-41f8553aa953'
$webSha='3d173a3943f25ecc40a10b549e15f8a491c991485dcfc1726a947d4ef9ce1a43'
$adminSha='18bb9c74b5b0759f71f6fbd2cbb5f3f4b746e92396519369099673867f59ef0d'
$runnerSha='a7d4c00faafd09a41af1286bf9471a66ea4dc0a79f9b474e659272d9cda7337c'
$web='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\web-overlay-super-agent-g13-v15r22-r3-20260816.zip'
$admin='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\evomind-admin-password.scrypt'
$runner='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\Deploy-CapabilityWebR2Remote.ps1'
$output=[IO.Path]::GetFullPath($OutputPath)
$errorOutput=[IO.Path]::ChangeExtension($output,'.error.json')
foreach($path in @($output,$errorOutput)){if(Test-Path -LiteralPath $path){throw "SUPER_AGENT_R3_OUTPUT_EXISTS:$path"}}
foreach($item in @(
  [pscustomobject]@{path=$web;sha=$webSha},
  [pscustomobject]@{path=$admin;sha=$adminSha},
  [pscustomobject]@{path=$runner;sha=$runnerSha}
)){
  if(-not(Test-Path -LiteralPath $item.path -PathType Leaf)){throw "SUPER_AGENT_R3_LOCAL_INPUT_MISSING:$($item.path)"}
  if((Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash.ToLowerInvariant()-ne$item.sha){throw "SUPER_AGENT_R3_LOCAL_HASH_REJECTED:$($item.path)"}
}
$create="if(Test-Path -LiteralPath '$remote'){throw 'SUPER_AGENT_R3_REMOTE_EXISTS'};New-Item -ItemType Directory -Path '$remote'|Out-Null"
$create|ssh $hostAlias 'powershell.exe -NoProfile -NonInteractive -Command -'
if($LASTEXITCODE-ne0){throw 'SUPER_AGENT_R3_REMOTE_CREATE_FAILED'}
foreach($item in @(
  [pscustomobject]@{path=$web;name='web.zip'},
  [pscustomobject]@{path=$admin;name='admin.scrypt'},
  [pscustomobject]@{path=$runner;name='Deploy-CapabilityWebR2Remote.ps1'}
)){
  scp $item.path "${hostAlias}:$remote\$($item.name)"
  if($LASTEXITCODE-ne0){throw "SUPER_AGENT_R3_STAGE_UPLOAD_FAILED:$($item.name)"}
}
$verify="`$files=@([pscustomobject]@{name='web.zip';sha='$webSha'},[pscustomobject]@{name='admin.scrypt';sha='$adminSha'},[pscustomobject]@{name='Deploy-CapabilityWebR2Remote.ps1';sha='$runnerSha'});foreach(`$i in `$files){`$p=Join-Path '$remote' `$i.name;if((Get-FileHash -LiteralPath `$p -Algorithm SHA256).Hash.ToLowerInvariant()-ne`$i.sha){throw 'SUPER_AGENT_R3_REMOTE_HASH_REJECTED'}};`$tokens=`$null;`$errors=`$null;[Management.Automation.Language.Parser]::ParseFile((Join-Path '$remote' 'Deploy-CapabilityWebR2Remote.ps1'),[ref]`$tokens,[ref]`$errors)|Out-Null;if(@(`$errors).Count){throw 'SUPER_AGENT_R3_REMOTE_AST_REJECTED'};[ordered]@{status='staged_verified';transaction_id='$TransactionId'}|ConvertTo-Json -Compress"
$staged=$verify|ssh $hostAlias 'powershell.exe -NoProfile -NonInteractive -Command -'
if($LASTEXITCODE-ne0){throw 'SUPER_AGENT_R3_REMOTE_VERIFY_FAILED'}
$staged
ssh $hostAlias "powershell.exe -NoProfile -NonInteractive -File $remote\Deploy-CapabilityWebR2Remote.ps1 -TransactionId $TransactionId -RemoteRoot $remote -WebSha256 $webSha -OldBuildId $old -NewBuildId $new"
if($LASTEXITCODE-ne0){
  scp "${hostAlias}:$remote\transaction-$TransactionId-error.json" $errorOutput 2>$null
  throw 'SUPER_AGENT_R3_TRANSACTION_FAILED'
}
scp "${hostAlias}:$remote\transaction-$TransactionId-result.json" $output
if($LASTEXITCODE-ne0-or-not(Test-Path -LiteralPath $output -PathType Leaf)){throw 'SUPER_AGENT_R3_RESULT_DOWNLOAD_FAILED'}
$result=Get-Content -LiteralPath $output -Raw -Encoding UTF8|ConvertFrom-Json
if([string]$result.status-ne'installed_and_verified'-or[string]$result.transaction_id-ne$TransactionId-or[string]$result.old_build_id-ne$old-or[string]$result.new_build_id-ne$new-or[string]$result.web_sha256-ne$webSha-or-not[bool]$result.hpc_binding_unchanged-or-not[bool]$result.byoa_stop_window_unchanged-or-not[bool]$result.profile_unchanged-or[bool]$result.hpc_accessed-or[bool]$result.gpu_touched){throw 'SUPER_AGENT_R3_RESULT_REJECTED'}
[ordered]@{status='installed_and_verified';transaction_id=$TransactionId;build_id=$new;result_path=$output;result_sha256=(Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()}|ConvertTo-Json -Compress
