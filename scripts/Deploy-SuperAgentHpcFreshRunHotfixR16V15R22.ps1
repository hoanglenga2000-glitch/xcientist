param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [string]$OutputPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-v15r22-r16-hpc-fresh-run-production-20260817.json'
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$hostAlias='evomind-shanghai'
$remote="C:\SecureInput\hpc-executor-overlay-hotfix-$TransactionId"
$build='overlay-super-agent-byoa-v15r22-r10-34f8af0e0451'
$hotfixSha='171ebca56062dc36e5c25dcac49497eee08a525792d223cd64b42dcf0c9e4253'
$runnerSha='299f294f32c4cb5f7089f353e7c2ef57d11cb6f08c049bef8871f5442c88c89b'
$sourceManifestSha='801cc1966a5c121a9f2dc6a05bfdc872e955b685e4681c5689b7d5ca6309091f'
$toolsSha='de1751cb5338e46d7baf9955af2427e3f6cdaa8e33101235372e3e6d0669c9ea'
$oldOverlaySha='b4b9bb6f3484e6cc8b7c9b97a6d074d02af65f9cb05bf8530a8445262f924cfb'
$overlaySha='605f46378f14f70a8a29ea8fe9041d1a731c70c946b9a936cf6c700183500a4c'
$hotfix='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-hpc-fresh-run-overlay-hotfix-v15r22-r16-20260817.zip'
$runner=Join-Path $PSScriptRoot 'Deploy-HpcExecutorOverlayHotfixR1Remote.ps1'
$output=[IO.Path]::GetFullPath($OutputPath)
$errorOutput=[IO.Path]::ChangeExtension($output,'.error.json')
foreach($path in @($output,$errorOutput)){if(Test-Path -LiteralPath $path){throw "SUPER_AGENT_HPC_FRESH_RUN_OUTPUT_EXISTS:$path"}}
foreach($item in @(
  [pscustomobject]@{path=$hotfix;sha=$hotfixSha},
  [pscustomobject]@{path=$runner;sha=$runnerSha}
)){
  if(-not(Test-Path -LiteralPath $item.path -PathType Leaf)){throw "SUPER_AGENT_HPC_FRESH_RUN_INPUT_MISSING:$($item.path)"}
  if((Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash.ToLowerInvariant()-ne$item.sha){throw "SUPER_AGENT_HPC_FRESH_RUN_HASH_REJECTED:$($item.path)"}
}
$create="if(Test-Path -LiteralPath '$remote'){throw 'SUPER_AGENT_HPC_FRESH_RUN_REMOTE_EXISTS'};New-Item -ItemType Directory -Path '$remote'|Out-Null"
$create|ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if($LASTEXITCODE-ne0){throw 'SUPER_AGENT_HPC_FRESH_RUN_REMOTE_CREATE_FAILED'}
foreach($item in @(
  [pscustomobject]@{path=$hotfix;name='hotfix.zip'},
  [pscustomobject]@{path=$runner;name='Deploy-HpcExecutorOverlayHotfixR1Remote.ps1'}
)){
  scp $item.path "${hostAlias}:$remote\$($item.name)"
  if($LASTEXITCODE-ne0){throw "SUPER_AGENT_HPC_FRESH_RUN_UPLOAD_FAILED:$($item.name)"}
}
$verify="`$files=@([pscustomobject]@{name='hotfix.zip';sha='$hotfixSha'},[pscustomobject]@{name='Deploy-HpcExecutorOverlayHotfixR1Remote.ps1';sha='$runnerSha'});foreach(`$i in `$files){`$p=Join-Path '$remote' `$i.name;if((Get-FileHash -LiteralPath `$p -Algorithm SHA256).Hash.ToLowerInvariant()-ne`$i.sha){throw 'SUPER_AGENT_HPC_FRESH_RUN_REMOTE_HASH_REJECTED'}};`$tokens=`$null;`$errors=`$null;[Management.Automation.Language.Parser]::ParseFile((Join-Path '$remote' 'Deploy-HpcExecutorOverlayHotfixR1Remote.ps1'),[ref]`$tokens,[ref]`$errors)|Out-Null;if(@(`$errors).Count){throw 'SUPER_AGENT_HPC_FRESH_RUN_REMOTE_AST_REJECTED'};[ordered]@{status='staged_verified';transaction_id='$TransactionId'}|ConvertTo-Json -Compress"
$staged=$verify|ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if($LASTEXITCODE-ne0){throw 'SUPER_AGENT_HPC_FRESH_RUN_REMOTE_VERIFY_FAILED'}
$staged
ssh $hostAlias "powershell.exe -NoProfile -NonInteractive -File $remote\Deploy-HpcExecutorOverlayHotfixR1Remote.ps1 -TransactionId $TransactionId -RemoteRoot $remote -HotfixSha256 $hotfixSha -ExpectedBuildId $build -ExpectedOldToolsSha256 $toolsSha -ExpectedOldOverlaySha256 $oldOverlaySha"
if($LASTEXITCODE-ne0){
  scp "${hostAlias}:$remote\transaction-$TransactionId-error.json" $errorOutput 2>$null
  throw 'SUPER_AGENT_HPC_FRESH_RUN_TRANSACTION_FAILED'
}
scp "${hostAlias}:$remote\transaction-$TransactionId-result.json" $output
if($LASTEXITCODE-ne0-or-not(Test-Path -LiteralPath $output -PathType Leaf)){throw 'SUPER_AGENT_HPC_FRESH_RUN_RESULT_DOWNLOAD_FAILED'}
$result=Get-Content -LiteralPath $output -Raw -Encoding UTF8|ConvertFrom-Json
if([string]$result.status-ne'installed_and_verified'-or[string]$result.transaction_id-ne$TransactionId-or[string]$result.build_id-ne$build-or[string]$result.hotfix_sha256-ne$hotfixSha-or[string]$result.source_manifest_sha256-ne$sourceManifestSha-or[string]$result.old_tools_sha256-ne$toolsSha-or[string]$result.old_overlay_sha256-ne$oldOverlaySha-or[string]$result.tools_sha256-ne$toolsSha-or[string]$result.overlay_sha256-ne$overlaySha-or[int]$result.updated_target_count-ne2-or-not[bool]$result.hpc_binding_unchanged-or-not[bool]$result.byoa_tree_unchanged-or-not[bool]$result.profile_unchanged-or[int]$result.job_id-ne91699-or[int]$result.allocation_generation-ne15-or[string]$result.binding_state-ne'active'-or[bool]$result.hpc_accessed-or[bool]$result.gpu_touched-or[bool]$result.training_started-or[int]$result.kaggle_submissions-ne0){throw 'SUPER_AGENT_HPC_FRESH_RUN_RESULT_REJECTED'}
[ordered]@{status='installed_and_verified';transaction_id=$TransactionId;build_id=$build;result_path=$output;result_sha256=(Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()}|ConvertTo-Json -Compress
