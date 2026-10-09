param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [string]$OutputPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-v15r22-r11-hpc-profile-production-20260817.json'
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$hostAlias='evomind-shanghai'
$remote="C:\SecureInput\hpc-profile-hotfix-$TransactionId"
$build='overlay-super-agent-byoa-v15r22-r9-92abb43074a5'
$hotfixSha='8ea0c641a9fb2e1f4b90453ea2a50c7f301dca001be67783bdfde720549bb2ed'
$runnerSha='83e588b30adff230bae95369322cfae28cf885027605cfcc7a04d7f00a9b598c'
$sourceManifestSha='5f0485440e97aae73be811a3a21cf3ae13a8cedb1259057f206966da06753d84'
$gpuSha='4e78dcfba1f1f692cd19b1561782b9d029eb4b943eb4c7d3025ef021a3c582cd'
$terminalSha='3f8401f6804fd5b508b98722519f05bde29bfe36ad6df8de3601078e7cc85881'
$hotfix='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-hpc-profile-hotfix-v15r22-r11-20260817.zip'
$runner=Join-Path $PSScriptRoot 'Deploy-HpcProfileHotfixR1Remote.ps1'
$output=[IO.Path]::GetFullPath($OutputPath)
$errorOutput=[IO.Path]::ChangeExtension($output,'.error.json')

foreach($path in @($output,$errorOutput)){if(Test-Path -LiteralPath $path){throw "SUPER_AGENT_HPC_PROFILE_OUTPUT_EXISTS:$path"}}
foreach($item in @([pscustomobject]@{path=$hotfix;sha=$hotfixSha},[pscustomobject]@{path=$runner;sha=$runnerSha})){
  if(-not(Test-Path -LiteralPath $item.path -PathType Leaf)){throw "SUPER_AGENT_HPC_PROFILE_LOCAL_INPUT_MISSING:$($item.path)"}
  if((Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash.ToLowerInvariant()-ne$item.sha){throw "SUPER_AGENT_HPC_PROFILE_LOCAL_HASH_REJECTED:$($item.path)"}
}
$create="if(Test-Path -LiteralPath '$remote'){throw 'SUPER_AGENT_HPC_PROFILE_REMOTE_EXISTS'};New-Item -ItemType Directory -Path '$remote'|Out-Null"
$create|ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if($LASTEXITCODE-ne0){throw 'SUPER_AGENT_HPC_PROFILE_REMOTE_CREATE_FAILED'}
foreach($item in @([pscustomobject]@{path=$hotfix;name='hotfix.zip'},[pscustomobject]@{path=$runner;name='Deploy-HpcProfileHotfixR1Remote.ps1'})){
  scp $item.path "${hostAlias}:$remote\$($item.name)"
  if($LASTEXITCODE-ne0){throw "SUPER_AGENT_HPC_PROFILE_UPLOAD_FAILED:$($item.name)"}
}
$verify="`$files=@([pscustomobject]@{name='hotfix.zip';sha='$hotfixSha'},[pscustomobject]@{name='Deploy-HpcProfileHotfixR1Remote.ps1';sha='$runnerSha'});foreach(`$i in `$files){`$p=Join-Path '$remote' `$i.name;if((Get-FileHash -LiteralPath `$p -Algorithm SHA256).Hash.ToLowerInvariant()-ne`$i.sha){throw 'SUPER_AGENT_HPC_PROFILE_REMOTE_HASH_REJECTED'}};`$tokens=`$null;`$errors=`$null;[Management.Automation.Language.Parser]::ParseFile((Join-Path '$remote' 'Deploy-HpcProfileHotfixR1Remote.ps1'),[ref]`$tokens,[ref]`$errors)|Out-Null;if(@(`$errors).Count){throw 'SUPER_AGENT_HPC_PROFILE_REMOTE_AST_REJECTED'};[ordered]@{status='staged_verified';transaction_id='$TransactionId'}|ConvertTo-Json -Compress"
$staged=$verify|ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if($LASTEXITCODE-ne0){throw 'SUPER_AGENT_HPC_PROFILE_REMOTE_VERIFY_FAILED'}
$staged
ssh $hostAlias "powershell.exe -NoProfile -NonInteractive -File $remote\Deploy-HpcProfileHotfixR1Remote.ps1 -TransactionId $TransactionId -RemoteRoot $remote -HotfixSha256 $hotfixSha -ExpectedBuildId $build"
if($LASTEXITCODE-ne0){scp "${hostAlias}:$remote\transaction-$TransactionId-error.json" $errorOutput 2>$null;throw 'SUPER_AGENT_HPC_PROFILE_TRANSACTION_FAILED'}
scp "${hostAlias}:$remote\transaction-$TransactionId-result.json" $output
if($LASTEXITCODE-ne0-or-not(Test-Path -LiteralPath $output -PathType Leaf)){throw 'SUPER_AGENT_HPC_PROFILE_RESULT_DOWNLOAD_FAILED'}
$result=Get-Content -LiteralPath $output -Raw -Encoding UTF8|ConvertFrom-Json
if([string]$result.status-ne'installed_and_verified'-or[string]$result.transaction_id-ne$TransactionId-or[string]$result.build_id-ne$build-or[string]$result.hotfix_sha256-ne$hotfixSha-or[string]$result.source_manifest_sha256-ne$sourceManifestSha-or[string]$result.gpu_sha256-ne$gpuSha-or[string]$result.terminal_sha256-ne$terminalSha-or-not[bool]$result.hpc_binding_unchanged-or-not[bool]$result.byoa_tree_unchanged-or-not[bool]$result.profile_unchanged-or[int]$result.job_id-ne91699-or[int]$result.allocation_generation-ne15-or[string]$result.binding_state-ne'active'-or[bool]$result.hpc_accessed-or[bool]$result.gpu_touched-or[bool]$result.training_started-or[int]$result.kaggle_submissions-ne0){throw 'SUPER_AGENT_HPC_PROFILE_RESULT_REJECTED'}
[ordered]@{status='installed_and_verified';transaction_id=$TransactionId;build_id=$build;result_path=$output;result_sha256=(Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()}|ConvertTo-Json -Compress
