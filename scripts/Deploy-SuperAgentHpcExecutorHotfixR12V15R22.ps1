param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [string]$OutputPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-v15r22-r12-hpc-executor-production-r2-20260817.json'
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$hostAlias='evomind-shanghai'
$remote="C:\SecureInput\hpc-executor-hotfix-$TransactionId"
$build='overlay-super-agent-byoa-v15r22-r9-92abb43074a5'
$hotfixSha='3a5f509a4c15ea69bf63c00bd11926cec797a1538bed136445493faabd4476e5'
$runnerSha='12436f43e5674ea8323ea79511424b15961d9fa73dc581793b73ad97781546f6'
$sourceManifestSha='bcd8a9f995b191b142db8af8a7bcc1be5a9193027316b65856587b8762c961a2'
$toolsSha='17b41a1c8ba0547309b510cb8aa8d0249714d1201631b0ab3db801458ddc6ab4'
$hpcRuntimeSha='866802b9aae6224eb547e705c47f2e37efdd79470693185c2a5db32c9583bba9'
$hotfix='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-hpc-executor-hotfix-v15r22-r12-20260817.zip'
$runner=Join-Path $PSScriptRoot 'Deploy-HpcExecutorHotfixR1Remote.ps1'
$output=[IO.Path]::GetFullPath($OutputPath)
$errorOutput=[IO.Path]::ChangeExtension($output,'.error.json')

foreach($path in @($output,$errorOutput)){if(Test-Path -LiteralPath $path){throw "SUPER_AGENT_HPC_EXECUTOR_OUTPUT_EXISTS:$path"}}
foreach($item in @([pscustomobject]@{path=$hotfix;sha=$hotfixSha},[pscustomobject]@{path=$runner;sha=$runnerSha})){
  if(-not(Test-Path -LiteralPath $item.path -PathType Leaf)){throw "SUPER_AGENT_HPC_EXECUTOR_LOCAL_INPUT_MISSING:$($item.path)"}
  if((Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash.ToLowerInvariant()-ne$item.sha){throw "SUPER_AGENT_HPC_EXECUTOR_LOCAL_HASH_REJECTED:$($item.path)"}
}
$create="if(Test-Path -LiteralPath '$remote'){throw 'SUPER_AGENT_HPC_EXECUTOR_REMOTE_EXISTS'};New-Item -ItemType Directory -Path '$remote'|Out-Null"
$create|ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if($LASTEXITCODE-ne0){throw 'SUPER_AGENT_HPC_EXECUTOR_REMOTE_CREATE_FAILED'}
foreach($item in @([pscustomobject]@{path=$hotfix;name='hotfix.zip'},[pscustomobject]@{path=$runner;name='Deploy-HpcExecutorHotfixR1Remote.ps1'})){
  scp $item.path "${hostAlias}:$remote\$($item.name)"
  if($LASTEXITCODE-ne0){throw "SUPER_AGENT_HPC_EXECUTOR_UPLOAD_FAILED:$($item.name)"}
}
$verify="`$files=@([pscustomobject]@{name='hotfix.zip';sha='$hotfixSha'},[pscustomobject]@{name='Deploy-HpcExecutorHotfixR1Remote.ps1';sha='$runnerSha'});foreach(`$i in `$files){`$p=Join-Path '$remote' `$i.name;if((Get-FileHash -LiteralPath `$p -Algorithm SHA256).Hash.ToLowerInvariant()-ne`$i.sha){throw 'SUPER_AGENT_HPC_EXECUTOR_REMOTE_HASH_REJECTED'}};`$tokens=`$null;`$errors=`$null;[Management.Automation.Language.Parser]::ParseFile((Join-Path '$remote' 'Deploy-HpcExecutorHotfixR1Remote.ps1'),[ref]`$tokens,[ref]`$errors)|Out-Null;if(@(`$errors).Count){throw 'SUPER_AGENT_HPC_EXECUTOR_REMOTE_AST_REJECTED'};[ordered]@{status='staged_verified';transaction_id='$TransactionId'}|ConvertTo-Json -Compress"
$staged=$verify|ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if($LASTEXITCODE-ne0){throw 'SUPER_AGENT_HPC_EXECUTOR_REMOTE_VERIFY_FAILED'}
$staged
ssh $hostAlias "powershell.exe -NoProfile -NonInteractive -File $remote\Deploy-HpcExecutorHotfixR1Remote.ps1 -TransactionId $TransactionId -RemoteRoot $remote -HotfixSha256 $hotfixSha -ExpectedBuildId $build"
if($LASTEXITCODE-ne0){scp "${hostAlias}:$remote\transaction-$TransactionId-error.json" $errorOutput 2>$null;throw 'SUPER_AGENT_HPC_EXECUTOR_TRANSACTION_FAILED'}
scp "${hostAlias}:$remote\transaction-$TransactionId-result.json" $output
if($LASTEXITCODE-ne0-or-not(Test-Path -LiteralPath $output -PathType Leaf)){throw 'SUPER_AGENT_HPC_EXECUTOR_RESULT_DOWNLOAD_FAILED'}
$result=Get-Content -LiteralPath $output -Raw -Encoding UTF8|ConvertFrom-Json
if(
  [string]$result.status-ne'installed_and_verified'-or
  [string]$result.transaction_id-ne$TransactionId-or
  [string]$result.build_id-ne$build-or
  [string]$result.hotfix_sha256-ne$hotfixSha-or
  [string]$result.source_manifest_sha256-ne$sourceManifestSha-or
  [string]$result.tools_sha256-ne$toolsSha-or
  [string]$result.hpc_runtime_sha256-ne$hpcRuntimeSha-or
  [int]$result.updated_target_count-ne2-or
  -not[bool]$result.tracked_release_source_unchanged-or
  -not[bool]$result.hpc_binding_unchanged-or
  -not[bool]$result.byoa_tree_unchanged-or
  -not[bool]$result.profile_unchanged-or
  [int]$result.job_id-ne91699-or
  [int]$result.allocation_generation-ne15-or
  [string]$result.binding_state-ne'active'-or
  [bool]$result.hpc_accessed-or
  [bool]$result.gpu_touched-or
  [bool]$result.training_started-or
  [int]$result.kaggle_submissions-ne0
){throw 'SUPER_AGENT_HPC_EXECUTOR_RESULT_REJECTED'}
[ordered]@{status='installed_and_verified';transaction_id=$TransactionId;build_id=$build;result_path=$output;result_sha256=(Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()}|ConvertTo-Json -Compress
