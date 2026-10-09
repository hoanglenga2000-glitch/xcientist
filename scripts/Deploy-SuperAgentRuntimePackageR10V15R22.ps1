param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [string]$OutputPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-v15r22-r10-runtime-production-r2-20260817.json'
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$hostAlias='evomind-shanghai'
$remote="C:\SecureInput\runtime-package-$TransactionId"
$build='overlay-super-agent-byoa-v15r22-r9-92abb43074a5'
$runtimeSha='a366115afd087c714578debccf161148c97fa981cf3e79bc9c116cb6e773182f'
$runnerSha='beeee30c006a68e77dd1857fe5e6893f67d018446de0e1424309d14366296448'
$sourceManifestSha='084831703ca5667af6fea856005c391d7306878cf1aca387e40ee4d5f3be3d97'
$toolsSha='35afc02f899eb6f2a01d046e74f9ce5bc00c76ccc8c042305b49deaafae1726b'
$runtime='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-runtime-hotfix-v15r22-r10-20260817.zip'
$runner=Join-Path $PSScriptRoot 'Deploy-RuntimePackageR1Remote.ps1'
$output=[IO.Path]::GetFullPath($OutputPath)
$errorOutput=[IO.Path]::ChangeExtension($output,'.error.json')

foreach($path in @($output,$errorOutput)){if(Test-Path -LiteralPath $path){throw "SUPER_AGENT_RUNTIME_PACKAGE_OUTPUT_EXISTS:$path"}}
foreach($item in @([pscustomobject]@{path=$runtime;sha=$runtimeSha},[pscustomobject]@{path=$runner;sha=$runnerSha})){
  if(-not(Test-Path -LiteralPath $item.path -PathType Leaf)){throw "SUPER_AGENT_RUNTIME_PACKAGE_LOCAL_INPUT_MISSING:$($item.path)"}
  if((Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash.ToLowerInvariant()-ne$item.sha){throw "SUPER_AGENT_RUNTIME_PACKAGE_LOCAL_HASH_REJECTED:$($item.path)"}
}
$create="if(Test-Path -LiteralPath '$remote'){throw 'SUPER_AGENT_RUNTIME_PACKAGE_REMOTE_EXISTS'};New-Item -ItemType Directory -Path '$remote'|Out-Null"
$create|ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if($LASTEXITCODE-ne0){throw 'SUPER_AGENT_RUNTIME_PACKAGE_REMOTE_CREATE_FAILED'}
foreach($item in @([pscustomobject]@{path=$runtime;name='runtime.zip'},[pscustomobject]@{path=$runner;name='Deploy-RuntimePackageR1Remote.ps1'})){
  scp $item.path "${hostAlias}:$remote\$($item.name)"
  if($LASTEXITCODE-ne0){throw "SUPER_AGENT_RUNTIME_PACKAGE_UPLOAD_FAILED:$($item.name)"}
}
$verify="`$files=@([pscustomobject]@{name='runtime.zip';sha='$runtimeSha'},[pscustomobject]@{name='Deploy-RuntimePackageR1Remote.ps1';sha='$runnerSha'});foreach(`$i in `$files){`$p=Join-Path '$remote' `$i.name;if((Get-FileHash -LiteralPath `$p -Algorithm SHA256).Hash.ToLowerInvariant()-ne`$i.sha){throw 'SUPER_AGENT_RUNTIME_PACKAGE_REMOTE_HASH_REJECTED'}};`$tokens=`$null;`$errors=`$null;[Management.Automation.Language.Parser]::ParseFile((Join-Path '$remote' 'Deploy-RuntimePackageR1Remote.ps1'),[ref]`$tokens,[ref]`$errors)|Out-Null;if(@(`$errors).Count){throw 'SUPER_AGENT_RUNTIME_PACKAGE_REMOTE_AST_REJECTED'};[ordered]@{status='staged_verified';transaction_id='$TransactionId'}|ConvertTo-Json -Compress"
$staged=$verify|ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if($LASTEXITCODE-ne0){throw 'SUPER_AGENT_RUNTIME_PACKAGE_REMOTE_VERIFY_FAILED'}
$staged
ssh $hostAlias "powershell.exe -NoProfile -NonInteractive -File $remote\Deploy-RuntimePackageR1Remote.ps1 -TransactionId $TransactionId -RemoteRoot $remote -RuntimeSha256 $runtimeSha -ExpectedBuildId $build"
if($LASTEXITCODE-ne0){scp "${hostAlias}:$remote\transaction-$TransactionId-error.json" $errorOutput 2>$null;throw 'SUPER_AGENT_RUNTIME_PACKAGE_TRANSACTION_FAILED'}
scp "${hostAlias}:$remote\transaction-$TransactionId-result.json" $output
if($LASTEXITCODE-ne0-or-not(Test-Path -LiteralPath $output -PathType Leaf)){throw 'SUPER_AGENT_RUNTIME_PACKAGE_RESULT_DOWNLOAD_FAILED'}
$result=Get-Content -LiteralPath $output -Raw -Encoding UTF8|ConvertFrom-Json
if([string]$result.status-ne'installed_and_verified'-or[string]$result.transaction_id-ne$TransactionId-or[string]$result.build_id-ne$build-or[string]$result.runtime_sha256-ne$runtimeSha-or[string]$result.source_manifest_sha256-ne$sourceManifestSha-or[string]$result.tools_sha256-ne$toolsSha-or-not[bool]$result.hpc_binding_unchanged-or-not[bool]$result.byoa_stop_window_unchanged-or-not[bool]$result.profile_unchanged-or[bool]$result.hpc_accessed-or[bool]$result.gpu_touched){throw 'SUPER_AGENT_RUNTIME_PACKAGE_RESULT_REJECTED'}
[ordered]@{status='installed_and_verified';transaction_id=$TransactionId;build_id=$build;result_path=$output;result_sha256=(Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()}|ConvertTo-Json -Compress
