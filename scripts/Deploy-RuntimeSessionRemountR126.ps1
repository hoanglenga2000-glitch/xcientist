param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [Parameter(Mandatory)][string]$ExpectedBuildId,
  [ValidatePattern('^(?:|run_[a-f0-9]{32})$')][string]$ExpectedRunId='',
  [string]$OutputPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\evomind-runtime-r126-session-remount-production-20260902.json',
  [string]$RuntimePath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\evomind-runtime-r126-session-remount-20260902.zip',
  [string]$SourceManifestPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\evomind-runtime-r126-session-remount-source-manifest-20260902.json',
  [string]$TestResultPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\evomind-runtime-r126-session-remount-test-results-20260902.json',
  [ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedRuntimeSha='7431700930fca5b22bcccf90770c6a94e2188d6b7c9aa95d9b3b9ff4d04b5e5e',
  [ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedSourceManifestSha='084fe6f14fca46dd4ca78669365f6c057009dd64a613be0a34d29a92ee2f5c1c',
  [ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedTestResultSha='ad1ef5aaea06fe2dcbc81527820c421aced457fcc4df2db1d8f834cf78f425fa',
  [ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedCompetitionDataSha='1e45df525064c16b6d0610e6769734e7645067941161c150adb8d9f774401c49'
)

Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$hostAlias='evomind-shanghai'
$remote="C:\SecureInput\runtime-package-$TransactionId"
$build=$ExpectedBuildId
$runtimeSha=$ExpectedRuntimeSha
$sourceManifestSha=$ExpectedSourceManifestSha
$testResultSha=$ExpectedTestResultSha
$runtime=[IO.Path]::GetFullPath($RuntimePath)
$sourceManifest=[IO.Path]::GetFullPath($SourceManifestPath)
$testResult=[IO.Path]::GetFullPath($TestResultPath)
$runner=Join-Path $PSScriptRoot 'Deploy-RuntimeSessionRemountR126Remote.ps1'
$runnerSha=(Get-FileHash -LiteralPath $runner -Algorithm SHA256).Hash.ToLowerInvariant()
$output=[IO.Path]::GetFullPath($OutputPath)
$errorOutput=[IO.Path]::ChangeExtension($output,'.error.json')

function Sha([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}

foreach($path in @($output,$errorOutput)){
  if(Test-Path -LiteralPath $path){throw "RUNTIME_R30_DEPLOY_OUTPUT_EXISTS:$path"}
}
$inputs=@(
  [pscustomobject]@{path=$runtime;sha=$runtimeSha;name='runtime.zip'},
  [pscustomobject]@{path=$sourceManifest;sha=$sourceManifestSha;name='source-manifest.json'},
  [pscustomobject]@{path=$testResult;sha=$testResultSha;name='test-results.json'},
  [pscustomobject]@{path=$runner;sha=$runnerSha;name='Deploy-RuntimeSessionRemountR126Remote.ps1'}
)
foreach($item in $inputs){
  if(-not(Test-Path -LiteralPath $item.path -PathType Leaf)){throw "RUNTIME_R30_LOCAL_INPUT_MISSING:$($item.path)"}
  if((Sha $item.path)-ne$item.sha){throw "RUNTIME_R30_LOCAL_HASH_REJECTED:$($item.path)"}
}

$sourcePayload=Get-Content -LiteralPath $sourceManifest -Raw -Encoding UTF8|ConvertFrom-Json
$testPayload=Get-Content -LiteralPath $testResult -Raw -Encoding UTF8|ConvertFrom-Json
if([string]$sourcePayload.schema-ne'evomind.super_agent_source_set.v1'-or-not[bool]$sourcePayload.frozen-or[int]$sourcePayload.file_count-ne32){
  throw 'RUNTIME_R30_SOURCE_MANIFEST_REJECTED'
}
if([string]$testPayload.status-ne'built_and_verified'-or[int]$testPayload.file_count-ne32-or@($testPayload.verification_rounds).Count-ne2-or@($testPayload.verification_rounds|Where-Object{[string]$_.status-ne'passed'-or[string]$_.cross_run_path-ne'fail_closed'-or[int]$_.pending_approvals-ne0-or[bool]$_.hpc_accessed-or[bool]$_.gpu_touched}).Count){
  throw 'RUNTIME_R30_TEST_RESULT_REJECTED'
}
$expected=@{}
foreach($name in @('assistant_runs.py','competition_data.py','runtime.py','store.py','super_agent_runtime.py','tools.py')){
  $entry=@($sourcePayload.files|Where-Object{[string]$_.path-eq"src/evomind_runtime/$name"})
  if($entry.Count-ne1){throw "RUNTIME_R30_SOURCE_ENTRY_REJECTED:$name"}
  $expected[$name]=[string]$entry[0].sha256
}
if($expected['competition_data.py']-ne$ExpectedCompetitionDataSha){throw 'RUNTIME_R30_COMPETITION_DATA_R82_DRIFT'}

$create="if(Test-Path -LiteralPath '$remote'){throw 'RUNTIME_R30_REMOTE_EXISTS'};New-Item -ItemType Directory -Path '$remote'|Out-Null"
$create|ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if($LASTEXITCODE-ne0){throw 'RUNTIME_R30_REMOTE_CREATE_FAILED'}
foreach($item in @(
  [pscustomobject]@{path=$runtime;name='runtime.zip'},
  [pscustomobject]@{path=$runner;name='Deploy-RuntimeSessionRemountR126Remote.ps1'}
)){
  scp $item.path "${hostAlias}:$remote\$($item.name)"
  if($LASTEXITCODE-ne0){throw "RUNTIME_R30_UPLOAD_FAILED:$($item.name)"}
}

$verify="`$files=@([pscustomobject]@{name='runtime.zip';sha='$runtimeSha'},[pscustomobject]@{name='Deploy-RuntimeSessionRemountR126Remote.ps1';sha='$runnerSha'});foreach(`$i in `$files){`$p=Join-Path '$remote' `$i.name;if((Get-FileHash -LiteralPath `$p -Algorithm SHA256).Hash.ToLowerInvariant()-ne`$i.sha){throw 'RUNTIME_R30_REMOTE_HASH_REJECTED'}};`$tokens=`$null;`$errors=`$null;[Management.Automation.Language.Parser]::ParseFile((Join-Path '$remote' 'Deploy-RuntimeSessionRemountR126Remote.ps1'),[ref]`$tokens,[ref]`$errors)|Out-Null;if(@(`$errors).Count){throw 'RUNTIME_R30_REMOTE_AST_REJECTED'};[ordered]@{status='staged_verified';transaction_id='$TransactionId'}|ConvertTo-Json -Compress"
$staged=$verify|ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if($LASTEXITCODE-ne0){throw 'RUNTIME_R30_REMOTE_VERIFY_FAILED'}
$staged

$targetRunArgument=if($ExpectedRunId){" -ExpectedRunId $ExpectedRunId"}else{''}
ssh $hostAlias "powershell.exe -NoProfile -NonInteractive -File $remote\Deploy-RuntimeSessionRemountR126Remote.ps1 -TransactionId $TransactionId -RemoteRoot $remote -RuntimeSha256 $runtimeSha -ExpectedBuildId $build$targetRunArgument"
if($LASTEXITCODE-ne0){
  scp "${hostAlias}:$remote\transaction-$TransactionId-error.json" $errorOutput 2>$null
  throw 'RUNTIME_R30_TRANSACTION_FAILED'
}
scp "${hostAlias}:$remote\transaction-$TransactionId-result.json" $output
if($LASTEXITCODE-ne0-or-not(Test-Path -LiteralPath $output -PathType Leaf)){throw 'RUNTIME_R30_RESULT_DOWNLOAD_FAILED'}

$result=Get-Content -LiteralPath $output -Raw -Encoding UTF8|ConvertFrom-Json
$loopback=@($result.loopback.samples)
$public=@($result.public.samples)
$targetRunValid=if($ExpectedRunId){
  [string]$result.target_run_preflight.run_id-eq$ExpectedRunId-and
  [string]$result.target_run_preflight.status-in@('blocked','failed','cancelled','completed')-and
  [int]$result.target_run_preflight.active_tool_count-eq0-and
  [int]$result.target_run_preflight.pending_approval_count-eq0
}else{
  $null-eq$result.target_run_preflight
}
if(
  [string]$result.status-ne'installed_and_verified'-or
  [string]$result.transaction_id-ne$TransactionId-or
  [string]$result.build_id-ne$build-or
  [string]$result.runtime_sha256-ne$runtimeSha-or
  [string]$result.source_manifest_sha256-ne$sourceManifestSha-or
  [string]$result.assistant_runs_sha256-ne$expected['assistant_runs.py']-or
  [string]$result.competition_data_sha256-ne$ExpectedCompetitionDataSha-or
  [string]$result.store_sha256-ne$expected['store.py']-or
  [string]$result.super_agent_runtime_sha256-ne$expected['super_agent_runtime.py']-or
  [string]$result.runtime_module_sha256-ne$expected['runtime.py']-or
  [string]$result.tools_sha256-ne$expected['tools.py']-or
  -not$targetRunValid-or
  -not[bool]$result.target_run_unchanged-or
  $loopback.Count-ne3-or@($loopback|Where-Object{-not[bool]$_.exact}).Count-or
  $public.Count-ne3-or@($public|Where-Object{-not[bool]$_.exact}).Count-or
  -not[bool]$result.hpc_binding_unchanged-or
  -not[bool]$result.byoa_stop_window_unchanged-or
  -not[bool]$result.profile_unchanged-or
  [bool]$result.hpc_accessed-or
  [bool]$result.gpu_touched-or
  [bool]$result.secrets_printed
){throw 'RUNTIME_R30_RESULT_REJECTED'}

[ordered]@{
  status='installed_and_verified'
  transaction_id=$TransactionId
  build_id=$build
  target_run_id=$ExpectedRunId
  result_path=$output
  result_sha256=Sha $output
  loopback_samples=$loopback.Count
  public_samples=$public.Count
  hpc_accessed=$false
  gpu_touched=$false
}|ConvertTo-Json -Compress
