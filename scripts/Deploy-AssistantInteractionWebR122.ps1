param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [string]$CandidateZip='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\web-overlay-assistant-interaction-r122.zip',
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$CandidateZipSha256,
  [string]$SourceManifestPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\web-overlay-assistant-interaction-r122-source-manifest.json',
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$SourceManifestSha256,
  [string]$BuildResultPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\web-overlay-assistant-interaction-r122-build-result.json',
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$BuildResultSha256,
  [string]$RemoteRunnerPath='',
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$RemoteRunnerSha256,
  [string]$HostAlias='evomind-shanghai',
  [string]$AdminHashPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\evomind-admin-password.scrypt',
  [string]$OutputPath=''
)

Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$expectedRuntimeTreeSha256='a9595ab4ed9a86786ca65f4456893d64741b449ec84cf8c9ddb42e9bf4110dc7'
$adminHashSha256='18bb9c74b5b0759f71f6fbd2cbb5f3f4b746e92396519369099673867f59ef0d'
$RemoteRunnerPath=if($RemoteRunnerPath){[IO.Path]::GetFullPath($RemoteRunnerPath)}else{Join-Path $PSScriptRoot 'Deploy-AssistantInteractionWebR122Remote.ps1'}
$remote="C:\SecureInput\assistant-interaction-r122-$TransactionId"
$output=if($OutputPath){[IO.Path]::GetFullPath($OutputPath)}else{"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\assistant-interaction-r122-production-$TransactionId.json"}
$errorOutput=[IO.Path]::ChangeExtension($output,'.error.json')

function Get-Sha256([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
function Read-Object([string]$Path,[string]$Code){if(-not(Test-Path -LiteralPath $Path -PathType Leaf)){throw "${Code}_MISSING"};try{$value=Get-Content -LiteralPath $Path -Raw -Encoding UTF8|ConvertFrom-Json}catch{throw "${Code}_INVALID_JSON"};if($null-eq$value-or$value-is[array]){throw "${Code}_OBJECT_REQUIRED"};$value}

foreach($path in @($output,$errorOutput)){if(Test-Path -LiteralPath $path){throw "R122_LOCAL_OUTPUT_EXISTS:$path"}}
$inputs=@(
  [pscustomobject]@{path=[IO.Path]::GetFullPath($CandidateZip);sha=$CandidateZipSha256;name='web.zip'},
  [pscustomobject]@{path=[IO.Path]::GetFullPath($SourceManifestPath);sha=$SourceManifestSha256;name='source-manifest.json'},
  [pscustomobject]@{path=[IO.Path]::GetFullPath($BuildResultPath);sha=$BuildResultSha256;name='build-result.json'},
  [pscustomobject]@{path=[IO.Path]::GetFullPath($RemoteRunnerPath);sha=$RemoteRunnerSha256;name='Deploy-AssistantInteractionWebR122Remote.ps1'},
  [pscustomobject]@{path=[IO.Path]::GetFullPath($AdminHashPath);sha=$adminHashSha256;name='admin.scrypt'}
)
foreach($item in $inputs){if(-not(Test-Path -LiteralPath $item.path -PathType Leaf)){throw "R122_LOCAL_INPUT_MISSING:$($item.path)"};if((Get-Sha256 $item.path)-cne$item.sha){throw "R122_LOCAL_INPUT_HASH_REJECTED:$($item.name)"}}

$source=Read-Object ([IO.Path]::GetFullPath($SourceManifestPath)) 'R122_LOCAL_SOURCE_MANIFEST'
$build=Read-Object ([IO.Path]::GetFullPath($BuildResultPath)) 'R122_LOCAL_BUILD_RESULT'
if([string]$source.schema-cne'evomind.assistant_interaction_web_source_set.r122.v1'-or-not[bool]$source.frozen-or[int]$source.production_allowlist_count-ne3-or[bool]$source.test_in_production-or[string]$source.expected_runtime_tree_sha256-cne$expectedRuntimeTreeSha256){throw 'R122_LOCAL_SOURCE_CONTRACT_REJECTED'}
if([string]$build.schema-cne'evomind.assistant_interaction_web_release.r122.v1'-or[string]$build.status-cne'built_and_verified'-or[string]$build.zip_sha256-cne$CandidateZipSha256-or[string]$build.source_manifest_sha256-cne$SourceManifestSha256-or[int]$build.production_allowlist_count-ne3-or[bool]$build.production_deployed-or[string]$build.expected_runtime_tree_sha256-cne$expectedRuntimeTreeSha256){throw 'R122_LOCAL_BUILD_CONTRACT_REJECTED'}
$newBuildId=[string]$build.candidate_id
if($newBuildId-cnotmatch'^overlay-assistant-interaction-r122-[a-f0-9]{12}$'){throw 'R122_LOCAL_BUILD_ID_REJECTED'}

$create="if(Test-Path -LiteralPath '$remote'){throw 'R122_REMOTE_EXISTS'};New-Item -ItemType Directory -Path '$remote'|Out-Null"
$create|ssh $HostAlias 'powershell.exe -NoProfile -NonInteractive -Command -'
if($LASTEXITCODE-ne0){throw 'R122_REMOTE_CREATE_REJECTED'}
foreach($item in $inputs){scp $item.path "${HostAlias}:$remote\$($item.name)";if($LASTEXITCODE-ne0){throw "R122_REMOTE_UPLOAD_REJECTED:$($item.name)"}}

$verify="`$files=@([pscustomobject]@{name='web.zip';sha='$CandidateZipSha256'},[pscustomobject]@{name='source-manifest.json';sha='$SourceManifestSha256'},[pscustomobject]@{name='build-result.json';sha='$BuildResultSha256'},[pscustomobject]@{name='Deploy-AssistantInteractionWebR122Remote.ps1';sha='$RemoteRunnerSha256'},[pscustomobject]@{name='admin.scrypt';sha='$adminHashSha256'});foreach(`$x in `$files){`$p=Join-Path '$remote' `$x.name;if((Get-FileHash -LiteralPath `$p -Algorithm SHA256).Hash.ToLowerInvariant()-cne`$x.sha){throw 'R122_REMOTE_HASH_REJECTED'}};`$tokens=`$null;`$errors=`$null;[Management.Automation.Language.Parser]::ParseFile((Join-Path '$remote' 'Deploy-AssistantInteractionWebR122Remote.ps1'),[ref]`$tokens,[ref]`$errors)|Out-Null;if(@(`$errors).Count){throw 'R122_REMOTE_AST_REJECTED'};[ordered]@{status='staged_verified';transaction_id='$TransactionId';new_build_id='$newBuildId'}|ConvertTo-Json -Compress"
$staged=$verify|ssh $HostAlias 'powershell.exe -NoProfile -NonInteractive -Command -'
if($LASTEXITCODE-ne0){throw 'R122_REMOTE_STAGE_VERIFY_REJECTED'}
$staged

$common="-TransactionId $TransactionId -RemoteRoot $remote -WebSha256 $CandidateZipSha256 -SourceManifestSha256 $SourceManifestSha256 -BuildResultSha256 $BuildResultSha256 -RemoteRunnerSha256 $RemoteRunnerSha256 -NewBuildId $newBuildId -ExpectedRuntimeTreeSha256 $expectedRuntimeTreeSha256"
$preflight=ssh $HostAlias "powershell.exe -NoProfile -NonInteractive -File $remote\Deploy-AssistantInteractionWebR122Remote.ps1 $common -PreflightOnly"
if($LASTEXITCODE-ne0){throw 'R122_REMOTE_PREFLIGHT_REJECTED'}
$preflightPayload=$preflight|ConvertFrom-Json
if([string]$preflightPayload.status-cne'preflight_verified'-or[string]$preflightPayload.transaction_id-cne$TransactionId-or[string]$preflightPayload.new_build_id-cne$newBuildId-or[bool]$preflightPayload.service_restarted-or[bool]$preflightPayload.hpc_accessed-or[bool]$preflightPayload.gpu_touched-or[bool]$preflightPayload.training_started){throw 'R122_REMOTE_PREFLIGHT_RESULT_REJECTED'}
$preflight

ssh $HostAlias "powershell.exe -NoProfile -NonInteractive -File $remote\Deploy-AssistantInteractionWebR122Remote.ps1 $common"
if($LASTEXITCODE-ne0){scp "${HostAlias}:$remote\transaction-$TransactionId-error.json" $errorOutput 2>$null;throw 'R122_REMOTE_TRANSACTION_REJECTED'}
scp "${HostAlias}:$remote\transaction-$TransactionId-result.json" $output
if($LASTEXITCODE-ne0-or-not(Test-Path -LiteralPath $output -PathType Leaf)){throw 'R122_RESULT_DOWNLOAD_REJECTED'}
$result=Read-Object $output 'R122_RESULT'
if([string]$result.schema-cne'evomind.assistant_interaction_web_deployment_result.r122.v1'-or[string]$result.status-cne'installed_and_verified'-or[string]$result.transaction_id-cne$TransactionId-or[string]$result.new_build_id-cne$newBuildId-or[string]$result.web_sha256-cne$CandidateZipSha256-or-not[bool]$result.run_unchanged-or-not[bool]$result.runtime_unchanged-or-not[bool]$result.hpc_binding_unchanged-or-not[bool]$result.byoa_unchanged-or-not[bool]$result.profile_unchanged-or-not[bool]$result.fresh_roles-or-not[bool]$result.fresh_listeners-or[bool]$result.hpc_accessed-or[bool]$result.gpu_touched-or[bool]$result.training_started-or[int]$result.hpc_remote_writes-ne0-or[int]$result.signals_sent-ne0-or[bool]$result.other_processes_modified){throw 'R122_RESULT_CONTRACT_REJECTED'}
[ordered]@{status='installed_and_verified';transaction_id=$TransactionId;build_id=$newBuildId;result_path=$output;result_sha256=Get-Sha256 $output;run_unchanged=$true;runtime_unchanged=$true;g24_unchanged=$true;hpc_accessed=$false;gpu_touched=$false;training_started=$false}|ConvertTo-Json -Compress
