param(
  [string]$CandidateTools='',
  [string]$CandidateHpcRuntime='',
  [string]$SourceManifestPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-v15r22-r12-hpc-executor-source-manifest-20260817.json',
  [string]$ZipPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-hpc-executor-hotfix-v15r22-r12-20260817.zip',
  [string]$TestResultPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-hpc-executor-hotfix-v15r22-r12-test-results-20260817.json'
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$CandidateTools=if($CandidateTools){[IO.Path]::GetFullPath($CandidateTools)}else{Join-Path $projectRoot 'src\evomind_runtime\tools.py'}
$CandidateHpcRuntime=if($CandidateHpcRuntime){[IO.Path]::GetFullPath($CandidateHpcRuntime)}else{Join-Path $projectRoot 'src\research_os\hpc_runtime.py'}
$fixture=Join-Path $projectRoot 'tests\test_hpc_execute_solution_workspace.py'
$buildId='overlay-super-agent-byoa-v15r22-r9-92abb43074a5'
$release='v15r22-r12'
$baseToolsSha='35afc02f899eb6f2a01d046e74f9ce5bc00c76ccc8c042305b49deaafae1726b'
$baseHpcRuntimeSha='f99c0fc52d97be222df4d923227905a4ea74dbdb76dd5dc6af22e3f320b116ce'
$sourceManifest=[IO.Path]::GetFullPath($SourceManifestPath)
$zip=[IO.Path]::GetFullPath($ZipPath)
$testResult=[IO.Path]::GetFullPath($TestResultPath)
$testFiles=@(
  'tests/test_hpc_execute_solution_workspace.py',
  'tests/test_aibuild_hpc_workflow.py',
  'tests/test_evomind_runtime.py',
  'tests/test_assistant_run_service.py',
  'tests/test_assistant_training.py',
  'tests/test_hpc_only_training_policy.py',
  'tests/test_hpc_runtime_contract.py'
)

function Sha([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
function Write-Utf8([string]$Path,[object]$Value){
  $parent=Split-Path -Parent $Path;if($parent){New-Item -ItemType Directory -Path $parent -Force|Out-Null}
  [IO.File]::WriteAllText($Path,(($Value|ConvertTo-Json -Depth 12)+"`n"),[Text.UTF8Encoding]::new($false))
}
function Assert-File([string]$Path){if(-not(Test-Path -LiteralPath $Path -PathType Leaf)){throw "R12_INPUT_MISSING:$Path"}}
function Read-ZipFiles([string]$Root){
  @(Get-ChildItem -LiteralPath $Root -Recurse -File -Force|Where-Object{$_.FullName-notmatch'[\\/]__pycache__[\\/]'-and$_.Extension-notin@('.pyc','.pyo')}|Sort-Object FullName|ForEach-Object{
    [ordered]@{path=$_.FullName.Substring($Root.Length).TrimStart('\').Replace('\','/');bytes=[long]$_.Length;sha256=Sha $_.FullName}
  })
}

foreach($path in @($CandidateTools,$CandidateHpcRuntime,$fixture)){Assert-File $path}
foreach($path in @($sourceManifest,$zip,$testResult)){if(Test-Path -LiteralPath $path){throw "R12_OUTPUT_EXISTS:$path"}}
if((Sha $CandidateTools)-eq$baseToolsSha-or(Sha $CandidateHpcRuntime)-eq$baseHpcRuntimeSha){throw 'R12_CANDIDATE_DID_NOT_CHANGE'}

$stamp=[guid]::NewGuid().ToString('N')
$asciiRoot='D:\EV12'
New-Item -ItemType Directory -Path $asciiRoot -Force|Out-Null
$stage=Join-Path $asciiRoot "r12-hpc-executor-$stamp"
$extractA=Join-Path $asciiRoot "r12-hpc-executor-verify-a-$stamp"
$extractB=Join-Path $asciiRoot "r12-hpc-executor-verify-b-$stamp"
foreach($path in @($stage,$extractA,$extractB)){if(Test-Path -LiteralPath $path){throw "R12_ASCII_STAGE_EXISTS:$path"}}
New-Item -ItemType Directory -Path (Join-Path $stage 'evomind_runtime') -Force|Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage 'research_os') -Force|Out-Null
Copy-Item -LiteralPath $CandidateTools -Destination (Join-Path $stage 'evomind_runtime\tools.py')
Copy-Item -LiteralPath $CandidateHpcRuntime -Destination (Join-Path $stage 'research_os\hpc_runtime.py')

$files=@(
  [ordered]@{path='evomind_runtime/tools.py';source='production_r10_baseline_plus_workspace_artifact_mapping_fix';base_sha256=$baseToolsSha;sha256=Sha (Join-Path $stage 'evomind_runtime\tools.py');bytes=[long](Get-Item -LiteralPath (Join-Path $stage 'evomind_runtime\tools.py')).Length},
  [ordered]@{path='research_os/hpc_runtime.py';source='active_release_baseline_plus_arbitrary_run_staging_contract';base_sha256=$baseHpcRuntimeSha;sha256=Sha (Join-Path $stage 'research_os\hpc_runtime.py');bytes=[long](Get-Item -LiteralPath (Join-Path $stage 'research_os\hpc_runtime.py')).Length}
)
$sourcePayload=[ordered]@{
  schema='evomind.super_agent_hpc_executor_source_set.v1'
  build_id=$buildId
  release=$release
  frozen=$true
  file_count=2
  files=$files
  fixture=[ordered]@{path='tests/test_hpc_execute_solution_workspace.py';sha256=Sha $fixture;bytes=[long](Get-Item -LiteralPath $fixture).Length}
  boundaries=[ordered]@{ascii_build_root=$asciiRoot;production_read_only_during_build=$true;hpc_accessed=$false;gpu_touched=$false;training_started=$false;kaggle_submissions=0;secrets_included=$false}
}
Write-Utf8 $sourceManifest $sourcePayload
$sourceManifestSha=Sha $sourceManifest
$hotfixManifest=[ordered]@{
  schema='evomind.super_agent_hpc_executor_hotfix.v1'
  build_id=$buildId
  release=$release
  source_manifest_sha256=$sourceManifestSha
  file_count=2
  files=$files
}
Write-Utf8 (Join-Path $stage 'hpc-executor-hotfix-manifest.json') $hotfixManifest

Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal
$zipSha=Sha $zip
Expand-Archive -LiteralPath $zip -DestinationPath $extractA
Expand-Archive -LiteralPath $zip -DestinationPath $extractB
$expected=@(
  [ordered]@{path='hpc-executor-hotfix-manifest.json';bytes=[long](Get-Item -LiteralPath (Join-Path $stage 'hpc-executor-hotfix-manifest.json')).Length;sha256=Sha (Join-Path $stage 'hpc-executor-hotfix-manifest.json')}
  $files
)

$verificationRuns=@()
foreach($root in @($extractA,$extractB)){
  $actual=Read-ZipFiles $root
  if($actual.Count-ne3){throw "R12_FILESET_COUNT_REJECTED:$root"}
  foreach($entry in $expected){
    $row=@($actual|Where-Object{$_.path-eq$entry.path})
    if($row.Count-ne1-or$row[0].bytes-ne$entry.bytes-or$row[0].sha256-ne$entry.sha256){throw "R12_FILESET_HASH_REJECTED:$($entry.path)"}
  }
  $overlayRoot=Join-Path $root 'verification-overlay'
  $overlaySrc=Join-Path $overlayRoot 'src'
  Copy-Item -LiteralPath (Join-Path $projectRoot 'src') -Destination $overlaySrc -Recurse
  Copy-Item -LiteralPath (Join-Path $root 'evomind_runtime\tools.py') -Destination (Join-Path $overlaySrc 'evomind_runtime\tools.py') -Force
  Copy-Item -LiteralPath (Join-Path $root 'research_os\hpc_runtime.py') -Destination (Join-Path $overlaySrc 'research_os\hpc_runtime.py') -Force
  $savedPythonPath=[Environment]::GetEnvironmentVariable('PYTHONPATH','Process')
  try{
    $env:PYTHONPATH=$overlaySrc
    $identityOutput=& uv run python -c "import hashlib,json;import evomind_runtime.tools as t;import research_os.hpc_runtime as h;p=lambda x:hashlib.sha256(open(x,'rb').read()).hexdigest();print(json.dumps({'tools_path':t.__file__,'tools_sha256':p(t.__file__),'hpc_path':h.__file__,'hpc_sha256':p(h.__file__)}))"
    if($LASTEXITCODE-ne0){throw "R12_IMPORT_IDENTITY_FAILED:$root"}
    $identity=$identityOutput|Out-String|ConvertFrom-Json
    if([string]$identity.tools_sha256-ne[string]$files[0].sha256-or[string]$identity.hpc_sha256-ne[string]$files[1].sha256-or-not([IO.Path]::GetFullPath([string]$identity.tools_path).StartsWith($overlaySrc,[StringComparison]::OrdinalIgnoreCase))-or-not([IO.Path]::GetFullPath([string]$identity.hpc_path).StartsWith($overlaySrc,[StringComparison]::OrdinalIgnoreCase))){throw "R12_IMPORT_IDENTITY_REJECTED:$root"}
    $compileOutput=& uv run python -m py_compile (Join-Path $overlaySrc 'evomind_runtime\tools.py') (Join-Path $overlaySrc 'research_os\hpc_runtime.py') 2>&1
    if($LASTEXITCODE-ne0){throw "R12_COMPILE_FAILED:$root`n$($compileOutput|Out-String)"}
    $testOutput=& uv run pytest -q @testFiles 2>&1
    if($LASTEXITCODE-ne0){throw "R12_TESTS_FAILED:$root`n$($testOutput|Out-String)"}
    $verificationRuns+=[ordered]@{extract_root=$root;overlay_src=$overlaySrc;identity=$identity;compile='passed';tests='passed';test_files=$testFiles;test_output_tail=(($testOutput|Out-String).Trim()|Select-Object -Last 1)}
  }finally{
    if($null-eq$savedPythonPath){Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue}else{[Environment]::SetEnvironmentVariable('PYTHONPATH',$savedPythonPath,'Process')}
  }
}

$testPayload=[ordered]@{
  schema='evomind.super_agent_hpc_executor_hotfix_test_result.v1'
  status='passed'
  build_id=$buildId
  release=$release
  zip_path=$zip
  zip_sha256=$zipSha
  source_manifest_path=$sourceManifest
  source_manifest_sha256=$sourceManifestSha
  independent_extract_runs=2
  verification_runs=$verificationRuns
  files=$files
  boundaries=[ordered]@{hpc_accessed=$false;gpu_touched=$false;remote_commands=0;training_started=$false;kaggle_submissions=0;secrets_included=$false}
}
Write-Utf8 $testResult $testPayload
[ordered]@{status='passed';zip_path=$zip;zip_sha256=$zipSha;source_manifest_path=$sourceManifest;source_manifest_sha256=$sourceManifestSha;test_result_path=$testResult;test_result_sha256=Sha $testResult}|ConvertTo-Json -Compress
