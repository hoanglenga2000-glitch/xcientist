param(
  [string]$CandidateGpuCredentials='D:\AI-Outputs\EvoMind-Cloud-Deploy\scratch\r11-production-baseline\gpu_credentials.production.py',
  [string]$CandidateTerminalTools='',
  [string]$SourceManifestPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-v15r22-r11-hpc-profile-source-manifest-20260817.json',
  [string]$ZipPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-hpc-profile-hotfix-v15r22-r11-20260817.zip',
  [string]$TestResultPath='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\super-agent-hpc-profile-hotfix-v15r22-r11-test-results-20260817.json'
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$CandidateTerminalTools=if($CandidateTerminalTools){[IO.Path]::GetFullPath($CandidateTerminalTools)}else{Join-Path $projectRoot 'src\xsci\terminal_tools.py'}
$fixture=Join-Path $PSScriptRoot 'test_tenant_hpc_profile_hotfix_r11.py'
$buildId='overlay-super-agent-byoa-v15r22-r9-92abb43074a5'
$baseGpuSha='a8d19ff2ceedcb5b5b4d716bb23380cef2ab307bdaaa10402ab691679e8bf804'
$baseTerminalSha='eec0457e427b271cf714fa2d229a18eed973d45be24d786692887a3419f4e9e2'
$sourceManifest=[IO.Path]::GetFullPath($SourceManifestPath)
$zip=[IO.Path]::GetFullPath($ZipPath)
$testResult=[IO.Path]::GetFullPath($TestResultPath)

function Sha([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
function Write-Utf8([string]$Path,[object]$Value){[IO.File]::WriteAllText($Path,(($Value|ConvertTo-Json -Depth 12)+"`n"),[Text.UTF8Encoding]::new($false))}
function Assert-File([string]$Path){if(-not(Test-Path -LiteralPath $Path -PathType Leaf)){throw "R11_INPUT_MISSING:$Path"}}
function Read-ZipFiles([string]$Root){
  @(Get-ChildItem -LiteralPath $Root -Recurse -File -Force|Where-Object{$_.FullName-notmatch'[\\/]__pycache__[\\/]'-and$_.Extension-notin@('.pyc','.pyo')}|Sort-Object FullName|ForEach-Object{
    [ordered]@{path=$_.FullName.Substring($Root.Length).TrimStart('\').Replace('\','/');bytes=[long]$_.Length;sha256=Sha $_.FullName}
  })
}

foreach($path in @($CandidateGpuCredentials,$CandidateTerminalTools,$fixture)){Assert-File $path}
foreach($path in @($sourceManifest,$zip,$testResult)){if(Test-Path -LiteralPath $path){throw "R11_OUTPUT_EXISTS:$path"}}

$stamp=[guid]::NewGuid().ToString('N')
$asciiRoot='D:\EV12'
New-Item -ItemType Directory -Path $asciiRoot -Force|Out-Null
$stage=Join-Path $asciiRoot "r11-hotfix-$stamp"
$extractA=Join-Path $asciiRoot "r11-verify-a-$stamp"
$extractB=Join-Path $asciiRoot "r11-verify-b-$stamp"
foreach($path in @($stage,$extractA,$extractB)){if(Test-Path -LiteralPath $path){throw "R11_ASCII_STAGE_EXISTS:$path"}}
New-Item -ItemType Directory -Path (Join-Path $stage 'research_agent_workstation\server\core') -Force|Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage 'xsci') -Force|Out-Null
Copy-Item -LiteralPath $CandidateGpuCredentials -Destination (Join-Path $stage 'research_agent_workstation\server\core\gpu_credentials.py')
Copy-Item -LiteralPath $CandidateTerminalTools -Destination (Join-Path $stage 'xsci\terminal_tools.py')

$gpuPath=Join-Path $stage 'research_agent_workstation\server\core\gpu_credentials.py'
$terminalPath=Join-Path $stage 'xsci\terminal_tools.py'
$files=@(
  [ordered]@{path='research_agent_workstation/server/core/gpu_credentials.py';source='production_read_only_baseline_plus_minimal_patch';base_sha256=$baseGpuSha;sha256=Sha $gpuPath;bytes=[long](Get-Item -LiteralPath $gpuPath).Length},
  [ordered]@{path='xsci/terminal_tools.py';source='workspace_matching_production_baseline_plus_minimal_patch';base_sha256=$baseTerminalSha;sha256=Sha $terminalPath;bytes=[long](Get-Item -LiteralPath $terminalPath).Length}
)
$sourcePayload=[ordered]@{
  schema='evomind.super_agent_hpc_profile_source_set.v1'
  build_id=$buildId
  release='v15r22-r11'
  file_count=2
  files=$files
  fixture=[ordered]@{path='scripts/test_tenant_hpc_profile_hotfix_r11.py';sha256=Sha $fixture;bytes=[long](Get-Item -LiteralPath $fixture).Length}
  boundaries=[ordered]@{production_baseline_read_only=$true;hpc_accessed=$false;gpu_touched=$false;training_started=$false;kaggle_submissions=0;secrets_included=$false}
}
Write-Utf8 $sourceManifest $sourcePayload
$sourceManifestSha=Sha $sourceManifest
$hotfixManifest=[ordered]@{
  schema='evomind.super_agent_hpc_profile_hotfix.v1'
  build_id=$buildId
  release='v15r22-r11'
  source_manifest_sha256=$sourceManifestSha
  file_count=2
  files=$files
}
Write-Utf8 (Join-Path $stage 'hpc-profile-hotfix-manifest.json') $hotfixManifest

Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal
$zipSha=Sha $zip
Expand-Archive -LiteralPath $zip -DestinationPath $extractA
Expand-Archive -LiteralPath $zip -DestinationPath $extractB
$expected=@(
  [ordered]@{path='hpc-profile-hotfix-manifest.json';bytes=[long](Get-Item -LiteralPath (Join-Path $stage 'hpc-profile-hotfix-manifest.json')).Length;sha256=Sha (Join-Path $stage 'hpc-profile-hotfix-manifest.json')}
  $files
)
foreach($root in @($extractA,$extractB)){
  $actual=Read-ZipFiles $root
  if($actual.Count-ne3){throw "R11_FILESET_COUNT_REJECTED:$root"}
  foreach($entry in $expected){
    $row=@($actual|Where-Object{$_.path-eq$entry.path})
    if($row.Count-ne1-or$row[0].bytes-ne$entry.bytes-or$row[0].sha256-ne$entry.sha256){throw "R11_FILESET_HASH_REJECTED:$($entry.path)"}
  }
}

$fixtureRuns=@()
foreach($root in @($extractA,$extractB)){
  $gpu=Join-Path $root 'research_agent_workstation\server\core\gpu_credentials.py'
  $terminal=Join-Path $root 'xsci\terminal_tools.py'
  $output=& uv run python $fixture --gpu-credentials $gpu --terminal-tools $terminal
  if($LASTEXITCODE-ne0){throw "R11_FIXTURE_FAILED:$root"}
  $payload=$output|Out-String|ConvertFrom-Json
  if([string]$payload.status-ne'passed'-or[int]$payload.offline_probe_samples-ne5-or[int]$payload.remote_commands-ne0-or[bool]$payload.training_started-or[int]$payload.kaggle_submissions-ne0){throw "R11_FIXTURE_RESULT_REJECTED:$root"}
  $fixtureRuns+=$payload
}

$testPayload=[ordered]@{
  schema='evomind.super_agent_hpc_profile_hotfix_test_result.v1'
  status='passed'
  build_id=$buildId
  release='v15r22-r11'
  zip_path=$zip
  zip_sha256=$zipSha
  source_manifest_path=$sourceManifest
  source_manifest_sha256=$sourceManifestSha
  extract_roots=@($extractA,$extractB)
  independent_extract_runs=2
  fixture_runs=$fixtureRuns
  files=$files
  boundaries=[ordered]@{hpc_accessed=$false;gpu_touched=$false;remote_commands=0;training_started=$false;kaggle_submissions=0;secrets_included=$false}
}
Write-Utf8 $testResult $testPayload
[ordered]@{status='passed';zip_path=$zip;zip_sha256=$zipSha;source_manifest_path=$sourceManifest;source_manifest_sha256=$sourceManifestSha;test_result_path=$testResult;test_result_sha256=Sha $testResult}|ConvertTo-Json -Compress
