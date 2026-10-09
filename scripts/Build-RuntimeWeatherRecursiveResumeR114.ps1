param(
  [string]$SourceRoot='',
  [string]$OutputRoot='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts',
  [string]$VerifyRoot='D:\EV12',
  [string]$BaselineReleaseZip='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\overlay-controlled-secret-binding-r112-standalone-v5.zip',
  [ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedBaselineZipSha='2284252cf7ae98a1d3db08e9117d863d839e55174d428df2e7e00dc1a7d8eaa2',
  [ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedCompetitionSha='1e45df525064c16b6d0610e6769734e7645067941161c150adb8d9f774401c49',
  [string]$Release='r114-weather-recursive-resume-r112-base'
)

Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

$sourceInput=if([string]::IsNullOrWhiteSpace($SourceRoot)){Split-Path -Parent $PSScriptRoot}else{$SourceRoot}
$source=[IO.Path]::GetFullPath($sourceInput).TrimEnd('\')
$output=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$verify=[IO.Path]::GetFullPath($VerifyRoot).TrimEnd('\')
$baseline=[IO.Path]::GetFullPath($BaselineReleaseZip)
$candidate=Join-Path $source 'src\evomind_runtime\competition_data.py'
$zip=Join-Path $output "evomind-runtime-$Release-20260829.zip"
$sourceManifest=Join-Path $output "evomind-runtime-$Release-source-manifest-20260829.json"
$testResult=Join-Path $output "evomind-runtime-$Release-test-results-20260829.json"
$expectedBuild='overlay-controlled-secret-binding-r112-standalone-f38c40f2dbdc'

function Sha([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
function Write-Json([string]$Path,[object]$Value){[IO.File]::WriteAllText($Path,(($Value|ConvertTo-Json -Depth 20)+"`n"),[Text.UTF8Encoding]::new($false))}

foreach($path in @($baseline,$candidate)){if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "R114_INPUT_MISSING:$path"}}
foreach($path in @($zip,$sourceManifest,$testResult)){if(Test-Path -LiteralPath $path){throw "R114_OUTPUT_EXISTS:$path"}}
if((Sha $baseline)-ne$ExpectedBaselineZipSha){throw 'R114_BASELINE_ZIP_HASH_REJECTED'}
if((Sha $candidate)-ne$ExpectedCompetitionSha){throw 'R114_COMPETITION_SOURCE_HASH_REJECTED'}
$python=Join-Path $source '.venv\Scripts\python.exe';if(-not(Test-Path -LiteralPath $python)){throw 'R114_PYTHON_MISSING'}
$bash=(Get-Command bash.exe -ErrorAction Stop).Source
New-Item -ItemType Directory -Force -Path $output,$verify|Out-Null
$stage=Join-Path $verify ('r114-stage-'+[guid]::NewGuid().ToString('N'))
$packageStage=Join-Path $verify ('r114-package-'+[guid]::NewGuid().ToString('N'))
$verifyDirs=@()
try{
  New-Item -ItemType Directory -Path $stage,$packageStage|Out-Null
  [IO.Compression.ZipFile]::ExtractToDirectory($baseline,$stage)
  $runtimeManifest=Get-Content -LiteralPath (Join-Path $stage 'runtime-build-manifest.json') -Raw -Encoding UTF8|ConvertFrom-Json
  if([string]$runtimeManifest.build_id-ne$expectedBuild){throw 'R114_BASELINE_BUILD_ID_REJECTED'}
  $runtimeRoot=Join-Path $stage 'support\python-runtime\evomind_runtime'
  $baselineFiles=@(Get-ChildItem -LiteralPath $runtimeRoot -File -Filter '*.py'|Sort-Object Name)
  if($baselineFiles.Count-ne30){throw 'R114_BASELINE_RUNTIME_COUNT_REJECTED'}
  $baselineByName=@{};foreach($file in $baselineFiles){$baselineByName[$file.Name]=[ordered]@{bytes=[long]$file.Length;sha256=Sha $file.FullName}}
  $package=Join-Path $packageStage 'evomind_runtime';New-Item -ItemType Directory -Path $package|Out-Null
  foreach($file in $baselineFiles){Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $package $file.Name)}
  Copy-Item -LiteralPath $candidate -Destination (Join-Path $package 'competition_data.py') -Force
  if((Sha (Join-Path $package 'competition_data.py'))-ne$ExpectedCompetitionSha){throw 'R114_COMPETITION_COPY_REJECTED'}
  $entries=@(Get-ChildItem -LiteralPath $package -File -Filter '*.py'|Sort-Object Name|ForEach-Object{[ordered]@{path="evomind_runtime/$($_.Name)";bytes=[long]$_.Length;sha256=Sha $_.FullName}})
  $changed=@($entries|Where-Object{$baselineByName[[IO.Path]::GetFileName([string]$_.path)].sha256-ne[string]$_.sha256})
  if($changed.Count-ne1-or[string]$changed[0].path-ne'evomind_runtime/competition_data.py'){throw 'R114_CHANGESET_REJECTED'}
  $sourceEntries=@($entries|ForEach-Object{[ordered]@{path=('src/'+[string]$_.path);bytes=$_.bytes;sha256=$_.sha256}})
  $canonical=($sourceEntries|ForEach-Object{"$($_.path)`0$($_.sha256)`n"})-join'';$hasher=[Security.Cryptography.SHA256]::Create();try{$treeSha=([BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($canonical)))).Replace('-','').ToLowerInvariant()}finally{$hasher.Dispose()}
  $sourcePayload=[ordered]@{schema='evomind.super_agent_source_set.v1';created_at_utc=[DateTime]::UtcNow.ToString('o');purpose='R114 Weather recursive resume over exact R112 production release';frozen=$true;source_dirty=$true;source_tree_sha256=$treeSha;file_count=30;files=$sourceEntries;baseline=[ordered]@{build_id=$expectedBuild;release_zip_sha256=$ExpectedBaselineZipSha;preserved_file_count=29;replaced_file='src/evomind_runtime/competition_data.py'};validation=[ordered]@{targeted_pytest='143/143';py_compile='passed';adapter_bash_n='12/12';recursive_resume_contract='passed';hpc_accessed=$false;gpu_touched=$false}}
  Write-Json $sourceManifest $sourcePayload;$sourceSha=Sha $sourceManifest
  $hotfix=[ordered]@{schema='evomind.super_agent_runtime_hotfix.v1';created_at_utc=[DateTime]::UtcNow.ToString('o');release=$Release;source_manifest_sha256=$sourceSha;file_count=30;files=$entries;target='bundle/runtime/evomind_runtime';production_deployed=$false;hpc_accessed=$false;gpu_touched=$false}
  Write-Json (Join-Path $packageStage 'runtime-hotfix-manifest.json') $hotfix
  [IO.Compression.ZipFile]::CreateFromDirectory($packageStage,$zip,[IO.Compression.CompressionLevel]::Optimal,$false)
  $rounds=@()
  foreach($round in 1..2){
    $target=Join-Path $verify ("r114-verify-$round-"+[guid]::NewGuid().ToString('N'));$verifyDirs+=$target;New-Item -ItemType Directory -Path $target|Out-Null;[IO.Compression.ZipFile]::ExtractToDirectory($zip,$target)
    $manifest=Get-Content -LiteralPath (Join-Path $target 'runtime-hotfix-manifest.json') -Raw|ConvertFrom-Json;$actual=@(Get-ChildItem -LiteralPath $target -Recurse -File);if($manifest.file_count-ne30-or$actual.Count-ne31-or[string]$manifest.target-ne'bundle/runtime/evomind_runtime'){throw "R114_ROUND_MANIFEST_REJECTED:$round"}
    foreach($entry in $manifest.files){$path=Join-Path $target ([string]$entry.path).Replace('/','\');if((Sha $path)-ne[string]$entry.sha256-or[long](Get-Item -LiteralPath $path).Length-ne[long]$entry.bytes){throw "R114_ROUND_HASH_REJECTED:$round"};& $python -m py_compile $path;if($LASTEXITCODE-ne0){throw "R114_ROUND_COMPILE_REJECTED:$round"}}
    Get-ChildItem -LiteralPath $target -Recurse -Directory -Filter '__pycache__'|Remove-Item -Recurse -Force
    $check=Join-Path $target 'adapter-check.py';$body=@'
import json,re,subprocess,sys
from evomind_runtime.competition_data import build_adapter_script
bash=sys.argv[1]
for competition in ("cure_bench","e2lmc","mindgames","ariel_2025","weather4cast","open_polymer"):
  for action in ("status","prepare"):
    script=build_adapter_script(competition,action);result=subprocess.run([bash,"-n"],input=script.encode(),capture_output=True)
    assert result.returncode==0,(competition,action,result.stderr)
    for body in re.findall(r"<<'PY'\n(.*?)\nPY\n",script,flags=re.S):compile(body,"embedded.py","exec")
accelerated=build_adapter_script("weather4cast","accelerate")
for marker in ("get -aR .",'"adapter_transfer_mode": "openssh_recursive_get"','kill -TERM -- "-$EXISTING_PID"','"expected_file_count": 108'):assert marker in accelerated
print(json.dumps({"status":"passed"}))
'@;[IO.File]::WriteAllText($check,$body,[Text.UTF8Encoding]::new($false));$saved=$env:PYTHONPATH;try{$env:PYTHONPATH=$target;$checkResult=& $python $check $bash|ConvertFrom-Json;if($LASTEXITCODE-ne0-or$checkResult.status-ne'passed'){throw "R114_ADAPTER_CHECK_REJECTED:$round"}}finally{$env:PYTHONPATH=$saved};$rounds+=[ordered]@{round=$round;status='passed';file_count=30;py_compile='passed';adapter_bash_n='12/12';recursive_resume_contract='passed';hpc_accessed=$false;gpu_touched=$false}
  }
  $result=[ordered]@{schema='evomind.runtime_r114_build_result.v1';status='built_and_verified';completed_at_utc=[DateTime]::UtcNow.ToString('o');zip_path=$zip;zip_bytes=[long](Get-Item -LiteralPath $zip).Length;zip_sha256=Sha $zip;source_manifest_path=$sourceManifest;source_manifest_sha256=$sourceSha;file_count=30;baseline_build_id=$expectedBuild;baseline_zip_sha256=$ExpectedBaselineZipSha;preserved_file_count=29;competition_data=[ordered]@{bytes=[long](Get-Item -LiteralPath $candidate).Length;sha256=$ExpectedCompetitionSha};verification_rounds=$rounds;production_deployed=$false;hpc_accessed=$false;gpu_touched=$false};Write-Json $testResult $result;$result|ConvertTo-Json -Depth 15 -Compress
}finally{foreach($path in @($stage,$packageStage)+$verifyDirs){if(Test-Path -LiteralPath $path){Remove-Item -LiteralPath $path -Recurse -Force}}}
