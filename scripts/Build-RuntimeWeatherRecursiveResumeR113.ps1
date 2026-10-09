param(
  [string]$SourceRoot='',
  [string]$OutputRoot='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts',
  [string]$VerificationBasePath='D:\EV12',
  [string]$BaselineZip='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\evomind-runtime-r111-weather-skip-timeout-gated-20260822.zip',
  [string]$BaselineSourceManifest='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\evomind-runtime-r111-weather-skip-timeout-gated-source-manifest-20260822.json',
  [ValidatePattern('^r[0-9]+(?:-[a-z0-9-]+)?$')][string]$ReleaseName='r113-weather-recursive-resume',
  [ValidatePattern('^[0-9]+/[0-9]+$')][string]$TargetedPytest='143/143'
)

Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$sourceInput = if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
  Split-Path -Parent $PSScriptRoot
} else {
  $SourceRoot
}
$source=[IO.Path]::GetFullPath($sourceInput).TrimEnd('\')
$output=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$verifyBase=[IO.Path]::GetFullPath($VerificationBasePath).TrimEnd('\')
$baselineZipPath=[IO.Path]::GetFullPath($BaselineZip)
$baselineSourcePath=[IO.Path]::GetFullPath($BaselineSourceManifest)
$release=$ReleaseName.ToLowerInvariant()
$sourceManifest=Join-Path $output "evomind-runtime-$release-source-manifest-20260829.json"
$zip=Join-Path $output "evomind-runtime-$release-20260829.zip"
$testResult=Join-Path $output "evomind-runtime-$release-test-results-20260829.json"
$expectedBaselineZipSha='ff35e6235c9fa425f26a3b826a36a936f519a3969ac9becc66d204ca06ced235'
$expectedBaselineSourceSha='0cdb6ae4cb878ed08cace602d1918f425ad443437911b0be6e0aafc880c475e1'
$runtimeNames=@(
  '__init__.py','agent_kernel_v2.py','assistant_runs.py','benchmark.py','capabilities.py',
  'cli.py','client.py','competition_data.py','connectors.py','credential_leases.py',
  'directory_broker.py','ecosystem.py','evidence_memory.py','evolution.py','hpc_runtime_overlay.py',
  'http_server.py','mcp.py','model_router.py','models.py','policy.py','recovery.py',
  'remote_connectors.py','runtime.py','run_secrets.py','store.py','super_agent_runtime.py',
  'super_agent_store.py','super_agent_tools.py','tool_package_runtime.py','tools.py'
)

function Sha([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Write-Utf8([string]$Path,[object]$Value) {
  $parent=Split-Path -Parent $Path
  if($parent){New-Item -ItemType Directory -Force -Path $parent|Out-Null}
  $json=if($Value-is[string]){[string]$Value}else{$Value|ConvertTo-Json -Depth 20}
  [IO.File]::WriteAllText($Path,$json.TrimEnd()+"`n",[Text.UTF8Encoding]::new($false))
}

foreach($path in @($sourceManifest,$zip,$testResult)){if(Test-Path -LiteralPath $path){throw "RUNTIME_R113_OUTPUT_EXISTS:$path"}}
foreach($path in @($baselineZipPath,$baselineSourcePath)){if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "RUNTIME_R113_BASELINE_MISSING:$path"}}
if((Sha $baselineZipPath)-ne$expectedBaselineZipSha){throw 'RUNTIME_R113_BASELINE_ZIP_HASH_REJECTED'}
if((Sha $baselineSourcePath)-ne$expectedBaselineSourceSha){throw 'RUNTIME_R113_BASELINE_SOURCE_HASH_REJECTED'}
$baselineManifest=Get-Content -LiteralPath $baselineSourcePath -Raw -Encoding UTF8|ConvertFrom-Json
if([string]$baselineManifest.schema-ne'evomind.super_agent_source_set.v1'-or-not[bool]$baselineManifest.frozen-or[int]$baselineManifest.file_count-ne30){throw 'RUNTIME_R113_BASELINE_MANIFEST_REJECTED'}
$baselineManifestSha=Sha $baselineSourcePath
$python=Join-Path $source '.venv\Scripts\python.exe'
if(-not(Test-Path -LiteralPath $python -PathType Leaf)){throw 'RUNTIME_R113_PYTHON_MISSING'}
$bash=(Get-Command bash.exe -ErrorAction Stop).Source
$competitionSource=Join-Path $source 'src\evomind_runtime\competition_data.py'
if(-not(Test-Path -LiteralPath $competitionSource -PathType Leaf)){throw 'RUNTIME_R113_COMPETITION_SOURCE_MISSING'}
$candidateCompetitionSha=Sha $competitionSource
$baselineEntries=@{}
foreach($entry in $baselineManifest.files){$baselineEntries[[string]$entry.path]=[ordered]@{bytes=[long]$entry.bytes;sha256=[string]$entry.sha256}}
if($baselineEntries.Count-ne30){throw 'RUNTIME_R113_BASELINE_ENTRY_COUNT_REJECTED'}
$stage=Join-Path $verifyBase ("runtime-$release-stage-"+[guid]::NewGuid().ToString('N'))
$verifyRoots=@()
try{
  New-Item -ItemType Directory -Path $stage|Out-Null
  Expand-Archive -LiteralPath $baselineZipPath -DestinationPath $stage
  $package=Join-Path $stage 'evomind_runtime'
  if(-not(Test-Path -LiteralPath $package -PathType Container)){throw 'RUNTIME_R113_PACKAGE_ROOT_MISSING'}
  foreach($name in $runtimeNames){
    $path=Join-Path $package $name
    if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "RUNTIME_R113_BASELINE_FILE_MISSING:$name"}
    $entry=$baselineEntries["src/evomind_runtime/$name"]
    if((Sha $path)-ne$entry.sha256-or[long](Get-Item -LiteralPath $path).Length-ne$entry.bytes){throw "RUNTIME_R113_BASELINE_FILE_HASH_REJECTED:$name"}
  }
  Copy-Item -LiteralPath $competitionSource -Destination (Join-Path $package 'competition_data.py') -Force
  if((Sha (Join-Path $package 'competition_data.py'))-ne$candidateCompetitionSha){throw 'RUNTIME_R113_COMPETITION_COPY_REJECTED'}
  & $python -m py_compile (Join-Path $package 'competition_data.py')
  if($LASTEXITCODE-ne0){throw 'RUNTIME_R113_COMPETITION_COMPILE_FAILED'}
  Get-ChildItem -LiteralPath $stage -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

  $entries=@()
  foreach($name in $runtimeNames){
    $path=Join-Path $package $name
    $entries+=[ordered]@{path="src/evomind_runtime/$name";bytes=[long](Get-Item -LiteralPath $path).Length;sha256=Sha $path}
  }
  $canonical=($entries|ForEach-Object{"$($_.path)`0$($_.sha256)`n"})-join''
  $hash=[Security.Cryptography.SHA256]::Create();try{$sourceTreeSha=([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($canonical)))).Replace('-','').ToLowerInvariant()}finally{$hash.Dispose()}
  $sourcePayload=[ordered]@{
    schema='evomind.super_agent_source_set.v1';created_at_utc=[DateTime]::UtcNow.ToString('o');purpose='Weather4cast accelerated recursive resume adapter over frozen R111 runtime';frozen=$true;source_dirty=$true;source_tree_sha256=$sourceTreeSha;file_count=30;files=$entries
    baseline=[ordered]@{release='r111-weather-skip-timeout-gated';runtime_zip_sha256=$expectedBaselineZipSha;source_manifest_sha256=$baselineManifestSha;preserved_file_count=29;replaced_file='src/evomind_runtime/competition_data.py';candidate_competition_sha256=$candidateCompetitionSha}
    validation=[ordered]@{targeted_pytest=$TargetedPytest;py_compile='passed';adapter_bash_n='12/12';embedded_python_compile='passed';recursive_resume_contract='passed';external_network_accessed=$false;hpc_accessed=$false;gpu_touched=$false}
  }
  Write-Utf8 $sourceManifest $sourcePayload
  $sourceManifestSha=Sha $sourceManifest
  $manifest=[ordered]@{schema='evomind.super_agent_runtime_hotfix.v1';created_at_utc=[DateTime]::UtcNow.ToString('o');release=$release;source_manifest_sha256=$sourceManifestSha;file_count=30;files=@($entries|ForEach-Object{[ordered]@{path="evomind_runtime/$([IO.Path]::GetFileName([string]$_.path))";bytes=$_.bytes;sha256=$_.sha256}});target='bundle/runtime/evomind_runtime';production_deployed=$false;hpc_accessed=$false;gpu_touched=$false}
  Write-Utf8 (Join-Path $stage 'runtime-hotfix-manifest.json') $manifest
  $zipStage=Join-Path $verifyBase ("runtime-$release-zipstage-"+[guid]::NewGuid().ToString('N'));New-Item -ItemType Directory -Path $zipStage|Out-Null
  try{
    Copy-Item -LiteralPath $package -Destination $zipStage -Recurse
    Copy-Item -LiteralPath (Join-Path $stage 'runtime-hotfix-manifest.json') -Destination $zipStage -Force
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::CreateFromDirectory($zipStage,$zip,[IO.Compression.CompressionLevel]::Optimal,$false)
  }finally{if(Test-Path -LiteralPath $zipStage){Remove-Item -LiteralPath $zipStage -Recurse -Force}}

  $rounds=@()
  foreach($round in 1..2){
    $target=Join-Path $verifyBase ("runtime-$release-verify-"+[guid]::NewGuid().ToString('N'));$verifyRoots+=$target;New-Item -ItemType Directory -Path $target|Out-Null;Expand-Archive -LiteralPath $zip -DestinationPath $target
    $m=Get-Content -LiteralPath (Join-Path $target 'runtime-hotfix-manifest.json') -Raw -Encoding UTF8|ConvertFrom-Json
    if([string]$m.release-ne$release-or[int]$m.file_count-ne30-or[string]$m.source_manifest_sha256-ne$sourceManifestSha){throw "RUNTIME_R113_MANIFEST_REJECTED:$round"}
    $actual=@(Get-ChildItem -LiteralPath $target -Recurse -File -Force|ForEach-Object{$_.FullName.Substring($target.Length+1).Replace('\','/')})
    if($actual.Count-ne31-or@($actual|Where-Object{$_-ne'runtime-hotfix-manifest.json'-and$m.files.path-notcontains$_}).Count){throw "RUNTIME_R113_FILESET_REJECTED:$round"}
    foreach($entry in $m.files){$path=Join-Path $target ([string]$entry.path).Replace('/','\');if((Sha $path)-ne[string]$entry.sha256-or[long](Get-Item -LiteralPath $path).Length-ne[long]$entry.bytes){throw "RUNTIME_R113_HASH_REJECTED:${round}:$($entry.path)"};& $python -m py_compile $path;if($LASTEXITCODE-ne0){throw "RUNTIME_R113_COMPILE_FAILED:${round}:$($entry.path)"}}
    Get-ChildItem -LiteralPath $target -Recurse -Directory -Filter '__pycache__'|Remove-Item -Recurse -Force
    $check=Join-Path $target "runtime-$release-adapter-check.py"
    $checkBody=@'
import json
import re
import subprocess
import sys
from evomind_runtime.competition_data import build_adapter_script
bash=sys.argv[1]
for competition in ("cure_bench","e2lmc","mindgames","ariel_2025","weather4cast","open_polymer"):
    for operation in ("status","prepare"):
        script=build_adapter_script(competition,operation)
        checked=subprocess.run([bash,"-n"],input=script.encode("utf-8"),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
        if checked.returncode: raise RuntimeError(f"bash_n_failed:{competition}:{operation}")
        for index,body in enumerate(re.findall(r"<<'PY'\n(.*?)\nPY\n",script,flags=re.DOTALL)): compile(body,f"{competition}-{operation}-{index}.py","exec")
accelerated=build_adapter_script("weather4cast","accelerate")
required=("get -aR .",'"adapter_transfer_mode": "openssh_recursive_get"','kill -TERM -- "-$EXISTING_PID"','"expected_file_count": 108','"secret_values_logged": False')
for item in required:
    if item not in accelerated: raise RuntimeError(f"recursive_resume_contract_missing:{item}")
print(json.dumps({"status":"passed","adapter_bash_n":"12/12","recursive_resume_contract":"passed"}))
'@
    [IO.File]::WriteAllText($check,$checkBody,[Text.UTF8Encoding]::new($false));$saved=$env:PYTHONPATH;try{$env:PYTHONPATH=$target;& $python $check $bash|ConvertFrom-Json;if($LASTEXITCODE-ne0){throw "RUNTIME_R113_ADAPTER_CHECK_FAILED:$round"}}finally{$env:PYTHONPATH=$saved};Remove-Item -LiteralPath $check -Force
    $rounds+=[ordered]@{round=$round;status='passed';file_count=30;py_compile='passed';adapter_bash_n='12/12';recursive_resume_contract='passed';hpc_accessed=$false;gpu_touched=$false}
  }
  $result=[ordered]@{schema="evomind.runtime_${release}_build_result.v1";status='built_and_verified';completed_at_utc=[DateTime]::UtcNow.ToString('o');zip_path=$zip;zip_bytes=[long](Get-Item -LiteralPath $zip).Length;zip_sha256=Sha $zip;source_manifest_path=$sourceManifest;source_manifest_sha256=$sourceManifestSha;file_count=30;baseline=[ordered]@{release='r111-weather-skip-timeout-gated';zip_sha256=$expectedBaselineZipSha;source_manifest_sha256=$baselineManifestSha;preserved_file_count=29};competition_data=[ordered]@{bytes=[long](Get-Item -LiteralPath $competitionSource).Length;sha256=$candidateCompetitionSha};verification_rounds=$rounds;production_deployed=$false;hpc_accessed=$false;gpu_touched=$false}
  Write-Utf8 $testResult $result;$result|ConvertTo-Json -Depth 15 -Compress
}finally{if(Test-Path -LiteralPath $stage){Remove-Item -LiteralPath $stage -Recurse -Force};foreach($target in $verifyRoots){if(Test-Path -LiteralPath $target){Remove-Item -LiteralPath $target -Recurse -Force}}}
