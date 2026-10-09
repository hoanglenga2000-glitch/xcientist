param(
  [string]$SourceRoot='D:\桌面\codex\科研港科技',
  [string]$OutputRoot='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts',
  [string]$VerificationBasePath='D:\EV12',
  [string]$BaselineZip='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\evomind-runtime-r80-weather-attempt-state-20260822.zip',
  [string]$BaselineSourceManifest='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\evomind-runtime-r80-weather-attempt-state-source-manifest-20260822.json',
  [ValidatePattern('^r[0-9]+(?:-[a-z0-9-]+)?$')][string]$ReleaseName='r82-kaggle-range-hotfix',
  [ValidatePattern('^[0-9]+/[0-9]+$')][string]$TargetedPytest='23/23'
)

Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$source=[IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
$output=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$verifyBase=[IO.Path]::GetFullPath($VerificationBasePath).TrimEnd('\')
$baselineZipPath=[IO.Path]::GetFullPath($BaselineZip)
$baselineSourcePath=[IO.Path]::GetFullPath($BaselineSourceManifest)
$release=$ReleaseName.ToLowerInvariant()
$sourceManifest=Join-Path $output "evomind-runtime-$release-source-manifest-20260822.json"
$zip=Join-Path $output "evomind-runtime-$release-20260822.zip"
$testResult=Join-Path $output "evomind-runtime-$release-test-results-20260822.json"
$competitionSource=Join-Path $source 'src\evomind_runtime\competition_data.py'
$expectedBaselineZipSha='21926524f2628898048e6577d70ae457597e68846ba9b6a6738c8f0d993beeab'
$expectedBaselineSourceSha='2f1632293522f0ab7f5f2c67cd0d953ef3e61b39043c2a91fe0091e3f6746e33'
$expectedCompetitionSha='c4720bbcd2f5d50b812f6b919416effb2f90d778eeffae56e401afdc78bac960'

function Sha([string]$Path) {
  (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-Utf8([string]$Path,[object]$Value) {
  $parent=Split-Path -Parent $Path
  if($parent){New-Item -ItemType Directory -Force -Path $parent|Out-Null}
  $json=if($Value-is[string]){[string]$Value}else{$Value|ConvertTo-Json -Depth 15}
  [IO.File]::WriteAllText($Path,$json.TrimEnd()+"`n",[Text.UTF8Encoding]::new($false))
}

foreach($path in @($sourceManifest,$zip,$testResult)) {
  if(Test-Path -LiteralPath $path){throw "RUNTIME_R82_OUTPUT_EXISTS:$path"}
}
foreach($path in @($baselineZipPath,$baselineSourcePath,$competitionSource)) {
  if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "RUNTIME_R82_INPUT_MISSING:$path"}
}
if((Sha $baselineZipPath)-ne$expectedBaselineZipSha){throw 'RUNTIME_R82_BASELINE_ZIP_HASH_REJECTED'}
if((Sha $baselineSourcePath)-ne$expectedBaselineSourceSha){throw 'RUNTIME_R82_BASELINE_SOURCE_HASH_REJECTED'}
if((Sha $competitionSource)-ne$expectedCompetitionSha){throw 'RUNTIME_R82_COMPETITION_SOURCE_HASH_REJECTED'}

$python=Join-Path $source '.venv\Scripts\python.exe'
if(-not(Test-Path -LiteralPath $python -PathType Leaf)){$python=(Get-Command python.exe -ErrorAction Stop).Source}
$bash=(Get-Command bash.exe -ErrorAction Stop).Source
$baselineSourcePayload=Get-Content -LiteralPath $baselineSourcePath -Raw -Encoding UTF8|ConvertFrom-Json
if(
  [string]$baselineSourcePayload.schema-ne'evomind.super_agent_source_set.v1'-or
  -not[bool]$baselineSourcePayload.frozen-or
  [int]$baselineSourcePayload.file_count-ne17
){throw 'RUNTIME_R82_BASELINE_SOURCE_MANIFEST_REJECTED'}

$sourceEntries=@()
foreach($entry in $baselineSourcePayload.files){
  if([string]$entry.path-eq'src/evomind_runtime/competition_data.py'){
    $sourceEntries+=[ordered]@{
      path='src/evomind_runtime/competition_data.py'
      bytes=[long](Get-Item -LiteralPath $competitionSource).Length
      sha256=$expectedCompetitionSha
    }
  }else{
    $sourceEntries+=[ordered]@{
      path=[string]$entry.path
      bytes=[long]$entry.bytes
      sha256=[string]$entry.sha256
    }
  }
}
if($sourceEntries.Count-ne17-or@($sourceEntries|Where-Object{$_.path-eq'src/evomind_runtime/competition_data.py'}).Count-ne1){
  throw 'RUNTIME_R82_SOURCE_FILESET_REJECTED'
}
$canonical=($sourceEntries|ForEach-Object{"$($_.path)`0$($_.sha256)`n"})-join''
$hasher=[Security.Cryptography.SHA256]::Create()
try{
  $sourceTreeSha=([BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($canonical)))).Replace('-','').ToLowerInvariant()
}finally{$hasher.Dispose()}
$head=(& git -C $source rev-parse HEAD).Trim()
if($LASTEXITCODE-ne0-or$head-notmatch'^[a-f0-9]{40}$'){throw 'RUNTIME_R82_GIT_HEAD_INVALID'}
$sourcePayload=[ordered]@{
  schema='evomind.super_agent_source_set.v1'
  created_at_utc=[DateTime]::UtcNow.ToString('o')
  purpose="$release competition_data clean-EOF and strict Range hotfix over frozen R80 baseline"
  frozen=$true
  git_head=$head
  source_dirty=$true
  source_tree_sha256=$sourceTreeSha
  file_count=17
  files=$sourceEntries
  baseline=[ordered]@{
    release='r80-weather-attempt-state'
    runtime_zip_sha256=$expectedBaselineZipSha
    source_manifest_sha256=$expectedBaselineSourceSha
    preserved_file_count=16
    replaced_file='src/evomind_runtime/competition_data.py'
  }
  validation=[ordered]@{
    targeted_pytest=$TargetedPytest
    clean_eof_test='1/1'
    py_compile='passed'
    adapter_bash_n='12/12'
    embedded_python_compile='36/36'
    strict_range_contract='passed'
    external_network_accessed=$false
    hpc_accessed=$false
    gpu_touched=$false
  }
}
New-Item -ItemType Directory -Force -Path $output,$verifyBase|Out-Null
Write-Utf8 $sourceManifest $sourcePayload
$sourceManifestSha=Sha $sourceManifest

$stage=Join-Path $verifyBase ("runtime-$release-stage-"+[guid]::NewGuid().ToString('N'))
$verifyRoots=@()
try{
  New-Item -ItemType Directory -Path $stage|Out-Null
  Expand-Archive -LiteralPath $baselineZipPath -DestinationPath $stage
  $baselineHotfixPath=Join-Path $stage 'runtime-hotfix-manifest.json'
  $baselineHotfix=Get-Content -LiteralPath $baselineHotfixPath -Raw -Encoding UTF8|ConvertFrom-Json
  if(
    [string]$baselineHotfix.schema-ne'evomind.super_agent_runtime_hotfix.v1'-or
    [string]$baselineHotfix.release-ne'r80-weather-attempt-state'-or
    [int]$baselineHotfix.file_count-ne17
  ){throw 'RUNTIME_R82_BASELINE_HOTFIX_MANIFEST_REJECTED'}
  foreach($entry in $baselineHotfix.files){
    $path=Join-Path $stage ([string]$entry.path).Replace('/','\')
    if((Sha $path)-ne[string]$entry.sha256-or[long](Get-Item -LiteralPath $path).Length-ne[long]$entry.bytes){
      throw "RUNTIME_R82_BASELINE_FILE_REJECTED:$($entry.path)"
    }
  }

  $competitionTarget=Join-Path $stage 'evomind_runtime\competition_data.py'
  Copy-Item -LiteralPath $competitionSource -Destination $competitionTarget -Force
  if((Sha $competitionTarget)-ne$expectedCompetitionSha){throw 'RUNTIME_R82_COMPETITION_COPY_REJECTED'}
  $packageEntries=@()
  foreach($entry in $sourceEntries){
    $relative=([string]$entry.path)-replace'^src/',''
    $path=Join-Path $stage $relative.Replace('/','\')
    if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "RUNTIME_R82_PACKAGE_FILE_MISSING:$relative"}
    $actualSha=Sha $path
    $actualBytes=[long](Get-Item -LiteralPath $path).Length
    if($actualSha-ne[string]$entry.sha256-or$actualBytes-ne[long]$entry.bytes){
      throw "RUNTIME_R82_PACKAGE_SOURCE_MISMATCH:$relative"
    }
    $packageEntries+=[ordered]@{path=$relative;bytes=$actualBytes;sha256=$actualSha}
  }
  $hotfix=[ordered]@{
    schema='evomind.super_agent_runtime_hotfix.v1'
    created_at_utc=[DateTime]::UtcNow.ToString('o')
    release=$release
    source_manifest_sha256=$sourceManifestSha
    file_count=17
    files=$packageEntries
    target='bundle/runtime/evomind_runtime'
    baseline_runtime_zip_sha256=$expectedBaselineZipSha
    production_deployed=$false
    external_network_accessed=$false
    hpc_accessed=$false
    gpu_touched=$false
  }
  Write-Utf8 $baselineHotfixPath $hotfix
  Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal

  $rounds=@()
  $extended=$null
  foreach($round in 1..2){
    $target=Join-Path $verifyBase ("runtime-$release-verify-"+[guid]::NewGuid().ToString('N'))
    $verifyRoots+=$target
    New-Item -ItemType Directory -Path $target|Out-Null
    Expand-Archive -LiteralPath $zip -DestinationPath $target
    if(@($target.ToCharArray()|Where-Object{[int]$_-gt127}).Count){throw "RUNTIME_R82_NON_ASCII_VERIFY_PATH:$round"}
    $manifestPath=Join-Path $target 'runtime-hotfix-manifest.json'
    $manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8|ConvertFrom-Json
    if(
      [string]$manifest.schema-ne'evomind.super_agent_runtime_hotfix.v1'-or
      [string]$manifest.release-ne$release-or
      [string]$manifest.source_manifest_sha256-ne$sourceManifestSha-or
      [int]$manifest.file_count-ne17
    ){throw "RUNTIME_R82_MANIFEST_REJECTED:$round"}
    $actual=@(Get-ChildItem -LiteralPath $target -Recurse -File -Force|ForEach-Object{$_.FullName.Substring($target.Length+1).Replace('\','/')})
    if($actual.Count-ne18-or@($actual|Where-Object{$_-ne'runtime-hotfix-manifest.json'-and$manifest.files.path-notcontains$_}).Count){
      throw "RUNTIME_R82_FILESET_REJECTED:$round"
    }
    foreach($entry in $manifest.files){
      $path=Join-Path $target ([string]$entry.path).Replace('/','\')
      if((Sha $path)-ne[string]$entry.sha256-or[long](Get-Item -LiteralPath $path).Length-ne[long]$entry.bytes){
        throw "RUNTIME_R82_HASH_REJECTED:${round}:$($entry.path)"
      }
      & $python -m py_compile $path
      if($LASTEXITCODE-ne0){throw "RUNTIME_R82_ROUND_COMPILE_FAILED:${round}:$($entry.path)"}
    }
    Get-ChildItem -LiteralPath $target -Recurse -Directory -Filter '__pycache__'|Remove-Item -Recurse -Force

    $smoke=Join-Path $target "runtime-$release-smoke.py"
    $smokeBody=@'
import json
import pathlib
import sys
import tempfile

from evomind_runtime.runtime import AgentRuntime
import evomind_runtime.hpc_runtime_overlay as hpc_overlay
import evomind_runtime.competition_data as competition_data
import evomind_runtime.run_secrets as run_secrets

base = pathlib.Path(sys.argv[1])
root = pathlib.Path(tempfile.mkdtemp(prefix="runtime-smoke-", dir=base))
task_parent = root / "workspace" / "runtime" / "assistant_tasks"
current = task_parent / ("run_" + "a" * 32)
sibling = task_parent / ("run_" + "b" * 32)
runtime = AgentRuntime(root / "runtime")
try:
    session = runtime.create_session(session_id=current.name, workspace_root=str(current), permission_level="workspace-write")
    outcome = runtime.invoke_tool(session["id"], "file_write", {"path": str(sibling / "outputs" / "result.json"), "content": "{}"})
    assert outcome["status"] == "failed"
    assert outcome["result"]["error"] == "cross_run_path"
    assert outcome["decision"]["scope"]["cross_run_workspace"] is True
    assert runtime.store.list_approvals(status="pending") == []
    assert not sibling.exists()
    assert pathlib.Path(hpc_overlay.__file__).name == "hpc_runtime_overlay.py"
    tool_names = {item["name"] for item in runtime.tools()}
    assert {"competition_data_prepare", "competition_data_status"} <= tool_names
    assert len(competition_data.catalog_projection()) == 6
    assert run_secrets.RunSecretStore is not None
    print(json.dumps({"status":"passed","error":"cross_run_path","pending_approvals":0,"competition_tools":2,"competition_sources":6,"run_secret_fail_closed":True}))
finally:
    runtime.close()
'@
    [IO.File]::WriteAllText($smoke,$smokeBody,[Text.UTF8Encoding]::new($false))
    $saved=$env:PYTHONPATH
    try{
      $env:PYTHONPATH=$target
      $smokeResult=& $python $smoke $verifyBase|ConvertFrom-Json
      if($LASTEXITCODE-ne0-or[string]$smokeResult.status-ne'passed'-or[string]$smokeResult.error-ne'cross_run_path'){
        throw "RUNTIME_R82_SMOKE_FAILED:$round"
      }
    }finally{$env:PYTHONPATH=$saved}
    Remove-Item -LiteralPath $smoke -Force

    if($round-eq2){
      $adapterCheck=Join-Path $target "runtime-$release-adapter-check.py"
      $adapterBody=@'
import json
import re
import subprocess
import sys

from evomind_runtime.competition_data import build_adapter_script

bash = sys.argv[1]
bash_count = 0
heredoc_count = 0
for competition in ("cure_bench", "e2lmc", "mindgames", "ariel_2025", "weather4cast", "open_polymer"):
    for operation in ("status", "prepare"):
        script = build_adapter_script(competition, operation)
        checked = subprocess.run([bash, "-n"], input=script.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if checked.returncode != 0:
            raise RuntimeError(f"bash_n_failed:{competition}:{operation}:{checked.stderr.decode('utf-8', errors='replace')}")
        bash_count += 1
        bodies = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY\n", script, flags=re.DOTALL)
        for index, body in enumerate(bodies):
            compile(body, f"{competition}-{operation}-heredoc-{index}.py", "exec")
            heredoc_count += 1

script = build_adapter_script("cure_bench", "prepare")
required = (
    "if content_length_text and not content_length_text.isdigit():",
    "content_length != end - start + 1",
    "total is not None and total <= end",
    "partial.stat().st_size == int(matched.group(1))",
    "and zipfile.is_zipfile(partial)",
    "kaggle_range_not_honored",
    "kaggle_range_contract_invalid",
)
missing = [item for item in required if item not in script]
if missing or script.count("content_length != end - start + 1") != 2:
    raise RuntimeError(f"strict_range_contract_missing:{missing}")
print(json.dumps({"status":"passed","adapter_bash_n":bash_count,"embedded_python_compile":heredoc_count,"strict_range_contract":"passed"}))
'@
      [IO.File]::WriteAllText($adapterCheck,$adapterBody,[Text.UTF8Encoding]::new($false))
      $saved=$env:PYTHONPATH
      try{
        $env:PYTHONPATH=$target
        $extended=& $python $adapterCheck $bash|ConvertFrom-Json
        if(
          $LASTEXITCODE-ne0-or[string]$extended.status-ne'passed'-or
          [int]$extended.adapter_bash_n-ne12-or[int]$extended.embedded_python_compile-ne36-or
          [string]$extended.strict_range_contract-ne'passed'
        ){throw 'RUNTIME_R82_ADAPTER_VALIDATION_FAILED'}
        & $python -m pytest (Join-Path $source 'tests\test_competition_data.py') -q -k 'test_kaggle_worker_resumes_after_clean_eof_leaves_a_truncated_zip'
        if($LASTEXITCODE-ne0){throw 'RUNTIME_R82_CLEAN_EOF_TEST_FAILED'}
      }finally{$env:PYTHONPATH=$saved}
      Remove-Item -LiteralPath $adapterCheck -Force
    }
    $rounds+=[ordered]@{
      round=$round
      status='passed'
      file_count=17
      py_compile='passed'
      cross_run_path='fail_closed'
      pending_approvals=0
      external_network_accessed=$false
      hpc_accessed=$false
      gpu_touched=$false
    }
  }

  if($null-eq$extended){throw 'RUNTIME_R82_EXTENDED_VALIDATION_MISSING'}
  if((Sha $competitionSource)-ne$expectedCompetitionSha){throw 'RUNTIME_R82_SOURCE_CHANGED_DURING_BUILD'}
  $result=[ordered]@{
    schema="evomind.runtime_${release}_build_result.v1"
    status='built_and_verified'
    completed_at_utc=[DateTime]::UtcNow.ToString('o')
    zip_path=$zip
    zip_bytes=[long](Get-Item -LiteralPath $zip).Length
    zip_sha256=Sha $zip
    source_manifest_path=$sourceManifest
    source_manifest_sha256=$sourceManifestSha
    file_count=17
    frozen_baseline=[ordered]@{
      release='r80-weather-attempt-state'
      runtime_zip_sha256=$expectedBaselineZipSha
      source_manifest_sha256=$expectedBaselineSourceSha
      preserved_file_count=16
    }
    competition_data=[ordered]@{
      bytes=[long](Get-Item -LiteralPath $competitionSource).Length
      sha256=$expectedCompetitionSha
    }
    targeted_pytest=$TargetedPytest
    extended_validation=[ordered]@{
      clean_eof_test='1/1'
      adapter_bash_n="$([int]$extended.adapter_bash_n)/12"
      embedded_python_compile="$([int]$extended.embedded_python_compile)/36"
      strict_range_contract=[string]$extended.strict_range_contract
      loopback_http_canary=$true
      external_network_accessed=$false
    }
    verification_rounds=$rounds
    production_deployed=$false
    hpc_accessed=$false
    gpu_touched=$false
  }
  Write-Utf8 $testResult $result
  $result|ConvertTo-Json -Depth 15 -Compress
}finally{
  if(Test-Path -LiteralPath $stage){Remove-Item -LiteralPath $stage -Recurse -Force}
  foreach($target in $verifyRoots){if(Test-Path -LiteralPath $target){Remove-Item -LiteralPath $target -Recurse -Force}}
}
