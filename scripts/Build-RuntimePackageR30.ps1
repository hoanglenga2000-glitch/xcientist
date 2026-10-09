param(
  [string]$SourceRoot='D:\桌面\codex\科研港科技',
  [string]$OutputRoot='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts',
  [string]$VerificationBasePath='D:\EV12',
  [ValidatePattern('^r[0-9]+(?:-[a-z0-9-]+)?$')][string]$ReleaseName='r30',
  [ValidatePattern('^[0-9]+/[0-9]+$')][string]$TargetedPytest='72/72',
  [ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedCompetitionDataSha='c4720bbcd2f5d50b812f6b919416effb2f90d778eeffae56e401afdc78bac960'
)

Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$source=[IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
$output=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$verifyBase=[IO.Path]::GetFullPath($VerificationBasePath).TrimEnd('\')
$release=$ReleaseName.ToLowerInvariant()
$sourceManifest=Join-Path $output "evomind-runtime-$release-source-manifest-20260822.json"
$zip=Join-Path $output "evomind-runtime-$release-20260822.zip"
$testResult=Join-Path $output "evomind-runtime-$release-test-results-20260822.json"
$runtimeNames=@(
  '__init__.py',
  'agent_kernel_v2.py',
  'assistant_runs.py',
  'benchmark.py',
  'capabilities.py',
  'cli.py',
  'client.py',
  'competition_data.py',
  'connectors.py',
  'credential_leases.py',
  'directory_broker.py',
  'ecosystem.py',
  'evidence_memory.py',
  'evolution.py',
  'hpc_runtime_overlay.py',
  'http_server.py',
  'mcp.py',
  'model_router.py',
  'models.py',
  'policy.py',
  'recovery.py',
  'remote_connectors.py',
  'runtime.py',
  'run_secrets.py',
  'store.py',
  'super_agent_runtime.py',
  'super_agent_store.py',
  'super_agent_tools.py',
  'tool_package_runtime.py',
  'tools.py'
)

function Sha([string]$Path) {
  (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-Utf8([string]$Path,[object]$Value) {
  $parent=Split-Path -Parent $Path
  if($parent){New-Item -ItemType Directory -Force -Path $parent|Out-Null}
  $json=if($Value-is[string]){[string]$Value}else{$Value|ConvertTo-Json -Depth 12}
  [IO.File]::WriteAllText($Path,$json.TrimEnd()+"`n",[Text.UTF8Encoding]::new($false))
}

foreach($path in @($sourceManifest,$zip,$testResult)) {
  if(Test-Path -LiteralPath $path){throw "RUNTIME_R30_OUTPUT_EXISTS:$path"}
}
if(-not(Test-Path -LiteralPath $source -PathType Container)){throw 'RUNTIME_R30_SOURCE_ROOT_MISSING'}
New-Item -ItemType Directory -Force -Path $output,$verifyBase|Out-Null

$python=Join-Path $source '.venv\Scripts\python.exe'
if(-not(Test-Path -LiteralPath $python -PathType Leaf)){
  $python=(Get-Command python.exe -ErrorAction Stop).Source
}

$entries=@()
foreach($name in $runtimeNames){
  $relative="src/evomind_runtime/$name"
  $path=Join-Path $source $relative.Replace('/','\')
  if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "RUNTIME_R30_SOURCE_MISSING:$relative"}
  $entries+=[ordered]@{path=$relative;bytes=[long](Get-Item -LiteralPath $path).Length;sha256=Sha $path}
}
$competitionEntry=@($entries|Where-Object{[string]$_.path-eq'src/evomind_runtime/competition_data.py'})
if($competitionEntry.Count-ne1-or[string]$competitionEntry[0].sha256-ne$ExpectedCompetitionDataSha){
  throw 'RUNTIME_R30_COMPETITION_DATA_R82_DRIFT'
}

$canonical=($entries|ForEach-Object{"$($_.path)`0$($_.sha256)`n"})-join''
$hash=[Security.Cryptography.SHA256]::Create()
try{
  $sourceTreeSha=([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($canonical)))).Replace('-','').ToLowerInvariant()
}finally{$hash.Dispose()}
$head=(& git -C $source rev-parse HEAD).Trim()
if($LASTEXITCODE-ne0-or$head-notmatch'^[a-f0-9]{40}$'){throw 'RUNTIME_R30_GIT_HEAD_INVALID'}
$dirty=@(& git -C $source status --porcelain=v1 --untracked-files=all)
$sourcePayload=[ordered]@{
  schema='evomind.super_agent_source_set.v1'
  created_at_utc=[DateTime]::UtcNow.ToString('o')
  purpose="$ReleaseName assistant Runtime package repair"
  frozen=$true
  git_head=$head
  source_dirty=($dirty.Count-gt0)
  source_tree_sha256=$sourceTreeSha
  file_count=$entries.Count
  files=$entries
  validation=[ordered]@{
    targeted_pytest=$TargetedPytest
    py_compile='passed'
    hpc_accessed=$false
    gpu_touched=$false
  }
}
Write-Utf8 $sourceManifest $sourcePayload
$sourceManifestSha=Sha $sourceManifest

$stage=Join-Path $verifyBase ("runtime-$release-stage-"+[guid]::NewGuid().ToString('N'))
$verifyRoots=@()
try{
  $package=Join-Path $stage 'evomind_runtime'
  New-Item -ItemType Directory -Force -Path $package|Out-Null
  $packageEntries=@()
  foreach($entry in $entries){
    $sourcePath=Join-Path $source ([string]$entry.path).Replace('/','\')
    if((Sha $sourcePath)-ne[string]$entry.sha256-or[long](Get-Item -LiteralPath $sourcePath).Length-ne[long]$entry.bytes){
      throw "RUNTIME_R30_SOURCE_CHANGED:$($entry.path)"
    }
    $destination=Join-Path $package ([IO.Path]::GetFileName($sourcePath))
    Copy-Item -LiteralPath $sourcePath -Destination $destination
    & $python -m py_compile $destination
    if($LASTEXITCODE-ne0){throw "RUNTIME_R30_COMPILE_FAILED:$($entry.path)"}
    $packageEntries+=[ordered]@{
      path=('evomind_runtime/'+[IO.Path]::GetFileName($sourcePath))
      bytes=[long](Get-Item -LiteralPath $destination).Length
      sha256=Sha $destination
    }
  }
  Get-ChildItem -LiteralPath $stage -Recurse -Directory -Filter '__pycache__'|Remove-Item -Recurse -Force
  $hotfix=[ordered]@{
    schema='evomind.super_agent_runtime_hotfix.v1'
    created_at_utc=[DateTime]::UtcNow.ToString('o')
    release=$release
    source_manifest_sha256=$sourceManifestSha
    file_count=$packageEntries.Count
    files=$packageEntries
    target='bundle/runtime/evomind_runtime'
    production_deployed=$false
    hpc_accessed=$false
    gpu_touched=$false
  }
  Write-Utf8 (Join-Path $stage 'runtime-hotfix-manifest.json') $hotfix
  Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal

  $rounds=@()
  foreach($round in 1..2){
    $target=Join-Path $verifyBase ("runtime-$release-verify-"+[guid]::NewGuid().ToString('N'))
    $verifyRoots+=$target
    New-Item -ItemType Directory -Path $target|Out-Null
    Expand-Archive -LiteralPath $zip -DestinationPath $target
    if(@($target.ToCharArray()|Where-Object{[int]$_-gt127}).Count){throw "RUNTIME_R30_NON_ASCII_VERIFY_PATH:$round"}
    $manifestPath=Join-Path $target 'runtime-hotfix-manifest.json'
    $manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8|ConvertFrom-Json
    if([string]$manifest.schema-ne'evomind.super_agent_runtime_hotfix.v1'-or[int]$manifest.file_count-ne$runtimeNames.Count){
      throw "RUNTIME_R30_MANIFEST_REJECTED:$round"
    }
    $actual=@(Get-ChildItem -LiteralPath $target -Recurse -File -Force|ForEach-Object{$_.FullName.Substring($target.Length+1).Replace('\','/')})
    if($actual.Count-ne($runtimeNames.Count+1)-or@($actual|Where-Object{$_-ne'runtime-hotfix-manifest.json'-and$manifest.files.path-notcontains$_}).Count){
      throw "RUNTIME_R30_FILESET_REJECTED:$round"
    }
    foreach($entry in $manifest.files){
      $path=Join-Path $target ([string]$entry.path).Replace('/','\')
      if((Sha $path)-ne[string]$entry.sha256-or[long](Get-Item -LiteralPath $path).Length-ne[long]$entry.bytes){
        throw "RUNTIME_R30_HASH_REJECTED:${round}:$($entry.path)"
      }
      & $python -m py_compile $path
      if($LASTEXITCODE-ne0){throw "RUNTIME_R30_ROUND_COMPILE_FAILED:${round}:$($entry.path)"}
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
    session = runtime.create_session(
        session_id=current.name,
        workspace_root=str(current),
        permission_level="workspace-write",
    )
    outcome = runtime.invoke_tool(
        session["id"],
        "file_write",
        {"path": str(sibling / "outputs" / "result.json"), "content": "{}"},
    )
    assert outcome["status"] == "failed"
    assert outcome["result"]["error"] == "cross_run_path"
    assert outcome["decision"]["scope"]["cross_run_workspace"] is True
    assert runtime.store.list_approvals(status="pending") == []
    assert not sibling.exists()
    assert pathlib.Path(hpc_overlay.__file__).name == "hpc_runtime_overlay.py"
    tool_names = {item["name"] for item in runtime.tools()}
    assert {"competition_data_prepare", "competition_data_status", "competition_data_progress"} <= tool_names
    assert {"super_agent_status", "capability_discover", "directory_read", "transfer_fetch", "job_execute"} <= tool_names
    assert len(competition_data.catalog_projection()) == 6
    assert run_secrets.RunSecretStore is not None
    super_agent = runtime.super_agent.status()
    assert super_agent["mode"] == "shadow"
    assert super_agent["migration_applied"] is True
    print(json.dumps({"status": "passed", "error": "cross_run_path", "pending_approvals": 0, "competition_tools": 3, "competition_sources": 6, "run_secret_fail_closed": True, "super_agent_mode": "shadow"}))
finally:
    runtime.close()
'@
    [IO.File]::WriteAllText($smoke,$smokeBody,[Text.UTF8Encoding]::new($false))
    $saved=$env:PYTHONPATH
    try{
      $env:PYTHONPATH=$target
      $smokeResult=& $python $smoke $verifyBase|ConvertFrom-Json
      if($LASTEXITCODE-ne0-or[string]$smokeResult.status-ne'passed'-or[string]$smokeResult.error-ne'cross_run_path'){
        throw "RUNTIME_R30_SMOKE_FAILED:$round"
      }
    }finally{$env:PYTHONPATH=$saved}
    Remove-Item -LiteralPath $smoke -Force
    $rounds+=[ordered]@{
      round=$round
      status='passed'
      file_count=$runtimeNames.Count
      py_compile='passed'
      cross_run_path='fail_closed'
      pending_approvals=0
      network_accessed=$false
      hpc_accessed=$false
      gpu_touched=$false
    }
  }

  $result=[ordered]@{
    schema="evomind.runtime_${release}_build_result.v1"
    status='built_and_verified'
    completed_at_utc=[DateTime]::UtcNow.ToString('o')
    zip_path=$zip
    zip_bytes=[long](Get-Item -LiteralPath $zip).Length
    zip_sha256=Sha $zip
    source_manifest_path=$sourceManifest
    source_manifest_sha256=$sourceManifestSha
    file_count=$runtimeNames.Count
    verification_rounds=$rounds
    production_deployed=$false
    hpc_accessed=$false
    gpu_touched=$false
  }
  Write-Utf8 $testResult $result
  $result|ConvertTo-Json -Depth 10 -Compress
}finally{
  if(Test-Path -LiteralPath $stage){Remove-Item -LiteralPath $stage -Recurse -Force}
  foreach($target in $verifyRoots){if(Test-Path -LiteralPath $target){Remove-Item -LiteralPath $target -Recurse -Force}}
}
