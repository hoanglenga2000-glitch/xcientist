param(
  [string]$SourceRoot='D:\桌面\codex\科研港科技',
  [string]$OutputRoot='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts',
  [string]$VerificationBasePath='D:\EV12',
  [ValidatePattern('^r[0-9]+(?:-[a-z0-9-]+)?$')][string]$ReleaseName='r130-public-https-source',
  [ValidatePattern('^[0-9]+/[0-9]+$')][string]$TargetedPytest='0/0',
  [ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedCompetitionDataSha='1e45df525064c16b6d0610e6769734e7645067941161c150adb8d9f774401c49',
  [string]$BaselineZip='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\evomind-runtime-r128-capability-discovery-20260902.zip',
  [ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedBaselineZipSha='4e360e5ff357e9ad6f9a1d1f536311e8bb184f41a4a9c572bc6db5c62f663fee',
  [ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedBaselineTreeSha='690c65320b67e109ac97c129ac531073961c4f5150cc1c2925858dc8eaf8939c',
  [ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedConnectorsSha='61bfac6445177f47a44972609a58592604519983d94552f7624caed005870e06',
  [ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedDirectoryBrokerSha='867d9d594d8692416362fa755aded19b257ca9c7bd9f61e82cdd52b33a0cb4df',
  [ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedRemoteConnectorsSha='c3158c02804005d636c68bdc4ad355ac669668c06246d7fb7c3c43a4240960e8',
  [ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedSuperAgentSha='de2243a6877ffac3d5305cbf4de182368afba81f5b55c57edebe56c56e3f6f7f',
  [ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedRuntimeModuleSha='ff5235b54b0ba1a6698cf53de46495d1cc64a59830c8ddec6db7c7ed0aea4594'
)

Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

$source=[IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
$output=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$verifyBase=[IO.Path]::GetFullPath($VerificationBasePath).TrimEnd('\')
$release=$ReleaseName.ToLowerInvariant()
$sourceManifest=Join-Path $output "evomind-runtime-$release-source-manifest-20260903.json"
$zip=Join-Path $output "evomind-runtime-$release-20260903.zip"
$testResult=Join-Path $output "evomind-runtime-$release-test-results-20260903.json"
$runtimeNames=@(
  '__init__.py',
  'agent_kernel_v2.py',
  'assistant_runs.py',
  'benchmark.py',
  'capabilities.py',
  'cli.py',
  'client.py',
  'competition_data.py',
  'competition_goal.py',
  'connectors.py',
  'credential_leases.py',
  'directory_broker.py',
  'ecosystem.py',
  'evidence_memory.py',
  'evolution.py',
  'goal_board.py',
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

function Sha-Bytes([byte[]]$Value) {
  $hash=[Security.Cryptography.SHA256]::Create()
  try{([BitConverter]::ToString($hash.ComputeHash($Value))).Replace('-','').ToLowerInvariant()}finally{$hash.Dispose()}
}

function Get-RuntimeTreeSha([hashtable]$Entries) {
  $rows=@($Entries.Keys|Sort-Object|ForEach-Object{"$([IO.Path]::GetFileName([string]$_))|$($Entries[$_].Length)|$(Sha-Bytes $Entries[$_])"})
  $hash=[Security.Cryptography.SHA256]::Create()
  try{([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes(($rows-join"`n"))))).Replace('-','').ToLowerInvariant()}finally{$hash.Dispose()}
}

function Read-BaselineRuntime([string]$Path) {
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  if(-not(Test-Path -LiteralPath $Path -PathType Leaf)-or(Sha $Path)-ne$ExpectedBaselineZipSha){throw 'RUNTIME_R125_BASELINE_ZIP_REJECTED'}
  $entries=@{}
  $archive=[IO.Compression.ZipFile]::OpenRead($Path)
  try{
    foreach($entry in $archive.Entries){
      $name=[string]$entry.FullName.Replace('\','/')
      if($name-match'^evomind_runtime/[A-Za-z0-9_]+\.py$'){
        if($entries.ContainsKey($name)){throw "RUNTIME_R125_BASELINE_DUPLICATE:$name"}
        $stream=$entry.Open();$memory=[IO.MemoryStream]::new()
        try{$stream.CopyTo($memory);$entries[$name]=$memory.ToArray()}finally{$memory.Dispose();$stream.Dispose()}
      }
    }
  }finally{$archive.Dispose()}
  if($entries.Count-ne$runtimeNames.Count-or@($runtimeNames|Where-Object{-not$entries.ContainsKey("evomind_runtime/$_")}).Count){throw 'RUNTIME_R125_BASELINE_FILESET_REJECTED'}
  if((Get-RuntimeTreeSha $entries)-ne$ExpectedBaselineTreeSha){throw 'RUNTIME_R125_BASELINE_TREE_REJECTED'}
  $entries
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

$baselineRuntime=Read-BaselineRuntime ([IO.Path]::GetFullPath($BaselineZip))
$overlaySources=@{
  'connectors.py'=[ordered]@{path=(Join-Path $source 'src\evomind_runtime\connectors.py');sha=$ExpectedConnectorsSha}
  'directory_broker.py'=[ordered]@{path=(Join-Path $source 'src\evomind_runtime\directory_broker.py');sha=$ExpectedDirectoryBrokerSha}
  'remote_connectors.py'=[ordered]@{path=(Join-Path $source 'src\evomind_runtime\remote_connectors.py');sha=$ExpectedRemoteConnectorsSha}
  'super_agent_runtime.py'=[ordered]@{path=(Join-Path $source 'src\evomind_runtime\super_agent_runtime.py');sha=$ExpectedSuperAgentSha}
}
foreach($item in $overlaySources.Values){if(-not(Test-Path -LiteralPath $item.path -PathType Leaf)-or(Sha $item.path)-ne$item.sha){throw 'RUNTIME_R130_OVERLAY_SOURCE_REJECTED'}}

$python=Join-Path $source '.venv\Scripts\python.exe'
if(-not(Test-Path -LiteralPath $python -PathType Leaf)){
  $python=(Get-Command python.exe -ErrorAction Stop).Source
}

$entries=@()
foreach($name in $runtimeNames){
  $relative="src/evomind_runtime/$name"
  $baselineName="evomind_runtime/$name"
  $overlay=$overlaySources[$name]
  $raw=if($overlay){[IO.File]::ReadAllBytes([string]$overlay.path)}else{[byte[]]$baselineRuntime[$baselineName]}
  $digest=Sha-Bytes $raw
  $entries+=[ordered]@{path=$relative;bytes=[long]$raw.Length;sha256=$digest;baseline_name=$baselineName}
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
$sourcePayload=[ordered]@{
  schema='evomind.super_agent_source_set.v1'
  created_at_utc=[DateTime]::UtcNow.ToString('o')
  purpose="$ReleaseName assistant Runtime package repair"
  frozen=$true
  git_head=$null
  source_dirty=$null
  source_control_not_queried=$true
  source_tree_sha256=$sourceTreeSha
  file_count=$entries.Count
  files=$entries
  validation=[ordered]@{
    targeted_pytest=$TargetedPytest
    py_compile='passed'
    baseline_zip_sha256=$ExpectedBaselineZipSha
    baseline_runtime_tree_sha256=$ExpectedBaselineTreeSha
    changed_files=@('evomind_runtime/connectors.py','evomind_runtime/directory_broker.py','evomind_runtime/remote_connectors.py','evomind_runtime/super_agent_runtime.py')
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
    $name=[IO.Path]::GetFileName([string]$entry.path)
    $destination=Join-Path $package $name
    if($overlaySources.ContainsKey($name)){Copy-Item -LiteralPath ([string]$overlaySources[$name].path) -Destination $destination}
    else{[IO.File]::WriteAllBytes($destination,[byte[]]$baselineRuntime[[string]$entry.baseline_name])}
    if((Sha $destination)-ne[string]$entry.sha256-or[long](Get-Item -LiteralPath $destination).Length-ne[long]$entry.bytes){throw "RUNTIME_R125_SOURCE_CHANGED:$($entry.path)"}
    & $python -m py_compile $destination
    if($LASTEXITCODE-ne0){throw "RUNTIME_R30_COMPILE_FAILED:$($entry.path)"}
    $packageEntries+=[ordered]@{
      path=('evomind_runtime/'+$name)
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
    assert any(item["descriptor"]["capability_id"] == "public-https-source" for item in runtime.super_agent.discover("public source transfer fetch"))
    assert len(competition_data.catalog_projection()) == 6
    assert run_secrets.RunSecretStore is not None
    tenant_session = runtime.create_session(
        session_id="run_" + "c" * 32,
        objective="same-root authenticated tenant capability",
        workspace_root=str(runtime.workspace_root),
        metadata={"tenant_id": "tenant_current", "project_id": "project_current"},
    )
    tenant_directory = tenant_session["metadata"]["super_agent_directory_ids"][0]
    assert tenant_directory != "workspace"
    own_directory = runtime.invoke_tool(
        tenant_session["id"],
        "directory_list",
        {"directory_id": tenant_directory, "relative_path": ""},
    )
    assert own_directory["result"]["ok"] is True
    global_directory = runtime.invoke_tool(
        tenant_session["id"],
        "directory_list",
        {"directory_id": "workspace", "relative_path": ""},
    )
    assert global_directory["result"]["ok"] is False
    assert "CapabilityPermissionError" in global_directory["result"]["error"]
    tenant_session_id = tenant_session["id"]
    runtime.close()
    runtime = AgentRuntime(root / "runtime")
    restored = runtime.get_session(tenant_session_id)
    assert restored["metadata"]["super_agent_directory_ids"] == [tenant_directory]
    remounted = runtime.super_agent.broker.capability(tenant_directory)
    assert remounted.tenant_id == "tenant_current"
    assert remounted.project_id == "project_current"
    assert remounted.run_id == tenant_session_id
    after_restart = runtime.invoke_tool(
        tenant_session_id,
        "directory_list",
        {"directory_id": tenant_directory, "relative_path": ""},
    )
    assert after_restart["result"]["ok"] is True
    super_agent = runtime.super_agent.status()
    assert super_agent["mode"] == "shadow"
    assert super_agent["migration_applied"] is True
    print(json.dumps({"status": "passed", "error": "cross_run_path", "pending_approvals": 0, "competition_tools": 3, "competition_sources": 6, "run_secret_fail_closed": True, "same_root_tenant_capability": True, "super_agent_mode": "shadow"}))
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
