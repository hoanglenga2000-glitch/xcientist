param(
  [string]$SourceRoot='D:\桌面\codex\科研港科技',
  [string]$PreviousOverlayRoot='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\web-overlay-20260824-competition-data-r41',
  [string]$PreviousSourceManifest='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\competition-data-r41-source-manifest-20260824.json',
  [Parameter(Mandatory)][string]$SourceManifestPath,
  [Parameter(Mandatory)][string]$OutputRoot,
  [Parameter(Mandatory)][ValidatePattern('^overlay-[a-z0-9-]+$')][string]$BuildId,
  [string]$VerificationBasePath='D:\EV12'
)

Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

function Sha([string]$Path) {
  (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-Utf8Json([string]$Path,[object]$Value) {
  [IO.File]::WriteAllText($Path,(($Value|ConvertTo-Json -Depth 12)+"`n"),[Text.UTF8Encoding]::new($false))
}

function Assert-ChildPath([string]$Path,[string]$Parent,[string]$Code) {
  $resolved=[IO.Path]::GetFullPath($Path).TrimEnd('\')
  $root=[IO.Path]::GetFullPath($Parent).TrimEnd('\')+'\'
  if(-not$resolved.StartsWith($root,[StringComparison]::OrdinalIgnoreCase)){throw $Code}
  return $resolved
}

$source=[IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
$web=Join-Path $source 'web\research-agent-workstation'
$previous=[IO.Path]::GetFullPath($PreviousOverlayRoot).TrimEnd('\')
$sourceManifestFile=[IO.Path]::GetFullPath($SourceManifestPath)
$previousManifestFile=[IO.Path]::GetFullPath($PreviousSourceManifest)
$output=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$verifyBase=[IO.Path]::GetFullPath($VerificationBasePath).TrimEnd('\')
$stage=Assert-ChildPath (Join-Path $verifyBase ('approval-sync-r43-'+[guid]::NewGuid().ToString('N'))) $verifyBase 'APPROVAL_SYNC_STAGE_PATH_REJECTED'
$stageCreated=$false

foreach($path in @($source,$web,$previous,$sourceManifestFile,$previousManifestFile,(Join-Path $web '.next'),(Join-Path $web '.next\standalone\server.js'))){
  if(-not(Test-Path -LiteralPath $path)){throw "APPROVAL_SYNC_INPUT_MISSING:$path"}
}
if(Test-Path -LiteralPath $output){throw 'APPROVAL_SYNC_OUTPUT_EXISTS'}

$sourceManifest=Get-Content -LiteralPath $sourceManifestFile -Raw -Encoding UTF8|ConvertFrom-Json
$previousSource=Get-Content -LiteralPath $previousManifestFile -Raw -Encoding UTF8|ConvertFrom-Json
$previousOverlay=Get-Content -LiteralPath (Join-Path $previous 'operational-overlay-manifest.json') -Raw -Encoding UTF8|ConvertFrom-Json
if([string]$sourceManifest.schema-ne'evomind.super_agent_source_set.v1'-or-not[bool]$sourceManifest.frozen){throw 'APPROVAL_SYNC_SOURCE_MANIFEST_REJECTED'}
if([string]$previousOverlay.schema-ne'evomind.web_operational_overlay.v1'){throw 'APPROVAL_SYNC_PREVIOUS_OVERLAY_REJECTED'}

try{
  Copy-Item -LiteralPath $previous -Destination $stage -Recurse -Force
  $stageCreated=$true
  foreach($relative in @('.next','src','public','support\python-runtime\evomind_runtime','support\python-runtime\xsci')){
    $target=Assert-ChildPath (Join-Path $stage $relative) $stage 'APPROVAL_SYNC_REMOVE_PATH_REJECTED'
    if(Test-Path -LiteralPath $target){Remove-Item -LiteralPath $target -Recurse -Force}
  }
  $oldManifest=Assert-ChildPath (Join-Path $stage 'operational-overlay-manifest.json') $stage 'APPROVAL_SYNC_MANIFEST_PATH_REJECTED'
  if(Test-Path -LiteralPath $oldManifest){Remove-Item -LiteralPath $oldManifest -Force}

  Copy-Item -LiteralPath (Join-Path $web '.next') -Destination (Join-Path $stage '.next') -Recurse -Force
  Copy-Item -LiteralPath (Join-Path $web 'src') -Destination (Join-Path $stage 'src') -Recurse -Force
  Copy-Item -LiteralPath (Join-Path $web 'public') -Destination (Join-Path $stage 'public') -Recurse -Force
  Copy-Item -LiteralPath (Join-Path $source 'src\evomind_runtime') -Destination (Join-Path $stage 'support\python-runtime\evomind_runtime') -Recurse -Force
  Copy-Item -LiteralPath (Join-Path $source 'src\xsci') -Destination (Join-Path $stage 'support\python-runtime\xsci') -Recurse -Force
  Get-ChildItem -LiteralPath (Join-Path $stage 'support\python-runtime') -Recurse -Directory -Filter '__pycache__'|ForEach-Object{
    $cache=Assert-ChildPath $_.FullName $stage 'APPROVAL_SYNC_CACHE_PATH_REJECTED'
    Remove-Item -LiteralPath $cache -Recurse -Force
  }
  Get-ChildItem -LiteralPath (Join-Path $stage 'support\python-runtime') -Recurse -File|Where-Object{$_.Extension-in'.pyc','.pyo'}|ForEach-Object{Remove-Item -LiteralPath $_.FullName -Force}

  foreach($name in @('package.json','package-lock.json','next.config.mjs','tsconfig.json','postcss.config.mjs','tailwind.config.ts')){
    Copy-Item -LiteralPath (Join-Path $web $name) -Destination (Join-Path $stage $name) -Force
  }
  Copy-Item -LiteralPath $sourceManifestFile -Destination (Join-Path $stage 'release-source-manifest.json') -Force

  $runtimeManifest=[ordered]@{
    schema='evomind.runtime_build.v1'
    commit_hash=[string]$previousOverlay.base_commit
    source_dirty=$true
    source_tree_sha256=[string]$sourceManifest.source_tree_sha256
    build_id=$BuildId
    build_time=[DateTime]::UtcNow.ToString('o')
    backend_version='0.3.0'
    frontend_version='0.3.0'
    database_schema_version='20260728171000_performance_indexes'
    database_schema_sha256='ce878bdbf737662fa2332c5bef1495f68762d73b9f05217cf752cf4014149f1f'
  }
  Write-Utf8Json (Join-Path $stage 'runtime-build-manifest.json') $runtimeManifest

  $previousByPath=@{};foreach($entry in $previousSource.files){$previousByPath[[string]$entry.path]=$entry}
  $changed=@($sourceManifest.files|Where-Object{
    -not$previousByPath.ContainsKey([string]$_.path)-or[string]$previousByPath[[string]$_.path].sha256-ne[string]$_.sha256
  })
  $files=@(Get-ChildItem -LiteralPath $stage -Recurse -File -Force|Sort-Object FullName|ForEach-Object{
    [ordered]@{
      path=$_.FullName.Substring($stage.Length+1).Replace('\','/')
      bytes=[long]$_.Length
      sha256=Sha $_.FullName
    }
  })
  $overlay=[ordered]@{
    schema='evomind.web_operational_overlay.v1'
    overlay_id=$BuildId
    created_at_utc=[DateTime]::UtcNow.ToString('o')
    base_commit=[string]$previousOverlay.base_commit
    source_head=[string]$sourceManifest.git_head
    source_dirty=$true
    entrypoint='server.js'
    source_identity_sha256=[string]$sourceManifest.source_tree_sha256
    source_manifest_sha256=Sha $sourceManifestFile
    changed_source_files=$changed
    file_count=$files.Count
    files=$files
    production_deployed=$false
    hpc_accessed=$false
    gpu_touched=$false
  }
  Write-Utf8Json (Join-Path $stage 'operational-overlay-manifest.json') $overlay

  $outputParent=Split-Path -Parent $output
  if(-not(Test-Path -LiteralPath $outputParent -PathType Container)){New-Item -ItemType Directory -Path $outputParent -Force|Out-Null}
  Move-Item -LiteralPath $stage -Destination $output
  $stageCreated=$false
  [ordered]@{
    status='overlay_built'
    build_id=$BuildId
    output_root=$output
    source_tree_sha256=[string]$sourceManifest.source_tree_sha256
    source_manifest_sha256=Sha $sourceManifestFile
    changed_source_files=$changed.Count
    file_count=$files.Count
    hpc_accessed=$false
    gpu_touched=$false
  }|ConvertTo-Json -Compress
}finally{
  if($stageCreated-and(Test-Path -LiteralPath $stage)){
    $safe=Assert-ChildPath $stage $verifyBase 'APPROVAL_SYNC_CLEANUP_PATH_REJECTED'
    Remove-Item -LiteralPath $safe -Recurse -Force
  }
}
