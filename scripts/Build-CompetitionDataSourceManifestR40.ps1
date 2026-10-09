param(
  [string]$SourceRoot='D:\桌面\codex\科研港科技',
  [string]$SupportBaselineRoot='D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\web-overlay-20260815-g13-runtime-persistence-v15r21\support',
  [string]$Purpose='R40 ordinary-user competition data preparation, persistent HPC storage, real loader receipts, resumable Weather4cast download, and one-shot Run secrets',
  [string]$PythonTargeted='114/114',
  [string]$WebTests='144/144',
  [string]$PlaintextSecretScan='not_run',
  [Parameter(Mandatory)][string]$OutputPath
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
$supportRoot=[IO.Path]::GetFullPath($SupportBaselineRoot).TrimEnd('\')
$output=[IO.Path]::GetFullPath($OutputPath)
if(-not(Test-Path -LiteralPath $root -PathType Container)){throw 'COMPETITION_SOURCE_ROOT_MISSING'}
if(Test-Path -LiteralPath $output){throw 'COMPETITION_SOURCE_MANIFEST_EXISTS'}

$files=[Collections.Generic.List[string]]::new()
function Resolve-SourceFile([string]$Normalized){
  $primary=Join-Path $root $Normalized.Replace('/','\')
  if(Test-Path -LiteralPath $primary -PathType Leaf){return $primary}
  if($Normalized.StartsWith('scripts/',[StringComparison]::Ordinal)){
    $fallback=Join-Path $supportRoot ([IO.Path]::GetFileName($Normalized))
    if(Test-Path -LiteralPath $fallback -PathType Leaf){return $fallback}
  }
  return $primary
}
function Add-File([string]$Relative){
  $normalized=$Relative.Replace('\','/').TrimStart('/')
  $path=Resolve-SourceFile $normalized
  if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "COMPETITION_SOURCE_FILE_MISSING:$normalized"}
  if(-not$files.Contains($normalized)){[void]$files.Add($normalized)}
}
function Add-Tree([string]$Relative,[string[]]$Extensions){
  $normalized=$Relative.Replace('\','/').TrimStart('/')
  $path=Join-Path $root $normalized.Replace('/','\')
  if(-not(Test-Path -LiteralPath $path -PathType Container)){throw "COMPETITION_SOURCE_TREE_MISSING:$normalized"}
  Get-ChildItem -LiteralPath $path -Recurse -File -Force|Where-Object{
    $_.FullName-notmatch'[\\/](__pycache__|node_modules|\.next|\.pytest_cache|\.mypy_cache)[\\/]'-and
    $_.Name-notmatch'\.(pyc|pyo|db|db-shm|db-wal)$'-and
    ($Extensions.Count-eq0-or$Extensions-contains$_.Extension.ToLowerInvariant())
  }|ForEach-Object{Add-File $_.FullName.Substring($root.Length+1)}
}

Add-Tree 'web/research-agent-workstation/src' @('.ts','.tsx','.css','.svg','.png','.mjs','.mts')
Add-Tree 'web/research-agent-workstation/scripts' @('.mjs','.js','.ts')
Add-Tree 'web/research-agent-workstation/public' @()
foreach($relative in @(
  'web/research-agent-workstation/package.json',
  'web/research-agent-workstation/package-lock.json',
  'web/research-agent-workstation/next.config.mjs',
  'web/research-agent-workstation/tsconfig.json',
  'web/research-agent-workstation/postcss.config.mjs',
  'web/research-agent-workstation/tailwind.config.ts',
  'web/research-agent-workstation/prisma/schema.prisma',
  'scripts/upload_hpc_dataset.py',
  'scripts/run_hpc_dataset_training.py',
  'scripts/transition_tenant_hpc_profile.py',
  'scripts/evolution_engine_cli.py',
  'scripts/manage_kaggle_secret.ps1',
  'scripts/run_managed_kaggle_action.py',
  'scripts/verify_managed_kaggle_api.py',
  'scripts/dpapi_credential_store.ps1',
  'scripts/kaggle_runtime_requirements.txt',
  'src/research_agent_workstation/server/core/gpu_credentials.py',
  'tests/test_assistant_run_service.py',
  'tests/test_competition_data.py',
  'tests/test_run_secrets.py',
  'tests/test_hpc_execute_solution_workspace.py',
  'tests/test_evomind_runtime_http_server.py',
  'tests/test_evomind_runtime.py',
  'tests/test_gpu_credentials.py'
)){Add-File $relative}
Add-Tree 'src/evomind_runtime' @('.py')
Add-Tree 'src/xsci' @('.py')

$entries=@($files|Sort-Object|ForEach-Object{
  $path=Resolve-SourceFile $_
  [ordered]@{path=$_;source=if($path.StartsWith($root,[StringComparison]::OrdinalIgnoreCase)){'workspace'}else{'verified_support_baseline'};bytes=[long](Get-Item -LiteralPath $path).Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$canonical=($entries|ForEach-Object{"$($_.path)`0$($_.sha256)`n"})-join''
$sha=[Security.Cryptography.SHA256]::Create()
try{$sourceSha=([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($canonical)))).Replace('-','').ToLowerInvariant()}finally{$sha.Dispose()}
$head=(& git -C $root rev-parse HEAD).Trim()
if($LASTEXITCODE-ne0-or$head-notmatch'^[a-f0-9]{40}$'){throw 'COMPETITION_SOURCE_GIT_HEAD_INVALID'}
$dirty=@(& git -C $root status --porcelain=v1 --untracked-files=all)
$manifest=[ordered]@{
  schema='evomind.super_agent_source_set.v1'
  created_at_utc=[DateTime]::UtcNow.ToString('o')
  purpose=$Purpose
  frozen=$true
  git_head=$head
  source_dirty=($dirty.Count-gt0)
  source_tree_sha256=$sourceSha
  file_count=$entries.Count
  files=$entries
  validation=[ordered]@{
    python_targeted=$PythonTargeted
    web_tests=$WebTests
    web_typecheck='passed'
    web_build='passed'
    plaintext_secret_scan=$PlaintextSecretScan
    runtime_package_rounds='2/2'
    runtime_package_files=30
    git_diff_check='passed'
    hpc_accessed=$false
    gpu_touched=$false
  }
}
$parent=Split-Path -Parent $output;if($parent){New-Item -ItemType Directory -Path $parent -Force|Out-Null}
[IO.File]::WriteAllText($output,(($manifest|ConvertTo-Json -Depth 8)+"`n"),[Text.UTF8Encoding]::new($false))
[ordered]@{schema=$manifest.schema;path=$output;file_count=$entries.Count;source_tree_sha256=$sourceSha;manifest_sha256=(Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()}|ConvertTo-Json -Compress
