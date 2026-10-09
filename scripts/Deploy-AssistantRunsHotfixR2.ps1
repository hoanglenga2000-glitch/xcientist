param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [Parameter(Mandatory)][string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$hostAlias = 'evomind-shanghai'
$remote = "C:\SecureInput\assistant-runs-hotfix-$TransactionId"
$build = 'overlay-activation-status-r52-0bc68a5a'
$candidateSha = '364156d16c56870ecddd987dc6dd77975d08c54795bbb86b4c655418bb13629c'
$runnerSha = '91b17f26de2680e0d4fa39995b7877f2d4cea301739b6e2e7af3ac5b353fbfc6'
$expectedOldSha = 'd9332641c2d359f23be7f77902d6515f60038a5f01bf797a89c2cb5ec24cf1cd'
$candidate = Join-Path $PSScriptRoot '..\src\evomind_runtime\assistant_runs.py'
$runner = Join-Path $PSScriptRoot 'Deploy-AssistantRunsHotfixR1Remote.ps1'
$output = [IO.Path]::GetFullPath($OutputPath)
$errorOutput = [IO.Path]::ChangeExtension($output, '.error.json')

foreach ($path in @($output, $errorOutput)) {
  if (Test-Path -LiteralPath $path) { throw "ASSISTANT_RUNS_R2_OUTPUT_EXISTS:$path" }
}
foreach ($item in @(
  [pscustomobject]@{ path = $candidate; sha = $candidateSha },
  [pscustomobject]@{ path = $runner; sha = $runnerSha }
)) {
  if (-not (Test-Path -LiteralPath $item.path -PathType Leaf)) {
    throw "ASSISTANT_RUNS_R2_LOCAL_INPUT_MISSING:$($item.path)"
  }
  if ((Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.sha) {
    throw "ASSISTANT_RUNS_R2_LOCAL_HASH_REJECTED:$($item.path)"
  }
}

$create = "if(Test-Path -LiteralPath '$remote'){throw 'ASSISTANT_RUNS_R2_REMOTE_EXISTS'};New-Item -ItemType Directory -Path '$remote'|Out-Null"
$create | ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if ($LASTEXITCODE -ne 0) { throw 'ASSISTANT_RUNS_R2_REMOTE_CREATE_FAILED' }

foreach ($item in @(
  [pscustomobject]@{ path = $candidate; name = 'assistant_runs.py' },
  [pscustomobject]@{ path = $runner; name = 'Deploy-AssistantRunsHotfixR1Remote.ps1' }
)) {
  scp $item.path "${hostAlias}:$remote\$($item.name)"
  if ($LASTEXITCODE -ne 0) { throw "ASSISTANT_RUNS_R2_UPLOAD_FAILED:$($item.name)" }
}

$verify = "`$files=@([pscustomobject]@{name='assistant_runs.py';sha='$candidateSha'},[pscustomobject]@{name='Deploy-AssistantRunsHotfixR1Remote.ps1';sha='$runnerSha'});foreach(`$i in `$files){`$p=Join-Path '$remote' `$i.name;if((Get-FileHash -LiteralPath `$p -Algorithm SHA256).Hash.ToLowerInvariant()-ne`$i.sha){throw 'ASSISTANT_RUNS_R2_REMOTE_HASH_REJECTED'}};`$tokens=`$null;`$errors=`$null;[Management.Automation.Language.Parser]::ParseFile((Join-Path '$remote' 'Deploy-AssistantRunsHotfixR1Remote.ps1'),[ref]`$tokens,[ref]`$errors)|Out-Null;if(@(`$errors).Count){throw 'ASSISTANT_RUNS_R2_REMOTE_AST_REJECTED'};[ordered]@{status='staged_verified';transaction_id='$TransactionId'}|ConvertTo-Json -Compress"
$staged = $verify | ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if ($LASTEXITCODE -ne 0) { throw 'ASSISTANT_RUNS_R2_REMOTE_VERIFY_FAILED' }
$staged

ssh $hostAlias "powershell.exe -NoProfile -NonInteractive -File $remote\Deploy-AssistantRunsHotfixR1Remote.ps1 -TransactionId $TransactionId -RemoteRoot $remote -CandidateSha256 $candidateSha -ExpectedOldSha256 $expectedOldSha -ExpectedBuildId $build -ExpectedJobId 92257 -ExpectedGeneration 21 -ExpectedTenantId tenant_ba0ef9d3767f2fb385b856e0"
if ($LASTEXITCODE -ne 0) {
  scp "${hostAlias}:$remote\transaction-$TransactionId-error.json" $errorOutput 2>$null
  throw 'ASSISTANT_RUNS_R2_TRANSACTION_FAILED'
}

scp "${hostAlias}:$remote\transaction-$TransactionId-result.json" $output
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $output -PathType Leaf)) {
  throw 'ASSISTANT_RUNS_R2_RESULT_DOWNLOAD_FAILED'
}
$result = Get-Content -LiteralPath $output -Raw -Encoding UTF8 | ConvertFrom-Json
if (
  [string]$result.status -ne 'installed_and_verified' -or
  [string]$result.transaction_id -ne $TransactionId -or
  [string]$result.build_id -ne $build -or
  [string]$result.old_sha256 -ne $expectedOldSha -or
  [string]$result.assistant_runs_sha256 -ne $candidateSha -or
  [int]$result.job_id -ne 92257 -or
  [int]$result.allocation_generation -ne 21 -or
  [string]$result.binding_state -ne 'active' -or
  -not [bool]$result.hpc_binding_unchanged -or
  -not [bool]$result.profile_unchanged -or
  -not [bool]$result.byoa_tree_unchanged -or
  [bool]$result.hpc_accessed -or
  [bool]$result.gpu_touched -or
  [bool]$result.training_started -or
  [int]$result.kaggle_submissions -ne 0 -or
  [bool]$result.secrets_printed
) {
  throw 'ASSISTANT_RUNS_R2_RESULT_REJECTED'
}

[ordered]@{
  status = 'installed_and_verified'
  transaction_id = $TransactionId
  build_id = $build
  result_path = $output
  result_sha256 = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()
} | ConvertTo-Json -Compress
