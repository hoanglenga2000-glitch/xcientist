param(
  [Parameter(Mandatory)][string]$OutputPath,
  [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$schema = 'evomind.g21_terminal_preserving_asset_probe_execution.r120.v1'
$fixedRun = 'run_7b1efb878afb40f396db431e91f093a5'
$fixedAllocation = 'G21'
$hostAlias = 'evomind-shanghai'
$runnerName = 'Invoke-G21TerminalPreservingAssetProbeR120Remote.ps1'
$runner = Join-Path $PSScriptRoot $runnerName
$sshOptions = @('-o','ConnectionAttempts=3','-o','ConnectTimeout=30','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3')

function Sha([string]$Path) {
  $stream = [IO.File]::OpenRead($Path)
  $hash = [Security.Cryptography.SHA256]::Create()
  try { ([BitConverter]::ToString($hash.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
  finally { $stream.Dispose(); $hash.Dispose() }
}

function Write-AtomicJson([string]$Path, [object]$Value) {
  $temporary = "$Path.$PID.tmp"
  [IO.File]::WriteAllText($temporary, (($Value | ConvertTo-Json -Depth 40) + "`n"), [Text.UTF8Encoding]::new($false))
  Move-Item -LiteralPath $temporary -Destination $Path -Force
}

if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) { throw 'R120_REMOTE_RUNNER_MISSING' }
$runnerItem = Get-Item -LiteralPath $runner -Force
if ($runnerItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'R120_REMOTE_RUNNER_REPARSE_REJECTED' }
$source = Get-Content -LiteralPath $runner -Raw -Encoding UTF8
if ($source -match '(?i)/v1/runs/.+/actions|action\s*[=:]\s*["'']?cancel|/v1/sessions/.+/cancel') { throw 'R120_RUNNER_CANCEL_SURFACE_REJECTED' }
if (($source | Select-String -Pattern 'Post-Api \"/v1/sessions/\$fixedRun/tools\"' -AllMatches).Matches.Count -ne 1) { throw 'R120_DIRECT_TOOL_CALLSITE_COUNT_REJECTED' }

$output = [IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $output) { throw 'R120_OUTPUT_EXISTS' }
$parent = [IO.Path]::GetDirectoryName($output)
if (-not (Test-Path -LiteralPath $parent -PathType Container)) { throw 'R120_OUTPUT_PARENT_MISSING' }

if ($ValidateOnly) {
  [ordered]@{
    schema=$schema
    status='validated_not_executed'
    run_id=$fixedRun
    allocation=$fixedAllocation
    host_alias=$hostAlias
    remote_runner_sha256=Sha $runner
    preserve_terminal_required=$true
    direct_tool_calls=0
    run_action_calls=0
    cancel_calls=0
    production_accessed=$false
    transport_mode='single_scp_runner'
    production_staging_writes=0
    production_staging_cleanup_policy='retain_for_audit_no_automatic_delete'
    hpc_accessed=$false
    gpu_touched=$false
  } | ConvertTo-Json -Compress
  exit 0
}

$transactionId = [guid]::NewGuid().ToString('N')
$remote = "C:\SecureInput\r120-probe-$transactionId"
$runnerSha = Sha $runner
$create = "if(Test-Path -LiteralPath '$remote'){throw 'R120_PROBE_REMOTE_EXISTS'};New-Item -ItemType Directory -Path '$remote'|Out-Null"
$create | & ssh @sshOptions $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if ($LASTEXITCODE -ne 0) { throw 'R120_PROBE_REMOTE_CREATE_FAILED' }
& scp @sshOptions $runner "${hostAlias}:$remote\$runnerName"
if ($LASTEXITCODE -ne 0) { throw 'R120_PROBE_RUNNER_UPLOAD_FAILED' }
$verify = "`$p=Join-Path '$remote' '$runnerName';`$s=[IO.File]::OpenRead(`$p);`$h=[Security.Cryptography.SHA256]::Create();try{`$v=([BitConverter]::ToString(`$h.ComputeHash(`$s))).Replace('-','').ToLowerInvariant()}finally{`$s.Dispose();`$h.Dispose()};if(`$v-cne'$runnerSha'){throw 'R120_PROBE_RUNNER_REMOTE_HASH_REJECTED'}"
$verify | & ssh @sshOptions $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if ($LASTEXITCODE -ne 0) { throw 'R120_PROBE_RUNNER_REMOTE_VERIFY_FAILED' }
& ssh @sshOptions $hostAlias "powershell.exe -NoProfile -NonInteractive -File $remote\$runnerName -TransactionId $transactionId -RemoteRoot $remote -RunnerSha256 $runnerSha"
$remoteExit = $LASTEXITCODE
if ($remoteExit -ne 0) {
  $errorPath = [IO.Path]::ChangeExtension($output, '.error.json')
  & scp @sshOptions "${hostAlias}:$remote\probe-$transactionId-error.json" $errorPath 2>$null
  throw 'R120_REMOTE_PROBE_FAILED_CLOSED'
}
& scp @sshOptions "${hostAlias}:$remote\probe-$transactionId-result.json" $output
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $output -PathType Leaf)) { throw 'R120_PROBE_RESULT_DOWNLOAD_FAILED' }
try { $result = Get-Content -LiteralPath $output -Raw -Encoding UTF8 | ConvertFrom-Json }
catch { throw 'R120_REMOTE_OUTPUT_INVALID_JSON' }
if (
  [string]$result.schema -cne $schema -or [string]$result.status -cne 'completed' -or
  [string]$result.run_id -cne $fixedRun -or [string]$result.allocation -cne $fixedAllocation -or
  $result.preserve_terminal -isnot [bool] -or -not [bool]$result.preserve_terminal -or
  [string]$result.run_before.status -cne 'cancelled' -or -not [bool]$result.run_before.terminal -or
  [string]$result.run_after.status -cne 'cancelled' -or -not [bool]$result.run_after.terminal -or
  [int]$result.run_after.effective_active_count -ne 0 -or [int]$result.run_after.pending_approval_count -ne 0 -or
  [int]$result.direct_tool_calls -ne 1 -or [int]$result.run_action_calls -ne 0 -or [int]$result.cancel_calls -ne 0 -or [int]$result.retries -ne 0 -or
  [int]$result.hpc_remote_writes -ne 0 -or [int]$result.signals_sent -ne 0 -or [bool]$result.other_processes_modified -or [bool]$result.training_started -or [int]$result.worker_control -ne 0 -or [bool]$result.secrets_printed
) { throw 'R120_REMOTE_RESULT_REJECTED' }
[ordered]@{
  schema=$schema
  status='completed'
  result_path=$output
  result_sha256=Sha $output
  run_id=$fixedRun
  allocation=$fixedAllocation
  preserve_terminal=$true
  probe_status=[string]$result.outcome.status
  direct_tool_calls=1
  cancel_calls=0
  transport_mode='single_scp_runner'
  production_staging_writes=3
  production_staging_cleanup_policy='retain_for_audit_no_automatic_delete'
} | ConvertTo-Json -Compress
