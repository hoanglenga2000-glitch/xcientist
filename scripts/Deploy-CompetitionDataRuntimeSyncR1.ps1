param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [Parameter(Mandatory)][string]$WebPath,
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$WebSha256,
  [Parameter(Mandatory)][string]$ExpectedBuildId,
  [Parameter(Mandatory)][string]$OutputPath,
  [string]$AdminHashPath = 'D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\evomind-admin-password.scrypt'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$hostAlias = 'evomind-shanghai'
$remote = "C:\SecureInput\runtime-sync-$TransactionId"
$runner = Join-Path $PSScriptRoot 'Deploy-RuntimeSyncR1Remote.ps1'
$output = [IO.Path]::GetFullPath($OutputPath)
$errorOutput = [IO.Path]::ChangeExtension($output, '.error.json')

function Sha([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }

foreach ($path in @($output, $errorOutput)) {
  if (Test-Path -LiteralPath $path) { throw "COMPETITION_RUNTIME_SYNC_OUTPUT_EXISTS:$path" }
}
foreach ($item in @(
    [pscustomobject]@{ path = [IO.Path]::GetFullPath($WebPath); sha = $WebSha256; name = 'web.zip' },
    [pscustomobject]@{ path = [IO.Path]::GetFullPath($AdminHashPath); sha = Sha ([IO.Path]::GetFullPath($AdminHashPath)); name = 'admin.scrypt' },
    [pscustomobject]@{ path = [IO.Path]::GetFullPath($runner); sha = Sha $runner; name = 'Deploy-RuntimeSyncR1Remote.ps1' }
  )) {
  if (-not (Test-Path -LiteralPath $item.path -PathType Leaf)) { throw "COMPETITION_RUNTIME_SYNC_LOCAL_INPUT_MISSING:$($item.path)" }
  if ((Sha $item.path) -ne $item.sha) { throw "COMPETITION_RUNTIME_SYNC_LOCAL_HASH_REJECTED:$($item.name)" }
}
$adminHash = (Get-Content -LiteralPath ([IO.Path]::GetFullPath($AdminHashPath)) -Raw).Trim()
if ($adminHash -notmatch '^scrypt\$16384\$8\$1\$[a-f0-9]{32}\$[a-f0-9]{64}$') { throw 'COMPETITION_RUNTIME_SYNC_ADMIN_HASH_INVALID' }
$adminHash = $null

$create = "if(Test-Path -LiteralPath '$remote'){throw 'COMPETITION_RUNTIME_SYNC_REMOTE_EXISTS'};New-Item -ItemType Directory -Path '$remote'|Out-Null"
$create | ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if ($LASTEXITCODE -ne 0) { throw 'COMPETITION_RUNTIME_SYNC_REMOTE_CREATE_FAILED' }
foreach ($item in @(
    [pscustomobject]@{ path = [IO.Path]::GetFullPath($WebPath); name = 'web.zip' },
    [pscustomobject]@{ path = [IO.Path]::GetFullPath($AdminHashPath); name = 'admin.scrypt' },
    [pscustomobject]@{ path = [IO.Path]::GetFullPath($runner); name = 'Deploy-RuntimeSyncR1Remote.ps1' }
  )) {
  scp $item.path "${hostAlias}:$remote\$($item.name)"
  if ($LASTEXITCODE -ne 0) { throw "COMPETITION_RUNTIME_SYNC_UPLOAD_FAILED:$($item.name)" }
}

$runnerSha = Sha $runner
$adminSha = Sha ([IO.Path]::GetFullPath($AdminHashPath))
$verify = "`$files=@([pscustomobject]@{name='web.zip';sha='$WebSha256'},[pscustomobject]@{name='admin.scrypt';sha='$adminSha'},[pscustomobject]@{name='Deploy-RuntimeSyncR1Remote.ps1';sha='$runnerSha'});foreach(`$i in `$files){`$p=Join-Path '$remote' `$i.name;if((Get-FileHash -LiteralPath `$p -Algorithm SHA256).Hash.ToLowerInvariant()-ne`$i.sha){throw 'COMPETITION_RUNTIME_SYNC_REMOTE_HASH_REJECTED'}};`$tokens=`$null;`$errors=`$null;[Management.Automation.Language.Parser]::ParseFile((Join-Path '$remote' 'Deploy-RuntimeSyncR1Remote.ps1'),[ref]`$tokens,[ref]`$errors)|Out-Null;if(@(`$errors).Count){throw 'COMPETITION_RUNTIME_SYNC_REMOTE_AST_REJECTED'};[ordered]@{status='staged_verified';transaction_id='$TransactionId'}|ConvertTo-Json -Compress"
$staged = $verify | ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if ($LASTEXITCODE -ne 0) { throw 'COMPETITION_RUNTIME_SYNC_REMOTE_VERIFY_FAILED' }
$staged

$preflight = ssh $hostAlias "powershell.exe -NoProfile -NonInteractive -File $remote\Deploy-RuntimeSyncR1Remote.ps1 -TransactionId $TransactionId -RemoteRoot $remote -WebSha256 $WebSha256 -ExpectedBuildId $ExpectedBuildId -PreflightOnly"
if ($LASTEXITCODE -ne 0) {
  scp "${hostAlias}:$remote\transaction-$TransactionId-error.json" $errorOutput 2>$null
  throw 'COMPETITION_RUNTIME_SYNC_PREFLIGHT_FAILED'
}
$preflight

ssh $hostAlias "powershell.exe -NoProfile -NonInteractive -File $remote\Deploy-RuntimeSyncR1Remote.ps1 -TransactionId $TransactionId -RemoteRoot $remote -WebSha256 $WebSha256 -ExpectedBuildId $ExpectedBuildId"
if ($LASTEXITCODE -ne 0) {
  scp "${hostAlias}:$remote\transaction-$TransactionId-error.json" $errorOutput 2>$null
  throw 'COMPETITION_RUNTIME_SYNC_TRANSACTION_FAILED'
}
scp "${hostAlias}:$remote\transaction-$TransactionId-result.json" $output
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $output -PathType Leaf)) { throw 'COMPETITION_RUNTIME_SYNC_RESULT_DOWNLOAD_FAILED' }

$result = Get-Content -LiteralPath $output -Raw -Encoding UTF8 | ConvertFrom-Json
if ([string]$result.status -ne 'installed_and_verified' -or
    [string]$result.transaction_id -ne $TransactionId -or
    [string]$result.build_id -ne $ExpectedBuildId -or
    [string]$result.web_sha256 -ne $WebSha256 -or
    -not [bool]$result.hpc_binding_unchanged -or
    -not [bool]$result.byoa_stop_window_unchanged -or
    -not [bool]$result.profile_unchanged -or
    [bool]$result.hpc_accessed -or
    [bool]$result.gpu_touched -or
    [bool]$result.secrets_printed) {
  throw 'COMPETITION_RUNTIME_SYNC_RESULT_REJECTED'
}

[ordered]@{
  status = 'installed_and_verified'
  transaction_id = $TransactionId
  build_id = $ExpectedBuildId
  web_sha256 = $WebSha256
  result_path = $output
  result_sha256 = Sha $output
  hpc_accessed = $false
  gpu_touched = $false
  secrets_printed = $false
} | ConvertTo-Json -Compress
