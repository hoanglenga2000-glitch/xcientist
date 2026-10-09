param(
  [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{32}$')][string]$TransactionId,
  [Parameter(Mandatory)][string]$OutputPath
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$hostAlias = 'evomind-shanghai'
$remoteRoot = "C:\SecureInput\capability-web-$TransactionId"
$oldBuild = 'overlay-weather4cast-secret-resume-r67-0896a68d234b'
$newBuild = 'overlay-weather4cast-secret-resume-r68-588023e13b18'
$webPath = 'D:/AI-Outputs/EvoMind-Cloud-Deploy/artifacts/web-overlay-20260826-weather4cast-secret-resume-r68.zip'
$adminPath = 'D:/AI-Outputs/EvoMind-Cloud-Deploy/artifacts/evomind-admin-password.scrypt'
$runnerPath = 'D:/AI-Outputs/EvoMind-Cloud-Deploy/artifacts/Deploy-CapabilityWebR2Remote.ps1'
$webSha = 'b9943af226429b9cc3b0feae02c403aeee7e375923ec195715cade2745d8bafc'
$adminSha = '18bb9c74b5b0759f71f6fbd2cbb5f3f4b746e92396519369099673867f59ef0d'
$runnerSha = 'a7d4c00faafd09a41af1286bf9471a66ea4dc0a79f9b474e659272d9cda7337c'
$output = [IO.Path]::GetFullPath($OutputPath)
$errorOutput = [IO.Path]::ChangeExtension($output, '.error.json')
function Sha([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
foreach ($path in @($output, $errorOutput)) {
  if (Test-Path -LiteralPath $path) { throw "R68_OUTPUT_EXISTS:$path" }
}
foreach ($item in @(
  [pscustomobject]@{ Path = $webPath; Sha = $webSha },
  [pscustomobject]@{ Path = $adminPath; Sha = $adminSha },
  [pscustomobject]@{ Path = $runnerPath; Sha = $runnerSha }
)) {
  if (-not (Test-Path -LiteralPath $item.Path -PathType Leaf)) { throw "R68_LOCAL_INPUT_MISSING:$($item.Path)" }
  if ((Sha $item.Path) -ne $item.Sha) { throw "R68_LOCAL_HASH_REJECTED:$($item.Path)" }
}
$create = "if(Test-Path -LiteralPath '$remoteRoot'){throw 'R68_REMOTE_EXISTS'};New-Item -ItemType Directory -Path '$remoteRoot'|Out-Null"
$create | ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if ($LASTEXITCODE -ne 0) { throw 'R68_REMOTE_CREATE_FAILED' }
foreach ($item in @(
  [pscustomobject]@{ Path = $webPath; Name = 'web.zip' },
  [pscustomobject]@{ Path = $adminPath; Name = 'admin.scrypt' },
  [pscustomobject]@{ Path = $runnerPath; Name = 'Deploy-CapabilityWebR2Remote.ps1' }
)) {
  scp $item.Path "$($hostAlias):$remoteRoot\$($item.Name)"
  if ($LASTEXITCODE -ne 0) { throw "R68_UPLOAD_FAILED:$($item.Name)" }
}
ssh $hostAlias "powershell.exe -NoProfile -NonInteractive -File $remoteRoot\Deploy-CapabilityWebR2Remote.ps1 -TransactionId $TransactionId -RemoteRoot $remoteRoot -WebSha256 $webSha -OldBuildId $oldBuild -NewBuildId $newBuild"
if ($LASTEXITCODE -ne 0) {
  scp "$($hostAlias):$remoteRoot\transaction-$TransactionId-error.json" $errorOutput 2>$null
  throw 'R68_TRANSACTION_FAILED'
}
scp "$($hostAlias):$remoteRoot\transaction-$TransactionId-result.json" $output
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $output -PathType Leaf)) { throw 'R68_RESULT_DOWNLOAD_FAILED' }
$result = Get-Content -LiteralPath $output -Raw -Encoding UTF8 | ConvertFrom-Json
if ([string]$result.status -ne 'installed_and_verified' -or [string]$result.transaction_id -ne $TransactionId -or [string]$result.old_build_id -ne $oldBuild -or [string]$result.new_build_id -ne $newBuild -or [string]$result.web_sha256 -ne $webSha -or @($result.loopback.samples).Count -ne 3 -or @($result.loopback.samples | Where-Object { -not [bool]$_.exact }).Count -ne 0 -or @($result.public.samples).Count -ne 3 -or @($result.public.samples | Where-Object { -not [bool]$_.exact }).Count -ne 0 -or -not [bool]$result.hpc_binding_unchanged -or -not [bool]$result.byoa_stop_window_unchanged -or -not [bool]$result.profile_unchanged -or [bool]$result.hpc_accessed -or [bool]$result.gpu_touched -or [bool]$result.secrets_printed) { throw 'R68_RESULT_REJECTED' }
[ordered]@{ status = 'installed_and_verified'; transaction_id = $TransactionId; build_id = $newBuild; result_path = $output; result_sha256 = Sha $output; loopback_samples = 3; public_samples = 3; hpc_accessed = $false; gpu_touched = $false } | ConvertTo-Json -Compress

