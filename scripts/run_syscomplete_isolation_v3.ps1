$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$stage = 'C:/ProgramData/EvoMind/staging/invitation-7925595228da-sys1'
$package = Join-Path $stage 'acceptance.zip'
if ((Get-FileHash -LiteralPath $package -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'b349c1d6f4ba6fc3f7f82b339ba5812a2dd27760712be52bbe85f75a79bc7746') { throw 'acceptance_package_hash_mismatch' }
if (Test-Path -LiteralPath (Join-Path $stage 'acceptance-manifest.json')) { throw 'isolation_requires_reconciliation_not_restart' }
Expand-Archive -LiteralPath $package -DestinationPath $stage
$manifest = Get-Content -LiteralPath (Join-Path $stage 'acceptance-manifest.json') -Raw | ConvertFrom-Json
foreach ($file in $manifest.files) {
  $path = Join-Path $stage $file.path
  if ((Get-Item -LiteralPath $path).Length -ne $file.bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) { throw 'acceptance_file_hash_mismatch' }
}
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$commands = Get-Content -LiteralPath (Join-Path $stage 'server-isolation-commands.json') -Raw | ConvertFrom-Json
foreach ($command in $commands.commands) {
  $program = [string]$command[0]
  $arguments = @($command | Select-Object -Skip 1)
  & $program @arguments
  if ($LASTEXITCODE -ne 0) { throw 'server_isolation_command_failed' }
}
