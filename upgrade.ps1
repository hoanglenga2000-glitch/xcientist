param(
  [Parameter(Mandatory=$true)][string]$ManifestPath,
  [string]$Version = ""
)
$ErrorActionPreference = "Stop"
$command = Get-Command evomind -ErrorAction SilentlyContinue
if (-not $command) {
  throw "SIGNED_BOOTSTRAP_REQUIRED: @evomind-ai/cli is required for online and offline upgrades."
}
$manifest = if ($ManifestPath -match '^https://') { $ManifestPath } elseif ($ManifestPath -match '^file:') { $ManifestPath } else { [IO.Path]::GetFullPath($ManifestPath) }
$arguments = @("upgrade", "--manifest", $manifest, "--json")
if ($Version) { $arguments += @("--version", $Version) }
# The npm bootstrapper authenticates the package-pinned Ed25519 key_id,
# min_bootstrap_version, anti-rollback state, archive byte length and SHA-256
# before extraction or before any EvoMind process is stopped.
& $command.Source @arguments
exit $LASTEXITCODE
