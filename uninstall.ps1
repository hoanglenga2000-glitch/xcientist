param([switch]$PurgeUserData, [switch]$Confirm)
$ErrorActionPreference = "Stop"
if (-not $Confirm) {
  throw "Uninstall requires -Confirm. User data is preserved unless -PurgeUserData is set."
}
$command = Get-Command evomind -ErrorAction SilentlyContinue
if (-not $command) {
  throw "SIGNED_BOOTSTRAP_REQUIRED: @evomind-ai/cli is required for uninstall and purge."
}
$arguments = @("uninstall", "--json")
if ($PurgeUserData) { $arguments += "--purge-data" }
& $command.Source @arguments
exit $LASTEXITCODE
