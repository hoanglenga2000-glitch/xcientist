param([switch]$NoRestart)
$ErrorActionPreference = "Stop"
$command = Get-Command evomind -ErrorAction SilentlyContinue
if (-not $command) {
  throw "SIGNED_BOOTSTRAP_REQUIRED: @evomind-ai/cli is required for rollback."
}
$arguments = @("rollback", "--json")
if ($NoRestart) { $arguments += "--no-restart" }
& $command.Source @arguments
exit $LASTEXITCODE
