param([int]$Port = 8088, [string]$HostName = "127.0.0.1")
if (-not $PSBoundParameters.ContainsKey("Port") -and -not [string]::IsNullOrWhiteSpace($env:WORKSTATION_PORT)) {
  $resolvedPort = 0
  if (-not [int]::TryParse($env:WORKSTATION_PORT, [ref]$resolvedPort) -or $resolvedPort -lt 1 -or $resolvedPort -gt 65535) {
    throw "WORKSTATION_PORT must be an integer between 1 and 65535."
  }
  $Port = $resolvedPort
}
if ($Port -lt 1 -or $Port -gt 65535) { throw "Port must be between 1 and 65535." }
& (Join-Path $PSScriptRoot "scripts\manage_workstation_lifecycle.ps1") status -InstallRoot $PSScriptRoot -Port $Port -HostName $HostName
exit $LASTEXITCODE
