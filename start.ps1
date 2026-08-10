param([int]$Port = 8088, [string]$HostName = "127.0.0.1", [string]$DataDir = "", [switch]$Build, [switch]$Smoke)
if (-not $PSBoundParameters.ContainsKey("Port") -and -not [string]::IsNullOrWhiteSpace($env:WORKSTATION_PORT)) {
  $resolvedPort = 0
  if (-not [int]::TryParse($env:WORKSTATION_PORT, [ref]$resolvedPort) -or $resolvedPort -lt 1 -or $resolvedPort -gt 65535) {
    throw "WORKSTATION_PORT must be an integer between 1 and 65535."
  }
  $Port = $resolvedPort
}
if ($Port -lt 1 -or $Port -gt 65535) { throw "Port must be between 1 and 65535." }
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPYCACHEPREFIX = $null
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot "app\server.js") -PathType Leaf) {
  $localBase = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData) }
  $roamingBase = if ($env:APPDATA) { $env:APPDATA } else { [Environment]::GetFolderPath([Environment+SpecialFolder]::ApplicationData) }
  $managedRoot = Join-Path $localBase "EvoMind"
  $env:WORKSTATION_DATA_DIR = Join-Path $managedRoot "data"
  if (-not $env:WORKSTATION_LOGS_DIR) { $env:WORKSTATION_LOGS_DIR = Join-Path $managedRoot "logs" }
  if (-not $env:WORKSTATION_BACKUPS_DIR) { $env:WORKSTATION_BACKUPS_DIR = Join-Path $managedRoot "backups" }
  if (-not $env:EVOMIND_PROFILES_DIR) { $env:EVOMIND_PROFILES_DIR = Join-Path $roamingBase "EvoMind\profiles" }
  if (-not $env:EVOMIND_SECRETS_DIR) { $env:EVOMIND_SECRETS_DIR = Join-Path $roamingBase "EvoMind\secrets" }
}
$argsList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "scripts\start_verified_workstation.ps1"), "-Command", $(if ($Smoke) { "smoke" } else { "start" }), "-Port", $Port, "-HostName", $HostName)
if ($Build) { $argsList += "-Build" }
& powershell @argsList
exit $LASTEXITCODE
