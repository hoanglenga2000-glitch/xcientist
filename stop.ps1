param([int]$Port = 8088, [string]$HostName = "127.0.0.1", [string]$DataDir = "")
if (-not $PSBoundParameters.ContainsKey("Port") -and -not [string]::IsNullOrWhiteSpace($env:WORKSTATION_PORT)) {
  $resolvedPort = 0
  if (-not [int]::TryParse($env:WORKSTATION_PORT, [ref]$resolvedPort) -or $resolvedPort -lt 1 -or $resolvedPort -gt 65535) {
    throw "WORKSTATION_PORT must be an integer between 1 and 65535."
  }
  $Port = $resolvedPort
}
if ($Port -lt 1 -or $Port -gt 65535) { throw "Port must be between 1 and 65535." }
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot "app\server.js") -PathType Leaf) {
  $localBase = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData) }
  $managedRoot = Join-Path $localBase "EvoMind"
  $env:WORKSTATION_DATA_DIR = Join-Path $managedRoot "data"
  if (-not $env:WORKSTATION_LOGS_DIR) { $env:WORKSTATION_LOGS_DIR = Join-Path $managedRoot "logs" }
}
$python = $null
$candidates = @()
if ($env:WORKSTATION_PYTHON) { $candidates += $env:WORKSTATION_PYTHON }
$candidates += @(
  (Join-Path $PSScriptRoot "runtime\python\python.exe"),
  (Join-Path $PSScriptRoot "runtime\python.exe"),
  (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
  "python.exe",
  "python"
)
foreach ($candidate in $candidates) {
  if (Test-Path -LiteralPath $candidate -PathType Leaf) { $python = (Resolve-Path -LiteralPath $candidate).Path; break }
  $command = Get-Command $candidate -ErrorAction SilentlyContinue
  if ($command -and $command.Source) { $python = $command.Source; break }
}
if (-not $python) { throw "WORKSTATION_STOP_FAILED: Python was not found." }
& $python (Join-Path $PSScriptRoot "scripts\manage_workstation_dashboard.py") stop --host $HostName --port $Port --timeout 30
$dashboardCode = $LASTEXITCODE
# The model gateway has an independent lifecycle and may be shared by other
# local clients. Stopping EvoMind therefore targets only the PID-bound
# dashboard manager.
exit $dashboardCode
