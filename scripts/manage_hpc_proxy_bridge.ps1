param(
  [ValidateSet("install-credential", "start", "stop", "status", "test")]
  [string]$Command = "status",
  [string]$ProxyUser = "",
  [System.Security.SecureString]$SecureProxyPassword,
  [switch]$SecretFromStdin,
  [int]$ListenPort = 7890,
  [string]$UpstreamHost = "",
  [int]$UpstreamPort = 0,
  [string]$DestinationHost = "",
  [int]$DestinationPort = 0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
. (Join-Path $Root "scripts\dpapi_credential_store.ps1")
$initializationMutex = Enter-EvoMindCredentialStoreLock
try {
  $StateDir = Initialize-EvoMindCredentialStateDirectory
} finally {
  Exit-EvoMindCredentialStoreLock $initializationMutex
}
$CredentialPath = Join-Path $StateDir "hpc_socks_credential.xml"
$PidPath = Join-Path $StateDir "hpc_socks_bridge.pid"
$OutLog = Join-Path $StateDir "hpc_socks_bridge.out.log"
$ErrLog = Join-Path $StateDir "hpc_socks_bridge.err.log"
$BridgeScript = Join-Path $Root "scripts\hpc_socks_bridge.py"
$BridgeLauncher = Join-Path $Root "scripts\start_hpc_socks_bridge.py"

function Get-PythonExe {
  $candidates = @(
    "C:\codex-python\python.exe",
    "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    "python.exe",
    "python"
  )
  foreach ($candidate in $candidates) {
    try {
      $resolved = Get-Command $candidate -ErrorAction Stop
      if ($resolved.Source) { return $resolved.Source }
    } catch {}
    if (Test-Path $candidate) { return $candidate }
  }
  throw "Python executable was not found."
}

function Resolve-NetworkEndpoint(
  [string]$EndpointHost,
  [int]$EndpointPort,
  [string]$HostEnvironmentName,
  [string]$PortEnvironmentName,
  [string]$Label
) {
  $resolvedHost = $EndpointHost.Trim()
  if ([string]::IsNullOrWhiteSpace($resolvedHost)) {
    $resolvedHost = [string][Environment]::GetEnvironmentVariable($HostEnvironmentName)
  }
  $resolvedPort = $EndpointPort
  if ($resolvedPort -eq 0) {
    $rawPort = [string][Environment]::GetEnvironmentVariable($PortEnvironmentName)
    if ($rawPort -match "^[0-9]{1,5}$") {
      $resolvedPort = [int]$rawPort
    }
  }
  Assert-EvoMindNetworkHost $resolvedHost
  if ($resolvedPort -lt 1 -or $resolvedPort -gt 65535) {
    throw "$Label port must be configured explicitly."
  }
  return [pscustomobject]@{ Host = $resolvedHost.Trim(); Port = $resolvedPort }
}

function Import-HpcCredential {
  if (-not (Test-Path $CredentialPath)) {
    throw "HPC SOCKS credential is not installed. Run install-credential with -ProxyUser and a hidden prompt."
  }
  $mutex = Enter-EvoMindCredentialStoreLock
  try {
    Assert-EvoMindCredentialDestination $CredentialPath
    Protect-EvoMindCredentialPath -Path $CredentialPath
    $credential = Import-Clixml -LiteralPath $CredentialPath
    if (
      $credential -isnot [System.Management.Automation.PSCredential] -or
      $credential.UserName -notmatch "^[A-Za-z0-9._@-]{1,256}$" -or
      $credential.Password.Length -eq 0
    ) {
      throw [System.IO.InvalidDataException]::new("HPC SOCKS credential is invalid.")
    }
    return $credential
  } finally {
    Exit-EvoMindCredentialStoreLock $mutex
  }
}

function Test-PortListening([int]$Port) {
  $conn = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($conn) {
    $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
    return [pscustomobject]@{ listening = $true; pid = $conn.OwningProcess; process = $proc.ProcessName }
  }
  [pscustomobject]@{ listening = $false; pid = $null; process = $null }
}

function Stop-Bridge {
  $status = Test-PortListening $ListenPort
  if ($status.listening -and $status.process -match "python") {
    Stop-Process -Id $status.pid -Force -ErrorAction SilentlyContinue
  }
  if (Test-Path $PidPath) {
    $pidValue = Get-Content $PidPath -ErrorAction SilentlyContinue
    if ($pidValue) {
      Stop-Process -Id ([int]$pidValue) -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $PidPath -Force -ErrorAction SilentlyContinue
  }
}

if ($Command -eq "install-credential") {
  if ($ProxyUser -notmatch "^[A-Za-z0-9._@-]{1,256}$") {
    throw "A valid ProxyUser is required for install-credential."
  }
  $secure = Read-EvoMindSecureInput `
    -Provided $SecureProxyPassword `
    -Prompt "HPC SOCKS password" `
    -FromStdin:$SecretFromStdin
  if ($secure.Length -eq 0) {
    throw "HPC SOCKS credential input was empty."
  }
  try {
    $credential = [pscredential]::new($ProxyUser, $secure)
    $mutex = Enter-EvoMindCredentialStoreLock
    try {
      $temporaryPath = Write-EvoMindCredentialTemp `
        -Credential $credential `
        -StateDirectory $StateDir `
        -Prefix "hpc-socks"
      Commit-EvoMindCredentialFiles @(
        [ordered]@{ TemporaryPath = $temporaryPath; DestinationPath = $CredentialPath }
      )
    } finally {
      Exit-EvoMindCredentialStoreLock $mutex
    }
  } finally {
    $secure.Dispose()
  }
  Write-Output (@{ status = "installed"; credential_path = $CredentialPath } | ConvertTo-Json -Depth 4)
  exit 0
}

if ($Command -eq "stop") {
  Stop-Bridge
  Write-Output (@{ status = "stopped"; listen_port = $ListenPort } | ConvertTo-Json -Depth 4)
  exit 0
}

if ($Command -eq "start") {
  $current = Test-PortListening $ListenPort
  if ($current.listening) {
    Write-Output (@{ status = "already_running"; listen_port = $ListenPort; pid = $current.pid; process = $current.process } | ConvertTo-Json -Depth 4)
    exit 0
  }
  $upstream = Resolve-NetworkEndpoint `
    -EndpointHost $UpstreamHost `
    -EndpointPort $UpstreamPort `
    -HostEnvironmentName "HPC_SOCKS_HOST" `
    -PortEnvironmentName "HPC_SOCKS_PORT" `
    -Label "HPC SOCKS upstream"
  $credential = Import-HpcCredential
  $python = Get-PythonExe
  $savedSensitiveEnvironment = [ordered]@{}
  foreach ($entry in [System.Environment]::GetEnvironmentVariables().GetEnumerator()) {
    $key = [string]$entry.Key
    if ($key -match "(?i)(api[_-]?key|authorization|cookie|credential|password|passwd|secret|token)") {
      $savedSensitiveEnvironment[$key] = [string]$entry.Value
      [Environment]::SetEnvironmentVariable($key, $null, "Process")
    }
  }
  $previousProxyUser = [Environment]::GetEnvironmentVariable("HPC_SOCKS_USER", "Process")
  try {
    [Environment]::SetEnvironmentVariable("HPC_SOCKS_USER", $credential.UserName, "Process")
    [Environment]::SetEnvironmentVariable(
      "HPC_SOCKS_PASSWORD",
      $credential.GetNetworkCredential().Password,
      "Process"
    )
    $launcherOutput = & $python $BridgeLauncher `
      --bridge-script $BridgeScript `
      --working-directory $Root `
      --listen-port $ListenPort `
      --upstream-host $upstream.Host `
      --upstream-port $upstream.Port `
      --stdout-log $OutLog `
      --stderr-log $ErrLog
    if ($LASTEXITCODE -ne 0) {
      throw "HPC SOCKS bridge launcher failed."
    }
    $launcherResult = $launcherOutput | ConvertFrom-Json
    $process = Get-Process -Id ([int]$launcherResult.pid) -ErrorAction Stop
  } finally {
    [Environment]::SetEnvironmentVariable("HPC_SOCKS_USER", $previousProxyUser, "Process")
    [Environment]::SetEnvironmentVariable("HPC_SOCKS_PASSWORD", $null, "Process")
    foreach ($key in $savedSensitiveEnvironment.Keys) {
      [Environment]::SetEnvironmentVariable(
        [string]$key,
        [string]$savedSensitiveEnvironment[$key],
        "Process"
      )
    }
  }
  Set-Content -Path $PidPath -Value $process.Id -Encoding ASCII
  $deadline = (Get-Date).AddSeconds(8)
  do {
    Start-Sleep -Milliseconds 200
    $status = Test-PortListening $ListenPort
  } while (-not $status.listening -and -not $process.HasExited -and (Get-Date) -lt $deadline)
  Write-Output (@{ status = $(if ($status.listening) { "started" } else { "failed" }); listen_port = $ListenPort; pid = $process.Id; process = $status.process; log = $OutLog } | ConvertTo-Json -Depth 4)
  exit $(if ($status.listening) { 0 } else { 1 })
}

if ($Command -eq "test") {
  $destination = Resolve-NetworkEndpoint `
    -EndpointHost $DestinationHost `
    -EndpointPort $DestinationPort `
    -HostEnvironmentName "GPU_SSH_HOST" `
    -PortEnvironmentName "GPU_SSH_PORT" `
    -Label "HPC test destination"
  $status = Test-PortListening $ListenPort
  if (-not $status.listening) {
    throw "HPC SOCKS bridge is not listening on 127.0.0.1:$ListenPort."
  }
  $python = Get-PythonExe
  $test = & $python (Join-Path $Root "scripts\verify_hpc_socks_gateway.py") `
    --proxy-host 127.0.0.1 `
    --proxy-port $ListenPort `
    --dest-host $destination.Host `
    --dest-port $destination.Port
  Write-Output $test
  exit 0
}

$status = Test-PortListening $ListenPort
Write-Output (@{ status = $(if ($status.listening) { "running" } else { "not_running" }); listen_port = $ListenPort; pid = $status.pid; process = $status.process; credential_installed = (Test-Path $CredentialPath) } | ConvertTo-Json -Depth 4)
