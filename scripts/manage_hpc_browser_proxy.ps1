param(
  [ValidateSet("start", "stop", "status", "test", "refresh-credential")]
  [string]$Command = "status",
  [int]$ListenPort = 17897,
  [string]$WebUrl = "http://100.85.169.63:1234/appform/login"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$StateDir = Join-Path $env:APPDATA "ResearchAgentWorkstation"
$CredentialPath = Join-Path $StateDir "hpc_socks_credential.xml"
$PidPath = Join-Path $StateDir "hpc_browser_proxy.pid"
$OutLog = Join-Path $StateDir "hpc_browser_proxy.out.log"
$ErrLog = Join-Path $StateDir "hpc_browser_proxy.err.log"
$BridgeScript = Join-Path $Root "scripts\hpc_socks_bridge.py"

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
    if (Test-Path -LiteralPath $candidate) { return $candidate }
  }
  throw "Python executable was not found."
}

function Get-Listener {
  $connection = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $ListenPort -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if (-not $connection) {
    return [pscustomobject]@{ listening = $false; pid = $null; process = $null; managed = $false }
  }

  $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$($connection.OwningProcess)" -ErrorAction SilentlyContinue
  $managed = $false
  if ($processInfo) {
    $managed = $processInfo.CommandLine -like "*$BridgeScript*" -and
      $processInfo.CommandLine -match "--listen-port\s+$ListenPort(?:\s|$)"
  }
  [pscustomobject]@{
    listening = $true
    pid = $connection.OwningProcess
    process = $processInfo.Name
    managed = $managed
  }
}

function Stop-ManagedProxy {
  $listener = Get-Listener
  if ($listener.listening -and -not $listener.managed) {
    throw "Port $ListenPort is owned by another process (PID $($listener.pid))."
  }
  if ($listener.managed) {
    Stop-Process -Id $listener.pid -Force -ErrorAction Stop
    Start-Sleep -Milliseconds 400
  }
  Remove-Item -LiteralPath $PidPath -Force -ErrorAction SilentlyContinue
}

function Read-NewLogText([string]$Path, [long]$Offset) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    return ""
  }
  $stream = [System.IO.File]::Open(
    $Path,
    [System.IO.FileMode]::Open,
    [System.IO.FileAccess]::Read,
    [System.IO.FileShare]::ReadWrite
  )
  try {
    $safeOffset = [Math]::Min([Math]::Max([long]0, $Offset), $stream.Length)
    [void]$stream.Seek($safeOffset, [System.IO.SeekOrigin]::Begin)
    $reader = [System.IO.StreamReader]::new(
      $stream,
      [System.Text.UTF8Encoding]::new($false),
      $true,
      1024,
      $true
    )
    try {
      return $reader.ReadToEnd()
    } finally {
      $reader.Dispose()
    }
  } finally {
    $stream.Dispose()
  }
}

function Resolve-PortalFailureReason([int]$CurlExit, [string]$BridgeDetail) {
  if ($BridgeDetail -match "username/password auth failed") {
    return "proxy_auth_failed"
  }
  if ($BridgeDetail -match "connect failed with code 3") {
    return "portal_destination_unreachable"
  }
  if ($BridgeDetail -match "timed out|ConnectionResetError|unexpected socket close") {
    return "upstream_proxy_unavailable"
  }
  if ($CurlExit -eq 97) {
    return "socks_connect_failed"
  }
  if ($CurlExit -eq 52) {
    return "portal_empty_reply"
  }
  return "portal_health_check_failed"
}

function Test-WebPortal {
  $curl = Get-Command "curl.exe" -ErrorAction Stop
  $logOffset = if (Test-Path -LiteralPath $ErrLog -PathType Leaf) {
    (Get-Item -LiteralPath $ErrLog).Length
  } else {
    0
  }
  $previousErrorAction = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    $httpCode = & $curl.Source @(
      "--silent",
      "--show-error",
      "--output", "NUL",
      "--write-out", "%{http_code}",
      "--max-time", "20",
      "--socks5-hostname", "127.0.0.1:$ListenPort",
      $WebUrl
    ) 2>$null
    $curlExit = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousErrorAction
  }
  Start-Sleep -Milliseconds 150
  $bridgeDetail = (Read-NewLogText -Path $ErrLog -Offset $logOffset).Trim()
  $passed = $curlExit -eq 0 -and [string]$httpCode -match "^[23][0-9][0-9]$"
  $failureReason = if ($passed) {
    $null
  } else {
    Resolve-PortalFailureReason -CurlExit $curlExit -BridgeDetail $bridgeDetail
  }
  [pscustomobject]@{
    status = $(if ($passed) { "passed" } else { "failed" })
    proxy = "127.0.0.1:$ListenPort"
    http_status = $(if ([string]$httpCode -match "^[0-9]{3}$") { [int]$httpCode } else { $null })
    curl_exit = $curlExit
    failure_reason = $failureReason
    credential_refresh_available = $failureReason -eq "proxy_auth_failed"
  }
}

New-Item -ItemType Directory -Path $StateDir -Force | Out-Null

if ($Command -eq "refresh-credential") {
  $existing = if (Test-Path -LiteralPath $CredentialPath -PathType Leaf) {
    Import-Clixml -LiteralPath $CredentialPath
  } else {
    $null
  }
  $existingUser = if ($existing -is [pscredential]) { $existing.UserName } else { "" }
  $credential = Get-Credential -UserName $existingUser -Message "Enter the school HPC portal proxy account. Do not enter the job SSH password."
  if ($credential -isnot [pscredential] -or [string]::IsNullOrWhiteSpace($credential.UserName)) {
    throw "HPC portal proxy credential refresh was cancelled."
  }
  $backupRoot = Join-Path $StateDir "credential_backups"
  $backupDir = Join-Path $backupRoot ("portal_proxy_" + (Get-Date -Format "yyyyMMdd-HHmmss"))
  New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
  $backupPath = Join-Path $backupDir "hpc_socks_credential.xml"
  if (Test-Path -LiteralPath $CredentialPath -PathType Leaf) {
    Copy-Item -LiteralPath $CredentialPath -Destination $backupPath
  }
  $credential | Export-Clixml -LiteralPath $CredentialPath
  Write-Output (@{
    status = "credential_refreshed"
    credential_path = $CredentialPath
    backup_path = $(if (Test-Path -LiteralPath $backupPath) { $backupPath } else { $null })
    storage = "windows_dpapi_current_user"
    training_ssh_proxy_modified = $false
  } | ConvertTo-Json -Compress)
  exit 0
}

if ($Command -eq "stop") {
  Stop-ManagedProxy
  Write-Output (@{ status = "stopped"; listen_port = $ListenPort } | ConvertTo-Json -Compress)
  exit 0
}

if ($Command -eq "start") {
  $current = Get-Listener
  if ($current.listening) {
    if (-not $current.managed) {
      throw "Port $ListenPort is already owned by another process (PID $($current.pid))."
    }
    Write-Output (@{ status = "already_running"; listen_port = $ListenPort; pid = $current.pid } | ConvertTo-Json -Compress)
    exit 0
  }
  if (-not (Test-Path -LiteralPath $CredentialPath -PathType Leaf)) {
    throw "The DPAPI-backed HPC portal proxy credential is not installed."
  }
  if (-not (Test-Path -LiteralPath $BridgeScript -PathType Leaf)) {
    throw "The HPC SOCKS bridge script is missing."
  }

  $credential = Import-Clixml -LiteralPath $CredentialPath
  if ($credential -isnot [pscredential] -or [string]::IsNullOrWhiteSpace($credential.UserName)) {
    throw "The DPAPI-backed HPC portal proxy credential is invalid."
  }
  $python = Get-PythonExe
  $oldUser = [Environment]::GetEnvironmentVariable("HPC_SOCKS_USER", "Process")
  $oldPassword = [Environment]::GetEnvironmentVariable("HPC_SOCKS_PASSWORD", "Process")
  try {
    $env:HPC_SOCKS_USER = $credential.UserName
    $env:HPC_SOCKS_PASSWORD = $credential.GetNetworkCredential().Password
    Set-Content -LiteralPath $OutLog -Value "" -Encoding UTF8
    Set-Content -LiteralPath $ErrLog -Value "" -Encoding UTF8
    $processInfo = Start-Process `
      -FilePath $python `
      -ArgumentList @("`"$BridgeScript`"", "--listen-port", [string]$ListenPort) `
      -WorkingDirectory $Root `
      -WindowStyle Hidden `
      -RedirectStandardOutput $OutLog `
      -RedirectStandardError $ErrLog `
      -PassThru
  } finally {
    if ($null -eq $oldUser) { Remove-Item Env:\HPC_SOCKS_USER -ErrorAction SilentlyContinue } else { $env:HPC_SOCKS_USER = $oldUser }
    if ($null -eq $oldPassword) { Remove-Item Env:\HPC_SOCKS_PASSWORD -ErrorAction SilentlyContinue } else { $env:HPC_SOCKS_PASSWORD = $oldPassword }
  }

  Set-Content -LiteralPath $PidPath -Value $processInfo.Id -Encoding ASCII
  Start-Sleep -Milliseconds 900
  $started = Get-Listener
  if (-not $started.managed) {
    $detail = (Get-Content -LiteralPath $ErrLog -Raw -ErrorAction SilentlyContinue).Trim()
    throw "The HPC portal proxy did not start. $detail"
  }
  Write-Output (@{ status = "started"; listen_port = $ListenPort; pid = $started.pid } | ConvertTo-Json -Compress)
  exit 0
}

if ($Command -eq "test") {
  $current = Get-Listener
  if (-not $current.managed) {
    throw "The managed HPC portal proxy is not listening on 127.0.0.1:$ListenPort."
  }
  $result = Test-WebPortal
  Write-Output ($result | ConvertTo-Json -Compress)
  exit $(if ($result.status -eq "passed") { 0 } else { 1 })
}

$status = Get-Listener
Write-Output (@{
  status = $(if ($status.managed) { "running" } elseif ($status.listening) { "port_conflict" } else { "not_running" })
  listen_port = $ListenPort
  pid = $status.pid
  credential_installed = (Test-Path -LiteralPath $CredentialPath -PathType Leaf)
} | ConvertTo-Json -Compress)
