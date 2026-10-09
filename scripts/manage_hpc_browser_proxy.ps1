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

function Read-ExactBytes([System.IO.Stream]$Stream, [int]$Count) {
  if ($Count -lt 0 -or $Count -gt 4096) {
    throw "Portal route reply length is invalid."
  }
  $buffer = [byte[]]::new($Count)
  $offset = 0
  while ($offset -lt $Count) {
    $read = $Stream.Read($buffer, $offset, $Count - $offset)
    if ($read -le 0) {
      throw "Portal route closed before the SOCKS reply completed."
    }
    $offset += $read
  }
  return ,$buffer
}

function Test-PortalRoute {
  $result = [ordered]@{
    route_verified = $false
    socks_reply_code = $null
    failure_reason = "portal_route_unverified"
  }
  $client = [System.Net.Sockets.TcpClient]::new()
  try {
    $uri = [Uri]$WebUrl
    if (-not $uri.IsAbsoluteUri -or $uri.Scheme -notin @("http", "https")) {
      throw "The HPC portal URL must use absolute HTTP or HTTPS."
    }
    $hostBytes = [System.Text.Encoding]::ASCII.GetBytes($uri.DnsSafeHost)
    if ($hostBytes.Length -lt 1 -or $hostBytes.Length -gt 255) {
      throw "The HPC portal host is invalid."
    }
    $port = if ($uri.IsDefaultPort) {
      if ($uri.Scheme -eq "https") { 443 } else { 80 }
    } else {
      $uri.Port
    }

    $connectTask = $client.ConnectAsync("127.0.0.1", $ListenPort)
    if (-not $connectTask.Wait(5000) -or -not $client.Connected) {
      $result.failure_reason = "local_proxy_unavailable"
      return [pscustomobject]$result
    }
    $stream = $client.GetStream()
    $stream.ReadTimeout = 8000
    $stream.WriteTimeout = 8000
    $methodRequest = [byte[]](5, 1, 0)
    $stream.Write($methodRequest, 0, $methodRequest.Length)
    $methodReply = Read-ExactBytes -Stream $stream -Count 2
    if ($methodReply[0] -ne 5 -or $methodReply[1] -ne 0) {
      $result.failure_reason = "local_proxy_method_rejected"
      return [pscustomobject]$result
    }

    $request = [System.Collections.Generic.List[byte]]::new()
    $request.AddRange([byte[]](5, 1, 0, 3, [byte]$hostBytes.Length))
    $request.AddRange($hostBytes)
    $request.Add([byte](($port -shr 8) -band 255))
    $request.Add([byte]($port -band 255))
    $requestBytes = $request.ToArray()
    $stream.Write($requestBytes, 0, $requestBytes.Length)

    $reply = Read-ExactBytes -Stream $stream -Count 4
    $result.socks_reply_code = [int]$reply[1]
    $addressBytes = switch ([int]$reply[3]) {
      1 { 4 }
      3 { [int](Read-ExactBytes -Stream $stream -Count 1)[0] }
      4 { 16 }
      default { throw "The portal route returned an unsupported address type." }
    }
    [void](Read-ExactBytes -Stream $stream -Count ($addressBytes + 2))
    if ($reply[0] -eq 5 -and $reply[1] -eq 0) {
      $result.route_verified = $true
      $result.failure_reason = ""
    } else {
      $result.failure_reason = "portal_destination_unreachable"
    }
  } catch [System.TimeoutException] {
    $result.failure_reason = "portal_route_timeout"
  } catch [System.IO.IOException] {
    $result.failure_reason = "portal_route_io_failed"
  } catch [System.Net.Sockets.SocketException] {
    $result.failure_reason = "portal_route_socket_failed"
  } finally {
    $client.Dispose()
  }
  return [pscustomobject]$result
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
  $route = if ($passed) {
    [pscustomobject]@{ route_verified = $true; socks_reply_code = 0; failure_reason = "" }
  } else {
    Test-PortalRoute
  }
  $routeReady = -not $passed -and $route.route_verified -and $curlExit -in @(28, 52)
  $failureReason = if ($passed) {
    $null
  } elseif ($routeReady) {
    "portal_response_pending"
  } else {
    $resolved = Resolve-PortalFailureReason -CurlExit $curlExit -BridgeDetail $bridgeDetail
    if ($resolved -eq "portal_health_check_failed" -and $route.failure_reason) {
      $route.failure_reason
    } else {
      $resolved
    }
  }
  [pscustomobject]@{
    status = $(if ($passed) { "passed" } elseif ($routeReady) { "route_ready" } else { "failed" })
    proxy = "127.0.0.1:$ListenPort"
    http_status = $(if ([string]$httpCode -match "^[0-9]{3}$") { [int]$httpCode } else { $null })
    curl_exit = $curlExit
    failure_reason = $failureReason
    credential_refresh_available = $failureReason -eq "proxy_auth_failed"
    route_verified = $route.route_verified
    socks_reply_code = $route.socks_reply_code
    http_response_pending = $routeReady
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
  exit $(if ($result.status -in @("passed", "route_ready")) { 0 } else { 1 })
}

$status = Get-Listener
Write-Output (@{
  status = $(if ($status.managed) { "running" } elseif ($status.listening) { "port_conflict" } else { "not_running" })
  listen_port = $ListenPort
  pid = $status.pid
  credential_installed = (Test-Path -LiteralPath $CredentialPath -PathType Leaf)
} | ConvertTo-Json -Compress)
