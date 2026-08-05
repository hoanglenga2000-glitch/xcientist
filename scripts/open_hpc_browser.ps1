param(
  [int]$ProxyPort = 17897,
  [switch]$SkipBrowser,
  [switch]$NoPauseOnError,
  [switch]$NoCredentialPrompt
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$Manager = Join-Path $Root "scripts\manage_hpc_browser_proxy.ps1"
$BrowserProfile = Join-Path $env:USERPROFILE "hkust-hpc-chrome-data"
$Url = "http://100.85.169.63:1234/appform/login"
$StateDir = Join-Path $env:APPDATA "ResearchAgentWorkstation"
$CredentialPath = Join-Path $StateDir "hpc_socks_credential.xml"

function Invoke-Manager {
  param(
    [string]$Command,
    [switch]$AllowFailure
  )
  $raw = @(
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Manager $Command -ListenPort $ProxyPort -WebUrl $Url
  )
  $managerExit = $LASTEXITCODE
  $jsonLine = $raw | Where-Object {
    ([string]$_).TrimStart().StartsWith("{")
  } | Select-Object -Last 1
  $payload = if ($jsonLine) {
    ([string]$jsonLine) | ConvertFrom-Json
  } else {
    $null
  }
  if ($managerExit -ne 0 -and -not $AllowFailure) {
    throw "HPC portal proxy command '$Command' failed."
  }
  if ($null -eq $payload) {
    if ($managerExit -ne 0) {
      return [pscustomobject]@{
        status = "failed"
        failure_reason = "manager_command_failed"
        manager_exit_code = $managerExit
      }
    }
    throw "HPC portal proxy command '$Command' returned no JSON status."
  }
  $payload | Add-Member -NotePropertyName manager_exit_code -NotePropertyValue $managerExit -Force
  return $payload
}

function Repair-PortalCredential {
  Write-Host ""
  Write-Host "The saved school portal proxy credential was rejected." -ForegroundColor Yellow
  Write-Host "Enter the school portal proxy account in the secure Windows prompt." -ForegroundColor Cyan
  Write-Host "Do not enter the job SSH password." -ForegroundColor DarkGray
  $existing = if (Test-Path -LiteralPath $CredentialPath -PathType Leaf) {
    Import-Clixml -LiteralPath $CredentialPath
  } else {
    $null
  }
  $existingUser = if ($existing -is [pscredential]) { $existing.UserName } else { "" }
  if ([string]::IsNullOrWhiteSpace($existingUser)) {
    throw "The school portal proxy username is not installed."
  }
  $proxyUser = $existingUser
  Write-Host "Using the saved school portal proxy username." -ForegroundColor DarkGray
  $securePassword = Read-Host "School portal proxy password (input hidden)" -AsSecureString
  $credential = [pscredential]::new($proxyUser, $securePassword)
  $backupDir = Join-Path $StateDir ("credential_backups\portal_proxy_" + (Get-Date -Format "yyyyMMdd-HHmmss"))
  New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
  if (Test-Path -LiteralPath $CredentialPath -PathType Leaf) {
    Copy-Item -LiteralPath $CredentialPath -Destination (Join-Path $backupDir "hpc_socks_credential.xml")
  }
  $credential | Export-Clixml -LiteralPath $CredentialPath
  $stopped = Invoke-Manager -Command "stop"
  $started = Invoke-Manager -Command "start"
  return Invoke-Manager -Command "test" -AllowFailure
}

function Get-PortalFailureMessage([string]$Reason) {
  switch ($Reason) {
    "proxy_auth_failed" {
      return "The saved school portal proxy credential was rejected."
    }
    "portal_destination_unreachable" {
      return "The school proxy accepted the connection but cannot currently reach the HPC portal."
    }
    "upstream_proxy_unavailable" {
      return "The school upstream proxy is temporarily unavailable."
    }
    "portal_empty_reply" {
      return "The HPC portal accepted a connection but returned an empty response."
    }
    default {
      return "The HPC portal proxy health check failed: $Reason."
    }
  }
}

function Stop-StaleDedicatedBrowser {
  $escapedProfile = [regex]::Escape($BrowserProfile)
  $processes = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -in @("chrome.exe", "msedge.exe") -and
    $_.CommandLine -match $escapedProfile -and
    $_.CommandLine -notmatch "--type=" -and
    $_.CommandLine -notmatch "--proxy-server=socks5://127\.0\.0\.1:$ProxyPort(?:\s|$)"
  }
  foreach ($processInfo in $processes) {
    Stop-Process -Id $processInfo.ProcessId -Force -ErrorAction SilentlyContinue
  }
  if ($processes) { Start-Sleep -Milliseconds 700 }
}

try {
  if (-not (Test-Path -LiteralPath $Manager -PathType Leaf)) {
    throw "The HPC portal proxy manager is missing."
  }
  $start = Invoke-Manager -Command "start"
  $health = Invoke-Manager -Command "test" -AllowFailure
  if (
    $health.status -ne "passed" -and
    $health.failure_reason -eq "proxy_auth_failed" -and
    -not $NoCredentialPrompt
  ) {
    $health = Repair-PortalCredential
  }
  if ($health.status -ne "passed") {
    $message = Get-PortalFailureMessage -Reason ([string]$health.failure_reason)
    if ($health.failure_reason -eq "proxy_auth_failed" -and $NoCredentialPrompt) {
      $message += " Reopen the desktop shortcut to refresh it securely."
    }
    throw $message
  }

  if ($SkipBrowser) {
    [pscustomobject]@{
      status = "ready"
      proxy = "127.0.0.1:$ProxyPort"
      http_status = $health.http_status
      browser_started = $false
    } | ConvertTo-Json -Compress
    exit 0
  }

  $browserCandidates = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LocalAppData\Google\Chrome\Application\chrome.exe",
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
  )
  $browser = $browserCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
  if (-not $browser) {
    throw "Chrome or Edge was not found."
  }

  Stop-StaleDedicatedBrowser
  Start-Process -FilePath $browser -ArgumentList @(
    "--proxy-server=socks5://127.0.0.1:$ProxyPort",
    "--user-data-dir=`"$BrowserProfile`"",
    "--no-first-run",
    "--new-window",
    $Url
  )
} catch {
  Write-Host ""
  Write-Host "HKUST HPC portal launcher failed." -ForegroundColor Red
  Write-Host $_.Exception.Message -ForegroundColor Yellow
  Write-Host "The training SSH proxy on 127.0.0.1:7890 was not changed." -ForegroundColor DarkGray
  if (-not $NoPauseOnError -and $Host.Name -eq "ConsoleHost") {
    Read-Host "Press Enter to close"
  }
  exit 1
}
