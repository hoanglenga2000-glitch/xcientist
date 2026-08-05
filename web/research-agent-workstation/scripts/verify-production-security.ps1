param(
  [int]$Port = 8091
)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $PSScriptRoot
$NextCli = Join-Path $AppRoot "node_modules\next\dist\bin\next"
$LogRoot = Join-Path ([System.IO.Path]::GetTempPath()) "evomind-runtime-security-$PID"
$Stdout = Join-Path $LogRoot "server.out.log"
$Stderr = Join-Path $LogRoot "server.err.log"
$BaseUrl = "http://127.0.0.1:$Port"
$DatabaseName = "ci-workstation-$PID.db"
$DatabasePath = Join-Path $AppRoot "prisma\$DatabaseName"
$PrismaPush = Join-Path $AppRoot "scripts\prisma-db-push.mjs"

function Invoke-Status([string[]]$CurlArgs) {
  $status = & curl.exe -sS -o NUL -w "%{http_code}" @CurlArgs
  if ($LASTEXITCODE -ne 0) {
    throw "curl failed with exit code $LASTEXITCODE"
  }
  return [string]$status
}

function Assert-Status([string]$Name, [string]$Expected, [string[]]$CurlArgs) {
  $actual = Invoke-Status $CurlArgs
  if ($actual -ne $Expected) {
    throw "$Name returned HTTP $actual, expected $Expected"
  }
}

if (-not (Test-Path -LiteralPath $NextCli)) {
  throw "Next.js CLI not found: $NextCli"
}
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$env:DATABASE_URL = "file:./$DatabaseName"
$env:NEXT_TELEMETRY_DISABLED = "1"

$server = $null
try {
  & node $PrismaPush "--skip-generate"
  if ($LASTEXITCODE -ne 0) {
    throw "Prisma schema initialization failed with exit code $LASTEXITCODE"
  }

  $server = Start-Process `
    -FilePath "node" `
    -ArgumentList @($NextCli, "start", "--hostname", "127.0.0.1", "--port", [string]$Port) `
    -WorkingDirectory $AppRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -PassThru

  $ready = $false
  for ($attempt = 0; $attempt -lt 40; $attempt++) {
    if ($server.HasExited) {
      throw "Next.js exited before the security smoke was ready"
    }
    $status = Invoke-Status @("$BaseUrl/api/workstation-summary")
    if ($status -eq "200") {
      $ready = $true
      break
    }
    Start-Sleep -Milliseconds 500
  }
  if (-not $ready) {
    throw "Next.js security smoke did not become ready"
  }

  Assert-Status "normal loopback request" "200" @("$BaseUrl/api/workstation-summary")
  Assert-Status "non-loopback Host" "403" @(
    "-H", "Host: attacker.invalid:$Port", "$BaseUrl/api/workstation-summary"
  )
  Assert-Status "userinfo Host confusion" "403" @(
    "-H", "Host: attacker@127.0.0.1:$Port", "$BaseUrl/api/workstation-summary"
  )
  Assert-Status "cross-origin mutation" "403" @(
    "-X", "POST", "-H", "Origin: https://attacker.invalid", "-H", "Content-Type: application/json",
    "--data", "{}", "$BaseUrl/api/workstation-actions"
  )
  Assert-Status "mutation without browser source" "403" @(
    "-X", "POST", "-H", "Content-Type: application/json",
    "--data", "{}", "$BaseUrl/api/workstation-actions"
  )
  Assert-Status "task path traversal" "400" @(
    "--path-as-is", "-X", "POST", "-H", "Origin: $BaseUrl", "-H", "Content-Type: application/json",
    "--data", "{}", "$BaseUrl/api/tasks/%2e%2e%5Cx/import-agent-patch"
  )
  Assert-Status "valid task route business validation" "400" @(
    "-X", "POST", "-H", "Origin: $BaseUrl", "-H", "Content-Type: application/json",
    "--data", "{}", "$BaseUrl/api/tasks/titanic/import-agent-patch"
  )
  Assert-Status "task trailing-dot alias" "400" @(
    "--path-as-is", "-X", "POST", "-H", "Origin: $BaseUrl", "-H", "Content-Type: application/json",
    "--data", "{}", "$BaseUrl/api/tasks/foo./import-agent-patch"
  )
  Write-Host "production security runtime smoke passed"
} finally {
  if ($server -and -not $server.HasExited) {
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    $server.WaitForExit(5000) | Out-Null
  }
  foreach ($path in @($DatabasePath, "$DatabasePath-journal", "$DatabasePath-shm", "$DatabasePath-wal")) {
    Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
  }
}
