param(
  [int]$Port = 8088,
  [string]$PythonExecutable = "",
  [string]$ProfileRoot = "",
  [switch]$SkipBuild,
  [switch]$SkipBrowserSmoke
)

$ErrorActionPreference = "Stop"
try {
  [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
  $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {}

$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location -LiteralPath $Root
$Web = Join-Path $Root "web\research-agent-workstation"
$BaseUrl = "http://127.0.0.1:$Port"
$ReportJson = Join-Path $Root "workspace\new_user_release_acceptance.json"
$ReportMd = Join-Path $Root "reports\NEW_USER_RELEASE_ACCEPTANCE.md"

if ($PythonExecutable) {
  $PythonExe = (Resolve-Path -LiteralPath $PythonExecutable -ErrorAction Stop).Path
} else {
  $PythonExe = (Get-Command python -ErrorAction Stop).Source
}
if (-not $ProfileRoot) {
  $ProfileRoot = Join-Path $Root "workspace\release-acceptance-profile"
}
$ProfileRoot = [IO.Path]::GetFullPath($ProfileRoot)
$env:USERPROFILE = $ProfileRoot
$env:HOME = $ProfileRoot
$env:HOMEDRIVE = [IO.Path]::GetPathRoot($ProfileRoot).TrimEnd('\')
$env:HOMEPATH = $ProfileRoot.Substring([IO.Path]::GetPathRoot($ProfileRoot).Length - 1)
$env:APPDATA = Join-Path $ProfileRoot "AppData\Roaming"
$env:LOCALAPPDATA = Join-Path $ProfileRoot "AppData\Local"
$env:XSCI_SHIM_DIR = Join-Path $ProfileRoot ".xsci\bin"
$env:PSModuleAnalysisCachePath = Join-Path $env:LOCALAPPDATA "evomind-powershell-cache\ModuleAnalysisCache"
$env:WORKSTATION_PYTHON = $PythonExe
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
New-Item -ItemType Directory -Force -Path $env:APPDATA,$env:LOCALAPPDATA,$env:XSCI_SHIM_DIR,(Split-Path -Parent $env:PSModuleAnalysisCachePath) | Out-Null

function Step([string]$Text) {
  Write-Host ""
  Write-Host ">>> $Text" -ForegroundColor Cyan
}

function Run-Check([string]$Id, [scriptblock]$Script) {
  $started = Get-Date
  Write-Host "  - $Id" -ForegroundColor Cyan
  $output = @()
  try {
    $global:LASTEXITCODE = 0
    $output = @(& $Script 2>&1)
    if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) { throw "$Id failed with exit code $LASTEXITCODE" }
    [pscustomobject][ordered]@{ id=$Id; ok=$true; seconds=[math]::Round(((Get-Date)-$started).TotalSeconds,3); error=$null; output_tail=@($output | Select-Object -Last 20 | ForEach-Object {[string]$_}) }
  } catch {
    [pscustomobject][ordered]@{ id=$Id; ok=$false; seconds=[math]::Round(((Get-Date)-$started).TotalSeconds,3); error=[string]$_.Exception.Message; output_tail=@($output | Select-Object -Last 20 | ForEach-Object {[string]$_}) }
  }
}

$checks = @()
Step "Isolated runtime and installer"
$checks += Run-Check "acceptance_test_dependencies" {
  $pytestRequirement = Join-Path $Root "requirements-dev.txt"
  & $PythonExe -m pip install $pytestRequirement --quiet
}
$checks += Run-Check "stop_existing_workstation_frontend" {
  & $PythonExe scripts\manage_workstation_dashboard.py stop --port $Port --force
}
$checks += Run-Check "installer_smoke_no_secrets" {
  $stdout = Join-Path $env:LOCALAPPDATA "evomind-installer.stdout.log"
  $stderr = Join-Path $env:LOCALAPPDATA "evomind-installer.stderr.log"
  $installerArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $Root "install.ps1"),
    "-PythonExecutable", $PythonExe,
    "-DataDir", (Join-Path $ProfileRoot "EvoMind\data"),
    "-LogsDir", (Join-Path $ProfileRoot "EvoMind\logs"),
    "-BackupsDir", (Join-Path $ProfileRoot "EvoMind\backups"),
    "-ProfilesDir", (Join-Path $ProfileRoot "EvoMind\profiles"),
    "-SecretsDir", (Join-Path $ProfileRoot "EvoMind\secrets"),
    "-SkipBuild", "-SkipSecretPrompt", "-SkipVerify", "-InstallUserShims", "-NoPathPrepend"
  )
  $installer = Start-Process -FilePath "powershell.exe" -ArgumentList $installerArgs -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru
  if ($installer.ExitCode -ne 0) { throw "Installer smoke failed with exit code $($installer.ExitCode): $((Get-Content -LiteralPath $stderr -Tail 30) -join "`n")" }
}

Step "Static, Python and Web checks"
$checks += Run-Check "python_core_compile" {
  & $PythonExe -m compileall -q src scripts
}
$checks += Run-Check "powershell_script_parse" {
  foreach ($rel in @("install.ps1","scripts\start_verified_workstation.ps1","scripts\restart_workstation_frontend.ps1","scripts\run_new_user_release_acceptance.ps1")) {
    $tokens=$null; $errors=$null
    [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $Root $rel),[ref]$tokens,[ref]$errors) | Out-Null
    if ($errors.Count) { throw (($errors | ForEach-Object { "$rel`: $($_.Message)" }) -join "; ") }
  }
}
$checks += Run-Check "cli_tests" {
  & $PythonExe -m pytest tests/test_kaggle_menu.py tests/test_xsci_cli.py tests/test_kaggle_stream.py -q
}
$checks += Run-Check "plaintext_secret_scan" {
  & $PythonExe scripts\verify_no_plaintext_secrets.py
}
$checks += Run-Check "frontend_typecheck" { Push-Location $Web; try { npm run typecheck } finally { Pop-Location } }
$checks += Run-Check "frontend_lint" { Push-Location $Web; try { npm run lint } finally { Pop-Location } }
$checks += Run-Check "frontend_test" { Push-Location $Web; try { npm run test } finally { Pop-Location } }
if (-not $SkipBuild) {
  $checks += Run-Check "frontend_build" { Push-Location $Web; try { npm run build } finally { Pop-Location } }
}

Step "Start production workstation and live gates"
$checks += Run-Check "start_production_workstation_frontend" {
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\restart_workstation_frontend.ps1 `
    -Port $Port `
    -PythonExecutable $PythonExe `
    -Mode production `
    -DatabaseUrl "file:./prisma/workstation.db"
}
$checks += Run-Check "new_user_release_readiness_live" { & $PythonExe scripts\verify_new_user_release_readiness.py --base-url $BaseUrl --write-report --require-live-server }
$checks += Run-Check "workstation_launch_readiness" { & $PythonExe scripts\verify_workstation_launch_readiness.py --base-url $BaseUrl --include-frontend --write-report }

if (-not $SkipBrowserSmoke) {
  Step "Browser and interaction checks"
  $checks += Run-Check "browser_render_smoke" { & $PythonExe scripts\verify_workstation_browser_render_smoke.py --base-url $BaseUrl --write-report }
  $checks += Run-Check "click_smoke" { node scripts\verify_workstation_click_smoke.mjs --base-url $BaseUrl --write-report }
  $checks += Run-Check "interactive_controls" { node scripts\verify_workstation_interactive_controls.mjs --base-url $BaseUrl --write-report }
}

$checkArray = @($checks)
$failed = @($checkArray | Where-Object { -not $_.ok })
$status = if ($failed.Count -eq 0) { "passed" } else { "failed" }
$summary = [ordered]@{
  schema = "xcientist.new_user_release_acceptance.v2"
  created_at = (Get-Date).ToString("s")
  status = $status
  default_gateway = "$BaseUrl/?page=assistant"
  python = $PythonExe
  profile_root = $ProfileRoot
  failed_checks = @($failed | ForEach-Object { $_.id })
  checks = $checkArray
  claim_boundary = "This validates the local EvoMind release surface. Kaggle submission remains human-gated and HPC readiness requires current job-container identity evidence."
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ReportJson),(Split-Path -Parent $ReportMd) | Out-Null
$summary | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $ReportJson
$md = @("# New User Release Acceptance","","- status: ``$status``","- default_gateway: $BaseUrl/?page=assistant","- failed_checks: ``$(@($summary.failed_checks) -join ', ')``","","## Checks","","| id | ok | seconds | error |","| --- | --- | ---: | --- |")
foreach ($check in $checks) { $err=if($check.error){$check.error.Replace('|','/')}else{''}; $md += "| ``$($check.id)`` | ``$($check.ok)`` | $($check.seconds) | $err |" }
$md += @("","## Claim Boundary","",$summary.claim_boundary,"")
$md | Set-Content -Encoding UTF8 $ReportMd
$summary | ConvertTo-Json -Depth 5
if ($status -ne "passed") { exit 1 }
