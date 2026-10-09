param(
  [int]$Port = 8088,
  [string]$PythonExecutable = "",
  [string]$ProfileRoot = "",
  [switch]$SkipBuild,
  [switch]$SkipBrowserSmoke,
  # Explicit, recorded waivers for launch-readiness checks that fail only
  # because of external / fresh-install conditions (see
  # EXTERNAL_WAIVABLE_CHECKS in verify_workstation_launch_readiness.py).
  [string[]]$WaiveExternal = @()
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
# Port-bound verifiers that read their own base URL (e.g. UI component wiring)
# must target the acceptance port, not the default 8088.
$env:WORKSTATION_TEST_BASE_URL = $BaseUrl
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
    # Native tools (node, npm, prisma, pip) write warnings to stderr. Under
    # Windows PowerShell 5.1 with ErrorActionPreference=Stop the first such
    # line aborts the check even when the tool exits 0, so collect output with
    # Continue and fail on the exit code or on any non-native error record.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $output = @(& $Script 2>&1) } finally { $ErrorActionPreference = $previousPreference }
    $scriptErrors = @($output | Where-Object { $_ -is [System.Management.Automation.ErrorRecord] -and $_.FullyQualifiedErrorId -notlike "NativeCommandError*" })
    if ($scriptErrors.Count -gt 0) { throw "$Id failed: $($scriptErrors[0])" }
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
  & $PythonExe -m pip install -r $pytestRequirement --quiet
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
  # The API proxy fails closed without a session secret, a hashed
  # local-automation token and a bound runtime build identity. Only the
  # managed launcher provides all three (and starts the runtime service), so
  # the release gate uses it instead of the legacy restart script.
  & $PythonExe scripts\manage_workstation_dashboard.py start `
    --host 127.0.0.1 `
    --port $Port `
    --build `
    --force `
    --timeout 180
  if ($LASTEXITCODE -ne 0) { return }
  $html = (Invoke-WebRequest -Uri "$BaseUrl/" -UseBasicParsing -TimeoutSec 30).Content
  $cssHref = [regex]::Match($html, 'href="([^"]*\.css[^"]*)"').Groups[1].Value
  if (-not $cssHref) { throw "No CSS bundle was referenced by the frontend HTML." }
  $css = Invoke-WebRequest -Uri "$BaseUrl$cssHref" -UseBasicParsing -TimeoutSec 30
  if ($css.StatusCode -ne 200 -or -not $css.Content.Contains("--tw-border-spacing-x")) {
    throw "CSS health check failed for $cssHref"
  }
  "frontend ready at $BaseUrl (css $cssHref)"
}
$checks += Run-Check "establish_release_check_session" {
  # Sign in the way a person does: exchange the server's one-time bootstrap
  # URL for the ordinary local session (loopback-only, single use). Live and
  # browser checks then see principal-scoped pages as the signed-in user. The
  # session file is owner-only and removed below and by the manager on stop.
  & $PythonExe scripts\workstation_local_auth.py establish --base-url $BaseUrl
}
$checks += Run-Check "new_user_release_readiness_live" { & $PythonExe scripts\verify_new_user_release_readiness.py --base-url $BaseUrl --write-report --require-live-server }
$checks += Run-Check "workstation_launch_readiness" {
  $waiverArgs = @($WaiveExternal | Where-Object { $_ } | ForEach-Object { "--waive-external"; $_ })
  & $PythonExe scripts\verify_workstation_launch_readiness.py --base-url $BaseUrl --include-frontend --write-report @waiverArgs
}

if (-not $SkipBrowserSmoke) {
  Step "Browser and interaction checks"
  $checks += Run-Check "browser_render_smoke" { & $PythonExe scripts\verify_workstation_browser_render_smoke.py --base-url $BaseUrl --write-report }
  $checks += Run-Check "click_smoke" { node scripts\verify_workstation_click_smoke.mjs --base-url $BaseUrl --write-report }
  $checks += Run-Check "interactive_controls" { node scripts\verify_workstation_interactive_controls.mjs --base-url $BaseUrl --write-report }
}
& $PythonExe scripts\workstation_local_auth.py remove --base-url $BaseUrl | Out-Null

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
  waived_external = @($WaiveExternal | Where-Object { $_ })
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
