# New-user release acceptance runner.
# This validates install and EvoMind gateway readiness only. It does not start model
# training, GPU jobs, external LLM calls, or official Kaggle submissions.
param(
  [int]$Port = 8088,
  [switch]$SkipBuild,
  [switch]$SkipBrowserSmoke
)

$ErrorActionPreference = "Stop"
try {
  [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
  $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {
  # Best effort for legacy Windows PowerShell.
}

$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$Web = Join-Path $Root "web\research-agent-workstation"
$BaseUrl = "http://127.0.0.1:$Port"
$ReportJson = Join-Path $Root "workspace\new_user_release_acceptance.json"
$ReportMd = Join-Path $Root "reports\NEW_USER_RELEASE_ACCEPTANCE.md"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

function Step([string]$Text) {
  Write-Host ""
  Write-Host ">>> $Text" -ForegroundColor Cyan
}

function Run-Check([string]$Id, [scriptblock]$Script) {
  $started = Get-Date
  Write-Host "  - $Id" -ForegroundColor Cyan
  try {
    $global:LASTEXITCODE = 0
    $output = @(& $Script 2>&1)
    if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
      throw "$Id failed with exit code $LASTEXITCODE"
    }
    return [pscustomobject][ordered]@{
      id = $Id
      ok = $true
      seconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 3)
      error = $null
      output_tail = @()
    }
  } catch {
    $tail = @()
    if ($output) {
      $tail = @($output | Select-Object -Last 20 | ForEach-Object { [string]$_ })
    }
    return [pscustomobject][ordered]@{
      id = $Id
      ok = $false
      seconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 3)
      error = [string]$_.Exception.Message
      output_tail = $tail
    }
  }
}

$checks = @()

Step "Static and build checks"
$checks += Run-Check "python_core_compile" {
  python -m py_compile `
    src/xsci/kaggle.py `
    src/xsci/kaggle_conversation.py `
    src/xsci/kaggle_actions.py `
    src/xsci/kaggle_competitions.py `
    src/xsci/config.py `
    src/xsci/kaggle_session.py `
    src/xsci/kaggle_intent.py `
    scripts/verify_new_user_release_readiness.py `
    scripts/hpc_socks_bridge.py `
    scripts/start_hpc_socks_bridge.py `
    scripts/verify_hpc_socks_gateway.py
}
$checks += Run-Check "powershell_script_parse" {
  $scripts = @(
    "install.ps1",
    "scripts\quick_setup.ps1",
    "scripts\install_autokaggle_cli.ps1",
    "scripts\start_verified_workstation.ps1",
    "scripts\restart_workstation_frontend.ps1",
    "scripts\dpapi_credential_store.ps1",
    "scripts\manage_deepseek_secret.ps1",
    "scripts\manage_kaggle_secret.ps1",
    "scripts\manage_hpc_ssh_secret.ps1",
    "scripts\manage_hpc_proxy_bridge.ps1",
    "scripts\open_hpc_browser.ps1",
    "scripts\run_new_user_release_acceptance.ps1"
  )
  foreach ($rel in $scripts) {
    $path = Join-Path $Root $rel
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors) | Out-Null
    if ($errors -and $errors.Count -gt 0) {
      $message = ($errors | Select-Object -First 3 | ForEach-Object { "${rel}:$($_.Message)" }) -join "; "
      throw $message
    }
  }
}
$checks += Run-Check "installer_smoke_no_secrets" {
  powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 -SkipBuild -SkipNpmInstall -SkipSecretPrompt -SkipVerify
}
$checks += Run-Check "cli_tests" {
  python -m pytest tests/test_kaggle_menu.py tests/test_autokaggle_cli.py tests/test_xsci_cli.py tests/test_xsci_phase2.py tests/test_xsci_phase3.py tests/test_kaggle_stream.py -q
}
$checks += Run-Check "frontend_typecheck" {
  Push-Location $Web
  try { npm run typecheck } finally { Pop-Location }
}
if (-not $SkipBuild) {
  $checks += Run-Check "frontend_build" {
    Push-Location $Web
    try { npm run build } finally { Pop-Location }
  }
}

Step "Start and live EvoMind gateway checks"
$checks += Run-Check "start_production_workstation_frontend" {
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\restart_workstation_frontend.ps1 -Port $Port -Mode production
}
$checks += Run-Check "workstation_launch_readiness" {
  python scripts\verify_workstation_launch_readiness.py --write-report --base-url $BaseUrl --fresh-install
}
$checks += Run-Check "new_user_release_readiness_live" {
  python scripts\verify_new_user_release_readiness.py --write-report --require-live-server --base-url $BaseUrl
}

if (-not $SkipBrowserSmoke) {
  Step "Browser and interaction checks"
  $checks += Run-Check "browser_render_smoke" {
    python scripts\verify_workstation_browser_render_smoke.py --write-report --base-url $BaseUrl
  }
  $checks += Run-Check "click_smoke" {
    node scripts\verify_workstation_click_smoke.mjs --write-report --base-url $BaseUrl
  }
  $checks += Run-Check "interactive_controls" {
    node scripts\verify_workstation_interactive_controls.mjs --write-report --base-url $BaseUrl
  }
}

Step "Security and CLI routing"
$checks += Run-Check "secret_scan" {
  python scripts\verify_no_plaintext_secrets.py
}
$checks += Run-Check "cli_routing" {
  $shim = Join-Path $env:USERPROFILE ".xsci\bin"
  $env:Path = "$shim;$env:Path"
  $where = where.exe evomind
  if (-not $where -or -not ($where[0] -like "*\.xsci\bin\evomind*")) {
    throw "evomind command is not routed through %USERPROFILE%\.xsci\bin"
  }
  evomind --help | Out-Null
  evomind ready | Out-Null
  evomind official --help | Out-Null
}

$checkArray = @($checks)
$failed = @($checkArray | Where-Object { -not $_.ok })
$failedIds = @($failed | ForEach-Object { $_.id })
$status = if ($failed.Count -eq 0) { "passed" } else { "failed" }
$summary = [ordered]@{
  schema = "xcientist.new_user_release_acceptance.v1"
  created_at = (Get-Date).ToString("s")
  status = $status
  default_gateway = "$BaseUrl/?page=control"
  failed_checks = $failedIds
  optional_training_blockers = @("gpu_resource_blocked", "deepseek_cache_below_80_for_batch_generation")
  claim_boundary = "This validates new-user EvoMind gateway release. It does not validate Kaggle training, official submission, rank, medal, or MLE-Bench-75 performance."
  checks = $checkArray
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ReportJson) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ReportMd) | Out-Null
$summary | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $ReportJson

$md = @()
$md += "# New User Release Acceptance"
$md += ""
$md += "- status: ``$status``"
$md += "- default_gateway: $BaseUrl/?page=control"
$md += "- failed_checks: ``$(@($summary.failed_checks) -join ', ')``"
$md += "- optional_training_blockers: ``$(@($summary.optional_training_blockers) -join ', ')``"
$md += ""
$md += "## Checks"
$md += ""
$md += "| id | ok | seconds | error |"
$md += "| --- | --- | ---: | --- |"
foreach ($check in $checks) {
  $err = if ($check.error) { $check.error.Replace("|", "/") } else { "" }
  $md += "| ``$($check.id)`` | ``$($check.ok)`` | $($check.seconds) | $err |"
}
$md += ""
$md += "## Claim Boundary"
$md += ""
$md += $summary.claim_boundary
$md += ""
$md | Set-Content -Encoding UTF8 $ReportMd

Write-Host ""
Write-Host "================================================================" -ForegroundColor $(if ($status -eq "passed") { "Green" } else { "Red" })
Write-Host "New-user release acceptance: $status" -ForegroundColor $(if ($status -eq "passed") { "Green" } else { "Red" })
Write-Host "Report: $ReportMd"
Write-Host "================================================================" -ForegroundColor $(if ($status -eq "passed") { "Green" } else { "Red" })

if ($status -ne "passed") {
  exit 1
}
