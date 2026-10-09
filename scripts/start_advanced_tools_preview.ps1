param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion.Major -lt 7) { throw '请在 PowerShell 7 中运行本机预览脚本。此脚本不会自动安装或升级环境。' }
$repo = Split-Path -Parent $PSScriptRoot
$web = Join-Path $repo 'web/research-agent-workstation'
$preview = Join-Path $web 'preview/advanced-tools'
$node = 'D:/下载/node.exe'
if (-not (Test-Path -LiteralPath $node)) { throw '请先核验 Node 路径。' }
if ((& $node --version) -ne 'v24.13.0') { throw 'Node 版本与计划不一致。' }
$next = Join-Path $web 'node_modules/next/dist/bin/next'
if (-not (Test-Path -LiteralPath $next)) { throw '未找到已安装依赖；此脚本不会自动安装。' }
if ($CheckOnly) { Write-Output 'Preview launcher prerequisites OK'; exit 0 }
# Sanitize the CHILD environment only; never alter the caller's shell credentials.
if (Get-NetTCPConnection -State Listen -LocalPort 8098 -ErrorAction SilentlyContinue) { throw '8098 已占用；请核验现有进程，不要直接结束其他服务。' }
$start = New-Object System.Diagnostics.ProcessStartInfo
$start.FileName = $node
$start.Arguments = '"' + $next + '" dev "' + $preview + '" --webpack --hostname 127.0.0.1 --port 8098'
$start.WorkingDirectory = $preview
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
@($start.EnvironmentVariables.Keys) | Where-Object { $_ -match '(TOKEN|SECRET|PASSWORD|API_KEY|DATABASE_URL|WORKSTATION|EVOMIND|ANTHROPIC|OPENAI|DEEPSEEK|HPC|KAGGLE|CLAUDE)' } | ForEach-Object { $start.EnvironmentVariables.Remove($_) }
$start.EnvironmentVariables['EVOMIND_USABILITY_PREVIEW'] = '1'
$start.EnvironmentVariables['NEXT_TELEMETRY_DISABLED'] = '1'
$start.EnvironmentVariables['DATABASE_URL'] = 'file:./unused-preview.db'
$start.EnvironmentVariables['WORKSTATION_ROOT'] = $preview
$start.EnvironmentVariables['WORKSTATION_DATA_DIR'] = Join-Path $preview '.next/isolated-data'
$start.EnvironmentVariables['EVOMIND_TEST_FIXTURE_ROOT'] = $start.EnvironmentVariables['WORKSTATION_DATA_DIR']
$start.EnvironmentVariables['WORKSTATION_SESSION_SECRET'] = [guid]::NewGuid().ToString('N')
$previewProcess = [System.Diagnostics.Process]::Start($start)
try { $previewProcess.WaitForExit(); exit $previewProcess.ExitCode }
finally {
  if (-not $previewProcess.HasExited) { $previewProcess.Kill($true) }
  $previewProcess.Dispose()
}
