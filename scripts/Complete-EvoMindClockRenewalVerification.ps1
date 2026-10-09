param([Parameter(Mandatory=$true)][string]$StageRoot,[Parameter(Mandatory=$true)][string]$ExpectedBuild)
$ErrorActionPreference='Stop'
$maintenance='C:/ProgramData/EvoMind/maintenance'
$watchdog=Join-Path $maintenance 'Watch-EvoMindNode.ps1'
$helper=Join-Path $maintenance 'Renew-EvoMindServiceClock.ps1'
if ((Get-FileHash -LiteralPath $watchdog).Hash -ne '9e07b262e6e70be73a17cdf784606080782d8c17cc843b418753015b02583c38' -or (Get-FileHash -LiteralPath $helper).Hash -ne '858270e4dfec8ff3e0c9c5432d8747511dcc4dc6df15e0436ad12053df631d8d') {throw 'INSTALLED_CLOCK_HASH_MISMATCH'}
$health=Invoke-RestMethod -Uri 'http://127.0.0.1:8088/api/healthz' -TimeoutSec 10
if ($health.build_id -ne $ExpectedBuild -or $health.status -ne 'ready') {throw 'BUILD_NOT_READY'}
$before=@(Get-NetTCPConnection -State Listen -LocalPort 8088,8765 | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique)
$renewal=& $helper -Apply -MinimumAgeHours 0
if ($renewal.status -ne 'renewed') {throw 'CLOCK_RENEWAL_NOT_VERIFIED'}
$clockText=& 'C:/Program Files/nodejs/node.exe' (Join-Path $StageRoot 'Verify-EvoMindServiceClock.cjs') $ExpectedBuild
if ($LASTEXITCODE -ne 0) {throw 'COMPILED_CLOCK_CHECK_FAILED'}
$clock=$clockText | ConvertFrom-Json
if ($clock.status -ne 'passed' -or $clock.utc_drift_ms -gt 5000) {throw 'CLOCK_CONSUMER_INVALID'}
$after=@(Get-NetTCPConnection -State Listen -LocalPort 8088,8765 | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique)
if (($before -join ',') -ne ($after -join ',')) {throw 'CLOCK_RENEWAL_RESTARTED_SERVICE'}
$scheduler=New-Object -ComObject 'Schedule.Service'; $scheduler.Connect()
$tasks=@()
foreach ($task in $scheduler.GetFolder('\').GetTasks(1)) {foreach ($action in $task.Definition.Actions) {if ([string]$action.Arguments -like '*Watch-EvoMindNode.ps1*' -and [string]$action.Arguments -like '*-Repair*' -and $task.Enabled) {$tasks += [pscustomobject]@{name=$task.Name;enabled=[bool]$task.Enabled;last_run=$task.LastRunTime.ToUniversalTime().ToString('o');last_result=$task.LastTaskResult;next_run=$task.NextRunTime.ToUniversalTime().ToString('o')}}}}
if ($tasks.Count -ne 1) {throw 'RECURRING_WATCHDOG_NOT_UNAMBIGUOUS'}
$receipt=[ordered]@{status='passed';build_id=$ExpectedBuild;renewal=$renewal;clock=$clock;listener_pids_unchanged=$true;listener_pids=$after;scheduled_watchdog=$tasks;http_probe='unauthenticated_endpoint_requires_session';initial_installer='files_installed_renewal_succeeded_http_check_returned_401';checked_at_utc=[DateTimeOffset]::UtcNow.ToString('o')}
[IO.File]::WriteAllText((Join-Path $StageRoot 'clock-renewal-installation.json'),($receipt | ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
$receipt | ConvertTo-Json -Depth 6 -Compress
