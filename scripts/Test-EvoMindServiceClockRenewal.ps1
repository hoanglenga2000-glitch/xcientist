param([Parameter(Mandatory=$true)][string]$StageRoot)
$ErrorActionPreference = 'Stop'
$helper = Join-Path $StageRoot 'Renew-EvoMindServiceClock.ps1'
$fixturePath = Join-Path $StageRoot ('clock-fixture-' + [guid]::NewGuid().ToString('N') + '.json')
$baseAnchor = Get-Content -LiteralPath 'C:\ProgramData\EvoMind\state\service-clock-anchor.json' -Raw | ConvertFrom-Json
$baseAnchor.PSObject.Properties.Remove('renewal')
$deltaMs = 25 * 3600 * 1000L
$baseAnchor.qpc_timestamp = [Diagnostics.Stopwatch]::GetTimestamp() - [long]($deltaMs * [Diagnostics.Stopwatch]::Frequency / 1000)
$baseAnchor.windows_utc_epoch_ms = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() - $deltaMs
$baseAnchor.created_at_utc = [DateTimeOffset]::FromUnixTimeMilliseconds($baseAnchor.windows_utc_epoch_ms).ToString('o')
function Write-Fixture($value) { [IO.File]::WriteAllText($fixturePath, ($value | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false)) }
Write-Fixture $baseAnchor
$before = Get-Content -LiteralPath $fixturePath -Raw | ConvertFrom-Json
$probe = & $helper -AnchorPath $fixturePath
if ($probe.status -ne 'renewal_due') { throw 'EXPECTED_STALE_FIXTURE' }
$result = & $helper -AnchorPath $fixturePath -Apply
if ($result.status -ne 'renewed') { throw 'FIXTURE_RENEWAL_FAILED' }
$after = Get-Content -LiteralPath $fixturePath -Raw | ConvertFrom-Json
foreach ($name in $before.PSObject.Properties.Name) { if ($before.$name -ne $after.$name) { throw 'ORIGINAL_ANCHOR_CHANGED' } }
if (-not $after.renewal -or $result.processes_restarted) { throw 'RENEWAL_RECEIPT_INVALID' }
$fresh = & $helper -AnchorPath $fixturePath -Apply
if ($fresh.status -ne 'fresh') { throw 'FRESH_ANCHOR_NOT_IDEMPOTENT' }
$rejected = 0
foreach ($case in @('boot','drift')) {
  $bad = $before | ConvertTo-Json -Depth 5 | ConvertFrom-Json
  if ($case -eq 'boot') { $bad.boot_id = 'wrong-boot' } else { $bad.windows_utc_epoch_ms += 60000; $bad.created_at_utc = [DateTimeOffset]::FromUnixTimeMilliseconds($bad.windows_utc_epoch_ms).ToString('o') }
  Write-Fixture $bad
  try { & $helper -AnchorPath $fixturePath -Apply | Out-Null } catch { $rejected++ }
}
if ($rejected -ne 2) { throw 'INVALID_CLOCK_NOT_REJECTED' }
Write-Fixture $after
[ordered]@{schema='evomind.clock_renewal_acceptance.v1';status='passed';expired_original_25h=$true;original_binding_unchanged=$true;fresh_idempotent=$true;boot_and_drift_rejected=$true;processes_restarted=$false;fixture_path=$fixturePath} | ConvertTo-Json -Compress
