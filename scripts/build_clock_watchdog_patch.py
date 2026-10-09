"""Patch the reviewed production watchdog, preserving unrelated local work."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
artifacts = root / 'artifacts/invitation-rebind-20260927'
source = (artifacts / 'Watch-EvoMindNode.production.ps1').read_text(encoding='utf-8-sig')
old = "$anchorPath = Join-Path $StateRoot 'service-clock-anchor.json'"
new = old + '''
$clockRenewal = $null
try {
  $renewalHelper = Join-Path $PSScriptRoot 'Renew-EvoMindServiceClock.ps1'
  $clockRenewal = & $renewalHelper -AnchorPath $anchorPath -Apply:$Repair
} catch { $clockRenewal = [pscustomobject]@{ status='renewal_rejected'; renewed=$false } }'''
assert source.count(old) == 1
source = source.replace(old, new)
source = source.replace('($now - $anchorCreated.UtcDateTime).TotalHours', '([DateTimeOffset]::UtcNow - $anchorCreated).TotalHours')
source = source.replace("if ($anchorDoc.schema -eq 'evomind.service_clock_anchor.v1' -and $anchorAgeHours -ge 0 -and $anchorAgeHours -lt 20) {", "if ($clockRenewal -and $clockRenewal.status -in @('fresh','renewed','renewal_due') -and $anchorDoc.schema -eq 'evomind.service_clock_anchor.v1' -and $anchorAgeHours -ge 0 -and $anchorAgeHours -lt 20) {")
old = "$anchorCreated = [DateTimeOffset]::Parse([string]$anchorDoc.created_at_utc)"
new = '''$clockSample = $anchorDoc
    if ($anchorDoc.renewal) { $clockSample = $anchorDoc.renewal }
    $anchorCreated = [DateTimeOffset]::Parse([string]$clockSample.created_at_utc)'''
assert source.count(old) == 1
source = source.replace(old, new)
source = source.replace("$repairOutcome = 'not_attempted'", "$repairOutcome = if ($clockRenewal -and $clockRenewal.renewed) { 'clock_renewed_without_restart' } else { 'not_attempted' }")
source = source.replace("if (((-not $coreOk) -or (-not $anchorOk)) -and $Repair) {", "if ((-not $coreOk) -and $Repair) {")
old = 'Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop'
new = "& 'C:/SecureInput/Invoke-ServiceAccountAction.ps1' -Action Start | Out-Null"
assert source.count(old) == 1
source = source.replace(old, new)
source = source.replace('if ($coreOk) { exit 0 } else { exit 1 }', 'if ($coreOk -and $anchorOk) { exit 0 } else { exit 1 }')
old_lock = "try { $mutexHeld = -not $mutex.WaitOne(0) } finally { $mutex.Dispose() }"
new_lock = """$watchdogHasLock = $false
    try {
      try { $watchdogHasLock = $mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $watchdogHasLock = $true }
      $mutexHeld = -not $watchdogHasLock
    } finally { if ($watchdogHasLock) { $mutex.ReleaseMutex() }; $mutex.Dispose() }"""
assert source.count(old_lock) == 1
source = source.replace(old_lock, new_lock)
source = source.replace('$mutexHeld = $false', '$mutexHeld = $true')
start = source.index('# The managed service clock anchor expires')
end = source.index('$anchorPath =', start)
source = source[:start] + '# Renew the trusted QPC sample without interrupting active work.\n# The immutable launch anchor and its environment binding stay unchanged.\n' + source[end:]
(artifacts / 'Watch-EvoMindNode.candidate.ps1').write_text(source, encoding='utf-8')
print('watchdog_candidate_created')
