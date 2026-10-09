param(
  [string]$AnchorPath = 'C:\ProgramData\EvoMind\state\service-clock-anchor.json',
  [ValidateRange(0,20)][double]$MinimumAgeHours = 12,
  [switch]$Apply
)
$ErrorActionPreference = 'Stop'
$clockMutex = New-Object System.Threading.Mutex($false, 'Global\EvoMind-Byoa-V12-Deployment')
$clockHeld = $false
$temporaryClock = $null
try {
  try { $clockHeld = $clockMutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $clockHeld = $true }
  if (-not $clockHeld) { return [pscustomobject]@{ status='deferred_deployment'; renewed=$false } }
  $clockFile = Get-Item -LiteralPath $AnchorPath
  if ($clockFile.Length -gt 16384 -or ($clockFile.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'SERVICE_CLOCK_FILE_INVALID' }
  $anchor = Get-Content -LiteralPath $AnchorPath -Raw | ConvertFrom-Json
  $bootText = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')
  $hasher = [Security.Cryptography.SHA256]::Create()
  try { $currentBoot = ([BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($bootText)))).Replace('-','').ToLowerInvariant() } finally { $hasher.Dispose() }
  $frequency = [Diagnostics.Stopwatch]::Frequency
  $sampleQpc = [Diagnostics.Stopwatch]::GetTimestamp()
  $sampleEpoch = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
  if ($anchor.schema -ne 'evomind.service_clock_anchor.v1' -or
      $anchor.boot_id -ne $currentBoot -or
      [string]$anchor.service_instance_id -notmatch '^[0-9a-f-]{36}$' -or
      $anchor.web_monotonic_origin -ne 'system_qpc' -or
      [long]$anchor.qpc_frequency -ne $frequency -or
      [long]$anchor.qpc_timestamp -le 0 -or [long]$anchor.qpc_timestamp -gt $sampleQpc -or
      [DateTimeOffset]::Parse([string]$anchor.created_at_utc).ToUnixTimeMilliseconds() -ne [long]$anchor.windows_utc_epoch_ms) { throw 'SERVICE_CLOCK_IDENTITY_INVALID' }
  $expectedEpoch = [double]$anchor.windows_utc_epoch_ms + ($sampleQpc - [long]$anchor.qpc_timestamp) * 1000.0 / $frequency
  $driftMs = [Math]::Abs($sampleEpoch - $expectedEpoch)
  if ($driftMs -gt 5000) { throw 'SERVICE_CLOCK_RENEWAL_DISCONTINUITY' }
  $previous = $anchor
  if ($anchor.PSObject.Properties.Name -contains 'renewal') {
    $previous = $anchor.renewal
    if (-not $previous -or $previous.schema -ne $anchor.schema -or
        $previous.service_instance_id -ne $anchor.service_instance_id -or
        $previous.boot_id -ne $anchor.boot_id -or $previous.web_monotonic_origin -ne 'system_qpc' -or
        [long]$previous.qpc_frequency -ne $frequency -or
        [long]$previous.qpc_timestamp -le [long]$anchor.qpc_timestamp -or
        [long]$previous.qpc_timestamp -gt $sampleQpc -or
        ($previous.PSObject.Properties.Name -contains 'renewal') -or
        [DateTimeOffset]::Parse([string]$previous.created_at_utc).ToUnixTimeMilliseconds() -ne [long]$previous.windows_utc_epoch_ms) { throw 'SERVICE_CLOCK_RENEWAL_INVALID' }
    $previousExpected = [double]$anchor.windows_utc_epoch_ms + ([long]$previous.qpc_timestamp - [long]$anchor.qpc_timestamp) * 1000.0 / $frequency
    if ([Math]::Abs([double]$previous.windows_utc_epoch_ms - $previousExpected) -gt 5000) { throw 'SERVICE_CLOCK_RENEWAL_DISCONTINUITY' }
  }
  $ageHours = ($sampleQpc - [long]$previous.qpc_timestamp) / [double]$frequency / 3600.0
  if ($ageHours -lt $MinimumAgeHours) { return [pscustomobject]@{ status='fresh'; renewed=$false; age_hours=$ageHours; drift_ms=$driftMs } }
  if (-not $Apply) { return [pscustomobject]@{ status='renewal_due'; renewed=$false; age_hours=$ageHours; drift_ms=$driftMs } }
  $renewal = [ordered]@{schema=$anchor.schema;service_instance_id=$anchor.service_instance_id;boot_id=$anchor.boot_id;windows_utc_epoch_ms=$sampleEpoch;qpc_timestamp=$sampleQpc;qpc_frequency=$frequency;created_at_utc=[DateTimeOffset]::FromUnixTimeMilliseconds($sampleEpoch).ToString('o');web_monotonic_origin='system_qpc'}
  $anchor | Add-Member -NotePropertyName renewal -NotePropertyValue $renewal -Force
  $temporaryClock = $AnchorPath + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
  [IO.File]::WriteAllText($temporaryClock, ($anchor | ConvertTo-Json -Depth 5) + "`n", [Text.UTF8Encoding]::new($false))
  Set-Acl -LiteralPath $temporaryClock -AclObject (Get-Acl -LiteralPath $AnchorPath)
  $previousClockPath = $AnchorPath + '.before-renewal-' + [guid]::NewGuid().ToString('N') + '.json'
  [IO.File]::Replace($temporaryClock, $AnchorPath, $previousClockPath)
  $temporaryClock = $null
  [pscustomobject]@{ status='renewed'; renewed=$true; age_hours=0; drift_ms=$driftMs; processes_restarted=$false }
} finally {
  if ($temporaryClock -and (Test-Path -LiteralPath $temporaryClock)) { Remove-Item -LiteralPath $temporaryClock }
  if ($clockHeld) { $clockMutex.ReleaseMutex() }
  $clockMutex.Dispose()
}
