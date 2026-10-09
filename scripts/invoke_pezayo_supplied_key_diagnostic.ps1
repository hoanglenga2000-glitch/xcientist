param(
    [Parameter(Mandatory=$true)][string]$SessionPath,
    [Parameter(Mandatory=$true)][string]$RemoteArguments
)
$ErrorActionPreference = 'Stop'
# Only read the explicitly selected current conversation's user connection object.
# Neither credentials nor raw conversation text is emitted or persisted here.
$providedKey = $null
$sessionStream = [IO.FileStream]::new((Resolve-Path -LiteralPath $SessionPath).Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
$sessionReader = [IO.StreamReader]::new($sessionStream)
try { while ($null -ne ($sessionLine = $sessionReader.ReadLine())) {
    if (-not $sessionLine.Contains('newapi')) { continue }
    try { $record = $sessionLine | ConvertFrom-Json } catch { continue }
    if ($record.type -ne 'response_item' -or $record.payload.role -ne 'user') { continue }
    foreach ($part in $record.payload.content) {
        $messageText = [string]$part.text
        if ($messageText -notmatch 'newapi' -or $messageText -notmatch 'api\.pezayo\.com') { continue }
        $keyMatch = [regex]::Match($messageText, '"key"\s*:\s*"(sk-[A-Za-z0-9_-]{20,200})"')
        if ($keyMatch.Success) { $providedKey = $keyMatch.Groups[1].Value }
    }
} } finally { $sessionReader.Dispose() }
if (-not $providedKey) { throw 'No unambiguous user-supplied connection found in selected conversation' }
if ($RemoteArguments -notmatch '^--case (inventory|simple1|simple7|actual1|actual7) ' -or $RemoteArguments -match '[";&|<>`\r\n]') {
    throw 'Unsafe diagnostic arguments'
}
$processInfo = New-Object System.Diagnostics.ProcessStartInfo
$processInfo.FileName = 'ssh.exe'
$processInfo.Arguments = '-o BatchMode=yes -o ConnectTimeout=12 evomind-shanghai C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv/Scripts/python.exe -X utf8 C:/ProgramData/EvoMind/staging/system-completeness-20260907/diagnose_pezayo_request_matrix.py --credential-stdin ' + $RemoteArguments
$processInfo.UseShellExecute = $false
$processInfo.CreateNoWindow = $true
$processInfo.RedirectStandardInput = $true
$diagnosticProcess = [Diagnostics.Process]::Start($processInfo)
$diagnosticProcess.StandardInput.WriteLine((@{key=$providedKey} | ConvertTo-Json -Compress))
$diagnosticProcess.StandardInput.Close()
$providedKey = $null
$diagnosticProcess.WaitForExit()
exit $diagnosticProcess.ExitCode
