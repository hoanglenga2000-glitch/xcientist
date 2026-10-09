[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$StopSignal,
    [int]$IntervalMilliseconds = 200
)

$ErrorActionPreference = "Stop"
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class EvoMindForegroundWindow {
    [DllImport("user32.dll")]
    public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
}
'@

$stop = [IO.Path]::GetFullPath($StopSignal)
while (-not (Test-Path -LiteralPath $stop)) {
    $chrome = Get-Process chrome -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -like "*EvoMind*" } |
        Select-Object -First 1
    if ($chrome) {
        [void][EvoMindForegroundWindow]::ShowWindowAsync($chrome.MainWindowHandle, 3)
        [void][EvoMindForegroundWindow]::SetForegroundWindow($chrome.MainWindowHandle)
    }
    Start-Sleep -Milliseconds ([Math]::Max(100, $IntervalMilliseconds))
}
