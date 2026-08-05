param([int]$Port = 8088, [string]$HostName = "127.0.0.1")
& (Join-Path $PSScriptRoot "scripts\manage_workstation_lifecycle.ps1") status -InstallRoot $PSScriptRoot -Port $Port -HostName $HostName
exit $LASTEXITCODE
