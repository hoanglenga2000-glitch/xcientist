@echo off
powershell.exe -NoProfile -NonInteractive -Command "$c=Import-Clixml -LiteralPath (Join-Path $env:APPDATA 'ResearchAgentWorkstation\hpc_ssh_credential.xml'); [Console]::Out.Write($c.GetNetworkCredential().Password)"
