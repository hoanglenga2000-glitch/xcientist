@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Write-Output $env:GPU_SSH_PASSWORD"
