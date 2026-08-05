@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%USERPROFILE%\.claude\launch-claude-code-deepseek.ps1" --resume
