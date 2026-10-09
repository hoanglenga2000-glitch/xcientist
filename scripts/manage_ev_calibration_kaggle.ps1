[CmdletBinding()]
param(
  [ValidateSet('install', 'probe', 'download', 'transfer')][string]$Action = 'probe',
  [switch]$SecretFromStdin
)
$ErrorActionPreference = 'Stop'
$campaignSecret = $null
$storeLock = $null
try {
  . (Join-Path $PSScriptRoot 'dpapi_credential_store.ps1')
  $storeLock = Enter-EvoMindCredentialStoreLock
  try {
    $stateDir = Initialize-EvoMindCredentialStateDirectory
    $secretPath = Join-Path $stateDir 'kaggle_ev_calibration_20260908.xml'
    if ($Action -eq 'install') {
      if (Test-Path -LiteralPath $secretPath) { throw 'Campaign credential already exists; refusing overwrite.' }
      $campaignSecret = Read-EvoMindSecureInput -Prompt 'Kaggle API token' -FromStdin:$SecretFromStdin
      $credential = [System.Management.Automation.PSCredential]::new('__KAGGLE_API_TOKEN__', $campaignSecret)
      $temporary = Write-EvoMindCredentialTemp -Credential $credential -StateDirectory $stateDir -Prefix 'ev-calibration-kaggle'
      Commit-EvoMindCredentialFiles @([ordered]@{TemporaryPath=$temporary;DestinationPath=$secretPath})
      Write-Output '{"status":"stored","storage":"windows_dpapi_current_user","scope":"ev_public_calibration_20260908","existing_global_credential_unchanged":true}'
      exit 0
    }
    Assert-EvoMindCredentialDestination $secretPath
    Protect-EvoMindCredentialPath -Path $secretPath
    $credential = Import-Clixml -LiteralPath $secretPath
    if ($credential -isnot [System.Management.Automation.PSCredential] -or $credential.UserName -ne '__KAGGLE_API_TOKEN__') { throw 'Invalid credential structure' }
  } finally {
    if ($null -ne $storeLock) { Exit-EvoMindCredentialStoreLock $storeLock; $storeLock = $null }
    if ($null -ne $campaignSecret) { $campaignSecret.Dispose(); $campaignSecret = $null }
  }
  $oldToken = $env:KAGGLE_API_TOKEN
  $oldUser = $env:KAGGLE_USERNAME
  $oldKey = $env:KAGGLE_KEY
  try {
    $env:KAGGLE_API_TOKEN = $credential.GetNetworkCredential().Password
    Remove-Item Env:KAGGLE_USERNAME -ErrorAction SilentlyContinue
    Remove-Item Env:KAGGLE_KEY -ErrorAction SilentlyContinue
    $python = Join-Path (Split-Path $PSScriptRoot) '.venv\Scripts\python.exe'
    if($Action -eq 'transfer'){
      & $python (Join-Path $PSScriptRoot 'transfer_ev_kaggle_secret.py')
    }else{
      & $python (Join-Path $PSScriptRoot 'probe_ev_kaggle_access.py') --action $Action
    }
    $probeExit = $LASTEXITCODE
  } finally {
    if ($null -eq $oldToken) { Remove-Item Env:KAGGLE_API_TOKEN -ErrorAction SilentlyContinue } else { $env:KAGGLE_API_TOKEN = $oldToken }
    if ($null -eq $oldUser) { Remove-Item Env:KAGGLE_USERNAME -ErrorAction SilentlyContinue } else { $env:KAGGLE_USERNAME = $oldUser }
    if ($null -eq $oldKey) { Remove-Item Env:KAGGLE_KEY -ErrorAction SilentlyContinue } else { $env:KAGGLE_KEY = $oldKey }
  }
  exit $probeExit
} catch {
  [pscustomobject]@{status='failed';error_type=$_.Exception.GetType().Name;secret_logged=$false} | ConvertTo-Json -Compress
  exit 1
}
