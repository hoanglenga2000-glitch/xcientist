[CmdletBinding()]
param(
  [ValidateSet('install','catalog','test','code-retest','transfer')][string]$Action='catalog',
  [string]$Model='',
  [switch]$SecretFromStdin
)
$ErrorActionPreference='Stop'
$secretValue=$null
try {
  . (Join-Path $PSScriptRoot 'dpapi_credential_store.ps1')
  $storeLock=Enter-EvoMindCredentialStoreLock
  try {
    $stateDir=Initialize-EvoMindCredentialStateDirectory
    $secretPath=Join-Path $stateDir 'pezayo_ev_calibration_20260908.xml'
    if ($Action -eq 'install') {
      if(Test-Path -LiteralPath $secretPath){throw 'Campaign credential already exists'}
      $secretValue=Read-EvoMindSecureInput -Prompt 'Model gateway API key' -FromStdin:$SecretFromStdin
      $credential=[System.Management.Automation.PSCredential]::new('__PEZAYO_EV_API__',$secretValue)
      $temporary=Write-EvoMindCredentialTemp -Credential $credential -StateDirectory $stateDir -Prefix 'ev-model-key'
      Commit-EvoMindCredentialFiles @([ordered]@{TemporaryPath=$temporary;DestinationPath=$secretPath})
      Write-Output '{"status":"stored","storage":"windows_dpapi_current_user","existing_model_credentials_unchanged":true}'
      exit 0
    }
    Assert-EvoMindCredentialDestination $secretPath
    Protect-EvoMindCredentialPath -Path $secretPath
    $credential=Import-Clixml -LiteralPath $secretPath
    if($credential -isnot [System.Management.Automation.PSCredential] -or $credential.UserName -ne '__PEZAYO_EV_API__'){throw 'Invalid credential structure'}
  } finally {
    Exit-EvoMindCredentialStoreLock $storeLock
    if($null -ne $secretValue){$secretValue.Dispose()}
  }
  $previousKey=$env:EVOMIND_PROBE_MODEL_KEY
  try {
    $env:EVOMIND_PROBE_MODEL_KEY=$credential.GetNetworkCredential().Password
    $python=Join-Path (Split-Path $PSScriptRoot) '.venv\Scripts\python.exe'
    if($Action -eq 'transfer'){
      & $python (Join-Path $PSScriptRoot 'transfer_ev_deepseek_secret.py')
    }else{
      & $python (Join-Path $PSScriptRoot 'probe_ev_domestic_models.py') --action $Action --model $Model
    }
    $resultCode=$LASTEXITCODE
  } finally {
    if($null -eq $previousKey){Remove-Item Env:EVOMIND_PROBE_MODEL_KEY -ErrorAction SilentlyContinue}else{$env:EVOMIND_PROBE_MODEL_KEY=$previousKey}
  }
  exit $resultCode
} catch {
  [pscustomobject]@{status='failed';error_type=$_.Exception.GetType().Name;secret_logged=$false}|ConvertTo-Json -Compress
  exit 1
}
