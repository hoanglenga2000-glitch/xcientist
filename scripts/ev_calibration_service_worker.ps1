param([ValidateSet('data')][string]$Mode='data')
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$stage='C:\ProgramData\EvoMind\staging\ev-public-calibration-20260908'
$rawBytes=$null;$secure=$null
try{
  . 'C:\ProgramData\EvoMind\bundle\scripts\lib\Common.ps1'
  Assert-DedicatedUserContext
  Add-Type -AssemblyName System.Security
  $paths=Get-NodeSecretPaths
  $dest=Join-Path $paths.root 'kaggle_ev_calibration_20260908.xml'
  if(-not(Test-Path -LiteralPath $dest)){
    $envelope=Get-Content -Raw -LiteralPath "$stage\kaggle-key.machine-dpapi.json"|ConvertFrom-Json
    if($envelope.schema -ne 'evomind.ev_kaggle.machine_dpapi_transit.v1'){throw 'Envelope mismatch'}
    $rawBytes=[Security.Cryptography.ProtectedData]::Unprotect([Convert]::FromBase64String($envelope.ciphertext),$null,[Security.Cryptography.DataProtectionScope]::LocalMachine)
    $plain=[Text.Encoding]::UTF8.GetString($rawBytes)
    $secure=ConvertTo-SecureString $plain -AsPlainText -Force;$plain=$null
    New-DpapiCredentialFile -Path $dest -UserName '__KAGGLE_API_TOKEN__' -Secret $secure
  }
  $credential=Import-DpapiCredentialChecked -Path $dest -ExpectedUserName '__KAGGLE_API_TOKEN__'
  $env:EVOMIND_EV_KAGGLE_KEY=$credential.GetNetworkCredential().Password
  & 'C:\EvoMind\releases\664a636ddd419a66f73cc10820c0a16429784866\.venv\Scripts\python.exe' -X utf8 "$stage\ev_calibration_provision.py" 1>> "$stage\service-output\data.stdout.log" 2>> "$stage\service-output\data.stderr.log"
  exit $LASTEXITCODE
}catch{[IO.File]::WriteAllText("$stage\service-output\worker-error.json",(@{error_type=$_.Exception.GetType().Name;status='failed'}|ConvertTo-Json -Compress));exit 1}
finally{Remove-Item Env:EVOMIND_EV_KAGGLE_KEY -ErrorAction SilentlyContinue;if($null -ne $rawBytes){[Array]::Clear($rawBytes,0,$rawBytes.Length)};if($null -ne $secure){$secure.Dispose()};$plain=$null}
