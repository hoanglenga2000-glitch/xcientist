function Assert-EvoMindWindowsDpapi {
  if ($env:OS -ne "Windows_NT") {
    throw [System.PlatformNotSupportedException]::new("Windows DPAPI is required.")
  }
}

function Enter-EvoMindCredentialStoreLock([int]$TimeoutMilliseconds = 120000) {
  Assert-EvoMindWindowsDpapi
  $sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
  $mutex = [System.Threading.Mutex]::new($false, "Local\EvoMindCredentialStore-$sid")
  $acquired = $false
  try {
    try {
      $acquired = $mutex.WaitOne($TimeoutMilliseconds)
    } catch [System.Threading.AbandonedMutexException] {
      $acquired = $true
    }
    if (-not $acquired) {
      throw [System.TimeoutException]::new("Credential store lock timed out.")
    }
    return $mutex
  } catch {
    $mutex.Dispose()
    throw
  }
}

function Exit-EvoMindCredentialStoreLock([System.Threading.Mutex]$Mutex) {
  if ($null -eq $Mutex) { return }
  try {
    $Mutex.ReleaseMutex()
  } finally {
    $Mutex.Dispose()
  }
}

function Assert-EvoMindNetworkHost([string]$Value, [switch]$AllowEmpty) {
  if ([string]::IsNullOrWhiteSpace($Value)) {
    if ($AllowEmpty) { return }
    throw [System.ArgumentException]::new("Host is required.")
  }
  if ($Value.Length -gt 253 -or $Value -match "[\r\n\s]") {
    throw [System.ArgumentException]::new("Host is invalid.")
  }
  $ipAddress = $null
  if ([System.Net.IPAddress]::TryParse($Value, [ref]$ipAddress)) {
    return
  }
  if ($Value -notmatch "^[A-Za-z0-9.-]+$") {
    throw [System.ArgumentException]::new("Host is invalid.")
  }
  foreach ($label in $Value.Split(".")) {
    if ($label.Length -lt 1 -or $label.Length -gt 63 -or $label.StartsWith("-") -or $label.EndsWith("-")) {
      throw [System.ArgumentException]::new("Host is invalid.")
    }
  }
}

function Assert-EvoMindRemoteWorkspace([string]$Value) {
  if ([string]::IsNullOrWhiteSpace($Value) -or $Value.Length -gt 1024 -or $Value -notmatch "^/[A-Za-z0-9._/-]+$") {
    throw [System.ArgumentException]::new("Remote workspace is invalid.")
  }
  $segments = @($Value.Split("/") | Where-Object { $_ -ne "" })
  if ($segments.Count -eq 0 -or @($segments | Where-Object { $_ -in @(".", "..") }).Count -gt 0) {
    throw [System.ArgumentException]::new("Remote workspace is invalid.")
  }
}

function Get-EvoMindSidValue([object]$Identity) {
  if ($Identity -is [System.Security.Principal.SecurityIdentifier]) {
    return $Identity.Value
  }
  if ($Identity -is [System.Security.Principal.IdentityReference]) {
    return $Identity.Translate([System.Security.Principal.SecurityIdentifier]).Value
  }
  $account = [System.Security.Principal.NTAccount]::new([string]$Identity)
  return $account.Translate([System.Security.Principal.SecurityIdentifier]).Value
}

function Get-EvoMindBuiltinAdministratorsSid {
  return [System.Security.Principal.SecurityIdentifier]::new(
    [System.Security.Principal.WellKnownSidType]::BuiltinAdministratorsSid,
    $null
  )
}

function Test-EvoMindCurrentUserIsAdministrator {
  $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
  $principal = [System.Security.Principal.WindowsPrincipal]::new($identity)
  if ($principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) {
    return $true
  }
  $administratorSid = (Get-EvoMindBuiltinAdministratorsSid).Value
  return @($identity.Groups | ForEach-Object { Get-EvoMindSidValue $_ }) -contains $administratorSid
}

function Set-EvoMindCredentialPathOwnerToCurrentUser([string]$Path) {
  $currentSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
  $acl = Get-Acl -LiteralPath $Path -ErrorAction Stop
  $acl.SetOwner($currentSid)
  Set-Acl -LiteralPath $Path -AclObject $acl -ErrorAction Stop
  $ownerSid = Get-EvoMindSidValue (Get-Acl -LiteralPath $Path -ErrorAction Stop).Owner
  if ($ownerSid -cne $currentSid.Value) {
    throw [System.UnauthorizedAccessException]::new("Credential path owner repair failed.")
  }
}

function Repair-EvoMindAdministratorOwnedCredentialPath([string]$Path, [string]$OwnerSid) {
  $administratorSid = (Get-EvoMindBuiltinAdministratorsSid).Value
  if ($OwnerSid -cne $administratorSid -or -not (Test-EvoMindCurrentUserIsAdministrator)) {
    return $false
  }
  Set-EvoMindCredentialPathOwnerToCurrentUser $Path
  return $true
}

function Protect-EvoMindCredentialPath {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [switch]$Directory
  )

  $currentSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
  $systemSid = [System.Security.Principal.SecurityIdentifier]::new(
    [System.Security.Principal.WellKnownSidType]::LocalSystemSid,
    $null
  )
  $allowed = @($currentSid.Value, $systemSid.Value)
  $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
  if ($Directory -and -not $item.PSIsContainer) {
    throw [System.IO.IOException]::new("Credential state path is not a directory.")
  }
  if (-not $Directory -and $item.PSIsContainer) {
    throw [System.IO.IOException]::new("Credential destination is not a file.")
  }
  if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw [System.IO.IOException]::new("Credential path cannot be a reparse point.")
  }

  $initialAcl = Get-Acl -LiteralPath $Path -ErrorAction Stop
  $ownerSid = Get-EvoMindSidValue $initialAcl.Owner
  if ($ownerSid -notin $allowed) {
    if (-not (Repair-EvoMindAdministratorOwnedCredentialPath -Path $Path -OwnerSid $ownerSid)) {
      throw [System.UnauthorizedAccessException]::new("Credential path owner is not trusted.")
    }
  }

  & icacls.exe $Path /inheritance:r *> $null
  if ($LASTEXITCODE -ne 0) {
    throw [System.UnauthorizedAccessException]::new("Failed to disable credential ACL inheritance.")
  }

  $acl = Get-Acl -LiteralPath $Path -ErrorAction Stop
  $unexpectedSids = @(
    $acl.Access |
      ForEach-Object { Get-EvoMindSidValue $_.IdentityReference } |
      Where-Object { $_ -notin $allowed } |
      Sort-Object -Unique
  )
  foreach ($sid in $unexpectedSids) {
    & icacls.exe $Path /remove "*$sid" *> $null
    if ($LASTEXITCODE -ne 0) {
      throw [System.UnauthorizedAccessException]::new("Failed to remove an unexpected credential ACL entry.")
    }
  }

  $currentGrant = if ($Directory) { "*$($currentSid.Value):(OI)(CI)(F)" } else { "*$($currentSid.Value):(F)" }
  $systemGrant = if ($Directory) { "*$($systemSid.Value):(OI)(CI)(F)" } else { "*$($systemSid.Value):(F)" }
  & icacls.exe $Path /grant:r $currentGrant $systemGrant *> $null
  if ($LASTEXITCODE -ne 0) {
    throw [System.UnauthorizedAccessException]::new("Failed to set the credential ACL.")
  }

  $verifiedAcl = Get-Acl -LiteralPath $Path -ErrorAction Stop
  if ((Get-EvoMindSidValue $verifiedAcl.Owner) -notin $allowed) {
    throw [System.UnauthorizedAccessException]::new("Credential path owner verification failed.")
  }
  if (-not $verifiedAcl.AreAccessRulesProtected) {
    throw [System.UnauthorizedAccessException]::new("Credential ACL inheritance remains enabled.")
  }
  foreach ($rule in $verifiedAcl.Access) {
    $sid = Get-EvoMindSidValue $rule.IdentityReference
    if ($sid -notin $allowed -or $rule.AccessControlType -ne [System.Security.AccessControl.AccessControlType]::Allow) {
      throw [System.UnauthorizedAccessException]::new("Credential ACL verification failed.")
    }
  }
}

function New-EvoMindSecureDirectory([string]$Path) {
  $currentSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
  $systemSid = [System.Security.Principal.SecurityIdentifier]::new(
    [System.Security.Principal.WellKnownSidType]::LocalSystemSid,
    $null
  )
  $security = [System.Security.AccessControl.DirectorySecurity]::new()
  $security.SetOwner($currentSid)
  $security.SetAccessRuleProtection($true, $false)
  $inheritance = [System.Security.AccessControl.InheritanceFlags]::ContainerInherit -bor `
    [System.Security.AccessControl.InheritanceFlags]::ObjectInherit
  foreach ($sid in @($currentSid, $systemSid)) {
    $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
      $sid,
      [System.Security.AccessControl.FileSystemRights]::FullControl,
      $inheritance,
      [System.Security.AccessControl.PropagationFlags]::None,
      [System.Security.AccessControl.AccessControlType]::Allow
    )
    [void]$security.AddAccessRule($rule)
  }
  $directory = [System.IO.DirectoryInfo]::new($Path)
  if ($PSVersionTable.PSEdition -eq "Core") {
    [System.IO.FileSystemAclExtensions]::Create($directory, $security)
  } else {
    $directory.Create($security)
  }
}

function Assert-EvoMindNoReparseAncestors([string]$Path, [string]$Anchor) {
  $fullPath = [System.IO.Path]::GetFullPath($Path)
  $fullAnchor = [System.IO.Path]::GetFullPath($Anchor).TrimEnd("\")
  $pathWithSeparator = $fullPath.TrimEnd("\") + "\"
  $anchorWithSeparator = $fullAnchor + "\"
  if ($fullPath -ne $fullAnchor -and -not $pathWithSeparator.StartsWith($anchorWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw [System.UnauthorizedAccessException]::new("Credential state path escaped its trusted anchor.")
  }
  $cursor = $fullAnchor
  if (Test-Path -LiteralPath $cursor) {
    $anchorItem = Get-Item -LiteralPath $cursor -Force -ErrorAction Stop
    if (($anchorItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
      throw [System.IO.IOException]::new("Credential state anchor cannot be a reparse point.")
    }
  }
  $relative = $fullPath.Substring($fullAnchor.Length).TrimStart("\")
  foreach ($segment in $relative.Split("\")) {
    if ([string]::IsNullOrWhiteSpace($segment)) { continue }
    $cursor = Join-Path $cursor $segment
    if (Test-Path -LiteralPath $cursor) {
      $ancestor = Get-Item -LiteralPath $cursor -Force -ErrorAction Stop
      if (($ancestor.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw [System.IO.IOException]::new("Credential state cannot traverse a reparse point.")
      }
    }
  }
}

function Get-EvoMindCredentialStateDirectory {
  Assert-EvoMindWindowsDpapi
  $override = [string]$env:EVOMIND_DPAPI_STATE_DIR
  if (-not [string]::IsNullOrWhiteSpace($override)) {
    if ($env:EVOMIND_ALLOW_TEST_STATE_DIR -ne "1") {
      throw [System.UnauthorizedAccessException]::new("Credential state override is disabled.")
    }
    $candidate = [System.IO.Path]::GetFullPath($override)
    $tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()).TrimEnd("\") + "\"
    $candidateWithSeparator = $candidate.TrimEnd("\") + "\"
    if ($candidateWithSeparator -eq $tempRoot -or -not $candidateWithSeparator.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
      throw [System.UnauthorizedAccessException]::new("Test credential state must stay under the system temp directory.")
    }
    if ([System.IO.Path]::GetFileName($candidate) -notlike "evomind-dpapi-test-*") {
      throw [System.UnauthorizedAccessException]::new("Test credential state must use a dedicated EvoMind directory.")
    }
    Assert-EvoMindNoReparseAncestors $candidate $tempRoot.TrimEnd("\")
  } else {
    $applicationData = [System.Environment]::GetFolderPath(
      [System.Environment+SpecialFolder]::ApplicationData
    )
    if ([string]::IsNullOrWhiteSpace($applicationData)) {
      throw [System.IO.DirectoryNotFoundException]::new("Windows application data is unavailable.")
    }
    $applicationData = [System.IO.Path]::GetFullPath($applicationData)
    if ($applicationData.StartsWith("\\")) {
      throw [System.UnauthorizedAccessException]::new("Credential state must use a local Windows profile path.")
    }
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $applicationData "ResearchAgentWorkstation"))
    Assert-EvoMindNoReparseAncestors $candidate ([System.IO.Path]::GetPathRoot($candidate))
  }

  return $candidate
}

function Initialize-EvoMindCredentialStateDirectory {
  $candidate = Get-EvoMindCredentialStateDirectory

  if (Test-Path -LiteralPath $candidate) {
    $existing = Get-Item -LiteralPath $candidate -Force -ErrorAction Stop
    if (-not $existing.PSIsContainer) {
      throw [System.IO.IOException]::new("Credential state path is not a directory.")
    }
    if (($existing.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
      throw [System.IO.IOException]::new("Credential state directory cannot be a reparse point.")
    }
  } else {
    New-EvoMindSecureDirectory $candidate
  }
  Protect-EvoMindCredentialPath -Path $candidate -Directory
  return $candidate
}

function Assert-EvoMindCredentialDestination([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) {
    return
  }
  $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
  if ($item.PSIsContainer) {
    throw [System.IO.IOException]::new("Credential destination collides with a directory.")
  }
  if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw [System.IO.IOException]::new("Credential destination cannot be a reparse point.")
  }
  $currentSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
  $systemSid = [System.Security.Principal.SecurityIdentifier]::new(
    [System.Security.Principal.WellKnownSidType]::LocalSystemSid,
    $null
  ).Value
  $ownerSid = Get-EvoMindSidValue (Get-Acl -LiteralPath $Path -ErrorAction Stop).Owner
  if ($ownerSid -notin @($currentSid, $systemSid)) {
    if (-not (Repair-EvoMindAdministratorOwnedCredentialPath -Path $Path -OwnerSid $ownerSid)) {
      throw [System.UnauthorizedAccessException]::new("Credential destination owner is not trusted.")
    }
  }
}

function New-EvoMindCredentialTempPath([string]$StateDirectory, [string]$Prefix) {
  return Join-Path $StateDirectory (".{0}-{1}.tmp" -f $Prefix, [guid]::NewGuid().ToString("N"))
}

function Sync-EvoMindCredentialFile([string]$Path) {
  $stream = [System.IO.File]::Open(
    $Path,
    [System.IO.FileMode]::Open,
    [System.IO.FileAccess]::ReadWrite,
    [System.IO.FileShare]::Read
  )
  try {
    $stream.Flush($true)
  } finally {
    $stream.Dispose()
  }
}

function Write-EvoMindCredentialTemp {
  param(
    [Parameter(Mandatory = $true)]
    [System.Management.Automation.PSCredential]$Credential,
    [Parameter(Mandatory = $true)]
    [string]$StateDirectory,
    [Parameter(Mandatory = $true)]
    [string]$Prefix
  )
  $temporaryPath = New-EvoMindCredentialTempPath $StateDirectory $Prefix
  try {
    $Credential | Export-Clixml -LiteralPath $temporaryPath -Depth 3
    Protect-EvoMindCredentialPath -Path $temporaryPath
    Sync-EvoMindCredentialFile $temporaryPath
    return $temporaryPath
  } catch {
    Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
    throw
  }
}

function Write-EvoMindJsonTemp {
  param(
    [Parameter(Mandatory = $true)]
    [System.Collections.IDictionary]$Payload,
    [Parameter(Mandatory = $true)]
    [string]$StateDirectory,
    [Parameter(Mandatory = $true)]
    [string]$Prefix
  )
  $temporaryPath = New-EvoMindCredentialTempPath $StateDirectory $Prefix
  try {
    [System.IO.File]::WriteAllText(
      $temporaryPath,
      ($Payload | ConvertTo-Json -Depth 8),
      [System.Text.UTF8Encoding]::new($false)
    )
    Protect-EvoMindCredentialPath -Path $temporaryPath
    Sync-EvoMindCredentialFile $temporaryPath
    return $temporaryPath
  } catch {
    Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
    throw
  }
}

function Commit-EvoMindCredentialFiles {
  param(
    [Parameter(Mandatory = $true)]
    [object[]]$Entries
  )
  if ($Entries.Count -eq 0) {
    throw [System.ArgumentException]::new("No credential files were supplied for commit.")
  }

  $states = @()
  foreach ($entry in $Entries) {
    $temporaryPath = [string]$entry.TemporaryPath
    $destinationPath = [string]$entry.DestinationPath
    if (-not (Test-Path -LiteralPath $temporaryPath -PathType Leaf)) {
      throw [System.IO.FileNotFoundException]::new("Credential staging file is missing.")
    }
    Assert-EvoMindCredentialDestination $destinationPath
    $states += [ordered]@{
      TemporaryPath = $temporaryPath
      DestinationPath = $destinationPath
      HadExisting = (Test-Path -LiteralPath $destinationPath -PathType Leaf)
      BackupPath = New-EvoMindCredentialTempPath (Split-Path -Parent $destinationPath) "rollback"
      Committed = $false
    }
  }

  try {
    foreach ($state in $states) {
      if ($state.HadExisting) {
        [System.IO.File]::Replace(
          $state.TemporaryPath,
          $state.DestinationPath,
          $state.BackupPath,
          $true
        )
      } else {
        [System.IO.File]::Move($state.TemporaryPath, $state.DestinationPath)
      }
      $state.Committed = $true
      Protect-EvoMindCredentialPath -Path $state.DestinationPath
      Sync-EvoMindCredentialFile $state.DestinationPath
    }
  } catch {
    for ($index = $states.Count - 1; $index -ge 0; $index--) {
      $state = $states[$index]
      if (-not $state.Committed) {
        continue
      }
      try {
        if ($state.HadExisting -and (Test-Path -LiteralPath $state.BackupPath -PathType Leaf)) {
          if (Test-Path -LiteralPath $state.DestinationPath -PathType Leaf) {
            [System.IO.File]::Replace($state.BackupPath, $state.DestinationPath, $null, $true)
          } else {
            [System.IO.File]::Move($state.BackupPath, $state.DestinationPath)
          }
        } elseif (Test-Path -LiteralPath $state.DestinationPath -PathType Leaf) {
          Remove-Item -LiteralPath $state.DestinationPath -Force -ErrorAction Stop
        }
      } catch {
        # The outer operation remains failed; continue restoring other entries.
      }
    }
    throw
  } finally {
    foreach ($state in $states) {
      Remove-Item -LiteralPath $state.TemporaryPath -Force -ErrorAction SilentlyContinue
      Remove-Item -LiteralPath $state.BackupPath -Force -ErrorAction SilentlyContinue
    }
  }
}

function Remove-EvoMindCredentialFile([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) {
    return
  }
  Assert-EvoMindCredentialDestination $Path
  Remove-Item -LiteralPath $Path -Force -ErrorAction Stop
  if (Test-Path -LiteralPath $Path) {
    throw [System.IO.IOException]::new("Credential file removal did not complete.")
  }
}

function Read-EvoMindSecureInput {
  param(
    [System.Security.SecureString]$Provided,
    [Parameter(Mandatory = $true)]
    [string]$Prompt,
    [switch]$FromStdin
  )
  if ($null -ne $Provided) {
    return $Provided.Copy()
  }
  if ($FromStdin) {
    $standardInput = [Console]::OpenStandardInput()
    $reader = [System.IO.StreamReader]::new(
      $standardInput,
      [System.Text.UTF8Encoding]::new($false),
      $true,
      1024,
      $true
    )
    try {
      $plainValue = $reader.ReadLine()
      if ([string]::IsNullOrWhiteSpace($plainValue)) {
        throw [System.ArgumentException]::new("Credential input was empty.")
      }
      return ConvertTo-SecureString $plainValue -AsPlainText -Force
    } finally {
      $plainValue = $null
      $reader.Dispose()
    }
  }
  return Read-Host $Prompt -AsSecureString
}

function Get-EvoMindHpcCredentialStorePaths([string]$StateDirectory) {
  $fullStateDirectory = [System.IO.Path]::GetFullPath($StateDirectory)
  return [ordered]@{
    PointerPath = Join-Path $fullStateDirectory "hpc_ssh_current.json"
    GenerationsDirectory = Join-Path $fullStateDirectory "hpc_ssh_generations"
    LegacyCredentialPath = Join-Path $fullStateDirectory "hpc_ssh_credential.xml"
    LegacyMetadataPath = Join-Path $fullStateDirectory "hpc_ssh_metadata.json"
    TestHookPath = Join-Path $fullStateDirectory ".hpc-before-pointer-publish.ready"
  }
}

function Get-EvoMindRequiredObjectProperty {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Object,
    [Parameter(Mandatory = $true)]
    [string]$Name
  )
  if ($Object -is [System.Collections.IDictionary]) {
    if (-not $Object.Contains($Name)) {
      throw [System.IO.InvalidDataException]::new("Credential data is incomplete.")
    }
    return $Object[$Name]
  }
  if ($null -eq $Object -or -not ($Object.PSObject.Properties.Name -contains $Name)) {
    throw [System.IO.InvalidDataException]::new("Credential data is incomplete.")
  }
  return $Object.$Name
}

function Get-EvoMindFileSha256([string]$Path) {
  $stream = [System.IO.File]::Open(
    $Path,
    [System.IO.FileMode]::Open,
    [System.IO.FileAccess]::Read,
    [System.IO.FileShare]::Read
  )
  $sha256 = [System.Security.Cryptography.SHA256]::Create()
  try {
    $hash = $sha256.ComputeHash($stream)
    return ([System.BitConverter]::ToString($hash)).Replace("-", "").ToLowerInvariant()
  } finally {
    $sha256.Dispose()
    $stream.Dispose()
  }
}

function Read-EvoMindProtectedJson([string]$Path) {
  Assert-EvoMindCredentialDestination $Path
  Protect-EvoMindCredentialPath -Path $Path
  try {
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
  } catch {
    throw [System.IO.InvalidDataException]::new("Credential JSON is invalid.")
  }
}

function Import-EvoMindHpcCredential([string]$Path) {
  Assert-EvoMindCredentialDestination $Path
  Protect-EvoMindCredentialPath -Path $Path
  try {
    $credential = Import-Clixml -LiteralPath $Path
  } catch {
    throw [System.IO.InvalidDataException]::new("HPC credential format is invalid.")
  }
  if ($credential -isnot [System.Management.Automation.PSCredential]) {
    throw [System.IO.InvalidDataException]::new("HPC credential format is invalid.")
  }
  if ($credential.UserName -notmatch "^[A-Za-z0-9._-]{1,128}$" -or $credential.Password.Length -eq 0) {
    throw [System.IO.InvalidDataException]::new("HPC credential structure is invalid.")
  }
  return $credential
}

function ConvertTo-EvoMindHpcMetadata {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Metadata,
    [string]$ExpectedGenerationId = "",
    [string]$ExpectedCredentialSha256 = "",
    [switch]$Legacy
  )

  $hostValue = [string](Get-EvoMindRequiredObjectProperty $Metadata "host")
  $remoteWorkspaceValue = [string](Get-EvoMindRequiredObjectProperty $Metadata "remote_workspace")
  $socksHostValue = [string](Get-EvoMindRequiredObjectProperty $Metadata "socks_host")
  try {
    $portValue = [int](Get-EvoMindRequiredObjectProperty $Metadata "port")
    $socksPortValue = [int](Get-EvoMindRequiredObjectProperty $Metadata "socks_port")
  } catch {
    throw [System.IO.InvalidDataException]::new("HPC metadata port is invalid.")
  }
  Assert-EvoMindNetworkHost $hostValue
  Assert-EvoMindNetworkHost $socksHostValue -AllowEmpty
  Assert-EvoMindRemoteWorkspace $remoteWorkspaceValue
  if ($portValue -lt 1 -or $portValue -gt 65535) {
    throw [System.IO.InvalidDataException]::new("HPC metadata port is invalid.")
  }
  $hasSocksHost = -not [string]::IsNullOrWhiteSpace($socksHostValue)
  $hasSocksPort = $socksPortValue -gt 0
  if ($hasSocksHost -ne $hasSocksPort -or ($hasSocksPort -and $socksPortValue -gt 65535)) {
    throw [System.IO.InvalidDataException]::new("HPC SOCKS metadata is invalid.")
  }

  $normalized = [ordered]@{
    host = $hostValue.Trim()
    port = $portValue
    remote_workspace = $remoteWorkspaceValue.Trim()
    socks_host = $socksHostValue.Trim()
    socks_port = $socksPortValue
  }
  if ($Legacy) {
    return $normalized
  }

  try {
    $schemaVersion = [int](Get-EvoMindRequiredObjectProperty $Metadata "schema_version")
  } catch {
    throw [System.IO.InvalidDataException]::new("HPC metadata schema is invalid.")
  }
  $generationId = [string](Get-EvoMindRequiredObjectProperty $Metadata "generation_id")
  $credentialSha256 = [string](Get-EvoMindRequiredObjectProperty $Metadata "credential_sha256")
  if ($schemaVersion -ne 1 -or $generationId -notmatch "^[a-f0-9]{32}$") {
    throw [System.IO.InvalidDataException]::new("HPC metadata generation is invalid.")
  }
  if ($credentialSha256 -notmatch "^[a-f0-9]{64}$") {
    throw [System.IO.InvalidDataException]::new("HPC metadata credential hash is invalid.")
  }
  if ($ExpectedGenerationId -and $generationId -cne $ExpectedGenerationId) {
    throw [System.IO.InvalidDataException]::new("HPC metadata generation does not match the current pointer.")
  }
  if ($ExpectedCredentialSha256 -and $credentialSha256 -cne $ExpectedCredentialSha256) {
    throw [System.IO.InvalidDataException]::new("HPC metadata credential hash does not match the current pointer.")
  }
  $normalized.schema_version = $schemaVersion
  $normalized.generation_id = $generationId
  $normalized.credential_sha256 = $credentialSha256
  return $normalized
}

function Initialize-EvoMindHpcGenerationsDirectory([string]$StateDirectory) {
  $paths = Get-EvoMindHpcCredentialStorePaths $StateDirectory
  $directory = $paths.GenerationsDirectory
  Assert-EvoMindNoReparseAncestors $directory $StateDirectory
  if (Test-Path -LiteralPath $directory) {
    $item = Get-Item -LiteralPath $directory -Force -ErrorAction Stop
    if (-not $item.PSIsContainer -or ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
      throw [System.IO.IOException]::new("HPC generation store is invalid.")
    }
  } else {
    New-EvoMindSecureDirectory $directory
  }
  Protect-EvoMindCredentialPath -Path $directory -Directory
  return $directory
}

function Resolve-EvoMindHpcGenerationPayload {
  param(
    [Parameter(Mandatory = $true)]
    [string]$StateDirectory,
    [Parameter(Mandatory = $true)]
    [object]$Pointer
  )
  try {
    $schemaVersion = [int](Get-EvoMindRequiredObjectProperty $Pointer "schema_version")
  } catch {
    throw [System.IO.InvalidDataException]::new("HPC current pointer schema is invalid.")
  }
  $generationId = [string](Get-EvoMindRequiredObjectProperty $Pointer "generation_id")
  $credentialSha256 = [string](Get-EvoMindRequiredObjectProperty $Pointer "credential_sha256")
  $metadataSha256 = [string](Get-EvoMindRequiredObjectProperty $Pointer "metadata_sha256")
  if ($schemaVersion -ne 1 -or $generationId -notmatch "^[a-f0-9]{32}$") {
    throw [System.IO.InvalidDataException]::new("HPC current pointer is invalid.")
  }
  if ($credentialSha256 -notmatch "^[a-f0-9]{64}$" -or $metadataSha256 -notmatch "^[a-f0-9]{64}$") {
    throw [System.IO.InvalidDataException]::new("HPC current pointer hash is invalid.")
  }

  $paths = Get-EvoMindHpcCredentialStorePaths $StateDirectory
  $generationsDirectory = $paths.GenerationsDirectory
  if (-not (Test-Path -LiteralPath $generationsDirectory -PathType Container)) {
    throw [System.IO.InvalidDataException]::new("HPC generation store is missing.")
  }
  Assert-EvoMindNoReparseAncestors $generationsDirectory $StateDirectory
  Protect-EvoMindCredentialPath -Path $generationsDirectory -Directory
  $generationDirectory = Join-Path $generationsDirectory $generationId
  Assert-EvoMindNoReparseAncestors $generationDirectory $generationsDirectory
  if (-not (Test-Path -LiteralPath $generationDirectory -PathType Container)) {
    throw [System.IO.InvalidDataException]::new("HPC current generation is missing.")
  }
  Protect-EvoMindCredentialPath -Path $generationDirectory -Directory

  $credentialPath = Join-Path $generationDirectory "credential.xml"
  $metadataPath = Join-Path $generationDirectory "metadata.json"
  if (-not (Test-Path -LiteralPath $credentialPath -PathType Leaf) -or -not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
    throw [System.IO.InvalidDataException]::new("HPC current generation is incomplete.")
  }
  $actualCredentialSha256 = Get-EvoMindFileSha256 $credentialPath
  $actualMetadataSha256 = Get-EvoMindFileSha256 $metadataPath
  if ($actualCredentialSha256 -cne $credentialSha256 -or $actualMetadataSha256 -cne $metadataSha256) {
    throw [System.IO.InvalidDataException]::new("HPC current generation integrity check failed.")
  }
  $credential = Import-EvoMindHpcCredential $credentialPath
  $metadataObject = Read-EvoMindProtectedJson $metadataPath
  $metadata = ConvertTo-EvoMindHpcMetadata `
    -Metadata $metadataObject `
    -ExpectedGenerationId $generationId `
    -ExpectedCredentialSha256 $credentialSha256

  return [ordered]@{
    Credential = $credential
    Metadata = $metadata
    GenerationId = $generationId
    GenerationDirectory = $generationDirectory
    CredentialPath = $credentialPath
    MetadataPath = $metadataPath
    PointerPath = $paths.PointerPath
    CredentialSha256 = $credentialSha256
    MetadataSha256 = $metadataSha256
  }
}

function Read-EvoMindHpcCurrentGeneration([string]$StateDirectory) {
  $paths = Get-EvoMindHpcCredentialStorePaths $StateDirectory
  if (-not (Test-Path -LiteralPath $paths.PointerPath)) {
    return $null
  }
  if (-not (Test-Path -LiteralPath $paths.PointerPath -PathType Leaf)) {
    throw [System.IO.InvalidDataException]::new("HPC current pointer is invalid.")
  }
  $pointer = Read-EvoMindProtectedJson $paths.PointerPath
  return Resolve-EvoMindHpcGenerationPayload -StateDirectory $StateDirectory -Pointer $pointer
}

function Invoke-EvoMindHpcBeforePointerTestHook([string]$StateDirectory) {
  if ($env:EVOMIND_ALLOW_TEST_STATE_DIR -ne "1" -or [string]::IsNullOrWhiteSpace([string]$env:EVOMIND_DPAPI_STATE_DIR)) {
    return
  }
  if ([System.IO.Path]::GetFullPath($env:EVOMIND_DPAPI_STATE_DIR) -ne [System.IO.Path]::GetFullPath($StateDirectory)) {
    return
  }
  $pauseValue = [string]$env:EVOMIND_HPC_TEST_PAUSE_BEFORE_POINTER_MS
  if ([string]::IsNullOrWhiteSpace($pauseValue)) {
    return
  }
  $pauseMilliseconds = 0
  if (-not [int]::TryParse($pauseValue, [ref]$pauseMilliseconds) -or $pauseMilliseconds -lt 1 -or $pauseMilliseconds -gt 120000) {
    throw [System.ArgumentOutOfRangeException]::new("HPC test pause is invalid.")
  }
  $paths = Get-EvoMindHpcCredentialStorePaths $StateDirectory
  try {
    Assert-EvoMindCredentialDestination $paths.TestHookPath
    [System.IO.File]::WriteAllText(
      $paths.TestHookPath,
      "ready",
      [System.Text.UTF8Encoding]::new($false)
    )
    Protect-EvoMindCredentialPath -Path $paths.TestHookPath
    Sync-EvoMindCredentialFile $paths.TestHookPath
    Start-Sleep -Milliseconds $pauseMilliseconds
  } finally {
    Remove-Item -LiteralPath $paths.TestHookPath -Force -ErrorAction SilentlyContinue
  }
}

function New-EvoMindHpcCredentialGeneration {
  param(
    [Parameter(Mandatory = $true)]
    [string]$StateDirectory,
    [Parameter(Mandatory = $true)]
    [System.Management.Automation.PSCredential]$Credential,
    [Parameter(Mandatory = $true)]
    [System.Collections.IDictionary]$Metadata
  )
  if ($Credential.UserName -notmatch "^[A-Za-z0-9._-]{1,128}$" -or $Credential.Password.Length -eq 0) {
    throw [System.IO.InvalidDataException]::new("HPC credential structure is invalid.")
  }
  $baseMetadata = ConvertTo-EvoMindHpcMetadata -Metadata $Metadata -Legacy
  $generationsDirectory = Initialize-EvoMindHpcGenerationsDirectory $StateDirectory
  $generationId = [guid]::NewGuid().ToString("N")
  $generationDirectory = Join-Path $generationsDirectory $generationId
  New-EvoMindSecureDirectory $generationDirectory
  Protect-EvoMindCredentialPath -Path $generationDirectory -Directory

  $credentialPath = Join-Path $generationDirectory "credential.xml"
  $metadataPath = Join-Path $generationDirectory "metadata.json"
  $credentialTemporaryPath = $null
  $metadataTemporaryPath = $null
  $pointerTemporaryPath = $null
  try {
    $credentialTemporaryPath = Write-EvoMindCredentialTemp `
      -Credential $Credential `
      -StateDirectory $generationDirectory `
      -Prefix "credential"
    Commit-EvoMindCredentialFiles @(
      [ordered]@{ TemporaryPath = $credentialTemporaryPath; DestinationPath = $credentialPath }
    )
    $credentialTemporaryPath = $null
    $credentialSha256 = Get-EvoMindFileSha256 $credentialPath

    $metadataPayload = [ordered]@{
      schema_version = 1
      generation_id = $generationId
      credential_sha256 = $credentialSha256
      host = $baseMetadata.host
      port = $baseMetadata.port
      remote_workspace = $baseMetadata.remote_workspace
      socks_host = $baseMetadata.socks_host
      socks_port = $baseMetadata.socks_port
    }
    $metadataTemporaryPath = Write-EvoMindJsonTemp `
      -Payload $metadataPayload `
      -StateDirectory $generationDirectory `
      -Prefix "metadata"
    Commit-EvoMindCredentialFiles @(
      [ordered]@{ TemporaryPath = $metadataTemporaryPath; DestinationPath = $metadataPath }
    )
    $metadataTemporaryPath = $null
    $metadataSha256 = Get-EvoMindFileSha256 $metadataPath

    $pointerPayload = [ordered]@{
      schema_version = 1
      generation_id = $generationId
      credential_sha256 = $credentialSha256
      metadata_sha256 = $metadataSha256
    }
    [void](Resolve-EvoMindHpcGenerationPayload -StateDirectory $StateDirectory -Pointer $pointerPayload)
    Invoke-EvoMindHpcBeforePointerTestHook $StateDirectory

    $paths = Get-EvoMindHpcCredentialStorePaths $StateDirectory
    $pointerTemporaryPath = Write-EvoMindJsonTemp `
      -Payload $pointerPayload `
      -StateDirectory $StateDirectory `
      -Prefix "hpc-current"
    Commit-EvoMindCredentialFiles @(
      [ordered]@{ TemporaryPath = $pointerTemporaryPath; DestinationPath = $paths.PointerPath }
    )
    $pointerTemporaryPath = $null
    return Read-EvoMindHpcCurrentGeneration $StateDirectory
  } finally {
    if ($credentialTemporaryPath) { Remove-Item -LiteralPath $credentialTemporaryPath -Force -ErrorAction SilentlyContinue }
    if ($metadataTemporaryPath) { Remove-Item -LiteralPath $metadataTemporaryPath -Force -ErrorAction SilentlyContinue }
    if ($pointerTemporaryPath) { Remove-Item -LiteralPath $pointerTemporaryPath -Force -ErrorAction SilentlyContinue }
  }
}

function Resolve-EvoMindHpcCredentialGeneration {
  param(
    [Parameter(Mandatory = $true)]
    [string]$StateDirectory,
    [switch]$MigrateLegacy
  )
  $current = Read-EvoMindHpcCurrentGeneration $StateDirectory
  if ($null -ne $current) {
    return $current
  }

  $paths = Get-EvoMindHpcCredentialStorePaths $StateDirectory
  $hasLegacyCredential = Test-Path -LiteralPath $paths.LegacyCredentialPath -PathType Leaf
  $hasLegacyMetadata = Test-Path -LiteralPath $paths.LegacyMetadataPath -PathType Leaf
  if ($hasLegacyCredential -xor $hasLegacyMetadata) {
    throw [System.IO.InvalidDataException]::new("Legacy HPC credential state is incomplete.")
  }
  if (-not $hasLegacyCredential) {
    if ((Test-Path -LiteralPath $paths.LegacyCredentialPath) -or (Test-Path -LiteralPath $paths.LegacyMetadataPath)) {
      throw [System.IO.InvalidDataException]::new("Legacy HPC credential state is invalid.")
    }
    return $null
  }
  if (-not $MigrateLegacy) {
    return $null
  }

  $legacyCredential = Import-EvoMindHpcCredential $paths.LegacyCredentialPath
  $legacyMetadataObject = Read-EvoMindProtectedJson $paths.LegacyMetadataPath
  $legacyMetadata = ConvertTo-EvoMindHpcMetadata -Metadata $legacyMetadataObject -Legacy
  $current = New-EvoMindHpcCredentialGeneration `
    -StateDirectory $StateDirectory `
    -Credential $legacyCredential `
    -Metadata $legacyMetadata
  try {
    Remove-EvoMindCredentialFile $paths.LegacyCredentialPath
    Remove-EvoMindCredentialFile $paths.LegacyMetadataPath
  } catch {
    # The atomically published generation remains authoritative and secure.
  }
  return $current
}

function Remove-EvoMindHpcCredentialStore([string]$StateDirectory) {
  $paths = Get-EvoMindHpcCredentialStorePaths $StateDirectory
  Remove-EvoMindCredentialFile $paths.PointerPath
  if (Test-Path -LiteralPath $paths.PointerPath) {
    throw [System.IO.IOException]::new("HPC current pointer removal failed.")
  }

  Remove-Item -LiteralPath $paths.TestHookPath -Force -ErrorAction SilentlyContinue
  if (Test-Path -LiteralPath $paths.GenerationsDirectory) {
    Assert-EvoMindNoReparseAncestors $paths.GenerationsDirectory $StateDirectory
    $root = Get-Item -LiteralPath $paths.GenerationsDirectory -Force -ErrorAction Stop
    if (-not $root.PSIsContainer -or ($root.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
      throw [System.IO.IOException]::new("HPC generation store is invalid.")
    }
    foreach ($item in Get-ChildItem -LiteralPath $paths.GenerationsDirectory -Force -Recurse -ErrorAction Stop) {
      if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw [System.IO.IOException]::new("HPC generation store contains a reparse point.")
      }
    }
    Remove-Item -LiteralPath $paths.GenerationsDirectory -Force -Recurse -ErrorAction Stop
  }
  Remove-EvoMindCredentialFile $paths.LegacyCredentialPath
  Remove-EvoMindCredentialFile $paths.LegacyMetadataPath
  if (
    (Test-Path -LiteralPath $paths.GenerationsDirectory) -or
    (Test-Path -LiteralPath $paths.LegacyCredentialPath) -or
    (Test-Path -LiteralPath $paths.LegacyMetadataPath)
  ) {
    throw [System.IO.IOException]::new("HPC credential removal failed.")
  }
}
