param([Parameter(Mandatory = $true)][string]$Url)
$ErrorActionPreference = "Stop"
$uri = [Uri]$Url
if ($uri.Scheme -ne "http" -or $uri.Host -notin @("127.0.0.1", "localhost")) { throw "Only the local EvoMind URL may be opened." }
Start-Process -FilePath $uri.AbsoluteUri
