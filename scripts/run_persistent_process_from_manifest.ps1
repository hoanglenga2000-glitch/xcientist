[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-AtomicJson {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Payload
    )
    $parent = Split-Path -Parent $Path
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    $temporary = "$Path.$PID.tmp"
    $json = $Payload | ConvertTo-Json -Depth 30
    [System.IO.File]::WriteAllText(
        $temporary,
        $json + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$manifestSha256 = Get-Sha256 -Path $manifestPath
$spec = Get-Content -Raw -LiteralPath $manifestPath -Encoding utf8 | ConvertFrom-Json
$statusPath = [string]$spec.status_path
$startedAt = Get-Date

try {
    if ($spec.schema -ne 'evomind.persistent_process_manifest.v1') {
        throw 'Unexpected persistent-process manifest schema'
    }
    if ($spec.process_signals_allowed -ne $false) {
        throw 'Persistent-process manifest permits process signals'
    }
    $executable = [string]$spec.executable
    $workingDirectory = [string]$spec.working_directory
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Executable is missing: $executable"
    }
    if (-not (Test-Path -LiteralPath $workingDirectory -PathType Container)) {
        throw "Working directory is missing: $workingDirectory"
    }
    foreach ($binding in @($spec.artifact_bindings)) {
        $path = [string]$binding.path
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Bound artifact is missing: $path"
        }
        if ((Get-Sha256 -Path $path) -ne [string]$binding.sha256) {
            throw "Bound artifact hash drifted: $path"
        }
    }

    $fragments = @($spec.unique_command_fragments | ForEach-Object { [string]$_ })
    if ($fragments.Count -gt 0) {
        $expectedProcessName = [System.IO.Path]::GetFileName($executable)
        $duplicates = @(Get-CimInstance Win32_Process | Where-Object {
            if (-not [string]::Equals(
                [string]$_.Name,
                $expectedProcessName,
                [StringComparison]::OrdinalIgnoreCase
            )) {
                return $false
            }
            $command = [string]$_.CommandLine
            $matches = $true
            foreach ($fragment in $fragments) {
                if (-not $command.Contains($fragment, [StringComparison]::OrdinalIgnoreCase)) {
                    $matches = $false
                    break
                }
            }
            $matches
        })
        if ($duplicates.Count -gt 0) {
            Write-AtomicJson -Path $statusPath -Payload ([ordered]@{
                schema = 'evomind.persistent_process_status.v1'
                created_at = (Get-Date -Format o)
                status = 'duplicate_guard_blocked'
                task_id = [string]$spec.task_id
                manifest_path = $manifestPath
                manifest_sha256 = $manifestSha256
                duplicate_pids = @($duplicates.ProcessId)
                wrapper_pid = $PID
                process_signals_sent = 0
            })
            exit 12
        }
    }

    $stdoutPath = [string]$spec.stdout_path
    $stderrPath = [string]$spec.stderr_path
    foreach ($logPath in @($stdoutPath, $stderrPath)) {
        $parent = Split-Path -Parent $logPath
        if ($parent) {
            New-Item -ItemType Directory -Force -Path $parent | Out-Null
        }
    }
    $stdoutStream = [System.IO.File]::Open(
        $stdoutPath,
        [System.IO.FileMode]::Append,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::ReadWrite
    )
    $stderrStream = [System.IO.File]::Open(
        $stderrPath,
        [System.IO.FileMode]::Append,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::ReadWrite
    )

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $executable
    $startInfo.WorkingDirectory = $workingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($argument in @($spec.arguments)) {
        [void]$startInfo.ArgumentList.Add([string]$argument)
    }
    if ($null -ne $spec.environment) {
        foreach ($property in $spec.environment.PSObject.Properties) {
            $startInfo.Environment[[string]$property.Name] = [string]$property.Value
        }
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw 'ProcessStartInfo.Start returned false'
    }
    $stdoutCopy = $process.StandardOutput.BaseStream.CopyToAsync($stdoutStream)
    $stderrCopy = $process.StandardError.BaseStream.CopyToAsync($stderrStream)
    Write-AtomicJson -Path $statusPath -Payload ([ordered]@{
        schema = 'evomind.persistent_process_status.v1'
        created_at = (Get-Date -Format o)
        status = 'running'
        task_id = [string]$spec.task_id
        manifest_path = $manifestPath
        manifest_sha256 = $manifestSha256
        wrapper_pid = $PID
        worker_pid = $process.Id
        executable = $executable
        arguments = @($spec.arguments)
        working_directory = $workingDirectory
        stdout_path = $stdoutPath
        stderr_path = $stderrPath
        process_signals_sent = 0
    })

    $process.WaitForExit()
    [System.Threading.Tasks.Task]::WaitAll(@($stdoutCopy, $stderrCopy))
    $stdoutStream.Flush()
    $stderrStream.Flush()
    $exitCode = $process.ExitCode
    $stdoutStream.Dispose()
    $stderrStream.Dispose()
    Write-AtomicJson -Path $statusPath -Payload ([ordered]@{
        schema = 'evomind.persistent_process_status.v1'
        created_at = (Get-Date -Format o)
        status = $(if ($exitCode -eq 0) { 'completed' } else { 'failed' })
        task_id = [string]$spec.task_id
        manifest_path = $manifestPath
        manifest_sha256 = $manifestSha256
        wrapper_pid = $PID
        worker_pid = $process.Id
        exit_code = $exitCode
        elapsed_seconds = ((Get-Date) - $startedAt).TotalSeconds
        stdout_path = $stdoutPath
        stderr_path = $stderrPath
        process_signals_sent = 0
    })
    exit $exitCode
}
catch {
    Write-AtomicJson -Path $statusPath -Payload ([ordered]@{
        schema = 'evomind.persistent_process_status.v1'
        created_at = (Get-Date -Format o)
        status = 'launcher_failed'
        task_id = $(if ($null -ne $spec.task_id) { [string]$spec.task_id } else { $null })
        manifest_path = $manifestPath
        manifest_sha256 = $manifestSha256
        wrapper_pid = $PID
        error_type = $_.Exception.GetType().FullName
        error = $_.Exception.Message
        process_signals_sent = 0
    })
    throw
}
