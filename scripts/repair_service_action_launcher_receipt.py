"""Narrow source patch: fresh launcher receipt proves same-second task starts."""
import argparse
import hashlib
from pathlib import Path


def patch(source: str) -> str:
    tick = chr(96)
    replacements = [
        ("$mainStderr = 'C:\\ProgramData\\EvoMind\\logs\\cloud-node-service-action.err.log'",
         "$mainStderr = 'C:\\ProgramData\\EvoMind\\logs\\cloud-node-service-action.err.log'\n$launcherResult = 'C:\\ProgramData\\EvoMind\\logs\\cloud-node-system-launcher.result.json'"),
        ("  [IO.File]::WriteAllText(\n    $launcherScript,",
         "  Remove-Item -LiteralPath $launcherResult -Force -ErrorAction SilentlyContinue\n"
         "  $launcherBody = @\"\n" + tick + "$ErrorActionPreference = 'Stop'\n"
         "Start-ScheduledTask -TaskName '$taskName'\n"
         "[IO.File]::WriteAllText('$launcherResult','{\"exit_code\":0}',[Text.UTF8Encoding]::new(" + tick + "$false))\n"
         "\"@\n  [IO.File]::WriteAllText(\n    $launcherScript,"),
        ("    \"Start-ScheduledTask -TaskName '$taskName'" + tick + "r" + tick + "n\",",
         "    $launcherBody,"),
        ("    -Deadline $launcherDeadline -TimeoutError 'SYSTEM_LAUNCHER_TIMEOUT'",
         "    -Deadline $launcherDeadline -TimeoutError 'SYSTEM_LAUNCHER_TIMEOUT' -ResultPath $launcherResult"),
        ("  if ([int]$launcherInfo.LastTaskResult -ne 0)",
         "  $launcherReceipt = Get-Content -LiteralPath $launcherResult -Raw | ConvertFrom-Json\n"
         "  if ([int]$launcherReceipt.exit_code -ne 0) { throw 'SYSTEM_LAUNCHER_RECEIPT_FAILED' }\n"
         "  if ([int]$launcherInfo.LastTaskResult -ne 0)"),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('managed_source_contract_changed')
        source = source.replace(before, after, 1)
    return source


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != args.expected_sha256:
        raise SystemExit('managed_source_hash_changed')
    if args.output.exists():
        raise SystemExit('output_already_exists')
    args.output.write_text(patch(args.source.read_text(encoding='utf-8-sig')), encoding='utf-8', newline='\n')
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())
