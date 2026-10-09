from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/Invoke-G21TerminalPreservingAssetProbeR120.ps1"
REMOTE = ROOT / "scripts/Invoke-G21TerminalPreservingAssetProbeR120Remote.ps1"
CANDIDATE_TREE = "a9595ab4ed9a86786ca65f4456893d64741b449ec84cf8c9ddb42e9bf4110dc7"


def test_runner_is_single_exact_probe_with_no_cancel_or_retry() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    remote = REMOTE.read_text(encoding="utf-8")
    assert remote.count('Post-Api "/v1/sessions/$fixedRun/tools"') == 1
    assert "tool_name=$toolName" in remote
    assert "scope=$scope" in remote
    assert "direct_tool_calls=1" in remote
    assert "run_action_calls=0" in remote
    assert "cancel_calls=0" in remote
    assert "preserve_terminal=$true" in remote
    assert "Assert-LifecyclePreserved" in remote
    assert CANDIDATE_TREE in remote
    forbidden = ("/actions", "action=cancel", "/cancel", "hpc_execute_solution", "training_route", "competition_data_accelerate")
    for value in forbidden:
        assert value not in remote
    assert "$source | & ssh" not in wrapper
    assert "& scp @sshOptions $runner" in wrapper
    assert "single_scp_runner" in wrapper
    assert "retain_for_audit_no_automatic_delete" in wrapper + remote
    assert "retries=0" in remote
    assert "tool_started" in remote and "tool_completed" in remote
    assert "payload.tool_call_id" in remote and "payload.tool" in remote
    assert "$settledType = if" in remote


def test_validate_only_does_not_open_ssh_or_write_result(tmp_path: Path) -> None:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    assert shell
    output = tmp_path / "probe.json"
    completed = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-File", str(WRAPPER), "-OutputPath", str(output), "-ValidateOnly"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["status"] == "validated_not_executed"
    assert payload["production_accessed"] is False
    assert payload["direct_tool_calls"] == 0
    assert payload["cancel_calls"] == 0
    assert not output.exists()


def test_powershell_5_and_7_parse_all_r120_scripts() -> None:
    scripts = [
        ROOT / "scripts/Invoke-G21TerminalPreservingAssetProbeR120.ps1",
        ROOT / "scripts/Invoke-G21TerminalPreservingAssetProbeR120Remote.ps1",
        ROOT / "scripts/Deploy-G21TerminalPreservingProbeRuntimeR120.ps1",
        ROOT / "scripts/Deploy-G21TerminalPreservingProbeRuntimeR120Remote.ps1",
    ]
    shells = [value for value in (shutil.which("pwsh"), shutil.which("powershell")) if value]
    assert shells
    for shell in shells:
        for path in scripts:
            escaped = str(path).replace("'", "''")
            command = f"$t=$null;$e=$null;[Management.Automation.Language.Parser]::ParseFile('{escaped}',[ref]$t,[ref]$e)|Out-Null;if(@($e).Count){{$e|% Message;exit 1}}"
            completed = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, text=True, check=False)
            assert completed.returncode == 0, f"{shell} {path}: {completed.stdout} {completed.stderr}"
