from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = Path(
    r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\r117-g21-conditional-policy-r116-base-20260831"
)
BASELINE = Path(
    r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\r116-g21-goal-r115-base-v2-20260831"
    r"\evomind-runtime-r116-g21-goal-r115-base-v2.zip"
)
PREFIX = "evomind-runtime-r117-g21-conditional-policy-r116-base"
WRAPPER = ROOT / "scripts/Deploy-G21ConditionalPolicyRuntimeR117.ps1"
RUNNER = ROOT / "scripts/Deploy-G21ConditionalPolicyRuntimeR117Remote.ps1"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=120,
    )


def test_formal_candidate_and_independent_receipt_are_closed() -> None:
    build = json.loads((ARTIFACT / f"{PREFIX}-build-result.json").read_text(encoding="utf-8"))
    receipt = json.loads(
        (ARTIFACT / f"{PREFIX}-independent-receipt.json").read_text(encoding="utf-8")
    )
    assert build["status"] == "built_and_verified"
    assert build["zip_sha256"] == "de7c8f33288ed12977125d6bf1813ec8a0e27ef42de269eeba92e87c56632fcc"
    assert build["candidate_runtime_tree_sha256"] == "c3f72a425374db31bac318c27bddd8752be9fb3d3ebecf1351f0265bbd1f8759"
    assert build["candidate_runtime_file_count"] == 32
    assert build["changed_file_count"] == 5
    assert build["unchanged_file_count"] == 27
    assert len(build["verification_rounds"]) == 2
    assert all(row["py_compile"] == 32 for row in build["verification_rounds"])
    assert receipt["status"] == "verified"
    assert receipt["candidate_zip_sha256"] == build["zip_sha256"]
    assert receipt["candidate_runtime_tree_sha256"] == build["candidate_runtime_tree_sha256"]
    assert len(receipt["verification_rounds"]) == 2
    assert receipt["production_deployed"] is False
    assert receipt["hpc_accessed"] is False
    assert receipt["gpu_touched"] is False


def test_builder_dry_run_rejects_worktree_drift_and_closes_exact_scope(tmp_path: Path) -> None:
    result = _run(
        sys.executable,
        "scripts/build_g21_conditional_policy_runtime_r117.py",
        "--baseline-zip",
        str(BASELINE),
        "--output-dir",
        str(tmp_path / "unused"),
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run_verified"
    assert payload["candidate_runtime_file_count"] == 32
    assert payload["changed_file_count"] == 5
    assert payload["unchanged_file_count"] == 27
    assert payload["old_manifest_removed"] is True
    assert payload["new_manifest_count"] == 1


def test_deployer_uses_managed_service_actions_and_has_atomic_rollback_contract() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "C:\\SecureInput\\Invoke-ServiceAccountAction.ps1" in runner
    assert "Invoke-ManagedAction Stop" in runner
    assert "Invoke-ManagedAction Start" in runner
    assert "Backup-Database" in runner
    assert "Restore-Database" in runner
    assert "runtime.sqlite3-wal" in runner or "@('','-wal','-shm')" in runner
    assert "Invoke-PolicyMigration" in runner
    assert "goal.policy_migrated" in runner
    assert "R117_CURRENT_RUNTIME_NOT_R116" in runner
    assert "R117_ROLLBACK_RUNTIME_REJECTED" in runner
    assert "-ValidateOnly" not in runner
    for forbidden in (
        "Start-ScheduledTask",
        "Stop-Process",
        "taskkill",
        "sqlite3.exe",
        "Get-FileHash",
    ):
        assert forbidden not in wrapper
        assert forbidden not in runner


def test_powershell_7_and_5_parse_deployers() -> None:
    scripts = (WRAPPER, RUNNER)
    powershells = [shutil.which("pwsh"), shutil.which("powershell.exe")]
    assert all(powershells)
    for executable in powershells:
        for script in scripts:
            command = (
                "$t=$null;$e=$null;"
                f"[Management.Automation.Language.Parser]::ParseFile('{script}',[ref]$t,[ref]$e)|Out-Null;"
                "if(@($e).Count){$e|ForEach-Object{$_.Message};exit 1}else{'PARSE_OK'}"
            )
            result = _run(str(executable), "-NoProfile", "-NonInteractive", "-Command", command)
            assert result.returncode == 0, result.stderr + result.stdout
            assert "PARSE_OK" in result.stdout


def test_fresh_local_approval_validates_on_ps7_and_ps5_without_production(tmp_path: Path) -> None:
    output = tmp_path / "approval"
    result = _run(
        sys.executable,
        "scripts/new_g21_conditional_policy_runtime_r117_deployment_approval.py",
        "--candidate-zip",
        str(ARTIFACT / f"{PREFIX}.zip"),
        "--source-manifest",
        str(ARTIFACT / f"{PREFIX}-source-manifest.json"),
        "--build-result",
        str(ARTIFACT / f"{PREFIX}-build-result.json"),
        "--test-receipt",
        str(ARTIFACT / f"{PREFIX}-independent-receipt.json"),
        "--old-spec",
        "configs/g21_five_competition_goal_v1_frozen.json",
        "--new-spec",
        "configs/g21_five_competition_goal.json",
        "--policy-evidence",
        "configs/g21_conditional_strong_baseline_policy_evidence.json",
        "--verifier",
        "scripts/verify_g21_conditional_policy_runtime_r117_candidate.py",
        "--wrapper",
        str(WRAPPER),
        "--remote-runner",
        str(RUNNER),
        "--output-dir",
        str(output),
        "--valid-minutes",
        "20",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    approval = output / "evomind-runtime-r117-g21-conditional-policy-deployment-approval.json"
    for executable in (shutil.which("pwsh"), shutil.which("powershell.exe")):
        validated = _run(
            str(executable),
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(WRAPPER),
            "-ApprovalPath",
            str(approval),
            "-ValidateOnly",
        )
        assert validated.returncode == 0, validated.stderr + validated.stdout
        payload = json.loads(validated.stdout.strip().splitlines()[-1])
        assert payload["status"] == "validated_not_deployed"
        assert payload["production_action_performed"] is False
        assert payload["hpc_accessed"] is False
        assert payload["gpu_touched"] is False
        assert payload["remote_writes"] == 0
