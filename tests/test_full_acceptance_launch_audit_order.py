from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_full_acceptance_supports_launcher_owned_audit() -> None:
    source = (ROOT / "scripts" / "run_full_acceptance.py").read_text(encoding="utf-8")

    assert '"--skip-verified-launch-audit"' in source
    assert "if not args.skip_verified_launch_audit:" in source
    assert 'commands.append([sys.executable, "scripts/verify_verified_workstation_launch_audit.py"])' in source


def test_launcher_writes_current_audit_before_self_verification() -> None:
    source = (ROOT / "scripts" / "start_verified_workstation.ps1").read_text(encoding="utf-8-sig")

    write_index = source.index("$auditPaths = Write-VerifiedAuditReport")
    verify_index = source.index('Invoke-JsonCommand -Label "verified_launch_audit"')
    output_index = source.index("Write-Output ([ordered]@{", verify_index)
    assert write_index < verify_index < output_index
    assert "output_excerpt" not in source
    assert "Write-PendingAuditReport -RunId $runId" in source


def test_launcher_runs_full_acceptance_only_through_the_audit_aware_path() -> None:
    source = (ROOT / "scripts" / "start_verified_workstation.ps1").read_text(encoding="utf-8-sig")

    marker = '$smokeResults += Invoke-JsonCommand -Label "full_acceptance"'
    assert source.count(marker) == 1
    acceptance_block = source[source.index(marker) : source.index("$dashboardStatus =", source.index(marker))]
    assert '"--skip-verified-launch-audit"' in acceptance_block
    assert "if ($ShouldRunFullAcceptance)" in source


def test_launcher_preserves_kaggle_configured_not_invoked_signal() -> None:
    source = (ROOT / "scripts" / "start_verified_workstation.ps1").read_text(encoding="utf-8-sig")

    assert '$payload.verification_state -eq "configured_not_invoked"' in source
    assert "$signals.kaggle_configured_not_invoked = $true" in source
