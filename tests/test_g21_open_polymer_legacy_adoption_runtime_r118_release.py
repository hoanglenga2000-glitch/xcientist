from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_g21_open_polymer_legacy_adoption_runtime_r118.py"
WRAPPER = ROOT / "scripts/Deploy-G21OpenPolymerLegacyAdoptionRuntimeR118.ps1"
REMOTE = ROOT / "scripts/Deploy-G21OpenPolymerLegacyAdoptionRuntimeR118Remote.ps1"
BASELINE = Path(
    r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\r117-g21-conditional-policy-r116-base-20260831\evomind-runtime-r117-g21-conditional-policy-r116-base.zip"
)
FINAL = Path(
    r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\r118-g21-open-polymer-legacy-adoption-r117-base-final-20260831"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builder_is_frozen_to_final_two_file_allowlist_and_dry_run() -> None:
    source = BUILDER.read_text(encoding="utf-8")
    ast.parse(source)
    assert "5d7a9d1977434796d490c380f1c1447d5a2e7359bd0cf9e0c124085df7d84977" in source
    assert "6c6cbf1591b71845ac20aa92c271f03517fedb19b06f2cc97bc008b3c714721a" in source
    completed = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--baseline-zip",
            str(BASELINE),
            "--output-dir",
            str(FINAL / "never-created-dry-run"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert '"changed_file_count": 2' in completed.stdout
    assert '"unchanged_file_count": 30' in completed.stdout


def test_final_candidate_and_receipt_are_hash_bound() -> None:
    assert _sha(FINAL / "evomind-runtime-r118-g21-open-polymer-legacy-adoption-r117-base.zip") == (
        "6e0c5c67a9cb69c68f5f2f88777e41a0db86c4340cfb2b0d04b981e84259908c"
    )
    assert _sha(FINAL / "evomind-runtime-r118-g21-open-polymer-legacy-adoption-r117-base-source-manifest.json") == (
        "5cf108c09395e236987bdf7924e84a15ffd6437c603c0ac34d722f068feda76d"
    )
    assert _sha(FINAL / "evomind-runtime-r118-g21-open-polymer-legacy-adoption-r117-base-independent-receipt.json") == (
        "96ac5ed2f891cfadbf70ce43cb2cdd428f609759a92e3e6febe57726dac5ad3d"
    )
    assert _sha(FINAL / "open-polymer-legacy-adoption-item-r118.json") == (
        "07eafe77bb33545dc98f59abefd60bd153ac491f67a5c283a2fb374796e006df"
    )


def test_deploy_harness_is_r117_to_r118_without_second_policy_migration() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    remote = REMOTE.read_text(encoding="utf-8")
    for forbidden in (
        "Invoke-PolicyMigration",
        "goal.policy_migrated",
        "Start-ScheduledTask",
        "Stop-Process",
        "sqlite3.connect",
        "SELECT * FROM",
        "UPDATE goal_records",
    ):
        assert forbidden not in wrapper
        assert forbidden not in remote
    for marker in (
        "Read-InitialGoalGate",
        "Invoke-LegacyAdoption",
        "Read-LegacyAdoptionState",
        "goal.updated",
        "R118_CURRENT_RUNTIME_NOT_R117",
        "R118_PYTHON_ROLE_PROCESS_REJECTED",
        "'python.exe'",
    ):
        assert marker in remote
    assert "throw'" not in wrapper
    assert "throw'" not in remote
    assert "throw 'R118_TRANSACTION_FAILED'" in wrapper
    assert "goal_before.board_sha256" in wrapper
    assert "goal_updated_event_count" in wrapper
    assert "completion_count" in wrapper
