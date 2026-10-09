from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = Path(r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\r121-g21-hpc-asset-probe-r120-base-v1-20260901")
APPROVAL_ARTIFACT = Path(r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\r121-g21-hpc-asset-probe-validation-approval-v4-20260901")
BASELINE = Path(r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\r120-g21-terminal-preserving-probe-r119-base-v3-20260901\evomind-runtime-r120-g21-terminal-preserving-probe-r119-base.zip")
ZIP = ARTIFACT / "evomind-runtime-r121-g21-hpc-asset-probe-r120-base.zip"
MANIFEST = ARTIFACT / "evomind-runtime-r121-g21-hpc-asset-probe-r120-base-source-manifest.json"
BUILD = ARTIFACT / "evomind-runtime-r121-g21-hpc-asset-probe-r120-base-build-result.json"
BASELINE_SHA = "979224d6baf2d779357d2d2a7ffacd57637289dce11a716857c15fe913eafa3c"
BASELINE_TREE = "5998ea21823c121c23a421bb145d8e2266bb79ab59eccc91a16122c3626b61ce"
CANDIDATE_TREE = "a9595ab4ed9a86786ca65f4456893d64741b449ec84cf8c9ddb42e9bf4110dc7"
PACKAGED_TOOLS = "8176c799277f7c56f67232dbfc3f4fda9c5fad88e1aa280087d94393433e97fd"


DEPLOY_ARTIFACTS_ROOT = Path(r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts")


def _require_deploy_artifacts() -> None:
    # Frozen release artifacts live in the out-of-repo deploy archive. Skip only when the
    # whole archive is absent on this machine; a missing file inside an existing archive
    # still fails the release-binding assertions below.
    if not DEPLOY_ARTIFACTS_ROOT.is_dir():
        pytest.skip("EvoMind-Cloud-Deploy release artifacts archive is not present on this machine")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_is_single_tools_overlay_on_r120() -> None:
    _require_deploy_artifacts()
    assert sha(BASELINE) == BASELINE_SHA
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    build = json.loads(BUILD.read_text(encoding="utf-8"))
    assert manifest["schema"] == "evomind.g21_hpc_asset_probe_runtime.r121.v1"
    assert manifest["baseline_runtime_tree_sha256"] == BASELINE_TREE
    assert manifest["candidate_runtime_tree_sha256"] == CANDIDATE_TREE
    assert manifest["packaged_tools_sha256"] == PACKAGED_TOOLS
    assert manifest["changed_files"] == ["evomind_runtime/tools.py"]
    assert manifest["changed_file_count"] == 1
    assert manifest["unchanged_file_count"] == 31
    assert manifest["output_bound"]["max_probe_stdout_bytes"] == 262144
    assert manifest["output_bound"]["model_file_details_included"] is False
    assert manifest["output_bound"]["model_metadata_closure_complete"] is True
    assert build["status"] == "built_and_verified"
    assert build["zip_sha256"] == sha(ZIP)
    assert len(build["verification_rounds"]) == 2
    assert all(row["py_compile"] == 32 for row in build["verification_rounds"])
    assert all(row["policy_smoke"]["output_bound_tests"] for row in build["verification_rounds"])
    with zipfile.ZipFile(BASELINE) as old_zip, zipfile.ZipFile(ZIP) as new_zip:
        old = {name: old_zip.read(name) for name in old_zip.namelist() if name.startswith("evomind_runtime/")}
        new = {name: new_zip.read(name) for name in new_zip.namelist() if name.startswith("evomind_runtime/")}
    assert set(old) == set(new)
    changed = [name for name in sorted(old) if old[name] != new[name]]
    assert changed == ["evomind_runtime/tools.py"]
    assert hashlib.sha256(new[changed[0]]).hexdigest() == PACKAGED_TOOLS
    assert hashlib.sha256(new["evomind_runtime/http_server.py"]).hexdigest() == "6d2d17f1257736683c4bc590f346c183c9feb25e06cce7a77d63d35843e23a9d"


def test_builder_dry_run_binds_r120_and_output_contract(tmp_path: Path) -> None:
    _require_deploy_artifacts()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_g21_hpc_asset_probe_runtime_r121.py"),
            "--baseline-zip", str(BASELINE),
            "--output-dir", str(tmp_path / "must-not-exist"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["status"] == "dry_run_verified"
    assert payload["candidate_runtime_tree_sha256"] == CANDIDATE_TREE
    assert payload["packaged_tools_sha256"] == PACKAGED_TOOLS
    assert payload["changed_files"] == ["evomind_runtime/tools.py"]
    assert not (tmp_path / "must-not-exist").exists()


def test_deployer_is_runtime_only_and_does_not_probe_hpc() -> None:
    wrapper = (ROOT / "scripts/Deploy-G21HpcAssetProbeRuntimeR121.ps1").read_text(encoding="utf-8")
    remote = (ROOT / "scripts/Deploy-G21HpcAssetProbeRuntimeR121Remote.ps1").read_text(encoding="utf-8")
    assert "single_scp_envelope" in wrapper
    assert "Invoke-ServiceAccountAction.ps1" in remote
    assert "BackupDb" in remote and "RestoreDb" in remote
    assert "function Assert-Fresh" in remote
    assert "Assert-Fresh $managedBefore $managedAfter" in remote
    assert "R121_LISTENER_NOT_FRESH" in remote
    assert "foreach($port in @('65068','8765','8088','7890'))" in remote
    assert "hpc_accessed=$false" in remote
    assert "hpc_tool_calls=0" in remote
    assert "Start-ScheduledTask" not in wrapper + remote
    assert "Stop-Process" not in wrapper + remote
    assert "10.120." not in wrapper + remote
    assert "/v1/sessions/" not in remote
    assert "tools/call" not in remote
    assert "hpc_execute_solution" not in remote


def test_release_scripts_py_compile() -> None:
    for path in (
        ROOT / "scripts/build_g21_hpc_asset_probe_runtime_r121.py",
        ROOT / "scripts/verify_g21_hpc_asset_probe_runtime_r121_candidate.py",
        ROOT / "scripts/new_g21_hpc_asset_probe_runtime_r121_deployment_approval.py",
    ):
        completed = subprocess.run([sys.executable, "-m", "py_compile", str(path)], capture_output=True, text=True, check=False)
        assert completed.returncode == 0, f"{path}: {completed.stderr}"


def test_validation_only_approval_and_deployer_are_non_authorizing(tmp_path: Path) -> None:
    _require_deploy_artifacts()
    approval = APPROVAL_ARTIFACT / "evomind-runtime-r121-hpc-asset-probe-deployment-approval.json"
    assert approval.is_file()
    payload = json.loads(approval.read_text(encoding="utf-8"))
    assert payload["schema"] == "evomind.g21_hpc_asset_probe_runtime_deployment_approval.r121.v1"
    assert payload["status"] == "validation_only_non_authorizing"
    assert payload["one_time"] is True
    assert payload["run_id"] == "run_7b1efb878afb40f396db431e91f093a5"
    assert payload["allocation"] == "G21"
    assert payload["candidate"]["installed_runtime_tree_sha256"] == CANDIDATE_TREE
    assert payload["candidate"]["packaged_tools_sha256"] == PACKAGED_TOOLS
    assert payload["production_invariants"]["output_bound_fix"] is True
    result = subprocess.run(
        [
            "pwsh", "-NoProfile", "-NonInteractive", "-File",
            str(APPROVAL_ARTIFACT / "Deploy-G21HpcAssetProbeRuntimeR121.ps1"),
            "-ApprovalPath", str(approval), "-ValidateOnly",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    checked = json.loads(result.stdout.strip().splitlines()[-1])
    assert checked["status"] == "validated_not_deployed"
    assert checked["production_action_performed"] is False
    assert checked["hpc_accessed"] is False
    assert checked["remote_writes"] == 0
