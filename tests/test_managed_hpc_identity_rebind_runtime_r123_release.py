from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_managed_hpc_identity_rebind_runtime_r123 as builder  # noqa: E402
import verify_managed_hpc_identity_rebind_runtime_r123_candidate as verifier  # noqa: E402


BASELINE_ZIP = Path(
    r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\r121-g21-hpc-asset-probe-production-20260901-d\evomind-runtime-r121-g21-hpc-asset-probe-r120-base.zip"
)


def test_r123_overlay_changes_only_three_allowlisted_files() -> None:
    assert BASELINE_ZIP.is_file()
    baseline, _ = builder.inspect_baseline(BASELINE_ZIP)
    overlays = builder.build_overlays(ROOT / "src", baseline)
    candidate = {**baseline, **overlays}
    changed = sorted(name for name in candidate if candidate[name] != baseline[name])
    unchanged = sorted(name for name in candidate if candidate[name] == baseline[name])

    assert changed == sorted(builder.CHANGED_FILES)
    assert len(candidate) == 32
    assert len(unchanged) == 29
    assert candidate["evomind_runtime/tools.py"] == baseline["evomind_runtime/tools.py"]
    assert candidate["evomind_runtime/__init__.py"] == baseline["evomind_runtime/__init__.py"]
    assistant = candidate["evomind_runtime/assistant_runs.py"].decode("utf-8-sig")
    store = candidate["evomind_runtime/store.py"].decode("utf-8-sig")
    http = candidate["evomind_runtime/http_server.py"].decode("utf-8-sig")
    assert "managed_hpc_identity.rebound" in assistant
    assert "def rebind_managed_hpc_identity(" in assistant
    assert "def rebind_session_managed_hpc_identity(" in store
    assert "WHERE session_id=? AND event_type=? ORDER BY seq" in store
    assert 'idempotency_key=str(body.get("idempotency_key") or "")' in http


def test_r123_builder_rejects_unbound_source_bytes(tmp_path: Path) -> None:
    assert BASELINE_ZIP.is_file()
    baseline, _ = builder.inspect_baseline(BASELINE_ZIP)
    source_root = tmp_path / "src"
    runtime_root = source_root / "evomind_runtime"
    runtime_root.mkdir(parents=True)
    for name in builder.CHANGED_FILES:
        shutil.copy2(ROOT / "src" / name, source_root / name)
    assistant = runtime_root / "assistant_runs.py"
    assistant.write_bytes(assistant.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SOURCE_RAW_SHA_MISMATCH"):
        builder.build_overlays(source_root, baseline)


def test_r123_junit_requires_at_least_one_passing_test(tmp_path: Path) -> None:
    zero = tmp_path / "zero.xml"
    zero.write_text('<testsuites><testsuite tests="0" failures="0" errors="0"/></testsuites>', encoding="utf-8")
    with pytest.raises(ValueError, match="ZERO_TESTS"):
        builder.parse_junit_receipt(zero)
    with pytest.raises(ValueError, match="ZERO_TESTS"):
        verifier.parse_junit_receipt(zero)

    passing = tmp_path / "passing.xml"
    passing.write_text(
        '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="1"/></testsuites>',
        encoding="utf-8",
    )
    assert builder.parse_junit_receipt(passing)["tests"] == 2
    assert verifier.parse_junit_receipt(passing)["tests"] == 2


def test_r123_verifier_has_independent_zip_and_manifest_validation() -> None:
    source = (SCRIPTS / "verify_managed_hpc_identity_rebind_runtime_r123_candidate.py").read_text(
        encoding="utf-8"
    )
    assert "from build_managed_hpc_identity_rebind_runtime_r123 import" not in source
    assert "def safe_member(" in source
    assert "def inspect_baseline(" in source
    assert "def inspect_candidate(" in source
    assert "def verify_extraction(" in source
    assert '[sys.executable, "-I", "-c", program]' in source


def test_r123_deployers_preserve_managed_service_and_hpc_boundaries() -> None:
    wrapper = (SCRIPTS / "Deploy-ManagedHpcIdentityRebindRuntimeR123.ps1").read_text(encoding="utf-8")
    remote = (SCRIPTS / "Deploy-ManagedHpcIdentityRebindRuntimeR123Remote.ps1").read_text(encoding="utf-8")
    combined = wrapper + "\n" + remote
    assert "Invoke-ServiceAccountAction.ps1" in remote
    assert "$fixedJob = 93015" in remote
    assert '"${fixedTenant}_job${fixedJob}_g${fixedGeneration}"' in remote
    assert "$fixedGeneration = 24" in remote
    assert "overlay-assistant-interaction-r122-1393046c9de7" in combined
    assert "hpc_profile_frozen.tombstone.json" in remote
    assert "hpc_profile_retired.tombstone.json" in remote
    assert "function InspectZip" in remote
    assert "R123_ZIP_UNSAFE_MEMBER" in remote
    assert "candidate_zip_inspection" in remote
    assert "R123_ROLLBACK_RUNTIME_HASH_REJECTED" in remote
    assert "function AssertRestoredDb" in remote
    assert "Start-ScheduledTask" not in combined
    assert "Stop-Process" not in combined
    assert "10.120." not in combined
    assert "hpc_accessed=$false" in combined
    assert "training_started=$false" in combined
