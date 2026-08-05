from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_job90353_profile_reenrollment_plan.py"
    spec = importlib.util.spec_from_file_location("build_job90353_profile_reenrollment_plan", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _readiness(path: Path, *, status: str = "failed_closed") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "evomind.hpc.profile_readiness.v1",
                "profile": "job90353",
                "status": status,
                "failed_checks": ["schema_v2"] if status != "ready" else [],
                "claim_boundary": "local profile metadata readiness only",
                "details": {"allocation_generation": None},
                "boundaries": {
                    "dpapi_decrypted": False,
                    "ssh_connections": 0,
                    "remote_commands": 0,
                    "training_started": False,
                    "grader_calls": 0,
                    "kaggle_submissions": 0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_reenrollment_plan_is_no_secret_and_actionable(tmp_path):
    builder = _load_builder()
    readiness = tmp_path / "profile.json"
    _readiness(readiness)

    plan = builder.build_plan(
        readiness,
        allocation_binding_id="job90353-reenroll-test",
        allocation_generation=2,
        cluster_name="AI-X86_NVIDIA",
        image_name="10.120.18.240:5000/admin/webide-jupyter:v6.2",
        environment_type="JupyterLab",
        allocation_created_at="2026-07-30 09:23:13",
        role_account="aimslab-TosNrOAl",
        allocation_private_host="10.120.18.240",
        allocation_private_port=6988,
    )
    wire = json.dumps(plan, ensure_ascii=False)

    assert plan["status"] == "ready_for_operator_secret_entry"
    assert plan["allocation_page_metadata"]["cluster_name"] == "AI-X86_NVIDIA"
    assert plan["allocation_page_metadata"]["role_account"] == "aimslab-TosNrOAl"
    assert plan["allocation_page_metadata"]["allocation_private_endpoint"]["host"] == "10.120.18.240"
    assert plan["allocation_page_metadata"]["allocation_private_endpoint_policy"] == "metadata_only_forbidden_off_campus_direct_target"
    assert plan["allocation_page_metadata"]["password_recorded"] is False
    assert plan["safety_boundaries"]["this_plan_decrypts_dpapi"] is False
    assert plan["safety_boundaries"]["this_plan_opens_ssh"] is False
    assert "install_hpc_ssh_credential_from_stdin.ps1" in plan["commands"]["install_provisioning_profile"]
    assert "-User aimslab-TosNrOAl" in plan["commands"]["install_provisioning_profile"]
    assert "-HostName 100.85.169.63" in plan["commands"]["install_provisioning_profile"]
    assert "-Port 1235" in plan["commands"]["install_provisioning_profile"]
    assert "-SocksPort 17897" in plan["commands"]["install_provisioning_profile"]
    assert "bootstrap_hpc_profile_identity.py" in " ".join(plan["commands"]["bootstrap_identity"])
    assert "verify_hpc_profile_readiness.py" in plan["commands"]["post_bootstrap_verification"]
    assert "password" in wire.lower()
    assert "secret-value" not in wire
    assert "10.120.0.0/16" in wire
    assert "Pht1wRlRuP" not in wire


def test_reenrollment_plan_rejects_secret_like_role_account(tmp_path):
    builder = _load_builder()
    readiness = tmp_path / "profile.json"
    _readiness(readiness)

    try:
        builder.build_plan(readiness, role_account="bad account with spaces")
    except ValueError as exc:
        assert "role account" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unsafe role account accepted")


def test_reenrollment_plan_rejects_unsafe_binding_id(tmp_path):
    builder = _load_builder()
    readiness = tmp_path / "profile.json"
    _readiness(readiness)

    try:
        builder.build_plan(readiness, allocation_binding_id="../bad", allocation_generation=1)
    except ValueError as exc:
        assert "allocation binding id" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unsafe binding accepted")
