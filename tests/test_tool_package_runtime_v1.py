from __future__ import annotations

import sqlite3

import pytest

from evomind_runtime.ecosystem import ExternalExecutionDisabled, ToolPackage
from evomind_runtime.super_agent_store import SuperAgentStore
from evomind_runtime.tool_package_runtime import (
    ToolPackageApprovalRequired,
    ToolPackageRuntime,
    ToolPackageStateError,
)


def _manifest(
    run_id: str,
    *,
    package_id: str = "mount-auditor",
    risk_class: str = "observe",
    operation: str = "audit_mount",
    reversible: bool = True,
) -> dict[str, object]:
    return {
        "schema": "evomind.tool_package.v1",
        "package_id": package_id,
        "run_id": run_id,
        "version": "1.0.0",
        "provider_id": "run-sandbox",
        "capabilities": [
            {
                "name": operation,
                "operations": [operation],
                "input_schema": {
                    "type": "object",
                    "properties": {"directory_id": {"type": "string"}},
                    "required": ["directory_id"],
                },
                "risk_class": risk_class,
                "idempotency": "idempotent",
            }
        ],
        "permissions": {"directories": ["declared-only"]},
        "tests": [{"id": "contract", "expected": "pass"}],
        "canary": {"scope": "isolated-fixture"},
        "rollback": {
            "action": "discard-run-package",
            "reversible": reversible,
        },
    }


def _verified(runtime: ToolPackageRuntime, package_id: str = "mount-auditor") -> None:
    runtime.record_tests(
        package_id,
        passed=True,
        evidence={"suite": "contract", "suite_sha256": "a" * 64},
    )
    runtime.record_canary(
        package_id,
        passed=True,
        evidence={"scope": "isolated", "receipt_sha256": "b" * 64},
    )


def test_lifecycle_persists_and_rehydrates_without_external_execution(tmp_path) -> None:
    path = tmp_path / "runtime.sqlite3"
    store = SuperAgentStore(path)
    runtime = ToolPackageRuntime(store, run_id="run-a")

    staged = runtime.stage(_manifest("run-a"))
    assert staged["promotion_status"] == "draft"
    assert staged["external_install_enabled"] is False
    assert staged["external_execution_enabled"] is False

    _verified(runtime)
    promoted = runtime.promote("mount-auditor")
    assert promoted["promotion_status"] == "promoted"
    assert promoted["approval_gate"]["status"] == "not_required"
    with pytest.raises(ToolPackageStateError, match="immutable"):
        runtime.reject("mount-auditor")
    with pytest.raises(ExternalExecutionDisabled):
        runtime.install("mount-auditor")
    with pytest.raises(ExternalExecutionDisabled):
        runtime.execute("mount-auditor")

    store.close()
    reopened_store = SuperAgentStore(path)
    reopened = ToolPackageRuntime(reopened_store, run_id="run-a")
    assert reopened.get("mount-auditor")["promotion_status"] == "promoted"
    reopened_store.close()


def test_test_and_canary_evidence_must_be_hash_bound(tmp_path) -> None:
    store = SuperAgentStore(tmp_path / "runtime.sqlite3")
    runtime = ToolPackageRuntime(store, run_id="run-a")
    runtime.stage(_manifest("run-a"))

    with pytest.raises(ToolPackageStateError, match="SHA-256"):
        runtime.record_tests("mount-auditor", passed=True, evidence={"status": "passed"})
    tested = runtime.record_tests(
        "mount-auditor",
        passed=True,
        evidence_sha256="c" * 64,
    )
    assert tested["test_evidence"]["evidence_sha256"] == "c" * 64
    with pytest.raises(ToolPackageStateError, match="SHA-256"):
        runtime.record_canary("mount-auditor", passed=True, evidence={"status": "passed"})
    store.close()


def test_stage_accepts_tool_package_as_a_reparsed_internal_copy(tmp_path) -> None:
    store = SuperAgentStore(tmp_path / "runtime.sqlite3")
    package = ToolPackage.from_manifest(_manifest("run-a"), run_id="run-a")
    runtime = ToolPackageRuntime(store, run_id="run-a")

    state = runtime.stage(package)
    package.promotion_status = "rejected"

    assert state["promotion_status"] == "draft"
    assert runtime.get("mount-auditor")["promotion_status"] == "draft"
    store.close()


def test_run_scope_uses_distinct_persisted_rows_and_rejects_cross_run_manifest(tmp_path) -> None:
    path = tmp_path / "runtime.sqlite3"
    store = SuperAgentStore(path)
    run_a = ToolPackageRuntime(store, run_id="run-a")
    run_b = ToolPackageRuntime(store, run_id="run-b")
    run_a.stage(_manifest("run-a"))
    run_b.stage(_manifest("run-b"))

    assert run_a.get("mount-auditor")["run_id"] == "run-a"
    assert run_b.get("mount-auditor")["run_id"] == "run-b"
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tool_packages_v2").fetchone()[0] == 2
    with pytest.raises(Exception, match="different run"):
        run_a.stage(_manifest("run-b", package_id="other"))
    store.close()


@pytest.mark.parametrize(
    ("risk_class", "operation", "reversible", "expected_reason"),
    [
        ("destructive", "delete_mount", True, "risk:destructive"),
        ("observe", "audit_mount", False, "irreversible"),
    ],
)
def test_risk_or_irreversible_package_requires_exact_approval_hash(
    tmp_path,
    risk_class: str,
    operation: str,
    reversible: bool,
    expected_reason: str,
) -> None:
    store = SuperAgentStore(tmp_path / "runtime.sqlite3")
    runtime = ToolPackageRuntime(store, run_id="run-a")
    runtime.stage(
        _manifest(
            "run-a",
            risk_class=risk_class,
            operation=operation,
            reversible=reversible,
        )
    )
    _verified(runtime)

    with pytest.raises(ToolPackageApprovalRequired, match="approval"):
        runtime.promote("mount-auditor")
    waiting = runtime.get("mount-auditor")
    assert waiting["promotion_status"] == "draft"
    assert waiting["approval_gate"]["status"] == "waiting_exact_gate"
    assert expected_reason in waiting["approval_reasons"]

    promoted = runtime.promote(
        "mount-auditor",
        approval_evidence_sha256="d" * 64,
    )
    assert promoted["promotion_status"] == "promoted"
    assert promoted["approval_gate"]["approval_evidence_sha256"] == "d" * 64
    store.close()


def test_noop_rollback_cannot_bypass_irreversible_gate(tmp_path) -> None:
    store = SuperAgentStore(tmp_path / "runtime.sqlite3")
    runtime = ToolPackageRuntime(store, run_id="run-a")
    manifest = _manifest("run-a")
    manifest["rollback"] = {"action": "noop", "reversible": True}
    runtime.stage(manifest)
    _verified(runtime)

    with pytest.raises(ToolPackageApprovalRequired):
        runtime.promote("mount-auditor")
    assert "irreversible" in runtime.get("mount-auditor")["approval_reasons"]
    store.close()


def test_failed_verification_blocks_promotion_and_reject_is_terminal(tmp_path) -> None:
    store = SuperAgentStore(tmp_path / "runtime.sqlite3")
    runtime = ToolPackageRuntime(store, run_id="run-a")
    runtime.stage(_manifest("run-a"))
    failed = runtime.record_tests(
        "mount-auditor",
        passed=False,
        evidence={"failure_sha256": "e" * 64},
    )
    assert failed["promotion_status"] == "blocked"
    with pytest.raises(ToolPackageStateError, match="tests"):
        runtime.promote("mount-auditor")

    # A new verified test result is reversible metadata and may unblock the draft.
    _verified(runtime)
    rejected = runtime.reject("mount-auditor")
    assert rejected["promotion_status"] == "rejected"
    with pytest.raises(ToolPackageStateError, match="immutable"):
        runtime.record_tests(
            "mount-auditor",
            passed=True,
            evidence={"suite_sha256": "f" * 64},
        )
    store.close()
