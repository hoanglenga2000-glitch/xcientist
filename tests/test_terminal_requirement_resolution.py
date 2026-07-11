from __future__ import annotations

from xsci.terminal_agent import _resolve_requirement_ledger_after_tools


def _ledger(*, gate: str, evidence: list[str]) -> dict:
    return {
        "schema": "evomind.ai_scientist.requirement_ledger.v1",
        "requirements": [{
            "id": "target",
            "description": "test requirement",
            "status": "planned",
            "gate": gate,
            "evidence_needed": evidence,
            "mapped_tools": ["scientist_execution_contract"],
        }],
    }


def test_ok_tool_with_blocking_result_does_not_close_acceptance_requirement():
    result = _resolve_requirement_ledger_after_tools(
        _ledger(gate="execution_contract_gate", evidence=["scientist_execution_contract.json"]),
        executed=[{"tool": "scientist_execution_contract", "ok": True}],
        artifacts=[".xsci/scientist_execution_contract.json"],
        blockers=[],
        tool_results={
            "scientist_execution_contract": {
                "ok": True,
                "status": "blocked",
                "go_no_go": "NO-GO",
                "blocking_gates": ["gpu_auth_pending"],
            }
        },
    )

    requirement = result["requirements"][0]
    assert requirement["status"] == "blocked"
    assert result["open_requirements"] == ["target"]
    assert requirement["execution_evidence"]["mapped_tool_hits"] == ["scientist_execution_contract"]
    assert requirement["execution_evidence"]["mapped_tool_clear_hits"] == []
    assert "gpu_auth_pending" in requirement["reason"]


def test_artifact_contract_can_close_after_tool_materializes_audit():
    ledger = _ledger(gate="capability_audit_gate", evidence=["scientist_self_audit.json"])
    ledger["requirements"][0]["mapped_tools"] = ["scientist_self_audit"]
    result = _resolve_requirement_ledger_after_tools(
        ledger,
        executed=[{"tool": "scientist_self_audit", "ok": True}],
        artifacts=[".xsci/scientist_self_audit.json"],
        blockers=[],
        tool_results={
            "scientist_self_audit": {
                "ok": True,
                "overall_score": 59,
                "behavior_benchmark_status": "not_measured",
            }
        },
    )

    assert result["requirements"][0]["status"] == "satisfied"
    assert result["satisfied_requirements"] == ["target"]
