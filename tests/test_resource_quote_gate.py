"""A plan can read the ledger and cannot publish an unverified zero fee.

The resources page and a quote question are local reads. No connection is opened.
"""
import json
from pathlib import Path

from evomind_runtime.personal_tool_boundary import (
    annotate_unverified_zero_cost,
    offered,
    planning_resource_clause,
    public_compute_status,
    zero_cost_was_confirmed,
)
from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.tools import ToolContext, _hpc_execution_budget_status, _quote_facts, build_default_registry
from research_os.agent import messaging


TENANT = "tenant_" + "ab" * 12


def _binding(root: Path, *, state: str = "active") -> dict:
    record = {
        "schema": "evomind.tenant_hpc_binding.v1",
        "tenant_id": TENANT,
        "username": "alice",
        "state": state,
        "job_id": 98270,
        "allocation_generation": 2,
        "credential_profile": f"{TENANT}_job98270_g2",
        "profile_instance_id": "a" * 32,
        "allocation_binding_id": "b" * 16,
    }
    path = root / "tenants" / TENANT
    path.mkdir(parents=True)
    (path / "hpc-binding.json").write_text(json.dumps(record), encoding="utf-8")
    return record


def test_personal_plan_can_read_the_ledger_and_still_cannot_run_python(tmp_path):
    assert offered(TENANT, "hpc_execution_budget_status") is True
    assert offered(TENANT, "shell_exec") is False
    assert offered(TENANT, "training_route") is False
    clause = planning_resource_clause(TENANT)
    assert "incremental_price_known" in clause and "do not say the cost is 0" in clause
    registry = build_default_registry()
    names = {item.name for item in registry.specs_for_prompt("这次基线的费用和配额是多少")}
    assert "hpc_execution_budget_status" in names
    runtime = AgentRuntime(tmp_path)
    try:
        session = {"metadata": {"http_access_scope": {"tenant_id": TENANT}}, "workspace_root": "/tmp"}
        offered_names = {item.name for item in runtime._message_tool_specs("报价，不要训练", session)}
    finally:
        runtime.close()
    assert "hpc_execution_budget_status" in offered_names
    assert "shell_exec" not in offered_names
    assert "training_route" not in offered_names


def test_ledger_read_reports_unverified_compute_and_refuses_a_zero_price(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSTATION_BYOA_STATE_ROOT", str(tmp_path))
    monkeypatch.delenv("EVOMIND_AIBUILD_POLICY_FILE", raising=False)
    monkeypatch.delenv("WORKSTATION_BYOA_RUNTIME_ROOT", raising=False)
    _binding(tmp_path)
    context = ToolContext("run_quote", tmp_path, tmp_path, tmp_path, tmp_path, None, {
        "http_access_scope": {"tenant_id": TENANT, "owner_principal_id": "alice"},
    })
    result = _hpc_execution_budget_status({}, context)
    assert result.ok
    assert result.content["compute_status"] == "计算资源未通过核对"
    assert result.content["compute_verified"] is False
    assert result.content["incremental_price_known"] is False
    assert result.content["zero_cost_confirmed"] is False
    assert result.content["python_execution_offered"] is False
    assert "engineering_used_seconds" not in result.content
    kept = _quote_facts({"engineering_used_seconds": 7200, "used_seconds": 73700}, context)
    assert kept["engineering_used_seconds"] == 7200 and kept["zero_cost_confirmed"] is False
    body = json.dumps(result.content)
    assert "98270" not in body and "credential_profile" not in body
    assert not (tmp_path / "gpu_budget.sqlite3").exists()


def test_completed_receipt_is_the_only_verified_label(tmp_path):
    record = _binding(tmp_path)
    assert public_compute_status(TENANT, tmp_path)["compute_status"] == "计算资源未通过核对"
    request_root = tmp_path / "tenants" / TENANT / "activation-requests"
    request_root.mkdir()
    (request_root / "g2.request.json").write_text(json.dumps({
        **{key: record[key] for key in ("tenant_id", "job_id", "credential_profile", "allocation_binding_id", "allocation_generation", "profile_instance_id")},
        "owner_principal_id": "alice",
        "state": "completed",
    }), encoding="utf-8")
    assert public_compute_status(TENANT, tmp_path)["compute_verified"] is False
    receipt = tmp_path / "tenants" / TENANT / "activation-receipts"
    receipt.mkdir()
    (receipt / "g2.json").write_text("{}", encoding="utf-8")
    status = public_compute_status(TENANT, tmp_path)
    assert status == {"compute_status": "计算资源已核对", "compute_verified": True}
    _binding_frozen = tmp_path / "tenants" / TENANT / "hpc-binding.json"
    record["state"] = "frozen"
    _binding_frozen.write_text(json.dumps(record), encoding="utf-8")
    assert public_compute_status(TENANT, tmp_path)["compute_status"] == "计算资源暂不可用"
    assert public_compute_status(TENANT, tmp_path / "missing")["compute_status"] == "未设置计算资源"


def test_unverified_zero_cost_is_marked_and_a_plan_without_one_is_not():
    claim = "这次费用为 0，不消耗配额。"
    marked = annotate_unverified_zero_cost(claim)
    assert "不能当成 0" in marked
    assert annotate_unverified_zero_cost(claim, confirmed=True) == claim
    plan = "已完成计划，未启动训练，未申请任何计费运行审批。"
    assert annotate_unverified_zero_cost(plan) == plan
    assert zero_cost_was_confirmed([{"tool_name": "hpc_execution_budget_status", "status": "completed", "result": {"content": {"zero_cost_confirmed": False}}}]) is False
    assert zero_cost_was_confirmed([{"tool_name": "hpc_execution_budget_status", "status": "completed", "result": {"content": {"zero_cost_confirmed": True}}}]) is True


def test_published_answer_marks_an_unverified_zero_fee(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-fixture")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9/anthropic")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("EVOLUTION_PRIMARY_PROVIDER", "anthropic")
    monkeypatch.setenv("EVOLUTION_PROVIDER_STRICT", "1")
    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    def post(url, headers, payload, timeout):
        return {
            "model": payload["model"], "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "这次费用为 0，不消耗配额。"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(messaging, "_post_json", post)
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="这次基线的费用和配额是多少", start=False)
        runtime.assistant._execute(run["id"], resume=False)
        answer = runtime.assistant.snapshot(run["id"])["answer"]
        assert "费用为 0" in answer
        assert "不能当成 0" in answer
        assert runtime.assistant.snapshot(run["id"])["status"] == "completed"
    finally:
        runtime.close()
