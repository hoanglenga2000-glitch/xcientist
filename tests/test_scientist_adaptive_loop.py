from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from research_os.agent.messaging import AssistantTurn, ToolCall
from xsci.scientist_adaptive_loop import run_adaptive_scientist_tool_loop


class ScriptedClient:
    def __init__(self, turns: list[AssistantTurn]) -> None:
        self.turns = list(turns)
        self.messages_seen: list[list[dict]] = []

    def is_available(self) -> bool:
        return True

    def send(self, messages, **kwargs):
        self.messages_seen.append(list(messages))
        return self.turns.pop(0)


def tool_turn(round_id: int, name: str, *, text: str = "") -> AssistantTurn:
    call = ToolCall(id=f"call_{round_id}", name=name, input={})
    content = []
    if text:
        content.append({"type": "text", "text": text})
    content.append({
        "type": "tool_use",
        "id": call.id,
        "name": call.name,
        "input": {},
    })
    return AssistantTurn(
        text=text,
        tool_calls=[call],
        stop_reason="tool_use",
        raw_content=content,
        provider="openai",
        model="test-model",
    )


def final_turn(text: str) -> AssistantTurn:
    return AssistantTurn(
        text=text,
        tool_calls=[],
        stop_reason="end_turn",
        raw_content=[{"type": "text", "text": text}],
        provider="openai",
        model="test-model",
    )


def plan() -> dict:
    return {
        "intent": {"kind": "planning"},
        "autonomy_level": "planner_observer",
        "tool_sequence": ["scientist_context_packet", "data_check", "scientist_workplan"],
        "scientific_critique": {"decision": "observe_then_plan"},
        "readiness": {"blocking_gates": []},
        "stop_conditions": ["stop before training"],
        "requirement_ledger": {
            "requirements": [
                {
                    "id": "repair_missing_data",
                    "status": "pending",
                    "mapped_tools": ["scientist_repair_plan"],
                }
            ]
        },
    }


def test_adaptive_loop_replans_after_tool_failure(tmp_path: Path):
    client = ScriptedClient([
        tool_turn(1, "data_check", text="First inspect the data contract."),
        tool_turn(2, "scientist_repair_plan", text="The data check failed, so diagnose it."),
        final_turn("The data gate is blocked; follow the repair plan before training."),
    ])
    executed: list[str] = []

    def dispatch(name, session, root):
        executed.append(name)
        if name == "data_check":
            return {"ok": False, "tool": name, "message": "train.csv is missing"}
        return {
            "ok": True,
            "tool": name,
            "message": "repair plan created",
            "artifact_path": str(root / ".xsci" / "scientist_repair_plan.json"),
        }

    result = run_adaptive_scientist_tool_loop(
        SimpleNamespace(selected_task="house-prices"),
        tmp_path,
        goal="Analyze the missing data and recover safely",
        turn_plan=plan(),
        client=client,
        dispatch=dispatch,
    )

    assert result["ok"] is True
    assert result["used"] is True
    assert executed == ["data_check", "scientist_repair_plan"]
    assert result["failure_observed"] is True
    assert result["replanned_after_failure"] is True
    assert result["requirement_resolution"]["resolved"] == ["repair_missing_data"]
    assert result["status"] == "completed"
    assert result["no_training_started"] is True
    persisted = json.loads((tmp_path / ".xsci" / "scientist_adaptive_tool_loop.json").read_text(encoding="utf-8"))
    assert "runtime_results" not in persisted
    assert persisted["executed_tools"] == executed


def test_adaptive_loop_blocks_unknown_and_duplicate_tools(tmp_path: Path):
    client = ScriptedClient([
        tool_turn(1, "delete_repository"),
        tool_turn(2, "system_status"),
        tool_turn(3, "system_status"),
        final_turn("The bounded status evidence is sufficient."),
    ])
    executed: list[str] = []

    def dispatch(name, session, root):
        executed.append(name)
        return {"ok": True, "tool": name, "message": "ready", "api_key": "must-not-persist"}

    result = run_adaptive_scientist_tool_loop(
        SimpleNamespace(selected_task="house-prices"),
        tmp_path,
        goal="Inspect status",
        turn_plan=plan(),
        client=client,
        dispatch=dispatch,
    )

    assert executed == ["system_status"]
    assert result["unsafe_or_unknown_calls_blocked"] == 1
    assert result["duplicate_calls_blocked"] == 1
    persisted = (tmp_path / ".xsci" / "scientist_adaptive_tool_loop.json").read_text(encoding="utf-8")
    assert "must-not-persist" not in persisted
    assert result["official_submit"].startswith("blocked")


def test_adaptive_loop_marks_open_requirements_as_needing_continuation(tmp_path: Path):
    client = ScriptedClient([final_turn("I inspected nothing and should not claim closure.")])
    result = run_adaptive_scientist_tool_loop(
        SimpleNamespace(selected_task="house-prices"),
        tmp_path,
        goal="Repair the missing data contract",
        turn_plan=plan(),
        client=client,
        dispatch=lambda *args: {"ok": True},
    )
    assert result["ok"] is True
    assert result["status"] == "needs_continuation"
    assert result["open_requirements"] == ["repair_missing_data"]
    assert result["next_safe_action"] == "evomind continuation-status"


def test_adaptive_loop_does_not_close_requirement_on_ok_but_blocked_result(tmp_path: Path):
    blocked_plan = plan()
    blocked_plan["requirement_ledger"]["requirements"][0]["mapped_tools"] = ["scientist_execution_contract"]
    client = ScriptedClient([
        tool_turn(1, "scientist_execution_contract"),
        final_turn("The contract is still NO-GO, so the requirement remains open."),
    ])

    result = run_adaptive_scientist_tool_loop(
        SimpleNamespace(selected_task="house-prices"),
        tmp_path,
        goal="Decide whether execution is ready",
        turn_plan=blocked_plan,
        client=client,
        dispatch=lambda *args: {
            "ok": True,
            "tool": "scientist_execution_contract",
            "status": "blocked",
            "go_no_go": "NO-GO",
            "blocking_gates": ["gpu_auth_pending"],
        },
    )

    assert result["status"] == "needs_continuation"
    assert result["requirement_resolution"]["resolved"] == []
    assert result["requirement_resolution"]["blocked"] == ["repair_missing_data"]
    signals = result["requirement_resolution"]["tool_blocking_signals"]["scientist_execution_contract"]
    assert "go_no_go=no-go" in signals
    assert "blocking_gates:gpu_auth_pending" in signals


def test_adaptive_loop_degrades_cleanly_without_provider(tmp_path: Path):
    class MissingClient:
        def is_available(self) -> bool:
            return False

    result = run_adaptive_scientist_tool_loop(
        SimpleNamespace(selected_task=""),
        tmp_path,
        goal="Inspect the project",
        turn_plan=plan(),
        client=MissingClient(),
    )
    assert result["ok"] is True
    assert result["available"] is False
    assert result["used"] is False
    assert result["stop_reason"] == "provider_unavailable"
