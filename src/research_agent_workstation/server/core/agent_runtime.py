from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Protocol

from ..schemas.agent import AgentInput, AgentOutput, AgentTrace
from .json_utils import append_jsonl, write_json


class RuntimeAgent(Protocol):
    name: str
    stage: str

    def run(self, agent_input: AgentInput) -> AgentOutput:
        ...


class AgentRuntime:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.traces: list[AgentTrace] = []

    def execute(self, agent: RuntimeAgent, agent_input: AgentInput, action: str = "run") -> AgentOutput:
        started = perf_counter()
        try:
            output = agent.run(agent_input)
        except Exception as exc:
            output = AgentOutput(status="failed", summary=f"{agent.name} failed: {exc}", error=str(exc))
        duration_ms = int((perf_counter() - started) * 1000)
        self.traces.append(
            AgentTrace.from_io(
                agent=agent.name,
                stage=agent.stage,
                action=action,
                agent_input=agent_input,
                output=output,
                duration_ms=duration_ms,
            )
        )
        return output

    def flush(self, output_dir: Path) -> None:
        trace_path = output_dir / "agent_trace.jsonl"
        if trace_path.exists():
            trace_path.unlink()
        for trace in self.traces:
            append_jsonl(trace_path, trace)
        write_json(output_dir / "agent_trace.json", {"traces": [asdict(trace) for trace in self.traces]})
