from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

AgentOutputStatus = Literal["success", "failed", "waiting_gate", "needs_human", "skipped"]


@dataclass(slots=True)
class CodePlan:
    plan_id: str
    task_id: str
    provider: str
    steps: list[str]
    risks: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CodeArtifact:
    artifact_id: str
    provider: str
    generated_files: list[Path]
    notes: str


@dataclass(slots=True)
class ReviewResult:
    status: str
    findings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PatchResult:
    patch_id: str
    source_agent: str
    patch_diff: str
    review_status: str = "pending"
    modified_files: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentPlan:
    plan_id: str
    hypothesis: str
    next_actions: list[str]


@dataclass(slots=True)
class AgentProviderStatus:
    code_agent: str
    codex_ready: bool
    claude_code_ready: bool


@dataclass(slots=True)
class AgentInput:
    task_id: str
    run_id: str | None
    stage: str
    research_question: str
    task_profile: dict[str, Any]
    current_artifacts: list[dict[str, Any]] = field(default_factory=list)
    previous_runs: list[dict[str, Any]] = field(default_factory=list)
    memory_context: list[dict[str, Any]] = field(default_factory=list)
    gate_status: dict[str, Any] = field(default_factory=dict)
    user_constraints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentOutput:
    status: AgentOutputStatus
    summary: str
    decisions: list[str] = field(default_factory=list)
    generated_artifacts: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    error: str | None = None
    suggested_gate: dict[str, Any] | None = None


@dataclass(slots=True)
class AgentTrace:
    timestamp: str
    agent: str
    stage: str
    action: str
    input_summary: str
    output_summary: str
    artifacts_generated: list[str] = field(default_factory=list)
    evidence_linked: list[str] = field(default_factory=list)
    status: AgentOutputStatus = "success"
    duration_ms: int = 0
    error: str | None = None
    run_id: str | None = None

    @classmethod
    def from_io(
        cls,
        *,
        agent: str,
        stage: str,
        action: str,
        agent_input: AgentInput,
        output: AgentOutput,
        duration_ms: int,
    ) -> "AgentTrace":
        return cls(
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            agent=agent,
            stage=stage,
            action=action,
            input_summary=f"task={agent_input.task_id}; stage={agent_input.stage}; artifacts={len(agent_input.current_artifacts)}",
            output_summary=output.summary,
            artifacts_generated=output.generated_artifacts,
            evidence_linked=output.evidence_refs,
            status=output.status,
            duration_ms=duration_ms,
            error=output.error,
            run_id=agent_input.run_id,
        )
