"""Deep AI-Scientist agent for research_os.

This package turns the engine's primitives into TOOLS that an LLM drives directly
(the "deep" control model): the model reasons, calls tools, reads results, and
decides the next move — instead of a fixed Python ladder calling the LLM as a
mere code generator.

The no-fabrication invariants are enforced OUTSIDE the model, in the tool
handlers and the deterministic engine functions they wrap:
  * a run's success/exit_code is decided by the Runner, never by the agent;
  * promotion is decided by ``SearchGraph.decide_promotion`` (a crashed run is
    never promotable, even if it flushed a score);
  * a conclusion is gated by ``claim_audit.audit_claim`` (reject on thin evidence);
  * Kaggle submission is always blocked behind a human gate.

So the agent may *request* these; it can never *fake* the outcome.
"""
from __future__ import annotations

from .aibuild_v1 import (
    build_aibuild_run,
    default_roles,
    read_current_run_pointer,
    run_directory,
    should_use_multi_agent,
    write_current_run_pointer,
)
from .guardrails import GuardrailDecision, ToolGuardrailController
from .ledger import MessageLedger
from .memory_library import MemoryLibrary
from .messaging import AgentMessageClient, ToolCall, ToolResult, ToolSpec
from .multi_agent import (
    AgentResult,
    AgentRoleSpec,
    AgentTask,
    HandoffEnvelope,
    MultiAgentStore,
    MultiAgentSupervisor,
    SupervisorRun,
    build_idempotency_key,
    create_run,
    validate_task_graph,
)
from .report import build_report, write_report
from .session import AgentSession, AgentSessionConfig
from .subagents import AUDIT_TOOLS, SubAgentResult, spawn_audit_agent
from .tools import ResearchToolbox

__all__ = [
    "AUDIT_TOOLS",
    "AgentMessageClient",
    "AgentResult",
    "AgentRoleSpec",
    "AgentSession",
    "AgentSessionConfig",
    "AgentTask",
    "GuardrailDecision",
    "HandoffEnvelope",
    "MemoryLibrary",
    "MessageLedger",
    "MultiAgentStore",
    "MultiAgentSupervisor",
    "ResearchToolbox",
    "SubAgentResult",
    "SupervisorRun",
    "ToolCall",
    "ToolGuardrailController",
    "ToolResult",
    "ToolSpec",
    "build_report",
    "build_idempotency_key",
    "build_aibuild_run",
    "create_run",
    "default_roles",
    "spawn_audit_agent",
    "read_current_run_pointer",
    "run_directory",
    "should_use_multi_agent",
    "validate_task_graph",
    "write_report",
    "write_current_run_pointer",
]
