from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class PermissionLevel(str, Enum):
    OBSERVE = "observe"
    WORKSPACE_WRITE = "workspace-write"
    COMPUTER_APPROVED = "computer-approved"
    FULL_AUTO = "full-auto"


class SessionStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    capability: str
    version: str = "1.0.0"
    read_only: bool = True
    supports_cancel: bool = False
    max_result_bytes: int = 64_000
    available: bool = True
    unavailable_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Session:
    id: str
    workspace_root: str
    permission_level: str = PermissionLevel.WORKSPACE_WRITE.value
    status: str = SessionStatus.CREATED.value
    title: str = ""
    objective: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    parent_session_id: str = ""
    selected_model_policy: str = "balanced"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCall:
    id: str
    session_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: str = "requested"
    created_at: str = field(default_factory=utc_now)
    started_at: str = ""
    completed_at: str = ""
    approval_id: str = ""
    idempotency_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    tool_call_id: str
    ok: bool
    content: Any
    summary: str
    error: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PermissionDecision:
    allowed: bool
    requires_approval: bool
    risk_level: str
    reason: str
    scope: dict[str, Any]
    normalized_arguments: dict[str, Any]
    reversible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ApprovalRequest:
    id: str
    session_id: str
    tool_call_id: str
    tool_name: str
    argument_fingerprint: str
    normalized_arguments: dict[str, Any]
    impact_scope: dict[str, Any]
    risk_level: str
    reversible: bool
    status: str = "pending"
    created_at: str = field(default_factory=utc_now)
    expires_at: str = ""
    decided_at: str = ""
    decision_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunEvent:
    id: str
    session_id: str
    seq: int
    event_type: str
    payload: dict[str, Any]
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Checkpoint:
    id: str
    session_id: str
    seq: int
    state: dict[str, Any]
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactRef:
    id: str
    session_id: str
    sha256: str
    path: str
    media_type: str
    bytes: int
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelDecision:
    purpose: str
    provider: str
    model: str
    reason: str
    fallback_chain: list[str]
    cross_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Task:
    id: str
    session_id: str
    goal: str
    role: str
    dependencies: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    status: str = "pending"
    budget: dict[str, Any] = field(default_factory=dict)
    artifact_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

