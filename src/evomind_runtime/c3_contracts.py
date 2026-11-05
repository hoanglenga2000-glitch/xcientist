"""C3 contracts + product-decision stubs (t237).

Scaffold only: no HTTP routes, no DB writes, no tool registration.
PRODUCT_MEMORY_* + Claude UX locks resolved (user t244u / Grok 2026-10-27).
See /workspace/ui-opus55-fix/t237-c3/PRODUCT_MEMORY_DECISIONS.md.
Spec 03 locked snapshot ab810a0e (Grok Claude UX locks in); README ba2d177b.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, NotRequired, TypedDict


class ProductDecisionPending(RuntimeError):
    """Raised when code would need a PRODUCT_MEMORY_* value that is still UNRESOLVED."""


class MemoryDefaultPolicy(str, Enum):
    """PRODUCT_MEMORY_DEFAULT — figma suggests ON; do not ship a default until resolved."""

    UNRESOLVED = "unresolved"
    ON = "on"
    OFF = "off"


class ManualAddMemory(str, Enum):
    """PRODUCT_MEMORY_MANUAL_ADD — whether UI exposes 「添加记忆」."""

    UNRESOLVED = "unresolved"
    YES = "yes"
    NO = "no"


class MemoryScope(str, Enum):
    """PRODUCT_MEMORY_SCOPE — figma suggests USER for v1; schema may reserve project_id."""

    UNRESOLVED = "unresolved"
    USER = "user"
    PROJECT = "project"


# --- user decisions (2026-10-27 09:43 ET, via Grok) -------------------------
# 1. 「使用记忆」defaults ON for users with no stored setting.
PRODUCT_MEMORY_DEFAULT: MemoryDefaultPolicy = MemoryDefaultPolicy.ON
# 2. The agent records memory automatically (incl. training / work experience);
#    no manual 「添加记忆」in C3. Edit / delete per spec 03 stay.
PRODUCT_MEMORY_MANUAL_ADD: ManualAddMemory = ManualAddMemory.NO
# 3. Long-term memory is per USER and readable from every project.
PRODUCT_MEMORY_SCOPE: MemoryScope = MemoryScope.USER
# 3b. Short-term memory (same-thread context) is isolated PER PROJECT: it never
#     crosses a project boundary (and, as today, never crosses conversations).
STM_SCOPE: MemoryScope = MemoryScope.PROJECT

# --- Claude UX locks (Grok, 2026-10-27; was TBD, now LOCKED) -----------------
# Edit textarea / PATCH text max length. Over → truncate + tip (FE); BE refuse > max.
MEMORY_TEXT_MAX_CHARS: int = 2000
# Clear-all: NO undo (03 copy 「不能撤销」). Single-item delete keeps soft-delete + 5s restore.
MEMORY_CLEAR_ALL_UNDO: bool = False
# Date display (FE via existing <time> / helper): same calendar year →「10月26日」;
# other year →「2025年10月26日」. Constants document the locked pattern only.
MEMORY_DATE_SAME_YEAR: str = "{m}月{d}日"
MEMORY_DATE_CROSS_YEAR: str = "{y}年{m}月{d}日"


def require_resolved(decision: Enum, decision_id: str) -> Enum:
    if getattr(decision, "value", decision) == "unresolved" or decision.name == "UNRESOLVED":
        raise ProductDecisionPending(
            f"{decision_id} is UNRESOLVED — waiting on Grok/user card; "
            "see /workspace/ui-opus55-fix/t237-c3/PRODUCT_MEMORY_DECISIONS.md"
        )
    return decision


# --- wire shapes from specs 03 / 05 / 06 (Claude may adjust field names) ----

class MemoryRow(TypedDict):
    id: str
    text: str
    source_task_id: NotRequired[str | None]
    source_task_title: NotRequired[str | None]
    created_at: str
    updated_at: str
    # Not used for LTM (scope is USER). Kept optional for provenance only;
    # never filter LTM reads by it.
    project_id: NotRequired[str | None]


class UserSettingsMemory(TypedDict):
    # Absent / null in storage means "never set" and reads as the product default (ON).
    memory_enabled: bool


class MemorySavedEvent(TypedDict):
    id: str
    text: str


class ProjectFileUnreadable(TypedDict):
    name: str
    error_class: str


class ExperienceSummary(TypedDict):
    count: int


class PlanReference(TypedDict):
    title: str
    summary: str


class PlanProposed(TypedDict):
    references: list[PlanReference]
    steps: list[str]
    needs_compute: bool


class PlanDecision(TypedDict):
    decision: str  # "confirm" | "cancel"


def memory_enabled_for_new_user() -> bool:
    """Return the default switch value — blocked until PRODUCT_MEMORY_DEFAULT resolves."""
    policy = require_resolved(PRODUCT_MEMORY_DEFAULT, "PRODUCT_MEMORY_DEFAULT")
    return policy is MemoryDefaultPolicy.ON


def manual_add_memory_allowed() -> bool:
    choice = require_resolved(PRODUCT_MEMORY_MANUAL_ADD, "PRODUCT_MEMORY_MANUAL_ADD")
    return choice is ManualAddMemory.YES


def memory_scope() -> MemoryScope:
    return require_resolved(PRODUCT_MEMORY_SCOPE, "PRODUCT_MEMORY_SCOPE")  # type: ignore[return-value]


def resolve_memory_enabled(stored: bool | None) -> bool:
    """User's stored switch, else the product default."""
    return memory_enabled_for_new_user() if stored is None else bool(stored)


def clamp_memory_text(text: str) -> tuple[str, bool]:
    """Return (text, truncated?). BE stores at most MEMORY_TEXT_MAX_CHARS."""
    raw = str(text or "")
    if len(raw) <= MEMORY_TEXT_MAX_CHARS:
        return raw, False
    return raw[:MEMORY_TEXT_MAX_CHARS], True


def clear_all_undo_allowed() -> bool:
    """False: clear-all has no undo path (LOCKED)."""
    return bool(MEMORY_CLEAR_ALL_UNDO)


def apply_scope_filter(rows: list[dict[str, Any]], *, project_id: str | None) -> list[dict[str, Any]]:
    """LTM rows readable by a run. USER scope: every live row, regardless of project_id."""
    scope = memory_scope()
    if scope is MemoryScope.USER:
        return list(rows)
    if scope is MemoryScope.PROJECT:
        if not project_id:
            return []
        return [r for r in rows if r.get("project_id") == project_id]
    raise ProductDecisionPending("PRODUCT_MEMORY_SCOPE")
