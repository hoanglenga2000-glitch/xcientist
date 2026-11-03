"""t237 C3/C4: experience entry (spec 06).

C3: summary, plan card, recorded decision.
C4: read-only retrieval via personal-allowed research tools; confirm+needs_compute
raises a Chinese product approval (no HPC launch in this tip).
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import sqlite3
from typing import Any

from .models import ApprovalRequest, SessionStatus, new_id, utc_now
from .policy import argument_fingerprint
from .tenant_access import AccessError, Principal

INTENTS = frozenset({"experience"})
DECISIONS = frozenset({"confirm", "cancel"})
NO_REFERENCES_ANSWER = "我先列了一个计划，你确认后我再继续。"
HAS_REFERENCES_ANSWER = "已根据相近经验整理计划，你确认后我再继续。"
STUB_STEPS = ("整理你的目标和已有条件", "对照相近的经验给出改进方案", "确认后再决定是否需要算力")
# Synthetic product tool for compute approval — never starts HPC in unit/pytest.
EXPERIENCE_COMPUTE_TOOL = "experience_compute"
HPC_OUTPUT_ROOT = "/hpc2hdd/home/aimslab/jinghw"
READ_TOOLS = ("research_capabilities", "skill_list", "verified_context")
# C4b: asks that need compute or a write after confirm (06 §4 step 4). Matched per clause.
_COMPUTE_ASK = re.compile(
    r"训练|微调|跑实验|算力|gpu|hpc|写回|写入(?:经验库|记忆)|memory\s*write-?back|fine-?tun\w*|\btrain(?:ing)?\b",
    re.I,
)
_NEGATION = re.compile(r"不要|不用|不需要|无需|不必|别|勿|禁止|\bdon'?t\b|\bdo not\b|\bno\b|\bwithout\b|\bnot\b", re.I)
_CLAUSE_SPLIT = re.compile(r"[，,。.;；!！?？\n]|但是|但|不过|而是")


def valid_intent(value: Any) -> str:
    """'' (no intent) or one of INTENTS; anything else is a 400."""
    if value in (None, ""):
        return ""
    if not isinstance(value, str) or value not in INTENTS:
        raise AccessError(400, "invalid_intent")
    return value


def summary(principal: Principal | None) -> dict:
    """Entry count. No owner-scoped store yet → count unknown (null).

    06 §2: unreadable/unknown count shows the 有经验 entry (line + 用经验库开发), not the empty
    state. Returning 0 would hide the only live entry after C4b E1 XOR (ACCEPT C4b blocker).
    """
    if principal is None:
        raise AccessError(403, "experience_principal_required")
    return {"count": None}


def proposed_plan(*, references: list[dict] | None = None, needs_compute: bool = False) -> dict:
    """PlanProposed (06 §4.1). Default needs_compute=False so live §2.5 smoke stays completed after confirm."""
    return {
        "references": list(references or []),
        "steps": list(STUB_STEPS),
        "needs_compute": bool(needs_compute),
    }


def detect_compute_intent(prompt: str) -> bool:
    """True when some clause asks for training / compute / memory writeback without negating it.

    A negation earlier in the same clause cancels every ask after it ("不要启动训练或算力").
    """
    for clause in _CLAUSE_SPLIT.split(str(prompt or "")):
        for match in _COMPUTE_ASK.finditer(clause):
            if not _NEGATION.search(clause[:match.start()]):
                return True
    return False


def _chinese_ref(title: str, summary_text: str) -> dict[str, str]:
    return {"title": title[:80], "summary": summary_text[:120]}


def retrieve_plan(runtime: Any, run_id: str) -> dict:
    """Invoke read-only research tools (policy a). Build references from verified_context hits only.

    research_capabilities / skill_list prove the tools run without inventing fake experience titles.
    Always returns a plan; caller marks the run completed. Never offers/starts write/train/HPC.
    """
    references: list[dict] = []
    for name in READ_TOOLS:
        try:
            result = runtime.invoke_tool(run_id, name, {})
        except Exception:
            continue
        if not isinstance(result, dict) or result.get("status") == "failed":
            # Boundary / missing tool must not block the experience run (§2.5 completed).
            continue
        payload = (result.get("result") or {}).get("output") or (result.get("result") or {})
        if name != "verified_context" or not isinstance(payload, dict):
            continue
        # Prefer explicit experience-like fields when present; otherwise skip inventing titles.
        for item in payload.get("experiences") or payload.get("references") or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("name") or "").strip()
            summary_text = str(item.get("summary") or item.get("detail") or "").strip()
            if title and summary_text:
                references.append(_chinese_ref(title, summary_text))
        task = str(payload.get("selected_task") or payload.get("task_title") or "").strip()
        if task and not references:
            note = str(payload.get("summary") or payload.get("status") or "已核对历史记录").strip()
            if note and "no matching" not in note.lower() and "empty" not in note.lower():
                references.append(_chinese_ref(task[:80], note[:120]))
    # Live smoke plan-only prompts stay needs_compute=False (H_m6_confirm expects completed).
    return proposed_plan(references=references, needs_compute=False)


def answer_for(plan: dict) -> str:
    return HAS_REFERENCES_ANSWER if plan.get("references") else NO_REFERENCES_ANSWER



def _public_approval_event(payload: dict) -> dict:
    """Copy an approval event payload without output_root (path contains /hpc2hdd/…)."""
    out = dict(payload)
    for key in ("normalized_arguments", "arguments", "impact_scope", "metadata", "meta"):
        value = out.get(key)
        if isinstance(value, dict) and "output_root" in value:
            cleaned = dict(value)
            cleaned.pop("output_root", None)
            out[key] = cleaned
    return out


def _pending_compute_approval(runtime: Any, run_id: str, plan: dict) -> dict:
    """Exact-approval card for compute; does not launch HPC (C4 tip)."""
    args = {
        "purpose": "experience_plan_compute",
        "estimated_cost": 0,
        "budget": 0,
        "steps": list(plan.get("steps") or []),
    }
    approval = ApprovalRequest(
        new_id("approval"),
        run_id,
        new_id("tool"),
        EXPERIENCE_COMPUTE_TOOL,
        argument_fingerprint(EXPERIENCE_COMPUTE_TOOL, args),
        args,
        {"personal_tool_boundary": False, "experience_compute": True},
        "low",
        True,
        expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
    )
    runtime.store.put_approval(approval)
    runtime.store.update_session(run_id, status=SessionStatus.WAITING_APPROVAL.value)
    runtime.store.update_assistant_run(
        run_id,
        status=SessionStatus.WAITING_APPROVAL.value,
        updated_at=utc_now(),
    )
    # Public /events must not contain smoke FORBIDDEN_EVENT_MARKERS (bare "hpc").
    # C6-sec: the stored approval carries no output_root either; the event copy is still scrubbed defensively.
    runtime.store.append_event(run_id, "approval.requested", _public_approval_event(approval.to_dict()))
    return approval.to_dict()


def run_plan_pending(runtime: Any, run: dict) -> bool:
    """C4c: an experience plan card the user has not decided yet (the run itself is already completed)."""
    session = runtime.store.get_session(str(run.get("session_id") or "")) or {}
    meta = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
    if meta.get("intent") != "experience" or not meta.get("plan_proposed"):
        return False
    return not PlanDecisions(runtime.runtime_root).get(str(run.get("id") or ""))


class PlanDecisions:
    def __init__(self, runtime_root: Path):
        self.path = Path(runtime_root) / "experience_plans.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS plan_decisions "
                "(run_id TEXT PRIMARY KEY, decision TEXT NOT NULL, decided_at TEXT NOT NULL)"
            )

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def get(self, run_id: str) -> str:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT decision FROM plan_decisions WHERE run_id=?", (str(run_id),)
            ).fetchone()
        return row[0] if row else ""

    def decide(self, runtime: Any, run_id: str, body: dict) -> dict:
        """Record confirm / cancel once. Confirm + needs_compute → pending approval (no HPC)."""
        if not isinstance(body, dict) or set(body) != {"decision"} or body["decision"] not in DECISIONS:
            raise AccessError(400, "invalid_plan_decision")
        session = runtime.store.get_session(run_id) or {}
        meta = session.get("metadata") or {}
        if meta.get("intent") != "experience":
            raise AccessError(409, "run_has_no_plan")
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO plan_decisions VALUES(?,?,?)",
                (run_id, body["decision"], utc_now()),
            )
            current = connection.execute(
                "SELECT decision FROM plan_decisions WHERE run_id=?", (run_id,)
            ).fetchone()[0]
        if current != body["decision"]:
            raise AccessError(409, "plan_already_decided")
        out: dict[str, Any] = {"run_id": run_id, "decision": current}
        if current == "confirm":
            plan = meta.get("plan_proposed") or {}
            if isinstance(plan, dict) and plan.get("needs_compute"):
                out["approval"] = _pending_compute_approval(runtime, run_id, plan)
        return out
