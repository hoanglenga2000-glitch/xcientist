"""Tool-call guardrail controller — hardens no-fabrication and stops token spin.

Ported from the sound idea in Hermes' ToolCallGuardrailController, specialized to
this research agent. It sits BETWEEN the model and the toolbox: the session asks
``before_call`` (may block a call before it runs) and reports ``after_call`` (may
force a halt). Nothing here decides research outcomes — that stays in the tools'
deterministic gates. This layer only prevents pathological loops:

  * IDENTICAL REPEATED FAILURE: the same tool called with the same args, failing
    N times, is blocked — the model is stuck; make it change something.
  * TRUNCATED CODE: a run_experiment whose code has an unbalanced ``` fence or ends
    mid-line is refused BEFORE execution (a truncated script wastes a run and can
    fail in confusing ways). This is the agent-side analogue of Hermes rejecting
    truncated tool_call arguments.
  * IDEMPOTENT READ SPIN: a read-only tool returning the identical result N times
    in a row is nudged (the model is re-reading instead of acting).
  * CONSECUTIVE FAILURE HALT: after N consecutive failed tool calls of ANY kind,
    the session halts — better to stop with an honest partial result than burn the
    whole budget flailing.

All limits are conservative and configurable; the goal is to catch loops, not to
second-guess a model that is making progress.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GuardrailDecision:
    """Returned by before_call. ``blocked`` short-circuits the tool; ``content`` is
    the message fed back to the model in place of the tool result."""

    blocked: bool = False
    content: str = ""
    summary: str = ""


# Read-only tools whose identical repetition is a spin signal (not progress).
_IDEMPOTENT_READS = {"inspect_data", "read_memory", "read_search_tree", "recommend_strategies"}
# Cheap tools whose SUCCESS is not "progress": a passing read or plan must not
# reset the consecutive-failure streak, otherwise the halt never fires in normal
# operation (plan_next_experiment always succeeds between failing runs).
_NON_PROGRESS_TOOLS = _IDEMPOTENT_READS | {"plan_next_experiment"}


def _args_fingerprint(name: str, args: dict[str, Any]) -> str:
    """Stable hash of (tool, args) so identical repeated calls collide. Large code
    blobs are hashed too, so a resubmitted-verbatim script is caught."""
    parts = [name]
    for key in sorted(args):
        parts.append(f"{key}={args[key]!r}")
    raw = "|".join(parts).encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:16]


def _looks_truncated(code: str) -> Optional[str]:
    """Return a reason string if the code looks truncated, else None.

    Cheap syntactic checks only — we are catching an obviously cut-off script
    (odd number of ``` fences, or a dangling continuation), not doing a full parse.
    """
    if code.count("```") % 2 != 0:
        return "unbalanced ``` code fence (script appears cut off mid-block)"
    stripped = code.rstrip()
    if stripped.endswith(("\\", ",", "(", "[", "{", "=", "+")):
        return "script ends on a dangling operator/bracket (appears truncated)"
    return None


class ToolGuardrailController:
    """Per-session loop-breaker. Cheap bookkeeping keyed by tool + args."""

    def __init__(self, *, repeat_failure_limit: int = 3, idempotent_repeat_limit: int = 4,
                 consecutive_failure_halt: int = 6) -> None:
        self.repeat_failure_limit = repeat_failure_limit
        self.idempotent_repeat_limit = idempotent_repeat_limit
        self.consecutive_failure_halt = consecutive_failure_halt
        # (fingerprint) -> consecutive failure count for that exact call
        self._fail_counts: dict[str, int] = {}
        # (tool) -> last result fingerprint + repeat count (idempotent read spin)
        self._last_read: dict[str, tuple[str, int]] = {}
        self._consecutive_failures = 0
        self._halted = False
        self._halt_reason = ""

    # ── before the tool runs ──────────────────────────────────────────────────
    def before_call(self, name: str, args: dict[str, Any]) -> GuardrailDecision:
        fp = _args_fingerprint(name, args)
        # 1) identical repeated failure: this exact call already failed N times.
        if self._fail_counts.get(fp, 0) >= self.repeat_failure_limit:
            return GuardrailDecision(
                blocked=True,
                content=(f"BLOCKED by guardrail: `{name}` with these exact arguments has already "
                         f"failed {self.repeat_failure_limit} times. Repeating it will not help — "
                         "change your approach (different code/hypothesis), or call finish if you "
                         "are genuinely stuck."),
                summary=f"{name} blocked (repeated {self.repeat_failure_limit}× failure)")
        # 2) truncated code on run_experiment: refuse before wasting a run.
        if name == "run_experiment":
            reason = _looks_truncated(args.get("code") or "")
            if reason:
                return GuardrailDecision(
                    blocked=True,
                    content=(f"BLOCKED by guardrail: the submitted code looks truncated — {reason}. "
                             "Resend the COMPLETE script (do not rely on it being continued)."),
                    summary=f"run_experiment blocked (truncated code: {reason[:40]})")
        return GuardrailDecision(blocked=False)

    # ── after the tool ran ────────────────────────────────────────────────────
    def after_call(self, name: str, args: dict[str, Any], *, ok: bool,
                   result_content: str = "") -> None:
        fp = _args_fingerprint(name, args)
        if ok:
            # This exact call succeeded → clear ITS identical-repeat count. But only
            # a substantive success (a run/promotion, not a cheap read/plan) resets
            # the global consecutive-failure streak — otherwise the halt never fires.
            self._fail_counts.pop(fp, None)
            if name not in _NON_PROGRESS_TOOLS:
                self._consecutive_failures = 0
        else:
            self._fail_counts[fp] = self._fail_counts.get(fp, 0) + 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.consecutive_failure_halt:
                self._halted = True
                self._halt_reason = (
                    f"{self._consecutive_failures} consecutive tool failures — halting to avoid "
                    "burning the budget. The run ends with an honest partial result.")
        # idempotent read spin: same read-only tool, identical result repeatedly.
        if name in _IDEMPOTENT_READS:
            digest = hashlib.sha256(result_content.encode("utf-8", errors="replace")).hexdigest()[:16]
            prev, count = self._last_read.get(name, ("", 0))
            self._last_read[name] = (digest, count + 1 if digest == prev else 1)

    def idempotent_spin_warning(self, name: str) -> Optional[str]:
        """A one-line nudge if a read-only tool has repeated its result too often."""
        _, count = self._last_read.get(name, ("", 0))
        if name in _IDEMPOTENT_READS and count >= self.idempotent_repeat_limit:
            return (f"note: `{name}` has returned the same result {count} times — you are "
                    "re-reading, not acting. Plan or run an experiment instead.")
        return None

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason
