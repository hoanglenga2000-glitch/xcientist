"""End-reason / retry signal for ended assistant runs (pure functions, no I/O, no model calls).

Wire fields (never carry error text or paths):
  end_reason  one of END_REASONS, present only when the run ended without success AND the reason was recorded
  retryable   bool; True only for RETRYABLE_REASONS

NO BACKFILL, NO READ-TIME DERIVATION. The values are decided exactly once, at the site that writes the failure
(`classify_failure` is called only there, with the full exception) and stored in session metadata as
  {"reason": <enum or "">, "retryable": <bool>, "status": <status written>, "sig": <signature of the row written>}.
The read path (`end_fields`) never looks at error_class / error_message to decide anything. If the record is absent,
malformed, or no longer matches the run row (status/error fields rewritten by another site), the fields are treated
as missing: no `end_reason`, `retryable` false. Old runs therefore never become retryable.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

END_REASONS = ("user_cancelled", "runtime_not_ready", "timeout", "model_error", "budget", "policy_blocked")
RETRYABLE_REASONS = frozenset({"runtime_not_ready", "timeout", "model_error"})
ENDED_STATUSES = frozenset({"failed", "blocked", "cancelled"})

# LLMError/<code> (model_transport.ModelTransportError) -> reason. Codes not listed here (refusal, truncated output,
# request too large, identity/contract/route-config problems, 4xx auth, ...) are deliberately NOT retryable: omitted.
_LLM_TIMEOUT = frozenset({"model_total_deadline_exceeded", "http_408", "http_504"})
_LLM_NOT_READY = frozenset({"http_429", "http_500", "http_502", "http_503", "model_network_error"})
_LLM_MODEL_ERROR = frozenset({
    "model_response_protocol_error", "response_not_object", "response_choices_invalid", "response_message_missing",
    "tool_calls_invalid", "tool_call_invalid", "tool_call_identity_invalid", "tool_arguments_invalid",
    "response_content_invalid", "empty_model_response",
})
_APPROVAL_CLASSES = frozenset({"approval_required", "approval_rejected", "approval_expired"})
_UNKNOWN_CLASSES = frozenset({"controlled_secret_required", "input_required", "incomplete_deliverables",
                              "execution_reconciliation_required", "approval_execution_unconfirmed",
                              "creation_interrupted", "worker_start_failed", "terminal"})
_LLM_CODE = re.compile(r"llmerror/([a-z0-9_]+)")
_TIMEOUT_TEXT = re.compile(r"timeout|timed out|deadline|\b504\b")
_NOT_READY_TEXT = re.compile(r"\b(?:429|500|502|503)\b|rate limit|connection|network|gateway|outage|unavailable")


def classify_failure(error_class: str, text: str) -> str:
    """-> reason from END_REASONS (never user_cancelled), or "" when unknown/uncertain.

    `text` is "<ExceptionType>: <message>" at failure time, or the stored error_message at read time."""
    ec = str(error_class or "")
    t = str(text or "").casefold()
    if ec in _APPROVAL_CLASSES:
        return "policy_blocked"
    if ec in _UNKNOWN_CLASSES or ec.startswith("approval_"):
        return ""
    if "budget" in t:  # before timeout: e.g. TimeoutError("download_budget_exhausted..."); the same budget would hit again
        return "budget"
    if "model_contract" in t:
        return ""
    match = _LLM_CODE.search(t)
    if match:
        code = match.group(1)
        if code in _LLM_TIMEOUT:
            return "timeout"
        if code in _LLM_NOT_READY:
            return "runtime_not_ready"
        if code in _LLM_MODEL_ERROR:
            return "model_error"
        return ""
    if ec == "transient":  # assistant_runs._failure_policy: timeout / 429 / 5xx / network / stream terms
        if _TIMEOUT_TEXT.search(t):
            return "timeout"
        if _NOT_READY_TEXT.search(t):
            return "runtime_not_ready"
    return ""


def failure_signature(status: Any, error_class: Any, error_message: Any) -> str:
    return hashlib.sha256(f"{status or ''}\x00{error_class or ''}\x00{error_message or ''}".encode("utf-8", "replace")).hexdigest()[:16]


def record_for(row: dict[str, Any], reason: str) -> dict[str, Any]:
    """The metadata record to store at the failure-write site. `row` is the assistant_runs row AFTER the failure was
    written, `reason` the classification (or "" when uncertain)."""
    reason = reason if reason in END_REASONS else ""
    return {"reason": reason, "retryable": reason in RETRYABLE_REASONS, "status": str(row.get("status") or ""),
            "sig": failure_signature(row.get("status"), row.get("error_class"), row.get("error_message"))}


def end_fields(run: dict[str, Any], session_metadata: dict[str, Any] | None, status: str | None = None) -> dict[str, Any]:
    """Wire fields for one run, from the STORED record only (see module docstring). `status` is the effective status."""
    metadata = session_metadata if isinstance(session_metadata, dict) else {}
    status = str(status if status is not None else run.get("status") or "")
    stored = metadata.get("end_reason")
    reason, stored_retryable = "", False
    if (
        status in ENDED_STATUSES
        and isinstance(stored, dict)
        and stored.get("status") == status == str(run.get("status") or "")
        and stored.get("sig") == failure_signature(run.get("status"), run.get("error_class"), run.get("error_message"))
        and stored.get("reason") in END_REASONS
    ):
        reason, stored_retryable = str(stored["reason"]), stored.get("retryable") is True
    retryable = (
        stored_retryable
        and reason in RETRYABLE_REASONS
        and not metadata.get("retried_by")                 # already retried once: do not offer it again
        and not metadata.get("user_task_id")               # task-linked runs bind task files; not replayable here
        and not metadata.get("run_secret_refs")            # one-shot controlled credentials cannot be replayed
    )
    result: dict[str, Any] = {"retryable": bool(retryable)}
    if reason:
        result["end_reason"] = reason
    return result
