"""POST /v1/runs/{id}/retry: one fresh attempt of the latest, retryable run of a conversation.

Contract: the caller already passed the ownership guard (tenant_access: non-owner -> 404 not_found). Here:
  * already retried (a run with session metadata retry_of == this run exists)  -> return that run, create nothing
  * not the newest run of its conversation                                       -> 409 {"ok":false,"code":"not_latest"}
  * snapshot retryable is not True                                               -> 409 {"ok":false,"code":"not_retryable"}
  * otherwise create ONE run through RunRequests.create (the normal admission path: ownership, project, model
    profile, policy/HPC guards and budgets all apply) with the same prompt, conversation, project, model profile,
    kind and attachments, linked with retry_of / retried_by.
Success body is only {"ok":true,"run_id":...,"conversation_id":...}; no error text or path is ever returned.
A process-wide lock serialises retries; the retry_of link written at creation makes a double click (or a crash
between creation and linking) converge on the same new run.
"""
from __future__ import annotations

import threading
from typing import Any

from .tenant_access import AccessError, Principal

_RETRY_LOCK = threading.RLock()


def _meta(store: Any, session_id: str) -> dict[str, Any]:
    session = store.get_session(session_id) or {}
    return session.get("metadata") if isinstance(session.get("metadata"), dict) else {}


def _existing_retry(store: Any, source: dict[str, Any]) -> dict[str, Any] | None:
    """The run created by an earlier retry of `source`, if any (persisted link, never in-memory state)."""
    source_id = str(source["id"])
    linked = str(_meta(store, str(source["session_id"])).get("retried_by") or "")
    if linked:
        child = store.get_assistant_run(linked)
        if child:
            return child
    for row in store.list_assistant_runs(str(source["conversation_id"]), 50):  # newest first
        if row["id"] == source_id:
            break
        if str(_meta(store, str(row["session_id"])).get("retry_of") or "") == source_id:
            return row
    return None


def _refuse(status: int, code: str) -> AccessError:
    return AccessError(status, code)


def retry_run(runtime: Any, access: Any, requests: Any, principal: Principal | None, run_id: str, body: dict) -> dict[str, Any]:
    store, assistant = runtime.store, runtime.assistant
    with _RETRY_LOCK:
        source = store.get_assistant_run(run_id)
        if not source:
            raise KeyError(run_id)
        conversation = str(source["conversation_id"])
        existing = _existing_retry(store, source)
        if existing is None:
            latest = store.latest_assistant_run(conversation)
            if not latest or latest["id"] != source["id"]:
                raise _refuse(409, "not_latest")
            if assistant.snapshot(run_id).get("retryable") is not True:
                raise _refuse(409, "not_retryable")
            metadata = _meta(store, str(source["session_id"]))
            attachment_ids = [str(value) for value in source.get("attachment_ids") or []]
            if principal is not None:
                # The source Run's input copies belong to the owner of that Run; register them so the normal
                # admission path (which requires attachment ownership) can copy them into the new Run.
                for attachment_id in attachment_ids:
                    access.bind("attachment", attachment_id, principal)
            request = {
                "prompt": str(source["prompt"]),
                "conversation_id": conversation,
                "selected_task": str(metadata.get("selected_task") or ""),
                "attachment_ids": attachment_ids,
                "secret_refs": [],
                "kind": assistant._raw_kind(source, metadata),
                # Server-bound identity supplied by the web tier, exactly as for POST /v1/runs.
                "managed_hpc_identity": body.get("managed_hpc_identity"),
            }
            if metadata.get("project_id"):
                request["project_id"] = str(metadata["project_id"])
            profile = metadata.get("model_profile")
            if isinstance(profile, dict) and profile.get("id") and type(profile.get("version")) is int:
                request["model_profile_id"], request["model_profile_version"] = str(profile["id"]), profile["version"]
            created = requests.create(runtime, access, principal, request, "", retry_of=run_id)
            existing = store.get_assistant_run(str(created["id"]))
        store.merge_session_metadata(str(source["session_id"]), {"retried_by": str(existing["id"])})
        return {"ok": True, "run_id": str(existing["id"]), "conversation_id": str(existing["conversation_id"])}
