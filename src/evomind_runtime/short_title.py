"""Async short sidebar titles (P2 §2).

After the first *completed* assistant turn, generate a ≤16-character label from the
conversation's first user question + that completed answer. Persist as session
metadata `short_title` + `title_source` (`auto`|`user`) on the conversation's
**chronologically first** assistant run session (sidebar / groupRuns reads that
row). Each run typically has session_id == run_id, so writing onto the completing
run alone leaves the first-run session empty and the sidebar hard-truncates.
If the chronologically first run fails/blocks, the first later completed run may
still write once — onto the first run's session. Silent on failure; ≤2 attempts;
never overwrite `title_source=user`.
"""
from __future__ import annotations

import re
import threading
from typing import Any, Callable

TITLE_MAX = 16
TITLE_SOURCES = frozenset({"auto", "user"})
_MAX_ATTEMPTS = 2
_MAX_PROMPT_CHARS = 1200
_MAX_ANSWER_CHARS = 1200

# Strip wrapping quotes / markdown emphasis the model sometimes adds.
_WRAP = re.compile(r'^[\s"\'「」『』《》【】\[\]()（）*`_~]+|[\s"\'「」『』《》【】\[\]()（）*`_~.。!！?？,，;；:：]+$')
_MULTISPACE = re.compile(r"\s+")

SYSTEM_PROMPT = (
    "You write short sidebar labels for chat sessions. "
    "Reply with ONLY the label text: at most 16 characters (Chinese characters count as one each). "
    "No quotes, no punctuation fluff, no markdown, no 'Untitled' or '未命名'. "
    "Prefer Chinese when the conversation is in Chinese; otherwise match the conversation language."
)


def clamp_title(text: str, limit: int = TITLE_MAX) -> str:
    """Normalize model/user title text to ≤limit grapheme-ish code points."""
    value = _MULTISPACE.sub(" ", str(text or "").replace("\n", " ").replace("\r", " ")).strip()
    value = _WRAP.sub("", value).strip()
    value = _MULTISPACE.sub(" ", value).strip()
    if not value:
        return ""
    # Prefer code-point length (matches CJK + frontend JS for BMP Chinese).
    if len(value) > limit:
        value = value[:limit].rstrip()
    # Reject empty / placeholder leftovers after clamp.
    lowered = value.lower()
    if lowered in {"untitled", "未命名", "未命名对话", "new chat", "new conversation"}:
        return ""
    return value


def should_generate(metadata: dict[str, Any] | None) -> bool:
    """True when we may auto-write a short_title (no user title, none set yet)."""
    meta = metadata if isinstance(metadata, dict) else {}
    if str(meta.get("title_source") or "") == "user":
        return False
    if str(meta.get("short_title") or "").strip():
        return False
    return True


def prefer_short_title(
    *,
    prompt: str,
    created_at: str,
    run_id: str,
    metadata: dict[str, Any] | None,
    sensitive: bool,
) -> str:
    """Title for list APIs: prefer stored short_title; else prompt / redaction."""
    meta = metadata if isinstance(metadata, dict) else {}
    short = clamp_title(str(meta.get("short_title") or ""))
    source = str(meta.get("title_source") or "").strip()
    if short and source in TITLE_SOURCES:
        return short
    if sensitive:
        return "Experiment " + str(created_at or "")[:10]
    title = " ".join(str(prompt or "").split())[:100]
    return title or str(run_id)


def build_title_user_message(question: str, answer: str) -> str:
    q = _MULTISPACE.sub(" ", str(question or "").strip())[:_MAX_PROMPT_CHARS]
    a = _MULTISPACE.sub(" ", str(answer or "").strip())[:_MAX_ANSWER_CHARS]
    return (
        f"First user question:\n{q or '(empty)'}\n\n"
        f"First assistant answer:\n{a or '(empty)'}\n\n"
        "Sidebar label:"
    )


def _client_send(runtime: Any, session: dict[str, Any], question: str, answer: str) -> str:
    """One-shot send via the run's own governed / personal client. tools=[]."""
    personal_binding = (session.get("metadata") or {}).get("model_profile")
    if personal_binding:
        from .personal_model_client import client_for_run

        client = client_for_run(runtime, session, before_attempt=None)
    else:
        from research_os.agent.messaging import AgentMessageClient

        from .model_transport import governed_client

        client = governed_client(AgentMessageClient())
    if not getattr(client, "is_available", lambda: True)():
        raise RuntimeError("title_model_unavailable")
    turn = client.send(
        [{"role": "user", "content": build_title_user_message(question, answer)}],
        system=SYSTEM_PROMPT,
        tools=[],
        max_tokens=40,
        temperature=0.2,
    )
    text = getattr(turn, "text", None)
    if text is None and isinstance(turn, dict):
        text = turn.get("text") or turn.get("content")
    return str(text or "")


def generate_and_store(
    *,
    store: Any,
    runtime: Any,
    run: dict[str, Any],
    answer: str,
    send_fn: Callable[..., str] | None = None,
) -> str:
    """Generate short_title for the first completed run; return stored title or ''."""
    session_id = str(run.get("session_id") or "")
    conversation_id = str(run.get("conversation_id") or "")
    run_id = str(run.get("id") or "")
    if not session_id or not conversation_id or not run_id:
        return ""
    # Gate on first *completed* run so a failed chronological-first does not
    # permanently leave the sidebar on hard-truncated prompt titles.
    if hasattr(store, "first_completed_assistant_run"):
        first_completed = store.first_completed_assistant_run(conversation_id)
    else:
        # Tests / older fakes: fall back to chronological first.
        first_completed = store.first_assistant_run(conversation_id)
    if not first_completed or str(first_completed.get("id") or "") != run_id:
        return ""
    # Prefer the conversation's first user prompt (chronological first run),
    # not only the successful retry's prompt.
    question = ""
    first_chrono = store.first_assistant_run(conversation_id)
    first_chrono_id = str((first_chrono or {}).get("id") or "")
    # Sidebar / groupRuns / prefer_short_title read the *first* run's session.
    # Runs use session_id == run_id, so metadata must land on that first session
    # — not the completing retry's session.
    target_session_id = str((first_chrono or {}).get("session_id") or "") or session_id
    if first_chrono_id:
        try:
            full_first = store.get_assistant_run(first_chrono_id)
        except Exception:
            full_first = None
        if isinstance(full_first, dict):
            question = str(full_first.get("prompt") or "").strip()
            # Prefer full row's session_id when the summary row omitted it.
            if not str((first_chrono or {}).get("session_id") or ""):
                target_session_id = str(full_first.get("session_id") or "") or target_session_id
        if not question and isinstance(first_chrono, dict):
            question = str(first_chrono.get("prompt") or "").strip()
    session = store.get_session(target_session_id) or {}
    metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
    if not should_generate(metadata):
        return ""
    if not question:
        question = str(run.get("prompt") or session.get("objective") or "").strip()
    # Model binding lives on the completing run's session; title metadata on first.
    client_session = store.get_session(session_id) or session
    last_title = ""
    for _ in range(_MAX_ATTEMPTS):
        try:
            # Re-read before each attempt so a concurrent user rename wins.
            session = store.get_session(target_session_id) or {}
            metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
            if not should_generate(metadata):
                return str(metadata.get("short_title") or "")
            if send_fn is None:
                raw = _client_send(runtime, client_session, question, answer)
            else:
                raw = send_fn(
                    runtime=runtime, session=client_session, question=question, answer=answer
                )
            title = clamp_title(raw)
            if not title:
                continue
            # Final race check under merge: skip if user won.
            session = store.get_session(target_session_id) or {}
            metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
            if not should_generate(metadata):
                return str(metadata.get("short_title") or "")
            store.merge_session_metadata(
                target_session_id, {"short_title": title, "title_source": "auto"}
            )
            return title
        except Exception:
            last_title = ""
            continue
    return last_title

def schedule_after_first_turn(assistant: Any, run_id: str, answer: str) -> None:
    """Fire a daemon thread; never blocks the completion path; swallows all errors."""
    try:
        if assistant._stopping.is_set():  # type: ignore[attr-defined]
            return
    except Exception:
        pass

    def worker() -> None:
        try:
            run = assistant.store.get_assistant_run(run_id)
            if not run or str(run.get("status") or "") != "completed":
                return
            generate_and_store(
                store=assistant.store,
                runtime=assistant.runtime,
                run=run,
                answer=answer,
            )
        except Exception:
            return

    try:
        threading.Thread(
            target=worker,
            name=f"evomind-short-title-{run_id[-12:]}",
            daemon=True,
        ).start()
    except Exception:
        return
