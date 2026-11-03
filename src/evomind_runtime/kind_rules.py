"""Conversation kind (user/test/acceptance/probe) and user-file visibility (user/internal).

WHEN IN DOUBT ALWAYS TAG `user`. Only clearly probe / smoke / acceptance-script conversations get `test` or
`acceptance`. A wrongly hidden real conversation is worse than a visible test one.

RUNTIME POLICY: NO KEYWORD RULES RUN AT RUNTIME. A conversation's kind is only
  1. the explicit `kind` the creating caller passed (mapped through emit_kind), or
  2. an entry in the static per-conversation table `kind_table.LEGACY_CONVERSATION_KINDS`, generated once from the
     existing conversations by the dry-run script (the keyword rules live only there), or
  3. `user` (anything else, including every new conversation without an explicit kind).
MANUAL_USER_CONVERSATIONS always wins over the table.

Pure functions, no I/O, no model calls.
"""
from __future__ import annotations

import os
import re

try:  # inside the evomind_runtime package
    from .kind_table import LEGACY_CONVERSATION_KINDS
except ImportError:  # standalone (dry-run / tests)
    from kind_table import LEGACY_CONVERSATION_KINDS

KINDS = ("user", "test", "acceptance", "probe")

# Legacy conversations that look test-like but are real work (decisions 2026-10-17): two real training runs,
# 0a816c44 (CURE-Bench R52 data-prep acceptance) and 007bdfddb8f34e (histopathologic study). Always `user`.
MANUAL_USER_CONVERSATIONS: frozenset[str] = frozenset(
    ('chat_a1b91c27a1564e', 'chat_be32e1a0e5bd41ada0a4958083882c24', 'chat_3feb9d0a816c44', 'chat_007bdfddb8f34e')
) | frozenset(value for value in os.environ.get("EVOMIND_KIND_MANUAL_USER", "").split(",") if value)


def emit_kind(kind: str) -> tuple[str, str]:
    """Wire format for the frontend, which only knows user/test/acceptance: probe -> kind "test" + kind_detail "probe"."""
    return ("test", "probe") if kind == "probe" else (kind, kind)


def conversation_kind(conversation_id: str) -> str:
    """Static lookup only (no prompt inspection): manual keep-user > legacy table > `user`."""
    cid = str(conversation_id or "")
    if cid in MANUAL_USER_CONVERSATIONS:
        return "user"
    kind = LEGACY_CONVERSATION_KINDS.get(cid, "user")
    return kind if kind in KINDS else "user"


# --- user files -------------------------------------------------------------------------------
_INTERNAL_CERTAIN = re.compile(
    r"(^cure-asset-probe-)|(_smoke\.(py|json|md|txt)$)|(-(owner|holdout)-authorization-[^/]*\.json$)", re.I)
_INTERNAL_EXTENDED = re.compile(
    r"(receipt[^/]*\.json$)|(visible-snapshot\.json$)|(_notebook_dom\.txt$)|(^verify_inputs\.py$)|(^run_mcgs_proof\.py$)"
    r"|(^source-search-graph)|(^acceptance-report)|(^independent-(evaluation|verification)\.json$)|(^private-evaluation\.json$)"
    r"|(^freeze-receipt)|(-gate\.md$)|(^run-manifest)|(^training_manifest)", re.I)


def file_visibility(name: str) -> tuple[str, str, str]:
    """-> (visibility, rule, confidence). Extended patterns only apply when EVOMIND_FILE_VISIBILITY_EXTENDED=1."""
    value = str(name or "")
    if _INTERNAL_CERTAIN.search(value):
        return "internal", "name pattern (plan section 3)", "certain"
    if _INTERNAL_EXTENDED.search(value):
        if os.environ.get("EVOMIND_FILE_VISIBILITY_EXTENDED") == "1":
            return "internal", "name pattern (extended)", "needs-confirm"
        return "user", "extended pattern disabled", "needs-confirm"
    return "user", "default", "default"
