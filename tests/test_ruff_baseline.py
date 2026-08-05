from __future__ import annotations

from collections import Counter

from scripts.verify_ruff_baseline import diagnostic_counts, unexpected_counts


def test_unexpected_counts_allows_resolved_legacy_diagnostics() -> None:
    fingerprint = ("scripts/legacy.py", "F401", "unused import")

    assert unexpected_counts(Counter({fingerprint: 1}), Counter({fingerprint: 2})) == Counter()


def test_unexpected_counts_rejects_new_occurrences() -> None:
    fingerprint = ("scripts/legacy.py", "F401", "unused import")

    assert unexpected_counts(Counter({fingerprint: 3}), Counter({fingerprint: 2})) == Counter({fingerprint: 1})


def test_diagnostic_counts_groups_stable_path_code_message_fingerprint() -> None:
    diagnostics = [
        {"filename": "scripts/legacy.py", "code": "F401", "message": "unused import"},
        {"filename": "scripts/legacy.py", "code": "F401", "message": "unused import"},
    ]

    counts = diagnostic_counts(diagnostics)

    assert sum(counts.values()) == 2
