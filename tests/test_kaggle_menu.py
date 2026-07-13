"""Tests for the arrow-key selection menu's numbered fallback.

The interactive arrow-key path needs a real TTY + keystrokes, so it can't be
unit-tested here; pytest sets ``PYTEST_CURRENT_TEST`` and ``select`` therefore
always takes the deterministic numbered fallback driven by an injected reader.
These lock that fallback's contract (the seam the setup wizard relies on).
"""
from __future__ import annotations

import io

from xsci import kaggle_menu as km


def _reader(answer: str):
    # Mimics kaggle._safe_input(prompt, default) -> str.
    return lambda prompt, default="": answer


def _choices():
    return [km.Choice("Anthropic (Claude)", "native tool-use"),
            km.Choice("DeepSeek", "openai-compatible"),
            km.Choice("OpenAI (GPT)")]


def test_numbered_pick_returns_zero_based_index():
    idx = km.select("pick", _choices(), reader=_reader("2"), stream=io.StringIO())
    assert idx == 1                       # "2" -> index 1 (DeepSeek)


def test_empty_answer_takes_the_default():
    idx = km.select("pick", _choices(), default=2, reader=_reader(""), stream=io.StringIO())
    assert idx == 2                       # blank -> the offered default


def test_skip_token_returns_minus_one_when_allowed():
    idx = km.select("pick", _choices(), allow_skip=True, reader=_reader("s"), stream=io.StringIO())
    assert idx == -1


def test_skip_token_ignored_when_not_allowed():
    # Without allow_skip, "s" is not a valid digit -> falls back to the default.
    idx = km.select("pick", _choices(), default=0, allow_skip=False, reader=_reader("s"),
                    stream=io.StringIO())
    assert idx == 0


def test_out_of_range_falls_back_to_default():
    idx = km.select("pick", _choices(), default=1, reader=_reader("99"), stream=io.StringIO())
    assert idx == 1


def test_empty_choice_list_returns_minus_one():
    assert km.select("pick", [], reader=_reader("1"), stream=io.StringIO()) == -1


def test_numbered_menu_renders_labels_and_hints():
    buf = io.StringIO()
    km.select("Choose a provider:", _choices(), allow_skip=True, reader=_reader("1"), stream=buf)
    out = buf.getvalue()
    assert "Anthropic (Claude)" in out and "DeepSeek" in out and "OpenAI (GPT)" in out
    assert "native tool-use" in out       # the hint is shown
    assert "s) skip" in out               # skip affordance shown when allowed


def test_interactive_disabled_under_pytest():
    # The guard: PYTEST_CURRENT_TEST is set, so the live arrow path is never taken
    # (which is why the numbered fallback above is exercised deterministically).
    assert km._interactive(io.StringIO()) is False
