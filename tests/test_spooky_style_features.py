"""Tests for deterministic Spooky stylometric features."""

from __future__ import annotations

import numpy as np

from scripts import spooky_style_features as style


def _column(name: str) -> int:
    return style.FEATURE_NAMES.index(name)


def test_style_features_are_fixed_finite_and_audited():
    values, contract = style.extract_style_features(["", "The raven spoke.", None])
    assert values.shape == (3, 28)
    assert np.isfinite(values).all()
    assert contract["feature_count"] == 28
    assert len(contract["vocabulary_sha256"]) == 64
    assert contract["fit_scope"] == "none_deterministic_public_text_only"
    assert contract["private_labels_used"] is False


def test_author_keywords_and_first_word_indicators_are_exact():
    values, _ = style.extract_style_features(
        [
            "The detective Dupin met Monsieur.",
            "My road led from Arkham to Innsmouth.",
            "She told Victor about Frankenstein and Perdita.",
        ]
    )
    assert values[0, _column("has_eap_keyword")] == 1.0
    assert values[1, _column("has_hpl_keyword")] == 1.0
    assert values[2, _column("has_mws_keyword")] == 1.0
    assert values[0, _column("first_word_is_the")] == 1.0
    assert values[1, _column("first_word_is_my")] == 1.0
    assert values[2, _column("first_word_is_she")] == 1.0
    assert values[2, _column("author_keyword_count")] == 3.0


def test_punctuation_densities_preserve_style_signal():
    values, _ = style.extract_style_features(
        ["One, two, three, four.", "One; two; three; four."]
    )
    assert values[0, _column("comma_per_word")] > values[1, _column("comma_per_word")]
    assert values[1, _column("semicolon_per_word")] > values[0, _column("semicolon_per_word")]

