"""Deterministic public-text stylometry for Spooky Author Identification."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

FEATURE_NAMES = (
    "character_count",
    "word_count",
    "comma_per_word",
    "semicolon_per_word",
    "period_per_word",
    "exclamation_per_word",
    "question_per_word",
    "apostrophe_per_word",
    "colon_per_word",
    "has_double_quote",
    "has_single_quote",
    "all_caps_count",
    "stopword_rate",
    "has_eap_keyword",
    "has_hpl_keyword",
    "has_mws_keyword",
    "dialect_word_count",
    "archaism_count",
    "long_word_rate",
    "characters_per_word",
    "type_token_ratio",
    "all_caps_rate",
    "double_to_single_quote_ratio",
    "first_word_is_she",
    "first_word_is_my",
    "first_word_is_the",
    "first_word_is_i",
    "author_keyword_count",
)

EAP_KEYWORDS = frozenset(
    {
        "amontillado",
        "beauvais",
        "chess",
        "dupin",
        "eleonora",
        "fortunato",
        "lalande",
        "ligeia",
        "madeline",
        "marie",
        "mesmeric",
        "monsieur",
        "morella",
        "prefect",
        "pym",
        "usher",
    }
)
HPL_KEYWORDS = frozenset(
    {
        "arkham",
        "cthulhu",
        "cyclopean",
        "daemoniac",
        "dunwich",
        "eldritch",
        "ghoul",
        "innsmouth",
        "keziah",
        "kingsport",
        "miskatonic",
        "necronomicon",
        "r'lyeh",
        "shoggoth",
        "tillinghast",
        "whateley",
        "wilbur",
        "yog-sothoth",
        "zadok",
    }
)
MWS_KEYWORDS = frozenset(
    {
        "adrian",
        "clara",
        "clerval",
        "cottage",
        "elizabeth",
        "evadne",
        "frankenstein",
        "idris",
        "justine",
        "lionel",
        "paradise",
        "perdita",
        "victor",
        "walton",
        "windsor",
    }
)
DIALECT = frozenset({"aout", "arter", "fur", "haow", "knewed", "sartin", "shet", "ud", "whar", "ye"})
ARCHAISMS = frozenset(
    {"betwixt", "doth", "ere", "hath", "hence", "hitherto", "methought", "thee", "thence", "therein", "thine", "thou", "thy", "wherein"}
)
STOPWORDS = frozenset(
    "a an the and or but if because as until while of at by for with about against between "
    "into through during before after above below to from up down in out on off over under "
    "again further then once here there when where why how all each every both few more most "
    "other some such no nor not only own same so than too very just also it its he she they "
    "them his her their this that these those is are was were be been being have has had do "
    "does did will would shall should may might must can could i me my we us our you your".split()
)
WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")


def _vocabulary_sha256() -> str:
    payload = {
        "eap": sorted(EAP_KEYWORDS),
        "hpl": sorted(HPL_KEYWORDS),
        "mws": sorted(MWS_KEYWORDS),
        "dialect": sorted(DIALECT),
        "archaisms": sorted(ARCHAISMS),
        "stopwords": sorted(STOPWORDS),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def extract_style_features(texts: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract 28 fixed public-text style features with no corpus fitting."""

    result = np.zeros((len(texts), len(FEATURE_NAMES)), dtype=np.float32)
    all_keywords = EAP_KEYWORDS | HPL_KEYWORDS | MWS_KEYWORDS
    for row, raw in enumerate(texts):
        text = "" if raw is None else str(raw)
        original_words = WORD_RE.findall(text)
        words = [value.lower() for value in original_words]
        word_set = set(words)
        word_count = len(words)
        denominator = max(word_count, 1)
        double_quotes = text.count('"')
        single_quotes = text.count("'")
        all_caps = sum(value.isupper() and len(value) > 1 for value in original_words)
        keyword_count = sum(value in all_keywords for value in words)
        result[row] = np.asarray(
            [
                len(text),
                word_count,
                text.count(",") / denominator,
                text.count(";") / denominator,
                text.count(".") / denominator,
                text.count("!") / denominator,
                text.count("?") / denominator,
                single_quotes / denominator,
                text.count(":") / denominator,
                float(double_quotes > 0),
                float(single_quotes > 0),
                all_caps,
                sum(value in STOPWORDS for value in words) / denominator,
                float(bool(word_set & EAP_KEYWORDS)),
                float(bool(word_set & HPL_KEYWORDS)),
                float(bool(word_set & MWS_KEYWORDS)),
                sum(value in DIALECT for value in words),
                sum(value in ARCHAISMS for value in words),
                sum(len(value) >= 10 for value in words) / denominator,
                len(text) / denominator,
                len(word_set) / denominator,
                all_caps / denominator,
                double_quotes / max(single_quotes, 1),
                float(bool(words) and words[0] == "she"),
                float(bool(words) and words[0] == "my"),
                float(bool(words) and words[0] == "the"),
                float(bool(words) and words[0] == "i"),
                keyword_count,
            ],
            dtype=np.float32,
        )
    if not np.isfinite(result).all():
        raise RuntimeError("Spooky style features include non-finite values")
    return result, {
        "schema": "evomind.spooky.style_feature_contract.v1",
        "rows": len(texts),
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "vocabulary_sha256": _vocabulary_sha256(),
        "fit_scope": "none_deterministic_public_text_only",
        "private_labels_used": False,
    }

