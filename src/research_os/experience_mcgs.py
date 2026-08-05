"""Deterministic experience layer for Experience-Guided MCGS.

The module is deliberately execution-agnostic.  It converts already-observed
public-validation outcomes into immutable cards, keeps an append-only task board,
selects parents with a quality/progress/novelty utility, and builds bounded,
operator-conditioned retrieval bundles.  Private grader feedback is rejected at
the data boundary and therefore cannot enter search prompts or summary caches.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import struct
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

SCHEMA_VERSION = "evomind.experience_mcgs.v1"
CANONICAL_HASH_SCHEMA = "evomind.canonical_json.f64.v1"
SUMMARY_CACHE_SCHEMA = "evomind.experience_summary_cache.v1"
SUMMARY_ENTRY_SCHEMA = "evomind.experience_summary.v1"
_FORBIDDEN_FIELD = re.compile(
    r"(?:"
    r"(?<![A-Za-z0-9])private(?=$|[\s_-])|"
    r"private[\s_-]*(?:grader|score|feedback|label(?:s)?|metric|evaluation|result|data|split|holdout|test|submission)|"
    r"leaderboard|"
    r"official[\s_-]*(?:rank|score|medal)"
    r")",
    re.I,
)
_SUMMARY_FIELDS = frozenset({"method_overview", "parent_comparison_experience"})
_MISSING = object()


class SearchOperator(str, Enum):
    DRAFT = "Draft"
    IMPROVE = "Improve"
    DEBUG = "Debug"
    CROSSOVER = "Crossover"


class SearchMode(str, Enum):
    LEGACY_UCT = "legacy_uct"
    EXPERIENCE_MCGS_V1 = "experience_mcgs_v1"

    @classmethod
    def parse(cls, value: str | None) -> "SearchMode":
        normalized = (value or "").strip().lower()
        if not normalized and os.environ.get("EXPERIENCE_MCGS_V1") == "1":
            normalized = cls.EXPERIENCE_MCGS_V1.value
        return cls.EXPERIENCE_MCGS_V1 if normalized == cls.EXPERIENCE_MCGS_V1.value else cls.LEGACY_UCT


_MAX_SAFE_INTEGER = (1 << 53) - 1


def _unicode_scalar_text(value: str, *, path: str) -> str:
    """Normalize valid UTF-16 surrogate pairs and reject lone surrogates."""

    result: list[str] = []
    index = 0
    while index < len(value):
        codepoint = ord(value[index])
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 1 >= len(value):
                raise ValueError(f"lone high surrogate is forbidden in canonical JSON: {path}")
            low = ord(value[index + 1])
            if not 0xDC00 <= low <= 0xDFFF:
                raise ValueError(f"lone high surrogate is forbidden in canonical JSON: {path}")
            scalar = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
            result.append(chr(scalar))
            index += 2
            continue
        if 0xDC00 <= codepoint <= 0xDFFF:
            raise ValueError(f"lone low surrogate is forbidden in canonical JSON: {path}")
        result.append(value[index])
        index += 1
    return "".join(result)


def _canonical_number(value: int | float, *, path: str) -> str:
    """Return the exact normalized IEEE-754 binary64 identity for a JSON number.

    JSON values cross the Python/JavaScript boundary.  Encoding the binary64
    payload avoids language-specific decimal rendering (for example ``0.0``
    versus ``0``).  Integer-valued numbers outside JavaScript's safe range are
    rejected so a parser cannot silently collapse distinct integer evidence.
    """

    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise ValueError(f"unsafe JSON integer is forbidden: {path}")
        number = float(value)
    else:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite JSON number is forbidden: {path}")
        if number.is_integer() and abs(number) > _MAX_SAFE_INTEGER:
            raise ValueError(f"unsafe integer-valued JSON number is forbidden: {path}")
    if number == 0.0:
        number = 0.0
    return f"f64:{struct.pack('>d', number).hex()}"


def _canonical_tree(value: Any, *, path: str, active: set[int]) -> dict[str, Any]:
    if value is None:
        return {"$null": True}
    if isinstance(value, bool):
        return {"$boolean": value}
    if isinstance(value, str):
        return {"$string": _unicode_scalar_text(value, path=path)}
    if isinstance(value, (int, float)):
        return {"$number": _canonical_number(value, path=path)}
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"cyclic JSON object is forbidden: {path}")
        active.add(identity)
        try:
            entries: list[tuple[str, dict[str, Any]]] = []
            normalized_keys: set[str] = set()
            for key, child in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"canonical JSON object keys must be strings: {path}")
                normalized_key = _unicode_scalar_text(key, path=f"{path}.<key>")
                if normalized_key in normalized_keys:
                    raise ValueError(f"duplicate normalized canonical JSON key: {path}.{normalized_key}")
                normalized_keys.add(normalized_key)
                entries.append((
                    normalized_key,
                    _canonical_tree(child, path=f"{path}.{normalized_key}", active=active),
                ))
            entries.sort(key=lambda entry: entry[0].encode("utf-8"))
            return {"$object": [[key, child] for key, child in entries]}
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"cyclic JSON array is forbidden: {path}")
        active.add(identity)
        try:
            return {
                "$array": [
                    _canonical_tree(child, path=f"{path}[{index}]", active=active)
                    for index, child in enumerate(value)
                ]
            }
        finally:
            active.remove(identity)
    if isinstance(value, (set, frozenset)):
        raise TypeError(f"sets are not canonical JSON values: {path}")
    raise TypeError(f"unsupported canonical JSON value at {path}: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize a JSON value with the versioned cross-language hash contract."""

    tagged = {"$canonical": [CANONICAL_HASH_SCHEMA, _canonical_tree(value, path="root", active=set())]}
    return json.dumps(tagged, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical(value: Any) -> str:
    return canonical_json(value)


def _sha(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_public_only(value: Any, path: str = "root") -> None:
    """Recursively reject private-evaluation material in keys *and* values.

    Checking keys alone is insufficient: an otherwise benign ``notes`` field can
    still carry a private score or grader response.  Every string leaf is
    therefore checked, including strings nested inside sequences.  The check is
    deliberately performed again when a card is appended and when a summary is
    loaded from disk so neither hand-built objects nor poisoned caches bypass the
    builder boundary.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if _FORBIDDEN_FIELD.search(key_text):
                raise ValueError(f"private evaluation field is forbidden in Experience Board: {path}.{key}")
            _assert_public_only(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, child in enumerate(value):
            _assert_public_only(child, f"{path}[{index}]")
    elif isinstance(value, str) and _FORBIDDEN_FIELD.search(value):
        raise ValueError(f"private evaluation value is forbidden in Experience Board: {path}")


def _deep_freeze_json(value: Any, path: str = "root") -> Any:
    """Return a canonical, deeply immutable representation of JSON content."""

    if isinstance(value, Mapping):
        items: list[tuple[str, Any]] = []
        seen: set[str] = set()
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"canonical JSON object keys must be strings: {path}")
            if key in seen:
                raise ValueError(f"duplicate canonical JSON key: {path}.{key}")
            seen.add(key)
            items.append((key, _deep_freeze_json(child, f"{path}.{key}")))
        return MappingProxyType(dict(sorted(items, key=lambda item: item[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_json(child, f"{path}[{index}]") for index, child in enumerate(value))
    if isinstance(value, (set, frozenset)):
        raise TypeError(f"sets are not canonical JSON values: {path}")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _unicode_scalar_text(value, path=path)
    if isinstance(value, int):
        _canonical_number(value, path=path)
        return value
    if isinstance(value, float):
        _canonical_number(value, path=path)
        return 0.0 if value == 0.0 else value
    raise TypeError(f"unsupported canonical JSON value at {path}: {type(value).__name__}")


def _deep_thaw_json(value: Any) -> Any:
    """Create a detached JSON-serializable copy of frozen canonical content."""

    if isinstance(value, Mapping):
        return {str(key): _deep_thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw_json(child) for child in value]
    return value


def _canonical_frozen_json(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    candidate: Mapping[str, Any] = value or {}
    _assert_public_only(candidate)
    frozen = _deep_freeze_json(candidate)
    if not isinstance(frozen, Mapping):  # defensive: the type contract is a mapping
        raise TypeError("canonical provenance must be a JSON object")
    return frozen


def normalize_error_signature(error: str) -> str:
    text = (error or "").replace("\\", "/")
    text = re.sub(r"[A-Za-z]:/[^\s:\n]+", "<path>", text)
    text = re.sub(r"/(?:[^\s/:]+/){1,}[^\s:\n]+", "<path>", text)
    text = re.sub(r"\b0x[0-9a-f]+\b", "<addr>", text, flags=re.I)
    text = re.sub(r"\b\d{4,}\b", "<n>", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    salient = next((line for line in reversed(lines) if "error" in line.lower() or "exception" in line.lower()), lines[-1] if lines else "")
    return _sha(salient.lower())[:24] if salient else ""


def _ast_token_stream(node: ast.AST) -> Iterator[str]:
    """Yield a deterministic, position-aware structural AST token stream.

    ``set(type(node) for ast.walk(...))`` loses ordering and multiplicity, making
    structurally different programs such as ``a + b * c`` and ``(a + b) * c``
    indistinguishable.  Field-boundary and enter/leave tokens retain that shape.
    Identifier and literal payloads are represented by type rather than value so
    novelty measures method structure instead of variable naming.
    """

    yield f"enter:{type(node).__name__}"
    for field_name, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            yield f"field:{field_name}"
            yield from _ast_token_stream(value)
        elif isinstance(value, list):
            # Ignore empty optional lists so common syntax is stable across
            # Python AST schema additions (for example 3.12 ``type_params``).
            if not value:
                continue
            yield f"list:{field_name}:{len(value)}"
            for index, child in enumerate(value):
                if isinstance(child, ast.AST):
                    yield f"item:{field_name}:{index}"
                    yield from _ast_token_stream(child)
        elif value is not None:
            if field_name in {"id", "arg", "name", "attr"}:
                scalar = "identifier"
            elif field_name in {"value", "kind"}:
                scalar = type(value).__name__
            else:
                scalar = str(value)
            yield f"scalar:{field_name}:{scalar}"
    yield f"leave:{type(node).__name__}"


def _deterministic_ast_shingles(tree: ast.AST, *, width: int = 4) -> tuple[str, ...]:
    tokens = tuple(_ast_token_stream(tree))
    if not tokens:
        return ()
    effective_width = max(1, min(int(width), len(tokens)))
    counts = Counter(
        _sha(_canonical(tokens[index:index + effective_width]))[:24]
        for index in range(0, len(tokens) - effective_width + 1)
    )
    # Count is encoded to retain multiplicity while the surrounding feature set
    # remains compact.  A whole-tree hash guards very short or unusual trees.
    shingles = [f"ast_shingle:{digest}:{counts[digest]}" for digest in sorted(counts)]
    shingles.append(f"ast_tree:{_sha(_canonical(tokens))[:24]}")
    return tuple(shingles)


def structural_features(code: str, operator: SearchOperator, method_family: str, error_signature: str = "") -> tuple[str, ...]:
    features = {f"operator:{operator.value}", f"family:{method_family.strip().lower() or 'unknown'}"}
    if error_signature:
        features.add(f"error:{error_signature}")
    try:
        tree = ast.parse(code or "")
        features.update(f"ast:{type(node).__name__}" for node in ast.walk(tree))
        features.update(_deterministic_ast_shingles(tree))
        features.update(
            f"call:{node.func.id.lower()}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        )
    except SyntaxError:
        features.add("ast:SyntaxError")
        features.add(f"ast_syntax_hash:{_sha(code or '')[:24]}")
    return tuple(sorted(features))


@dataclass(frozen=True)
class ExperienceCost:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    wall_seconds: float = 0.0
    gpu_seconds: float = 0.0
    estimated_cost_usd: float = 0.0


@dataclass(frozen=True)
class ExperienceCard:
    schema: str
    card_id: str
    task_id: str
    node_id: str
    parent_ids: tuple[str, ...]
    operator: SearchOperator
    method_family: str
    code_hash: str
    data_hashes: tuple[str, ...]
    prompt_hash: str
    execution_hash: str
    execution_id: str
    status: str
    public_validation_score: float | None
    quality: float | None
    progress: float
    novelty: float
    metric_direction: str
    error_signature: str
    structural_fingerprint: tuple[str, ...]
    cost: ExperienceCost
    provenance: Mapping[str, Any] = field(default_factory=dict)
    summary: str = ""

    def __post_init__(self) -> None:
        operator = self.operator if isinstance(self.operator, SearchOperator) else SearchOperator(str(self.operator))
        resolved_cost = self.cost
        if isinstance(resolved_cost, Mapping):
            resolved_cost = ExperienceCost(**dict(resolved_cost))
        if not isinstance(resolved_cost, ExperienceCost):
            raise TypeError("cost must be ExperienceCost or a compatible mapping")
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "parent_ids", tuple(str(item) for item in self.parent_ids))
        object.__setattr__(self, "data_hashes", tuple(str(item) for item in self.data_hashes))
        object.__setattr__(self, "structural_fingerprint", tuple(str(item) for item in self.structural_fingerprint))
        object.__setattr__(self, "cost", resolved_cost)
        object.__setattr__(self, "provenance", _canonical_frozen_json(self.provenance))
        object.__setattr__(self, "summary", str(self.summary or ""))
        if self.schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported Experience Card schema: {self.schema!r}")
        if self.public_validation_score is not None and not math.isfinite(float(self.public_validation_score)):
            raise ValueError("public_validation_score must be finite")
        if not math.isfinite(float(self.progress)) or not -1.0 <= float(self.progress) <= 1.0:
            raise ValueError("progress must be finite and in [-1,1]")
        if not math.isfinite(float(self.novelty)) or not 0.0 <= float(self.novelty) <= 1.0:
            raise ValueError("novelty must be finite and in [0,1]")
        _assert_public_only(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        # Explicit serialization is intentional: dataclasses.asdict performs a
        # deepcopy, which is incompatible with MappingProxyType and would also
        # obscure the guarantee that the returned object is detached.
        return {
            "schema": self.schema,
            "card_id": self.card_id,
            "task_id": self.task_id,
            "node_id": self.node_id,
            "parent_ids": list(self.parent_ids),
            "operator": self.operator.value,
            "method_family": self.method_family,
            "code_hash": self.code_hash,
            "data_hashes": list(self.data_hashes),
            "prompt_hash": self.prompt_hash,
            "execution_hash": self.execution_hash,
            "execution_id": self.execution_id,
            "status": self.status,
            "public_validation_score": self.public_validation_score,
            "quality": self.quality,
            "progress": self.progress,
            "novelty": self.novelty,
            "metric_direction": self.metric_direction,
            "error_signature": self.error_signature,
            "structural_fingerprint": list(self.structural_fingerprint),
            "cost": asdict(self.cost),
            "provenance": _deep_thaw_json(self.provenance),
            "summary": self.summary,
        }

    @property
    def content_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("card_id", None)
        return _sha(_canonical(payload))


class ExperienceCardBuilder:
    @staticmethod
    def build(
        *,
        task_id: str,
        node_id: str,
        parent_ids: Sequence[str] = (),
        operator: SearchOperator | str,
        method_family: str,
        code: str,
        data_hashes: Sequence[str] = (),
        prompt: str = "",
        execution_id: str = "",
        status: str,
        public_validation_score: float | None,
        metric_direction: str = "maximize",
        error: str = "",
        cost: ExperienceCost | None = None,
        provenance: Mapping[str, Any] | None = None,
        summary: str = "",
        progress: float = 0.0,
        novelty: float = 1.0,
    ) -> ExperienceCard:
        op = operator if isinstance(operator, SearchOperator) else SearchOperator(str(operator))
        frozen_prov = _canonical_frozen_json(provenance)
        prov = _deep_thaw_json(frozen_prov)
        if public_validation_score is not None and not math.isfinite(float(public_validation_score)):
            raise ValueError("public_validation_score must be finite")
        if not math.isfinite(float(progress)) or not -1.0 <= float(progress) <= 1.0:
            raise ValueError("progress must be finite and in [-1,1]")
        if not math.isfinite(float(novelty)) or not 0.0 <= float(novelty) <= 1.0:
            raise ValueError("novelty must be finite and in [0,1]")
        error_signature = normalize_error_signature(error)
        resolved_cost = cost or ExperienceCost()
        direction = -1.0 if str(metric_direction).lower() in {"minimize", "lower", "lower_is_better"} else 1.0
        quality = None if public_validation_score is None else direction * float(public_validation_score)
        execution_hash = _sha(_canonical({
            "execution_id": str(execution_id or node_id),
            "status": str(status),
            "public_validation_score": public_validation_score,
            "metric_direction": str(metric_direction or "maximize"),
            "error_signature": error_signature,
            "cost": asdict(resolved_cost),
        }))
        base = {
            "schema": SCHEMA_VERSION,
            "task_id": str(task_id),
            "node_id": str(node_id),
            "parent_ids": sorted({str(item) for item in parent_ids if item}),
            "operator": op.value,
            "method_family": str(method_family or "unknown"),
            "code_hash": _sha(code or ""),
            "data_hashes": sorted({str(item).lower() for item in data_hashes if item}),
            "prompt_hash": _sha(prompt or ""),
            "execution_hash": execution_hash,
            "execution_id": str(execution_id or node_id),
            "status": str(status),
            "public_validation_score": None if public_validation_score is None else float(public_validation_score),
            "quality": quality,
            "progress": float(progress),
            "novelty": float(novelty),
            "metric_direction": str(metric_direction or "maximize"),
            "error_signature": error_signature,
            "structural_fingerprint": list(structural_features(code, op, method_family, error_signature)),
            "cost": asdict(resolved_cost),
            "provenance": prov,
            "summary": str(summary or ""),
        }
        card_id = f"exp_{_sha(_canonical(base))[:24]}"
        return ExperienceCard(
            schema=SCHEMA_VERSION,
            card_id=card_id,
            task_id=base["task_id"],
            node_id=base["node_id"],
            parent_ids=tuple(base["parent_ids"]),
            operator=op,
            method_family=base["method_family"],
            code_hash=base["code_hash"],
            data_hashes=tuple(base["data_hashes"]),
            prompt_hash=base["prompt_hash"],
            execution_hash=base["execution_hash"],
            execution_id=base["execution_id"],
            status=base["status"],
            public_validation_score=base["public_validation_score"],
            quality=base["quality"],
            progress=base["progress"],
            novelty=base["novelty"],
            metric_direction=base["metric_direction"],
            error_signature=error_signature,
            structural_fingerprint=tuple(base["structural_fingerprint"]),
            cost=resolved_cost,
            provenance=frozen_prov,
            summary=base["summary"],
        )


class _AppendOnlyCardView(Mapping[str, ExperienceCard]):
    """Read-only live view used by legacy selectors.

    The historical selector called ``cards.pop(id, None)`` before appending a
    verified observation.  Destructive removal would violate the paper's global
    board contract, so ``pop`` is retained solely as a non-destructive peek.  All
    other mutation methods are absent and item assignment/deletion raises the
    normal Mapping ``TypeError``.
    """

    __slots__ = ("__store",)

    def __init__(self, store: dict[str, ExperienceCard]) -> None:
        self.__store = store

    def __getitem__(self, key: str) -> ExperienceCard:
        return self.__store[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__store)

    def __len__(self) -> int:
        return len(self.__store)

    def pop(self, key: str, default: Any = _MISSING) -> ExperienceCard | Any:
        if key in self.__store:
            return self.__store[key]
        if default is _MISSING:
            raise KeyError(key)
        return default


class ExperienceBoard:
    """Task-global, hash-deduplicated and strictly append-only experience log."""

    __slots__ = ("_task_id", "_cards", "_append_order", "_cards_view")

    def __init__(
        self,
        task_id: str,
        cards: Mapping[str, ExperienceCard] | Iterable[ExperienceCard] | None = None,
    ) -> None:
        self._task_id = str(task_id)
        self._cards: dict[str, ExperienceCard] = {}
        self._append_order: list[str] = []
        self._cards_view = _AppendOnlyCardView(self._cards)
        if cards:
            source = cards.values() if isinstance(cards, Mapping) else cards
            for card in source:
                self.append(card)

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def cards(self) -> Mapping[str, ExperienceCard]:
        return self._cards_view

    @property
    def append_order(self) -> tuple[str, ...]:
        return tuple(self._append_order)

    def append(self, card: ExperienceCard) -> bool:
        if not isinstance(card, ExperienceCard):
            raise TypeError("Experience Board accepts ExperienceCard instances only")
        if card.task_id != self.task_id:
            raise ValueError(f"card task {card.task_id!r} does not match board {self.task_id!r}")
        payload = card.to_dict()
        _assert_public_only(payload)
        existing = self._cards.get(card.card_id)
        if existing is not None:
            if existing != card:
                raise ValueError(f"immutable card collision: {card.card_id}")
            return False
        expected_card_id = f"exp_{card.content_hash[:24]}"
        if card.card_id != expected_card_id:
            raise ValueError(
                f"Experience Card content hash mismatch: expected {expected_card_id}, got {card.card_id}"
            )
        self._cards[card.card_id] = card
        self._append_order.append(card.card_id)
        return True

    def by_node(self, node_id: str) -> ExperienceCard | None:
        node = str(node_id)
        for card_id in reversed(self._append_order):
            candidate = self._cards[card_id]
            if candidate.node_id == node:
                return candidate
        return None

    def successful(self) -> list[ExperienceCard]:
        return [
            self._cards[card_id]
            for card_id in self._append_order
            if self._cards[card_id].status == "success"
            and self._cards[card_id].public_validation_score is not None
        ]

    def aggregate(self) -> dict[str, Any]:
        cards = [self._cards[card_id] for card_id in self._append_order]
        operator_counts = Counter(card.operator.value for card in cards)
        family_counts = Counter(card.method_family for card in cards)
        status_counts = Counter(card.status for card in cards)
        direction_counts = Counter(card.metric_direction for card in cards if card.public_validation_score is not None)
        error_counts = Counter(card.error_signature for card in cards if card.error_signature)
        node_ids = {card.node_id for card in cards}
        lineage_edges = sorted({(parent_id, card.node_id) for card in cards for parent_id in card.parent_ids})
        root_nodes = sorted({card.node_id for card in cards if not card.parent_ids})
        latest_by_node: dict[str, str] = {}
        for card in cards:
            latest_by_node[card.node_id] = card.card_id

        scored = [
            card for card in cards
            if card.status == "success" and card.public_validation_score is not None and card.quality is not None
        ]
        ranked = sorted(scored, key=lambda card: (-float(card.quality), card.card_id))
        best = ranked[0] if ranked else None
        running_best: float | None = None
        new_best_count = 0
        for card in scored:
            quality = float(card.quality)
            if running_best is None or quality > running_best:
                running_best = quality
                new_best_count += 1

        return {
            "scope": "task_global",
            "card_count": len(cards),
            "unique_node_count": len(node_ids),
            "scored_card_count": len(scored),
            "operator_counts": dict(sorted(operator_counts.items())),
            "method_family_counts": dict(sorted(family_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "metric_direction_counts": dict(sorted(direction_counts.items())),
            "error_signature_counts": dict(sorted(error_counts.items())),
            "lineage": {
                "root_node_ids": root_nodes,
                "edge_count": len(lineage_edges),
                "edges": [
                    {"parent_node_id": parent_id, "child_node_id": child_id}
                    for parent_id, child_id in lineage_edges
                ],
                "latest_card_by_node": dict(sorted(latest_by_node.items())),
            },
            "public_validation": {
                "best_card_id": best.card_id if best else None,
                "best_node_id": best.node_id if best else None,
                "best_score": best.public_validation_score if best else None,
                "best_quality": best.quality if best else None,
                "new_best_count": new_best_count,
            },
            "cost": {
                "prompt_tokens": sum(card.cost.prompt_tokens for card in cards),
                "completion_tokens": sum(card.cost.completion_tokens for card in cards),
                "total_tokens": sum(card.cost.total_tokens for card in cards),
                "wall_seconds": sum(card.cost.wall_seconds for card in cards),
                "gpu_seconds": sum(card.cost.gpu_seconds for card in cards),
                "estimated_cost_usd": sum(card.cost.estimated_cost_usd for card in cards),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        cards = [self._cards[card_id].to_dict() for card_id in self._append_order]
        aggregate = self.aggregate()
        chain_head = "0" * 64
        for card_id in self._append_order:
            card = self._cards[card_id]
            chain_head = _sha(_canonical({
                "previous": chain_head,
                "card_id": card.card_id,
                "card_hash": card.content_hash,
            }))
        hash_payload = {
            "schema": SCHEMA_VERSION,
            "hash_canonicalization": CANONICAL_HASH_SCHEMA,
            "task_id": self.task_id,
            "append_order": list(self._append_order),
            "append_chain_head": chain_head,
            "cards": cards,
            "task_global_aggregation": aggregate,
        }
        _assert_public_only(hash_payload)
        return {
            **hash_payload,
            "append_only": True,
            "deduplication_key": "canonical_card_hash",
            "board_hash": _sha(_canonical(hash_payload)),
        }

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
        return output


@dataclass(frozen=True)
class SelectionScore:
    node_id: str
    card_id: str
    quality: float
    progress: float
    novelty: float
    exploration: float
    utility: float
    visits: int
    parent_visits: int


@dataclass(frozen=True)
class SelectionTrace:
    schema: str
    selected_node_id: str
    selected_card_id: str
    exploration_c: float
    candidates: tuple[SelectionScore, ...]
    tie_break: str = "utility_desc,quality_desc,card_id_asc"
    operator: SearchOperator | None = None
    selection_reason: str = ""
    selected_parent_ids: tuple[str, ...] = ()
    budget_snapshot: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        resolved_operator = (
            self.operator
            if self.operator is None or isinstance(self.operator, SearchOperator)
            else SearchOperator(str(self.operator))
        )
        object.__setattr__(self, "operator", resolved_operator)
        object.__setattr__(self, "selected_parent_ids", tuple(str(item) for item in self.selected_parent_ids))
        object.__setattr__(self, "budget_snapshot", _canonical_frozen_json(self.budget_snapshot))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "selected_node_id": self.selected_node_id,
            "selected_card_id": self.selected_card_id,
            "exploration_c": self.exploration_c,
            "candidates": [asdict(item) for item in self.candidates],
            "tie_break": self.tie_break,
            "operator": self.operator.value if self.operator is not None else "",
            "selection_reason": self.selection_reason,
            "selected_parent_ids": list(self.selected_parent_ids),
            "budget_snapshot": _deep_thaw_json(self.budget_snapshot),
        }


def _direction(card: ExperienceCard) -> int:
    return -1 if card.metric_direction.lower() in {"minimize", "lower", "lower_is_better"} else 1


def _quality_percentiles(cards: Sequence[ExperienceCard]) -> dict[str, float]:
    successful = [card for card in cards if card.status == "success" and card.public_validation_score is not None]
    if not successful:
        return {card.card_id: 0.0 for card in cards}
    ordered = sorted(successful, key=lambda card: (_direction(card) * float(card.public_validation_score), card.card_id))
    if len(ordered) == 1:
        return {card.card_id: (1.0 if card.card_id == ordered[0].card_id else 0.0) for card in cards}
    denominator = max(1, len(ordered) - 1)
    result = {card.card_id: index / denominator for index, card in enumerate(ordered)}
    result.update({card.card_id: 0.0 for card in cards if card.card_id not in result})
    return result


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def select_experience_parent(
    board: ExperienceBoard,
    *,
    visits: Mapping[str, int] | None = None,
    parent_visits: Mapping[str, int] | None = None,
    exploration_c: float = 1.0,
) -> SelectionTrace:
    cards = sorted(board.cards.values(), key=lambda card: card.card_id)
    if not cards:
        raise ValueError("Experience Board has no cards")
    quality = _quality_percentiles(cards)
    scores: list[SelectionScore] = []
    for card in cards:
        parent_quality = [quality[parent.card_id] for parent_id in card.parent_ids if (parent := board.by_node(parent_id)) is not None]
        progress = max(-1.0, min(1.0, quality[card.card_id] - max(parent_quality))) if parent_quality else 0.0
        peers = [peer for peer in cards if peer.card_id != card.card_id]
        novelty = 1.0 - max((_jaccard(card.structural_fingerprint, peer.structural_fingerprint) for peer in peers), default=0.0)
        node_visits = max(0, int((visits or {}).get(card.node_id, 0)))
        p_visits = max(0, int((parent_visits or visits or {}).get(card.parent_ids[0], 0))) if card.parent_ids else sum(max(0, int(v)) for v in (visits or {}).values())
        exploration = exploration_c * math.sqrt(math.log1p(p_visits) / (1 + node_visits)) if p_visits > 0 else 0.0
        utility = quality[card.card_id] + 0.6 * progress + 0.3 * novelty + exploration
        scores.append(SelectionScore(card.node_id, card.card_id, quality[card.card_id], progress, novelty, exploration, utility, node_visits, p_visits))
    ranked = sorted(scores, key=lambda score: (-score.utility, -score.quality, score.card_id))
    return SelectionTrace(SCHEMA_VERSION, ranked[0].node_id, ranked[0].card_id, exploration_c, tuple(ranked))


def route_operator(board: ExperienceBoard, *, no_new_best_expansions: int = 0) -> SearchOperator:
    cards = sorted(board.cards.values(), key=lambda card: (card.execution_id, card.card_id))
    if not cards:
        return SearchOperator.DRAFT
    latest = cards[-1]
    if latest.status != "success" or latest.error_signature:
        return SearchOperator.DEBUG
    successful_families = {card.method_family for card in cards if card.status == "success" and card.public_validation_score is not None}
    if no_new_best_expansions >= 3 and len(successful_families) >= 2:
        return SearchOperator.CROSSOVER
    return SearchOperator.IMPROVE


@dataclass(frozen=True)
class RetrievalBundle:
    schema: str
    operator: SearchOperator
    card_ids: tuple[str, ...]
    ranking_scores: tuple[tuple[str, float], ...]
    estimated_tokens: int
    max_cards: int
    max_tokens: int
    truncated_by: str
    cache_key: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["operator"] = self.operator.value
        return payload


def _retrieval_card_context_line(card: ExperienceCard) -> str:
    """Render the compact public projection that is actually sent to a prompt.

    Retrieval must budget the prompt-facing projection, not the full canonical
    card.  The canonical card intentionally retains hashes, provenance and a
    potentially large AST fingerprint; using that audit payload for token
    admission can reject every card before the lazy summary cache gets a
    chance to run.
    """

    score = (
        "not_recorded"
        if card.public_validation_score is None
        else f"{card.public_validation_score:.8g}"
    )
    overview = str(
        card.summary
        or card.provenance.get("changes_summary", "")
        or card.method_family
    )
    return (
        f"- {card.node_id} | {card.operator.value} | family={card.method_family} | "
        f"status={card.status} | public_validation={score} | "
        f"error_signature={card.error_signature or 'none'} | {overview}"
    )


def retrieve_experience(
    board: ExperienceBoard,
    operator: SearchOperator,
    *,
    selected_node_id: str = "",
    max_cards: int = 8,
    max_tokens: int = 6000,
) -> RetrievalBundle:
    cards = list(board.cards.values())
    selected = board.by_node(selected_node_id)
    quality = _quality_percentiles(cards)

    def parent_progress(card: ExperienceCard) -> float:
        parent_scores = [quality[parent.card_id] for parent_id in card.parent_ids if (parent := board.by_node(parent_id)) is not None]
        return max(-1.0, min(1.0, quality[card.card_id] - max(parent_scores))) if parent_scores else 0.0

    successful = [card for card in cards if card.status == "success" and card.public_validation_score is not None]
    global_best = sorted(successful, key=lambda card: (-quality[card.card_id], card.card_id))[:1]

    def diverse_fill(primary: list[ExperienceCard], candidates: Iterable[ExperienceCard]) -> list[ExperienceCard]:
        result = list(primary)
        remaining = [card for card in candidates if card not in result]
        while remaining:
            card = max(
                remaining,
                key=lambda item: (
                    min((1.0 - _jaccard(item.structural_fingerprint, chosen.structural_fingerprint) for chosen in result), default=1.0),
                    quality[item.card_id],
                    -len(item.method_family),
                    item.card_id,
                ),
            )
            result.append(card)
            remaining.remove(card)
        return result

    if operator is SearchOperator.DRAFT:
        # One high-quality representative per family before any family repeats.
        family_best: dict[str, ExperienceCard] = {}
        for card in sorted(successful, key=lambda item: (-quality[item.card_id], item.card_id)):
            family_best.setdefault(card.method_family, card)
        ordered = diverse_fill([], family_best.values())
        ordered.extend(card for card in sorted(successful, key=lambda item: (-quality[item.card_id], item.card_id)) if card not in ordered)
    elif operator is SearchOperator.IMPROVE:
        same_family = [
            card for card in successful
            if selected is not None and card.method_family == selected.method_family and parent_progress(card) > 0
        ]
        same_family.sort(key=lambda item: (-parent_progress(item), -quality[item.card_id], item.card_id))
        ordered = list(same_family)
        for card in global_best:
            if card not in ordered:
                ordered.append(card)
        ordered = diverse_fill(ordered, successful)
    elif operator is SearchOperator.DEBUG:
        signature = selected.error_signature if selected else ""
        repaired = [
            card for card in successful
            if signature and signature in {
                str(card.provenance.get("resolved_error_signature") or ""),
                str(card.provenance.get("fixes_error_signature") or ""),
                str(card.provenance.get("parent_error_signature") or ""),
            }
        ]
        repaired.sort(key=lambda item: (-quality[item.card_id], item.card_id))
        same_failure = [card for card in cards if signature and card.error_signature == signature and card not in repaired]
        same_failure.sort(key=lambda item: (item.status != "success", -quality[item.card_id], item.card_id))
        recent = sorted((card for card in cards if card not in repaired and card not in same_failure), key=lambda item: (item.execution_id, item.card_id), reverse=True)
        ordered = repaired + same_failure + recent
    else:
        # Two complementary method families, capped at three cards each, then
        # diversity fill.  Prefer the selected family plus the best other family.
        by_family: dict[str, list[ExperienceCard]] = {}
        for card in successful:
            by_family.setdefault(card.method_family, []).append(card)
        for values in by_family.values():
            values.sort(key=lambda item: (-quality[item.card_id], item.card_id))
        ranked_families = sorted(by_family, key=lambda family: (-quality[by_family[family][0].card_id], family))
        first_family = selected.method_family if selected and selected.method_family in by_family else (ranked_families[0] if ranked_families else "")
        second_family = next((family for family in ranked_families if family != first_family), "")
        ordered = list(by_family.get(first_family, [])[:3]) + list(by_family.get(second_family, [])[:3])
        ordered = diverse_fill(ordered, successful)

    # Preserve deterministic order and remove any accidental duplicates.
    deduplicated: list[ExperienceCard] = []
    seen_card_ids: set[str] = set()
    for card in ordered:
        if card.card_id not in seen_card_ids:
            deduplicated.append(card)
            seen_card_ids.add(card.card_id)
    ordered = deduplicated
    chosen: list[str] = []
    ranking_scores: list[tuple[str, float]] = []
    estimated = 0
    truncated_by = "none"
    for rank_index, card in enumerate(ordered):
        if len(chosen) >= max_cards:
            truncated_by = "card_limit"
            break
        text = _retrieval_card_context_line(card)
        tokens = max(1, math.ceil(len(text) / 4))
        if estimated + tokens > max_tokens:
            truncated_by = "token_limit"
            break
        chosen.append(card.card_id)
        ranking_scores.append((card.card_id, float(len(ordered) - rank_index) + quality[card.card_id]))
        estimated += tokens
    key_payload = {"schema": SCHEMA_VERSION, "operator": operator.value, "selected_node_id": selected_node_id, "cards": chosen, "max_cards": max_cards, "max_tokens": max_tokens}
    return RetrievalBundle(SCHEMA_VERSION, operator, tuple(chosen), tuple(ranking_scores), estimated, max_cards, max_tokens, truncated_by, _sha(_canonical(key_payload)))


class LazySummaryCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, Mapping[str, Any]] = {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                isinstance(raw, Mapping)
                and set(raw) == {"schema", "entry_schema", "entries"}
                and raw.get("schema") == SUMMARY_CACHE_SCHEMA
                and raw.get("entry_schema") == SUMMARY_ENTRY_SCHEMA
                and isinstance(raw.get("entries"), Mapping)
            ):
                for key, entry in raw["entries"].items():
                    try:
                        validated = self._validate_persisted_entry(str(key), entry)
                    except (TypeError, ValueError):
                        continue
                    self._entries[str(key)] = validated
        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
            pass

    @property
    def entries(self) -> Mapping[str, Mapping[str, Any]]:
        # Expose detached immutable snapshots; callers cannot poison a future hit.
        return MappingProxyType({
            key: MappingProxyType(_deep_thaw_json(value))
            for key, value in sorted(self._entries.items())
        })

    @staticmethod
    def _cache_key_from_hash(card_hash: str, summarizer_model: str, prompt_template_hash: str) -> str:
        # Preserve the original deterministic cache-key contract (schema +
        # canonical card id + model + template hash) while persisting the full
        # content hash in the stricter entry schema.
        normalized_hash = str(card_hash).lower()
        card_id = f"exp_{normalized_hash[:24]}" if re.fullmatch(r"[0-9a-f]{64}", normalized_hash) else str(card_hash)
        return _sha(json.dumps({
            "schema": SCHEMA_VERSION,
            "card_id": card_id,
            "summarizer_model": str(summarizer_model),
            "prompt_template_hash": str(prompt_template_hash),
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))

    @staticmethod
    def cache_key(card: ExperienceCard, summarizer_model: str, prompt_template_hash: str) -> str:
        return LazySummaryCache._cache_key_from_hash(
            card.content_hash,
            summarizer_model,
            prompt_template_hash,
        )

    @staticmethod
    def _validate_summary_content(value: Any) -> Mapping[str, str]:
        if not isinstance(value, Mapping):
            raise TypeError(
                "summary must be a mapping with method_overview and parent_comparison_experience"
            )
        if set(value) != _SUMMARY_FIELDS:
            raise ValueError(
                "summary fields must be exactly method_overview and parent_comparison_experience"
            )
        normalized: dict[str, str] = {}
        for field_name in sorted(_SUMMARY_FIELDS):
            field_value = value[field_name]
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"summary field {field_name} must be a non-empty string")
            normalized[field_name] = field_value.strip()
        _assert_public_only(normalized, "summary")
        return MappingProxyType(normalized)

    @classmethod
    def _validate_persisted_entry(cls, key: str, value: Any) -> Mapping[str, Any]:
        required = {
            "schema",
            "card_hash",
            "summarizer_model",
            "prompt_template_hash",
            "temperature",
            "method_overview",
            "parent_comparison_experience",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("invalid summary cache entry schema")
        if value.get("schema") != SUMMARY_ENTRY_SCHEMA or value.get("temperature") != 0:
            raise ValueError("invalid summary cache entry metadata")
        if not isinstance(value.get("summarizer_model"), str) or not value["summarizer_model"].strip():
            raise ValueError("invalid summary cache entry summarizer_model")
        if not isinstance(value.get("prompt_template_hash"), str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", value["prompt_template_hash"]
        ):
            raise ValueError("invalid summary cache entry prompt_template_hash")
        if not isinstance(value.get("card_hash"), str) or not re.fullmatch(r"[0-9a-f]{64}", value["card_hash"]):
            raise ValueError("invalid summary cache entry card_hash")
        summary = cls._validate_summary_content({
            "method_overview": value["method_overview"],
            "parent_comparison_experience": value["parent_comparison_experience"],
        })
        expected_key = cls._cache_key_from_hash(
            value["card_hash"],
            value["summarizer_model"],
            value["prompt_template_hash"],
        )
        if key != expected_key:
            raise ValueError("summary cache key does not match entry metadata")
        entry = {
            "schema": SUMMARY_ENTRY_SCHEMA,
            "card_hash": value["card_hash"],
            "summarizer_model": value["summarizer_model"],
            "prompt_template_hash": value["prompt_template_hash"],
            "temperature": 0,
            "method_overview": summary["method_overview"],
            "parent_comparison_experience": summary["parent_comparison_experience"],
        }
        _assert_public_only(entry, "summary_cache_entry")
        return _deep_freeze_json(entry, "summary_cache_entry")

    def _persist(self) -> None:
        payload = {
            "schema": SUMMARY_CACHE_SCHEMA,
            "entry_schema": SUMMARY_ENTRY_SCHEMA,
            "entries": {
                key: _deep_thaw_json(value)
                for key, value in sorted(self._entries.items())
            },
        }
        _assert_public_only(payload, "summary_cache")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.path)

    def get_or_create(
        self,
        card: ExperienceCard,
        *,
        summarizer_model: str,
        prompt_template_hash: str,
        summarize: Callable[[ExperienceCard], Mapping[str, str]],
    ) -> tuple[dict[str, str], bool]:
        if not isinstance(summarizer_model, str) or not summarizer_model.strip():
            raise ValueError("summarizer_model must be a non-empty string")
        if not isinstance(prompt_template_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", prompt_template_hash):
            raise ValueError("prompt_template_hash must be a 64-character SHA-256 hex digest")
        key = self.cache_key(card, summarizer_model, prompt_template_hash)
        expected_metadata = {
            "card_hash": card.content_hash,
            "summarizer_model": str(summarizer_model),
            "prompt_template_hash": str(prompt_template_hash),
        }
        cached = self._entries.get(key)
        if cached is not None and all(cached.get(name) == value for name, value in expected_metadata.items()):
            return {
                "method_overview": str(cached["method_overview"]),
                "parent_comparison_experience": str(cached["parent_comparison_experience"]),
            }, True

        summary = self._validate_summary_content(summarize(card))
        entry = {
            "schema": SUMMARY_ENTRY_SCHEMA,
            **expected_metadata,
            "temperature": 0,
            "method_overview": summary["method_overview"],
            "parent_comparison_experience": summary["parent_comparison_experience"],
        }
        validated = self._validate_persisted_entry(key, entry)
        self._entries[key] = validated
        self._persist()
        return {
            "method_overview": str(summary["method_overview"]),
            "parent_comparison_experience": str(summary["parent_comparison_experience"]),
        }, False


@dataclass
class BudgetLedger:
    max_nodes: int
    max_total_tokens: int
    max_wall_seconds: float
    max_cost_usd: float | None = None
    nodes: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    wall_seconds: float = 0.0
    gpu_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    terminal_reason: str = ""

    def record(self, cost: ExperienceCost) -> None:
        self.nodes += 1
        self.prompt_tokens += max(0, int(cost.prompt_tokens))
        self.completion_tokens += max(0, int(cost.completion_tokens))
        self.total_tokens += max(0, int(cost.total_tokens or cost.prompt_tokens + cost.completion_tokens))
        self.wall_seconds += max(0.0, float(cost.wall_seconds))
        self.gpu_seconds += max(0.0, float(cost.gpu_seconds))
        self.estimated_cost_usd += max(0.0, float(cost.estimated_cost_usd))
        self.terminal_reason = self.exceeded_reason()

    def exceeded_reason(self) -> str:
        if self.nodes >= self.max_nodes:
            return "node_budget_exhausted"
        if self.total_tokens >= self.max_total_tokens:
            return "token_budget_exhausted"
        if self.wall_seconds >= self.max_wall_seconds:
            return "wall_clock_budget_exhausted"
        if self.max_cost_usd is not None and self.estimated_cost_usd >= self.max_cost_usd:
            return "cost_budget_exhausted"
        return ""

    @property
    def can_continue(self) -> bool:
        return not self.exceeded_reason()

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA_VERSION, **asdict(self), "can_continue": self.can_continue}


def render_retrieval_context(board: ExperienceBoard, bundle: RetrievalBundle) -> str:
    lines = [f"operator={bundle.operator.value} cards={len(bundle.card_ids)} estimated_tokens={bundle.estimated_tokens}"]
    for card_id in bundle.card_ids:
        card = board.cards[card_id]
        lines.append(_retrieval_card_context_line(card))
    return "\n".join(lines)


def shadow_replay(board: ExperienceBoard, *, visits: Mapping[str, int] | None = None) -> dict[str, Any]:
    first = select_experience_parent(board, visits=visits)
    second = select_experience_parent(board, visits=visits)
    first_payload = first.to_dict()
    second_payload = second.to_dict()
    deterministic = _canonical(first_payload) == _canonical(second_payload)
    serialized = board.to_dict()
    _assert_public_only(serialized)
    return {
        "schema": SCHEMA_VERSION,
        "mode": "shadow_replay_no_llm_no_grader",
        "card_count": len(board.cards),
        "selected_node_id": first.selected_node_id,
        "deterministic": deterministic,
        "private_feedback_absent": True,
        "board_hash": serialized["board_hash"],
        "trace_hash": _sha(_canonical(first_payload)),
    }
