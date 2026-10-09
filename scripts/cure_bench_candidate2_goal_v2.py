from __future__ import annotations

"""CURE-Bench candidate-2 with a frozen, offline language representation.

This module is intentionally independent from ``cure_bench_goal_v2.py``.  It
does not download models or data, does not inspect test answers, and refuses
ambiguous answer encodings instead of guessing.  The production runner should
provide a SHA-bound local encoder manifest; the test suite injects a tiny fake
encoder so the split and artifact contracts can be exercised without a GPU.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


SEED = 20260901
PRIOR_SEED = 20260830
HOLDOUT_FRACTION = 0.20
BOOTSTRAP_ROUNDS = 2000
MIN_OPTIONS = 2
MAX_OPTIONS = 26
PRIMARY_BASELINE_SCHEMA = "evomind.cure_bench.primary_baseline_manifest.v1"
PRIMARY_BASELINE_TYPES = {"official_organizer", "publicly_reproducible_strong"}
PRIMARY_BASELINE_SOURCE_AUTHORITIES = {"official_organizer", "peer_reviewed", "official_dataset"}
PRIMARY_BASELINE_METRIC = "accuracy on scored choice rows"
PRIMARY_BASELINE_COMMON_ROLES = {"source", "protocol"}
PRIMARY_BASELINE_STRONG_ROLES = {
    "source",
    "protocol",
    "implementation",
    "data_manifest",
    "environment",
    "reproduction_receipt",
}

_CHOICE_TYPES = {
    "multi_choice": "multi_choice",
    "multiple_choice": "multi_choice",
    "multiple-choice": "multi_choice",
    "open_ended_multi_choice": "open_ended_multi_choice",
    "open-ended-multiple-choice": "open_ended_multi_choice",
    "open-ended-multiple_choice": "open_ended_multi_choice",
}
_OPEN_ENDED_TYPES = {"open_ended", "open-ended", "free_response", "free-response"}
_LABEL_KEYS = {
    "answer",
    "correct_answer",
    "correctanswer",
    "gold",
    "gold_answer",
    "label",
    "labels",
    "target",
    "targets",
    "solution",
    "ground_truth",
    "groundtruth",
}


class ExactGateError(ValueError):
    """A precise, actionable gate rather than an unstructured runtime error."""

    def __init__(self, code: str, reason: str, required_action: str, resume_point: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.required_action = required_action
        self.resume_point = resume_point

    def as_dict(self) -> dict[str, str]:
        return {
            "status": "waiting_exact_gate",
            "code": self.code,
            "reason": self.reason,
            "required_action": self.required_action,
            "resume_point": self.resume_point,
        }


class EncoderProtocol(Protocol):
    manifest_info: Mapping[str, Any]

    def encode(self, texts: Sequence[str], *, batch_size: int = 16) -> np.ndarray:
        ...


@dataclass(frozen=True)
class CanonicalRow:
    row_id: str
    question_type: str
    question: str
    option_keys: tuple[str, ...]
    option_values: tuple[str, ...]
    label: int | None
    scored: bool

    @property
    def option_count(self) -> int:
        return len(self.option_keys)


@dataclass(frozen=True)
class DatasetBundle:
    validation: tuple[CanonicalRow, ...]
    test_ids: tuple[str, ...]
    test_schema_rows: int
    validation_sha256: str
    test_file_sha256: Mapping[str, str]
    answer_schema_audit: Mapping[str, Any]


@dataclass(frozen=True)
class HoldoutLedger:
    """Read-only view of a supervisor-owned, atomically claimed holdout ledger."""

    path: str
    sha256: str
    schema: str
    consumed_hashes: frozenset[str]
    available_hashes: frozenset[str]


@dataclass(frozen=True)
class SplitBundle:
    development: np.ndarray
    holdout: np.ndarray
    prior_holdout: np.ndarray
    inner_train: np.ndarray
    inner_validation: np.ndarray
    prior_holdout_id_sha256: str
    holdout_id_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def atomic_copyfile(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def atomic_joblib_dump(value: Any, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, destination)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    return path.is_symlink() or bool(attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)))


def _has_link_or_reparse_ancestor(path: Path) -> bool:
    absolute = path.expanduser().absolute()
    return any(_is_link_or_reparse(candidate) for candidate in (absolute, *absolute.parents))


def _norm(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).strip().split())


def _fold(value: Any) -> str:
    return _norm(value).casefold()


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def _find_label_keys(value: Any, *, path: str = "", depth: int = 0) -> list[str]:
    """Find answer-like keys recursively without inspecting ordinary text values."""

    if depth > 32:
        raise ExactGateError(
            "TEST_SCHEMA_NESTING_TOO_DEEP",
            "test metadata nesting exceeds the bounded inspection depth",
            "Flatten or validate the official test schema before use",
            "test_schema_validation",
        )
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_folded = re.sub(r"[\s\-]+", "_", _fold(key))
            if key_folded in _LABEL_KEYS:
                found.append(f"{path}.{key}" if path else str(key))
            found.extend(_find_label_keys(child, path=f"{path}.{key}" if path else str(key), depth=depth + 1))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_label_keys(child, path=f"{path}[{index}]", depth=depth + 1))
    return found


def canonical_question_type(value: Any) -> str:
    key = _fold(value).replace(" ", "_")
    if key in _CHOICE_TYPES:
        return _CHOICE_TYPES[key]
    if key in _OPEN_ENDED_TYPES:
        return "open_ended"
    raise ExactGateError(
        "UNSUPPORTED_QUESTION_TYPE",
        f"question_type {key!r} is outside the scored CURE-Bench contract",
        "Provide a frozen adapter for this question type or use the organizer-approved choice subset",
        "dataset_schema_validation",
    )


def options_list(value: Any) -> tuple[list[str], list[str]]:
    if isinstance(value, Mapping):
        if not MIN_OPTIONS <= len(value) <= MAX_OPTIONS:
            raise ExactGateError(
                "OPTION_COUNT_OUT_OF_RANGE",
                "options object must contain between 2 and 26 choices",
                "Repair the validation schema before running candidate-2",
                "dataset_schema_validation",
            )
        raw_pairs = list(value.items())
    elif isinstance(value, list):
        if not MIN_OPTIONS <= len(value) <= MAX_OPTIONS:
            raise ExactGateError(
                "OPTION_COUNT_OUT_OF_RANGE",
                "options list must contain between 2 and 26 choices",
                "Repair the validation schema before running candidate-2",
                "dataset_schema_validation",
            )
        raw_pairs = [(chr(ord("A") + index), item) for index, item in enumerate(value)]
    else:
        raise ExactGateError(
            "OPTIONS_SCHEMA_MISMATCH",
            "choice row options must be an object or list",
            "Use the official CURE-Bench options schema",
            "dataset_schema_validation",
        )

    keys = [_norm(key) for key, _ in raw_pairs]
    values = [_norm(item) for _, item in raw_pairs]
    folded_keys = [_fold(key) for key in keys]
    if any(not key for key in keys) or len(set(folded_keys)) != len(folded_keys):
        raise ExactGateError(
            "OPTION_KEYS_INVALID",
            "option labels must be non-empty and unique",
            "Repair duplicate or empty option labels",
            "dataset_schema_validation",
        )
    if any(not item for item in values):
        raise ExactGateError(
            "OPTION_TEXT_EMPTY",
            "option text must be non-empty",
            "Repair empty option text before training",
            "dataset_schema_validation",
        )
    return keys, values


def answer_index(row: Mapping[str, Any], keys: Sequence[str], values: Sequence[str]) -> tuple[int, str]:
    """Map an answer only when its encoding is unambiguous.

    Integer answers in the interior of the range are deliberately rejected
    unless the row declares ``answer_index_base``.  The old candidate-1 code
    silently interpreted these as zero-based and could shift every label.
    """

    answer = row.get("correct_answer")
    base = row.get("answer_index_base")
    if base is not None:
        if isinstance(base, bool) or str(base) not in {"0", "1"}:
            raise ExactGateError(
                "ANSWER_INDEX_BASE_INVALID",
                "answer_index_base must be explicitly 0 or 1",
                "Correct the row-level answer_index_base metadata",
                "dataset_schema_validation",
            )
        base_int = int(base)
    else:
        base_int = None

    numeric: int | None = None
    if isinstance(answer, int) and not isinstance(answer, bool):
        numeric = int(answer)
    elif isinstance(answer, str) and answer.strip().lstrip("+-").isdigit():
        numeric = int(answer.strip())
    if numeric is not None:
        if base_int is None:
            if numeric == 0:
                return 0, "integer_zero_based_inferred_at_zero"
            if numeric == len(values):
                return numeric - 1, "integer_one_based_inferred_at_length"
            if 1 <= numeric < len(values):
                raise ExactGateError(
                    "AMBIGUOUS_NUMERIC_ANSWER_ENCODING",
                    "numeric correct_answer has both plausible zero-based and one-based meanings",
                    "Declare answer_index_base=0 or answer_index_base=1 in the frozen source metadata",
                    "dataset_schema_validation",
                )
            raise ExactGateError(
                "ANSWER_INDEX_OUT_OF_RANGE",
                "numeric correct_answer is outside the option range",
                "Repair the official answer value",
                "dataset_schema_validation",
            )
        index = numeric if base_int == 0 else numeric - 1
        if not 0 <= index < len(values):
            raise ExactGateError(
                "ANSWER_INDEX_OUT_OF_RANGE",
                "correct_answer does not map into the declared option range",
                "Repair the official answer value or answer_index_base",
                "dataset_schema_validation",
            )
        return index, f"integer_explicit_{base_int}_based"

    folded = _fold(answer)
    key_matches = [index for index, key in enumerate(keys) if _fold(key) == folded]
    if len(key_matches) == 1:
        return key_matches[0], "label"
    value_matches = [index for index, text in enumerate(values) if _fold(text) == folded]
    if len(value_matches) == 1:
        return value_matches[0], "text"
    if len(key_matches) > 1 or len(value_matches) > 1:
        raise ExactGateError(
            "ANSWER_MATCH_AMBIGUOUS",
            "correct_answer matches more than one option",
            "Repair the answer key so each answer maps to one option",
            "dataset_schema_validation",
        )
    raise ExactGateError(
        "ANSWER_UNMAPPABLE",
        "correct_answer cannot be mapped to an option label or text",
        "Repair the official answer key before running candidate-2",
        "dataset_schema_validation",
    )


def _canonical_row(raw: Mapping[str, Any], *, require_answer: bool, row_number: int) -> tuple[CanonicalRow, str]:
    required = {"id", "question", "question_type"}
    if not required.issubset(raw):
        missing = sorted(required - set(raw))
        raise ExactGateError(
            "ROW_SCHEMA_MISMATCH",
            f"row {row_number} is missing required fields: {missing}",
            "Restore the official CURE-Bench row schema",
            "dataset_schema_validation",
        )
    if not _is_scalar(raw["id"]) or not _norm(raw["id"]):
        raise ExactGateError(
            "ROW_ID_INVALID",
            f"row {row_number} has an invalid id",
            "Repair the official row id",
            "dataset_schema_validation",
        )
    row_id = _norm(raw["id"])
    question = _norm(raw["question"])
    if not question:
        raise ExactGateError(
            "QUESTION_EMPTY",
            f"row {row_number} has empty question text",
            "Repair the official question text",
            "dataset_schema_validation",
        )
    question_type = canonical_question_type(raw["question_type"])
    if not require_answer and _find_label_keys(raw):
        raise ExactGateError(
            "TEST_LABELS_PRESENT",
            f"test row {row_number} contains an answer-like field",
            "Provide test files without labels; candidate-2 will not read hidden answers",
            "test_schema_validation",
        )

    keys: list[str] = []
    values: list[str] = []
    label: int | None = None
    answer_encoding = "unscored"
    if question_type in {"multi_choice", "open_ended_multi_choice"}:
        if "options" not in raw:
            raise ExactGateError(
                "OPTIONS_MISSING",
                f"choice row {row_number} has no options field",
                "Restore the official options field",
                "dataset_schema_validation",
            )
        keys, values = options_list(raw["options"])
        if require_answer:
            if "correct_answer" not in raw:
                raise ExactGateError(
                    "VALIDATION_LABEL_MISSING",
                    f"validation row {row_number} lacks correct_answer",
                    "Restore validation labels in the authorized validation file",
                    "dataset_schema_validation",
                )
            label, answer_encoding = answer_index(raw, keys, values)
        scored = require_answer
    else:
        if require_answer and "correct_answer" not in raw:
            raise ExactGateError(
                "VALIDATION_LABEL_MISSING",
                f"validation row {row_number} lacks correct_answer",
                "Restore the authorized validation answer field",
                "dataset_schema_validation",
            )
        # Open-ended rows are retained only for schema accounting.  Their
        # answer text is never loaded into model features or output artifacts.
        scored = False

    return (
        CanonicalRow(
            row_id=row_id,
            question_type=question_type,
            question=question,
            option_keys=tuple(keys),
            option_values=tuple(values),
            label=label,
            scored=scored,
        ),
        answer_encoding,
    )


def load_jsonl(path: Path, *, require_answer: bool) -> tuple[list[CanonicalRow], list[str], dict[str, int]]:
    rows: list[CanonicalRow] = []
    ids: list[str] = []
    encoding_counts: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for row_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ExactGateError(
                        "JSONL_PARSE_ERROR",
                        f"invalid JSON on line {row_number}",
                        "Repair the official JSONL encoding",
                        "dataset_schema_validation",
                    ) from exc
                if not isinstance(value, Mapping):
                    raise ExactGateError(
                        "ROW_NOT_OBJECT",
                        f"line {row_number} is not a JSON object",
                        "Repair the official JSONL rows",
                        "dataset_schema_validation",
                    )
                row, encoding = _canonical_row(value, require_answer=require_answer, row_number=row_number)
                rows.append(row)
                ids.append(row.row_id)
                encoding_counts[encoding] = encoding_counts.get(encoding, 0) + 1
    except FileNotFoundError as exc:
        raise ExactGateError(
            "DATA_FILE_MISSING",
            f"required data file {path.name} is missing",
            "Mount the authorized CURE-Bench data directory",
            "dataset_discovery",
        ) from exc
    if not rows:
        raise ExactGateError(
            "EMPTY_JSONL",
            f"data file {path.name} contains no rows",
            "Mount a non-empty official CURE-Bench file",
            "dataset_schema_validation",
        )
    if len(ids) != len(set(ids)):
        raise ExactGateError(
            "DUPLICATE_ROW_IDS",
            f"data file {path.name} contains duplicate ids",
            "Repair duplicate IDs before splitting",
            "dataset_schema_validation",
        )
    return rows, ids, encoding_counts


def _resolve_data_file(root: Path, names: Iterable[str]) -> Path:
    candidates = [root / "data" / name for name in names]
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate.resolve(strict=True)
    raise ExactGateError(
        "DATA_FILE_MISSING",
        f"none of the fixed data filenames are present: {[name for name in names]}",
        "Mount the authorized CURE-Bench data directory with its official filenames",
        "dataset_discovery",
    )


def load_dataset(data_root: Path) -> DatasetBundle:
    validation_path = _resolve_data_file(
        data_root,
        ("curebench_valset_pharse1.jsonl", "curebench_valset_phase1.jsonl", "curebench_valset.jsonl"),
    )
    test_paths = [
        _resolve_data_file(data_root, ("curebench_testset_phase1.jsonl",)),
        _resolve_data_file(data_root, ("curebench_testset_phase2.jsonl",)),
    ]
    validation, validation_ids, encoding_counts = load_jsonl(validation_path, require_answer=True)
    test_ids: list[str] = []
    test_rows = 0
    test_hashes: dict[str, str] = {}
    for path in test_paths:
        rows, ids, _ = load_jsonl(path, require_answer=False)
        test_rows += len(rows)
        test_ids.extend(ids)
        test_hashes[path.name] = sha256_file(path)
    if len(test_ids) != len(set(test_ids)):
        raise ExactGateError(
            "DUPLICATE_TEST_IDS",
            "test files contain duplicate ids",
            "Repair duplicate test IDs before inference",
            "test_schema_validation",
        )
    overlap = set(validation_ids) & set(test_ids)
    if overlap:
        raise ExactGateError(
            "VALIDATION_TEST_ID_OVERLAP",
            "validation and test IDs overlap",
            "Use the official disjoint validation/test files",
            "dataset_schema_validation",
        )
    return DatasetBundle(
        validation=tuple(validation),
        test_ids=tuple(test_ids),
        test_schema_rows=test_rows,
        validation_sha256=sha256_file(validation_path),
        test_file_sha256=test_hashes,
        answer_schema_audit={
            "schema": "evomind.cure_bench.answer_schema_audit.v3",
            "validation_rows": len(validation),
            "scored_choice_rows": sum(row.scored for row in validation),
            "unscored_open_ended_rows": sum(not row.scored for row in validation),
            "answer_encoding_counts": encoding_counts,
            "ambiguous_numeric_answers": 0,
            "test_labels_read": False,
            "validation_test_id_overlap": 0,
        },
    )


def _id_hash(ids: Sequence[str]) -> str:
    return sha256_bytes(("\n".join(sorted(map(str, ids))) + "\n").encode("utf-8"))


def _stratified_split(indices: np.ndarray, labels: np.ndarray, *, test_size: int, random_state: int, stage: str) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) <= test_size or test_size < 1:
        raise ExactGateError(
            "HOLDOUT_TOO_SMALL",
            f"{stage} cannot allocate a non-empty holdout",
            "Provide enough labeled validation rows for an independent holdout",
            stage,
        )
    try:
        train, holdout = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=labels[indices],
        )
    except ValueError as exc:
        raise ExactGateError(
            "STRATIFIED_SPLIT_UNAVAILABLE",
            f"{stage} cannot form a stratified split: {exc}",
            "Provide a validation set with at least two examples per class in each split",
            stage,
        ) from exc
    return np.asarray(train, dtype=np.int64), np.asarray(holdout, dtype=np.int64)


def make_fresh_splits(rows: Sequence[CanonicalRow]) -> SplitBundle:
    scored = np.asarray([index for index, row in enumerate(rows) if row.scored], dtype=np.int64)
    labels = np.asarray([int(row.label) for row in rows], dtype=np.int64)
    if len(scored) < 30:
        raise ExactGateError(
            "INSUFFICIENT_SCORED_ROWS",
            "candidate-2 requires at least 30 scored validation rows",
            "Mount the complete authorized validation set",
            "split_creation",
        )
    # Reconstruct candidate-1's deterministic outer split only to exclude it;
    # its labels are never used for candidate-2 model selection.
    prior_development, prior_holdout = _stratified_split(
        scored,
        labels,
        test_size=max(1, int(round(len(scored) * HOLDOUT_FRACTION))),
        random_state=PRIOR_SEED,
        stage="prior_split_reconstruction",
    )
    candidate_holdout_size = max(1, int(round(len(scored) * HOLDOUT_FRACTION)))
    development, holdout = _stratified_split(
        prior_development,
        labels,
        test_size=candidate_holdout_size,
        random_state=SEED,
        stage="fresh_holdout_split",
    )
    if set(prior_holdout.tolist()) & set(holdout.tolist()):
        raise ExactGateError(
            "HOLDOUT_OVERLAP",
            "candidate-2 holdout overlaps candidate-1 holdout",
            "Use a new split salt/seed and regenerate the candidate-2 split",
            "split_creation",
        )
    inner_train, inner_validation = _stratified_split(
        development,
        labels,
        test_size=max(1, int(round(len(development) * 0.20))),
        random_state=SEED + 1,
        stage="inner_selection_split",
    )
    if set(inner_train.tolist()) & set(inner_validation.tolist()):
        raise ExactGateError(
            "INNER_SPLIT_OVERLAP",
            "inner train and validation IDs overlap",
            "Regenerate the deterministic inner split",
            "split_creation",
        )
    return SplitBundle(
        development=development,
        holdout=holdout,
        prior_holdout=prior_holdout,
        inner_train=inner_train,
        inner_validation=inner_validation,
        prior_holdout_id_sha256=_id_hash([rows[index].row_id for index in prior_holdout]),
        holdout_id_sha256=_id_hash([rows[index].row_id for index in holdout]),
    )


def row_text(row: CanonicalRow) -> str:
    prefix = f"TYPE_{row.question_type} QUESTION {row.question}"
    if row.question_type == "multi_choice":
        options = " ".join(
            f"OPTION_{row.option_keys[index]} {row.option_values[index]}" for index in range(row.option_count)
        )
        return f"{prefix} {options}"
    # The official protocol forbids presenting options in the first pass for
    # open-ended-multiple-choice rows, so this representation stays question-only.
    return prefix


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        raise ExactGateError("EMPTY_EVALUATION", "cannot score an empty holdout", "Provide a non-empty holdout", "evaluation")
    return float(np.mean(y_true == y_pred))


def _estimator_scores(estimator: Any, matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    classes = np.asarray(getattr(estimator, "classes_", []), dtype=np.int64)
    if len(classes) < 2:
        raise ExactGateError(
            "HEAD_CLASSES_INCOMPLETE",
            "fitted head contains fewer than two answer classes",
            "Provide a balanced validation set with all answer classes represented",
            "head_fit",
        )
    if hasattr(estimator, "predict_proba"):
        scores = np.asarray(estimator.predict_proba(matrix), dtype=float)
    else:
        raw = np.asarray(estimator.decision_function(matrix), dtype=float)
        if raw.ndim == 1:
            scores = np.column_stack((-raw, raw))
        else:
            scores = raw
    if scores.ndim != 2 or scores.shape[1] != len(classes):
        raise ExactGateError(
            "HEAD_SCORE_SHAPE_INVALID",
            "fitted head returned scores with an unexpected class shape",
            "Use a deterministic sklearn classifier head",
            "head_prediction",
        )
    return classes, scores


def predict_valid_options(estimator: Any, matrix: Any, rows: Sequence[CanonicalRow], indices: Sequence[int]) -> np.ndarray:
    classes, scores = _estimator_scores(estimator, matrix)
    predictions: list[int] = []
    for position, row_index in enumerate(indices):
        row = rows[int(row_index)]
        valid = [class_position for class_position, class_value in enumerate(classes) if 0 <= int(class_value) < row.option_count]
        if not valid:
            raise ExactGateError(
                "NO_VALID_OPTION_CLASS",
                f"head has no class valid for row {row.row_id}",
                "Ensure every option class is represented in development labels",
                "head_prediction",
            )
        best = max(valid, key=lambda class_position: (float(scores[position, class_position]), -int(classes[class_position])))
        predictions.append(int(classes[best]))
    return np.asarray(predictions, dtype=np.int64)


def paired_bootstrap(
    y_true: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    *,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = SEED + 2,
) -> dict[str, float | int]:
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 2:
        raise ExactGateError(
            "BOOTSTRAP_ROUNDS_INVALID",
            "paired bootstrap requires an integer number of rounds of at least two",
            "Use the frozen bootstrap round count from the training contract",
            "bootstrap",
        )
    if not (len(y_true) == len(baseline_prediction) == len(candidate_prediction)):
        raise ExactGateError("BOOTSTRAP_LENGTH_MISMATCH", "paired bootstrap arrays differ in length", "Repair holdout prediction alignment", "bootstrap")
    if len(y_true) < 2:
        raise ExactGateError("BOOTSTRAP_HOLDOUT_TOO_SMALL", "at least two holdout rows are required", "Provide a larger fresh holdout", "bootstrap")
    baseline_correct = (y_true == baseline_prediction).astype(float)
    candidate_correct = (y_true == candidate_prediction).astype(float)
    differences = candidate_correct - baseline_correct
    rng = np.random.default_rng(seed)
    samples = np.empty(rounds, dtype=float)
    candidate_samples = np.empty(rounds, dtype=float)
    for index in range(rounds):
        sample = rng.integers(0, len(differences), len(differences))
        samples[index] = float(np.mean(differences[sample]))
        candidate_samples[index] = float(np.mean(candidate_correct[sample]))
    standard_error = float(samples.std(ddof=1))
    ci_lower = float(np.quantile(samples, 0.025))
    ci_upper = float(np.quantile(samples, 0.975))
    return {
        "rounds": int(rounds),
        "difference_candidate_minus_baseline_mean": float(np.mean(differences)),
        "standard_error": standard_error,
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "candidate_ci95_lower": float(np.quantile(candidate_samples, 0.025)),
        "candidate_ci95_upper": float(np.quantile(candidate_samples, 0.975)),
    }


def _valid_https_source(value: Any) -> bool:
    try:
        parsed = urlsplit(str(value or "").strip())
    except ValueError:
        return False
    hostname = str(parsed.hostname or "").casefold()
    return bool(
        parsed.scheme.casefold() == "https"
        and hostname
        and hostname not in {"localhost", "example.com", "example.org", "example.net"}
        and not hostname.endswith((".invalid", ".localhost", ".test"))
    )


def discover_primary_baseline_manifest(data_root: Path, explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        (
            data_root / "managed_runtime" / "cure-bench-primary-baseline.json",
            data_root / ".evomind" / "cure-bench" / "primary-baseline-manifest.json",
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ExactGateError(
        "PRIMARY_BASELINE_MANIFEST_MISSING",
        "no qualified organizer or publicly reproducible strong primary baseline manifest is available",
        "Provide a source-bound primary baseline manifest with a comparable protocol and exact evidence-file SHA closure; the internal TF-IDF reference is not a public strong baseline",
        "primary_baseline_discovery",
    )


def load_primary_baseline_manifest(manifest_path: Path, data_root: Path) -> dict[str, Any]:
    manifest_input = manifest_path.expanduser().absolute()
    if _has_link_or_reparse_ancestor(manifest_input):
        raise ExactGateError(
            "PRIMARY_BASELINE_MANIFEST_UNSAFE",
            "primary baseline manifest traverses a symlink or reparse point",
            "Use a regular manifest and evidence directory below the authorized data root",
            "primary_baseline_validation",
        )
    try:
        resolved = manifest_input.resolve(strict=True)
        authorized_root = data_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ExactGateError(
            "PRIMARY_BASELINE_MANIFEST_MISSING",
            "primary baseline manifest is missing",
            "Restore the exact source-bound primary baseline manifest",
            "primary_baseline_discovery",
        ) from exc
    if not resolved.is_file() or (authorized_root != resolved.parent and authorized_root not in resolved.parents):
        raise ExactGateError(
            "PRIMARY_BASELINE_MANIFEST_OUTSIDE_ROOT",
            "primary baseline manifest is outside the authorized data root",
            "Place the manifest under the fixed CURE-Bench data root",
            "primary_baseline_validation",
        )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExactGateError(
            "PRIMARY_BASELINE_MANIFEST_INVALID",
            "primary baseline manifest is not valid UTF-8 JSON",
            "Repair the source-bound primary baseline manifest",
            "primary_baseline_validation",
        ) from exc
    if not isinstance(value, dict) or value.get("schema") != PRIMARY_BASELINE_SCHEMA:
        raise ExactGateError(
            "PRIMARY_BASELINE_SCHEMA_MISMATCH",
            "primary baseline manifest schema is not the CURE-Bench v1 contract",
            f"Provide {PRIMARY_BASELINE_SCHEMA}",
            "primary_baseline_validation",
        )
    baseline_type = str(value.get("baseline_type") or "")
    if baseline_type not in PRIMARY_BASELINE_TYPES:
        raise ExactGateError(
            "PRIMARY_BASELINE_TYPE_INVALID",
            "internal or self-declared references are not qualified primary baselines",
            "Use an official organizer baseline or a publicly reproducible strong baseline; keep internal TF-IDF labeled as an internal reference",
            "primary_baseline_validation",
        )
    protocol = str(value.get("protocol") or "").strip()
    expected_protocol_sha = hashlib.sha256(protocol.encode("utf-8")).hexdigest()
    score = value.get("value")
    if (
        value.get("competition") != "cure_bench"
        or value.get("status") != "VERIFIED"
        or not _valid_https_source(value.get("source_url"))
        or value.get("source_authority") not in PRIMARY_BASELINE_SOURCE_AUTHORITIES
        or not str(value.get("title") or "").strip()
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}", str(value.get("protocol_id") or ""))
        or len(protocol) < 20
        or value.get("protocol_sha256") != expected_protocol_sha
        or value.get("protocol_comparable") is not True
        or value.get("metric") != PRIMARY_BASELINE_METRIC
        or value.get("direction") != "higher_is_better"
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
    ):
        raise ExactGateError(
            "PRIMARY_BASELINE_SOURCE_OR_PROTOCOL_INVALID",
            "primary baseline source, comparable protocol, metric, direction, or value is invalid",
            "Bind a real authoritative HTTPS source and the exact comparable CURE-Bench accuracy protocol",
            "primary_baseline_validation",
        )
    if baseline_type == "official_organizer":
        if value.get("source_authority") != "official_organizer" or value.get("official_organizer_baseline") is not True:
            raise ExactGateError(
                "PRIMARY_BASELINE_ORGANIZER_STATUS_INVALID",
                "official organizer baseline status is not proven",
                "Provide organizer-authored source evidence and official_organizer_baseline=true",
                "primary_baseline_validation",
            )
        required_roles = PRIMARY_BASELINE_COMMON_ROLES
    else:
        if value.get("publicly_reproducible") is not True:
            raise ExactGateError(
                "PRIMARY_BASELINE_REPRODUCIBILITY_INVALID",
                "public strong baseline does not declare a reproducible source bundle",
                "Provide implementation, data, environment, protocol, and reproduction receipt evidence",
                "primary_baseline_validation",
            )
        required_roles = PRIMARY_BASELINE_STRONG_ROLES
    evidence_dir_value = value.get("evidence_dir")
    if not isinstance(evidence_dir_value, str) or not evidence_dir_value or Path(evidence_dir_value).is_absolute() or ".." in Path(evidence_dir_value).parts:
        raise ExactGateError(
            "PRIMARY_BASELINE_EVIDENCE_PATH_INVALID",
            "primary baseline evidence_dir must be a safe relative path",
            "Place the closed evidence bundle below the manifest directory",
            "primary_baseline_validation",
        )
    evidence_input = resolved.parent / evidence_dir_value
    if _has_link_or_reparse_ancestor(evidence_input):
        raise ExactGateError(
            "PRIMARY_BASELINE_EVIDENCE_UNSAFE",
            "primary baseline evidence bundle traverses a symlink or reparse point",
            "Materialize a regular-file-only evidence bundle",
            "primary_baseline_validation",
        )
    try:
        evidence_root = evidence_input.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ExactGateError(
            "PRIMARY_BASELINE_EVIDENCE_MISSING",
            "primary baseline evidence directory is missing",
            "Restore the exact source-bound evidence bundle",
            "primary_baseline_validation",
        ) from exc
    if not evidence_root.is_dir() or resolved.parent not in evidence_root.parents:
        raise ExactGateError(
            "PRIMARY_BASELINE_EVIDENCE_OUTSIDE_ROOT",
            "primary baseline evidence bundle escapes the manifest directory",
            "Use a relative evidence directory next to the manifest",
            "primary_baseline_validation",
        )
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ExactGateError(
            "PRIMARY_BASELINE_FILE_MANIFEST_MISSING",
            "primary baseline evidence file manifest is empty",
            "List every regular evidence file with role, bytes, and SHA-256",
            "primary_baseline_validation",
        )
    seen_paths: set[str] = set()
    seen_roles: set[str] = set()
    verified: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise ExactGateError("PRIMARY_BASELINE_FILE_ENTRY_INVALID", "primary baseline file entry is malformed", "Repair the evidence file manifest", "primary_baseline_validation")
        relative = Path(str(entry.get("path") or ""))
        role = str(entry.get("role") or "")
        expected_sha = str(entry.get("sha256") or "").lower()
        expected_bytes = entry.get("bytes")
        key = relative.as_posix()
        if relative.is_absolute() or not key or ".." in relative.parts or key in seen_paths or not role or role in seen_roles:
            raise ExactGateError("PRIMARY_BASELINE_FILE_ENTRY_INVALID", "primary baseline file path or role is unsafe or duplicated", "Use unique relative paths and roles", "primary_baseline_validation")
        seen_paths.add(key)
        seen_roles.add(role)
        path_input = evidence_root / relative
        if _has_link_or_reparse_ancestor(path_input):
            raise ExactGateError("PRIMARY_BASELINE_FILE_UNSAFE", "primary baseline evidence contains a link or reparse entry", "Use regular files only", "primary_baseline_validation")
        path = path_input.resolve(strict=True)
        if not path.is_file() or evidence_root not in path.parents:
            raise ExactGateError("PRIMARY_BASELINE_FILE_UNSAFE", "primary baseline evidence file escapes the bundle", "Use regular files inside evidence_dir", "primary_baseline_validation")
        actual_sha = sha256_file(path)
        actual_bytes = path.stat().st_size
        if len(expected_sha) != 64 or expected_sha != actual_sha or isinstance(expected_bytes, bool) or expected_bytes != actual_bytes:
            raise ExactGateError("PRIMARY_BASELINE_FILE_HASH_MISMATCH", "primary baseline evidence bytes or SHA-256 do not match", "Restore the exact evidence bundle", "primary_baseline_validation")
        verified.append({"role": role, "path": key, "bytes": actual_bytes, "sha256": actual_sha})
    if not required_roles <= seen_roles:
        raise ExactGateError(
            "PRIMARY_BASELINE_FILE_ROLES_MISSING",
            "primary baseline evidence bundle is missing required source or reproduction roles",
            "Add the exact source, protocol, implementation, data manifest, environment, and reproduction receipt as applicable",
            "primary_baseline_validation",
        )
    if any(_is_link_or_reparse(path) for path in evidence_root.rglob("*")):
        raise ExactGateError(
            "PRIMARY_BASELINE_EVIDENCE_UNSAFE",
            "primary baseline evidence bundle contains a symlink or reparse entry",
            "Materialize a regular-file-only evidence bundle",
            "primary_baseline_validation",
        )
    actual_files = {
        path.relative_to(evidence_root).as_posix()
        for path in evidence_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_files != seen_paths:
        raise ExactGateError(
            "PRIMARY_BASELINE_FILE_CLOSURE_MISMATCH",
            "primary baseline manifest has missing or extra evidence files",
            "Regenerate the exact regular-file evidence closure",
            "primary_baseline_validation",
        )
    return {
        **value,
        "manifest_sha256": sha256_file(resolved),
        "manifest_path": str(resolved),
        "verified_files": verified,
    }


def _validate_encoder_manifest(manifest_path: Path, data_root: Path) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    manifest_input = manifest_path.expanduser().absolute()
    if _has_link_or_reparse_ancestor(manifest_input):
        raise ExactGateError(
            "PRETRAINED_ENCODER_MANIFEST_UNSAFE",
            "encoder manifest must not be a symlink or reparse point",
            "Use a regular manifest inside the managed model root",
            "encoder_discovery",
        )
    try:
        manifest_resolved = manifest_input.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ExactGateError(
            "PRETRAINED_ENCODER_MANIFEST_MISSING",
            "no SHA-bound frozen encoder manifest is available",
            "Provision an existing managed-runtime encoder manifest; do not download a model",
            "encoder_discovery",
        ) from exc
    if not manifest_resolved.is_file():
        raise ExactGateError(
            "PRETRAINED_ENCODER_MANIFEST_UNSAFE",
            "encoder manifest must be a regular non-symlink file",
            "Use a regular manifest inside the managed model root",
            "encoder_discovery",
        )
    managed_root_env = os.environ.get("EVOMIND_MANAGED_MODEL_ROOT", "").strip()
    allowed_roots = [data_root.resolve(strict=True)]
    if managed_root_env:
        managed_root = Path(managed_root_env).expanduser()
        if not managed_root.is_absolute():
            raise ExactGateError(
                "PRETRAINED_ENCODER_ROOT_INVALID",
                "EVOMIND_MANAGED_MODEL_ROOT must be absolute",
                "Bind the encoder to the managed runtime model root",
                "encoder_discovery",
            )
        allowed_roots.append(managed_root.resolve(strict=True))
    if not any(root == manifest_resolved.parent or root in manifest_resolved.parents for root in allowed_roots):
        raise ExactGateError(
            "PRETRAINED_ENCODER_OUTSIDE_MANAGED_ROOT",
            "encoder manifest is outside the authorized data/model roots",
            "Use a manifest under data-root or EVOMIND_MANAGED_MODEL_ROOT",
            "encoder_discovery",
        )
    try:
        manifest = json.loads(manifest_resolved.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExactGateError(
            "PRETRAINED_ENCODER_MANIFEST_INVALID",
            "encoder manifest is not valid UTF-8 JSON",
            "Repair the managed encoder manifest",
            "encoder_validation",
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != "evomind.cure_bench.frozen_encoder_manifest.v1":
        raise ExactGateError(
            "PRETRAINED_ENCODER_SCHEMA_MISMATCH",
            "encoder manifest schema is not the frozen CURE-Bench v1 contract",
            "Create a managed-runtime frozen_encoder_manifest.v1",
            "encoder_validation",
        )
    if manifest.get("backend") != "transformers" or manifest.get("frozen") is not True or manifest.get("offline_only") is not True:
        raise ExactGateError(
            "PRETRAINED_ENCODER_NOT_FROZEN_OFFLINE",
            "encoder must be a frozen transformers snapshot with offline_only=true",
            "Bind an existing frozen local Transformer snapshot; no online model resolution is allowed",
            "encoder_validation",
        )
    if manifest.get("trust_remote_code") is True or manifest.get("remote_code") is True:
        raise ExactGateError(
            "PRETRAINED_ENCODER_REMOTE_CODE",
            "remote code execution is forbidden for the encoder",
            "Use a standard local Transformers model with trust_remote_code=false",
            "encoder_validation",
        )
    model_dir_value = manifest.get("model_dir")
    if not isinstance(model_dir_value, str) or not model_dir_value or Path(model_dir_value).is_absolute():
        raise ExactGateError(
            "PRETRAINED_ENCODER_MODEL_PATH_INVALID",
            "model_dir must be a relative path in the encoder manifest",
            "Repair model_dir to point inside the manifest directory",
            "encoder_validation",
        )
    model_dir_input = manifest_resolved.parent / model_dir_value
    if _has_link_or_reparse_ancestor(model_dir_input):
        raise ExactGateError(
            "PRETRAINED_ENCODER_MODEL_UNSAFE",
            "model snapshot directory must not be a symlink or reparse point",
            "Use a regular directory in the managed model root",
            "encoder_validation",
        )
    model_dir = model_dir_input.resolve(strict=True)
    if not any(root == model_dir or root in model_dir.parents for root in allowed_roots):
        raise ExactGateError(
            "PRETRAINED_ENCODER_MODEL_OUTSIDE_ROOT",
            "model snapshot escapes the authorized managed model root",
            "Use a local model snapshot below the managed model root",
            "encoder_validation",
        )
    if not model_dir.is_dir():
        raise ExactGateError(
            "PRETRAINED_ENCODER_MODEL_MISSING",
            "declared model snapshot is not a regular directory",
            "Provision an existing local model snapshot; do not download during a run",
            "encoder_discovery",
        )
    hidden_size = manifest.get("hidden_size")
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ExactGateError(
            "PRETRAINED_ENCODER_HIDDEN_SIZE_INVALID",
            "encoder manifest hidden_size must be a positive integer",
            "Record the exact hidden dimension of the frozen local snapshot",
            "encoder_validation",
        )
    max_length = manifest.get("max_length", 512)
    if isinstance(max_length, bool) or not isinstance(max_length, int) or not 1 <= max_length <= 4096:
        raise ExactGateError(
            "PRETRAINED_ENCODER_MAX_LENGTH_INVALID",
            "encoder manifest max_length must be an integer from 1 through 4096",
            "Record the bounded tokenizer sequence length",
            "encoder_validation",
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ExactGateError(
            "PRETRAINED_ENCODER_FILE_MANIFEST_MISSING",
            "encoder manifest does not contain file hashes",
            "Add regular-file SHA-256 entries for the local model snapshot",
            "encoder_validation",
        )
    verified_files: list[dict[str, Any]] = []
    seen_relative: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise ExactGateError(
                "PRETRAINED_ENCODER_FILE_ENTRY_INVALID",
                "encoder file manifest entry is malformed",
                "Repair the local encoder file manifest",
                "encoder_validation",
            )
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ExactGateError(
                "PRETRAINED_ENCODER_FILE_PATH_INVALID",
                "encoder file entry escapes model_dir",
                "Use relative regular-file paths only",
                "encoder_validation",
            )
        relative_key = relative.as_posix()
        if relative_key in seen_relative:
            raise ExactGateError(
                "PRETRAINED_ENCODER_FILE_DUPLICATE",
                "encoder file manifest contains duplicate paths",
                "List each frozen snapshot file exactly once",
                "encoder_validation",
            )
        seen_relative.add(relative_key)
        file_input = model_dir / relative
        if _has_link_or_reparse_ancestor(file_input):
            raise ExactGateError(
                "PRETRAINED_ENCODER_FILE_UNSAFE",
                "encoder file entry is symlinked or reparse-backed",
                "Use regular files in the frozen local snapshot",
                "encoder_validation",
            )
        file_path = file_input.resolve(strict=True)
        if not file_path.is_file() or model_dir not in file_path.parents:
            raise ExactGateError(
                "PRETRAINED_ENCODER_FILE_UNSAFE",
                "encoder file entry is missing, symlinked, or outside model_dir",
                "Use a regular file in the frozen local snapshot",
                "encoder_validation",
            )
        expected = item["sha256"].lower()
        actual = sha256_file(file_path)
        if len(expected) != 64 or actual != expected:
            raise ExactGateError(
                "PRETRAINED_ENCODER_FILE_HASH_MISMATCH",
                "encoder snapshot file hash does not match the frozen manifest",
                "Restore the exact managed-runtime encoder snapshot",
                "encoder_validation",
            )
        verified_files.append({"path": relative_key, "bytes": file_path.stat().st_size, "sha256": actual})
    reparse_entries = [path for path in model_dir.rglob("*") if _is_link_or_reparse(path)]
    if reparse_entries:
        raise ExactGateError(
            "PRETRAINED_ENCODER_REPARSE_ENTRY",
            "encoder snapshot contains a symlink or reparse entry",
            "Materialize a regular-file-only frozen snapshot",
            "encoder_validation",
        )
    actual_regular = {
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_regular != seen_relative:
        raise ExactGateError(
            "PRETRAINED_ENCODER_FILE_CLOSURE_MISMATCH",
            "encoder manifest does not exactly cover the regular files in the snapshot",
            "Regenerate the frozen encoder file manifest without extra or missing files",
            "encoder_validation",
        )
    return manifest, model_dir, verified_files


class FrozenTransformerEncoder:
    def __init__(self, manifest_path: Path, data_root: Path) -> None:
        manifest, model_dir, verified_files = _validate_encoder_manifest(manifest_path, data_root)
        self.manifest_info = {
            "schema": manifest["schema"],
            "backend": manifest["backend"],
            "model_id": str(manifest.get("model_id", "")),
            "revision": str(manifest.get("revision", "")),
            "pooling": str(manifest.get("pooling", "mean_attention_mask")),
            "max_length": int(manifest.get("max_length", 512)),
            "hidden_size": int(manifest.get("hidden_size", 0)),
            "device": str(manifest.get("device", "auto")),
            "manifest_sha256": sha256_file(manifest_path.resolve(strict=True)),
            "model_dir_relative": str(manifest.get("model_dir")),
            "files": verified_files,
            "frozen": True,
            "offline_only": True,
        }
        self._model_dir = model_dir
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ExactGateError(
                "PRETRAINED_ENCODER_DEPENDENCY_MISSING",
                "managed runtime lacks torch/transformers for the frozen encoder",
                "Use a managed runtime that already contains the declared dependencies; do not install online",
                "encoder_load",
            ) from exc
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_dir,
                local_files_only=True,
                trust_remote_code=False,
                use_fast=True,
            )
            model = AutoModel.from_pretrained(
                self._model_dir,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise ExactGateError(
                "PRETRAINED_ENCODER_LOAD_FAILED",
                "the frozen local Transformer snapshot could not be loaded offline",
                "Restore the exact local snapshot and compatible managed dependencies",
                "encoder_load",
            ) from exc
        requested_device = str(self.manifest_info.get("device", "auto")).strip().lower()
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise ExactGateError(
                "PRETRAINED_ENCODER_DEVICE_INVALID",
                "encoder manifest device must be auto, cpu, or cuda",
                "Repair the managed encoder manifest device policy",
                "encoder_load",
            )
        if requested_device == "cuda" or (requested_device == "auto" and torch.cuda.is_available()):
            if not torch.cuda.is_available():
                raise ExactGateError(
                    "PRETRAINED_ENCODER_GPU_UNAVAILABLE",
                    "encoder manifest requires CUDA but no managed CUDA device is available",
                    "Use the bound managed GPU runtime or declare a CPU-compatible frozen snapshot",
                    "encoder_load",
                )
            model.to(torch.device("cuda:0"))
        else:
            model.to(torch.device("cpu"))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._tokenizer = tokenizer
        self._model = model
        self._device = next(model.parameters()).device

    def encode(self, texts: Sequence[str], *, batch_size: int = 16) -> np.ndarray:
        self._load()
        import torch

        max_length = int(self.manifest_info["max_length"])
        vectors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), max(1, int(batch_size))):
                batch = list(texts[start : start + max(1, int(batch_size))])
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                output = self._model(**encoded)
                hidden = getattr(output, "last_hidden_state", None)
                if hidden is None:
                    raise ExactGateError(
                        "PRETRAINED_ENCODER_OUTPUT_INVALID",
                        "frozen encoder did not return last_hidden_state",
                        "Use an AutoModel-compatible encoder snapshot",
                        "encoder_encode",
                    )
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.detach().float().cpu().numpy())
        if not vectors:
            raise ExactGateError("EMPTY_ENCODER_INPUT", "encoder received no rows", "Provide scored validation rows", "encoder_encode")
        result = np.concatenate(vectors, axis=0).astype(np.float32, copy=False)
        expected_size = int(self.manifest_info.get("hidden_size") or result.shape[1])
        if result.shape[1] != expected_size or not np.all(np.isfinite(result)):
            raise ExactGateError(
                "PRETRAINED_ENCODER_DIMENSION_INVALID",
                "frozen encoder output dimension or values do not match the manifest",
                "Restore the exact encoder snapshot and manifest",
                "encoder_encode",
            )
        return result


def discover_encoder_manifest(data_root: Path, explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_value = os.environ.get("EVOMIND_CURE_ENCODER_MANIFEST", "").strip()
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(
        [
            data_root / ".evomind" / "cure-bench" / "frozen-encoder.json",
            data_root / "managed_runtime" / "frozen-encoder.json",
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and not _is_link_or_reparse(candidate):
            return candidate
    raise ExactGateError(
        "PRETRAINED_ENCODER_MANIFEST_MISSING",
        "no existing managed-runtime frozen encoder manifest was found",
        "Bind one existing local Transformer snapshot; candidate-2 never downloads a model",
        "encoder_discovery",
    )


def discover_holdout_ledger(data_root: Path, explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_value = os.environ.get("EVOMIND_CURE_HOLDOUT_LEDGER", "").strip()
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(data_root / ".evomind" / "cure-bench" / "holdout-ledger.json")
    for candidate in candidates:
        if candidate.is_file() and not _has_link_or_reparse_ancestor(candidate):
            return candidate
    raise ExactGateError(
        "HOLDOUT_LEDGER_MISSING",
        "no managed ledger proving that the fresh holdout is available and unconsumed",
        "Provision a SHA-bound holdout ledger; do not reuse an evaluated holdout",
        "holdout_ledger",
    )


def load_holdout_ledger(path: Path) -> HoldoutLedger:
    if _has_link_or_reparse_ancestor(path):
        raise ExactGateError(
            "HOLDOUT_LEDGER_UNSAFE",
            "holdout ledger path traverses a symlink or reparse point",
            "Use a regular managed evaluator ledger path",
            "holdout_ledger",
        )
    try:
        resolved_path = path.resolve(strict=True)
        payload = json.loads(resolved_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExactGateError(
            "HOLDOUT_LEDGER_INVALID",
            "holdout ledger is not valid UTF-8 JSON",
            "Repair the managed holdout ledger",
            "holdout_ledger",
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != "evomind.cure_bench.holdout_ledger.v1":
        raise ExactGateError(
            "HOLDOUT_LEDGER_SCHEMA_MISMATCH",
            "holdout ledger schema is not v1",
            "Provide evomind.cure_bench.holdout_ledger.v1",
            "holdout_ledger",
        )
    consumed = payload.get("consumed_holdout_id_sha256", [])
    available = payload.get("available_holdout_id_sha256", [])
    if not isinstance(consumed, list) or not isinstance(available, list):
        raise ExactGateError(
            "HOLDOUT_LEDGER_ENTRIES_INVALID",
            "holdout ledger hash lists are malformed",
            "Repair the holdout ledger hash lists",
            "holdout_ledger",
        )
    hashes = [str(item).lower() for item in [*consumed, *available]]
    if any(len(item) != 64 or any(char not in "0123456789abcdef" for char in item) for item in hashes):
        raise ExactGateError(
            "HOLDOUT_LEDGER_HASH_INVALID",
            "holdout ledger contains a non-SHA-256 hash",
            "Record canonical SHA-256 holdout identities",
            "holdout_ledger",
        )
    return HoldoutLedger(
        path=str(resolved_path),
        sha256=sha256_file(resolved_path),
        schema=str(payload["schema"]),
        consumed_hashes=frozenset(str(item).lower() for item in consumed),
        available_hashes=frozenset(str(item).lower() for item in available),
    )


def _fit_baseline(texts: Sequence[str], labels: np.ndarray, train_indices: np.ndarray) -> tuple[Any, Any]:
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=30_000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([texts[int(index)] for index in train_indices])
    model = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced", random_state=SEED)
    model.fit(matrix, labels[train_indices])
    return vectorizer, model


def _fit_candidate(matrix: np.ndarray, labels: np.ndarray, train_indices: np.ndarray, c_value: float) -> Any:
    model = LinearSVC(C=float(c_value), class_weight="balanced", random_state=SEED)
    model.fit(matrix[train_indices], labels[train_indices])
    return model


def run_experiment(
    data_root: Path,
    output_root: Path,
    *,
    encoder: EncoderProtocol,
    source_path: Path | None = None,
    allow_test_encoder: bool = False,
    holdout_ledger: HoldoutLedger | None = None,
    primary_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = utc_now()
    data_input = data_root.expanduser().absolute()
    if _has_link_or_reparse_ancestor(data_input):
        raise ExactGateError(
            "DATA_ROOT_UNSAFE",
            "data-root traverses a symlink or reparse point",
            "Use a regular managed CURE-Bench data root",
            "dataset_discovery",
        )
    data_root = data_input.resolve()
    output_input = output_root.expanduser().absolute()
    if _has_link_or_reparse_ancestor(output_input):
        raise ExactGateError(
            "OUTPUT_DIRECTORY_UNSAFE",
            "candidate-2 output path traverses a symlink or reparse point",
            "Use a regular managed solution output directory",
            "artifact_materialization",
        )
    output_root = output_input.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ExactGateError(
            "OUTPUT_DIRECTORY_NOT_EMPTY",
            "candidate-2 output directory already contains files",
            "Use a new empty solution output directory",
            "artifact_materialization",
        )
    data = load_dataset(data_root)
    rows = data.validation
    splits = make_fresh_splits(rows)
    if holdout_ledger is None and not allow_test_encoder:
        raise ExactGateError(
            "HOLDOUT_LEDGER_MISSING",
            "candidate-2 has no managed ledger proving this holdout is unconsumed",
            "Bind the fresh holdout to a managed evaluator ledger before execution",
            "holdout_ledger",
        )
    if holdout_ledger is not None:
        if splits.holdout_id_sha256 in holdout_ledger.consumed_hashes:
            raise ExactGateError(
                "HOLDOUT_ALREADY_CONSUMED",
                "candidate-2 fresh holdout is already marked consumed",
                "Allocate a different untouched holdout; do not rerun this split",
                "holdout_ledger",
            )
        if holdout_ledger.available_hashes and splits.holdout_id_sha256 not in holdout_ledger.available_hashes:
            raise ExactGateError(
                "HOLDOUT_LEDGER_BINDING_MISMATCH",
                "managed holdout ledger does not authorize this split identity",
                "Register the exact candidate-2 holdout hash in the evaluator ledger",
                "holdout_ledger",
            )
    if primary_baseline is None and not allow_test_encoder:
        raise ExactGateError(
            "PRIMARY_BASELINE_MANIFEST_MISSING",
            "candidate-2 has no qualified organizer or publicly reproducible strong primary baseline",
            "Bind a validated primary baseline manifest; the internal TF-IDF model remains a local reference only",
            "primary_baseline_discovery",
        )
    texts = [row_text(row) for row in rows]
    labels = np.asarray([int(row.label) if row.label is not None else -1 for row in rows], dtype=np.int64)
    if any(labels[index] < 0 for index in splits.holdout):
        raise ExactGateError("HOLDOUT_LABEL_MISSING", "fresh holdout contains an unscored row", "Regenerate the choice-only split", "split_creation")

    encoder_info = dict(getattr(encoder, "manifest_info", {}))
    manifest_sha = str(encoder_info.get("manifest_sha256") or "").lower()
    sha_valid = len(manifest_sha) == 64 and all(character in "0123456789abcdef" for character in manifest_sha)
    expected_backend = "test-fixture" if allow_test_encoder else "transformers"
    if (
        encoder_info.get("schema") != "evomind.cure_bench.frozen_encoder_manifest.v1"
        or encoder_info.get("backend") != expected_backend
        or encoder_info.get("frozen") is not True
        or encoder_info.get("offline_only") is not True
        or not sha_valid
    ):
        raise ExactGateError(
            "ENCODER_PROVENANCE_INVALID",
            "encoder provenance is not a frozen, offline, SHA-bound managed Transformer contract",
            "Bind the exact managed frozen encoder manifest before training",
            "encoder_provenance",
        )
    # The encoder is loaded once and never fit or updated.  Encoding all rows
    # is label-free; head/hyperparameter selection remains inner-only.
    embedding_matrix = np.asarray(encoder.encode(texts), dtype=np.float32)
    if embedding_matrix.ndim != 2 or embedding_matrix.shape[0] != len(rows):
        raise ExactGateError("ENCODER_ROW_COUNT_MISMATCH", "frozen encoder returned the wrong row count", "Restore the encoder adapter", "encoder_encode")
    if not np.all(np.isfinite(embedding_matrix)):
        raise ExactGateError("ENCODER_NONFINITE", "frozen encoder returned non-finite values", "Restore the encoder snapshot", "encoder_encode")

    baseline_inner_vectorizer, baseline_inner = _fit_baseline(texts, labels, splits.inner_train)
    baseline_inner_matrix = baseline_inner_vectorizer.transform([texts[int(index)] for index in splits.inner_validation])
    baseline_inner_prediction = predict_valid_options(baseline_inner, baseline_inner_matrix, rows, splits.inner_validation)
    baseline_inner_score = accuracy(labels[splits.inner_validation], baseline_inner_prediction)
    baseline_vectorizer, baseline = _fit_baseline(texts, labels, splits.development)

    candidate_history: list[dict[str, Any]] = []
    for c_value in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        candidate_inner = _fit_candidate(embedding_matrix, labels, splits.inner_train, c_value)
        prediction = predict_valid_options(
            candidate_inner,
            embedding_matrix[splits.inner_validation],
            rows,
            splits.inner_validation,
        )
        score = accuracy(labels[splits.inner_validation], prediction)
        candidate_history.append({"candidate_id": f"frozen-transformer-linearsvc-{c_value}", "C": c_value, "inner_accuracy": score})
    selected = max(candidate_history, key=lambda item: (item["inner_accuracy"], -float(item["C"]), item["candidate_id"]))
    candidate = _fit_candidate(embedding_matrix, labels, splits.development, float(selected["C"]))

    baseline_prediction = predict_valid_options(
        baseline,
        baseline_vectorizer.transform([texts[int(index)] for index in splits.holdout]),
        rows,
        splits.holdout,
    )
    candidate_prediction = predict_valid_options(candidate, embedding_matrix[splits.holdout], rows, splits.holdout)
    y_holdout = labels[splits.holdout]
    baseline_score = accuracy(y_holdout, baseline_prediction)
    candidate_score = accuracy(y_holdout, candidate_prediction)
    bootstrap = paired_bootstrap(y_holdout, baseline_prediction, candidate_prediction)
    minimum_margin = max(0.01 * max(baseline_score, 1e-12), float(bootstrap["standard_error"]))
    internal_reference_gate_passed = bool(
        candidate_score > baseline_score and float(bootstrap["ci95_lower"]) > minimum_margin
    )
    primary_value = float(primary_baseline["value"]) if primary_baseline is not None else None
    primary_gate_passed = bool(
        primary_value is not None
        and candidate_score > primary_value
        and float(bootstrap["candidate_ci95_lower"]) > primary_value
    )
    gate_passed = bool(
        internal_reference_gate_passed
        and (primary_gate_passed if primary_baseline is not None else allow_test_encoder)
    )
    production_goal_eligible = bool(primary_baseline is not None and gate_passed)

    source = (source_path or Path(__file__)).resolve(strict=True)
    source_sha = sha256_file(source)
    holdout_predictions = [
        {
            "id": rows[int(index)].row_id,
            "question_type": rows[int(index)].question_type,
            "option_count": rows[int(index)].option_count,
            "y_true": int(labels[int(index)]),
            "baseline_prediction": int(baseline_prediction[position]),
            "candidate_prediction": int(candidate_prediction[position]),
            "baseline_correct": bool(labels[int(index)] == baseline_prediction[position]),
            "candidate_correct": bool(labels[int(index)] == candidate_prediction[position]),
        }
        for position, index in enumerate(splits.holdout)
    ]
    recomputed_baseline = accuracy(
        np.asarray([row["y_true"] for row in holdout_predictions], dtype=np.int64),
        np.asarray([row["baseline_prediction"] for row in holdout_predictions], dtype=np.int64),
    )
    recomputed_candidate = accuracy(
        np.asarray([row["y_true"] for row in holdout_predictions], dtype=np.int64),
        np.asarray([row["candidate_prediction"] for row in holdout_predictions], dtype=np.int64),
    )
    independent = {
        "schema": "evomind.cure_bench.independent_verification.v3",
        "holdout_rows": len(holdout_predictions),
        "baseline_recomputed": recomputed_baseline,
        "candidate_recomputed": recomputed_candidate,
        "baseline_match": abs(recomputed_baseline - baseline_score) < 1e-12,
        "candidate_match": abs(recomputed_candidate - candidate_score) < 1e-12,
        "id_overlap_with_prior_candidate": 0,
        "test_labels_used": False,
        "gate_passed": gate_passed,
        "internal_reference_gate_passed": internal_reference_gate_passed,
        "primary_baseline_gate_passed": primary_gate_passed,
        "production_goal_eligible": production_goal_eligible,
    }
    output_root.mkdir(parents=True, exist_ok=True)

    task_contract = {
        "schema": "evomind.cure_bench.task_contract.v3",
        "task": "multiple-choice medical question answering",
        "metric": "accuracy on scored choice rows",
        "direction": "higher_is_better",
        "entity": "question id",
        "candidate_id": "cure-bench-v2-frozen-language-candidate-2",
        "representation": "frozen local Transformer mean attention-mask pooling",
        "head": "class-balanced LinearSVC with C selected on inner validation",
        "open_ended_multi_choice_protocol": "question-only representation; options withheld from first pass",
        "test_labels_used": False,
        "network_access": False,
    }
    baseline_evidence = {
        "schema": "evomind.cure_bench.baseline_evidence.v3",
        "classification": "INTERNAL_REPRODUCIBLE_REFERENCE",
        "method": "word 1-2 gram TF-IDF plus class-balanced LogisticRegression (local reference; no official score claim)",
        "official_reference": None,
        "implementation_sha256": source_sha,
        "validation_sha256": data.validation_sha256,
        "reproduction_accuracy": baseline_score,
        "eligible_as_primary_baseline": False,
    }
    primary_baseline_evidence = (
        {
            **dict(primary_baseline),
            "candidate_accuracy": candidate_score,
            "candidate_ci95_lower": bootstrap["candidate_ci95_lower"],
            "primary_gate_passed": primary_gate_passed,
            "production_goal_eligible": production_goal_eligible,
        }
        if primary_baseline is not None
        else {
            "schema": "evomind.cure_bench.primary_baseline_unavailable.v1",
            "status": "TEST_FIXTURE_ONLY",
            "internal_reference_is_primary": False,
            "production_goal_eligible": False,
            "recovery_gate": "PRIMARY_BASELINE_MANIFEST_MISSING",
        }
    )
    split_manifest = {
        "schema": "evomind.cure_bench.split_manifest.v3",
        "seed": SEED,
        "prior_seed": PRIOR_SEED,
        "development_rows": int(len(splits.development)),
        "holdout_rows": int(len(splits.holdout)),
        "inner_train_rows": int(len(splits.inner_train)),
        "inner_validation_rows": int(len(splits.inner_validation)),
        "prior_holdout_rows": int(len(splits.prior_holdout)),
        "id_overlap_with_prior_candidate": 0,
        "prior_holdout_id_sha256": splits.prior_holdout_id_sha256,
        "holdout_id_sha256": splits.holdout_id_sha256,
        "prior_split_source": "candidate1_legacy_seed_reconstruction",
        "holdout_blind_for_selection": True,
        "previous_holdout_excluded": True,
        "previous_holdout_promoted_to_development": False,
    }
    metrics = {
        "schema": "evomind.cure_bench.metrics.v3",
        "baseline_accuracy": baseline_score,
        "candidate_accuracy": candidate_score,
        "absolute_improvement": candidate_score - baseline_score,
        "relative_improvement": (candidate_score - baseline_score) / max(baseline_score, 1e-12),
        "baseline_inner_accuracy": baseline_inner_score,
        "candidate_history": candidate_history,
        "bootstrap": bootstrap,
        "minimum_margin": minimum_margin,
        "test_labels_used": False,
        "holdout_ledger_sha256": holdout_ledger.sha256 if holdout_ledger is not None else None,
        "primary_baseline_manifest_sha256": (
            str(primary_baseline.get("manifest_sha256") or "") if primary_baseline is not None else None
        ),
        "internal_reference_gate_passed": internal_reference_gate_passed,
        "primary_baseline_gate_passed": primary_gate_passed,
        "production_goal_eligible": production_goal_eligible,
    }
    comparison = {
        "schema": "evomind.cure_bench.candidate_vs_baseline.v3",
        "baseline": baseline_score,
        "candidate": candidate_score,
        "minimum_margin": minimum_margin,
        "difference_candidate_minus_baseline_ci95_lower": bootstrap["ci95_lower"],
        "gate_passed": gate_passed,
        "primary_baseline_value": primary_value,
        "candidate_ci95_lower": bootstrap["candidate_ci95_lower"],
        "primary_baseline_gate_passed": primary_gate_passed,
        "production_goal_eligible": production_goal_eligible,
    }

    atomic_copyfile(source, output_root / "solution.py")
    write_json(output_root / "task-contract-v3.json", task_contract)
    write_json(output_root / "baseline-evidence.json", baseline_evidence)
    write_json(output_root / "baseline-reproduction.json", {**baseline_evidence, "completed_at_utc": utc_now()})
    write_json(output_root / "primary-baseline-evidence.json", primary_baseline_evidence)
    write_json(output_root / "dataset-audit.json", {**dict(data.answer_schema_audit), "schema": "evomind.cure_bench.dataset_audit.v3", "validation_sha256": data.validation_sha256, "test_schema_rows": data.test_schema_rows, "test_file_sha256": dict(data.test_file_sha256), "test_id_sha256": _id_hash(data.test_ids), "test_labels_used": False})
    write_json(output_root / "answer-schema-audit.json", dict(data.answer_schema_audit))
    if holdout_ledger is not None:
        split_manifest["holdout_ledger_sha256"] = holdout_ledger.sha256
        split_manifest["holdout_ledger_path"] = holdout_ledger.path
        split_manifest["holdout_consumption_contract"] = "managed_ledger_claim_once"
    write_json(output_root / "split-manifest.json", split_manifest)
    write_json(output_root / "frozen-encoder-provenance.json", encoder_info)
    write_json(output_root / "training-config.json", {"schema": "evomind.cure_bench.training_config.v3", "seed": SEED, "baseline": baseline_evidence["method"], "baseline_classification": baseline_evidence["classification"], "internal_reference_is_primary": False, "primary_baseline_manifest_sha256": primary_baseline_evidence.get("manifest_sha256"), "candidate": task_contract["representation"] + " + " + task_contract["head"], "selected_candidate": selected, "bootstrap_rounds": BOOTSTRAP_ROUNDS, "encoder_manifest_sha256": encoder_info["manifest_sha256"]})
    write_jsonl(output_root / "candidate-history.jsonl", candidate_history)
    write_json(output_root / "holdout-predictions.json", holdout_predictions)
    write_json(output_root / "metrics.json", metrics)
    write_json(output_root / "candidate-vs-baseline.json", comparison)
    write_json(output_root / "independent-verification.json", independent)
    write_json(output_root / "environment-lock.json", {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__, "encoder_manifest_sha256": encoder_info["manifest_sha256"]})
    write_jsonl(
        output_root / "training.log",
        [
            {"at_utc": started, "event": "start", "source_sha256": source_sha, "test_labels_used": False},
            {"at_utc": utc_now(), "event": "frozen_encoder_loaded", "encoder_manifest_sha256": encoder_info["manifest_sha256"]},
            {"at_utc": utc_now(), "event": "baseline_complete", "accuracy": baseline_score},
            {"at_utc": utc_now(), "event": "candidate_complete", "accuracy": candidate_score},
            {"at_utc": utc_now(), "event": "independent_verification", "gate_passed": gate_passed},
        ],
    )
    atomic_joblib_dump({"baseline_vectorizer": baseline_vectorizer, "baseline": baseline, "candidate": candidate, "representation": "frozen_transformer_mean_attention_mask", "encoder_manifest_sha256": encoder_info["manifest_sha256"]}, output_root / "model-bundle.joblib")
    write_json(output_root / "retrospective-memory.json", {"schema": "evomind.retrospective_memory.v3", "parent_failure_corrected": ["ambiguous_numeric_answer_encoding", "test_label_recursion", "fresh_holdout_overlap"], "candidate": task_contract["representation"], "gate_passed": gate_passed, "test_labels_used": False, "memory_writeback_allowed": gate_passed, "next_step_if_failed": "Use a new external blind evaluator or a separately authorized fresh holdout; do not reuse this holdout."})

    files: list[dict[str, Any]] = []
    for path in sorted(output_root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name not in {"artifact-manifest.json", "artifact-manifest-receipt.json"}:
            files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"schema": "evomind.cure_bench.artifact_manifest.v3", "candidate_id": task_contract["candidate_id"], "started_at_utc": started, "completed_at_utc": utc_now(), "source_sha256": source_sha, "encoder_manifest_sha256": encoder_info["manifest_sha256"], "holdout_ledger_sha256": holdout_ledger.sha256 if holdout_ledger is not None else None, "exit_code": 0, "gate_passed": gate_passed, "test_labels_used": False, "files": files}
    write_json(output_root / "artifact-manifest.json", manifest)
    write_json(output_root / "artifact-manifest-receipt.json", {"schema": "evomind.cure_bench.artifact_manifest_receipt.v3", "artifact": "artifact-manifest.json", "bytes": (output_root / "artifact-manifest.json").stat().st_size, "sha256": sha256_file(output_root / "artifact-manifest.json")})
    return {"status": "completed", **comparison, "source_sha256": source_sha, "encoder_manifest_sha256": encoder_info["manifest_sha256"], "holdout_ledger_sha256": holdout_ledger.sha256 if holdout_ledger is not None else None, "primary_baseline_manifest_sha256": primary_baseline_evidence.get("manifest_sha256"), "production_goal_eligible": production_goal_eligible, "holdout_rows": len(splits.holdout), "id_overlap_with_prior_candidate": 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--encoder-manifest")
    parser.add_argument("--holdout-ledger")
    parser.add_argument("--primary-baseline-manifest")
    args = parser.parse_args()
    data_input = Path(args.data_dir).expanduser().absolute()
    output_input = Path(args.out_dir).expanduser().absolute()
    data_root = data_input.resolve()
    output_root = output_input.resolve()
    try:
        if _has_link_or_reparse_ancestor(data_input) or not data_root.is_dir():
            raise ExactGateError("DATA_ROOT_INVALID", "data-dir is not a regular directory", "Mount the authorized CURE-Bench data root", "dataset_discovery")
        if _has_link_or_reparse_ancestor(output_input):
            raise ExactGateError("OUTPUT_DIRECTORY_UNSAFE", "out-dir traverses a symlink or reparse point", "Use a regular managed solution output directory", "artifact_materialization")
        manifest_path = discover_encoder_manifest(data_root, args.encoder_manifest)
        encoder = FrozenTransformerEncoder(manifest_path, data_root)
        ledger_path = discover_holdout_ledger(data_root, args.holdout_ledger)
        ledger = load_holdout_ledger(ledger_path)
        primary_manifest_path = discover_primary_baseline_manifest(data_root, args.primary_baseline_manifest)
        primary_baseline = load_primary_baseline_manifest(primary_manifest_path, data_root)
        result = run_experiment(
            data_root,
            output_root,
            encoder=encoder,
            holdout_ledger=ledger,
            primary_baseline=primary_baseline,
        )
    except ExactGateError as exc:
        if not _has_link_or_reparse_ancestor(output_input):
            output_root.mkdir(parents=True, exist_ok=True)
            write_json(output_root / "exact-gate.json", exc.as_dict())
        print(json.dumps(exc.as_dict(), ensure_ascii=False, sort_keys=True))
        return 2
    except Exception as exc:  # fail closed without echoing potentially sensitive details
        gate = ExactGateError(
            "UNEXPECTED_LOCAL_FAILURE",
            f"candidate-2 stopped at {type(exc).__name__}",
            "Inspect the local fixture/runtime and rerun only after the precondition changes",
            "local_execution",
        )
        if not _has_link_or_reparse_ancestor(output_input):
            output_root.mkdir(parents=True, exist_ok=True)
            write_json(output_root / "exact-gate.json", gate.as_dict())
        print(json.dumps(gate.as_dict(), ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
