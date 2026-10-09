from __future__ import annotations

"""Offline E2LMC candidate-2 experiment.

This candidate is deliberately separate from :mod:`e2lmc_goal_v2`.  The
first-generation script selected two architecture holdouts and sampled the
points at iterations divisible by 4000 (with an all-points fallback).  A
second run must therefore not look at either of those architectures or those
iteration points.  This module accepts only a locally mounted, SHA-bound
official snapshot and a supervisor-owned read-only holdout ledger.  It never
fetches data, mutates the ledger, contacts a remote service, or consumes test
labels.

The candidate is a signal-quality comparison, not an official leaderboard
score.  ``mmluvar`` remains the fixed organizer baseline.  Candidate metric
selection is performed only on the development portion of fresh points; the
latest fresh time block in each eligible experiment is opened once as a blind
holdout.  The paired bootstrap resamples experiment groups so baseline and
candidate are compared on exactly the same units.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SEED = 20260902
BOOTSTRAP_ROUNDS = 2000
BASELINE_METRIC = "mmluvar"
SOURCE_URL = (
    "https://raw.githubusercontent.com/et-slm-evaluation/et-slm-evaluation.github.io/"
    "refs/heads/main/assets/data/plot-data-combined-all.json"
)
SOURCE_SHA256 = "afbea8052cc0b3c3c08a2b4d6000e0c1b960f9d9b4bdd6671d167abb01278d69"
LEDGER_SCHEMA = "evomind.e2lmc.holdout_ledger.v1"
MIN_POINTS_PER_PARTITION = 4
HOLDOUT_FRACTION = 0.25

# These were the two architecture holdouts consumed by candidate-1.  Keep the
# deny-list even if an older ledger omits one of the names.
PRIOR_ARCHITECTURE_HOLDOUTS = frozenset({"dense-500m-arch2", "dense-1b-arch2"})


class ExactGateError(ValueError):
    """A bounded, actionable gate instead of a generic runtime exception."""

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


@dataclass(frozen=True)
class Point:
    experiment: str
    iteration: float
    value: float

    @property
    def token(self) -> str:
        return point_token(self.experiment, self.iteration)


@dataclass(frozen=True)
class Ledger:
    path: Path
    sha256: str
    source_sha256: str
    consumed_experiments: frozenset[str]
    consumed_points: frozenset[str]
    available_experiments: frozenset[str]
    available_points: frozenset[str]


@dataclass(frozen=True)
class ExperimentSplit:
    experiment: str
    development: tuple[float, ...]
    holdout: tuple[float, ...]


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
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def atomic_copyfile(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    with source.open("rb") as source_handle, temporary.open("wb") as destination_handle:
        while block := source_handle.read(1024 * 1024):
            destination_handle.write(block)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    os.replace(temporary, destination)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return path.is_symlink() or bool(attributes & reparse)


def _has_link_ancestor(path: Path) -> bool:
    absolute = path.expanduser().absolute()
    return any(_is_link_or_reparse(candidate) for candidate in (absolute, *absolute.parents))


def _regular_file(path: Path, *, code: str, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ExactGateError(
            code,
            f"{label} is missing",
            f"Mount the authorized local {label} snapshot and retry candidate-2",
            "local_input_mount",
        ) from exc
    if not resolved.is_file() or _has_link_ancestor(path) or path.is_symlink():
        raise ExactGateError(
            f"{code}_NOT_REGULAR",
            f"{label} must be a regular non-symlink file",
            f"Provide a regular local {label} snapshot without symlink or reparse indirection",
            "local_input_mount",
        )
    return resolved


def _norm_name(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _fold(value: Any) -> str:
    return _norm_name(value).casefold()


def canonical_iteration(value: Any) -> float:
    if isinstance(value, bool):
        raise ExactGateError(
            "ITERATION_NOT_NUMERIC",
            "iteration values must be finite numbers",
            "Restore numeric iteration values in the official snapshot",
            "source_schema_validation",
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExactGateError(
            "ITERATION_NOT_NUMERIC",
            "iteration values must be finite numbers",
            "Restore numeric iteration values in the official snapshot",
            "source_schema_validation",
        ) from exc
    if not math.isfinite(result):
        raise ExactGateError(
            "ITERATION_NOT_FINITE",
            "iteration values must be finite",
            "Restore finite iteration values in the official snapshot",
            "source_schema_validation",
        )
    return result


def point_token(experiment: str, iteration: Any) -> str:
    value = canonical_iteration(iteration)
    rendered = str(int(value)) if value.is_integer() else format(value, ".15g")
    return f"{_norm_name(experiment)}|{rendered}"


def _array(value: Any, *, experiment: str, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ExactGateError(
            "CURVE_NOT_ARRAY",
            f"{experiment}.{field} must be an array",
            "Provide the frozen official learning-curve arrays",
            "source_schema_validation",
        )
    result: list[float] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            result.append(float("nan"))
            continue
        result.append(number if math.isfinite(number) else float("nan"))
    return result


def load_official_source(path: Path, *, expected_sha256: str | None = None) -> tuple[bytes, dict[str, Any]]:
    source_path = _regular_file(path, code="OFFICIAL_SOURCE_SNAPSHOT_MISSING", label="official E2LMC source")
    raw = source_path.read_bytes()
    actual_sha = sha256_bytes(raw)
    if expected_sha256 is None:
        expected_sha256 = SOURCE_SHA256
    if actual_sha != expected_sha256:
        raise ExactGateError(
            "OFFICIAL_SOURCE_SHA_MISMATCH",
            "the local official E2LMC snapshot does not match the frozen source SHA-256",
            f"Mount the official snapshot with SHA-256 {expected_sha256}",
            "official_source_validation",
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExactGateError(
            "OFFICIAL_SOURCE_JSON_INVALID",
            "the local official E2LMC snapshot is not valid JSON",
            "Restore the byte-identical official JSON snapshot",
            "official_source_validation",
        ) from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), Mapping):
        raise ExactGateError(
            "OFFICIAL_SOURCE_SCHEMA_MISMATCH",
            "official E2LMC source must contain a data object",
            "Mount the organizer source matching the frozen E2LMC contract",
            "official_source_validation",
        )
    metadata = payload.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or not isinstance(metadata.get("metrics"), Sequence)
        or isinstance(metadata.get("metrics"), (str, bytes, bytearray))
    ):
        raise ExactGateError(
            "OFFICIAL_SOURCE_METADATA_MISSING",
            "official E2LMC source metadata.metrics is missing",
            "Restore metadata.metrics from the frozen organizer source",
            "official_source_validation",
        )
    return raw, dict(payload)


_HIDDEN_LABEL_KEYS = {
    "answer",
    "answers",
    "gold",
    "gold_answer",
    "ground_truth",
    "groundtruth",
    "label",
    "labels",
    "test_labels",
    "hidden_labels",
}


def _find_hidden_label_keys(value: Any, *, path: str = "", depth: int = 0) -> list[str]:
    if depth > 32:
        raise ExactGateError(
            "SOURCE_NESTING_TOO_DEEP",
            "official source nesting exceeds the bounded audit depth",
            "Provide a normalized official source snapshot",
            "source_schema_validation",
        )
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            folded = re.sub(r"[\s\-]+", "_", _fold(key))
            current = f"{path}.{key}" if path else str(key)
            if folded in _HIDDEN_LABEL_KEYS and ("test" in current.casefold() or folded in {"test_labels", "hidden_labels"}):
                found.append(current)
            found.extend(_find_hidden_label_keys(child, path=current, depth=depth + 1))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_hidden_label_keys(child, path=f"{path}[{index}]", depth=depth + 1))
    return found


def _point_entry(value: Any) -> tuple[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    experiment = value.get("experiment", value.get("architecture", value.get("model")))
    iteration = value.get("iteration", value.get("step", value.get("global_step")))
    if experiment is None or iteration is None:
        return None
    return _norm_name(experiment), canonical_iteration(iteration)


def _collect_point_entries(value: Any, *, depth: int = 0) -> list[tuple[str, float]]:
    if depth > 16:
        raise ExactGateError(
            "LEDGER_NESTING_TOO_DEEP",
            "holdout ledger nesting exceeds the bounded audit depth",
            "Provide a normalized holdout-ledger.v1",
            "ledger_validation",
        )
    entries: list[tuple[str, float]] = []
    if isinstance(value, Mapping):
        point = _point_entry(value)
        if point is not None:
            entries.append(point)
        for child in value.values():
            entries.extend(_collect_point_entries(child, depth=depth + 1))
    elif isinstance(value, list):
        for child in value:
            entries.extend(_collect_point_entries(child, depth=depth + 1))
    return entries


def load_holdout_ledger(path: Path, *, source_sha256: str) -> Ledger:
    ledger_path = _regular_file(path, code="HOLDOUT_LEDGER_MISSING", label="holdout-ledger.v1")
    raw = ledger_path.read_bytes()
    ledger_sha = sha256_bytes(raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExactGateError(
            "HOLDOUT_LEDGER_JSON_INVALID",
            "holdout-ledger.v1 is not valid JSON",
            "Restore the supervisor-owned immutable ledger",
            "ledger_validation",
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != LEDGER_SCHEMA:
        raise ExactGateError(
            "HOLDOUT_LEDGER_SCHEMA_MISMATCH",
            "ledger schema is not evomind.e2lmc.holdout_ledger.v1",
            "Provide the supervisor-owned holdout-ledger.v1",
            "ledger_validation",
        )
    declared_source = str(payload.get("official_source_sha256", ""))
    if len(declared_source) != 64 or any(char not in "0123456789abcdefABCDEF" for char in declared_source):
        raise ExactGateError(
            "HOLDOUT_LEDGER_SOURCE_SHA_INVALID",
            "holdout ledger source binding is not a SHA-256 value",
            "Publish a ledger with a canonical SHA-256 source binding",
            "ledger_validation",
        )
    if declared_source != source_sha256:
        raise ExactGateError(
            "HOLDOUT_LEDGER_SOURCE_MISMATCH",
            "holdout ledger is bound to a different official source snapshot",
            f"Issue a read-only ledger bound to source SHA-256 {source_sha256}",
            "ledger_validation",
        )
    if bool(payload.get("hidden_test_labels_used", False)) or bool(payload.get("test_labels_used", False)):
        raise ExactGateError(
            "HOLDOUT_LEDGER_TEST_LABELS",
            "holdout ledger reports hidden/test labels were used",
            "Provide a clean ledger that contains no test labels",
            "ledger_validation",
        )
    if payload.get("read_only") is False or payload.get("immutable") is False:
        raise ExactGateError(
            "HOLDOUT_LEDGER_NOT_READ_ONLY",
            "holdout ledger is explicitly mutable",
            "Publish a supervisor-owned read-only immutable ledger",
            "ledger_validation",
        )
    consumed_experiments: set[str] = set()
    for key in (
        "consumed_experiments",
        "consumed_experiment_ids",
        "consumed_architectures",
        "prior_holdout_experiments",
        "holdout_experiments",
    ):
        value = payload.get(key, [])
        if value is None:
            continue
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ExactGateError(
                "HOLDOUT_LEDGER_FIELD_INVALID",
                f"{key} must be an array",
                "Normalize consumed experiment identities in holdout-ledger.v1",
                "ledger_validation",
            )
        consumed_experiments.update(_norm_name(item) for item in value if _norm_name(item))
    consumed_points_raw: list[Any] = []
    for key in (
        "consumed_points",
        "consumed_iteration_points",
        "consumed_iteration_point_ids",
        "point_holdouts",
        "entries",
    ):
        value = payload.get(key, [])
        if value is not None:
            consumed_points_raw.append(value)
    point_entries: list[tuple[str, float]] = []
    for value in consumed_points_raw:
        point_entries.extend(_collect_point_entries(value))
    # An explicit non-empty point ledger must carry identities, otherwise a
    # hash-only row would be impossible to compare without reading secrets.
    declared_point_count = payload.get("consumed_point_count")
    if declared_point_count not in (None, 0) and not point_entries:
        raise ExactGateError(
            "HOLDOUT_LEDGER_POINT_IDENTITIES_MISSING",
            "ledger declares consumed iteration points but provides no experiment/iteration identities",
            "Publish a read-only ledger with canonical experiment and iteration entries",
            "ledger_validation",
        )
    consumed_points = {point_token(experiment, iteration) for experiment, iteration in point_entries}
    available_experiments: set[str] = set()
    for key in ("available_experiments", "available_experiment_ids", "fresh_experiments"):
        value = payload.get(key, [])
        if value is None:
            continue
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ExactGateError(
                "HOLDOUT_LEDGER_FIELD_INVALID",
                f"{key} must be an array",
                "Normalize available experiment identities in holdout-ledger.v1",
                "ledger_validation",
            )
        available_experiments.update(_norm_name(item) for item in value if _norm_name(item))
    available_points_raw: list[Any] = []
    for key in ("available_points", "available_iteration_points", "available_point_holdouts", "fresh_points"):
        value = payload.get(key, [])
        if value is not None:
            available_points_raw.append(value)
    available_entries: list[tuple[str, float]] = []
    for value in available_points_raw:
        available_entries.extend(_collect_point_entries(value))
    available_points = {point_token(experiment, iteration) for experiment, iteration in available_entries}
    if available_points and consumed_points & available_points:
        raise ExactGateError(
            "HOLDOUT_LEDGER_CONFLICT",
            "a point is listed as both consumed and available",
            "Repair the immutable holdout ledger before candidate-2",
            "ledger_validation",
        )
    # A candidate-1 architecture holdout must be represented even if the
    # ledger only records its names; all points in those experiments are denied.
    consumed_experiments.update(PRIOR_ARCHITECTURE_HOLDOUTS)
    return Ledger(
        path=ledger_path,
        sha256=ledger_sha,
        source_sha256=declared_source,
        consumed_experiments=frozenset(consumed_experiments),
        consumed_points=frozenset(consumed_points),
        available_experiments=frozenset(available_experiments),
        available_points=frozenset(available_points),
    )


def rank_correlation(values: np.ndarray) -> float:
    if len(values) < 3 or not np.all(np.isfinite(values)):
        return 0.0
    x_rank = np.arange(len(values), dtype=float)
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    y_rank = np.asarray(order, dtype=float)
    n = len(values)
    d_squared_sum = float(np.sum((x_rank - y_rank) ** 2))
    return max(0.0, 1.0 - (6.0 * d_squared_sum) / (n * (n * n - 1)))


def autocorrelation(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(values):
        return 0.0
    left = values[lag:]
    right = values[:-lag]
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    result = float(np.corrcoef(left, right)[0, 1])
    return result if math.isfinite(result) else 0.0


def signal_quality(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if len(array) < 4 or not np.all(np.isfinite(array)):
        raise ExactGateError(
            "SIGNAL_SEQUENCE_TOO_SHORT",
            "RC signal quality requires at least four finite ordered points",
            "Provide a fresh time block with enough finite iterations",
            "fresh_point_split",
        )
    lags = range(1, max(2, len(array) // 4))
    autocorrelation_score = float(np.mean([abs(autocorrelation(array, lag)) for lag in lags]))
    return {
        "monotonicity": rank_correlation(array),
        "autocorrelation": autocorrelation_score,
        "score": 0.5 * (rank_correlation(array) + autocorrelation_score),
    }


def candidate1_selected_tokens(experiment: Mapping[str, Any], name: str, metric: str = BASELINE_METRIC) -> frozenset[str]:
    iterations = _array(experiment.get("iteration", []), experiment=name, field="iteration")
    values = _array(experiment.get(metric, []), experiment=name, field=metric)
    if len(iterations) != len(values) or len(values) < 4:
        return frozenset()
    finite = [index for index, value in enumerate(values) if math.isfinite(value) and math.isfinite(iterations[index])]
    selected = [index for index in finite if iterations[index] % 4000 == 0]
    if len(selected) < 4:
        selected = finite
    return frozenset(point_token(name, iterations[index]) for index in selected)


def fresh_points_for_experiment(
    name: str,
    experiment: Mapping[str, Any],
    metrics: Sequence[str],
    denied_tokens: frozenset[str],
    available_points: frozenset[str] = frozenset(),
) -> dict[str, dict[str, float]]:
    iterations = _array(experiment.get("iteration", []), experiment=name, field="iteration")
    arrays = {metric: _array(experiment.get(metric, []), experiment=name, field=metric) for metric in metrics}
    lengths = {len(iterations), *(len(values) for values in arrays.values())}
    if len(lengths) != 1:
        raise ExactGateError(
            "CURVE_LENGTH_MISMATCH",
            f"{name} has mismatched iteration/metric lengths",
            "Restore aligned official learning-curve arrays",
            "source_schema_validation",
        )
    rows: dict[str, dict[str, float]] = {}
    for index, iteration in enumerate(iterations):
        token = point_token(name, iteration)
        if token in denied_tokens or not math.isfinite(iteration):
            continue
        if available_points and token not in available_points:
            continue
        values = {metric: arrays[metric][index] for metric in metrics}
        if not all(math.isfinite(value) for value in values.values()):
            continue
        if token in rows:
            raise ExactGateError(
                "DUPLICATE_ITERATION_POINT",
                f"{name} contains duplicate iteration identity {token}",
                "Restore unique iteration points in the official snapshot",
                "source_schema_validation",
            )
        rows[token] = {"iteration": iteration, **values}
    return rows


def temporal_split(rows: Mapping[str, Mapping[str, float]], *, experiment: str) -> ExperimentSplit:
    ordered = sorted((float(item["iteration"]) for item in rows.values()))
    if len(ordered) < 2 * MIN_POINTS_PER_PARTITION:
        raise ExactGateError(
            "INSUFFICIENT_FRESH_POINTS",
            f"{experiment} has fewer than eight fresh aligned iteration points",
            "Provide a new unconsumed source snapshot or a supervisor ledger with more fresh points",
            "fresh_point_split",
        )
    holdout_count = max(MIN_POINTS_PER_PARTITION, int(math.ceil(len(ordered) * HOLDOUT_FRACTION)))
    holdout_count = min(holdout_count, len(ordered) - MIN_POINTS_PER_PARTITION)
    if holdout_count < MIN_POINTS_PER_PARTITION:
        raise ExactGateError(
            "INSUFFICIENT_FRESH_HOLDOUT_POINTS",
            f"{experiment} cannot form a four-point fresh temporal holdout",
            "Provide more unconsumed iteration points",
            "fresh_point_split",
        )
    return ExperimentSplit(
        experiment=experiment,
        development=tuple(ordered[:-holdout_count]),
        holdout=tuple(ordered[-holdout_count:]),
    )


def _values_for_iterations(rows: Mapping[str, Mapping[str, float]], iterations: Sequence[float], metric: str) -> np.ndarray:
    by_iteration = {float(item["iteration"]): float(item[metric]) for item in rows.values()}
    try:
        return np.asarray([by_iteration[float(iteration)] for iteration in iterations], dtype=float)
    except KeyError as exc:
        raise ExactGateError(
            "ITERATION_ALIGNMENT_FAILED",
            "metric values are not aligned to the selected iteration points",
            "Restore a complete aligned metric curve",
            "source_schema_validation",
        ) from exc


def paired_group_bootstrap(rows: Sequence[Mapping[str, Any]], *, rounds: int = BOOTSTRAP_ROUNDS, seed: int = SEED) -> dict[str, float | int]:
    if len(rows) < 2:
        raise ExactGateError(
            "BOOTSTRAP_GROUPS_INSUFFICIENT",
            "paired bootstrap requires at least two independent experiment groups",
            "Provide at least two eligible experiments with fresh holdout points",
            "paired_bootstrap",
        )
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(int(rounds)):
        indices = rng.integers(0, len(rows), len(rows))
        deltas.append(
            float(
                np.mean(
                    [float(rows[index]["candidate"]) - float(rows[index]["baseline"]) for index in indices]
                )
            )
        )
    values = np.asarray(deltas, dtype=float)
    return {
        "rounds": int(rounds),
        "resampling_unit": "experiment_group_paired_delta",
        "difference_mean": float(values.mean()),
        "difference_standard_error": float(values.std(ddof=1)),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
    }


def _resolve_source(args: argparse.Namespace) -> Path:
    if args.source_json:
        return Path(args.source_json)
    if args.data_dir:
        root = Path(args.data_dir)
        for name in ("plot-data-combined-all.json", "official-source.json", "e2lmc-source.json"):
            candidate = root / name
            if candidate.is_file():
                return candidate
    raise ExactGateError(
        "OFFICIAL_SOURCE_SNAPSHOT_MISSING",
        "candidate-2 requires an explicit local official source snapshot",
        "Mount the SHA-bound E2LMC JSON and pass --source-json",
        "local_input_mount",
    )


def _resolve_ledger(args: argparse.Namespace) -> Path:
    if args.holdout_ledger:
        return Path(args.holdout_ledger)
    if args.data_dir:
        root = Path(args.data_dir)
        for name in ("holdout-ledger.v1.json", "holdout-ledger.json"):
            candidate = root / name
            if candidate.is_file():
                return candidate
    raise ExactGateError(
        "HOLDOUT_LEDGER_MISSING",
        "candidate-2 requires a supervisor-owned read-only holdout-ledger.v1",
        "Provide the immutable ledger before opening a new holdout",
        "local_input_mount",
    )


def run_candidate(source_raw: bytes, payload: Mapping[str, Any], ledger: Ledger, *, source_sha: str) -> dict[str, Any]:
    hidden = _find_hidden_label_keys(payload)
    if hidden:
        raise ExactGateError(
            "HIDDEN_TEST_LABELS_PRESENT",
            "official source contains answer/test-label fields outside the public curve contract",
            "Provide a public curve snapshot without hidden/test labels",
            "source_schema_validation",
        )
    data = payload["data"]
    name_map: dict[str, Any] = {}
    for raw_name, experiment in data.items():
        name = _norm_name(raw_name)
        if not name or name in name_map:
            raise ExactGateError(
                "EXPERIMENT_NAME_COLLISION",
                "official source contains empty or colliding normalized experiment names",
                "Restore unique canonical experiment identities",
                "source_schema_validation",
            )
        if not isinstance(experiment, Mapping):
            raise ExactGateError(
                "EXPERIMENT_SCHEMA_MISMATCH",
                f"experiment {name!r} must be an object of aligned curves",
                "Restore the official experiment object schema",
                "source_schema_validation",
            )
        name_map[name] = experiment
    names = sorted(name_map)
    if len(names) < 3:
        raise ExactGateError(
            "INSUFFICIENT_EXPERIMENTS",
            "official source contains fewer than three experiments",
            "Mount the complete organizer curve snapshot",
            "source_schema_validation",
        )
    metadata_metrics = [_norm_name(metric) for metric in payload["metadata"]["metrics"]]
    if BASELINE_METRIC not in metadata_metrics:
        raise ExactGateError(
            "BASELINE_METRIC_MISSING",
            "official source does not expose the fixed mmluvar baseline metric",
            "Restore the organizer mmluvar series",
            "source_schema_validation",
        )
    candidate_metrics = [
        metric
        for metric in metadata_metrics
        if metric not in {"iteration", "gtokens", BASELINE_METRIC}
        and all(isinstance(name_map[name], Mapping) and metric in name_map[name] for name in names)
    ]
    if not candidate_metrics:
        raise ExactGateError(
            "CANDIDATE_METRICS_MISSING",
            "no complete non-baseline official metrics are available",
            "Mount the complete official RC metric set",
            "source_schema_validation",
        )

    denied_tokens: set[str] = set(ledger.consumed_points)
    denied_experiments = set(ledger.consumed_experiments) | set(PRIOR_ARCHITECTURE_HOLDOUTS)
    # Candidate-1's numeric_curve selector was applied independently to each
    # metric.  Union every selected metric point, rather than only mmluvar,
    # so a point selected for any prior feature cannot become candidate-2's
    # holdout by accident.
    for name in names:
        for metric in metadata_metrics:
            # ``gtokens`` is an x-axis/metadata field and was explicitly
            # excluded from candidate-1's metric set.
            if metric not in {"iteration", "gtokens"} and metric in name_map[name]:
                denied_tokens.update(candidate1_selected_tokens(name_map[name], name, metric))
    eligible_names = [name for name in names if name not in denied_experiments]
    if ledger.available_experiments:
        eligible_names = [name for name in eligible_names if name in ledger.available_experiments]
    if len(eligible_names) < 2:
        raise ExactGateError(
            "NO_ELIGIBLE_EXPERIMENTS",
            "all experiments are consumed architecture holdouts or denied by the ledger",
            "Issue a ledger with at least two new model-training experiments",
            "fresh_point_split",
        )

    aligned_metrics = [BASELINE_METRIC, *candidate_metrics]
    fresh_rows: dict[str, dict[str, dict[str, float]]] = {}
    splits: list[ExperimentSplit] = []
    for name in eligible_names:
        rows = fresh_points_for_experiment(
            name,
            name_map[name],
            aligned_metrics,
            frozenset(denied_tokens),
            ledger.available_points,
        )
        split = temporal_split(rows, experiment=name)
        fresh_rows[name] = rows
        splits.append(split)
    if len(splits) < 2:
        raise ExactGateError(
            "BOOTSTRAP_GROUPS_INSUFFICIENT",
            "fewer than two eligible experiment groups remain after the fresh split",
            "Provide at least two independent unconsumed experiments",
            "paired_bootstrap",
        )

    # All metric normalization statistics come from development points only.
    normalization: dict[str, dict[str, float]] = {}
    for metric in candidate_metrics:
        development_values = np.asarray(
            [
                fresh_rows[item.experiment][point_token(item.experiment, iteration)][metric]
                for item in splits
                for iteration in item.development
            ],
            dtype=float,
        )
        minimum = float(np.min(development_values))
        maximum = float(np.max(development_values))
        normalization[metric] = {"min": minimum, "max": maximum, "range": max(maximum - minimum, 1e-20)}

    def normalized(values: np.ndarray, metric: str) -> np.ndarray:
        stats = normalization[metric]
        return (values - stats["min"]) / stats["range"]

    development_metric_scores: dict[str, float] = {}
    for metric in candidate_metrics:
        scores: list[float] = []
        for item in splits:
            values = _values_for_iterations(fresh_rows[item.experiment], item.development, metric)
            scores.append(signal_quality(normalized(values, metric))["score"])
        development_metric_scores[metric] = float(np.mean(scores))
    ordered_metrics = sorted(candidate_metrics, key=lambda metric: (-development_metric_scores[metric], metric))
    candidate_history: list[dict[str, Any]] = []
    for count in (2, 3, 5, 8):
        selected = ordered_metrics[: min(count, len(ordered_metrics))]
        scores: list[float] = []
        for item in splits:
            curves = [
                normalized(_values_for_iterations(fresh_rows[item.experiment], item.development, metric), metric)
                for metric in selected
            ]
            scores.append(signal_quality(np.mean(np.stack(curves), axis=0))["score"])
        candidate_history.append(
            {
                "candidate_id": f"top-{len(selected)}-metric-composite",
                "metrics": selected,
                "development_score": float(np.mean(scores)),
            }
        )
    selected_candidate = max(
        candidate_history,
        key=lambda item: (item["development_score"], -len(item["metrics"]), item["candidate_id"]),
    )
    selected_metrics = list(selected_candidate["metrics"])

    holdout_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for item in splits:
        baseline_values = _values_for_iterations(fresh_rows[item.experiment], item.holdout, BASELINE_METRIC)
        candidate_curves = [
            normalized(_values_for_iterations(fresh_rows[item.experiment], item.holdout, metric), metric)
            for metric in selected_metrics
        ]
        candidate_values = np.mean(np.stack(candidate_curves), axis=0)
        baseline_score = signal_quality(baseline_values)["score"]
        candidate_score = signal_quality(candidate_values)["score"]
        holdout_rows.append(
            {
                "experiment": item.experiment,
                "holdout_points": len(item.holdout),
                "baseline": baseline_score,
                "candidate": candidate_score,
                "difference_candidate_minus_baseline": candidate_score - baseline_score,
            }
        )
        for iteration, baseline_value, candidate_value in zip(item.holdout, baseline_values, candidate_values):
            point_rows.append(
                {
                    "experiment": item.experiment,
                    "iteration": iteration,
                    "baseline_value": float(baseline_value),
                    "candidate_value": float(candidate_value),
                }
            )
    baseline_score = float(np.mean([row["baseline"] for row in holdout_rows]))
    candidate_score = float(np.mean([row["candidate"] for row in holdout_rows]))
    bootstrap = paired_group_bootstrap(holdout_rows)
    minimum_margin = max(0.01 * baseline_score, float(bootstrap["difference_standard_error"]))
    gate_passed = bool(candidate_score > baseline_score and float(bootstrap["ci95_lower"]) > minimum_margin)

    # Independent verification reconstructs each experiment's ordered curves
    # from the persisted point rows; it does not merely copy the aggregate
    # values calculated above.
    recomputed_groups: dict[str, dict[str, list[tuple[float, float, float]]]] = {}
    for row in point_rows:
        recomputed_groups.setdefault(row["experiment"], {"rows": []})["rows"].append(
            (float(row["iteration"]), float(row["baseline_value"]), float(row["candidate_value"]))
        )
    recomputed_rows: list[dict[str, float | str]] = []
    for experiment, group in sorted(recomputed_groups.items()):
        ordered = sorted(group["rows"], key=lambda item: item[0])
        baseline_values = np.asarray([item[1] for item in ordered], dtype=float)
        candidate_values = np.asarray([item[2] for item in ordered], dtype=float)
        recomputed_rows.append(
            {
                "experiment": experiment,
                "baseline": signal_quality(baseline_values)["score"],
                "candidate": signal_quality(candidate_values)["score"],
            }
        )
    recomputed_baseline = float(np.mean([row["baseline"] for row in recomputed_rows]))
    recomputed_candidate = float(np.mean([row["candidate"] for row in recomputed_rows]))
    holdout_tokens = {
        point_token(row["experiment"], row["iteration"]) for row in point_rows
    }
    overlap_with_denied = sorted(holdout_tokens & (denied_tokens | {point_token(name, 0) for name in denied_experiments}))
    if overlap_with_denied:
        raise ExactGateError(
            "HOLDOUT_OVERLAP",
            "fresh holdout contains an iteration point or architecture consumed by candidate-1",
            "Issue a corrected immutable holdout ledger and restart from source validation",
            "fresh_point_split",
        )

    return {
        "source_raw_bytes": len(source_raw),
        "source_sha256": source_sha,
        "official_source_url": SOURCE_URL,
        "ledger_sha256": ledger.sha256,
        "ledger_source_sha256": ledger.source_sha256,
        "all_experiments": names,
        "denied_architecture_holdouts": sorted(denied_experiments),
        "denied_iteration_points": len(denied_tokens),
        "eligible_experiments": eligible_names,
        "splits": [
            {
                "experiment": item.experiment,
                "development_points": len(item.development),
                "holdout_points": len(item.holdout),
                "development_iteration_sha256": sha256_bytes(
                    "\n".join(str(value) for value in item.development).encode()
                ),
                "holdout_iteration_sha256": sha256_bytes(
                    "\n".join(str(value) for value in item.holdout).encode()
                ),
            }
            for item in splits
        ],
        "development_metric_scores": development_metric_scores,
        "normalization": normalization,
        "selected_metrics": selected_metrics,
        "candidate_history": candidate_history,
        "holdout_rows": holdout_rows,
        "holdout_point_rows": point_rows,
        "baseline": baseline_score,
        "candidate": candidate_score,
        "absolute_improvement": candidate_score - baseline_score,
        "relative_improvement": (candidate_score - baseline_score) / max(abs(baseline_score), 1e-12),
        "bootstrap": bootstrap,
        "minimum_margin": minimum_margin,
        "gate_passed": gate_passed,
        "independent": {
            "baseline_recomputed": recomputed_baseline,
            "candidate_recomputed": recomputed_candidate,
            "baseline_match": abs(recomputed_baseline - baseline_score) < 1e-12,
            "candidate_match": abs(recomputed_candidate - candidate_score) < 1e-12,
            "group_scores_recomputed": recomputed_rows,
            "holdout_point_overlap": 0,
            "prior_architecture_overlap": 0,
            "hidden_test_labels_used": False,
            "test_labels_used": False,
            "gate_passed": gate_passed,
        },
    }


def _artifact_files(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name not in {"artifact-manifest.json", "artifact-manifest-receipt.json"}:
            files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return files


def _write_exact_gate(output_root: Path, error: ExactGateError) -> int:
    output_input = output_root.absolute()
    if _has_link_ancestor(output_input):
        print(json.dumps(error.as_dict(), sort_keys=True))
        return 2
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "exact-gate.json", error.as_dict())
    print(json.dumps(error.as_dict(), sort_keys=True))
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline, fresh-point E2LMC candidate-2 evaluation")
    parser.add_argument("--data-dir", help="local directory containing source JSON and ledger")
    parser.add_argument("--source-json", help="local SHA-bound official source JSON; no network fallback")
    parser.add_argument("--holdout-ledger", "--ledger", dest="holdout_ledger", help="read-only holdout-ledger.v1")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    output_root = Path(args.out_dir).expanduser().resolve()
    try:
        output_input = Path(args.out_dir).expanduser().absolute()
        if _has_link_ancestor(output_input):
            raise ExactGateError(
                "OUTPUT_DIRECTORY_UNSAFE",
                "candidate-2 output path traverses a symlink or reparse point",
                "Use a regular local output directory",
                "artifact_materialization",
            )
        output_root.mkdir(parents=True, exist_ok=True)
        if any(output_root.iterdir()):
            raise ExactGateError(
                "OUTPUT_DIRECTORY_NOT_EMPTY",
                "candidate-2 output directory must be empty before an immutable run",
                "Choose a new empty output directory",
                "artifact_materialization",
            )
        source_path = _resolve_source(args)
        raw, payload = load_official_source(source_path)
        source_sha = sha256_bytes(raw)
        ledger = load_holdout_ledger(_resolve_ledger(args), source_sha256=source_sha)
        result = run_candidate(raw, payload, ledger, source_sha=source_sha)
    except ExactGateError as error:
        return _write_exact_gate(output_root, error)
    except Exception as error:  # fail closed without exposing local input details
        gate = ExactGateError(
            "UNEXPECTED_LOCAL_FAILURE",
            f"candidate-2 stopped at {type(error).__name__}",
            "Inspect the local source/ledger fixture and retry only after the precondition changes",
            "local_execution",
        )
        return _write_exact_gate(output_root, gate)

    source_path = Path(__file__).resolve(strict=True)
    source_sha = sha256_file(source_path)
    started = utc_now()
    task_contract = {
        "schema": "evomind.e2lmc.candidate2.task_contract.v2",
        "task": "fresh unconsumed iteration-point RC signal-quality comparison",
        "official_component": "RC results consistency / signal quality",
        "metric": "mean of monotonicity Spearman and mean absolute multi-lag autocorrelation",
        "baseline_metric": BASELINE_METRIC,
        "direction": "higher_is_better",
        "holdout_unit": "new temporal iteration block within eligible experiment",
        "candidate1_holdout_architectures_denied": sorted(PRIOR_ARCHITECTURE_HOLDOUTS),
        "candidate1_numeric_curve_points_denied": True,
        "hidden_test_labels_used": False,
        "test_labels_used": False,
        "external_submission": False,
        "network_access": False,
    }
    baseline_evidence = {
        "schema": "evomind.e2lmc.candidate2.baseline_evidence.v2",
        "classification": "OFFICIAL_ORGANIZER_BASELINE",
        "method": f"official {BASELINE_METRIC} learning curve scored by organizer RC formula",
        "official_source_url": SOURCE_URL,
        "official_source_sha256": result["source_sha256"],
        "implementation_sha256": source_sha,
        "reproduction_value": result["baseline"],
    }
    split_manifest = {
        "schema": "evomind.e2lmc.candidate2.split_manifest.v2",
        "seed": SEED,
        "holdout_strategy": "latest fresh temporal block per experiment",
        "candidate1_architecture_overlap": 0,
        "candidate1_iteration_overlap": 0,
        "holdout_blind_for_selection": True,
        "eligible_experiments": result["eligible_experiments"],
        "splits": result["splits"],
        "holdout_ledger_sha256": result["ledger_sha256"],
    }
    metrics = {
        "schema": "evomind.e2lmc.candidate2.metrics.v2",
        "baseline": result["baseline"],
        "candidate": result["candidate"],
        "absolute_improvement": result["absolute_improvement"],
        "relative_improvement": result["relative_improvement"],
        "selected_metrics": result["selected_metrics"],
        "candidate_history": result["candidate_history"],
        "holdout_rows": result["holdout_rows"],
        "bootstrap": result["bootstrap"],
    }
    comparison = {
        "schema": "evomind.e2lmc.candidate2.candidate_vs_baseline.v2",
        "baseline": result["baseline"],
        "candidate": result["candidate"],
        "minimum_margin": result["minimum_margin"],
        "difference_candidate_minus_baseline_ci95_lower": result["bootstrap"]["ci95_lower"],
        "gate_passed": result["gate_passed"],
    }
    independent = {
        "schema": "evomind.e2lmc.candidate2.independent_verification.v2",
        **result["independent"],
        "holdout_point_rows": len(result["holdout_point_rows"]),
    }
    dataset_audit = {
        "schema": "evomind.e2lmc.candidate2.dataset_audit.v2",
        "official_source_bytes": result["source_raw_bytes"],
        "official_source_sha256": result["source_sha256"],
        "all_experiments": result["all_experiments"],
        "eligible_experiments": result["eligible_experiments"],
        "denied_architecture_holdouts": result["denied_architecture_holdouts"],
        "denied_iteration_points": result["denied_iteration_points"],
        "holdout_ledger_sha256": result["ledger_sha256"],
        "hidden_test_labels_used": False,
        "test_labels_used": False,
        "network_access": False,
    }
    training_config = {
        "schema": "evomind.e2lmc.candidate2.training_config.v2",
        "candidate_id": "e2lmc-v2-fresh-iteration-candidate-2",
        "parent_candidate_id": "e2lmc-v2-baseline-candidate-1",
        "single_primary_change": "fresh unconsumed temporal iteration holdout plus development-only metric selection",
        "seed": SEED,
        "baseline_metric": BASELINE_METRIC,
        "candidate": "development-only selected composite over fresh non-candidate-1 iteration points",
        "selected_metrics": result["selected_metrics"],
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "single_gpu_solution": True,
        "external_submission": False,
        "network_access": False,
    }

    # All writes below are local and atomic.  Raw source values are only
    # public curve evidence; no hidden labels or credentials are copied.
    atomic_copyfile(source_path, output_root / "solution.py")
    write_json(output_root / "official-source-receipt.json", {"url": SOURCE_URL, "bytes": result["source_raw_bytes"], "sha256": result["source_sha256"]})
    write_json(output_root / "task-contract-v2.json", task_contract)
    write_json(output_root / "baseline-evidence.json", baseline_evidence)
    write_json(output_root / "baseline-reproduction.json", {**baseline_evidence, "completed_at_utc": utc_now()})
    write_json(output_root / "dataset-audit.json", dataset_audit)
    write_json(output_root / "split-manifest.json", split_manifest)
    write_json(output_root / "training-config.json", training_config)
    write_json(
        output_root / "human-baseline-evidence.json",
        {
            "schema": "evomind.e2lmc.human_baseline_evidence.v1",
            "status": "HUMAN_BASELINE_UNDEFINED",
            "sources_checked": [
                {
                    "url": SOURCE_URL,
                    "sha256": result["source_sha256"],
                    "role": "official RC curve data, not a human-participant result",
                }
            ],
            "missing_evidence": "protocol-comparable human participant mean with sample size, uncertainty, and evaluation protocol",
            "required_action": "Provide a first-party or peer-reviewed human-participant mean for the same E2LMC RC protocol",
            "resume_point": "human_gate_review",
        },
    )
    write_json(
        output_root / "citation-audit.json",
        {
            "schema": "evomind.e2lmc.citation_audit.v1",
            "status": "HUMAN_BASELINE_UNDEFINED",
            "official_data_source_verified": True,
            "human_baseline_source_verified": False,
            "reason": "the frozen public curve source does not report a protocol-comparable human participant mean",
            "required_action": "Run citation_audit against a supplied first-party or peer-reviewed human source before claiming human superiority",
        },
    )
    write_json(
        output_root / "checkpoint-index.json",
        {
            "schema": "evomind.e2lmc.candidate2.checkpoint_index.v1",
            "candidate_id": "e2lmc-v2-fresh-iteration-candidate-2",
            "completed_nodes": ["source_validated", "ledger_validated", "fresh_split", "development_selection", "holdout_evaluation", "independent_recompute"],
            "resume_node": "human_gate_review" if result["gate_passed"] else "new_unconsumed_strategy",
            "network_access": False,
        },
    )
    with (output_root / "failure-history.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        if not result["gate_passed"]:
            handle.write(
                json.dumps(
                    {
                        "at_utc": utc_now(),
                        "candidate_id": "e2lmc-v2-fresh-iteration-candidate-2",
                        "failure_code": "CANDIDATE_GATE_NOT_PASSED",
                        "next_action": "Do not reuse this holdout; request a new unconsumed strategy or organizer private evaluator",
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    write_json(output_root / "metrics.json", metrics)
    write_json(output_root / "candidate-vs-baseline.json", comparison)
    write_json(output_root / "independent-verification.json", independent)
    write_json(output_root / "normalization.json", result["normalization"])
    write_json(output_root / "holdout-point-rows.json", result["holdout_point_rows"])
    write_json(
        output_root / "environment-lock.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "network_access": False,
            "command": "python scripts/e2lmc_candidate2_goal_v2.py --source-json <LOCAL> --holdout-ledger <LOCAL> --out-dir <OUT>",
            "command_sha256": sha256_bytes(
                b"python scripts/e2lmc_candidate2_goal_v2.py --source-json <LOCAL> --holdout-ledger <LOCAL> --out-dir <OUT>"
            ),
        },
    )
    with (output_root / "candidate-history.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in result["candidate_history"]:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (output_root / "training.log").open("w", encoding="utf-8", newline="\n") as handle:
        for event in (
            {"at_utc": started, "event": "start", "source_sha256": result["source_sha256"], "network_access": False},
            {"at_utc": utc_now(), "event": "fresh_split_complete", "eligible_experiments": len(result["eligible_experiments"])},
            {"at_utc": utc_now(), "event": "baseline_complete", "score": result["baseline"]},
            {"at_utc": utc_now(), "event": "candidate_complete", "score": result["candidate"]},
            {"at_utc": utc_now(), "event": "independent_verification", "gate_passed": result["gate_passed"]},
        ):
            handle.write(json.dumps(event, separators=(",", ":")) + "\n")
    write_json(
        output_root / "retrospective-memory.json",
        {
            "schema": "evomind.retrospective_memory.v2",
            "baseline": baseline_evidence["method"],
            "candidate": "fresh temporal iteration-point composite",
            "gate_passed": result["gate_passed"],
            "lessons": [
                "Never reuse dense-500m-arch2 or dense-1b-arch2 holdouts.",
                "Exclude every point consumed by candidate-1 numeric_curve selection.",
                "Select candidate metrics only on development points and use a paired experiment bootstrap.",
            ],
            "memory_writeback_allowed": bool(result["gate_passed"] and independent["baseline_match"] and independent["candidate_match"]),
        },
    )
    write_json(
        output_root / "candidate-vs-human.json",
        {
            "schema": "evomind.e2lmc.candidate2.candidate_vs_human.v1",
            "status": "HUMAN_BASELINE_UNDEFINED",
            "candidate_independent_gate_passed": bool(result["gate_passed"]),
            "human_gate_passed": False,
            "required_evidence": "protocol-comparable human participant mean and uncertainty",
        },
    )
    files = _artifact_files(output_root)
    manifest = {
        "schema": "evomind.e2lmc.candidate2.artifact_manifest.v2",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "source_sha256": source_sha,
        "official_source_sha256": result["source_sha256"],
        "holdout_ledger_sha256": result["ledger_sha256"],
        "environment_lock": "environment-lock.json",
        "exit_code": 0,
        "gate_passed": result["gate_passed"],
        "network_access": False,
        "files": files,
    }
    write_json(output_root / "artifact-manifest.json", manifest)
    write_json(
        output_root / "artifact-manifest-receipt.json",
        {
            "artifact": "artifact-manifest.json",
            "bytes": (output_root / "artifact-manifest.json").stat().st_size,
            "sha256": sha256_file(output_root / "artifact-manifest.json"),
        },
    )
    print(json.dumps({"status": "completed", **comparison}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
