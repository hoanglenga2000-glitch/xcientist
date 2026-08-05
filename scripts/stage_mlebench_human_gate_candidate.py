#!/usr/bin/env python3
"""Stage a verified Human Gate package as a frozen one-shot regrade candidate.

This producer is offline and candidate-only.  It verifies every source package
hash, validates the withheld CSV against the public sample-submission schema,
and builds the exact ``result.json``/manifest/approval shape consumed by
``regrade_mlebench_lite_run.py``.  The generated approval is always a template
with ``approved=false``; this command never grades, submits, approves, trains,
or accesses private labels.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import zip_longest
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts import regrade_mlebench_lite_run as regrade  # noqa: E402

PACKAGE_MANIFEST_SCHEMA = "evomind.human_gate.candidate_package.v1"
PACKAGE_VERIFICATION_SCHEMA = "evomind.human_gate.package_verification.v1"
STAGED_RESULT_SCHEMA = "evomind.mlebench_lite.candidate_only_result.v1"
STAGED_CHECKPOINT_SCHEMA = "evomind.mlebench_lite.human_gate_staged_checkpoint.v1"
STAGED_RESULTS_SCHEMA = "evomind.mlebench_lite.human_gate_staged_results.v1"
STAGED_SUMMARY_SCHEMA = "evomind.mlebench_lite.human_gate_staged_summary.v1"
STAGING_VERIFICATION_SCHEMA = "evomind.mlebench_lite.human_gate_staging_verification.v1"
STAGING_PRODUCER_SCHEMA = "evomind.mlebench_lite.human_gate_staging_producer.v1"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_PACKAGE_ROOT = PROJECT_ROOT / "workspace" / "human_gate"
ALLOWED_OUTPUT_ROOT = PROJECT_ROOT / "workspace" / "human_gate_staged_runs"
DEFAULT_PUBLIC_DATA_ROOT = PROJECT_ROOT / "workspace" / "local_gpu" / "mlebench_official_data"
MANDATORY_PACKAGE_FILES = {
    "manifest.json",
    "package_verification.json",
    "independent_verification.json",
    "candidate_submission_withheld.csv",
    "frozen_plan.json",
}


class StagingError(RuntimeError):
    """Raised when a source package or staged candidate fails closed."""


@dataclass(frozen=True)
class FileRecord:
    name: str
    path: Path
    bytes: int
    sha256: str

    def portable(self, *, relative_to: Path | None = None) -> dict[str, Any]:
        path = self.path
        if relative_to is not None:
            path_value = path.relative_to(relative_to).as_posix()
        else:
            path_value = str(path)
        return {
            "name": self.name,
            "path": path_value,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class VerifiedPackage:
    root: Path
    competition_id: str
    package_manifest: dict[str, Any]
    package_verification: dict[str, Any]
    independent_verification: dict[str, Any]
    source_result: dict[str, Any]
    frozen_plan: dict[str, Any]
    files: dict[str, FileRecord]
    source_result_name: str
    metric: str
    direction: str
    cv_score: float
    model_seeds: tuple[int, ...]
    public_csv_validation: dict[str, Any]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, name: str | None = None) -> FileRecord:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise StagingError(f"Expected a regular non-symlink file: {resolved}")
    return FileRecord(
        name=name or resolved.name,
        path=resolved,
        bytes=resolved.stat().st_size,
        sha256=sha256_file(resolved),
    )


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(payload))
    return path


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise StagingError(f"Missing regular {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StagingError(f"Invalid {label} JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise StagingError(f"{label} must contain a JSON object: {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StagingError(message)


def normalized_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not SHA256_PATTERN.fullmatch(digest):
        raise StagingError(f"{label} must be a SHA-256 digest")
    return digest


def ensure_within(path: Path, root: Path, *, allow_root: bool = False) -> Path:
    root_resolved = Path(root).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise StagingError(f"Path escaped allowed root {root_resolved}: {resolved}") from exc
    if not allow_root and relative == Path("."):
        raise StagingError(f"A child path is required below {root_resolved}")
    return resolved


def validate_run_id(value: str) -> str:
    run_id = str(value or "").strip()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise StagingError("run_id contains unsafe characters")
    return run_id


def _record_path(package_root: Path, item: Mapping[str, Any], *, label: str) -> Path:
    raw_name = str(item.get("name") or "").strip()
    raw_path = str(item.get("path") or "").strip()
    basename = raw_name or (Path(raw_path).name if raw_path else "")
    if not basename or basename != Path(basename).name:
        raise StagingError(f"{label} has an unsafe file name")
    candidate = package_root / basename
    if raw_path:
        declared = Path(raw_path).expanduser()
        if not declared.is_absolute():
            declared = package_root / declared
        try:
            declared_resolved = declared.resolve()
            declared_resolved.relative_to(package_root)
            candidate = declared_resolved
        except (OSError, ValueError):
            candidate = package_root / basename
    return ensure_within(candidate, package_root)


def _index_declared_records(
    package_root: Path,
    items: Any,
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list) or not items:
        raise StagingError(f"{label}.files must be a non-empty list")
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(items):
        if not isinstance(raw, Mapping):
            raise StagingError(f"{label}.files[{index}] must be an object")
        path = _record_path(package_root, raw, label=f"{label}.files[{index}]")
        name = path.name
        if name in indexed:
            raise StagingError(f"{label} contains duplicate file record: {name}")
        expected_bytes = raw.get("bytes")
        if type(expected_bytes) is not int or int(expected_bytes) < 0:
            raise StagingError(f"{label} has invalid byte count for {name}")
        expected_sha = normalized_sha256(raw.get("sha256"), label=f"{label} {name}")
        actual = file_record(path, name=name)
        if actual.bytes != expected_bytes or actual.sha256 != expected_sha:
            raise StagingError(
                f"{label} hash drift for {name}: expected {expected_bytes}/{expected_sha}, "
                f"got {actual.bytes}/{actual.sha256}"
            )
        indexed[name] = {
            "path": actual.path,
            "bytes": actual.bytes,
            "sha256": actual.sha256,
        }
    return indexed


def _assert_boundaries(payload: Mapping[str, Any], *, label: str) -> None:
    for key in ("private_labels_used", "official_grader_executed", "kaggle_submission_executed"):
        require(payload.get(key) is False, f"{label}.{key} must be false")
    if "automatic_submission" in payload:
        require(payload.get("automatic_submission") is False, f"{label}.automatic_submission must be false")
    if "process_signals_sent" in payload:
        require(payload.get("process_signals_sent") == 0, f"{label}.process_signals_sent must be zero")
    if "human_gate_preserved" in payload:
        require(payload.get("human_gate_preserved") is True, f"{label}.human_gate_preserved must be true")


def _recursive_sha_records(payload: Any, key_name: str) -> list[str]:
    found: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key == key_name and isinstance(value, Mapping) and value.get("sha256"):
                found.append(normalized_sha256(value.get("sha256"), label=f"{key_name}.sha256"))
            found.extend(_recursive_sha_records(value, key_name))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(_recursive_sha_records(value, key_name))
    return found


def validate_public_submission_csv(
    candidate_path: Path,
    sample_path: Path,
    *,
    prediction_bounds: tuple[float, float] | None = (0.0, 1.0),
) -> dict[str, Any]:
    """Validate exact public schema, ID order, and finite predictions.

    Classification packages keep the default probability bounds. Regression
    packages pass ``None`` and remain subject to the same exact schema, row,
    ID-order, numeric, and finiteness checks.
    """

    candidate_record = file_record(candidate_path)
    sample_record = file_record(sample_path)
    errors: list[str] = []
    row_count = 0
    prediction_columns = 0
    min_probability = math.inf
    max_probability = -math.inf
    with candidate_record.path.open("r", encoding="utf-8-sig", newline="") as candidate_handle, sample_record.path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as sample_handle:
        candidate_reader = csv.reader(candidate_handle)
        sample_reader = csv.reader(sample_handle)
        candidate_header = next(candidate_reader, None)
        sample_header = next(sample_reader, None)
        if not candidate_header or candidate_header != sample_header or len(candidate_header) < 2:
            errors.append("header_or_column_order_mismatch")
        else:
            prediction_columns = len(candidate_header) - 1
        for row_index, pair in enumerate(
            zip_longest(candidate_reader, sample_reader, fillvalue=None), start=2
        ):
            candidate_row, sample_row = pair
            if candidate_row is None or sample_row is None:
                errors.append("row_count_mismatch")
                break
            row_count += 1
            if candidate_header and len(candidate_row) != len(candidate_header):
                errors.append(f"candidate_column_count_mismatch_at_row_{row_index}")
                break
            if not sample_row or candidate_row[0] != sample_row[0]:
                errors.append(f"id_order_mismatch_at_row_{row_index}")
                break
            for value in candidate_row[1:]:
                try:
                    probability = float(value)
                except ValueError:
                    errors.append(f"non_numeric_prediction_at_row_{row_index}")
                    break
                if not math.isfinite(probability) or (
                    prediction_bounds is not None
                    and not (prediction_bounds[0] <= probability <= prediction_bounds[1])
                ):
                    errors.append(f"invalid_probability_at_row_{row_index}")
                    break
                min_probability = min(min_probability, probability)
                max_probability = max(max_probability, probability)
            if errors:
                break
    return {
        "schema": "evomind.mlebench_lite.public_submission_validation.v1",
        "valid": not errors and row_count > 0,
        "errors": errors,
        "candidate": candidate_record.portable(),
        "sample_submission": sample_record.portable(),
        "columns": candidate_header or [],
        "prediction_columns": prediction_columns,
        "rows": row_count,
        "minimum_probability": min_probability if math.isfinite(min_probability) else None,
        "maximum_probability": max_probability if math.isfinite(max_probability) else None,
        "private_labels_used": False,
    }


def _extract_metric(source_result: Mapping[str, Any]) -> tuple[str, str, float]:
    gate = source_result.get("confirmation_gate")
    metrics = source_result.get("metrics")
    require(isinstance(gate, Mapping), "Source confirmation_gate is missing")
    require(gate.get("passed") is True, "Source confirmation gate did not pass")
    require(isinstance(metrics, Mapping), "Source metrics are missing")
    metric = str(gate.get("metric") or "").strip()
    direction = str(gate.get("direction") or "").strip()
    candidates: Sequence[tuple[str, str]] = (
        ("ensemble_oof_auc", "maximize"),
        ("ensemble_oof_log_loss", "minimize"),
        ("ensemble_oof_rmse", "minimize"),
    )
    for key, fallback_direction in candidates:
        if key in metrics:
            value = float(metrics[key])
            require(math.isfinite(value), f"Source metric {key} is not finite")
            return metric or key, direction or fallback_direction, value
    raise StagingError("Source result has no supported ensemble OOF metric")


def _verify_recomputed_metric(
    independent: Mapping[str, Any], source_result: Mapping[str, Any], cv_score: float
) -> None:
    metrics = source_result.get("metrics") or {}
    if "ensemble_oof_auc" in metrics:
        key = "recomputed_ensemble_oof_auc"
    elif "ensemble_oof_log_loss" in metrics:
        key = "recomputed_ensemble_oof_log_loss"
    else:
        key = "recomputed_ensemble_oof_rmse"
    require(key in independent, f"Independent verification lacks {key}")
    recomputed = float(independent[key])
    require(math.isfinite(recomputed) and abs(recomputed - cv_score) <= 1e-12, "Independent ensemble metric drift")


def verify_human_gate_package(
    package_dir: Path,
    *,
    public_data_root: Path = DEFAULT_PUBLIC_DATA_ROOT,
    allowed_package_root: Path = DEFAULT_PACKAGE_ROOT,
) -> VerifiedPackage:
    package_root = ensure_within(package_dir, allowed_package_root)
    require(package_root.is_dir() and not package_root.is_symlink(), "Package root must be a non-symlink directory")
    manifest_path = package_root / "manifest.json"
    package_verification_path = package_root / "package_verification.json"
    independent_path = package_root / "independent_verification.json"
    package_manifest = read_json(manifest_path, label="package manifest")
    package_verification = read_json(package_verification_path, label="package verification")
    independent = read_json(independent_path, label="independent verification")
    require(package_manifest.get("schema") == PACKAGE_MANIFEST_SCHEMA, "Package manifest schema mismatch")
    require(package_manifest.get("status") == "ready_for_human_review_not_submitted", "Package is not Human Gate ready")
    require(package_manifest.get("candidate_ready_for_human_gate") is True, "Package candidate-ready flag is false")
    require(package_verification.get("schema") == PACKAGE_VERIFICATION_SCHEMA, "Package verification schema mismatch")
    require(package_verification.get("status") == "verified", "Package verification did not pass")
    require(package_verification.get("candidate_csv_present") is True, "Package verification lacks candidate CSV")
    require(
        package_verification.get("independent_verification_passed") is True,
        "Package verification lacks independent pass",
    )
    require(package_verification.get("automatic_submission") is False, "Package verification enabled submission")
    require(independent.get("status") == "verification_passed", "Independent verification status did not pass")
    require(independent.get("ok") is True, "Independent verification ok flag is false")
    require(independent.get("candidate_ready_for_human_gate") is True, "Independent candidate-ready flag is false")
    require(independent.get("errors") == [], "Independent verification contains errors")
    _assert_boundaries(package_manifest, label="package manifest")
    _assert_boundaries(independent, label="independent verification")

    manifest_records = _index_declared_records(
        package_root, package_manifest.get("files"), label="package manifest"
    )
    verification_records = _index_declared_records(
        package_root, package_verification.get("files"), label="package verification"
    )
    shared = set(manifest_records) & set(verification_records)
    for name in shared:
        require(
            manifest_records[name]["bytes"] == verification_records[name]["bytes"]
            and manifest_records[name]["sha256"] == verification_records[name]["sha256"],
            f"Manifest and package verification disagree for {name}",
        )
    for mandatory in MANDATORY_PACKAGE_FILES - {"manifest.json", "package_verification.json"}:
        require(
            mandatory in manifest_records and mandatory in verification_records,
            f"Mandatory package artifact is not hash-bound twice: {mandatory}",
        )
    if "manifest.json" in verification_records:
        actual_manifest = file_record(manifest_path)
        expected = verification_records["manifest.json"]
        require(
            expected["bytes"] == actual_manifest.bytes and expected["sha256"] == actual_manifest.sha256,
            "Package manifest self-record drift",
        )

    unsafe_entries = [
        path.name
        for path in package_root.iterdir()
        if path.is_symlink() or not path.is_file()
    ]
    require(not unsafe_entries, f"Package contains non-regular entries: {sorted(unsafe_entries)}")
    actual_names = {
        path.name
        for path in package_root.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    declared_names = set(manifest_records) | set(verification_records) | {"package_verification.json", "manifest.json"}
    require(actual_names == declared_names, f"Package has undeclared or missing files: {sorted(actual_names ^ declared_names)}")
    all_files = {name: file_record(package_root / name, name=name) for name in sorted(actual_names)}

    competition_id = str(package_manifest.get("competition_id") or "").strip()
    require(bool(competition_id), "Package competition_id is missing")
    independent_result_sha = normalized_sha256(
        independent.get("result_sha256"), label="Independent result SHA256"
    )
    result_matches = [record for record in all_files.values() if record.sha256 == independent_result_sha]
    require(len(result_matches) == 1, "Independent result SHA256 does not identify exactly one package file")
    source_result_record = result_matches[0]
    source_result = read_json(source_result_record.path, label="source confirmation result")
    require(source_result.get("competition_id") == competition_id, "Source result competition mismatch")
    require(source_result.get("status") == "confirmation_passed_human_gate_pending", "Source result is not Human Gate pending")
    require(source_result.get("candidate_ready_for_human_gate") is True, "Source result candidate-ready flag is false")
    _assert_boundaries(source_result, label="source result")
    metric, direction, cv_score = _extract_metric(source_result)
    _verify_recomputed_metric(independent, source_result, cv_score)

    frozen_plan_record = all_files["frozen_plan.json"]
    require(
        frozen_plan_record.sha256
        == normalized_sha256(independent.get("plan_sha256"), label="Independent plan SHA256"),
        "Frozen plan SHA256 differs from independent verification",
    )
    frozen_plan = read_json(frozen_plan_record.path, label="frozen plan")
    require(frozen_plan.get("competition_id") == competition_id, "Frozen plan competition mismatch")
    _assert_boundaries(frozen_plan.get("boundaries") or {}, label="frozen plan boundaries")

    candidate_record = all_files["candidate_submission_withheld.csv"]
    withheld = source_result.get("submission_withheld")
    require(isinstance(withheld, Mapping), "Source result submission_withheld record is missing")
    require(
        candidate_record.sha256
        == normalized_sha256(withheld.get("sha256"), label="Source withheld submission SHA256"),
        "Withheld CSV SHA256 differs from source result",
    )
    sample_path = ensure_within(
        Path(public_data_root) / competition_id / "prepared" / "public" / "sample_submission.csv",
        public_data_root,
    )
    prediction_bounds = None if "rmse" in metric.lower() else (0.0, 1.0)
    public_validation = validate_public_submission_csv(
        candidate_record.path,
        sample_path,
        prediction_bounds=prediction_bounds,
    )
    require(public_validation.get("valid") is True, f"Candidate CSV failed public validation: {public_validation['errors']}")
    sample_sha = public_validation["sample_submission"]["sha256"]
    declared_sample_hashes = set(_recursive_sha_records(frozen_plan, "sample_submission"))
    declared_sample_hashes.update(_recursive_sha_records(source_result, "sample_submission"))
    require(sample_sha in declared_sample_hashes, "Public sample-submission SHA256 is not frozen by source evidence")

    package_public_metrics = package_manifest.get("public_oof_metrics")
    require(isinstance(package_public_metrics, Mapping), "Package public OOF metrics are missing")
    source_metrics = source_result.get("metrics") or {}
    if "ensemble_oof_auc" in source_metrics:
        package_metric = float(package_public_metrics.get("ensemble_oof_auc"))
    elif "ensemble_oof_log_loss" in source_metrics:
        package_metric = float(package_public_metrics.get("ensemble_oof_log_loss"))
    else:
        package_metric = float(package_public_metrics.get("ensemble_oof_rmse"))
    require(abs(package_metric - cv_score) <= 1e-12, "Package manifest and source result metric drift")
    seeds = tuple(int(item.get("model_seed")) for item in source_result.get("seed_records") or [])
    declared_seeds_raw = (
        (frozen_plan.get("confirmation_gate") or {}).get("confirmation_seeds")
        or frozen_plan.get("seeds")
    )
    require(isinstance(declared_seeds_raw, list) and declared_seeds_raw, "Frozen plan lacks a confirmation seed contract")
    declared_seeds = tuple(int(value) for value in declared_seeds_raw)
    if len(declared_seeds) == 1:
        require(
            frozen_plan.get("single_seed_nested_oof_confirmation") is True,
            "Single-seed packages require an explicit nested-OOF confirmation contract",
        )
    if len(set(declared_seeds)) != len(declared_seeds):
        require(
            frozen_plan.get("allow_duplicate_model_seeds_for_distinct_runs") is True,
            "Frozen plan seeds are not unique",
        )
        source_run_ids = tuple(
            str(item.get("run_id") or "") for item in source_result.get("seed_records") or []
        )
        require(
            all(source_run_ids) and len(set(source_run_ids)) == len(source_run_ids),
            "Duplicate model seeds require distinct frozen run IDs",
        )
    require(seeds == declared_seeds, "Source result seeds differ from the frozen plan")

    return VerifiedPackage(
        root=package_root,
        competition_id=competition_id,
        package_manifest=package_manifest,
        package_verification=package_verification,
        independent_verification=independent,
        source_result=source_result,
        frozen_plan=frozen_plan,
        files=all_files,
        source_result_name=source_result_record.name,
        metric=metric,
        direction=direction,
        cv_score=cv_score,
        model_seeds=seeds,
        public_csv_validation=public_validation,
    )


def _copy_verified(record: FileRecord, destination: Path) -> FileRecord:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(record.path, destination)
    copied = file_record(destination, name=destination.name)
    require(
        copied.bytes == record.bytes and copied.sha256 == record.sha256,
        f"Copied file hash drift: {record.name}",
    )
    return copied


def _portable_source_records(
    verified: VerifiedPackage,
    staged_records: Mapping[str, FileRecord],
    run_root: Path,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name in sorted(verified.files):
        source = verified.files[name]
        staged = staged_records[name]
        output.append(
            {
                "name": name,
                "source_path": str(source.path),
                "staged_path": staged.path.relative_to(run_root).as_posix(),
                "bytes": source.bytes,
                "sha256": source.sha256,
                "copied_as_submission": name == "candidate_submission_withheld.csv",
            }
        )
    return output


def _build_candidate_result(
    verified: VerifiedPackage,
    *,
    submission_sha256: str,
    source_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_by_name = {str(item["name"]): item for item in source_records}
    return {
        "schema": STAGED_RESULT_SCHEMA,
        "created_at": now_iso(),
        "competition_id": verified.competition_id,
        "status": regrade.CONFIRMATION_STATUS,
        "promotion_gate": {
            "name": "verified_human_gate_package_confirmation",
            "passed": True,
            "source_confirmation_status": verified.source_result["status"],
            "source_confirmation_gate_passed": True,
            "package_verification_passed": True,
            "independent_verification_passed": True,
        },
        "candidate_only": True,
        "official_grader_confirmation_pending": True,
        "official_grader_withheld": True,
        "official_grader_executed": False,
        "valid_submission": True,
        "private_grader": {
            "status": "withheld",
            "reason": regrade.CONFIRMATION_REASON,
            "official_mlebench_grader_executed": False,
            "score": None,
        },
        "submission_path": "submission.csv",
        "submission_sha256": submission_sha256,
        "budget": {
            "seed": 42,
            "confirmation_model_seeds": list(verified.model_seeds),
            "training_reused": True,
            "training_executed": False,
        },
        "metric": verified.metric,
        "direction": verified.direction,
        "cv_score": verified.cv_score,
        "source_human_gate_package": {
            "path": str(verified.root),
            "package_manifest_sha256": source_by_name["manifest.json"]["sha256"],
            "package_verification_sha256": source_by_name["package_verification.json"]["sha256"],
            "independent_verification_sha256": source_by_name["independent_verification.json"]["sha256"],
            "confirmation_result_name": verified.source_result_name,
            "confirmation_result_sha256": source_by_name[verified.source_result_name]["sha256"],
            "frozen_plan_sha256": source_by_name["frozen_plan.json"]["sha256"],
            "all_source_artifacts_hash_verified": True,
        },
        "public_submission_validation": verified.public_csv_validation,
        "private_labels_used": False,
        "training_executed": False,
        "training_reused": True,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "human_gate_preserved": True,
        "claim_boundary": "Frozen candidate-only staging; explicit external human approval remains required before one-shot grading.",
    }


def _tree_records(run_root: Path, *, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = exclude or set()
    records: list[dict[str, Any]] = []
    for path in sorted(run_root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(run_root).as_posix()
        if relative in excluded:
            continue
        record = file_record(path, name=relative)
        records.append(
            {
                "path": relative,
                "bytes": record.bytes,
                "sha256": record.sha256,
            }
        )
    return records


def stage_candidate_run(
    package_dir: Path,
    *,
    run_id: str,
    public_data_root: Path = DEFAULT_PUBLIC_DATA_ROOT,
    output_root: Path = ALLOWED_OUTPUT_ROOT,
    allowed_package_root: Path = DEFAULT_PACKAGE_ROOT,
) -> dict[str, Any]:
    run_id = validate_run_id(run_id)
    output_root = ensure_within(output_root, ALLOWED_OUTPUT_ROOT, allow_root=True)
    output_root.mkdir(parents=True, exist_ok=True)
    target = ensure_within(output_root / run_id, ALLOWED_OUTPUT_ROOT)
    verified = verify_human_gate_package(
        package_dir,
        public_data_root=public_data_root,
        allowed_package_root=allowed_package_root,
    )
    temporary = ensure_within(
        output_root / f".staging-{run_id}-{os.getpid()}", ALLOWED_OUTPUT_ROOT
    )
    if temporary.exists():
        raise StagingError(f"Temporary staging path already exists: {temporary}")
    if target.exists():
        raise StagingError(
            f"Immutable staged run already exists; choose a new reviewed run_id: {target}"
        )
    try:
        task_root = temporary / verified.competition_id
        source_snapshot = task_root / "source_package"
        approvals_root = temporary / "approvals"
        source_snapshot.mkdir(parents=True)
        approvals_root.mkdir(parents=True)
        staged_source_records: dict[str, FileRecord] = {}
        for name, source in verified.files.items():
            destination = (
                task_root / "submission.csv"
                if name == "candidate_submission_withheld.csv"
                else source_snapshot / name
            )
            staged_source_records[name] = _copy_verified(source, destination)
        portable_sources = _portable_source_records(
            verified, staged_source_records, temporary
        )
        submission = staged_source_records["candidate_submission_withheld.csv"]
        result = _build_candidate_result(
            verified,
            submission_sha256=submission.sha256,
            source_records=portable_sources,
        )
        result_path = write_json(task_root / "result.json", result)
        regrade._validate_source_result(result, verified.competition_id)
        result_record = file_record(result_path)
        candidate_manifest = {
            "schema": regrade.CANDIDATE_MANIFEST_SCHEMA,
            "created_at": now_iso(),
            "frozen": True,
            "run_id": run_id,
            "competition_id": verified.competition_id,
            "source_result_sha256": result_record.sha256,
            "submission": {
                "path": "submission.csv",
                "sha256": submission.sha256,
                "size_bytes": submission.bytes,
            },
            "source_human_gate_package": {
                "path": str(verified.root),
                "files": portable_sources,
                "package_manifest_schema": verified.package_manifest["schema"],
                "package_verification_schema": verified.package_verification["schema"],
                "independent_verification_schema": verified.independent_verification["schema"],
                "all_hashes_verified": True,
            },
            "public_submission_validation": verified.public_csv_validation,
            "candidate_only": True,
            "approved": False,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
        }
        manifest_path = write_json(
            task_root / "frozen_candidate_manifest.json", candidate_manifest
        )
        manifest_record = file_record(manifest_path)
        approval_relative = PurePosixPath("..") / verified.competition_id / manifest_path.name
        approval_template = {
            "schema": regrade.APPROVAL_SCHEMA,
            "created_at": now_iso(),
            "approval_id": None,
            "approved": False,
            "approved_by": None,
            "approved_at": None,
            "run_id": run_id,
            "competition_id": verified.competition_id,
            "candidate_manifest_path": approval_relative.as_posix(),
            "candidate_manifest_sha256": manifest_record.sha256,
            "submission_sha256": submission.sha256,
            "instructions": (
                "Do not edit this frozen template in place. After human review, copy it to a new "
                "approval file, set approved=true, and fill approval_id/approved_by/approved_at."
            ),
            "automatic_approval": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        }
        approval_path = write_json(
            approvals_root / f"{verified.competition_id}.approval_template.json",
            approval_template,
        )
        approval_record = file_record(approval_path)
        checkpoint = {
            "schema": STAGED_CHECKPOINT_SCHEMA,
            "created_at": now_iso(),
            "run_id": run_id,
            "status": "human_approval_pending",
            "requested": [verified.competition_id],
            "completed": [verified.competition_id],
            "candidate_only": True,
            "frozen": True,
            "approved": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        }
        checkpoint_path = write_json(temporary / "checkpoint.json", checkpoint)
        results_current = {
            "schema": STAGED_RESULTS_SCHEMA,
            "created_at": now_iso(),
            "run_id": run_id,
            "status": "human_approval_pending",
            "results": [
                {
                    "competition_id": verified.competition_id,
                    "result_path": f"{verified.competition_id}/result.json",
                    "result_sha256": result_record.sha256,
                    "candidate_manifest_path": f"{verified.competition_id}/{manifest_path.name}",
                    "candidate_manifest_sha256": manifest_record.sha256,
                    "submission_sha256": submission.sha256,
                }
            ],
            "approved": False,
        }
        results_path = write_json(temporary / "results_current.json", results_current)
        summary = {
            "schema": STAGED_SUMMARY_SCHEMA,
            "created_at": now_iso(),
            "run_id": run_id,
            "status": "human_approval_pending",
            "competition_count": 1,
            "requested": [verified.competition_id],
            "staged": [verified.competition_id],
            "candidate_only": True,
            "source_package_verified": True,
            "independent_verification_passed": True,
            "public_submission_valid": True,
            "frozen_candidate_manifest_sha256": manifest_record.sha256,
            "submission_sha256": submission.sha256,
            "approval_template_path": approval_path.relative_to(temporary).as_posix(),
            "approval_template_sha256": approval_record.sha256,
            "approved": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "private_labels_used": False,
            "training_executed": False,
            "process_signals_sent": 0,
        }
        summary_path = write_json(temporary / "summary.json", summary)
        staged_files = _tree_records(temporary, exclude={"verification.json"})
        verification = {
            "schema": STAGING_VERIFICATION_SCHEMA,
            "created_at": now_iso(),
            "status": "verified_human_approval_pending",
            "run_id": run_id,
            "competition_id": verified.competition_id,
            "checks": {
                "package_manifest_hashes_verified": True,
                "package_verification_hashes_verified": True,
                "candidate_csv_hash_verified": True,
                "independent_verification_hash_verified": True,
                "independent_verification_passed": True,
                "source_confirmation_gate_passed": True,
                "public_submission_schema_and_ids_valid": True,
                "candidate_result_matches_regrade_contract": True,
                "frozen_candidate_manifest_bound": True,
                "approval_template_unapproved": True,
            },
            "source_package": {
                "path": str(verified.root),
                "files": portable_sources,
            },
            "public_submission_validation": verified.public_csv_validation,
            "staged_files": staged_files,
            "approved": False,
            "automatic_approval": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "private_labels_used": False,
            "network_accessed": False,
            "process_signals_sent": 0,
        }
        verification_path = write_json(temporary / "verification.json", verification)
        verify_staged_run(temporary, public_data_root=public_data_root)

        temporary.rename(target)
        final_verification = verify_staged_run(target, public_data_root=public_data_root)
        report = {
            "schema": STAGING_PRODUCER_SCHEMA,
            "created_at": now_iso(),
            "status": "staged_human_approval_pending",
            "run_id": run_id,
            "competition_id": verified.competition_id,
            "run_dir": str(target),
            "checkpoint": str(target / checkpoint_path.relative_to(temporary)),
            "summary": str(target / summary_path.relative_to(temporary)),
            "results_current": str(target / results_path.relative_to(temporary)),
            "verification": str(target / verification_path.relative_to(temporary)),
            "approval_template": str(target / approval_path.relative_to(temporary)),
            "candidate_manifest": str(target / manifest_path.relative_to(temporary)),
            "submission": str(target / task_root.relative_to(temporary) / "submission.csv"),
            "submission_sha256": submission.sha256,
            "candidate_manifest_sha256": manifest_record.sha256,
            "verification_sha256": sha256_file(target / "verification.json"),
            "staged_verification": final_verification,
            "approved": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "network_accessed": False,
        }
        return report
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def verify_staged_run(
    run_dir: Path,
    *,
    public_data_root: Path = DEFAULT_PUBLIC_DATA_ROOT,
) -> dict[str, Any]:
    run_root = ensure_within(run_dir, ALLOWED_OUTPUT_ROOT)
    checkpoint = read_json(run_root / "checkpoint.json", label="staged checkpoint")
    summary = read_json(run_root / "summary.json", label="staged summary")
    results_current = read_json(run_root / "results_current.json", label="staged results")
    verification = read_json(run_root / "verification.json", label="staging verification")
    require(checkpoint.get("schema") == STAGED_CHECKPOINT_SCHEMA, "Staged checkpoint schema mismatch")
    require(summary.get("schema") == STAGED_SUMMARY_SCHEMA, "Staged summary schema mismatch")
    require(results_current.get("schema") == STAGED_RESULTS_SCHEMA, "Staged results schema mismatch")
    require(verification.get("schema") == STAGING_VERIFICATION_SCHEMA, "Staging verification schema mismatch")
    run_id = str(checkpoint.get("run_id") or "")
    require(run_root.name == run_id or run_root.name.startswith(f".staging-{run_id}-"), "Staged run_id/path mismatch")
    requested = checkpoint.get("requested")
    require(isinstance(requested, list) and len(requested) == 1, "Staged checkpoint must request one competition")
    competition_id = str(requested[0])
    require(summary.get("requested") == requested, "Staged summary requested set mismatch")
    require(verification.get("competition_id") == competition_id, "Staging verification competition mismatch")
    task_root = ensure_within(run_root / competition_id, run_root)
    result_path = task_root / "result.json"
    submission_path = task_root / "submission.csv"
    manifest_path = task_root / "frozen_candidate_manifest.json"
    approval_path = run_root / "approvals" / f"{competition_id}.approval_template.json"
    result = read_json(result_path, label="candidate-only result")
    manifest = read_json(manifest_path, label="frozen candidate manifest")
    approval = read_json(approval_path, label="approval template")
    regrade._validate_source_result(result, competition_id)
    require(manifest.get("schema") == regrade.CANDIDATE_MANIFEST_SCHEMA, "Candidate manifest schema mismatch")
    require(manifest.get("frozen") is True and manifest.get("run_id") == run_id, "Candidate manifest is not frozen for run")
    require(manifest.get("competition_id") == competition_id, "Candidate manifest competition mismatch")
    require(manifest.get("source_result_sha256") == sha256_file(result_path), "Candidate result hash drift")
    submission_record = file_record(submission_path)
    manifest_submission = manifest.get("submission") or {}
    require(manifest_submission.get("path") == "submission.csv", "Candidate manifest submission path is not portable")
    require(manifest_submission.get("sha256") == submission_record.sha256, "Candidate manifest submission hash drift")
    require(manifest_submission.get("size_bytes") == submission_record.bytes, "Candidate manifest submission size drift")
    require(result.get("submission_path") == "submission.csv", "Candidate result submission path is not portable")
    require(result.get("submission_sha256") == submission_record.sha256, "Candidate result submission hash drift")
    require(approval.get("schema") == regrade.APPROVAL_SCHEMA, "Approval template schema mismatch")
    require(approval.get("approved") is False, "Approval template must remain unapproved")
    require(approval.get("automatic_approval") is False, "Approval template enabled automatic approval")
    require(approval.get("run_id") == run_id and approval.get("competition_id") == competition_id, "Approval binding mismatch")
    require(approval.get("candidate_manifest_sha256") == sha256_file(manifest_path), "Approval manifest hash drift")
    require(approval.get("submission_sha256") == submission_record.sha256, "Approval submission hash drift")
    approval_manifest_path = (approval_path.parent / str(approval["candidate_manifest_path"])).resolve()
    require(approval_manifest_path == manifest_path.resolve(), "Approval manifest path drift")
    source_package = manifest.get("source_human_gate_package")
    require(isinstance(source_package, Mapping), "Candidate manifest lacks source package evidence")
    source_files = source_package.get("files")
    require(isinstance(source_files, list) and source_files, "Candidate manifest lacks source file records")
    for item in source_files:
        require(isinstance(item, Mapping), "Candidate source file record is invalid")
        staged_path = ensure_within(run_root / str(item.get("staged_path") or ""), run_root)
        staged_record = file_record(staged_path)
        require(
            staged_record.bytes == item.get("bytes")
            and staged_record.sha256 == item.get("sha256"),
            f"Candidate source package snapshot drift: {item.get('name')}",
        )
    result_entries = results_current.get("results")
    require(isinstance(result_entries, list) and len(result_entries) == 1, "Staged results must bind one candidate")
    result_entry = result_entries[0]
    require(result_entry.get("competition_id") == competition_id, "Staged results competition mismatch")
    require(result_entry.get("result_sha256") == sha256_file(result_path), "Staged results result hash drift")
    require(
        result_entry.get("candidate_manifest_sha256") == sha256_file(manifest_path),
        "Staged results manifest hash drift",
    )
    require(result_entry.get("submission_sha256") == submission_record.sha256, "Staged results submission hash drift")
    require(summary.get("frozen_candidate_manifest_sha256") == sha256_file(manifest_path), "Summary manifest hash drift")
    require(summary.get("submission_sha256") == submission_record.sha256, "Summary submission hash drift")
    sample_path = Path(public_data_root) / competition_id / "prepared" / "public" / "sample_submission.csv"
    public_validation = validate_public_submission_csv(submission_path, sample_path)
    require(public_validation.get("valid") is True, "Staged CSV failed public validation")
    expected_records = verification.get("staged_files")
    require(isinstance(expected_records, list) and expected_records, "Staging verification lacks file records")
    expected_by_path = {str(item.get("path")): item for item in expected_records if isinstance(item, Mapping)}
    actual_records = _tree_records(run_root, exclude={"verification.json"})
    actual_by_path = {str(item["path"]): item for item in actual_records}
    require(set(expected_by_path) == set(actual_by_path), "Staged file inventory drift")
    for path, actual in actual_by_path.items():
        expected = expected_by_path[path]
        require(
            actual["bytes"] == expected.get("bytes") and actual["sha256"] == expected.get("sha256"),
            f"Staged file hash drift: {path}",
        )
    verification_checks = verification.get("checks")
    require(
        isinstance(verification_checks, Mapping)
        and verification_checks
        and all(value is True for value in verification_checks.values()),
        "Staging verification contains a failed check",
    )
    require(verification.get("approved") is False, "Staging verification was approved")
    require(verification.get("automatic_approval") is False, "Staging verification enabled approval")
    require(verification.get("official_grader_executed") is False, "Staging verification claims grader execution")
    require(verification.get("kaggle_submission_executed") is False, "Staging verification claims Kaggle execution")
    require(verification.get("network_accessed") is False, "Staging verification claims network access")
    require(not (task_root / "regrades").exists(), "Staged candidate already contains regrade evidence")
    require(summary.get("approved") is False and checkpoint.get("approved") is False, "Staged run was approved")
    require(summary.get("official_grader_executed") is False, "Staged summary claims grader execution")
    require(summary.get("kaggle_submission_executed") is False, "Staged summary claims Kaggle execution")
    return {
        "schema": STAGING_VERIFICATION_SCHEMA,
        "status": "verified_human_approval_pending",
        "run_id": run_id,
        "competition_id": competition_id,
        "staged_file_count": len(actual_records) + 1,
        "submission_sha256": submission_record.sha256,
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "verification_sha256": sha256_file(run_root / "verification.json"),
        "approved": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "private_labels_used": False,
        "network_accessed": False,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--public-data-root", type=Path, default=DEFAULT_PUBLIC_DATA_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify-package")
    stage = subparsers.add_parser("stage")
    stage.add_argument("--run-id", required=True)
    stage.add_argument("--output-root", type=Path, default=ALLOWED_OUTPUT_ROOT)
    verify = subparsers.add_parser("verify-staged")
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--output-root", type=Path, default=ALLOWED_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "verify-package":
        verified = verify_human_gate_package(
            args.package_dir,
            public_data_root=args.public_data_root,
        )
        payload = {
            "schema": STAGING_PRODUCER_SCHEMA,
            "created_at": now_iso(),
            "status": "package_verified",
            "package_dir": str(verified.root),
            "competition_id": verified.competition_id,
            "file_count": len(verified.files),
            "cv_score": verified.cv_score,
            "approved": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "network_accessed": False,
        }
    elif args.command == "stage":
        payload = stage_candidate_run(
            args.package_dir,
            run_id=args.run_id,
            public_data_root=args.public_data_root,
            output_root=args.output_root,
        )
    else:
        run_id = validate_run_id(args.run_id)
        output_root = ensure_within(args.output_root, ALLOWED_OUTPUT_ROOT, allow_root=True)
        payload = verify_staged_run(
            output_root / run_id,
            public_data_root=args.public_data_root,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
