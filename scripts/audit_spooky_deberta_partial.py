"""Audit completed folds of a running Spooky DeBERTa OOF job.

The audit is deliberately read-only.  It reconstructs labels from the public
``train.csv``, loads NumPy archives with ``allow_pickle=False``, verifies fold
identity and duplicate-group isolation, and independently recomputes log-loss
for every completed fold.  A partial component score is diagnostic only: the
cross-fit meta candidate requires all outer folds and is never approximated by
this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


CLASS_COLUMNS = ("EAP", "HPL", "MWS")
COMPONENTS = ("transformer", "sparse")
EXPECTED_PLAN_SCHEMA = "evomind.spooky.deberta_oof_frozen_plan.v2"
EXPECTED_MANIFEST_SCHEMA = "evomind.spooky.deberta_oof_run.v2"


@dataclass(frozen=True)
class AuditConfig:
    run_dir: Path
    public_dir: Path
    plan_path: Path
    cross_run_plan_path: Path
    output_path: Path


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_sha256(indices: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_duplicate_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value)).lower()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return " ".join(re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", text))


def duplicate_group_report(
    texts: Sequence[str], truth: np.ndarray, folds: np.ndarray, *, seed: int
) -> dict[str, Any]:
    normalized = [normalize_duplicate_text(value) for value in texts]
    keys = [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in normalized]
    unique_keys = {value: index for index, value in enumerate(sorted(set(keys)))}
    groups = np.asarray([unique_keys[value] for value in keys], dtype=np.int64)
    counts = np.bincount(groups)
    conflicting_labels = 0
    crossing_folds = 0
    for group in range(len(counts)):
        mask = groups == group
        conflicting_labels += int(len(set(np.asarray(truth)[mask].tolist())) != 1)
        crossing_folds += int(len(set(np.asarray(folds)[mask].tolist())) != 1)
    return {
        "schema": "evomind.spooky.duplicate_groups.v1",
        "normalization": "NFKC_lower_word_number_tokens_v1",
        "rows": len(truth),
        "unique_groups": int(len(counts)),
        "duplicate_groups": int(np.sum(counts > 1)),
        "duplicate_rows_beyond_first": int(np.sum(np.maximum(counts - 1, 0))),
        "maximum_group_size": int(counts.max(initial=0)),
        "conflicting_label_groups": conflicting_labels,
        "folds": int(len(np.unique(folds))),
        "seed": seed,
        "group_isolation": crossing_folds == 0,
        "groups_crossing_folds": crossing_folds,
    }


def probability_report(values: np.ndarray, *, rows: int, name: str) -> dict[str, Any]:
    raw = np.asarray(values)
    shape_ok = raw.shape == (rows, len(CLASS_COLUMNS))
    finite = bool(np.isfinite(raw).all())
    nonnegative = bool(finite and np.all(raw >= 0.0))
    bounded = bool(finite and np.all(raw <= 1.0))
    normalized = bool(
        shape_ok
        and finite
        and np.allclose(raw.sum(axis=1), 1.0, atol=1e-7, rtol=0.0)
    )
    return {
        "name": name,
        "shape": list(raw.shape),
        "expected_shape": [rows, len(CLASS_COLUMNS)],
        "dtype": str(raw.dtype),
        "finite": finite,
        "nonnegative": nonnegative,
        "bounded": bounded,
        "normalized": normalized,
        "passed": shape_ok and finite and nonnegative and bounded and normalized,
    }


def multiclass_log_loss(truth: np.ndarray, probability: np.ndarray) -> float:
    return float(
        log_loss(
            np.asarray(truth, dtype=np.int64),
            np.asarray(probability, dtype=np.float64),
            labels=np.arange(len(CLASS_COLUMNS)),
        )
    )


def load_numeric_bundle(path: Path, names: Sequence[str]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(names) - set(archive.files))
        if missing:
            raise KeyError(f"Numeric bundle is missing arrays: {missing}")
        for name in names:
            arrays[name] = np.asarray(archive[name])
    return arrays


def discover_completed_folds(run_dir: Path, expected_folds: int) -> list[int]:
    completed: list[int] = []
    for metadata_path in sorted(run_dir.glob("fold_*_result.json")):
        match = re.fullmatch(r"fold_(\d+)_result\.json", metadata_path.name)
        if match is None:
            continue
        fold = int(match.group(1))
        if not 0 <= fold < expected_folds:
            continue
        prediction_path = run_dir / f"fold_{fold}_predictions.npz"
        metadata = read_json(metadata_path)
        if metadata.get("status") == "passed" and prediction_path.is_file():
            completed.append(fold)
    return sorted(set(completed))


def _load_fold_assignment(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"train_id", "truth", "fold"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"Fold assignment is missing arrays: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in sorted(required)}
    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "allow_pickle": False,
        "train_id_dtype": str(arrays["train_id"].dtype),
        "train_id_safe_unicode": arrays["train_id"].dtype.kind == "U",
    }
    return arrays, report


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def audit(config: AuditConfig) -> dict[str, Any]:
    run_dir = config.run_dir.resolve()
    public_dir = config.public_dir.resolve()
    plan_path = config.plan_path.resolve()
    cross_run_plan_path = config.cross_run_plan_path.resolve()
    manifest_path = run_dir / "manifest.json"
    fold_assignment_path = run_dir / "fold_assignments.npz"
    heartbeat_path = run_dir / "heartbeat.json"
    train_path = public_dir / "train.csv"
    required = (
        plan_path,
        cross_run_plan_path,
        manifest_path,
        fold_assignment_path,
        train_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    plan = read_json(plan_path)
    cross_run_plan = read_json(cross_run_plan_path)
    manifest = read_json(manifest_path)
    heartbeat = read_json(heartbeat_path) if heartbeat_path.is_file() else {}
    plan_sha256 = sha256_file(plan_path)
    train_sha256 = sha256_file(train_path)
    expected_folds = int(plan["training"]["folds"])
    class_order = tuple(plan["inputs"]["class_order"])
    if class_order != CLASS_COLUMNS:
        raise ValueError(f"Unexpected class order: {class_order}")

    train = pd.read_csv(train_path)
    if list(train.columns) != ["id", "text", "author"]:
        raise ValueError("Spooky public train schema is not exact")
    mapped = train["author"].map({value: index for index, value in enumerate(CLASS_COLUMNS)})
    if mapped.isna().any():
        raise ValueError("Spooky public train contains an unexpected author label")
    public_truth = mapped.to_numpy(dtype=np.int64)
    public_ids = train["id"].astype(str).to_numpy(dtype=np.str_)
    public_text = train["text"].fillna("").astype(str).tolist()

    assignment, assignment_report = _load_fold_assignment(fold_assignment_path)
    stored_truth = np.asarray(assignment["truth"], dtype=np.int64)
    stored_folds = np.asarray(assignment["fold"], dtype=np.int16)
    stored_ids = np.asarray(assignment["train_id"])
    duplicate_report = duplicate_group_report(
        public_text,
        public_truth,
        stored_folds,
        seed=int(plan["training"]["seed"]),
    )
    duplicate_expected = dict(plan["inputs"]["duplicate_group_report"])
    duplicate_comparable = {key: duplicate_report[key] for key in duplicate_expected}

    prior_summary_path = Path(cross_run_plan["prior"]["summary"]["path"]).resolve()
    prior_bundle_path = Path(cross_run_plan["prior"]["bundle"]["path"]).resolve()
    prior_summary = read_json(prior_summary_path)
    prior_names = (
        "truth",
        "fold",
        "transformer_oof",
        "sparse_oof",
        "byte_oof",
        "candidate_oof",
    )
    prior = load_numeric_bundle(prior_bundle_path, prior_names)
    prior_alignment = {
        "allow_pickle": False,
        "summary_hash": sha256_file(prior_summary_path)
        == cross_run_plan["prior"]["summary"]["sha256"],
        "bundle_hash": sha256_file(prior_bundle_path)
        == cross_run_plan["prior"]["bundle"]["sha256"],
        "truth": np.array_equal(np.asarray(prior["truth"], dtype=np.int64), public_truth),
        "fold": np.array_equal(np.asarray(prior["fold"], dtype=np.int16), stored_folds),
    }
    prior_alignment_passed = bool(
        prior_alignment["allow_pickle"] is False
        and prior_alignment["summary_hash"]
        and prior_alignment["bundle_hash"]
        and prior_alignment["truth"]
        and prior_alignment["fold"]
    )

    completed_folds = discover_completed_folds(run_dir, expected_folds)
    rows = len(train)
    current_oof = {
        name: np.full((rows, len(CLASS_COLUMNS)), np.nan, dtype=np.float64)
        for name in COMPONENTS
    }
    write_counts = np.zeros(rows, dtype=np.uint8)
    fold_records: list[dict[str, Any]] = []
    runner_sha256 = manifest.get("source_contract", {}).get("runner", {}).get(
        "actual_sha256"
    )
    for fold in completed_folds:
        metadata_path = run_dir / f"fold_{fold}_result.json"
        prediction_path = run_dir / f"fold_{fold}_predictions.npz"
        metadata = read_json(metadata_path)
        expected_valid = np.flatnonzero(stored_folds == fold).astype(np.int64)
        expected_fit = np.flatnonzero(stored_folds != fold).astype(np.int64)
        with np.load(prediction_path, allow_pickle=False) as archive:
            required_arrays = {
                "valid_indices",
                "test_indices",
                *(f"{name}_{scope}" for name in COMPONENTS for scope in ("valid", "test")),
            }
            missing = sorted(required_arrays - set(archive.files))
            if missing:
                raise KeyError(f"Fold {fold} archive is missing arrays: {missing}")
            valid_indices = np.asarray(archive["valid_indices"], dtype=np.int64)
            test_indices = np.asarray(archive["test_indices"], dtype=np.int64)
            checks: dict[str, bool] = {
                "metadata_status": metadata.get("status") == "passed",
                "metadata_fold": metadata.get("fold") == fold,
                "prediction_hash": metadata.get("prediction_sha256")
                == sha256_file(prediction_path),
                "plan_hash": metadata.get("plan_sha256") == plan_sha256,
                "runner_hash": metadata.get("runner_sha256") == runner_sha256,
                "valid_indices": np.array_equal(valid_indices, expected_valid),
                "test_indices": np.array_equal(
                    test_indices, np.arange(int(plan["inputs"]["test_rows"]), dtype=np.int64)
                ),
                "fit_index_hash": metadata.get("fit_index_sha256")
                == index_sha256(expected_fit),
                "valid_index_hash": metadata.get("valid_index_sha256")
                == index_sha256(expected_valid),
                "fit_valid_disjoint": metadata.get("fit_valid_disjoint") is True
                and not bool(np.intersect1d(expected_fit, expected_valid).size),
                "fixed_checkpoint_budget": metadata.get(
                    "checkpoint_selection_used_outer_fold"
                )
                is False,
                "private_labels_unused": metadata.get("private_labels_used") is False,
                "official_grader_not_executed": metadata.get(
                    "official_grader_executed"
                )
                is False,
                "kaggle_submission_not_executed": metadata.get(
                    "kaggle_submission_executed"
                )
                is False,
                "no_process_signals": metadata.get("process_signals_sent") == 0,
            }
            component_records: dict[str, Any] = {}
            for name in COMPONENTS:
                valid_values = np.asarray(archive[f"{name}_valid"], dtype=np.float64)
                test_values = np.asarray(archive[f"{name}_test"], dtype=np.float64)
                valid_report = probability_report(
                    valid_values, rows=len(expected_valid), name=f"fold_{fold}_{name}_valid"
                )
                test_report = probability_report(
                    test_values,
                    rows=int(plan["inputs"]["test_rows"]),
                    name=f"fold_{fold}_{name}_test",
                )
                checks[f"{name}_valid_probability"] = valid_report["passed"]
                checks[f"{name}_test_probability"] = test_report["passed"]
                independent_score = (
                    multiclass_log_loss(public_truth[expected_valid], valid_values)
                    if valid_report["passed"]
                    else None
                )
                metadata_score = _safe_float(
                    metadata.get("component_log_loss", {}).get(name)
                )
                score_matches = bool(
                    independent_score is not None
                    and metadata_score is not None
                    and math.isclose(
                        independent_score,
                        metadata_score,
                        rel_tol=0.0,
                        abs_tol=1e-10,
                    )
                )
                checks[f"{name}_metadata_score"] = score_matches
                if valid_report["passed"] and np.array_equal(valid_indices, expected_valid):
                    current_oof[name][expected_valid] = valid_values
                prior_key = f"{name}_oof"
                component_records[name] = {
                    "independent_log_loss": independent_score,
                    "metadata_log_loss": metadata_score,
                    "metadata_score_matches": score_matches,
                    "prior_same_rows_log_loss": multiclass_log_loss(
                        public_truth[expected_valid], prior[prior_key][expected_valid]
                    ),
                    "valid_probability": valid_report,
                    "test_probability": test_report,
                }
            if np.array_equal(valid_indices, expected_valid):
                write_counts[expected_valid] += 1
        fold_records.append(
            {
                "fold": fold,
                "fit_rows": len(expected_fit),
                "valid_rows": len(expected_valid),
                "metadata": {"path": str(metadata_path), "sha256": sha256_file(metadata_path)},
                "prediction": {
                    "path": str(prediction_path),
                    "sha256": sha256_file(prediction_path),
                    "allow_pickle": False,
                },
                "components": component_records,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    completed_mask = np.isin(stored_folds, np.asarray(completed_folds, dtype=np.int16))
    covered_mask = write_counts == 1
    aggregate_current: dict[str, float | None] = {}
    aggregate_prior: dict[str, float | None] = {}
    aggregate_delta: dict[str, float | None] = {}
    for name in COMPONENTS:
        usable = bool(
            completed_folds
            and np.array_equal(covered_mask, completed_mask)
            and np.isfinite(current_oof[name][completed_mask]).all()
        )
        current_score = (
            multiclass_log_loss(public_truth[completed_mask], current_oof[name][completed_mask])
            if usable
            else None
        )
        prior_score = (
            multiclass_log_loss(
                public_truth[completed_mask], prior[f"{name}_oof"][completed_mask]
            )
            if completed_folds
            else None
        )
        aggregate_current[name] = current_score
        aggregate_prior[name] = prior_score
        aggregate_delta[name] = (
            current_score - prior_score
            if current_score is not None and prior_score is not None
            else None
        )
    for name in ("byte", "candidate"):
        aggregate_prior[name] = (
            multiclass_log_loss(
                public_truth[completed_mask], prior[f"{name}_oof"][completed_mask]
            )
            if completed_folds
            else None
        )

    bronze_threshold = float(plan["promotion_gate"]["bronze_threshold_reference"])
    single_seed_threshold = float(
        plan["promotion_gate"]["single_seed_log_loss_threshold"]
    )
    component_bronze_signal = {
        name: score is not None and score <= bronze_threshold
        for name, score in aggregate_current.items()
    }
    global_checks = {
        "plan_schema": plan.get("schema") == EXPECTED_PLAN_SCHEMA,
        "manifest_schema": manifest.get("schema") == EXPECTED_MANIFEST_SCHEMA,
        "plan_frozen": plan.get("status") == "frozen_before_training",
        "plan_hash": manifest.get("plan_sha256") == plan_sha256,
        "cross_run_current_plan_hash": cross_run_plan.get("current", {})
        .get("plan", {})
        .get("sha256")
        == plan_sha256,
        "run_identity": manifest.get("run_id")
        == cross_run_plan.get("current", {}).get("run_id"),
        "public_train_hash": train_sha256
        == plan.get("inputs", {}).get("public_train_sha256"),
        "public_rows": len(train) == int(plan["inputs"]["train_rows"]),
        "class_order": class_order == CLASS_COLUMNS,
        "fold_assignment_hash": assignment_report["sha256"]
        == manifest.get("fold_assignment", {}).get("sha256"),
        "fold_ids_safe": assignment_report["train_id_safe_unicode"],
        "fold_ids_match_public": np.array_equal(stored_ids, public_ids),
        "truth_matches_public": np.array_equal(stored_truth, public_truth),
        "fold_domain": set(np.unique(stored_folds).tolist()) == set(range(expected_folds)),
        "duplicate_report": duplicate_comparable == duplicate_expected,
        "duplicate_group_isolation": duplicate_report["group_isolation"] is True,
        "prior_alignment": prior_alignment_passed,
        "has_completed_folds": bool(completed_folds),
        "completed_fold_artifacts": bool(fold_records)
        and all(record["passed"] for record in fold_records),
        "completed_coverage_exact_once": np.array_equal(covered_mask, completed_mask),
        "manifest_private_labels_unused": manifest.get("private_labels_used") is False,
        "manifest_grader_not_executed": manifest.get("official_grader_executed")
        is False,
        "manifest_kaggle_not_executed": manifest.get("kaggle_submission_executed")
        is False,
        "manifest_no_process_signals": manifest.get("process_signals_sent") == 0,
    }
    passed = all(global_checks.values())
    all_folds_complete = completed_folds == list(range(expected_folds))
    quality_verdict = (
        "AWAIT_FINAL_INDEPENDENT_VERIFIER"
        if all_folds_complete
        else "INTERIM_NO_SINGLE_COMPONENT_BRONZE_SIGNAL_CONTINUE_FULL_RUN"
        if not any(component_bronze_signal.values())
        else "INTERIM_COMPONENT_SIGNAL_CONTINUE_FULL_RUN"
    )
    report: dict[str, Any] = {
        "schema": "evomind.spooky.deberta_partial_quality_audit.v1",
        "created_at": now_iso(),
        "run_id": manifest.get("run_id"),
        "status": "passed_partial_audit" if passed else "failed_partial_audit",
        "phase": "mid_training_public_oof_diagnostic",
        "plan": {"path": str(plan_path), "sha256": plan_sha256},
        "cross_run_plan": {
            "path": str(cross_run_plan_path),
            "sha256": sha256_file(cross_run_plan_path),
        },
        "heartbeat_snapshot": {
            "status": heartbeat.get("status"),
            "fold": heartbeat.get("fold"),
            "epoch": heartbeat.get("epoch"),
            "update": heartbeat.get("update"),
            "total_updates": heartbeat.get("total_updates"),
            "process_signals_sent": heartbeat.get("process_signals_sent"),
        },
        "public_input": {
            "train": {"path": str(train_path), "sha256": train_sha256},
            "rows": len(train),
            "class_order": list(CLASS_COLUMNS),
            "labels_source": "public_train_csv_only",
        },
        "fold_assignment": assignment_report,
        "duplicate_group_report": duplicate_report,
        "completed_folds": completed_folds,
        "expected_folds": list(range(expected_folds)),
        "coverage": {
            "completed_rows": int(np.sum(completed_mask)),
            "total_rows": rows,
            "fraction": float(np.mean(completed_mask)),
            "write_count_min_completed": (
                int(write_counts[completed_mask].min()) if completed_mask.any() else 0
            ),
            "write_count_max": int(write_counts.max(initial=0)),
            "exact_once_completed": np.array_equal(covered_mask, completed_mask),
        },
        "fold_records": fold_records,
        "aggregate_completed_folds": {
            "current_component_log_loss": aggregate_current,
            "prior_same_rows_log_loss": aggregate_prior,
            "current_minus_prior": aggregate_delta,
        },
        "reference": {
            "bronze_threshold": bronze_threshold,
            "single_seed_threshold": single_seed_threshold,
            "prior_run_id": cross_run_plan["prior"]["run_id"],
            "prior_full_candidate_oof_log_loss": _safe_float(
                prior_summary.get("candidate_oof_log_loss")
            ),
            "prior_terminal_status": prior_summary.get("status"),
            "prior_alignment": prior_alignment,
        },
        "decision": {
            "quality_verdict": quality_verdict,
            "training_action": (
                "AWAIT_FINAL_VERIFIER"
                if all_folds_complete
                else "CONTINUE_TO_FULL_FIVE_FOLD_VERIFICATION"
            ),
            "promotion_action": "NO_PROMOTION_DECISION_FROM_PARTIAL_FOLDS",
            "meta_candidate_available": False,
            "component_bronze_signal": component_bronze_signal,
            "reason": (
                "The frozen cross-fit meta candidate requires all five outer folds; "
                "partial component metrics are diagnostic and cannot authorize promotion."
            ),
        },
        "global_checks": global_checks,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "human_gate_preserved": True,
        "claim_boundary": (
            "Completed public OOF folds only. This report is not a full-run candidate score, "
            "official MLE-Bench score, Kaggle score, rank, or medal claim."
        ),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cross-run-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AuditConfig(
        run_dir=args.run_dir,
        public_dir=args.public_dir,
        plan_path=args.plan,
        cross_run_plan_path=args.cross_run_plan,
        output_path=args.output,
    )
    report = audit(config)
    write_json_atomic(config.output_path.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed_partial_audit" else 3


if __name__ == "__main__":
    raise SystemExit(main())
