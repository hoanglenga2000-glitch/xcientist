#!/usr/bin/env python3
"""Independently verify a completed public-data SIIM candidate run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from scripts import queue_siim_final_candidate as queue
except ModuleNotFoundError:
    import queue_siim_final_candidate as queue

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = PROJECT_ROOT / "workspace" / "mlebench_plans" / "siim_final_candidate_frozen_plan.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def ensure_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Path is outside the verified run: {resolved}") from exc
    return resolved


def verify_artifact_manifest(task_dir: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    records = manifest.get("artifacts") or []
    checks: dict[str, bool] = {
        "schema": manifest.get("schema") == "evomind.siim_artifact_manifest.v1",
        "hash_algorithm": manifest.get("hash_algorithm") == "sha256",
        "artifact_count": manifest.get("artifact_count") == len(records) and bool(records),
    }
    seen: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        relative = str(record.get("path", ""))
        path = ensure_within(task_dir / relative, task_dir)
        unique = relative not in seen
        seen.add(relative)
        exists = path.is_file()
        size_ok = exists and path.stat().st_size == record.get("bytes")
        hash_ok = exists and sha256_file(path) == record.get("sha256")
        checks[f"artifact_{index:03d}"] = bool(relative and unique and size_ok and hash_ok)
        artifacts.append({
            "path": str(path),
            "role": record.get("role"),
            "exists": exists,
            "size_ok": size_ok,
            "hash_ok": hash_ok,
        })
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_sha256": sha256_file(manifest_path),
        "artifacts": artifacts,
    }


def verify_nested_isolation(payload: dict[str, Any], expected_outer: int, expected_inner: int) -> dict[str, Any]:
    folds = payload.get("folds") or []
    checks: dict[str, bool] = {
        "schema": payload.get("schema") == "evomind.siim_nested_patient_folds.v2",
        "outer_fold_count": payload.get("outer_fold_count") == expected_outer,
        "requested_inner_fold_count": payload.get("requested_inner_fold_count") == expected_inner,
        "all_inner_folds_aggregated": payload.get("all_inner_folds_aggregated") is True,
        "outer_validation_role": payload.get("outer_validation_role") == "final_oof_only",
        "fixed_epoch_outer_refit": payload.get("fixed_epoch_outer_refit") is True,
        "channel_specific_epoch_selection": payload.get("channel_specific_epoch_selection") is True,
        "fixed_iteration_metadata_outer_refit": (
            payload.get("fixed_iteration_metadata_outer_refit") is True
        ),
        "fold_record_count": len(folds) == expected_outer,
    }
    for outer_index, fold in enumerate(folds):
        roles = fold.get("roles") or {}
        outer_fit = set((roles.get("outer_fit") or {}).get("leakage_groups") or [])
        outer_valid = set((roles.get("outer_valid") or {}).get("leakage_groups") or [])
        refit = set((roles.get("refit") or {}).get("leakage_groups") or [])
        inner_folds = fold.get("inner_folds") or []
        checks[f"outer_{outer_index}_number"] = fold.get("outer_fold") == outer_index
        checks[f"outer_{outer_index}_fit_valid_disjoint"] = bool(
            outer_fit and outer_valid and outer_fit.isdisjoint(outer_valid)
        )
        checks[f"outer_{outer_index}_fresh_refit_scope"] = refit == outer_fit
        checks[f"outer_{outer_index}_all_inner_aggregated"] = (
            fold.get("all_inner_folds_aggregated") is True
        )
        checks[f"outer_{outer_index}_inner_count"] = len(inner_folds) == expected_inner
        inner_valid_union: set[str] = set()
        for inner_index, inner in enumerate(inner_folds):
            inner_roles = inner.get("roles") or {}
            inner_fit = set((inner_roles.get("inner_fit") or {}).get("leakage_groups") or [])
            inner_valid = set((inner_roles.get("inner_valid") or {}).get("leakage_groups") or [])
            inner_valid_union.update(inner_valid)
            prefix = f"outer_{outer_index}_inner_{inner_index}"
            checks[f"{prefix}_number"] = inner.get("inner_fold") == inner_index
            checks[f"{prefix}_disjoint"] = bool(
                inner_fit and inner_valid and inner_fit.isdisjoint(inner_valid)
            )
            checks[f"{prefix}_within_outer_fit"] = (
                inner_fit.issubset(outer_fit) and inner_valid.issubset(outer_fit)
            )
        checks[f"outer_{outer_index}_inner_valid_coverage"] = inner_valid_union == outer_fit
    return {"passed": all(checks.values()), "checks": checks}


def verify_oof(task_dir: Path, result: dict[str, Any], expected_rows: int) -> dict[str, Any]:
    path = task_dir / "siim_oof_predictions.csv"
    frame = pd.read_csv(path)
    required = {
        "image_name",
        "patient_id",
        "leakage_group",
        "target",
        "fold",
        "blended_probability",
    }
    prediction = frame.get("blended_probability", pd.Series(dtype=float)).to_numpy(dtype=float)
    truth = frame.get("target", pd.Series(dtype=float)).to_numpy(dtype=float)
    auc = float(roc_auc_score(truth, prediction)) if len(frame) and set(np.unique(truth)) == {0.0, 1.0} else None
    group_fold_max = (
        int(frame.groupby("leakage_group")["fold"].nunique().max())
        if required.issubset(frame.columns) and len(frame)
        else None
    )
    nonempty_patient = frame[frame.get("patient_id", "").astype(str).str.len() > 0]
    patient_fold_max = (
        int(nonempty_patient.groupby("patient_id")["fold"].nunique().max())
        if len(nonempty_patient)
        else 1
    )
    expected_auc = result.get("cv_score")
    checks = {
        "required_columns": required.issubset(frame.columns),
        "row_count": len(frame) == expected_rows,
        "unique_image_ids": frame.get("image_name", pd.Series(dtype=str)).nunique() == len(frame),
        "finite_probability": len(prediction) == len(frame) and bool(np.isfinite(prediction).all()),
        "probability_range": len(prediction) == len(frame)
        and bool(((prediction >= 0.0) & (prediction <= 1.0)).all()),
        "binary_target": set(np.unique(truth)) == {0.0, 1.0},
        "fold_coverage": sorted(frame.get("fold", pd.Series(dtype=int)).unique().tolist())
        == list(range(int(result.get("budget", {}).get("folds", 0)))),
        "leakage_group_single_fold": group_fold_max == 1,
        "patient_single_fold": patient_fold_max == 1,
        "auc_recomputed": auc is not None
        and expected_auc is not None
        and abs(auc - float(expected_auc)) <= 1e-12,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "path": str(path),
        "sha256": sha256_file(path),
        "recomputed_auc": auc,
        "reported_auc": expected_auc,
        "leakage_group_max_fold_count": group_fold_max,
        "patient_max_fold_count": patient_fold_max,
    }


def verify_submission_rebuild(
    task_dir: Path,
    sample_submission_path: Path,
    expected_rows: int,
) -> dict[str, Any]:
    submission_path = task_dir / "submission.csv"
    components_path = task_dir / "siim_test_components.csv"
    ensemble_path = task_dir / "siim_fold_ensemble.npz"
    submission = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_submission_path)
    components = pd.read_csv(components_path)
    with np.load(ensemble_path, allow_pickle=False) as arrays:
        test_id = arrays["test_id"].astype(str)
        test_probability = arrays["test_probability"].astype(float)
        pure_by_fold = arrays["pure_image_test_by_fold"]
        lesion_by_fold = arrays["lesion_focus_test_by_fold"]
        fusion_by_fold = arrays["image_metadata_fusion_test_by_fold"]
        metadata_by_fold = arrays["metadata_catboost_test_by_fold"]
    rebuilt = pd.DataFrame({"image_name": test_id, "target": test_probability})
    rebuilt = sample[["image_name"]].merge(rebuilt, on="image_name", how="left", validate="one_to_one")
    joined = submission[["image_name", "target"]].merge(
        rebuilt,
        on="image_name",
        how="outer",
        suffixes=("_actual", "_rebuilt"),
        validate="one_to_one",
    )
    component_map = components.set_index("image_name")["blended_probability"]
    component_reordered = submission["image_name"].map(component_map).to_numpy(dtype=float)
    checks = {
        "row_count": len(submission) == expected_rows == len(sample) == len(components),
        "schema": submission.columns.tolist() == sample.columns.tolist() == ["image_name", "target"],
        "unique_ids": submission["image_name"].nunique() == len(submission),
        "sample_id_order": submission["image_name"].astype(str).tolist()
        == sample["image_name"].astype(str).tolist(),
        "finite": bool(np.isfinite(submission["target"].to_numpy(dtype=float)).all()),
        "range": bool(
            (
                (submission["target"].to_numpy(dtype=float) >= 0.0)
                & (submission["target"].to_numpy(dtype=float) <= 1.0)
            ).all()
        ),
        "npz_id_count": len(test_id) == expected_rows and len(set(test_id.tolist())) == expected_rows,
        "npz_rebuild": len(joined) == expected_rows
        and bool(
            np.allclose(
                joined["target_actual"].to_numpy(dtype=float),
                joined["target_rebuilt"].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "component_rebuild": bool(
            np.allclose(
                submission["target"].to_numpy(dtype=float),
                component_reordered,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "fold_array_shapes": all(
            array.ndim == 2 and array.shape[1] == expected_rows
            for array in (pure_by_fold, lesion_by_fold, fusion_by_fold, metadata_by_fold)
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "submission_path": str(submission_path),
        "submission_sha256": sha256_file(submission_path),
        "fold_ensemble_sha256": sha256_file(ensemble_path),
    }


def verify_run(plan: dict[str, Any], run_dir: Path, queue_status_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    result_path = run_dir / plan["competition_id"] / "result.json"
    result = read_json(result_path)
    task_dir = ensure_within(Path(result["attempt_dir"]), run_dir)
    attempt_result = read_json(task_dir / "result.json")
    manifest = read_json(run_dir / "manifest.json")
    queue_status = read_json(queue_status_path.resolve())
    ablation = read_json(Path(plan["serial_dependency"]["report"]).resolve())
    ablation_gate = read_json(task_dir / "siim_preprocessing_ablation_gate.json")
    nested_path = task_dir / "siim_nested_patient_folds.json"
    nested = verify_nested_isolation(
        read_json(nested_path),
        expected_outer=int(plan["training"]["outer_folds"]),
        expected_inner=int(plan["training"]["inner_folds"]),
    )
    artifact_path = task_dir / "siim_artifact_manifest.json"
    artifacts = verify_artifact_manifest(task_dir, artifact_path)
    oof = verify_oof(task_dir, result, int(plan["data_contract"]["train_rows"]))
    submission = verify_submission_rebuild(
        task_dir,
        Path(plan["data_contract"]["sample_submission"]).resolve(),
        int(plan["data_contract"]["test_rows"]),
    )

    promotion = result.get("promotion_gate") or {}
    fold_records = result.get("folds") or []
    checkpoint_hashes_ok = all(
        sha256_file(task_dir / f"siim_fusion_fold_{index + 1:02d}.pt")
        == record.get("vision_checkpoint_sha256")
        and sha256_file(task_dir / f"siim_metadata_fold_{index + 1:02d}.cbm")
        == record.get("metadata_model_sha256")
        for index, record in enumerate(fold_records)
    )
    fresh_refit_ok = all(
        record.get("outer_validation_role") == "final_oof_only"
        and record.get("refit_epoch_count") == record.get("selected_epoch")
        and record.get("inner_fold_count") == plan["training"]["inner_folds"]
        and record.get("refit_seed") not in (record.get("selection_seeds") or [])
        and record.get("preprocessing_profile") == ablation.get("selected_profile")
        for record in fold_records
    )
    expected_code_hashes = {
        str(Path(plan["implementation"][name]["path"]).resolve()): plan["implementation"][name][
            "sha256"
        ]
        for name in ("full_runner", "wave0", "wave2", "adapter")
    }
    manifest_hashes = manifest.get("code_sha256") or {}
    code_hashes_ok = all(manifest_hashes.get(path) == digest for path, digest in expected_code_hashes.items())
    status_ok = result.get("status") in {
        "promotion_gate_passed_confirmation_pending",
        "promotion_gate_failed",
    }
    candidate_ready = result.get("status") == "promotion_gate_passed_confirmation_pending"
    gate_consistent = (
        promotion.get("passed") is (float(oof["recomputed_auc"]) >= float(plan["objective"]["promotion_auc"]))
        if oof["recomputed_auc"] is not None
        else False
    )
    checks = {
        "result_current_matches_attempt": result == attempt_result,
        "competition": result.get("competition_id") == plan["competition_id"],
        "terminal_candidate_status": status_ok,
        "candidate_only": result.get("candidate_only") is True,
        "official_grader_not_executed": result.get("official_grader_executed") is False,
        "official_grader_withheld": result.get("official_grader_withheld") is True,
        "private_grader_score_absent": result.get("mle_private_grader_score") is None,
        "kaggle_scores_absent": result.get("kaggle_public_score") is None
        and result.get("kaggle_private_score") is None,
        "manifest_human_gate": manifest.get("human_gate_preserved") is True,
        "manifest_candidate_only": manifest.get("candidate_only") is True,
        "manifest_cuda_lease": manifest.get("hold_cuda_lease") is True,
        "manifest_code_hashes": code_hashes_ok,
        "queue_no_signals": queue_status.get("process_signals_sent") == 0,
        "queue_no_grader": queue_status.get("official_grader_executed") is False,
        "queue_no_kaggle": queue_status.get("kaggle_submission_executed") is False,
        "ablation_selected_profile": ablation_gate.get("validated") is True
        and ablation_gate.get("selected_profile") == ablation.get("selected_profile"),
        "ablation_report_hash": ablation_gate.get("report_sha256")
        == sha256_file(Path(plan["serial_dependency"]["report"]).resolve()),
        "nested_isolation": nested["passed"],
        "artifact_manifest": artifacts["passed"],
        "artifact_manifest_hash_in_result": result.get("artifact_manifest_sha256")
        == sha256_file(artifact_path),
        "oof_recomputed": oof["passed"],
        "submission_rebuilt": submission["passed"],
        "checkpoint_hashes": checkpoint_hashes_ok,
        "nested_selection_fresh_refit": fresh_refit_ok,
        "promotion_gate_consistent": gate_consistent,
        "no_private_labels_in_plan": plan["training"]["private_labels_used"] is False,
    }
    passed = all(checks.values())
    return {
        "schema": "evomind.siim.final_candidate_independent_verification.v1",
        "created_at": now_iso(),
        "status": (
            "verification_passed_confirmation_pending"
            if passed and candidate_ready
            else "verified_promotion_gate_failed"
            if passed
            else "verification_failed"
        ),
        "passed": passed,
        "candidate_ready": candidate_ready and passed,
        "competition_id": plan["competition_id"],
        "run_dir": str(run_dir),
        "task_dir": str(task_dir),
        "plan_path": plan["_plan_path"],
        "plan_sha256": plan["_plan_sha256"],
        "checks": checks,
        "oof": oof,
        "nested": nested,
        "artifacts": artifacts,
        "submission": submission,
        "promotion_gate": promotion,
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "claim_boundary": (
            "Verified public-data candidate only; official MLE-Bench grading remains pending "
            "explicit human confirmation."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--queue-status", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = queue.validate_frozen_plan(args.plan)
    run_dir = args.run_dir or (
        Path(plan["training"]["output_root"]) / plan["training"]["run_id"]
    )
    queue_status = args.queue_status or Path(plan["launch_contract"]["queue_status"])
    output = args.output or Path(run_dir) / "independent_verification.json"
    report = verify_run(plan, Path(run_dir), Path(queue_status))
    write_json_atomic(Path(output).resolve(), report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
