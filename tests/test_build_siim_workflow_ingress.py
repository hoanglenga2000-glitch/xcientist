from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from research_os.agent.siim_hpc_workflow import run_siim_hpc_research
from scripts import build_siim_workflow_ingress as ingress
from xsci.user_request import parse_user_request


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write(path: Path, value: bytes = b"evidence\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict:
    return {"path": str(path.resolve()), "local": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _run_dir(root: Path, run_id: str) -> Path:
    run_dir = root / "workspace" / "evomind_runs" / run_id
    _write_json(run_dir / "request.json", {"objective": "SIIM fixture"})
    return run_dir


def _preflight_fixture(root: Path, run_id: str, *, passed: bool = True) -> tuple[Path, Path]:
    campaign = root / "workspace" / "hpc" / "job89508_siim_campaign" / run_id
    runtime_path = root / "workspace" / "hpc" / "job89508_siim_runtime" / "verify_current.json"
    created_at = datetime.now(timezone.utc).isoformat()
    bundle_sha = "a" * 64
    requirements_sha = "b" * 64
    plan = {
        "schema": "evomind.siim.hpc_campaign_plan.v1",
        "run_id": run_id,
        "job_id": 89508,
        "credential_profile": "job89508",
        "remote_root": ingress.REMOTE_ROOT,
        "bundle_sha256": bundle_sha,
        "runtime_requirements_sha256": requirements_sha,
        "dataset": {
            "competition_id": ingress.TASK_ID,
            "manifest_sha256": ingress.EXPECTED_DATASET_MANIFEST_SHA256,
            "file_count": 33_129,
            "total_bytes": 25_765_345_055,
            "train_images": 28_984,
            "test_images": 4_142,
            "positive_rows": 513,
            "patients": 2_056,
        },
        "ablation_seeds": [40, 41, 42],
        "formal_seeds": [43, 44, 45],
        "formal_runs": [
            {"seed": seed, "run_id": f"{run_id}_s{seed}"} for seed in (43, 44, 45)
        ],
        "resource_policy": {
            "memory_limit_mib": 55 * 1024,
            "effective_batch_size": 384,
            "max_workers": 8,
            "other_processes_modified": False,
            "signals_sent": 0,
        },
        "official_submission": "forbidden",
        "private_grader": "once_after_candidate_freeze",
        "post_grader_tuning": "forbidden",
    }
    sample = {
        "captured_at_epoch": 1,
        "compute_apps": [],
        "gpus": [
            {
                "name": "NVIDIA A800-SXM4-80GB",
                "uuid": "GPU-fixture",
                "memory_total_mib": 81_920,
                "memory_used_mib": 1_920,
                "memory_free_mib": 80_000,
                "utilization_percent": 0,
            }
        ],
        "eligible": passed,
        "probe_errors": [],
        "hold_reasons": [] if passed else ["fixture_hold"],
        "other_processes_modified": False,
    }
    identity = {
        "host_uuid": "host-fixture",
        "gpu_uuids": ["gpu-fixture"],
        "expected_host_uuid_bound": True,
        "expected_gpu_uuid_bound": True,
        "stable": True,
    }
    gate = {
        "schema": "evomind.hpc_gpu_resource_gate.v2",
        "created_at": created_at,
        "remote_root": ingress.REMOTE_ROOT,
        "credential_profile": "job89508",
        "policy": {"samples_required": 5},
        "samples": [dict(sample) for _ in range(5)],
        "identity": identity,
        "hold_reasons": [] if passed else ["fixture_hold"],
        "read_only_gate_passed": passed,
        "dedicated_root_writable": True,
        "other_processes_modified": False,
        "signals_sent": 0,
        "passed": passed,
    }
    remote = {
        "campaign": f"{ingress.REMOTE_ROOT}/siim_job89508/campaigns/{run_id}",
        "bundle": f"{ingress.REMOTE_ROOT}/siim_job89508/bundles/{bundle_sha}",
        "runtime_verification": f"{ingress.REMOTE_ROOT}/siim_job89508/runtime/{requirements_sha}/verification.json",
    }
    deployment = {
        "schema": "evomind.siim.job89508.deployment.v1",
        "run_id": run_id,
        "status": "deployed_not_started",
        "job_id": 89508,
        "credential_profile": "job89508",
        "remote_root": ingress.REMOTE_ROOT,
        "bundle_sha256": bundle_sha,
        "remote": remote,
        "import_smoke": ["import_smoke=passed"],
        "dataset_uploaded": False,
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    launch = {
        "schema": "evomind.siim.job89508.launch.v1",
        "run_id": run_id,
        "status": "started",
        "job_id": 89508,
        "credential_profile": "job89508",
        "action": "new_supervisor_started",
        "supervisor_pid": 1234,
        "gate": {
            "passed": True,
            "created_at": created_at,
            "age_seconds": 1,
            "host_uuid": identity["host_uuid"],
            "gpu_uuids": identity["gpu_uuids"],
        },
        "bundle_sha256": bundle_sha,
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    runtime = {
        "schema": "evomind.siim.isolated_runtime_verification.v1",
        "status": "passed",
        "job_id": 89508,
        "credential_profile": "job89508",
        "requirements_sha256": requirements_sha,
        "python": "3.11.15",
        "versions": {"torch": "2.5.1+cu118"},
        "cuda": {"available": True, "device": "NVIDIA A800-SXM4-80GB", "bf16": True},
        "catboost_smoke": True,
        "signals_sent": 0,
        "other_processes_modified": False,
    }
    _write_json(campaign / "campaign_plan.json", plan)
    _write_json(campaign / "gpu_gate.json", gate)
    _write_json(campaign / "deployment.json", deployment)
    _write_json(campaign / "launch.json", launch)
    _write_json(runtime_path, runtime)
    return campaign, runtime_path


def test_hold_gate_fails_before_any_ingress_is_written(tmp_path):
    run_id = "siim_hold_fixture"
    run_dir = _run_dir(tmp_path, run_id)
    campaign, runtime = _preflight_fixture(tmp_path, run_id, passed=False)

    with pytest.raises(ingress.SiimWorkflowIngressError, match="GPU gate is HOLD"):
        ingress.stage_preflight_ingress(
            tmp_path,
            run_id,
            campaign_dir=campaign,
            runtime_verification_path=runtime,
            resume=False,
        )

    assert not (run_dir / "ingress").exists()


def test_preflight_rejects_runtime_from_a_different_allocation(tmp_path):
    run_id = "siim_cross_allocation_runtime"
    run_dir = _run_dir(tmp_path, run_id)
    campaign, runtime = _preflight_fixture(tmp_path, run_id)
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    payload["job_id"] = 90353
    payload["credential_profile"] = "job90353"
    _write_json(runtime, payload)

    with pytest.raises(ingress.SiimWorkflowIngressError, match="runtime job binding"):
        ingress.stage_preflight_ingress(
            tmp_path,
            run_id,
            campaign_dir=campaign,
            runtime_verification_path=runtime,
            resume=False,
        )

    assert not (run_dir / "ingress").exists()


def test_preflight_ingress_is_hash_bound_idempotent_and_drift_rejecting(tmp_path):
    run_id = "siim_preflight_fixture"
    run_dir = _run_dir(tmp_path, run_id)
    campaign, runtime = _preflight_fixture(tmp_path, run_id)

    first = ingress.stage_preflight_ingress(
        tmp_path,
        run_id,
        campaign_dir=campaign,
        runtime_verification_path=runtime,
        resume=False,
    )
    second = ingress.stage_preflight_ingress(
        tmp_path,
        run_id,
        campaign_dir=campaign,
        runtime_verification_path=runtime,
        resume=False,
    )

    assert first["status"] == second["status"] == "staged"
    manifest = json.loads(
        (run_dir / "ingress" / "workflow_ingress_manifest.json").read_text(encoding="utf-8")
    )
    records = {
        item["path"]: item for item in manifest["stages"]["preflight"]["staged_artifacts"]
    }
    assert set(records) == {"hpc_preflight.json", "dataset_profile.json"}
    for relative, record in records.items():
        path = run_dir / "ingress" / relative
        assert record["bytes"] == path.stat().st_size
        assert record["sha256"] == _sha256(path)

    (run_dir / "ingress" / "dataset_profile.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ingress.SiimWorkflowIngressError, match="existing ingress bytes drifted"):
        ingress.stage_preflight_ingress(
            tmp_path,
            run_id,
            campaign_dir=campaign,
            runtime_verification_path=runtime,
            resume=False,
        )


def test_candidate_manifest_rejects_hash_drift(tmp_path):
    run_id = "siim_candidate_manifest_fixture"
    candidate = tmp_path / "candidate"
    names = (
        "candidate_submission_withheld.csv",
        "ensemble_oof_predictions.csv",
        "fold_metrics.csv",
        "metrics.json",
        "frozen_plan.json",
        "candidate_freeze.json",
        "independent_verification.json",
    )
    files = [_write(candidate / name, f"{name}\n".encode()) for name in names]
    _write_json(
        candidate / "artifact_manifest.json",
        {
            "schema": "evomind.siim.frozen_artifact_manifest.v1",
            "run_id": run_id,
            "artifacts": [
                {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in files
            ],
            "artifact_count": len(files),
            "all_sha256_bound": True,
        },
    )

    indexed = ingress.validate_candidate_manifest(candidate, run_id)
    assert str((candidate / "metrics.json").resolve()) in indexed

    (candidate / "metrics.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ingress.SiimWorkflowIngressError, match="byte count changed|SHA-256 changed"):
        ingress.validate_candidate_manifest(candidate, run_id)


def test_campaign_evidence_builds_real_counts_without_inventing_metrics(tmp_path):
    run_id = "siim_campaign_fixture"
    initial = run_siim_hpc_research(
        tmp_path,
        parse_user_request(
            "请使用 SIIM-ISIC Melanoma Classification 皮肤镜图像，在 HPC A800 上训练，"
            "按患者分组验证、独立复核，并生成报告和证据包；不提交公开榜单。"
        ),
        run_id=run_id,
    )
    assert initial.status == "needs_continuation"
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    campaign, _runtime = _preflight_fixture(tmp_path, run_id)
    preflight = ingress.stage_preflight_ingress(tmp_path, run_id, campaign_dir=campaign, resume=True)
    assert preflight["status"] == "staged"
    plan_path = campaign / "campaign_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    collected_root = tmp_path / "workspace" / "hpc" / "mlebench_remote_ops" / "collected"
    sample_rows = [{"image_name": f"test_{index:05d}", "target": 0.0} for index in range(4_142)]
    sample = _write_csv(
        tmp_path / "sample_submission.csv",
        ["image_name", "target"],
        sample_rows,
    )

    seed_collections = []
    seed_records = []
    for seed in (43, 44, 45):
        formal_id = f"{run_id}_s{seed}"
        formal_root = collected_root / formal_id
        task = formal_root / ingress.TASK_ID
        result = _write_json(
            task / "result.json",
            {
                "competition_id": ingress.TASK_ID,
                "status": "promotion_gate_failed" if seed == 43 else "passed",
                "metric": "roc_auc",
                "direction": "maximize",
                "valid_submission": True,
                "official_grader_executed": False,
                "mle_private_grader_score": None,
                "kaggle_public_score": None,
                "kaggle_private_score": None,
                "budget": {"seed": seed},
                "cv_score": 0.9,
            },
        )
        bundle = _write(task / "siim_fold_ensemble.npz", f"bundle-{seed}\n".encode())
        seed_oof = _write(task / "siim_oof_predictions.csv", f"seed-oof-{seed}\n".encode())
        seed_submission = _write(task / "submission.csv", f"seed-submission-{seed}\n".encode())
        history = _write_json(
            task / "siim_training_history.json",
            {
                "performance": {
                    "physical_batch_size": 96,
                    "effective_batch_size": 384,
                    "dataloader_workers": 8,
                    "channels_last": True,
                    "amp_dtype": "bfloat16",
                    "preprocessing_profile": "raw_multiview_v1",
                    "preprocessing_ablation": {"validated": True},
                },
                "folds": [{"fold": fold} for fold in range(5)],
            },
        )
        nested = _write_json(
            task / "siim_nested_patient_folds.json",
            {
                "schema": "evomind.siim_nested_patient_folds.v2",
                "outer_fold_count": 5,
                "requested_inner_fold_count": 3,
                "all_inner_folds_aggregated": True,
                "outer_validation_role": "final_oof_only",
            },
        )
        collected_files = [result, bundle, seed_oof, seed_submission, history, nested]
        collection = {
            "schema": "evomind.mlebench_remote_ops.collection.v1",
            "run_id": formal_id,
            "remote_run": f"{ingress.REMOTE_ROOT}/{formal_id}",
            "local_root": str(formal_root.resolve()),
            "include_checkpoints": False,
            "file_count": len(collected_files),
            "files": [_record(path) for path in collected_files],
            "passed": True,
        }
        _write_json(formal_root / "collection_manifest.json", collection)
        seed_collections.append(collection)
        seed_records.append(
            {
                "model_seed": seed,
                "run_id": formal_id,
                "oof_roc_auc": 0.9,
                "result": _record(result),
                "prediction_bundle": _record(bundle),
                "oof_predictions": _record(seed_oof),
                "submission": _record(seed_submission),
            }
        )

    evidence = campaign / "remote_evidence"
    _write_json(evidence / "campaign_plan.json", plan)
    _write_json(
        evidence / "campaign_state.json",
        {
            "schema": "evomind.siim.job89508.campaign_state.v1",
            "run_id": run_id,
            "job_id": 89508,
            "credential_profile": "job89508",
            "status": "awaiting_collection_and_freeze",
            "completed_seeds": [43, 44, 45],
            "selected_profile": "raw_multiview_v1",
            "selected_batch_size": 96,
            "signals_sent": 0,
            "other_processes_modified": False,
        },
    )
    telemetry = evidence / "hpc_telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "schema": "evomind.siim.hpc_telemetry.v1",
                "run_id": run_id,
                "job_id": 89508,
                "credential_profile": "job89508",
                "gpu": {"name": "NVIDIA A800-SXM4-80GB"},
                "signals_sent": 0,
                "other_processes_modified": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ablation_root = evidence / "ablation" / "runs" / f"{run_id}_ablation"
    image_manifest = _write(ablation_root / "siim_image_content_manifest.csv", b"source,image_name\n")
    profile_names = list(ingress.PROFILE_MAP)
    profiles = []
    for offset, name in enumerate(profile_names):
        profiles.append(
            {
                "profile": name,
                "seed_fold_auc": {
                    str(seed): [0.80 + offset / 100, 0.81 + offset / 100, 0.82 + offset / 100]
                    for seed in (40, 41, 42)
                },
                "mean_auc": 0.81 + offset / 100,
                "worst_fold_auc": 0.80 + offset / 100,
                "mean_gain": None if offset == 0 else 0.01,
                "worst_fold_delta": None if offset == 0 else 0.01,
                "seed_stability_pass_count": 3,
                "eligible_for_selection": offset == 0,
            }
        )
    _write_json(
        ablation_root / "siim_preprocessing_ablation.json",
        {
            "schema": "evomind.siim_preprocessing_ablation.v1",
            "passed": True,
            "selected_profile": "raw_multiview_v1",
            "profile_order": profile_names,
            "profiles": profiles,
            "fold_count": 3,
            "evaluation_seed_count": 3,
            "evaluation_seeds": [40, 41, 42],
            "score_count_per_profile": 9,
            "leakage_group_policy": "patient_exact_file_decoded_pixel_v2",
            "perceptual_edge_policy": "audit_only_no_cross_patient_union_v1",
            "image_content_manifest_sha256": _sha256(image_manifest),
            "private_labels_used_for_training": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
    )
    _write_json(
        ablation_root / "siim_duplicate_connected_groups.json",
        {
            "schema": "evomind.siim_patient_content_connected_groups.v3",
            "rows": 28_984,
            "private_labels_used": False,
        },
    )
    _write_json(
        campaign / "collection.json",
        {
            "schema": "evomind.siim.job89508.collection.v1",
            "run_id": run_id,
            "job_id": 89508,
            "credential_profile": "job89508",
            "status": "collected",
            "seed_collections": seed_collections,
            "remote_evidence_root": str(evidence.resolve()),
            "signals_sent": 0,
            "other_processes_modified": False,
        },
    )

    row_count = 28_984
    index = np.arange(row_count)
    patient_index = index % 2_056
    folds = patient_index % 5
    truth = (index < 513).astype(int)
    probability = np.where(truth == 1, 0.8 + (index % 10) / 100, 0.05 + (index % 20) / 100)
    oof = pd.DataFrame(
        {
            "image_name": [f"train_{value:05d}" for value in index],
            "patient_id": [f"patient_{value:04d}" for value in patient_index],
            "leakage_group": [f"group_{value:04d}" for value in patient_index],
            "target": truth,
            "fold": folds,
            "probability": probability,
        }
    )
    candidate = tmp_path / "workspace" / "siim_job89508" / "candidates" / run_id
    candidate.mkdir(parents=True)
    oof_path = candidate / "ensemble_oof_predictions.csv"
    oof.to_csv(oof_path, index=False)
    candidate_submission = candidate / "candidate_submission_withheld.csv"
    pd.DataFrame(
        {
            "image_name": [row["image_name"] for row in sample_rows],
            "target": np.linspace(0.01, 0.99, 4_142),
        }
    ).to_csv(candidate_submission, index=False)
    fold_rows = []
    for fold in range(5):
        mask = folds == fold
        fold_rows.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                "positives": int(truth[mask].sum()),
                "roc_auc": float(roc_auc_score(truth[mask], probability[mask])),
                "pr_auc": float(average_precision_score(truth[mask], probability[mask])),
                "brier": float(brier_score_loss(truth[mask], probability[mask])),
            }
        )
    fold_path = candidate / "fold_metrics.csv"
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    metrics_payload = {
        "schema": "evomind.siim.multiseed_metrics.v1",
        "run_id": run_id,
        "roc_auc": float(roc_auc_score(truth, probability)),
        "pr_auc": float(average_precision_score(truth, probability)),
        "brier": float(brier_score_loss(truth, probability)),
        "private_grader_execution_count": 0,
        "kaggle_submission_executed": False,
        "clinical_diagnosis_claimed": False,
    }
    metrics_path = _write_json(candidate / "metrics.json", metrics_payload)
    frozen_plan = candidate / "frozen_plan.json"
    frozen_plan.write_bytes(plan_path.read_bytes())
    freeze = _write_json(
        candidate / "candidate_freeze.json",
        {
            "schema": "evomind.siim.candidate_freeze.v1",
            "run_id": run_id,
            "status": "frozen_before_private_grader",
            "formal_seeds": [43, 44, 45],
            "ablation_seeds": [40, 41, 42],
            "seed_sets_disjoint": True,
            "seed_records": seed_records,
            "candidate_submission": _record(candidate_submission),
            "ensemble_oof_predictions": _record(oof_path),
            "metrics": _record(metrics_path),
            "private_grader_execution_count_before_freeze": 0,
            "official_submission_executed": False,
            "candidate_hash_bound": True,
        },
    )
    independent = _write_json(
        candidate / "independent_verification.json",
        {
            "schema": "evomind.siim.multiseed_independent_verification.v1",
            "run_id": run_id,
            "status": "passed",
            "passed": True,
            "train_rows": 28_984,
            "test_rows": 4_142,
            "positive_rows": 513,
            "oof_coverage_exactly_once": True,
            "patient_content_grouping_bound": True,
            "train_id_order_verified": True,
            "test_id_order_verified": True,
            "submission_schema_verified": True,
            "formal_ablation_seed_separation_verified": True,
            "recomputed_metrics": metrics_payload,
            "candidate_freeze_sha256": _sha256(freeze),
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "other_processes_modified": False,
            "signals_sent": 0,
        },
    )
    material = [candidate_submission, oof_path, fold_path, metrics_path, frozen_plan, freeze, independent]
    _write_json(
        candidate / "artifact_manifest.json",
        {
            "schema": "evomind.siim.frozen_artifact_manifest.v1",
            "run_id": run_id,
            "artifacts": [_record(path) for path in material],
            "artifact_count": len(material),
            "all_sha256_bound": True,
        },
    )

    result = ingress.stage_campaign_ingress(
        tmp_path,
        run_id,
        campaign_dir=campaign,
        candidate_root=candidate,
        collected_root=collected_root,
        sample_submission=sample,
        resume=True,
    )

    assert result["status"] == "staged"
    audit = json.loads((run_dir / "ingress" / "data_audit.json").read_text(encoding="utf-8"))
    staged_metrics = json.loads((run_dir / "ingress" / "training" / "metrics.json").read_text(encoding="utf-8"))
    assert audit["train_rows"] == 28_984
    assert audit["patients"] == 2_056
    assert audit["patient_group_overlap"] == audit["content_group_overlap"] == 0
    assert staged_metrics["roc_auc"] == pytest.approx(metrics_payload["roc_auc"], abs=1e-12)
    assert staged_metrics["mle_private_grader_score"] is None
    assert result["review_freeze"]["candidate_freeze_sha256"] == _sha256(
        run_dir / "candidate_freeze.json"
    )
    workflow = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert workflow["tasks"]["independent_review_freeze"]["status"] == "completed"
    assert workflow["tasks"]["terminal_private_grader"]["status"] == "failed"


def test_three_fold_ablation_ingress_requires_exact_finite_fold_arrays(tmp_path):
    image_manifest = _write(tmp_path / "siim_image_content_manifest.csv", b"source,image_name\n")
    profile_names = list(ingress.PROFILE_MAP)
    profiles = [
        {
            "profile": name,
            "seed_fold_auc": {
                str(seed): [
                    0.80 + offset / 100,
                    0.81 + offset / 100,
                    0.82 + offset / 100,
                ]
                for seed in (40, 41, 42)
            },
            "mean_auc": 0.81 + offset / 100,
            "worst_fold_auc": 0.80 + offset / 100,
            "mean_gain": None if offset == 0 else 0.01,
            "worst_fold_delta": None if offset == 0 else 0.01,
            "seed_stability_pass_count": 3,
            "eligible_for_selection": offset == 0,
        }
        for offset, name in enumerate(profile_names)
    ]
    ablation = {
        "schema": "evomind.siim_preprocessing_ablation.v1",
        "passed": True,
        "selected_profile": "raw_multiview_v1",
        "profile_order": profile_names,
        "profiles": profiles,
        "fold_count": 3,
        "evaluation_seed_count": 3,
        "evaluation_seeds": [40, 41, 42],
        "score_count_per_profile": 9,
        "leakage_group_policy": "patient_exact_file_decoded_pixel_v2",
        "perceptual_edge_policy": "audit_only_no_cross_patient_union_v1",
        "image_content_manifest_sha256": _sha256(image_manifest),
        "private_labels_used_for_training": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }

    normalized, selected = ingress._validate_ablation(ablation, image_manifest)

    assert ingress.MIN_SIIM_ABLATION_FOLDS == 3
    assert normalized["fold_count"] == ingress.MIN_SIIM_ABLATION_FOLDS
    assert normalized["score_count_per_profile"] == 9
    assert selected == "raw_multiview_v1"

    mismatched = json.loads(json.dumps(ablation))
    mismatched["profiles"][-1]["seed_fold_auc"]["42"].append(0.82)
    with pytest.raises(ingress.SiimWorkflowIngressError, match="fold evidence is invalid"):
        ingress._validate_ablation(mismatched, image_manifest)

    non_finite = json.loads(json.dumps(ablation))
    non_finite["profiles"][0]["seed_fold_auc"]["40"][1] = float("nan")
    with pytest.raises(ingress.SiimWorkflowIngressError, match="fold evidence is invalid"):
        ingress._validate_ablation(non_finite, image_manifest)


def test_private_grader_must_bind_exact_workflow_freeze(tmp_path):
    run_id = "siim_private_grader_fixture"
    run_dir = _run_dir(tmp_path, run_id)
    frozen_artifact = _write(run_dir / "metrics.json", b'{"roc_auc":0.9}\n')
    freeze = _write_json(
        run_dir / "candidate_freeze.json",
        {
            "schema": "evomind.siim.candidate_freeze.v1",
            "run_id": run_id,
            "status": "frozen_before_private_grader",
            "tuning_closed": True,
            "artifacts": [
                {
                    "path": "metrics.json",
                    "bytes": frozen_artifact.stat().st_size,
                    "sha256": _sha256(frozen_artifact),
                }
            ],
        },
    )
    grader = _write_json(
        tmp_path / "private_grader.json",
        {
            "schema": "evomind.siim.private_grader.v1",
            "run_id": run_id,
            "status": "passed",
            "execution_index": 1,
            "candidate_freeze_sha256": "f" * 64,
            "executed_after_freeze": True,
            "feedback_used_for_tuning": False,
            "official_submission_executed": False,
            "mle_private_grader_score": 0.92,
        },
    )

    with pytest.raises(ingress.SiimWorkflowIngressError, match="different freeze"):
        ingress.stage_private_grader_ingress(tmp_path, run_id, grader, resume=False)
    assert not (run_dir / "ingress" / "private_grader.json").exists()

    payload = json.loads(grader.read_text(encoding="utf-8"))
    payload["candidate_freeze_sha256"] = _sha256(freeze)
    _write_json(grader, payload)
    result = ingress.stage_private_grader_ingress(tmp_path, run_id, grader, resume=False)
    assert result["status"] == "staged"
    assert _sha256(run_dir / "ingress" / "private_grader.json") == _sha256(grader)
