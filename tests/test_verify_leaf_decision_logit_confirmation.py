from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts import run_leaf_decision_logit_confirmation as runner
from scripts import verify_leaf_decision_logit_confirmation as verifier


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _relative_record(path: Path, run_dir: Path) -> dict[str, Any]:
    return runner.file_record(path, relative_to=run_dir)


def _build_verified_fixture(tmp_path: Path) -> dict[str, Any]:
    data_root = tmp_path / "data"
    public = data_root / runner.COMPETITION_ID / "prepared" / "public"
    public.mkdir(parents=True)
    classes = [f"Class_{index:03d}" for index in range(99)]
    feature_columns = [f"feature_{index:03d}" for index in range(192)]
    train_rows = []
    for class_index, class_name in enumerate(classes):
        for within_class in range(5):
            train_rows.append(
                {
                    "id": class_index * 10 + within_class,
                    "species": class_name,
                    **{
                        column: float(class_index + within_class / 10.0 + offset / 1000.0)
                        for offset, column in enumerate(feature_columns)
                    },
                }
            )
    train = pd.DataFrame(train_rows)
    test = pd.DataFrame(
        {
            "id": [1000, 1001, 1002],
            **{
                column: [
                    1.0 + offset / 1000.0,
                    2.0 + offset / 1000.0,
                    3.0 + offset / 1000.0,
                ]
                for offset, column in enumerate(feature_columns)
            },
        }
    )
    sample = pd.DataFrame({"id": [1002, 1000, 1001]})
    for class_name in classes:
        sample[class_name] = 0.0
    train_path = public / "train.csv"
    test_path = public / "test.csv"
    sample_path = public / "sample_submission.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    sample.to_csv(sample_path, index=False)

    image_rows = []
    for index, value in enumerate(train["id"]):
        image_rows.append(
            {
                "split": "train",
                "index": index,
                "id": value,
                "sha256": f"train_hash_{index:04d}",
            }
        )
    for index, value in enumerate(test["id"]):
        image_rows.append(
            {
                "split": "test",
                "index": index,
                "id": value,
                "sha256": f"test_hash_{index:04d}",
            }
        )
    image_manifest = pd.DataFrame(image_rows)
    image_manifest_path = tmp_path / "reference" / "leaf_image_manifest.csv"
    image_manifest_path.parent.mkdir(parents=True)
    image_manifest.to_csv(image_manifest_path, index=False)
    reference_report_path = tmp_path / "reference" / "reference_report.json"
    _write_json(reference_report_path, {"competition_id": runner.COMPETITION_ID})

    cache_records = []
    cache_root = tmp_path / "external_cache"
    cache_root.mkdir()
    for backbone, width in zip(runner.BACKBONES, [768, 1280], strict=True):
        cache_path = cache_root / f"{backbone}.npy"
        values = np.zeros((len(train) + len(test), width), dtype=np.float32)
        np.save(cache_path, values, allow_pickle=False)
        contract = {
            "backbone": backbone,
            "private_labels_used": False,
            "cache_key_sha256": f"cache-key-{backbone}",
        }
        metadata_path = cache_root / f"{backbone}.json"
        _write_json(
            metadata_path,
            {
                "cache_sha256": runner.sha256_file(cache_path),
                "contract": contract,
                "pretrained_weight_identity": {"backbone": backbone},
            },
        )
        cache_records.append(
            {
                "backbone": backbone,
                "cache": runner.file_record(cache_path),
                "metadata": runner.file_record(metadata_path),
                "contract": contract,
                "pretrained_weight_identity": {"backbone": backbone},
                "rows": len(values),
                "width": width,
                "dtype": str(values.dtype),
            }
        )

    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    models = artifacts / "models"
    seeds_root = artifacts / "seeds"
    models.mkdir(parents=True)
    seeds_root.mkdir()

    source_files = []
    source_root = tmp_path / "sources"
    source_root.mkdir()
    for name in [
        "run_leaf_decision_logit_confirmation.py",
        "verify_leaf_decision_logit_confirmation.py",
        "run_leaf_multibackbone_oof.py",
        "mlebench_medal_recovery_adapters.py",
        "mlebench_wave2_adapters.py",
        "russian_transliteration.py",
        "mlebench_phase_a.py",
    ]:
        source_path = source_root / name
        source_path.write_text(f"# {name}\n", encoding="utf-8")
        source_files.append(runner.file_record(source_path))
    source_manifest = {
        "schema": runner.SOURCE_SCHEMA,
        "created_at": runner.now_iso(),
        "files": source_files,
        "runtime": {
            "python": sys.version,
            "packages": runner.dependency_versions(),
        },
    }
    source_manifest_path = artifacts / "source_manifest.json"
    _write_json(source_manifest_path, source_manifest)

    input_manifest = {
        "schema": runner.INPUT_SCHEMA,
        "created_at": runner.now_iso(),
        "competition_id": runner.COMPETITION_ID,
        "public_dir": str(public),
        "public_files": {
            "train": runner.file_record(train_path),
            "test": runner.file_record(test_path),
            "sample_submission": runner.file_record(sample_path),
        },
        "reference_report": runner.file_record(reference_report_path),
        "reference_image_manifest": runner.file_record(image_manifest_path),
        "embedding_caches": cache_records,
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_columns": feature_columns,
        "feature_columns_sha256": runner.sha256_json(feature_columns),
        "classes": classes,
        "classes_sha256": runner.sha256_json(classes),
        "train_ids_sha256": runner.sha256_json(
            [runner.canonical_id(value) for value in train["id"]]
        ),
        "test_ids_sha256": runner.sha256_json(
            [runner.canonical_id(value) for value in test["id"]]
        ),
        "private_labels_used": False,
    }
    input_manifest_path = artifacts / "input_manifest.json"
    _write_json(input_manifest_path, input_manifest)

    target = train["species"].astype(str).to_numpy()
    class_index = {value: index for index, value in enumerate(classes)}
    target_index = np.asarray([class_index[value] for value in target], dtype=np.int64)
    folds = np.tile(np.arange(5, dtype=np.int16), 99)
    oof_logits = np.zeros((len(train), len(classes)), dtype=np.float64)
    oof_logits[np.arange(len(train)), target_index] = 2.0
    oof_probability = runner.stable_softmax(oof_logits)
    seed_records = []
    seed_oof = []
    seed_test = []
    for seed in runner.CONFIRMATION_SEEDS:
        test_fold_logits = np.zeros((5, len(test), len(classes)), dtype=np.float64)
        for row in range(len(test)):
            test_fold_logits[:, row, row] = 2.0
        test_fold_probability = np.stack(
            [runner.stable_softmax(test_fold_logits[fold]) for fold in range(5)],
            axis=0,
        )
        test_probability = test_fold_probability.mean(axis=0)
        bundle_path = seeds_root / f"seed_{seed}.npz"
        np.savez_compressed(
            bundle_path,
            seed=np.asarray([seed], dtype=np.int64),
            train_id=runner.portable_unicode_array(
                [runner.canonical_id(value) for value in train["id"]]
            ),
            target=runner.portable_unicode_array(target),
            classes=runner.portable_unicode_array(classes),
            fold_assignment=folds,
            oof_decision_logits=oof_logits,
            oof_probability=oof_probability,
            test_id=runner.portable_unicode_array(
                [runner.canonical_id(value) for value in test["id"]]
            ),
            test_fold_decision_logits=test_fold_logits,
            test_fold_probability=test_fold_probability,
            test_probability=test_probability,
        )
        model_records = []
        fold_records = []
        for fold in range(5):
            model_path = models / f"seed_{seed}_fold_{fold}.joblib"
            model_path.write_bytes(f"model-{seed}-{fold}".encode())
            model_records.append(
                {
                    "fold": fold,
                    "fold_seed": seed * 100 + fold,
                    **_relative_record(model_path, run_dir),
                }
            )
            fold_records.append(
                {
                    "fold": fold,
                    "fit_rows": len(train) - 99,
                    "valid_rows": 99,
                    "fit_class_count": 99,
                    "valid_class_count": 99,
                    "group_overlap_count": 0,
                }
            )
        score = runner.multiclass_log_loss(target, oof_probability, classes)
        seed_records.append(
            {
                "seed": seed,
                "cross_fitted_log_loss": score,
                "gate_passed": score <= runner.LOG_LOSS_THRESHOLD,
                "folds": fold_records,
                "models": model_records,
                "bundle": _relative_record(bundle_path, run_dir),
            }
        )
        seed_oof.append(oof_probability)
        seed_test.append(test_probability)

    ensemble_oof = np.mean(seed_oof, axis=0)
    ensemble_test = np.mean(seed_test, axis=0)
    ensemble_score = runner.multiclass_log_loss(target, ensemble_oof, classes)
    ensemble_path = artifacts / "ensemble_oof_and_test.npz"
    np.savez_compressed(
        ensemble_path,
        seeds=np.asarray(runner.CONFIRMATION_SEEDS, dtype=np.int64),
        train_id=runner.portable_unicode_array(
            [runner.canonical_id(value) for value in train["id"]]
        ),
        target=runner.portable_unicode_array(target),
        classes=runner.portable_unicode_array(classes),
        oof_probability=ensemble_oof,
        test_id=runner.portable_unicode_array(
            [runner.canonical_id(value) for value in test["id"]]
        ),
        test_probability=ensemble_test,
    )
    candidate = runner.align_submission(sample, test["id"], ensemble_test, classes)
    candidate_path = artifacts / "submission_withheld.csv"
    candidate.to_csv(candidate_path, index=False)
    gate = runner.evaluate_gate(
        [record["cross_fitted_log_loss"] for record in seed_records],
        ensemble_score,
    )
    report = {
        "schema": runner.SCHEMA,
        "created_at": runner.now_iso(),
        "status": "confirmation_gate_passed_verification_pending",
        "run_id": "fixture-run",
        "competition_id": runner.COMPETITION_ID,
        "contract": runner.fixed_contract(),
        "source_manifest": _relative_record(source_manifest_path, run_dir),
        "input_manifest": _relative_record(input_manifest_path, run_dir),
        "seed_records": seed_records,
        "ensemble": {
            "cross_fitted_log_loss": ensemble_score,
            "bundle": _relative_record(ensemble_path, run_dir),
            "submission": _relative_record(candidate_path, run_dir),
        },
        "promotion_gate": gate,
        "candidate_only": True,
        "candidate_ready": False,
        "independent_verification_required": True,
        "private_labels_used": False,
        "private_scores_used_for_tuning": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    report_path = artifacts / "confirmation_report.json"
    _write_json(report_path, report)
    artifact_manifest = runner.build_artifact_manifest(run_dir)
    _write_json(run_dir / "artifact_manifest.json", artifact_manifest)
    return {
        "data_root": data_root,
        "run_dir": run_dir,
        "report": report,
        "train": train,
        "test": test,
        "classes": classes,
        "target": target,
        "groups": image_manifest.iloc[: len(train)]["sha256"].astype(str).to_numpy(),
    }


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def test_independent_verifier_rebuilds_logits_probabilities_and_submission(
    tmp_path: Path,
) -> None:
    fixture = _build_verified_fixture(tmp_path)

    result = verifier.verify_run(fixture["run_dir"], fixture["data_root"])

    assert result["status"] == "verification_passed"
    assert result["ok"] is True
    assert result["candidate_ready"] is True
    assert result["promotion_gate"]["passed"] is True
    assert [row["seed"] for row in result["seed_results"]] == [43, 44, 45]
    assert result["official_grader_executed"] is False
    assert result["kaggle_submission_executed"] is False


def test_verifier_rejects_artifact_hash_drift(tmp_path: Path) -> None:
    fixture = _build_verified_fixture(tmp_path)
    model = fixture["run_dir"] / "artifacts" / "models" / "seed_43_fold_0.joblib"
    model.write_bytes(model.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="size differs|hash differs"):
        verifier.verify_run(fixture["run_dir"], fixture["data_root"])


@pytest.mark.parametrize("field", ["classes", "train_id"])
def test_seed_verifier_rejects_class_or_id_drift(tmp_path: Path, field: str) -> None:
    fixture = _build_verified_fixture(tmp_path)
    seed_record = dict(fixture["report"]["seed_records"][0])
    original = fixture["run_dir"] / seed_record["bundle"]["path"]
    arrays = _load_npz_arrays(original)
    if field == "classes":
        changed = arrays["classes"].astype(str)
        changed[[0, 1]] = changed[[1, 0]]
        arrays["classes"] = runner.portable_unicode_array(changed)
    else:
        changed = arrays["train_id"].astype(str)
        changed[0] = "drifted-id"
        arrays["train_id"] = runner.portable_unicode_array(changed)
    np.savez_compressed(original, **arrays)
    seed_record["bundle"] = _relative_record(original, fixture["run_dir"])

    with pytest.raises(RuntimeError, match="classes differ|train IDs differ"):
        verifier.verify_seed_bundle(
            fixture["run_dir"],
            seed_record,
            train_ids=[runner.canonical_id(value) for value in fixture["train"]["id"]],
            target=fixture["target"],
            test_ids=[runner.canonical_id(value) for value in fixture["test"]["id"]],
            classes=fixture["classes"],
            groups=fixture["groups"],
        )


def test_seed_verifier_rejects_object_arrays_without_pickle(tmp_path: Path) -> None:
    fixture = _build_verified_fixture(tmp_path)
    seed_record = dict(fixture["report"]["seed_records"][0])
    original = fixture["run_dir"] / seed_record["bundle"]["path"]
    arrays = _load_npz_arrays(original)
    arrays["target"] = np.asarray(arrays["target"].astype(str), dtype=object)
    np.savez_compressed(original, **arrays)
    seed_record["bundle"] = _relative_record(original, fixture["run_dir"])

    with pytest.raises((RuntimeError, ValueError), match="pickle|Object arrays"):
        verifier.verify_seed_bundle(
            fixture["run_dir"],
            seed_record,
            train_ids=[runner.canonical_id(value) for value in fixture["train"]["id"]],
            target=fixture["target"],
            test_ids=[runner.canonical_id(value) for value in fixture["test"]["id"]],
            classes=fixture["classes"],
            groups=fixture["groups"],
        )


def test_failure_report_never_marks_candidate_ready(tmp_path: Path) -> None:
    output = tmp_path / "verification.json"
    code = verifier.main(
        [
            "--run-dir",
            str(tmp_path / "missing-run"),
            "--data-root",
            str(tmp_path / "missing-data"),
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert code == 1
    assert report["status"] == "verification_failed"
    assert report["candidate_ready"] is False
    assert report["candidate_only"] is True
    assert report["official_grader_executed"] is False
    assert report["kaggle_submission_executed"] is False
