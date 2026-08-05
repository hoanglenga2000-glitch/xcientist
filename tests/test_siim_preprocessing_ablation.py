"""Contract tests for the real SIIM preprocessing-ablation executor."""
from __future__ import annotations

import inspect
import json
import logging

import numpy as np
import pandas as pd
import pytest

from scripts import run_siim_preprocessing_ablation as ablation


def _cache_contract(**overrides):
    values = {
        "profile": "raw_multiview_v1",
        "view_name": "full_image",
        "backbone": "convnext_small",
        "weight_sha256": "1" * 64,
        "image_vector_sha256": "2" * 64,
        "image_content_manifest_sha256": "3" * 64,
        "image_size": 224,
        "adapter_source_sha256": "4" * 64,
        "wave2_source_sha256": "5" * 64,
    }
    values.update(overrides)
    return ablation.build_embedding_cache_contract(**values)


def test_siim_embedding_cache_key_binds_profile_content_backbone_and_weights():
    baseline = _cache_contract()
    replay = _cache_contract()
    assert baseline == replay
    assert baseline["cache_key_sha256"] == replay["cache_key_sha256"]

    mutations = [
        _cache_contract(profile="border_multiview_v1"),
        _cache_contract(view_name="lesion_focus"),
        _cache_contract(backbone="efficientnet_v2_s"),
        _cache_contract(weight_sha256="a" * 64),
        _cache_contract(image_vector_sha256="b" * 64),
        _cache_contract(image_content_manifest_sha256="c" * 64),
        _cache_contract(image_size=256),
    ]
    assert all(
        candidate["cache_key_sha256"] != baseline["cache_key_sha256"]
        for candidate in mutations
    )
    assert baseline["private_labels_used"] is False


def test_siim_embedding_cache_contract_rejects_invalid_sha256():
    with pytest.raises(ValueError, match="Invalid SHA256 fields"):
        _cache_contract(weight_sha256="not-a-sha")


def test_siim_evaluation_seed_parser_requires_three_distinct_integers():
    assert ablation.parse_evaluation_seeds("40, 41,42") == [40, 41, 42]
    assert ablation.parse_evaluation_seeds([3, 5, 7]) == [3, 5, 7]
    with pytest.raises(ValueError, match="three distinct"):
        ablation.parse_evaluation_seeds("40,41")
    with pytest.raises(ValueError, match="three distinct"):
        ablation.parse_evaluation_seeds("40,40,42")
    with pytest.raises(ValueError, match="comma-separated integers"):
        ablation.parse_evaluation_seeds("40,bad,42")


def test_siim_ablation_run_directory_requires_explicit_resume(tmp_path):
    run_dir = tmp_path / "runs" / "same-run"

    assert ablation.prepare_run_directory(run_dir, resume=False) == "created"
    with pytest.raises(FileExistsError, match="already exists"):
        ablation.prepare_run_directory(run_dir, resume=False)
    assert ablation.prepare_run_directory(run_dir, resume=True) == "resumed"

    file_path = tmp_path / "not-a-directory"
    file_path.write_text("evidence", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a directory"):
        ablation.prepare_run_directory(file_path, resume=True)


def test_siim_ablation_exposes_governed_cuda_memory_limit(tmp_path):
    args = ablation.parse_args(["--data-root", str(tmp_path)])
    assert args.memory_limit_mib == 0
    assert args.image_content_manifest is None
    assert len(ablation.FROZEN_EMBEDDING_ADAPTER_CONTRACT_SHA256) == 64
    source = inspect.getsource(ablation.main)
    assert "configure_siim_cuda_memory_limit" in source
    assert '"memory_limit_mib"' in source


def test_siim_frozen_linear_head_is_deterministic_and_covers_every_fold():
    rng = np.random.default_rng(123)
    rows = 90
    labels = (np.arange(rows) % 2).astype(np.int8)
    signal = labels.astype(np.float32)[:, None] * 2.0 - 1.0
    full = np.concatenate(
        [signal + rng.normal(0.0, 0.15, size=(rows, 1)), rng.normal(size=(rows, 7))],
        axis=1,
    ).astype(np.float32)
    lesion = np.concatenate(
        [signal + rng.normal(0.0, 0.20, size=(rows, 1)), rng.normal(size=(rows, 5))],
        axis=1,
    ).astype(np.float32)
    indices = np.arange(rows)
    splits = []
    for fold in range(3):
        valid = indices[indices % 3 == fold]
        fit = indices[indices % 3 != fold]
        splits.append((fit, valid))

    first = ablation.evaluate_frozen_linear_head(
        full,
        lesion,
        labels,
        splits,
        seed=42,
        alpha=1e-4,
        max_iter=500,
    )
    second = ablation.evaluate_frozen_linear_head(
        full,
        lesion,
        labels,
        splits,
        seed=42,
        alpha=1e-4,
        max_iter=500,
    )

    assert first["fold_auc"] == second["fold_auc"]
    np.testing.assert_array_equal(first["oof_prediction"], second["oof_prediction"])
    assert len(first["folds"]) == 3
    assert all(record["auc"] > 0.95 for record in first["folds"])
    assert all(record["standardizer_fit_scope"] == "fold_fit_only" for record in first["folds"])
    assert all(record["head_fit_scope"] == "fold_fit_only" for record in first["folds"])


def test_siim_fold_contract_covers_three_external_seeds(tmp_path, monkeypatch):
    rows = 12
    train_ids = [f"train_{index}" for index in range(rows)]
    test_ids = ["test_0", "test_1"]
    train = pd.DataFrame({
        "image_name": train_ids,
        "patient_id": [f"patient_{index}" for index in range(rows)],
        "target": np.arange(rows) % 2,
    })
    test = pd.DataFrame({"image_name": test_ids})
    target = train["target"].to_numpy(dtype=np.int8)
    dataset = {
        "train": train,
        "test": test,
        "target": target,
        "train_paths": [tmp_path / f"{value}.jpg" for value in train_ids],
        "test_paths": [tmp_path / f"{value}.jpg" for value in test_ids],
    }
    manifest = pd.DataFrame({
        "source": ["train"] * rows + ["test"] * len(test_ids),
        "image_name": train_ids + test_ids,
        "content_sha256": [f"{index:064x}" for index in range(rows + len(test_ids))],
        "decoded_pixel_sha256": [
            f"{index + 100:064x}" for index in range(rows + len(test_ids))
        ],
        "perceptual_dhash64": [
            f"{index:016x}" for index in range(rows + len(test_ids))
        ],
    })
    groups = np.asarray([f"group_{index}" for index in range(rows)], dtype=object)

    monkeypatch.setattr(
        ablation.recovery,
        "build_siim_image_content_manifest",
        lambda *args, **kwargs: (manifest.copy(), {"rows": len(manifest)}),
    )
    monkeypatch.setattr(
        ablation.recovery,
        "build_siim_content_connected_groups",
        lambda *args, **kwargs: (groups.copy(), {"group_count": rows}),
    )

    def fake_patient_folds(*args, seed, group_values, **kwargs):
        indices = np.arange(rows)
        split_list = []
        for fold in range(3):
            valid = indices[(indices + int(seed)) % 3 == fold]
            fit = indices[(indices + int(seed)) % 3 != fold]
            split_list.append((fit, valid))
        return split_list, np.asarray(group_values), "mock_grouped_three_fold"

    monkeypatch.setattr(ablation.recovery, "build_siim_patient_folds", fake_patient_folds)
    contract = ablation.build_manifest_and_folds(
        dataset,
        run_dir=tmp_path,
        folds=3,
        evaluation_seeds=[40, 41, 42],
        workers=1,
    )

    assert set(contract["splits_by_seed"]) == {40, 41, 42}
    assert set(contract["assignment_paths"]) == {40, 41, 42}
    assert all(path.is_file() for path in contract["assignment_paths"].values())
    split_payload = json.loads(
        (tmp_path / "siim_preprocessing_ablation_split_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert split_payload["evaluation_seed_count"] == 3
    assert split_payload["evaluation_seeds"] == [40, 41, 42]
    assert len(split_payload["seed_contracts"]) == 3
    assert all(len(record["folds"]) == 3 for record in split_payload["seed_contracts"])

    with pytest.raises(RuntimeError, match="three distinct"):
        ablation.build_manifest_and_folds(
            dataset,
            run_dir=tmp_path / "invalid",
            folds=3,
            evaluation_seeds=[40, 40, 42],
            workers=1,
        )


def test_siim_profile_embeddings_are_reused_across_three_seed_oof_runs(
    tmp_path, monkeypatch
):
    rows = 12
    target = (np.arange(rows) % 2).astype(np.int8)
    seeds = [40, 41, 42]
    profile_embeddings = {
        profile: {
            "full_image": np.full((rows, 3), index, dtype=np.float32),
            "lesion_focus": np.full((rows, 2), index + 10, dtype=np.float32),
        }
        for index, profile in enumerate(ablation.recovery.SIIM_PREPROCESSING_PROFILES)
    }
    assignment_paths = {}
    for seed in seeds:
        path = tmp_path / f"folds_s{seed}.csv"
        pd.DataFrame({"fold": (np.arange(rows) + seed) % 3}).to_csv(path, index=False)
        assignment_paths[seed] = path
    fold_contract = {
        "assignment_paths": assignment_paths,
        "splits_by_seed": {seed: [(np.array([]), np.array([]))] * 3 for seed in seeds},
    }
    calls = []

    def fake_evaluate(full, lesion, labels, splits, *, seed, alpha, max_iter):
        calls.append((id(full), id(lesion), int(seed), float(alpha), int(max_iter)))
        return {
            "fold_auc": [0.80 + seed * 1e-5, 0.81, 0.82],
            "oof_auc": 0.81,
            "folds": [{"fold": index} for index in range(3)],
            "oof_prediction": np.linspace(-1.0, 1.0, len(labels)),
        }

    monkeypatch.setattr(ablation, "evaluate_frozen_linear_head", fake_evaluate)
    fold_auc, evaluations = ablation.evaluate_profiles_across_seeds(
        profile_embeddings=profile_embeddings,
        train_image_names=[f"image_{index}" for index in range(rows)],
        target=target,
        fold_contract=fold_contract,
        evaluation_seeds=seeds,
        run_dir=tmp_path,
        alpha=1e-4,
        max_iter=500,
        logger=logging.getLogger("siim-multiseed-test"),
    )

    assert len(calls) == len(ablation.recovery.SIIM_PREPROCESSING_PROFILES) * 3
    assert all(set(seed_scores) == set(seeds) for seed_scores in fold_auc.values())
    assert all(set(seed_records) == {"40", "41", "42"} for seed_records in evaluations.values())
    for profile in ablation.recovery.SIIM_PREPROCESSING_PROFILES:
        expected_full_id = id(profile_embeddings[profile]["full_image"])
        expected_lesion_id = id(profile_embeddings[profile]["lesion_focus"])
        profile_calls = [
            record
            for record in calls
            if record[0] == expected_full_id and record[1] == expected_lesion_id
        ]
        assert [record[2] for record in profile_calls] == seeds
    oof_paths = sorted((tmp_path / "oof").glob("*_oof.csv"))
    assert len(oof_paths) == 15
    assert all(path.stat().st_size > 0 for path in oof_paths)


def test_siim_ablation_main_wires_full_public_grouped_five_profile_contract():
    source = inspect.getsource(ablation.main)
    manifest_source = inspect.getsource(ablation.build_manifest_and_folds)
    assert "recovery.SIIM_PREPROCESSING_PROFILES" in source
    assert "select_siim_preprocessing_ablation" in source
    assert "parse_evaluation_seeds" in source
    assert "evaluate_profiles_across_seeds" in source
    assert '"split_strategy_by_seed"' in source
    assert '"split_records_by_seed"' in source
    assert '"full_public_train_scope": True' in source
    assert '"official_score_claimed": False' in source
    assert "build_siim_image_content_manifest" in manifest_source
    assert "build_siim_content_connected_groups" in manifest_source
    assert "build_siim_patient_folds" in manifest_source
    assert "require_requested_folds=True" in manifest_source
    assert "load_precomputed_siim_image_content_manifest" in manifest_source
    assert '"patient_content_group_isolation": True' in manifest_source
    assert '"validation_coverage_exactly_once": True' in manifest_source
