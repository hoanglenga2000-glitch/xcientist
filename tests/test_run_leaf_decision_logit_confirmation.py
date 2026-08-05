from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_leaf_decision_logit_confirmation as runner


def _legacy_sklearn_172_assignment(
    target: np.ndarray, groups: np.ndarray, *, seed: int
) -> np.ndarray:
    """Reproduce the shuffled-group mapping bug observed in sklearn 1.7.2."""
    _, target_inverse, target_count = np.unique(
        target,
        return_inverse=True,
        return_counts=True,
    )
    _, group_inverse, group_count = np.unique(
        groups,
        return_inverse=True,
        return_counts=True,
    )
    group_target_count = np.zeros((len(group_count), len(target_count)))
    for class_index, group_index in zip(
        target_inverse,
        group_inverse,
        strict=True,
    ):
        group_target_count[group_index, class_index] += 1
    fold_target_count = np.zeros((runner.FOLD_COUNT, len(target_count)))
    groups_per_fold: defaultdict[int, set[int]] = defaultdict(set)
    np.random.RandomState(seed).shuffle(group_target_count)
    group_order = np.argsort(
        -np.std(group_target_count, axis=1),
        kind="mergesort",
    )
    for group_index in group_order:
        group_counts = group_target_count[group_index]
        best_fold = -1
        minimum_evaluation = np.inf
        minimum_samples = np.inf
        for fold in range(runner.FOLD_COUNT):
            fold_target_count[fold] += group_counts
            evaluation = np.mean(
                np.std(
                    fold_target_count / target_count.reshape(1, -1),
                    axis=0,
                )
            )
            fold_target_count[fold] -= group_counts
            samples = np.sum(fold_target_count[fold])
            if evaluation < minimum_evaluation or (
                np.isclose(evaluation, minimum_evaluation)
                and samples < minimum_samples
            ):
                minimum_evaluation = evaluation
                minimum_samples = samples
                best_fold = fold
        fold_target_count[best_fold] += group_counts
        groups_per_fold[best_fold].add(int(group_index))
    assignment = np.full(len(target), -1, dtype=np.int16)
    for fold in range(runner.FOLD_COUNT):
        assignment[
            np.asarray(
                [value in groups_per_fold[fold] for value in group_inverse],
                dtype=bool,
            )
        ] = fold
    return assignment


def test_frozen_contract_has_no_confirmation_seed_tuning_surface() -> None:
    contract = runner.fixed_contract()

    assert contract["seeds"] == [43, 44, 45]
    assert contract["development_seeds_excluded"] == [40, 41, 42]
    assert contract["calibration"] == {
        "source": "decision_function",
        "function": "stable_softmax",
        "temperature": 0.15,
        "frozen_before_confirmation": True,
        "confirmation_seed_tuning_allowed": False,
    }
    assert contract["svc"]["C"] == 10.0
    assert contract["svc"]["probability"] is False
    assert contract["candidate_only"] is True
    assert contract["automatic_official_grader"] is False
    assert contract["automatic_kaggle_submission"] is False
    parser = runner.build_parser()
    option_names = {option for action in parser._actions for option in action.option_strings}
    assert "--seeds" not in option_names
    assert "--temperature" not in option_names
    assert "--svc-c" not in option_names


def test_stable_softmax_is_normalized_and_temperature_is_immutable() -> None:
    logits = np.asarray([[4.0, 1.0, -2.0], [-1.0, 0.0, 3.0]], dtype=np.float64)
    probability = runner.stable_softmax(logits)

    assert probability.shape == logits.shape
    assert np.isfinite(probability).all()
    assert np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    assert probability.argmax(axis=1).tolist() == [0, 2]
    with pytest.raises(ValueError, match="frozen"):
        runner.stable_softmax(logits, temperature=0.2)


def test_portable_unicode_and_manual_log_loss_contract() -> None:
    target = runner.portable_unicode_array(["Acer", "Quercus"])
    probability = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)

    assert target.dtype.kind == "U"
    assert target.dtype.hasobject is False
    score = runner.multiclass_log_loss(target, probability, ["Acer", "Quercus"])
    assert score == pytest.approx(-0.5 * (np.log(0.9) + np.log(0.8)))


def test_frozen_reference_basenames_are_portable_across_operating_systems() -> None:
    assert (
        runner.portable_reference_basename(
            r"D:\cache\convnext_small__frozen.npy"
        )
        == "convnext_small__frozen.npy"
    )
    assert (
        runner.portable_reference_basename(
            "/cache/efficientnet_v2_s__frozen.json"
        )
        == "efficientnet_v2_s__frozen.json"
    )
    with pytest.raises(ValueError, match="portable basename"):
        runner.portable_reference_basename("")


def test_gate_requires_every_seed_mean_max_and_ensemble() -> None:
    passed = runner.evaluate_gate([0.010, 0.011, 0.012], 0.009)
    one_seed_failed = runner.evaluate_gate([0.010, 0.014, 0.011], 0.009)
    ensemble_failed = runner.evaluate_gate([0.010, 0.011, 0.012], 0.014)

    assert passed["passed"] is True
    assert all(passed["checks"].values())
    assert one_seed_failed["passed"] is False
    assert one_seed_failed["checks"]["all_per_seed_at_or_below_threshold"] is False
    assert one_seed_failed["checks"]["maximum_seed_log_loss_at_or_below_threshold"] is False
    assert ensemble_failed["passed"] is False
    assert ensemble_failed["checks"]["ensemble_log_loss_at_or_below_threshold"] is False


def test_grouped_folds_cover_every_class_and_row() -> None:
    classes = [f"Class_{index:02d}" for index in range(6)]
    target = np.asarray([value for value in classes for _ in range(5)])
    groups = np.asarray([f"group_{index}" for index in range(len(target))])
    features = np.arange(len(target), dtype=np.float64).reshape(-1, 1)

    _, assignment, records = runner.build_folds(
        features,
        target,
        groups,
        seed=43,
        classes=classes,
    )

    assert sorted(np.unique(assignment).tolist()) == list(range(5))
    assert len(records) == 5
    assert all(record["fit_class_count"] == len(classes) for record in records)
    assert all(record["valid_class_count"] == len(classes) for record in records)
    assert all(record["group_overlap_count"] == 0 for record in records)


def test_class_balanced_folds_regress_remote_sklearn_172_missing_classes() -> None:
    classes = [f"Class_{index:03d}" for index in range(99)]
    target = np.asarray([value for value in classes for _ in range(9)])
    groups = np.asarray([f"group_{index:04d}" for index in range(len(target))])
    features = np.zeros((len(target), 1), dtype=np.float64)
    expected_classes = set(classes)
    assignments = []

    for seed in runner.CONFIRMATION_SEEDS:
        legacy = _legacy_sklearn_172_assignment(target, groups, seed=seed)
        assert any(
            set(target[legacy == fold]) != expected_classes
            for fold in range(runner.FOLD_COUNT)
        )

        splits, assignment, records = runner.build_folds(
            features,
            target,
            groups,
            seed=seed,
            classes=classes,
        )
        assignments.append(assignment)
        assert np.array_equal(
            assignment,
            runner.build_folds(
                features,
                target,
                groups,
                seed=seed,
                classes=classes,
            )[1],
        )
        assert np.array_equal(
            np.sort(np.concatenate([valid for _, valid in splits])),
            np.arange(len(target)),
        )
        assert sorted(np.unique(assignment).tolist()) == list(
            range(runner.FOLD_COUNT)
        )
        assert all(record["valid_class_count"] == 99 for record in records)
        assert all(record["fit_class_count"] == 99 for record in records)
        assert all(record["group_overlap_count"] == 0 for record in records)
        for fold, (fit, valid) in enumerate(splits):
            assert np.all(assignment[valid] == fold)
            assert set(target[valid]) == expected_classes
            assert set(target[fit]) == expected_classes
            assert not (set(groups[fit]) & set(groups[valid]))
    assert len({value.tobytes() for value in assignments}) == len(
        runner.CONFIRMATION_SEEDS
    )


def test_class_balanced_folds_keep_groups_atomic_and_fail_closed() -> None:
    classes = [f"Class_{index:02d}" for index in range(6)]
    target = np.asarray(
        [class_name for class_name in classes for _group in range(6) for _row in range(2)]
    )
    groups = np.asarray(
        [
            f"{class_name}_group_{group}"
            for class_name in classes
            for group in range(6)
            for _row in range(2)
        ]
    )
    features = np.zeros((len(target), 1), dtype=np.float64)

    _, assignment, _ = runner.build_folds(
        features,
        target,
        groups,
        seed=43,
        classes=classes,
    )
    for group_name in np.unique(groups):
        assert len(set(assignment[groups == group_name])) == 1

    crossed = groups.copy()
    crossed[np.flatnonzero(target == classes[1])[0]] = groups[0]
    with pytest.raises(RuntimeError, match="spans target classes"):
        runner.build_folds(
            features,
            target,
            crossed,
            seed=43,
            classes=classes,
        )

    too_few_groups = groups.copy()
    too_few_groups[target == classes[0]] = np.resize(
        np.asarray([f"limited_{index}" for index in range(4)]),
        int(np.sum(target == classes[0])),
    )
    with pytest.raises(RuntimeError, match="fewer than 5 distinct groups"):
        runner.build_folds(
            features,
            target,
            too_few_groups,
            seed=43,
            classes=classes,
        )


def test_submission_alignment_uses_ids_not_frame_order() -> None:
    sample = pd.DataFrame({"id": [30, 10, 20], "A": 0.0, "B": 0.0})
    test_ids = [10, 20, 30]
    probability = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])

    result = runner.align_submission(sample, test_ids, probability, ["A", "B"])

    assert result["id"].tolist() == [30, 10, 20]
    assert np.allclose(
        result[["A", "B"]].to_numpy(),
        [[0.6, 0.4], [0.9, 0.1], [0.2, 0.8]],
    )


def test_artifact_manifest_requires_exact_artifact_tree(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for relative in runner.expected_artifact_paths():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"path": relative}), encoding="utf-8")

    manifest = runner.build_artifact_manifest(run_dir)

    assert manifest["exact_inventory_required"] is True
    assert manifest["expected_paths"] == runner.expected_artifact_paths()
    assert manifest["artifact_count"] == len(runner.expected_artifact_paths())


def test_source_has_no_grader_kaggle_or_network_execution() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")

    assert "private_grade(" not in source
    assert "subprocess" not in source
    assert "paramiko" not in source
    assert "urllib" not in source
    assert '"candidate_only": True' in source
    assert '"official_grader_executed": False' in source
    assert '"kaggle_submission_executed": False' in source
