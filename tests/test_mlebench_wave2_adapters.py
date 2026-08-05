"""Fast unit contracts for Wave 2 helpers; no GPU or dataset is required."""
from __future__ import annotations

import hashlib
import inspect
import pickle
import zipfile
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile
from scipy.sparse import csr_matrix

from scripts import mlebench_medal_recovery_adapters as recovery
from scripts import mlebench_wave2_adapters as wave2
from scripts import russian_transliteration as ru_transliteration


def test_wave2_vision_datasets_are_windows_spawn_picklable():
    teacher = wave2._DogBreedTeacherImages([], None)
    training = wave2._VisionImages([], None, None, "multiclass")

    assert "<locals>" not in type(teacher).__qualname__
    assert "<locals>" not in type(training).__qualname__
    pickle.loads(pickle.dumps(teacher))
    pickle.loads(pickle.dumps(training))


def test_wave2_registry_is_exact_and_complete():
    assert len(wave2.WAVE2_COMPETITIONS) == 11
    assert set(wave2.RUNNERS) == set(wave2.WAVE2_COMPETITIONS)


def test_wave2_promotion_contract_registry_is_exact_for_eight_unscored_tasks():
    expected = {
        "aptos2019-blindness-detection",
        "dog-breed-identification",
        "histopathologic-cancer-detection",
        "jigsaw-toxic-comment-classification-challenge",
        "mlsp-2013-birds",
        "plant-pathology-2020-fgvc7",
        "ranzcr-clip-catheter-line-classification",
        "the-icml-2013-whale-challenge-right-whale-redux",
    }
    assert set(wave2.WAVE2_PROMOTION_CONTRACTS) == expected
    assert len(wave2.WAVE2_PROMOTION_CONTRACTS) == 8


def test_jigsaw_parallel_nbsvm_matches_serial_coefficients_and_order():
    rng = np.random.RandomState(42)
    word = csr_matrix(rng.uniform(size=(48, 23)).astype(np.float32))
    char = csr_matrix(rng.uniform(size=(48, 31)).astype(np.float32))
    truth = np.column_stack(
        [
            np.asarray([(row + column) % (column + 2) == 0 for row in range(48)], dtype=np.int8)
            for column in range(3)
        ]
    )
    names = ("toxic", "severe_toxic", "obscene")
    kwargs = {
        "target_names": names,
        "c_value": 2.0,
        "max_iter": 100,
        "seed_base": 314,
    }

    serial = wave2.fit_jigsaw_nbsvm_channels(word, char, truth, workers=1, **kwargs)
    parallel = wave2.fit_jigsaw_nbsvm_channels(word, char, truth, workers=6, **kwargs)

    assert [(item["channel"], item["index"], item["target"], item["seed"]) for item in serial] == [
        (item["channel"], item["index"], item["target"], item["seed"]) for item in parallel
    ]
    for expected, actual in zip(serial, parallel, strict=True):
        np.testing.assert_allclose(actual["ratio"], expected["ratio"], rtol=0, atol=0)
        np.testing.assert_allclose(actual["model"].coef_, expected["model"].coef_, rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual["model"].intercept_, expected["model"].intercept_, rtol=0, atol=1e-12)


def test_binary_ranking_gate_counts_ties_as_violations():
    truth = np.asarray([1, 1, 0, 0])
    probability = np.asarray([0.9, 0.5, 0.5, 0.1])
    counts = wave2.binary_ranking_violations(truth, probability)
    assert counts["strict_inversions"] == 0
    assert counts["positive_negative_ties"] == 1
    assert counts["non_strict_violations"] == 1


def test_vision_promotion_gate_requires_perfect_fold_ranking():
    contract = {
        "name": "aerial",
        "metric": "roc_auc",
        "direction": "maximize",
        "threshold": 1.0,
        "require_all_folds": True,
        "require_zero_binary_ranking_violations": True,
    }
    passed = wave2.evaluate_vision_promotion_gate(
        contract,
        metric="roc_auc",
        cv_score=1.0,
        fold_scores=[1.0, 1.0],
        truth=np.asarray([1, 1, 0, 0]),
        oof_probability=np.asarray([0.9, 0.8, 0.2, 0.1]),
    )
    failed = wave2.evaluate_vision_promotion_gate(
        contract,
        metric="roc_auc",
        cv_score=1.0,
        fold_scores=[1.0, 0.999],
        truth=np.asarray([1, 1, 0, 0]),
        oof_probability=np.asarray([0.9, 0.5, 0.5, 0.1]),
    )
    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["ranking"]["non_strict_violations"] == 1


def test_vision_promotion_gate_separates_aggregate_and_fold_thresholds():
    contract = {
        "name": "separate_thresholds",
        "metric": "roc_auc",
        "direction": "maximize",
        "threshold": 0.90,
        "fold_threshold": 0.80,
        "require_all_folds": True,
    }
    aggregate_failed = wave2.evaluate_vision_promotion_gate(
        contract,
        metric="roc_auc",
        cv_score=0.89,
        fold_scores=[0.88, 0.87],
        truth=np.asarray([0, 1]),
        oof_probability=np.asarray([0.1, 0.9]),
    )
    fold_failed = wave2.evaluate_vision_promotion_gate(
        contract,
        metric="roc_auc",
        cv_score=0.91,
        fold_scores=[0.88, 0.79],
        truth=np.asarray([0, 1]),
        oof_probability=np.asarray([0.1, 0.9]),
    )
    assert aggregate_failed["checks"]["aggregate_threshold"] is False
    assert aggregate_failed["checks"]["every_fold_threshold"] is True
    assert fold_failed["checks"]["aggregate_threshold"] is True
    assert fold_failed["checks"]["every_fold_threshold"] is False
    assert aggregate_failed["passed"] is False
    assert fold_failed["passed"] is False


def test_cross_fitted_ordinal_thresholds_are_deterministic_and_cover_every_row():
    truth = np.tile(np.arange(5), 12)
    expected = truth.astype(float) + np.linspace(-0.16, 0.16, len(truth))
    folds = np.arange(len(truth)) % 3
    first_prediction, first_records = wave2.cross_fit_ordinal_thresholds(
        expected, truth, folds
    )
    second_prediction, second_records = wave2.cross_fit_ordinal_thresholds(
        expected, truth, folds
    )
    np.testing.assert_array_equal(first_prediction, second_prediction)
    assert first_records == second_records
    assert len(first_prediction) == len(truth)
    assert set(np.unique(first_prediction)) <= set(range(5))
    assert sum(record["validation_rows"] for record in first_records) == len(truth)
    assert {record["fold"] for record in first_records} == {0, 1, 2}


def test_jigsaw_blend_improvement_failure_blocks_promotion():
    rows = np.arange(60)
    folds = rows % 3
    truth = np.column_stack([
        ((rows // 3 + offset) % 2).astype(np.int8)
        for offset in range(len(wave2.JIGSAW_TARGET_COLUMNS))
    ])
    perfect = 0.1 + 0.8 * truth
    gate = wave2.build_jigsaw_promotion_gate(
        wave2.WAVE2_PROMOTION_CONTRACTS[
            "jigsaw-toxic-comment-classification-challenge"
        ],
        target_columns=wave2.JIGSAW_TARGET_COLUMNS,
        truth=truth,
        fold_assignment=folds,
        oof_word=perfect,
        oof_char=perfect,
        oof_blend=perfect,
    )
    assert gate["checks"]["aggregate_threshold"] is True
    assert gate["checks"]["every_label_scoreable_in_every_fold"] is True
    assert gate["checks"]["minimum_cross_fitted_blend_improvement"] is False
    assert gate["evidence"]["cross_fitted_blend_improvement"] == pytest.approx(0.0)
    assert gate["passed"] is False


def test_jigsaw_promotion_uses_same_fold_clean_rank_protocol_for_all_components():
    rows = np.arange(90)
    folds = rows % 3
    truth = np.column_stack([
        ((rows // 3 + offset) % 2).astype(np.int8)
        for offset in range(len(wave2.JIGSAW_TARGET_COLUMNS))
    ])
    base = truth.astype(float) + rows[:, None] * 1e-5
    blend = np.empty_like(base)
    for fold in np.unique(folds):
        held_out = folds == fold
        for label in range(base.shape[1]):
            blend[held_out, label] = wave2._fractional_rank(base[held_out, label])
    gate = wave2.build_jigsaw_promotion_gate(
        wave2.WAVE2_PROMOTION_CONTRACTS[
            "jigsaw-toxic-comment-classification-challenge"
        ],
        target_columns=wave2.JIGSAW_TARGET_COLUMNS,
        truth=truth,
        fold_assignment=folds,
        oof_word=base,
        oof_char=base,
        oof_blend=blend,
    )
    assert gate["evidence"]["promotion_normalization"] == (
        "same_fold_clean_fractional_rank_for_all_declared_channels_and_candidate"
    )
    assert gate["evidence"]["cross_fitted_blend_improvement"] == pytest.approx(0.0)


def test_jigsaw_promotion_actually_fold_ranks_candidate_before_scoring():
    folds = np.repeat(np.arange(2), 4)
    one_label_truth = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    truth = np.column_stack(
        [one_label_truth for _ in wave2.JIGSAW_TARGET_COLUMNS]
    )
    one_label_candidate = np.asarray([0.80, 0.90, 0.81, 0.91, 0.10, 0.20, 0.11, 0.21])
    candidate = np.column_stack(
        [one_label_candidate for _ in wave2.JIGSAW_TARGET_COLUMNS]
    )
    base = np.column_stack(
        [np.linspace(0.1, 0.9, len(folds)) for _ in wave2.JIGSAW_TARGET_COLUMNS]
    )
    raw_auc = wave2.compute_metric("roc_auc", one_label_truth, one_label_candidate)
    assert raw_auc < 1.0
    gate = wave2.build_jigsaw_promotion_gate(
        wave2.WAVE2_PROMOTION_CONTRACTS[
            "jigsaw-toxic-comment-classification-challenge"
        ],
        target_columns=wave2.JIGSAW_TARGET_COLUMNS,
        truth=truth,
        fold_assignment=folds,
        oof_word=base,
        oof_char=base,
        oof_blend=candidate,
    )
    assert gate["evidence"]["cross_fitted_candidate_unweighted_mean_auc"] == pytest.approx(1.0)


def _synthetic_jigsaw_stacker_fixture():
    rng = np.random.default_rng(123)
    rows = np.arange(180)
    truth = np.column_stack(
        [
            ((rows // 3 + label) % (2 + (label % 2)) == 0).astype(np.int8)
            for label in range(len(wave2.JIGSAW_TARGET_COLUMNS))
        ]
    )
    _, folds = wave2.make_multilabel_stratified_folds(
        truth, requested_folds=3, seed=17
    )
    word = np.clip(0.15 + 0.68 * truth + rng.normal(0, 0.12, truth.shape), 0.001, 0.999)
    char = np.clip(0.20 + 0.60 * truth + rng.normal(0, 0.14, truth.shape), 0.001, 0.999)
    baseline = 0.55 * word + 0.45 * char
    features, names = wave2.build_jigsaw_meta_features(
        word,
        char,
        baseline,
        folds,
        target_columns=wave2.JIGSAW_TARGET_COLUMNS,
    )
    test_rows = 23
    test_word = rng.uniform(0.01, 0.99, size=(3, test_rows, truth.shape[1]))
    test_char = rng.uniform(0.01, 0.99, size=(3, test_rows, truth.shape[1]))
    weights = np.full((3, truth.shape[1]), 0.5)
    test_blend = wave2.build_jigsaw_fold_test_blends(test_word, test_char, weights)
    test_features, test_names = wave2.build_jigsaw_test_meta_features_by_fold(
        test_word,
        test_char,
        test_blend,
        target_columns=wave2.JIGSAW_TARGET_COLUMNS,
    )
    assert names == test_names
    return truth, folds, features, test_features, names


def test_jigsaw_hybrid30_stacker_is_exact_once_and_outer_label_isolated():
    truth, folds, features, test_features, names = _synthetic_jigsaw_stacker_fixture()
    first, test_prediction, counts, contract = wave2.cross_fit_jigsaw_label_stacker(
        features,
        test_features,
        truth,
        folds,
        target_columns=wave2.JIGSAW_TARGET_COLUMNS,
        feature_names=names,
        c_grid=[0.1],
        max_iter=120,
        seed=91,
    )
    assert first.shape == truth.shape
    assert test_prediction.shape == (test_features.shape[1], truth.shape[1])
    assert np.all(counts == 1)
    assert np.isfinite(first).all() and np.isfinite(test_prediction).all()
    assert contract["feature_set"] == "hybrid30"
    assert len(contract["records"]) == len(np.unique(folds)) * truth.shape[1]

    mutated = truth.copy()
    held_out = folds == 0
    mutated[held_out, 0] = 1 - mutated[held_out, 0]
    repeated, _, _, _ = wave2.cross_fit_jigsaw_label_stacker(
        features,
        test_features,
        mutated,
        folds,
        target_columns=wave2.JIGSAW_TARGET_COLUMNS,
        feature_names=names,
        c_grid=[0.1],
        max_iter=120,
        seed=91,
    )
    np.testing.assert_allclose(first[held_out, 0], repeated[held_out, 0], atol=1e-12)


def test_jigsaw_provenance_rejects_missing_or_duplicate_oof_writes():
    truth, folds, features, test_features, names = _synthetic_jigsaw_stacker_fixture()
    _, _, counts, contract = wave2.cross_fit_jigsaw_label_stacker(
        features,
        test_features,
        truth,
        folds,
        target_columns=wave2.JIGSAW_TARGET_COLUMNS,
        feature_names=names,
        c_grid=[0.1],
        max_iter=120,
        seed=92,
    )
    base_counts = {
        "word": np.ones_like(counts),
        "char_wb": np.ones_like(counts),
        "rank_blend": np.ones_like(counts),
    }
    provenance = wave2.validate_jigsaw_prediction_provenance(
        target_columns=wave2.JIGSAW_TARGET_COLUMNS,
        fold_assignment=folds,
        base_write_counts=base_counts,
        stacker_write_counts=counts,
        stacker_contract=contract,
    )
    assert provenance["passed"] is True
    broken = counts.copy()
    broken[0, 0] = 0
    broken[1, 0] = 2
    with pytest.raises(RuntimeError, match="exact-once"):
        wave2.validate_jigsaw_prediction_provenance(
            target_columns=wave2.JIGSAW_TARGET_COLUMNS,
            fold_assignment=folds,
            base_write_counts=base_counts,
            stacker_write_counts=broken,
            stacker_contract=contract,
        )


def test_jigsaw_prediction_bundle_is_content_addressed_and_pickle_free(tmp_path):
    truth, folds, features, test_features, names = _synthetic_jigsaw_stacker_fixture()
    oof_stacker, test_stacker, counts, contract = wave2.cross_fit_jigsaw_label_stacker(
        features,
        test_features,
        truth,
        folds,
        target_columns=wave2.JIGSAW_TARGET_COLUMNS,
        feature_names=names,
        c_grid=[0.1],
        max_iter=120,
        seed=93,
    )
    rng = np.random.default_rng(93)
    word = np.clip(0.15 + 0.7 * truth, 0.001, 0.999)
    char = np.clip(0.20 + 0.6 * truth, 0.001, 0.999)
    baseline = 0.5 * word + 0.5 * char
    test_word = rng.uniform(0.01, 0.99, size=(3, len(test_stacker), truth.shape[1]))
    test_char = rng.uniform(0.01, 0.99, size=test_word.shape)
    test_blend = wave2.build_jigsaw_fold_test_blends(
        test_word, test_char, np.full((3, truth.shape[1]), 0.5)
    )
    base_counts = {
        "word": np.ones_like(counts),
        "char_wb": np.ones_like(counts),
        "rank_blend": np.ones_like(counts),
    }
    provenance = wave2.validate_jigsaw_prediction_provenance(
        target_columns=wave2.JIGSAW_TARGET_COLUMNS,
        fold_assignment=folds,
        base_write_counts=base_counts,
        stacker_write_counts=counts,
        stacker_contract=contract,
    )
    gate = {"passed": True, "checks": {"fixture": True}}
    record = wave2.save_jigsaw_prediction_bundle(
        tmp_path,
        target_columns=wave2.JIGSAW_TARGET_COLUMNS,
        train_ids=np.asarray([f"train-{index}" for index in range(len(truth))]),
        test_ids=np.asarray([f"test-{index}" for index in range(len(test_stacker))]),
        truth=truth,
        fold_assignment=folds,
        oof_word=word,
        oof_char=char,
        oof_baseline_blend=baseline,
        test_word_by_fold=test_word,
        test_char_by_fold=test_char,
        test_blend_by_fold=test_blend,
        oof_stacker=oof_stacker,
        test_stacker=test_stacker,
        base_write_counts=base_counts,
        stacker_write_counts=counts,
        stacker_contract=contract,
        provenance=provenance,
        promotion_gate=gate,
    )
    bundle = tmp_path / record["bundle"]
    manifest = tmp_path / record["manifest"]
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == record["bundle_sha256"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == record["manifest_sha256"]
    with np.load(bundle, allow_pickle=False) as arrays:
        assert arrays["oof_by_channel"].shape == (3, len(truth), truth.shape[1])
        assert arrays["test_by_fold_channel"].shape == (
            3,
            3,
            len(test_stacker),
            truth.shape[1],
        )
        assert np.all(arrays["stacker_write_counts"] == 1)


def test_all_eight_promoted_runners_bind_and_forward_frozen_gates():
    vision_runners = (
        wave2.run_aptos,
        wave2.run_dog_breed,
        wave2.run_histopath,
        wave2.run_plant,
        wave2.run_ranzcr,
    )
    for runner in vision_runners:
        assert "promotion_contract=WAVE2_PROMOTION_CONTRACTS[cid]" in inspect.getsource(runner)
    vision_source = inspect.getsource(wave2._run_vision)
    assert "promotion_gate=promotion_gate" in vision_source

    for runner in (wave2.run_jigsaw, wave2.run_birds, wave2.run_whale):
        source = inspect.getsource(runner)
        assert "promotion_gate =" in source
        assert "promotion_gate=promotion_gate" in source


def test_nomad_feature_engineering_is_finite():
    frame = pd.DataFrame({
        "id": [1], "spacegroup": [33], "number_of_total_atoms": [40],
        "percent_atom_al": [0.5], "percent_atom_ga": [0.1], "percent_atom_in": [0.4],
        "lattice_vector_1_ang": [5.2], "lattice_vector_2_ang": [8.9], "lattice_vector_3_ang": [9.5],
        "lattice_angle_alpha_degree": [90], "lattice_angle_beta_degree": [90],
        "lattice_angle_gamma_degree": [90],
    })
    features = wave2._nomad_features(frame)
    assert features.loc[0, "cell_volume"] > 0
    assert np.isfinite(features.to_numpy()).all()


def test_nomad_process_pool_preserves_input_order(monkeypatch, tmp_path):
    paths = []
    for value in (3, 1, 2):
        path = tmp_path / str(value) / "geometry.xyz"
        path.parent.mkdir()
        path.write_text("fixture", encoding="utf-8")
        paths.append(path)

    observed = {}

    def fake_features(path):
        return {"geom_marker": float(path.parent.name)}

    class RecordingPool:
        def __init__(self, *, max_workers):
            observed["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def map(self, function, values, *, chunksize):
            observed["chunksize"] = chunksize
            return map(function, values)

    monkeypatch.setattr(wave2, "nomad_structure_features", fake_features)
    monkeypatch.setattr(wave2, "ProcessPoolExecutor", RecordingPool)
    rows = wave2._parallel_nomad_structure_features(paths, workers=16)

    assert [row["geom_marker"] for row in rows] == [3.0, 1.0, 2.0]
    assert observed == {"max_workers": 3, "chunksize": 1}


def test_normalization_map_and_english_rules():
    frame = pd.DataFrame({
        "before": ["Dr.", "Dr.", "Dr.", "same"],
        "after": ["doctor", "doctor", "drive", "same"],
    })
    mapping = wave2.build_normalization_map(frame)
    assert mapping == {"Dr.": "doctor", "same": "same"}
    assert wave2.normalize_token("Dr.", mapping, "english") == "doctor"
    assert wave2.normalize_token("42", mapping, "english") == "forty two"
    assert wave2.normalize_token("12.5", mapping, "english") == "twelve point five"
    assert wave2.normalize_token("42", {}, "russian") == "\u0441\u043e\u0440\u043e\u043a \u0434\u0432\u0430"
    assert wave2.english_integer_words(1_000_001) == "one million one"
    assert wave2.normalize_token("21st", {}, "english") == "twenty first"
    assert wave2.normalize_token("3/4", {}, "english") == "three fourths"


def test_english_normalization_extended_rule_families():
    examples = {
        "2006": "two thousand six",
        "1987": "nineteen eighty seven",
        "4 March 2014": "the fourth of march twenty fourteen",
        "April 10, 2013": "april tenth twenty thirteen",
        "August 1999": "august nineteen ninety nine",
        "Sep 7, 2009": "september seventh two thousand nine",
        "24 April 350": "the twenty fourth of april three fifty",
        "Thursday 17 February 2011": "thursday the seventeenth of february twenty eleven",
        "7 May,": "the seventh of may",
        "1830s": "eighteen thirties",
        "IUCN": "i u c n",
        "BC": "b c",
        "C.": "c",
        "ZRs": "z r's",
        "200-": "two o o",
        "08": "o eight",
        "007": "double o seven",
        "978-0-253-34916-3": "nine seven eight sil o sil two five three sil three four nine one six sil three",
        "animalsvoice.com": "a n i m a l s v o i c e dot c o m",
        "60 km": "sixty kilometers",
        "79.20%": "seventy nine point two o percent",
        "459.0/km²": "four hundred fifty nine point o per square kilometers",
        "$1": "one dollar",
        "$22,750": "twenty two thousand seven hundred fifty dollars",
        "$1,096,796.23": "one million ninety six thousand seven hundred ninety six dollars and twenty three cents",
        ".161": "point one six one",
    }
    for before, expected in examples.items():
        assert wave2.normalize_token(before, {}, "english") == expected
    assert wave2.normalize_token("XIV", {}, "english", "ORDINAL") == "fourteenth"


def test_text_normalization_submission_alignment_is_id_keyed_and_strict():
    sample = pd.DataFrame({"id": ["8_2", "7_1", "7_0"], "after": ["", "", ""]})
    test = pd.DataFrame({
        "sentence_id": ["7", "8", "7"],
        "token_id": ["0", "2", "1"],
        "before": ["A", "B", "C"],
    })
    aligned = wave2.align_text_normalization_submission_by_id(
        sample,
        test,
        ["alpha", "bravo", "charlie"],
    )
    assert aligned["id"].tolist() == ["8_2", "7_1", "7_0"]
    assert aligned["after"].tolist() == ["bravo", "charlie", "alpha"]

    with pytest.raises(RuntimeError, match="not unique"):
        wave2.align_text_normalization_submission_by_id(
            sample,
            pd.concat([test.iloc[:1], test.iloc[:1], test.iloc[2:]], ignore_index=True),
            ["a", "b", "c"],
        )

    with pytest.raises(RuntimeError, match="ID sets differ"):
        wave2.align_text_normalization_submission_by_id(
            sample,
            test.assign(token_id=["0", "3", "1"]),
            ["a", "b", "c"],
        )


def test_text_normalization_empty_prediction_survives_csv_round_trip():
    sample = pd.DataFrame({"id": ["8_2", "7_1", "7_0"], "after": ["", "", ""]})
    test = pd.DataFrame({
        "sentence_id": ["7", "8", "7"],
        "token_id": ["0", "2", "1"],
        "before": ["A", "", "C"],
    })

    aligned = wave2.align_text_normalization_submission_by_id(
        sample,
        test,
        ["alpha", "", "charlie"],
    )

    assert aligned["id"].tolist() == ["8_2", "7_1", "7_0"]
    assert aligned["after"].tolist() == [" ", "charlie", "alpha"]
    round_tripped = pd.read_csv(StringIO(aligned.to_csv(index=False)), dtype=str)
    assert round_tripped["after"].isna().sum() == 0
    assert round_tripped.to_dict("records") == aligned.to_dict("records")


def test_russian_normalization_extended_rule_families():
    assert wave2.russian_integer_words(1999) == "тысяча девятьсот девяносто девять"
    examples = {
        "2010 года": "две тысячи десятого года",
        "1999 году": "тысяча девятьсот девяносто девятом году",
        "12 февраля 2013": "двенадцатого февраля две тысячи тринадцатого года",
        "30.10.1943": "тридцатое октября тысяча девятьсот сорок третьего года",
        "1954 гг.": "тысяча девятьсот пятьдесят четвертый год",
        "1-й": "первый",
        "1-го": "первого",
        "22-х": "двадцать вторых",
        "1,5": "одна целая и пять десятых",
        "0,3": "ноль целых и три десятых",
        "432 с.": "четыреста тридцать две секунды",
        "82 т.": "восемьдесят две тонны",
        "10 %": "десять процентов",
        "ISBN": "i s b n",
        "СССР": "с с с р",
        "08": "ноль восемь",
        "5-200-00643-0": "пять sil двести sil ноль ноль шестьсот сорок три sil ноль",
        "3/48": "три сорок восьмых",
        "20:06": "двадцать часов шесть минут",
        "11:15 UTC": "одиннадцать часов пятнадцать минут по часовому поясу u t c",
        "12 км": "двенадцать километров",
    }
    for before, expected in examples.items():
        assert wave2.normalize_token(before, {}, "russian") == expected


def test_russian_contextual_rules_and_frame_boundaries():
    assert wave2.russian_cardinal_genitive_words(25) == "двадцати пяти"
    assert wave2.russian_contextual_token_words("25", "от", "—") == "двадцати пяти"
    assert wave2.russian_contextual_token_words("2 км", "в", "от") == "двух километрах"
    assert wave2.russian_contextual_token_words("XIX", "в", "веке") == "девятнадцатом"
    assert wave2.russian_contextual_token_words("2007", "по", "год") == "две тысячи седьмой"
    assert wave2.russian_contextual_token_words("1927", "с", "по") == "тысяча девятьсот двадцать седьмого"
    frame = pd.DataFrame({
        "sentence_id": ["1", "1", "1", "2"],
        "before": ["от", "25", "—", "3"],
    })
    assert wave2.normalize_frame_tokens(frame, {}, "russian") == [
        "от", "двадцати пяти", "—", "три",
    ]
    context_train = pd.DataFrame({
        "sentence_id": ["a", "a", "b", "b", "c", "c"],
        "before": ["около", "10", "около", "10", "ровно", "10"],
        "after": ["около", "десяти", "около", "десяти", "ровно", "десять"],
    })
    context_maps = wave2.build_normalization_context_maps(context_train)
    context_test = pd.DataFrame({"sentence_id": ["z", "z"], "before": ["около", "10"]})
    assert wave2.normalize_frame_tokens(context_test, {"10": "десять"}, "russian", context_maps) == [
        "около", "десяти",
    ]


def test_russian_transliteration_pair_extraction_and_application(monkeypatch):
    assert ru_transliteration.extract_russian_transliteration_target(
        "т_trans е_trans х_trans а_trans с_trans"
    ) == "техас"
    assert ru_transliteration.extract_russian_transliteration_target("т е х а с") is None
    training = pd.DataFrame({
        "before": ["Texas", "Texas", "Texas", "plain"],
        "after": [
            "т_trans е_trans х_trans а_trans с_trans",
            "т_trans е_trans х_trans а_trans с_trans",
            "т_trans э_trans к_trans с_trans а_trans с_trans",
            "plain",
        ],
    })
    assert ru_transliteration.collect_russian_transliteration_pairs(training) == [("texas", "техас")]

    def fake_predict(*args, **kwargs):
        return {"texas": "т_trans е_trans х_trans а_trans с_trans"}, {"training_pairs": 1}

    monkeypatch.setattr(wave2.ru_transliteration, "predict_russian_transliterations", fake_predict)
    prediction_frame = pd.DataFrame({"before": ["Texas", "known"]})
    result, diagnostics = wave2.apply_russian_transliteration_model(
        training,
        prediction_frame,
        {"known": "известный"},
        ["Texas", "известный"],
        epochs=40,
        seed=42,
        device="cpu",
    )
    assert result == ["т_trans е_trans х_trans а_trans с_trans", "известный"]
    assert diagnostics["applied_rows"] == 1


def test_russian_transliterator_uses_matching_boolean_masks_and_bounded_threads():
    import inspect

    source = inspect.getsource(ru_transliteration.predict_russian_transliterations)
    assert "dtype=torch.bool" in source
    assert "generate_square_subsequent_mask" not in source
    assert "torch.set_num_threads(min(8" in source
    assert '"epochs_completed": len(history)' in source


def test_ordinal_thresholds_are_monotonic_and_score_well():
    truth = np.repeat(np.arange(5), 10)
    expected = truth.astype(float) + np.linspace(-0.1, 0.1, len(truth))
    thresholds = wave2.optimize_ordinal_thresholds(expected, truth)
    assert thresholds == sorted(thresholds)
    prediction = np.digitize(expected, thresholds)
    assert wave2.compute_metric("quadratic_weighted_kappa", truth, prediction) > 0.99


def test_audio_features_are_fixed_width_and_finite(tmp_path):
    rate = 16_000
    time = np.arange(rate, dtype=np.float32) / rate
    signal = (np.sin(2 * np.pi * 440 * time) * 30_000).astype(np.int16)
    path = tmp_path / "tone.wav"
    wavfile.write(path, rate, signal)
    features = wave2.audio_features(path)
    assert features.shape == (wave2.AUDIO_FEATURE_WIDTH,)
    assert np.isfinite(features).all()


def test_audio_features_are_sample_rate_stable_for_same_tone():
    one_second_8k = np.arange(8_000, dtype=np.float32) / 8_000
    one_second_16k = np.arange(16_000, dtype=np.float32) / 16_000
    low_rate = np.sin(2 * np.pi * 440 * one_second_8k).astype(np.float32)
    high_rate = np.sin(2 * np.pi * 440 * one_second_16k).astype(np.float32)
    low_features = wave2.extract_audio_feature_vector(low_rate, 8_000)
    high_features = wave2.extract_audio_feature_vector(high_rate, 16_000)
    cosine = float(np.dot(low_features, high_features) / (
        np.linalg.norm(low_features) * np.linalg.norm(high_features)
    ))
    assert cosine > 0.98


def test_nomad_geometry_features_are_fixed_and_periodic(tmp_path):
    geometry = tmp_path / "geometry.xyz"
    geometry.write_text(
        "\n".join([
            "lattice_vector 4 0 0",
            "lattice_vector 0 4 0",
            "lattice_vector 0 0 4",
            "atom 0 0 0 Al",
            "atom 3.9 0 0 O",
        ]),
        encoding="utf-8",
    )
    cell, atoms, elements = wave2.parse_nomad_geometry(geometry)
    assert cell.shape == (3, 3)
    assert atoms.shape == (2, 3)
    assert elements == ("Al", "O")
    features = wave2.nomad_structure_features(geometry)
    assert features["geom_volume"] == pytest.approx(64.0)
    assert features["geom_atom_count"] == 2.0
    assert features["geom_pair_all_min"] == pytest.approx(0.1, abs=1e-8)
    assert np.isfinite(np.fromiter(features.values(), dtype=float)).all()


def test_nomad_geometry_parser_converts_fractional_coordinates(tmp_path):
    geometry = tmp_path / "geometry.xyz"
    geometry.write_text(
        "\n".join([
            "lattice_vector 4 0 0",
            "lattice_vector 0 6 0",
            "lattice_vector 0 0 8",
            "atom_frac 0.5 0.25 0.125 In",
            "atom_frac 0 0 0 O",
        ]),
        encoding="utf-8",
    )
    cell, atoms, elements = wave2.parse_nomad_geometry(geometry)
    np.testing.assert_allclose(cell, np.diag([4.0, 6.0, 8.0]))
    np.testing.assert_allclose(atoms[0], [2.0, 1.5, 1.0])
    np.testing.assert_allclose(atoms[1], [0.0, 0.0, 0.0])
    assert elements == ("In", "O")
    features = wave2.nomad_structure_features(geometry)
    assert features["geom_atom_count"] == 2.0
    assert np.isfinite(np.fromiter(features.values(), dtype=float)).all()


def test_nomad_blend_is_cross_fitted_and_complete():
    truth = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    catboost = np.asarray([0.11, 0.18, 0.29, 0.45, 0.47, 0.62])
    extra_trees = np.asarray([0.08, 0.23, 0.34, 0.38, 0.54, 0.57])
    folds = np.asarray([0, 0, 1, 1, 2, 2])
    prediction, records = wave2.cross_fit_nomad_log_blend(
        truth, catboost, extra_trees, folds
    )
    assert prediction.shape == truth.shape
    assert np.isfinite(prediction).all()
    assert [record["fold"] for record in records] == [0.0, 1.0, 2.0]
    assert all(0.0 <= record["catboost_weight"] <= 1.0 for record in records)
    assert all(record["extra_trees_weight"] == pytest.approx(1.0 - record["catboost_weight"]) for record in records)


def test_nb_log_count_ratio_matches_manual_formula():
    matrix = csr_matrix(np.asarray([
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ], dtype=np.float32))
    labels = np.asarray([1, 1, 0, 0])
    ratio = wave2.compute_nb_log_count_ratio(matrix, labels)
    positive = np.asarray([(1 + 2) / 4, (1 + 1) / 4, (1 + 1) / 4])
    negative = np.asarray([(1 + 0) / 4, (1 + 1) / 4, (1 + 2) / 4])
    np.testing.assert_allclose(ratio, np.log(positive / negative), rtol=1e-6)


def test_multilabel_stratified_folds_are_deterministic_balanced_and_complete():
    rows = np.arange(120)
    truth = np.column_stack([
        rows % 2 == 0,
        rows % 4 == 0,
        rows % 10 == 0,
        (rows % 7 == 0) | (rows % 11 == 0),
    ]).astype(np.int8)
    splits, assignment = wave2.make_multilabel_stratified_folds(
        truth,
        requested_folds=5,
        seed=42,
    )
    repeated, repeated_assignment = wave2.make_multilabel_stratified_folds(
        truth,
        requested_folds=5,
        seed=42,
    )
    assert np.array_equal(assignment, repeated_assignment)
    assert len(splits) == len(repeated) == 5
    assert sorted(np.concatenate([valid for _, valid in splits]).tolist()) == rows.tolist()
    assert max(np.bincount(assignment)) - min(np.bincount(assignment)) <= 1
    for train_indices, valid_indices in splits:
        assert not set(train_indices) & set(valid_indices)
        for label in range(truth.shape[1]):
            assert set(np.unique(truth[valid_indices, label])) == {0, 1}


def test_multilabel_rank_blend_is_cross_fitted_and_label_specific():
    rng = np.random.RandomState(7)
    rows = np.arange(150)
    truth = np.column_stack([
        rows % 2 == 0,
        rows % 3 == 0,
    ]).astype(np.int8)
    _, fold = wave2.make_multilabel_stratified_folds(
        truth,
        requested_folds=5,
        seed=9,
    )
    word = np.column_stack([
        truth[:, 0] + rng.normal(0, 0.12, len(rows)),
        rng.normal(0, 1, len(rows)),
    ])
    char = np.column_stack([
        rng.normal(0, 1, len(rows)),
        truth[:, 1] + rng.normal(0, 0.12, len(rows)),
    ])
    test_word = rng.normal(size=(31, 2))
    test_char = rng.normal(size=(31, 2))
    cross_fitted, test_blend, fold_weights, final_weights = (
        wave2.cross_fit_multilabel_rank_blend(
            word,
            char,
            test_word,
            test_char,
            truth,
            fold,
        )
    )
    assert cross_fitted.shape == truth.shape
    assert test_blend.shape == (31, 2)
    assert fold_weights.shape == (5, 2)
    assert np.isfinite(cross_fitted).all() and np.isfinite(test_blend).all()
    assert np.all((fold_weights >= 0) & (fold_weights <= 1))
    assert final_weights[0] >= 0.5
    assert final_weights[1] <= 0.5


def test_grouped_multilabel_vision_splits_are_disjoint_and_scoreable():
    groups = np.repeat(np.asarray([f"p{index}" for index in range(12)]), 2)
    group_index = np.repeat(np.arange(12), 2)
    truth = np.column_stack([
        group_index % 2 == 0,
        group_index % 3 == 0,
        group_index % 4 < 2,
    ]).astype(np.int8)
    splits, strategy = wave2.make_vision_splits(
        sample_count=len(groups),
        requested_folds=3,
        seed=42,
        mode="multilabel",
        target=truth,
        groups=groups,
    )
    assert strategy == "multilabel_stratified_group_kfold"
    assert wave2.vision_splits_are_scoreable(
        mode="multilabel", target=truth, splits=splits
    )
    for train_indices, valid_indices in splits:
        assert not set(groups[train_indices]) & set(groups[valid_indices])


def test_multilabel_submission_alignment_is_id_keyed():
    sample = pd.DataFrame({
        "id": ["b", "a", "c"],
        "toxic": [0.0, 0.0, 0.0],
        "threat": [0.0, 0.0, 0.0],
    })
    test = pd.DataFrame({"id": ["a", "b", "c"], "comment_text": ["A", "B", "C"]})
    prediction = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    aligned = wave2.align_multilabel_submission_by_id(
        sample,
        test,
        prediction,
        id_column="id",
        target_columns=["toxic", "threat"],
    )
    assert aligned["id"].tolist() == ["b", "a", "c"]
    np.testing.assert_allclose(
        aligned[["toxic", "threat"]].to_numpy(),
        [[0.3, 0.4], [0.1, 0.2], [0.5, 0.6]],
    )


def test_vision_splits_fall_back_when_all_rows_share_one_group():
    splits, strategy = wave2.make_vision_splits(
        sample_count=6,
        requested_folds=5,
        seed=42,
        mode="multilabel",
        target=np.zeros((6, 2), dtype=np.float32),
        groups=np.asarray(["same-patient"] * 6),
    )
    assert strategy == "kfold_single_group_fallback"
    assert len(splits) == 5
    assert sorted(np.concatenate([valid for _, valid in splits]).tolist()) == list(range(6))


def test_vision_splits_reject_single_row():
    with pytest.raises(RuntimeError, match="at least two"):
        wave2.make_vision_splits(
            sample_count=1,
            requested_folds=5,
            seed=42,
            mode="multilabel",
            target=np.zeros((1, 2), dtype=np.float32),
        )


def test_vision_splits_keep_classification_groups_disjoint():
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    target = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    splits, strategy = wave2.make_vision_splits(
        sample_count=len(target),
        requested_folds=2,
        seed=42,
        mode="binary",
        target=target,
        groups=groups,
    )
    assert strategy == "stratified_group_kfold"
    for fit_indices, valid_indices in splits:
        assert set(groups[fit_indices]).isdisjoint(set(groups[valid_indices]))


def test_binary_logloss_calibration_is_cross_fitted_and_improves_overconfidence():
    rng = np.random.default_rng(123)
    latent = rng.normal(size=1_000)
    truth = (rng.random(1_000) < wave2._sigmoid_array(latent)).astype(float)
    overconfident_logits = 4.0 * latent + 0.8
    folds = np.arange(len(truth)) % 5
    base_test_logits = np.linspace(-12.0, 12.0, 41)
    test_logits = np.column_stack(
        [base_test_logits + offset for offset in (-0.4, -0.2, 0.0, 0.2, 0.4)]
    )
    calibrated_oof, calibrated_test, records, final = (
        wave2.cross_fit_binary_logloss_calibration(
            overconfident_logits,
            test_logits,
            truth,
            folds,
        )
    )
    raw_loss = wave2._binary_log_loss_array(
        truth,
        wave2._sigmoid_array(overconfident_logits),
    )
    calibrated_loss = wave2._binary_log_loss_array(truth, calibrated_oof)
    assert calibrated_loss < raw_loss
    assert len(records) == 5
    assert final["cross_fitted_log_loss"] == pytest.approx(calibrated_loss)
    assert final["test_component_count"] == 5
    assert final["test_aggregation"] == "per_fold_calibrate_then_probability_average"
    expected_test = np.mean(
        np.column_stack(
            [
                wave2.apply_binary_temperature_intercept(test_logits[:, index], record)
                for index, record in enumerate(records)
            ]
        ),
        axis=1,
    )
    np.testing.assert_allclose(calibrated_test, expected_test)
    assert np.isfinite(calibrated_oof).all() and np.isfinite(calibrated_test).all()
    assert np.all((calibrated_test > 0.0) & (calibrated_test < 1.0))


def test_binary_logloss_calibration_rejects_incomplete_folds():
    with pytest.raises(RuntimeError, match="multiple folds"):
        wave2.cross_fit_binary_logloss_calibration(
            np.asarray([-1.0, -0.5, 0.5, 1.0]),
            np.asarray([0.0]),
            np.asarray([0.0, 0.0, 1.0, 1.0]),
            np.zeros(4, dtype=int),
        )


def test_binary_logloss_calibration_rejects_wrong_test_component_count():
    with pytest.raises(RuntimeError, match="one component per OOF fold"):
        wave2.cross_fit_binary_logloss_calibration(
            np.asarray([-1.0, -0.5, 0.5, 1.0]),
            np.zeros((3, 3)),
            np.asarray([0.0, 0.0, 1.0, 1.0]),
            np.asarray([0, 1, 0, 1]),
        )


def test_vision_backbone_weight_identities_are_fully_pinned():
    assert wave2.VISION_BACKBONE_SPECS == {
        "convnext_tiny": {
            "filename": "convnext_tiny-983f1562.pth",
            "sha256": "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d",
            "classifier": "classifier.2",
        },
        "convnext_small": {
            "filename": "convnext_small-0c510722.pth",
            "sha256": "0c510722adfd92966a2bd72b92f785ca05966bbac03cafe2f7a90b1f54bfab9a",
            "classifier": "classifier.2",
        },
        "efficientnet_v2_s": {
            "filename": "efficientnet_v2_s-dd5fe13b.pth",
            "sha256": "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74",
            "classifier": "classifier.1",
        },
    }
    assert all(len(spec["sha256"]) == 64 for spec in wave2.VISION_BACKBONE_SPECS.values())
    assert wave2.VISION_DATALOADER_PREFETCH_FACTOR == 4


def test_a40_vision_runners_bind_reviewed_backbones_and_batches():
    expected = {
        wave2.run_histopath: ("efficientnet_v2_s", 192, 128),
        wave2.run_plant: ("convnext_small", 448, 32),
        wave2.run_aptos: ("efficientnet_v2_s", 512, 24),
    }
    for runner, (backbone, image_size, batch_size) in expected.items():
        source = inspect.getsource(runner)
        assert f'backbone="{backbone}"' in source
        assert f"image_size={image_size}" in source
        assert f"batch_size={batch_size}" in source

    dog_breed_source = inspect.getsource(wave2.run_dog_breed)
    assert 'getattr(args, "wave2_dog_breed_backbone", "convnext_small")' in dog_breed_source
    assert "image_size=384" in dog_breed_source
    assert "batch_size=args.wave2_dog_breed_batch_size" in dog_breed_source

    ranzcr_source = inspect.getsource(wave2.run_ranzcr)
    assert 'getattr(args, "wave2_ranzcr_backbone", "efficientnet_v2_s")' in ranzcr_source
    assert 'getattr(args, "wave2_ranzcr_image_size", 512)' in ranzcr_source
    assert 'getattr(args, "wave2_ranzcr_batch_size", 24)' in ranzcr_source
    assert "worker_count=32" in ranzcr_source
    histopath_source = inspect.getsource(wave2.run_histopath)
    assert "workers=max(32, args.wave2_workers)" in histopath_source
    assert "worker_count=32" in histopath_source

    vision_source = inspect.getsource(wave2._run_vision)
    assert '"backbone": backbone' in vision_source
    assert 'f"{backbone}_ImageNet1K_fold_ensemble"' in vision_source


def test_vision_model_is_cache_only_and_never_downloads_missing_optional_weights():
    source = inspect.getsource(wave2._vision_model)
    missing_cache_branch = source.split("if not weight_path.is_file():", 1)[1].split(
        "if weight_path.is_file()", 1
    )[0]
    assert "model_factory(weights=None)" in missing_cache_branch
    assert "model_factory(weights=weights)" not in missing_cache_branch


def test_imagenet_classifier_row_initialization_copies_exact_rows():
    torch = pytest.importorskip("torch")

    source = torch.nn.Linear(4, 6)
    with torch.no_grad():
        source.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(6, 4))
        source.bias.copy_(torch.arange(6, dtype=torch.float32) + 100)

    replacement, report = wave2.initialize_classifier_from_imagenet_rows(
        source,
        3,
        [4, 1, 5],
    )

    assert torch.equal(replacement.weight, source.weight[[4, 1, 5]])
    assert torch.equal(replacement.bias, source.bias[[4, 1, 5]])
    assert replacement.weight.data_ptr() != source.weight.data_ptr()
    assert report == {
        "mode": "exact_pretrained_imagenet_classifier_rows",
        "output_count": 3,
        "source_output_count": 6,
        "source_indices": [4, 1, 5],
        "unique_source_rows": True,
        "weights_copied": True,
        "bias_copied": True,
    }


@pytest.mark.parametrize("indices", [[0, 0, 1], [0, 1], [0, 1, 6]])
def test_imagenet_classifier_row_initialization_rejects_invalid_indices(indices):
    torch = pytest.importorskip("torch")

    source = torch.nn.Linear(4, 6)
    with pytest.raises(ValueError):
        wave2.initialize_classifier_from_imagenet_rows(source, 3, indices)


def test_dog_vision_model_uses_teacher_mapping_for_classifier_initialization():
    source = inspect.getsource(wave2._run_vision)
    assert "dog_breed_classifier_source_indices" in source
    assert "classifier_source_indices=dog_breed_classifier_source_indices" in source
    assert "imagenet_classifier_rows_initialized_all_folds" in source
    assert "pretrained_backbone_0.1x_classifier_1.0x" in source
    assert "dog_breed_optimizer_contract_all_folds" in source
    assert "cross_fit_dog_breed_class_bias_calibration" in source
    assert "select_dog_breed_probability_replacement" in source


def test_dog_breed_imagenet_mapping_is_complete_and_cardigan_is_disambiguated():
    categories = [
        "tench",
        "cardigan",
        "Chihuahua",
        "Japanese spaniel",
        "Cardigan",
        "African hunting dog",
        "soccer ball",
    ]

    indices, report = wave2.build_dog_breed_imagenet_mapping(
        ["chihuahua", "cardigan", "african_hunting_dog"],
        categories,
    )

    np.testing.assert_array_equal(indices, np.asarray([2, 4, 5], dtype=np.int64))
    assert report["coverage_count"] == 3
    assert report["class_count"] == 3
    assert report["coverage_fraction"] == pytest.approx(1.0)
    assert report["category_names_by_class"]["cardigan"] == "Cardigan"
    assert report["canine_span"] == [2, 5]


def test_dog_breed_imagenet_mapping_fails_closed_for_missing_classes():
    categories = ["Chihuahua", "Japanese spaniel", "African hunting dog"]

    with pytest.raises(RuntimeError, match="incomplete or ambiguous"):
        wave2.build_dog_breed_imagenet_mapping(["chihuahua", "unknown_breed"], categories)


def test_dog_breed_probability_blend_normalizes_and_crossfits_teacher_gain():
    fine_tuned = np.asarray([
        [0.40, 0.30, 0.30],
        [0.38, 0.31, 0.31],
        [0.30, 0.40, 0.30],
        [0.31, 0.38, 0.31],
        [0.30, 0.30, 0.40],
        [0.31, 0.31, 0.38],
    ])
    teacher = np.asarray([
        [0.96, 0.02, 0.02],
        [0.95, 0.03, 0.02],
        [0.02, 0.96, 0.02],
        [0.03, 0.95, 0.02],
        [0.02, 0.02, 0.96],
        [0.02, 0.03, 0.95],
    ])
    target = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    folds = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int16)

    blended = wave2.apply_dog_breed_probability_blend(
        fine_tuned,
        teacher,
        fine_tuned_weight=0.25,
        temperature=1.0,
    )
    assert np.isfinite(blended).all()
    np.testing.assert_allclose(blended.sum(axis=1), 1.0)

    selected, parameters = wave2.select_dog_breed_probability_blend(
        fine_tuned,
        teacher,
        target,
    )
    assert parameters["teacher_weight"] > 0.0
    assert parameters["blend_family"] in wave2.DOG_BREED_BLEND_FAMILIES
    assert parameters["candidate_count"] > 0
    assert parameters["search_contract"] == "joint_family_weight_temperature_coarse_to_fine_v1"
    assert wave2._indexed_multiclass_log_loss(target, selected) < parameters["fine_tuned_only_log_loss"]

    crossfit, records = wave2.cross_fit_dog_breed_probability_blend(
        fine_tuned,
        teacher,
        target,
        folds,
    )
    assert len(records) == 2
    assert all(record["blend_family"] in wave2.DOG_BREED_BLEND_FAMILIES for record in records)
    assert {record["fold"] for record in records} == {0, 1}
    assert sum(record["validation_rows"] for record in records) == len(target)
    assert wave2._indexed_multiclass_log_loss(target, crossfit) < wave2._indexed_multiclass_log_loss(
        target,
        fine_tuned,
    )
    np.testing.assert_allclose(crossfit.sum(axis=1), 1.0)


def test_dog_breed_blend_families_match_their_closed_forms():
    fine_tuned = np.asarray([[0.70, 0.20, 0.10], [0.15, 0.75, 0.10]], dtype=np.float64)
    teacher = np.asarray([[0.45, 0.40, 0.15], [0.30, 0.55, 0.15]], dtype=np.float64)
    weight = 0.35
    temperature = 0.8

    arithmetic = wave2.apply_dog_breed_probability_blend(
        fine_tuned,
        teacher,
        fine_tuned_weight=weight,
        temperature=temperature,
        blend_family="arithmetic_probability",
    )
    arithmetic_logits = np.log(weight * fine_tuned + (1.0 - weight) * teacher) / temperature
    arithmetic_expected = np.exp(arithmetic_logits - arithmetic_logits.max(axis=1, keepdims=True))
    arithmetic_expected /= arithmetic_expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(arithmetic, arithmetic_expected, atol=1e-12)

    geometric = wave2.apply_dog_breed_probability_blend(
        fine_tuned,
        teacher,
        fine_tuned_weight=weight,
        temperature=temperature,
        blend_family="geometric_log_probability",
    )
    geometric_logits = (
        weight * np.log(fine_tuned) + (1.0 - weight) * np.log(teacher)
    ) / temperature
    geometric_expected = np.exp(geometric_logits - geometric_logits.max(axis=1, keepdims=True))
    geometric_expected /= geometric_expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(geometric, geometric_expected, atol=1e-12)
    np.testing.assert_allclose(geometric.sum(axis=1), 1.0, atol=1e-12)


def test_dog_breed_blend_rejects_unknown_family():
    probability = np.asarray([[0.6, 0.4]], dtype=np.float64)
    with pytest.raises(ValueError, match="blend family"):
        wave2.apply_dog_breed_probability_blend(
            probability,
            probability,
            fine_tuned_weight=0.5,
            temperature=1.0,
            blend_family="unknown",
        )


def test_dog_breed_class_bias_outer_predictions_ignore_outer_labels():
    rng = np.random.RandomState(20260726)
    truth = np.tile(np.arange(4, dtype=np.int64), 30)
    folds = np.arange(len(truth), dtype=np.int16) % 5
    logits = rng.normal(0.0, 0.08, size=(len(truth), 4))
    logits[np.arange(len(truth)), truth] += 2.4
    logits += np.asarray([1.1, -0.7, 0.35, -0.75])
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)

    baseline, records, report = wave2.cross_fit_dog_breed_class_bias_calibration(
        probability,
        truth,
        folds,
        enforce_early_checkpoint=False,
    )
    perturbed_truth = truth.copy()
    perturbed_truth[folds == 0] = (perturbed_truth[folds == 0] + 1) % 4
    perturbed, _, perturbed_report = wave2.cross_fit_dog_breed_class_bias_calibration(
        probability,
        perturbed_truth,
        folds,
        enforce_early_checkpoint=False,
    )

    assert report["candidate_complete"] is True
    assert perturbed_report["candidate_complete"] is True
    assert len(records) == 5
    np.testing.assert_allclose(
        baseline[folds == 0],
        perturbed[folds == 0],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(baseline.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert all(record["private_labels_used"] is False for record in records)


def test_dog_breed_replacement_fallback_preserves_incumbent_bit_for_bit():
    truth = np.tile(np.arange(3, dtype=np.int64), 10)
    folds = np.arange(len(truth), dtype=np.int16) % 5
    incumbent = np.full((len(truth), 3), 0.025, dtype=np.float64)
    incumbent[np.arange(len(truth)), truth] = 0.95
    candidate = np.full_like(incumbent, 1.0 / 3.0)

    selected, report = wave2.select_dog_breed_probability_replacement(
        incumbent,
        candidate,
        truth,
        folds,
        candidate_complete=True,
    )

    assert report["replacement_selected"] is False
    assert report["fallback_preserved_incumbent_exactly"] is True
    assert selected.tobytes() == incumbent.tobytes()
    assert report["candidate_oof_log_loss"] > report["incumbent_oof_log_loss"]


def test_dog_breed_stability_optimizer_separates_head_norm_and_decay():
    class Parameter:
        def __init__(self, ndim):
            self.ndim = ndim
            self.requires_grad = True

    class Head:
        def __init__(self):
            self.weight = Parameter(2)
            self.bias = Parameter(1)

        def parameters(self, recurse=False):
            assert recurse is False
            return [self.weight, self.bias]

    class Model:
        def __init__(self):
            self.backbone_weight = Parameter(4)
            self.backbone_bias = Parameter(1)
            self.norm_weight = Parameter(1)
            self.norm_bias = Parameter(1)
            self.head = Head()
            self.classifier = [object(), object(), self.head]

        def named_parameters(self):
            return iter([
                ("features.0.weight", self.backbone_weight),
                ("features.0.bias", self.backbone_bias),
                ("classifier.0.weight", self.norm_weight),
                ("classifier.0.bias", self.norm_bias),
                ("classifier.2.weight", self.head.weight),
                ("classifier.2.bias", self.head.bias),
            ])

    groups, contract = wave2.build_dog_breed_stability_optimizer_groups(
        Model(),
        base_learning_rate=2e-4,
    )

    assert len(groups) == 4
    assert contract["groups_disjoint"] is True
    assert contract["groups_cover_all_trainable_parameters"] is True
    assert contract["task_head_parameter_names"] == [
        "classifier.2.weight",
        "classifier.2.bias",
    ]
    assert contract["classifier_norm_parameter_names"] == [
        "classifier.0.weight",
        "classifier.0.bias",
    ]
    by_name = {item["group_name"]: item for item in contract["groups"]}
    assert by_name["pretrained_no_decay"]["learning_rate"] == pytest.approx(2e-5)
    assert by_name["pretrained_no_decay"]["weight_decay"] == 0.0
    assert by_name["task_head_decay"]["learning_rate"] == pytest.approx(2e-4)
    assert by_name["task_head_decay"]["weight_decay"] == pytest.approx(2e-4)
    assert by_name["task_head_no_decay"]["weight_decay"] == 0.0


def test_dog_breed_frozen_head_optimizer_trains_only_mapped_classifier():
    class Parameter:
        def __init__(self, ndim):
            self.ndim = ndim
            self.requires_grad = True

    class Head:
        def __init__(self):
            self.weight = Parameter(2)
            self.bias = Parameter(1)

        def parameters(self, recurse=True):
            del recurse
            return iter((self.weight, self.bias))

    class Classifier:
        def __init__(self):
            self._modules = {"2": Head()}

        def __getitem__(self, index):
            return self._modules[str(index)]

    class Model:
        def __init__(self):
            self.classifier = Classifier()
            self.backbone_weight = Parameter(2)

        def named_parameters(self):
            head = self.classifier._modules["2"]
            return iter(
                [
                    ("features.0.weight", self.backbone_weight),
                    ("classifier.2.weight", head.weight),
                    ("classifier.2.bias", head.bias),
                ]
            )

    model = Model()
    groups, contract = wave2.build_dog_breed_frozen_head_optimizer_groups(
        model,
        base_learning_rate=1e-3,
    )

    assert len(groups) == 2
    assert model.backbone_weight.requires_grad is False
    assert model.classifier._modules["2"].weight.requires_grad is True
    assert model.classifier._modules["2"].bias.requires_grad is True
    assert contract["mode"] == "frozen_backbone_imagenet_head_only"
    assert contract["backbone_frozen"] is True
    assert contract["groups_disjoint"] is True
    assert contract["groups_cover_all_trainable_parameters"] is True
    by_name = {item["group_name"]: item for item in contract["groups"]}
    assert by_name["mapped_head_decay"]["learning_rate"] == pytest.approx(1e-3)
    assert by_name["mapped_head_decay"]["weight_decay"] == pytest.approx(1e-4)
    assert by_name["mapped_head_no_decay"]["weight_decay"] == 0.0


def test_dog_breed_stability_schedule_warms_then_decays_to_five_percent():
    total_steps = 100
    factors = [
        wave2.dog_breed_stability_lr_factor(step, total_steps=total_steps)
        for step in range(total_steps)
    ]
    assert factors[0] == pytest.approx(0.1)
    assert factors[9] == pytest.approx(1.0)
    assert factors[-1] == pytest.approx(0.05)
    assert all(left <= right for left, right in zip(factors[:9], factors[1:10]))
    assert all(left >= right for left, right in zip(factors[9:-1], factors[10:]))


def test_dog_breed_stab1_has_single_fold_no_submission_contract():
    source = inspect.getsource(wave2._run_vision)
    assert "splits = splits[:diagnostic_fold_limit]" in source
    assert '"fixed_fold": 0' in source
    assert '"valid_submission": False' in source
    assert '"submission_created": False' in source
    assert '"official_grader_executed": False' in source
    assert "DOG_BREED_STAB_EPOCH_ONE_MAX_LOG_LOSS" in source
    assert "DOG_BREED_STAB_MIN_EPOCH_IMPROVEMENT" in source


def test_run_dog_breed_binds_shared_imagenet_teacher_cache():
    source = inspect.getsource(wave2.run_dog_breed)
    assert "convnext_small_imagenet1k_teacher_v1.npz" in source
    assert "dog_breed_teacher_cache_path=wave0.ensure_within" in source
    assert "dog_breed_teacher_source_manifest_sha256" in source


def test_leaf_multimodal_d4_codes_and_explicit_id_join(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for identifier in (1, 2):
        (image_dir / f"{identifier}.jpg").write_bytes(b"fixture")

    paths = recovery.resolve_leaf_image_paths(tmp_path, [1, 2.0])
    assert [path.name for path in paths] == ["1.jpg", "2.jpg"]
    assert recovery.leaf_d4_variant_codes(1) == ((0, False),)
    assert recovery.leaf_d4_variant_codes(4) == tuple(
        (rotation, False) for rotation in range(4)
    )
    codes = recovery.leaf_d4_variant_codes(8)
    assert len(codes) == 8 and len(set(codes)) == 8
    with pytest.raises(ValueError, match="one of 1, 4, or 8"):
        recovery.leaf_d4_variant_codes(2)
    with pytest.raises(FileNotFoundError, match="join is incomplete"):
        recovery.resolve_leaf_image_paths(tmp_path, [3])


def test_leaf_multimodal_runner_has_gpu_throughput_and_fold_clean_components():
    import inspect

    embedding_source = inspect.getsource(recovery.extract_leaf_convnext_embeddings)
    runner_source = inspect.getsource(recovery.run_leaf_multimodal)
    assert "memory_format=torch.channels_last" in embedding_source
    assert "torch.autocast" in embedding_source
    assert "torch.backends.cudnn.benchmark = True" in embedding_source
    assert "VISION_DATALOADER_PREFETCH_FACTOR" in embedding_source
    assert "numeric_rbf_svc" in runner_source
    assert "image_cosine_rbf_svc" in runner_source
    assert "multimodal_group_balanced_rbf_svc" in runner_source
    assert "cross_fit_multiclass_logloss_blend" in runner_source
    assert '"private_labels_used": False' in runner_source


def test_vision_runner_aligns_logloss_tta_and_preserves_fast_other_task_validation():
    import inspect

    source = inspect.getsource(wave2._run_vision)
    assert 'model.to(device="cuda", memory_format=torch.channels_last)' in source
    assert ".cuda(memory_format=" not in source
    assert "fused=True" in source
    assert "prefetch_factor" in source
    assert 'with torch.autocast(device_type="cuda", dtype=amp_dtype):' in source
    assert "logits = logits.float()" in source
    assert "checkpoint_selection_tta = bool(" in source
    assert "apply_tta=checkpoint_selection_tta" in source
    assert "fold_valid, truth, fold_valid_logits = infer" in source
    assert "valid_loader, apply_tta=tta_flips" in source
    assert "fold_test, _, fold_test_logits = infer" in source
    assert "test_loader, apply_tta=tta_flips" in source
    assert "binary_test_logits_by_fold[:, fold] = fold_test_logits[:, 0]" in source
    assert '"per_fold_calibrate_then_probability_average"' in inspect.getsource(
        wave2.cross_fit_binary_logloss_calibration
    )
    assert "torch.backends.cudnn.benchmark = fast_kernel_mode" in source
    assert '"a40_fast_kernel_mode": fast_kernel_mode' in source
    assert '"epoch_selection_tta": bool(' in source
    assert '"train_images_per_second"' in source
    assert "torch.save(best_state, best_path)" in source


def test_histopath_transform_keeps_labelled_centre_geometry():
    source = inspect.getsource(wave2._image_transforms)
    assert "if center_patch:" in source
    assert "transforms.CenterCrop(64)" in source
    assert "RandomResizedCrop" in source
    centre_branch = source.split("if center_patch:", 1)[1].split("else:", 1)[0]
    assert "transforms.RandomResizedCrop(" not in centre_branch


def test_retina_preprocessing_crops_black_border_and_returns_square():
    from PIL import Image

    values = np.zeros((12, 20, 3), dtype=np.uint8)
    values[3:9, 5:15] = np.asarray([80, 120, 160], dtype=np.uint8)
    result = wave2.prepare_retina_image(Image.fromarray(values, mode="RGB"))
    assert result.size == (10, 10)
    result_values = np.asarray(result)
    assert result_values.shape == (10, 10, 3)
    assert int(result_values.max()) > 0


def test_full_frame_padding_preserves_all_source_pixels():
    from PIL import Image

    values = np.zeros((2, 4, 3), dtype=np.uint8)
    values[0, 0] = [255, 0, 0]
    values[0, -1] = [0, 255, 0]
    values[-1, 0] = [0, 0, 255]
    values[-1, -1] = [255, 255, 0]
    result = np.asarray(wave2.pad_to_square_image(Image.fromarray(values, mode="RGB")))
    assert result.shape == (4, 4, 3)
    np.testing.assert_array_equal(result[1:3], values)


def test_reviewed_vision_profiles_and_tta_evidence_are_bound_to_runners():
    aptos = inspect.getsource(wave2.run_aptos)
    plant = inspect.getsource(wave2.run_plant)
    ranzcr = inspect.getsource(wave2.run_ranzcr)
    vision = inspect.getsource(wave2._run_vision)
    assert 'preprocessing_profile="retina"' in aptos
    assert "early_stopping_min_epochs=4" in aptos
    assert 'preprocessing_profile="full_frame"' in plant
    assert "fold_count=min(3, args.wave2_vision_folds)" in plant
    assert "RANZCR requires a complete PatientID" in ranzcr
    assert '"epoch_selection_tta": bool(tta_flips)' in vision
    assert "epoch + 1 >= minimum_epochs_before_stop" in vision


def test_multiclass_image_duplicate_contract_groups_and_overrides(tmp_path):
    train_paths = []
    for name, content in (("a.jpg", b"breed-a"), ("b.jpg", b"breed-b"), ("c.jpg", b"breed-a")):
        path = tmp_path / name
        path.write_bytes(content)
        train_paths.append(path)
    test_paths = [tmp_path / "x.jpg", tmp_path / "y.jpg"]
    test_paths[0].write_bytes(b"breed-b")
    test_paths[1].write_bytes(b"unseen")
    cache = tmp_path / "cache" / "dog_hashes.npz"
    groups, overrides, contract = wave2.build_multiclass_image_duplicate_contract(
        train_paths,
        test_paths,
        np.asarray(["a", "b", "a"]),
        ["a", "b"],
        workers=2,
        cache_path=cache,
    )
    assert groups[0] == groups[2]
    np.testing.assert_allclose(overrides[0].sum(), 1.0)
    assert int(np.argmax(overrides[0])) == 1
    assert np.isnan(overrides[1]).all()
    assert contract["duplicate_train_rows"] == 1
    assert contract["exact_train_test_matches"] == 1
    assert contract["cache_hit"] is False

    _, repeated, repeated_contract = wave2.build_multiclass_image_duplicate_contract(
        train_paths,
        test_paths,
        np.asarray(["a", "b", "a"]),
        ["a", "b"],
        workers=2,
        cache_path=cache,
    )
    np.testing.assert_equal(repeated, overrides)
    assert repeated_contract["cache_hit"] is True


def test_multiclass_image_duplicate_contract_quarantines_conflicting_labels(tmp_path):
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    matching_test = tmp_path / "test.jpg"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    matching_test.write_bytes(b"same")
    groups, overrides, contract = wave2.build_multiclass_image_duplicate_contract(
        [first, second],
        [matching_test],
        np.asarray(["a", "b"]),
        ["a", "b"],
        workers=1,
    )
    assert groups[0] == groups[1]
    assert np.isnan(overrides[0]).all()
    assert contract["conflicting_duplicate_hash_groups"] == 1
    assert contract["conflicting_duplicate_train_rows"] == 2
    assert contract["conflicting_test_hash_matches"] == 1
    assert contract["conflicting_test_overrides"] == 0
    assert contract["conflicting_hash_policy"] == "group_together_no_test_override"


def test_histopath_duplicate_contract_groups_hashes_and_reuses_cache(tmp_path):
    train_paths = []
    for name, content in (("a.tif", b"negative"), ("b.tif", b"positive"), ("c.tif", b"negative")):
        path = tmp_path / name
        path.write_bytes(content)
        train_paths.append(path)
    test_paths = [tmp_path / "x.tif", tmp_path / "y.tif"]
    test_paths[0].write_bytes(b"positive")
    test_paths[1].write_bytes(b"unseen")
    cache = tmp_path / "cache" / "histopath_hashes.npz"

    hashes, overrides, contract = wave2.build_histopath_duplicate_contract(
        train_paths,
        test_paths,
        np.asarray([0, 1, 0]),
        workers=2,
        cache_path=cache,
    )
    assert hashes[0] == hashes[2]
    assert overrides[0] == 1.0
    assert np.isnan(overrides[1])
    assert contract["duplicate_train_rows"] == 1
    assert contract["exact_train_test_matches"] == 1
    assert contract["cache_hit"] is False

    _, repeated_overrides, repeated_contract = wave2.build_histopath_duplicate_contract(
        train_paths,
        test_paths,
        np.asarray([0, 1, 0]),
        workers=2,
        cache_path=cache,
    )
    np.testing.assert_equal(repeated_overrides, overrides)
    assert repeated_contract["cache_hit"] is True


def test_large_vision_manifest_metadata_is_parallel_bounded_and_ordered(tmp_path, monkeypatch):
    paths = []
    for name, content in (("c.tif", b"ccc"), ("a.tif", b"a"), ("b.tif", b"bb")):
        path = tmp_path / name
        path.write_bytes(content)
        paths.append(path)
    observed = {}

    class RecordingPool:
        def __init__(self, *, max_workers):
            observed["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, function, values, *, chunksize):
            observed["chunksize"] = chunksize
            return map(function, values)

    monkeypatch.setattr(wave2, "ThreadPoolExecutor", RecordingPool)
    manifest = wave2._parallel_file_manifest(paths, workers=32)

    assert observed == {"max_workers": 3, "chunksize": 64}
    assert [value[0] for value in manifest] == [str(path.resolve()) for path in paths]
    assert [value[1] for value in manifest] == [3, 1, 2]
    assert "_parallel_file_manifest(all_paths, workers)" in inspect.getsource(
        wave2.build_histopath_duplicate_contract
    )
    assert "_parallel_file_manifest(all_paths, workers)" in inspect.getsource(
        wave2.build_multiclass_image_duplicate_contract
    )


def test_histopath_duplicate_contract_rejects_conflicting_labels(tmp_path):
    first = tmp_path / "a.tif"
    second = tmp_path / "b.tif"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    with pytest.raises(RuntimeError, match="conflicting labels"):
        wave2.build_histopath_duplicate_contract(
            [first, second],
            [],
            np.asarray([0, 1]),
            workers=1,
        )


def test_histopath_promotion_checks_require_grouped_disjoint_zero_conflict():
    groups = np.asarray(["a", "a", "b", "c"])
    passed = wave2.histopath_duplicate_promotion_checks(
        groups=groups,
        split_strategy="stratified_group_kfold",
        group_overlap_by_fold={"0": 0, "1": 0},
        duplicate_contract={"conflicting_duplicate_hash_groups": 0},
    )
    assert all(passed.values())
    failed = wave2.histopath_duplicate_promotion_checks(
        groups=groups,
        split_strategy="stratified_kfold",
        group_overlap_by_fold={"0": 1},
        duplicate_contract={"conflicting_duplicate_hash_groups": 1},
    )
    assert not any(failed.values())


def test_bird_supplemental_features_align_ids_and_aggregate_segments(tmp_path):
    (tmp_path / "histogram_of_segments.txt").write_text(
        "rec_id,h1,h2\n2,20,21\n1,10,11\n",
        encoding="utf-8",
    )
    (tmp_path / "segment_features.txt").write_text(
        "rec_id,segment_id,f1,f2\n1,0,1,2\n1,1,3,4\n2,0,5,6\n",
        encoding="utf-8",
    )

    features, contract = wave2.load_bird_supplemental_features(tmp_path, [1, 2])

    assert features.shape == (2, 17)
    np.testing.assert_allclose(features[:, :2], [[10, 11], [20, 21]])
    np.testing.assert_allclose(features[0, 2:5], [2.0, np.log1p(2), 0.0])
    assert {key: contract[key] for key in (
        "recordings",
        "histogram_width",
        "segment_source_width",
        "segment_aggregate_width",
        "combined_width",
        "missing_segment_recordings",
        "private_labels_used",
    )} == {
        "recordings": 2,
        "histogram_width": 2,
        "segment_source_width": 2,
        "segment_aggregate_width": 15,
        "combined_width": 17,
        "missing_segment_recordings": 0,
        "private_labels_used": False,
    }
    assert len(contract["fingerprint"]) == 64
    assert set(contract["source_files"]) == {
        "histogram_of_segments.txt",
        "segment_features.txt",
    }


def test_bird_supplemental_features_fail_closed_on_missing_or_duplicate_ids(tmp_path):
    segment = "rec_id,segment_id,f1\n1,0,1\n2,0,2\n"
    (tmp_path / "segment_features.txt").write_text(segment, encoding="utf-8")
    (tmp_path / "histogram_of_segments.txt").write_text(
        "rec_id,h1\n1,10\n1,11\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="duplicate recording IDs"):
        wave2.load_bird_supplemental_features(tmp_path, [1])

    (tmp_path / "histogram_of_segments.txt").write_text(
        "rec_id,h1\n1,10\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="missing 1 recording IDs"):
        wave2.load_bird_supplemental_features(tmp_path, [1, 2])

    (tmp_path / "histogram_of_segments.txt").write_text(
        "rec_id,h1\n1,10\n2,20\n",
        encoding="utf-8",
    )
    (tmp_path / "segment_features.txt").write_text(
        "rec_id,segment_id,f1\n1,0,1\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="segment alignment is missing 1"):
        wave2.load_bird_supplemental_features(tmp_path, [1, 2])


def test_bird_supplemental_features_accept_zero_histogram_without_segments(tmp_path):
    (tmp_path / "histogram_of_segments.txt").write_text(
        "rec_id,h1,h2\n1,0.4,0.6\n2,0,0\n",
        encoding="utf-8",
    )
    (tmp_path / "segment_features.txt").write_text(
        "rec_id,segment_id,f1,f2\n1,0,1,2\n",
        encoding="utf-8",
    )

    features, contract = wave2.load_bird_supplemental_features(tmp_path, [1, 2])

    assert features.shape == (2, 17)
    np.testing.assert_allclose(features[1, :5], [0.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(features[1, 5:], 0.0)
    assert np.isfinite(features).all()
    assert contract["missing_segment_recordings"] == 1
    assert contract["missing_segment_nonzero_histograms"] == 0
    assert contract["missing_segment_policy"] == "zero_fill_only_when_histogram_all_zero"


def test_binary_auc_crossfit_blend_is_complete_and_fold_clean():
    truth = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    folds = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    strong = np.asarray([0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.25, 0.75])
    weak = np.asarray([0.45, 0.55, 0.4, 0.6, 0.48, 0.52, 0.42, 0.58])
    oof, test, fold_weights, final_weight = wave2.cross_fit_binary_auc_blend(
        strong,
        weak,
        np.asarray([0.2, 0.8]),
        np.asarray([0.4, 0.6]),
        truth,
        folds,
    )
    assert np.isfinite(oof).all() and np.isfinite(test).all()
    assert len(fold_weights) == 2
    assert all(0.0 <= value <= 1.0 for value in [*fold_weights, final_weight])
    assert wave2.compute_metric("roc_auc", truth, oof) == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="multiple folds"):
        wave2.cross_fit_binary_auc_blend(strong, weak, [0.2], [0.4], truth, -folds - 1)


def test_multichannel_logit_auc_blend_is_complete_and_on_simplex():
    truth = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    folds = np.repeat(np.arange(3), 4)
    strong = np.where(truth == 1, 0.9, 0.1)
    medium = np.where(truth == 1, 0.7, 0.3)
    weak = np.asarray([0.45, 0.55] * 6)
    oof, test, fold_weights, final_weights = wave2.cross_fit_multichannel_logit_auc_blend(
        np.column_stack([strong, medium, weak]),
        np.asarray([[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]]),
        truth,
        folds,
    )
    assert np.isfinite(oof).all() and np.isfinite(test).all()
    assert len(fold_weights) == 3
    for weights in [*fold_weights, final_weights]:
        assert len(weights) == 3
        assert sum(weights) == pytest.approx(1.0)
        assert all(0.0 <= value <= 1.0 for value in weights)
    assert wave2.compute_metric("roc_auc", truth, oof) == pytest.approx(1.0)


def test_birds_linear_species_channel_is_fold_isolated_and_finite(tmp_path):
    rng = np.random.default_rng(42)
    train = rng.normal(size=(20, 12)).astype(np.float32)
    test = rng.normal(size=(4, 12)).astype(np.float32)
    rows = np.arange(len(train))
    truth = np.asarray(
        [[(row // 2 + species) % 2 for species in range(19)] for row in rows],
        dtype=np.int8,
    )
    fold_assignment = rows % 2
    splits = [
        (np.flatnonzero(fold_assignment != fold), np.flatnonzero(fold_assignment == fold))
        for fold in range(2)
    ]

    oof, test_prediction, contract = wave2.fit_birds_linear_species_channel(
        train,
        test,
        truth,
        splits,
        seed=42,
        task_dir=tmp_path,
    )

    assert oof.shape == truth.shape
    assert test_prediction.shape == (len(test), 19)
    assert np.isfinite(oof).all() and np.isfinite(test_prediction).all()
    assert np.all((oof >= 0.0) & (oof <= 1.0))
    assert contract["regularization_c"] == pytest.approx(0.01)
    assert contract["private_labels_used"] is False
    assert len(contract["fold_artifacts"]) == 2
    assert all((tmp_path / name).is_file() for name in contract["fold_artifacts"])


def test_audio_feature_cache_reuses_matrix_across_runs(monkeypatch, tmp_path):
    paths = [tmp_path / "a.wav", tmp_path / "b.wav"]
    for index, path in enumerate(paths):
        path.write_bytes(bytes([index + 1]))
    calls = []

    def fake_extract(values, workers):
        calls.append((list(values), workers))
        return np.full((len(values), wave2.AUDIO_FEATURE_WIDTH), 3.0, dtype=np.float32)

    monkeypatch.setattr(wave2, "_parallel_audio_features", fake_extract)
    first, first_contract = wave2.load_or_compute_audio_feature_matrix(paths, 16, tmp_path / "cache")
    second, second_contract = wave2.load_or_compute_audio_feature_matrix(paths, 16, tmp_path / "cache")

    assert len(calls) == 1
    assert first_contract["cache_hit"] is False
    assert second_contract["cache_hit"] is True
    assert first_contract["path_manifest"]["cache_hit"] is False
    assert second_contract["path_manifest"]["cache_hit"] is True
    assert len(first_contract["path_manifest"]["waveform_sha256"]) == 64
    np.testing.assert_array_equal(first, second)


def test_shared_zip_cache_extracts_once(tmp_path):
    archive = tmp_path / "audio.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("clips/1.aif", b"audio")
    first, first_contract = wave2.extract_zip_to_shared_cache(archive, tmp_path / "cache")
    second, second_contract = wave2.extract_zip_to_shared_cache(archive, tmp_path / "cache")
    assert (first / "clips" / "1.aif").read_bytes() == b"audio"
    assert first == second
    assert first_contract["cache_hit"] is False
    assert second_contract["cache_hit"] is True


def test_whale_duplicate_aware_splits_group_hashes_and_reject_conflicts():
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    hashes = np.asarray(["a", "a", "b", "c", "c", "d"], dtype=object)
    splits, strategy, contract = wave2.make_whale_duplicate_aware_splits(
        labels,
        hashes,
        requested_folds=2,
        seed=42,
    )
    assert strategy == "stratified_group_kfold_exact_waveform_sha256"
    assert contract["duplicate_rows"] == 2
    assert len(splits) == 2
    for train_indices, valid_indices in splits:
        assert set(hashes[train_indices]).isdisjoint(set(hashes[valid_indices]))
        assert set(labels[valid_indices]) == {0, 1}

    with pytest.raises(RuntimeError, match="conflicting labels"):
        wave2.make_whale_duplicate_aware_splits(
            np.asarray([0, 1, 0, 1]),
            np.asarray(["same", "same", "zero", "one"], dtype=object),
            requested_folds=2,
            seed=42,
        )


def test_birds_and_whale_runners_encode_official_oof_and_cache_contracts():
    birds_source = inspect.getsource(wave2.run_birds)
    precompute_source = inspect.getsource(wave2.precompute_birds_cpu_artifacts)
    whale_source = inspect.getsource(wave2.run_whale)
    assert "pooled_recording_species_binary_roc_auc" in birds_source
    assert "cross_fit_multichannel_logit_auc_blend" in birds_source
    assert "fit_birds_linear_species_channel" in birds_source
    assert "birds_oof_and_test.npz" in birds_source
    assert '"private_labels_used": False' in birds_source
    assert 'eval_metric="Logloss"' in inspect.getsource(wave2.load_or_compute_birds_species_precompute)
    assert "thread_count=species_thread_count" in inspect.getsource(
        wave2.load_or_compute_birds_species_precompute
    )
    assert "load_or_compute_audio_feature_matrix" in inspect.getsource(wave2.build_birds_feature_matrices)
    assert "load_or_compute_birds_species_precompute" in birds_source
    assert "gpu_training_started" in precompute_source
    assert "cuda_required" in precompute_source
    assert "audio_waveform_sha256" in inspect.getsource(wave2._birds_precompute_identity)
    assert "make_whale_duplicate_aware_splits" in whale_source
    assert "whale_oof_and_test.npz" in whale_source
    assert "exact_duplicate_test_override" in whale_source
    assert "fold_test_ensemble" in whale_source


def test_strict_dogs_cats_filename_contract():
    labels = recovery.strict_dogs_cats_labels([
        recovery.Path("cat.1.jpg"),
        recovery.Path("dog.1.jpg"),
    ])
    np.testing.assert_array_equal(labels, np.asarray([0, 1]))
    with pytest.raises(ValueError, match="Unexpected Dogs-vs-Cats"):
        recovery.strict_dogs_cats_labels([
            recovery.Path("cat.1.jpg"),
            recovery.Path("invalid.jpg"),
        ])


def test_dogs_image_decode_manifest_verifies_rgb_and_hashes(tmp_path):
    from PIL import Image

    cat = tmp_path / "cat.1.jpg"
    dog = tmp_path / "dog.1.jpg"
    Image.new("L", (8, 6), color=128).save(cat)
    Image.new("RGB", (7, 5), color=(1, 2, 3)).save(dog)
    manifest = recovery.verify_image_decode_manifest(
        [cat, dog],
        split="train",
        labels=np.asarray([0, 1]),
    )
    assert manifest["rgb_decode"].all()
    assert manifest["label"].tolist() == [0, 1]
    assert manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert manifest[["width", "height"]].to_numpy().tolist() == [[8, 6], [7, 5]]

    corrupt = tmp_path / "cat.2.jpg"
    corrupt.write_bytes(b"not-an-image")
    with pytest.raises(RuntimeError, match="Image decode failed"):
        recovery.verify_image_decode_manifest([corrupt], split="train")

    duplicate = tmp_path / "dog.2.jpg"
    duplicate.write_bytes(cat.read_bytes())
    duplicate_manifest = recovery.verify_image_decode_manifest(
        [cat, duplicate],
        split="train",
        labels=np.asarray([0, 1]),
    )
    assert duplicate_manifest["sha256"].nunique() == 1

    source = inspect.getsource(recovery.verify_image_decode_manifest)
    assert "ThreadPoolExecutor" in source
    assert "workers or 64" in source
    decode_source = inspect.getsource(recovery._decode_manifest_record)
    assert "path.read_bytes()" in decode_source
    assert "io.BytesIO(encoded)" in decode_source
    runner_source = inspect.getsource(recovery.run_dogs_cats_convnext)
    assert "workers=args.dogs_workers" in runner_source


def test_exact_image_hash_groups_and_label_conflict(tmp_path):
    first = tmp_path / "first.jpg"
    duplicate = tmp_path / "duplicate.jpg"
    unique = tmp_path / "unique.jpg"
    first.write_bytes(b"same-image")
    duplicate.write_bytes(b"same-image")
    unique.write_bytes(b"different-image")

    groups, manifest = recovery.build_exact_image_hash_groups(
        [first, duplicate, unique],
        np.asarray([1, 1, 0]),
    )
    assert groups is not None
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]
    assert manifest["duplicate_group"].tolist() == [True, True, False]

    with pytest.raises(RuntimeError, match="conflicting labels"):
        recovery.build_exact_image_hash_groups(
            [first, duplicate],
            np.asarray([0, 1]),
        )


def test_multiclass_submission_alignment_is_id_keyed():
    sample = pd.DataFrame({
        "id": ["row-b", "row-a"],
        "EAP": [0.0, 0.0],
        "HPL": [0.0, 0.0],
        "MWS": [0.0, 0.0],
    })
    test_ids = pd.Series(["row-a", "row-b"])
    probability = np.asarray([
        [0.1, 0.2, 0.7],
        [0.6, 0.3, 0.1],
    ])
    aligned = recovery.align_multiclass_submission(
        sample,
        test_ids,
        probability,
        ["EAP", "HPL", "MWS"],
    )
    probability_columns = ["EAP", "HPL", "MWS"]
    np.testing.assert_allclose(
        aligned.loc[[0], probability_columns].to_numpy(dtype=np.float64)[0],
        [0.6, 0.3, 0.1],
    )
    np.testing.assert_allclose(
        aligned.loc[[1], probability_columns].to_numpy(dtype=np.float64)[0],
        [0.1, 0.2, 0.7],
    )
    with pytest.raises(RuntimeError, match="ID sets differ"):
        recovery.align_multiclass_submission(
            sample,
            pd.Series(["row-a", "row-c"]),
            probability,
            ["EAP", "HPL", "MWS"],
        )


def test_spooky_probability_blend_and_crossfit_are_complete():
    word = np.asarray([
        [0.80, 0.10, 0.10],
        [0.70, 0.20, 0.10],
        [0.10, 0.80, 0.10],
        [0.10, 0.70, 0.20],
        [0.10, 0.10, 0.80],
        [0.20, 0.10, 0.70],
    ])
    char = 0.9 * word + 0.1 / 3.0
    labels = np.asarray(["EAP", "EAP", "HPL", "HPL", "MWS", "MWS"])
    classes = ["EAP", "HPL", "MWS"]
    blended = recovery.apply_spooky_probability_blend(
        word,
        char,
        word_weight=0.5,
        temperature=1.0,
    )
    assert np.isfinite(blended).all()
    np.testing.assert_allclose(blended.sum(axis=1), 1.0)

    crossfit, records = recovery.cross_fit_spooky_probability_blend(
        word,
        char,
        labels,
        classes,
        np.asarray([0, 1, 0, 1, 0, 1]),
    )
    assert len(records) == 2
    assert np.isfinite(crossfit).all()
    np.testing.assert_allclose(crossfit.sum(axis=1), 1.0)


def test_spooky_stylometry_and_multicomponent_blend_are_complete():
    texts = pd.Series([
        "I am here, and I am ready.",
        'SHE CRIED: "Never!"',
        "Longer words nevertheless characterize this peculiar sentence.",
        "He was there; he was not alone.",
        "What? What! What...",
        "We have been in the house before.",
    ])
    features = recovery.build_spooky_stylometric_features(texts)
    assert len(features) == len(texts)
    assert "character_entropy" in features
    assert "function_the" in features
    assert "punct_exclamation_per_word" in features
    assert np.isfinite(features.to_numpy()).all()
    assert features.nunique().max() > 1

    classes = ["EAP", "HPL", "MWS"]
    labels = np.asarray(["EAP", "HPL", "MWS", "EAP", "HPL", "MWS"])
    base = np.asarray([
        [0.80, 0.10, 0.10],
        [0.10, 0.80, 0.10],
        [0.10, 0.10, 0.80],
        [0.75, 0.15, 0.10],
        [0.15, 0.75, 0.10],
        [0.10, 0.15, 0.75],
    ])
    components = [
        base,
        0.9 * base + 0.1 / 3.0,
        0.8 * base + 0.2 / 3.0,
        0.7 * base + 0.3 / 3.0,
    ]
    for mode in ("arithmetic", "log_probability"):
        direct = recovery.apply_spooky_multicomponent_blend(
            components,
            weights=[0.25, 0.25, 0.25, 0.25],
            temperature=1.0,
            mode=mode,
        )
        assert np.isfinite(direct).all()
        np.testing.assert_allclose(direct.sum(axis=1), 1.0)
    blended, records = recovery.cross_fit_spooky_multicomponent_blend(
        components,
        labels,
        classes,
        np.asarray([0, 0, 0, 1, 1, 1]),
    )
    assert len(records) == 2
    assert np.isfinite(blended).all()
    np.testing.assert_allclose(blended.sum(axis=1), 1.0)
    for record in records:
        assert sum(record["weights"]) == pytest.approx(1.0)
        assert record["mode"] in {"arithmetic", "log_probability"}


def test_spooky_duplicate_groups_are_stable_and_conflicts_fail_closed():
    texts = pd.Series([
        "The SAME text!",
        "  the same text  ",
        "Unique EAP",
        "Unique HPL",
        "Unique MWS",
    ])
    labels = np.asarray(["EAP", "EAP", "EAP", "HPL", "MWS"])
    groups, report = recovery.build_spooky_duplicate_groups(texts, labels)
    assert groups[0] == groups[1]
    assert report["duplicate_groups"] == 1
    assert report["duplicate_rows"] == 1

    conflicting = labels.copy()
    conflicting[1] = "HPL"
    with pytest.raises(RuntimeError, match="conflicting labels"):
        recovery.build_spooky_duplicate_groups(texts, conflicting)


def test_spooky_duplicate_groups_never_cross_grouped_folds():
    from sklearn.model_selection import StratifiedGroupKFold

    labels = np.asarray(["EAP", "EAP", "EAP", "EAP", "HPL", "HPL", "MWS", "MWS"])
    groups = np.asarray([0, 0, 1, 2, 3, 4, 5, 6])
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    for fit_indices, valid_indices in splitter.split(np.zeros(len(labels)), labels, groups):
        assert not (set(groups[fit_indices]) & set(groups[valid_indices]))


def test_spooky_nbsvm_channel_shapes():
    matrix = csr_matrix(np.asarray([
        [2, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 2, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 2, 1],
        [0, 0, 1, 1],
    ], dtype=np.float64))
    labels = np.asarray(["EAP", "EAP", "HPL", "HPL", "MWS", "MWS"])
    valid, test, models = recovery.fit_spooky_nbsvm_channel(
        matrix,
        labels,
        matrix[:2],
        matrix[2:5],
        ["EAP", "HPL", "MWS"],
        c_value=1.0,
        seed=42,
    )
    assert valid.shape == (2, 3)
    assert test.shape == (3, 3)
    assert len(models) == 3
    np.testing.assert_allclose(valid.sum(axis=1), 1.0)
    np.testing.assert_allclose(test.sum(axis=1), 1.0)


def test_may2022_crossfit_blend_and_scalar_id_alignment():
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    folds = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    strong = np.asarray([0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.25, 0.75])
    components = {
        "residual_mlp": strong,
        "xgboost": 0.9 * strong + 0.05,
        "catboost": np.asarray([0.4, 0.6, 0.35, 0.65, 0.45, 0.55, 0.3, 0.7]),
    }
    prediction, records = recovery.cross_fit_may2022_multimodel_blend(
        components, labels, folds
    )
    assert len(records) == 2
    assert np.isfinite(prediction).all()
    assert all(record["outer_auc"] == pytest.approx(1.0) for record in records)

    sample = pd.DataFrame({"id": [2, 1], "target": [0.0, 0.0]})
    aligned = recovery.align_scalar_submission_by_id(
        sample,
        pd.Series([1, 2]),
        np.asarray([0.1, 0.9]),
        id_column="id",
        target_column="target",
    )
    assert aligned["id"].tolist() == [2, 1]
    assert aligned["target"].tolist() == [0.9, 0.1]
    with pytest.raises(RuntimeError, match="ID sets differ"):
        recovery.align_scalar_submission_by_id(
            sample,
            pd.Series([1, 3]),
            np.asarray([0.1, 0.9]),
            id_column="id",
            target_column="target",
        )


def test_denoising_submission_coordinates_require_exact_pixel_coverage(tmp_path):
    from PIL import Image

    image = np.asarray([[0, 64], [128, 255]], dtype=np.uint8)
    Image.fromarray(image, mode="L").save(tmp_path / "doc_1.png")
    sample = pd.DataFrame({
        "id": ["doc_1_1_1", "doc_1_1_2", "doc_1_2_1", "doc_1_2_2"],
        "value": [0.0, 0.0, 0.0, 0.0],
    })
    coordinates = recovery.validate_denoising_submission_coordinates(sample, tmp_path)
    assert coordinates["sample_position"].tolist() == [0, 1, 2, 3]
    assert set(coordinates["image_id"]) == {"doc_1"}

    with pytest.raises(RuntimeError, match="out of bounds"):
        recovery.validate_denoising_submission_coordinates(
            pd.DataFrame({"id": ["doc_1_1_1", "doc_1_1_2", "doc_1_2_1", "doc_1_3_2"]}),
            tmp_path,
        )
    with pytest.raises(RuntimeError, match="every pixel exactly once"):
        recovery.validate_denoising_submission_coordinates(sample.iloc[:3], tmp_path)
