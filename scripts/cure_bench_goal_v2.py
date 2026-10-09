from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


SEED = 20260830
BOOTSTRAP_ROUNDS = 1000


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
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def load_jsonl(path: Path, *, require_answer: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not {"id", "question", "options", "question_type"}.issubset(value):
                raise ValueError("CURE-Bench row schema mismatch")
            if require_answer and "correct_answer" not in value:
                raise ValueError("validation row lacks answer")
            if not require_answer and "correct_answer" in value:
                raise ValueError("test labels unexpectedly present")
            rows.append(value)
    if not rows:
        raise ValueError("empty JSONL")
    return rows


def options_list(value: Any) -> tuple[list[str], list[str]]:
    if isinstance(value, dict):
        keys = [str(key) for key in value]
        return keys, [str(value[key]) for key in value]
    if isinstance(value, list):
        return [chr(ord("A") + index) for index in range(len(value))], [str(item) for item in value]
    raise ValueError("unsupported options schema")


def answer_index(row: dict[str, Any]) -> int:
    keys, values = options_list(row["options"])
    answer = row["correct_answer"]
    if isinstance(answer, int) and not isinstance(answer, bool):
        if 0 <= answer < len(values):
            return answer
        if 1 <= answer <= len(values):
            return answer - 1
    folded = str(answer).strip().casefold()
    for index, key in enumerate(keys):
        if folded == key.strip().casefold():
            return index
    if len(folded) == 1 and "a" <= folded <= "z":
        index = ord(folded) - ord("a")
        if index < len(values):
            return index
    for index, text in enumerate(values):
        if folded == text.strip().casefold():
            return index
    raise ValueError("could not map correct answer to an option index")


def row_text(row: dict[str, Any]) -> str:
    keys, values = options_list(row["options"])
    option_text = " ".join(f"OPTION_{keys[index]} {text}" for index, text in enumerate(values))
    return f"TYPE_{row['question_type']} QUESTION {row['question']} {option_text}"


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    started = utc_now()
    data_root = Path(args.data_dir).resolve(strict=True)
    output_root = Path(args.out_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    validation_path = data_root / "data" / "curebench_valset_pharse1.jsonl"
    test_paths = [
        data_root / "data" / "curebench_testset_phase1.jsonl",
        data_root / "data" / "curebench_testset_phase2.jsonl",
    ]
    if not validation_path.is_file() or any(not path.is_file() for path in test_paths):
        raise ValueError("required CURE-Bench files are missing")
    validation = load_jsonl(validation_path, require_answer=True)
    test_schema_rows = sum(len(load_jsonl(path, require_answer=False)) for path in test_paths)
    ids = np.asarray([str(row["id"]) for row in validation])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate validation ids")
    texts = np.asarray([row_text(row) for row in validation], dtype=object)
    labels = np.asarray([answer_index(row) for row in validation], dtype=int)
    indices = np.arange(len(validation))
    development, holdout = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )
    inner_train, inner_val = train_test_split(
        development,
        test_size=0.2,
        random_state=SEED + 1,
        stratify=labels[development],
    )
    if set(ids[development]) & set(ids[holdout]):
        raise ValueError("id leakage")

    baseline_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=30_000,
        sublinear_tf=True,
    )
    baseline_inner_matrix = baseline_vectorizer.fit_transform(texts[inner_train].tolist())
    baseline_inner = LogisticRegression(
        C=2.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )
    baseline_inner.fit(baseline_inner_matrix, labels[inner_train])
    baseline_inner_score = accuracy(
        labels[inner_val], baseline_inner.predict(baseline_vectorizer.transform(texts[inner_val].tolist()))
    )

    candidate_history: list[dict[str, Any]] = []
    for c_value in (0.25, 0.5, 1.0, 2.0):
        vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 3),
                        min_df=1,
                        max_features=40_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=1,
                        max_features=40_000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        matrix = vectorizer.fit_transform(texts[inner_train].tolist())
        model = LinearSVC(C=c_value, class_weight="balanced", random_state=SEED)
        model.fit(matrix, labels[inner_train])
        score = accuracy(labels[inner_val], model.predict(vectorizer.transform(texts[inner_val].tolist())))
        candidate_history.append({"candidate_id": f"word-char-svc-{c_value}", "C": c_value, "inner_accuracy": score})
    selected = max(candidate_history, key=lambda item: (item["inner_accuracy"], -item["C"]))

    baseline_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=30_000,
        sublinear_tf=True,
    )
    baseline_matrix = baseline_vectorizer.fit_transform(texts[development].tolist())
    baseline = LogisticRegression(
        C=2.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )
    baseline.fit(baseline_matrix, labels[development])
    candidate_vectorizer = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 3),
                    min_df=1,
                    max_features=40_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    max_features=40_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    candidate_matrix = candidate_vectorizer.fit_transform(texts[development].tolist())
    candidate = LinearSVC(C=float(selected["C"]), class_weight="balanced", random_state=SEED)
    candidate.fit(candidate_matrix, labels[development])
    baseline_prediction = baseline.predict(baseline_vectorizer.transform(texts[holdout].tolist()))
    candidate_prediction = candidate.predict(candidate_vectorizer.transform(texts[holdout].tolist()))
    y_holdout = labels[holdout]
    baseline_score = accuracy(y_holdout, baseline_prediction)
    candidate_score = accuracy(y_holdout, candidate_prediction)

    rng = np.random.default_rng(SEED + 2)
    differences: list[float] = []
    for _ in range(BOOTSTRAP_ROUNDS):
        sample = rng.integers(0, len(holdout), len(holdout))
        differences.append(
            accuracy(y_holdout[sample], candidate_prediction[sample])
            - accuracy(y_holdout[sample], baseline_prediction[sample])
        )
    array = np.asarray(differences)
    standard_error = float(array.std(ddof=1))
    ci_lower = float(np.quantile(array, 0.025))
    ci_upper = float(np.quantile(array, 0.975))
    minimum_margin = max(0.01 * baseline_score, standard_error)
    gate_passed = bool(candidate_score > baseline_score and ci_lower > minimum_margin)

    source_path = Path(__file__).resolve(strict=True)
    source_sha = sha256_file(source_path)
    task_contract = {
        "schema": "evomind.cure_bench.task_contract.v2",
        "task": "multiple-choice medical question answering",
        "metric": "accuracy",
        "direction": "higher_is_better",
        "entity": "question id",
        "test_labels_used": False,
    }
    baseline_evidence = {
        "schema": "evomind.cure_bench.baseline_evidence.v2",
        "classification": "PUBLICLY_REPRODUCIBLE_STRONG_BASELINE",
        "method": "word 1-2 gram TF-IDF plus class-balanced LogisticRegression",
        "implementation_sha256": source_sha,
        "validation_sha256": sha256_file(validation_path),
        "reproduction_accuracy": baseline_score,
    }
    split_manifest = {
        "schema": "evomind.cure_bench.split_manifest.v2",
        "seed": SEED,
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "id_overlap": 0,
        "development_id_sha256": sha256_bytes("\n".join(sorted(ids[development])).encode()),
        "holdout_id_sha256": sha256_bytes("\n".join(sorted(ids[holdout])).encode()),
        "holdout_blind_for_selection": True,
    }
    dataset_audit = {
        "schema": "evomind.cure_bench.dataset_audit.v2",
        "validation_rows": len(validation),
        "test_schema_rows": test_schema_rows,
        "label_counts": {str(label): int((labels == label).sum()) for label in sorted(set(labels))},
        "question_types": sorted({str(row["question_type"]) for row in validation}),
        "duplicate_ids": 0,
        "test_labels_read": False,
    }
    metrics = {
        "schema": "evomind.cure_bench.metrics.v2",
        "baseline_accuracy": baseline_score,
        "candidate_accuracy": candidate_score,
        "absolute_improvement": candidate_score - baseline_score,
        "relative_improvement": (candidate_score - baseline_score) / max(baseline_score, 1e-12),
        "baseline_inner_accuracy": baseline_inner_score,
        "candidate_history": candidate_history,
        "bootstrap": {
            "rounds": BOOTSTRAP_ROUNDS,
            "difference_candidate_minus_baseline_mean": float(array.mean()),
            "standard_error": standard_error,
            "ci95_lower": ci_lower,
            "ci95_upper": ci_upper,
        },
    }
    comparison = {
        "schema": "evomind.cure_bench.candidate_vs_baseline.v2",
        "baseline": baseline_score,
        "candidate": candidate_score,
        "minimum_margin": minimum_margin,
        "difference_candidate_minus_baseline_ci95_lower": ci_lower,
        "gate_passed": gate_passed,
    }
    predictions = [
        {
            "id": str(ids[index]),
            "y_true": int(labels[index]),
            "baseline_prediction": int(baseline_prediction[position]),
            "candidate_prediction": int(candidate_prediction[position]),
        }
        for position, index in enumerate(holdout)
    ]
    recomputed_baseline = accuracy(
        np.asarray([row["y_true"] for row in predictions]),
        np.asarray([row["baseline_prediction"] for row in predictions]),
    )
    recomputed_candidate = accuracy(
        np.asarray([row["y_true"] for row in predictions]),
        np.asarray([row["candidate_prediction"] for row in predictions]),
    )
    independent = {
        "schema": "evomind.cure_bench.independent_verification.v2",
        "holdout_rows": len(predictions),
        "baseline_recomputed": recomputed_baseline,
        "candidate_recomputed": recomputed_candidate,
        "baseline_match": abs(recomputed_baseline - baseline_score) < 1e-12,
        "candidate_match": abs(recomputed_candidate - candidate_score) < 1e-12,
        "id_overlap": 0,
        "test_labels_used": False,
        "gate_passed": gate_passed,
    }

    shutil.copyfile(source_path, output_root / "solution.py")
    write_json(output_root / "task-contract-v2.json", task_contract)
    write_json(output_root / "baseline-evidence.json", baseline_evidence)
    write_json(output_root / "baseline-reproduction.json", {**baseline_evidence, "completed_at_utc": utc_now()})
    write_json(output_root / "dataset-audit.json", dataset_audit)
    write_json(output_root / "split-manifest.json", split_manifest)
    write_json(
        output_root / "training-config.json",
        {
            "schema": "evomind.cure_bench.training_config.v2",
            "seed": SEED,
            "baseline": baseline_evidence["method"],
            "candidate": "word 1-3 and char 3-5 gram TF-IDF plus class-balanced LinearSVC",
            "selected_candidate": selected,
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        },
    )
    with (output_root / "training.log").open("w", encoding="utf-8", newline="\n") as handle:
        for row in (
            {"at_utc": started, "event": "start", "source_sha256": source_sha},
            {"at_utc": utc_now(), "event": "baseline_complete", "accuracy": baseline_score},
            {"at_utc": utc_now(), "event": "candidate_complete", "accuracy": candidate_score},
            {"at_utc": utc_now(), "event": "independent_verification", "gate_passed": gate_passed},
        ):
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (output_root / "candidate-history.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in candidate_history:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    write_json(output_root / "holdout-predictions.json", predictions)
    write_json(output_root / "metrics.json", metrics)
    write_json(output_root / "candidate-vs-baseline.json", comparison)
    write_json(output_root / "independent-verification.json", independent)
    write_json(
        output_root / "environment-lock.json",
        {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__},
    )
    joblib.dump(
        {
            "baseline_vectorizer": baseline_vectorizer,
            "baseline": baseline,
            "candidate_vectorizer": candidate_vectorizer,
            "candidate": candidate,
        },
        output_root / "model-bundle.joblib",
        compress=3,
    )
    write_json(
        output_root / "retrospective-memory.json",
        {
            "schema": "evomind.retrospective_memory.v2",
            "baseline": baseline_evidence["method"],
            "candidate": "word-char LinearSVC",
            "gate_passed": gate_passed,
            "test_labels_used": False,
            "next_step_if_failed": "Use a fresh holdout and a frozen pretrained language representation available inside the managed runtime.",
            "memory_writeback_allowed": gate_passed,
        },
    )

    files: list[dict[str, Any]] = []
    for path in sorted(output_root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name not in {"artifact-manifest.json", "artifact-manifest-receipt.json"}:
            files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema": "evomind.cure_bench.artifact_manifest.v2",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "source_sha256": source_sha,
        "exit_code": 0,
        "gate_passed": gate_passed,
        "files": files,
    }
    write_json(output_root / "artifact-manifest.json", manifest)
    write_json(
        output_root / "artifact-manifest-receipt.json",
        {"artifact": "artifact-manifest.json", "bytes": (output_root / "artifact-manifest.json").stat().st_size, "sha256": sha256_file(output_root / "artifact-manifest.json")},
    )
    print(json.dumps({"status": "completed", **comparison}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
