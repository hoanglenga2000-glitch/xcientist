#!/usr/bin/env python3
"""Independently reconstruct and verify a completed Spooky neural OOF run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

CLASS_COLUMNS = ("EAP", "HPL", "MWS")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_probability(values: np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != len(CLASS_COLUMNS):
        raise ValueError("Independent Spooky probability shape is invalid")
    if not np.isfinite(probability).all():
        raise ValueError("Independent Spooky probability contains non-finite values")
    probability = np.clip(probability, 1e-7, 1.0)
    return probability / probability.sum(axis=1, keepdims=True)


def independent_log_loss(truth: np.ndarray, probability: np.ndarray) -> float:
    return float(
        log_loss(
            np.asarray(truth, dtype=np.int64),
            normalize_probability(probability),
            labels=np.arange(len(CLASS_COLUMNS)),
        )
    )


def _simplex_weights(component_count: int, steps: int) -> list[tuple[float, ...]]:
    def compositions(total: int, parts: int, prefix: tuple[int, ...] = ()) -> Iterable[tuple[int, ...]]:
        if parts == 1:
            yield (*prefix, total)
            return
        for value in range(total + 1):
            yield from compositions(total - value, parts - 1, (*prefix, value))

    if component_count < 1 or steps < 1:
        raise ValueError("Independent simplex dimensions must be positive")
    return [tuple(value / steps for value in raw) for raw in compositions(steps, component_count)]


def apply_blend(
    components: Sequence[np.ndarray],
    weights: Sequence[float],
    *,
    temperature: float,
    mode: str,
) -> np.ndarray:
    numeric_weights = np.asarray(weights, dtype=np.float64)
    if len(components) != len(numeric_weights) or not components:
        raise ValueError("Independent Spooky blend inputs do not match")
    if np.any(numeric_weights < 0.0) or not math.isclose(
        float(numeric_weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("Independent Spooky blend weights are invalid")
    if temperature <= 0.0:
        raise ValueError("Independent Spooky temperature is invalid")
    stack = np.stack([normalize_probability(value) for value in components], axis=0)
    if mode == "probability":
        pooled = np.tensordot(numeric_weights, stack, axes=(0, 0))
        logits = np.log(np.clip(pooled, 1e-7, 1.0)) / temperature
    elif mode == "log_probability":
        logits = np.tensordot(numeric_weights, np.log(stack), axes=(0, 0)) / temperature
    else:
        raise ValueError(f"Unknown independent Spooky blend mode: {mode}")
    logits -= logits.max(axis=1, keepdims=True)
    exponential = np.exp(logits)
    return normalize_probability(exponential)


def select_blend(
    components: Sequence[np.ndarray],
    truth: np.ndarray,
    *,
    steps: int,
    temperatures: Sequence[float],
) -> dict[str, Any]:
    candidates = []
    weights = _simplex_weights(len(components), steps)
    for mode in ("probability", "log_probability"):
        for temperature in temperatures:
            for value in weights:
                probability = apply_blend(
                    components,
                    value,
                    temperature=float(temperature),
                    mode=mode,
                )
                candidates.append(
                    {
                        "mode": mode,
                        "temperature": float(temperature),
                        "weights": list(value),
                        "log_loss": independent_log_loss(truth, probability),
                    }
                )
    return min(
        candidates,
        key=lambda item: (
            item["log_loss"],
            abs(item["temperature"] - 1.0),
            item["mode"] != "probability",
            tuple(-value for value in item["weights"]),
        ),
    )


def reconstruct_exact_deployment(
    component_oof: dict[str, np.ndarray],
    component_test_by_fold: dict[str, np.ndarray],
    truth: np.ndarray,
    fold_assignment: np.ndarray,
    *,
    steps: int,
    temperatures: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    names = tuple(component_oof)
    if names != tuple(component_test_by_fold) or len(names) < 2:
        raise ValueError("Independent OOF/test component order differs")
    labels = np.asarray(truth, dtype=np.int64).reshape(-1)
    folds = np.asarray(fold_assignment, dtype=np.int16).reshape(-1)
    unique_folds = sorted(int(value) for value in np.unique(folds))
    if unique_folds != list(range(len(unique_folds))) or len(unique_folds) < 3:
        raise ValueError("Independent Spooky folds are not contiguous")
    oof_components = {name: normalize_probability(component_oof[name]) for name in names}
    if any(len(values) != len(labels) for values in oof_components.values()):
        raise ValueError("Independent Spooky OOF rows differ from truth")
    test_components = {
        name: np.asarray(component_test_by_fold[name], dtype=np.float64) for name in names
    }
    test_rows = test_components[names[0]].shape[1]
    expected = (len(unique_folds), test_rows, len(CLASS_COLUMNS))
    if any(values.shape != expected for values in test_components.values()):
        raise ValueError("Independent Spooky test component shape is invalid")

    candidate_oof = np.full((len(labels), len(CLASS_COLUMNS)), np.nan, dtype=np.float64)
    counts = np.zeros(len(labels), dtype=np.uint8)
    fold_test = np.zeros(expected, dtype=np.float64)
    records = []
    for fold in unique_folds:
        fit_mask = folds != fold
        score_mask = folds == fold
        selected = select_blend(
            [oof_components[name][fit_mask] for name in names],
            labels[fit_mask],
            steps=steps,
            temperatures=temperatures,
        )
        candidate_oof[score_mask] = apply_blend(
            [oof_components[name][score_mask] for name in names],
            selected["weights"],
            temperature=selected["temperature"],
            mode=selected["mode"],
        )
        counts[score_mask] += 1
        fold_test[fold] = apply_blend(
            [test_components[name][fold] for name in names],
            selected["weights"],
            temperature=selected["temperature"],
            mode=selected["mode"],
        )
        records.append(
            {
                "score_fold": fold,
                "fit_folds": [value for value in unique_folds if value != fold],
                "component_order": list(names),
                "selected": selected,
            }
        )
    if not np.all(counts == 1) or not np.isfinite(candidate_oof).all():
        raise RuntimeError("Independent Spooky OOF was not written exactly once")
    return candidate_oof, normalize_probability(fold_test.mean(axis=0)), counts, records


def normalize_duplicate_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value)).lower()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return " ".join(re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", text))


def verify_duplicate_fold_isolation(
    texts: Sequence[str],
    labels: np.ndarray,
    folds: np.ndarray,
) -> dict[str, Any]:
    groups: dict[str, dict[str, set[int]]] = {}
    for text, label, fold in zip(texts, labels, folds, strict=True):
        key = hashlib.sha256(normalize_duplicate_text(text).encode("utf-8")).hexdigest()
        record = groups.setdefault(key, {"labels": set(), "folds": set()})
        record["labels"].add(int(label))
        record["folds"].add(int(fold))
    conflicting = sum(len(value["labels"]) != 1 for value in groups.values())
    crossed = sum(len(value["folds"]) != 1 for value in groups.values())
    duplicate = len(texts) - len(groups)
    return {
        "rows": len(texts),
        "unique_groups": len(groups),
        "duplicate_rows_beyond_first": duplicate,
        "conflicting_label_groups": conflicting,
        "cross_fold_groups": crossed,
        "passed": conflicting == 0 and crossed == 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    public_dir = args.public_dir.resolve()
    plan_path = args.plan.resolve()
    output = (args.output or run_dir / "independent_verification.json").resolve()
    summary_path = run_dir / "summary.json"
    bundle_path = run_dir / "spooky_transformer_oof_and_test.npz"
    submission_path = run_dir / "candidate_submission_withheld.csv"
    for path in (plan_path, summary_path, bundle_path, submission_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    sample_path = public_dir / "sample_submission.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    submission = pd.read_csv(submission_path)
    class_to_index = {value: index for index, value in enumerate(CLASS_COLUMNS)}
    truth = train["author"].map(class_to_index).to_numpy(dtype=np.int64)
    with np.load(bundle_path, allow_pickle=False) as archive:
        required = {
            "truth",
            "fold",
            "sparse_oof",
            "byte_oof",
            "transformer_oof",
            "sparse_test_by_fold",
            "byte_test_by_fold",
            "transformer_test_by_fold",
            "candidate_oof",
            "candidate_test",
            "component_write_counts",
            "candidate_write_counts",
            "train_id",
            "test_id",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(f"Spooky prediction bundle is missing arrays: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    input_hashes = {
        "public_train_sha256": sha256_file(train_path),
        "public_test_sha256": sha256_file(test_path),
        "public_sample_submission_sha256": sha256_file(sample_path),
    }
    input_checks = {
        name: plan.get("inputs", {}).get(name) == value for name, value in input_hashes.items()
    }
    identity_checks = {
        "truth": np.array_equal(arrays["truth"].astype(np.int64), truth),
        "train_id": np.array_equal(
            arrays["train_id"].astype(str), train["id"].astype(str).to_numpy()
        ),
        "test_id": np.array_equal(
            arrays["test_id"].astype(str), test["id"].astype(str).to_numpy()
        ),
        "sample_id": sample["id"].astype(str).tolist() == test["id"].astype(str).tolist(),
        "submission_id": submission["id"].astype(str).tolist()
        == test["id"].astype(str).tolist(),
    }
    component_oof = {
        "sparse": arrays["sparse_oof"],
        "byte": arrays["byte_oof"],
        "transformer": arrays["transformer_oof"],
    }
    component_test = {
        "sparse": arrays["sparse_test_by_fold"],
        "byte": arrays["byte_test_by_fold"],
        "transformer": arrays["transformer_test_by_fold"],
    }
    reconstructed_oof, reconstructed_test, reconstructed_counts, records = (
        reconstruct_exact_deployment(
            component_oof,
            component_test,
            truth,
            arrays["fold"],
            steps=int(plan["ensemble"]["weight_grid_steps"]),
            temperatures=tuple(float(value) for value in plan["ensemble"]["temperatures"]),
        )
    )
    score = independent_log_loss(truth, reconstructed_oof)
    duplicate = verify_duplicate_fold_isolation(
        train["text"].fillna("").astype(str).tolist(),
        truth,
        arrays["fold"],
    )
    probability_checks = {
        "candidate_oof_reconstructed": bool(
            np.allclose(reconstructed_oof, arrays["candidate_oof"], rtol=0.0, atol=1e-12)
        ),
        "candidate_test_reconstructed": bool(
            np.allclose(reconstructed_test, arrays["candidate_test"], rtol=0.0, atol=1e-12)
        ),
        "candidate_counts": bool(
            np.all(reconstructed_counts == 1) and np.all(arrays["candidate_write_counts"] == 1)
        ),
        "component_counts": bool(np.all(arrays["component_write_counts"] == 1)),
        "submission_values": bool(
            np.allclose(
                submission[list(CLASS_COLUMNS)].to_numpy(dtype=np.float64),
                reconstructed_test,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "oof_normalized": bool(np.allclose(reconstructed_oof.sum(axis=1), 1.0, atol=1e-12)),
        "test_normalized": bool(np.allclose(reconstructed_test.sum(axis=1), 1.0, atol=1e-12)),
        "summary_score": math.isclose(
            score,
            float(summary.get("candidate_oof_log_loss", math.nan)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    contract_checks = {
        "plan_hash": summary.get("plan_sha256") == sha256_file(plan_path),
        "terminal_status": summary.get("status")
        in {"single_seed_gate_passed_confirmation_pending", "single_seed_gate_failed"},
        "input_hashes": all(input_checks.values()),
        "identities": all(identity_checks.values()),
        "duplicate_group_isolation": duplicate["passed"],
        "probabilities": all(probability_checks.values()),
        "private_labels_unused": summary.get("private_labels_used") is False,
        "official_grader_not_executed": summary.get("official_grader_executed") is False,
        "kaggle_submission_not_executed": summary.get("kaggle_submission_executed") is False,
    }
    report = {
        "schema": "evomind.spooky.independent_verification.v1",
        "created_at": now_iso(),
        "status": "passed" if all(contract_checks.values()) else "failed",
        "run_dir": str(run_dir),
        "run_id": summary.get("run_id"),
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "prediction_bundle": {"path": str(bundle_path), "sha256": sha256_file(bundle_path)},
        "submission_withheld": {
            "path": str(submission_path),
            "sha256": sha256_file(submission_path),
        },
        "input_hashes": input_hashes,
        "input_checks": input_checks,
        "identity_checks": identity_checks,
        "duplicate_groups": duplicate,
        "probability_checks": probability_checks,
        "contract_checks": contract_checks,
        "recomputed_candidate_oof_log_loss": score,
        "reconstructed_blend_records": records,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
    }
    write_json_atomic(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
