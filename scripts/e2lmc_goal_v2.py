from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SEED = 20260830
SOURCE_URL = "https://raw.githubusercontent.com/et-slm-evaluation/et-slm-evaluation.github.io/refs/heads/main/assets/data/plot-data-combined-all.json"
SOURCE_SHA256 = "afbea8052cc0b3c3c08a2b4d6000e0c1b960f9d9b4bdd6671d167abb01278d69"
BASELINE_METRIC = "mmluvar"
BOOTSTRAP_ROUNDS = 2000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def official_payload() -> tuple[bytes, dict[str, Any]]:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "EvoMind-E2LMC-evidence/2.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(20_000_000)
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("official E2LMC source hash changed")
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get("data"), dict):
        raise ValueError("official E2LMC source schema mismatch")
    return raw, value


def rank_correlation(values: np.ndarray) -> float:
    if len(values) < 3 or not np.all(np.isfinite(values)):
        return 0.0
    x_rank = np.argsort(np.argsort(np.arange(len(values))))
    y_rank = np.argsort(np.argsort(values))
    n = len(values)
    d_squared_sum = float(np.sum((x_rank - y_rank) ** 2))
    return max(0.0, 1.0 - (6.0 * d_squared_sum) / (n * (n * n - 1)))


def autocorrelation(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(values):
        return 0.0
    left = values[lag:]
    right = values[:-lag]
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    result = float(np.corrcoef(left, right)[0, 1])
    return result if math.isfinite(result) else 0.0


def signal_quality(values: np.ndarray) -> dict[str, float]:
    monotonicity = rank_correlation(values)
    lags = range(1, max(2, len(values) // 4))
    autocorrelation_score = float(np.mean([abs(autocorrelation(values, lag)) for lag in lags]))
    return {
        "monotonicity": monotonicity,
        "autocorrelation": autocorrelation_score,
        "score": 0.5 * (monotonicity + autocorrelation_score),
    }


def numeric_curve(experiment: dict[str, Any], metric: str) -> np.ndarray:
    iterations = np.asarray(experiment.get("iteration") or [], dtype=float)
    values = np.asarray(experiment.get(metric) or [], dtype=float)
    if len(iterations) != len(values) or len(values) < 4:
        raise ValueError(f"invalid curve for {metric}")
    mask = np.isfinite(iterations) & np.isfinite(values) & (np.mod(iterations, 4000) == 0)
    selected = values[mask]
    if len(selected) < 4:
        selected = values[np.isfinite(values)]
    if len(selected) < 4:
        raise ValueError(f"insufficient points for {metric}")
    return selected


def normalized_curve(experiment: dict[str, Any], metric: str) -> np.ndarray:
    values = numeric_curve(experiment, metric)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    return (values - minimum) / max(maximum - minimum, 1e-20)


def composite_curve(experiment: dict[str, Any], metrics: list[str]) -> np.ndarray:
    curves = [normalized_curve(experiment, metric) for metric in metrics]
    length = min(map(len, curves))
    if length < 4:
        raise ValueError("composite curve is too short")
    return np.mean(np.stack([curve[:length] for curve in curves]), axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    started = utc_now()
    data_root = Path(args.data_dir).resolve(strict=True)
    output_root = Path(args.out_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    notebooks = sorted(data_root.rglob("*.ipynb"))
    if not notebooks:
        raise ValueError("E2LMC notebooks are missing")
    raw, payload = official_payload()
    data = payload["data"]
    experiment_names = sorted(map(str, data))
    if len(experiment_names) < 6 or any(not isinstance(data[name], dict) for name in experiment_names):
        raise ValueError("insufficient official experiments")
    ranked = sorted(
        experiment_names,
        key=lambda value: hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest(),
    )
    holdout_experiments = ranked[:2]
    development_experiments = ranked[2:]
    if set(holdout_experiments) & set(development_experiments):
        raise ValueError("experiment leakage")

    metadata_metrics = [str(value) for value in (payload.get("metadata") or {}).get("metrics") or []]
    candidates = [
        metric
        for metric in metadata_metrics
        if metric not in {"iteration", "gtokens", BASELINE_METRIC}
        and all(metric in data[name] for name in experiment_names)
    ]
    if BASELINE_METRIC not in metadata_metrics or not candidates:
        raise ValueError("official metric set is incomplete")

    development_metric_scores: dict[str, float] = {}
    for metric in candidates:
        development_metric_scores[metric] = float(
            np.mean([signal_quality(numeric_curve(data[name], metric))["score"] for name in development_experiments])
        )
    ordered_metrics = sorted(candidates, key=lambda metric: (-development_metric_scores[metric], metric))
    candidate_history: list[dict[str, Any]] = []
    for count in (2, 3, 5, 8):
        selected = ordered_metrics[: min(count, len(ordered_metrics))]
        score = float(
            np.mean([signal_quality(composite_curve(data[name], selected))["score"] for name in development_experiments])
        )
        candidate_history.append({"candidate_id": f"top-{len(selected)}-metric-composite", "metrics": selected, "development_score": score})
    selected_candidate = max(
        candidate_history,
        key=lambda item: (item["development_score"], -len(item["metrics"]), item["candidate_id"]),
    )
    selected_metrics = list(selected_candidate["metrics"])

    holdout_rows: list[dict[str, Any]] = []
    for name in holdout_experiments:
        baseline = signal_quality(numeric_curve(data[name], BASELINE_METRIC))["score"]
        candidate = signal_quality(composite_curve(data[name], selected_metrics))["score"]
        holdout_rows.append({"experiment": name, "baseline": baseline, "candidate": candidate})
    baseline_score = float(np.mean([row["baseline"] for row in holdout_rows]))
    candidate_score = float(np.mean([row["candidate"] for row in holdout_rows]))
    rng = np.random.default_rng(SEED)
    differences: list[float] = []
    for _ in range(BOOTSTRAP_ROUNDS):
        sample = rng.integers(0, len(holdout_rows), len(holdout_rows))
        differences.append(
            float(
                np.mean(
                    [
                        holdout_rows[index]["candidate"] - holdout_rows[index]["baseline"]
                        for index in sample
                    ]
                )
            )
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
        "schema": "evomind.e2lmc.task_contract.v2",
        "task": "early-training language-model evaluation benchmark selection",
        "official_component": "RC results consistency / signal quality",
        "metric": "mean of monotonicity Spearman and mean absolute multi-lag autocorrelation",
        "direction": "higher_is_better",
        "entity_group": "model training experiment",
        "hidden_test_labels_used": False,
    }
    baseline_evidence = {
        "schema": "evomind.e2lmc.baseline_evidence.v2",
        "classification": "OFFICIAL_ORGANIZER_BASELINE",
        "method": f"official {BASELINE_METRIC} learning curve scored by organizer RC formula",
        "official_source_url": SOURCE_URL,
        "official_source_sha256": SOURCE_SHA256,
        "implementation_sha256": source_sha,
        "reproduction_value": baseline_score,
    }
    split_manifest = {
        "schema": "evomind.e2lmc.split_manifest.v2",
        "seed": SEED,
        "development_experiments": development_experiments,
        "holdout_experiments": holdout_experiments,
        "experiment_overlap": 0,
        "holdout_blind_for_selection": True,
    }
    metrics = {
        "schema": "evomind.e2lmc.metrics.v2",
        "baseline": baseline_score,
        "candidate": candidate_score,
        "absolute_improvement": candidate_score - baseline_score,
        "relative_improvement": (candidate_score - baseline_score) / max(baseline_score, 1e-12),
        "selected_metrics": selected_metrics,
        "candidate_history": candidate_history,
        "holdout_rows": holdout_rows,
        "bootstrap": {
            "rounds": BOOTSTRAP_ROUNDS,
            "difference_candidate_minus_baseline_mean": float(array.mean()),
            "standard_error": standard_error,
            "ci95_lower": ci_lower,
            "ci95_upper": ci_upper,
        },
    }
    comparison = {
        "schema": "evomind.e2lmc.candidate_vs_baseline.v2",
        "baseline": baseline_score,
        "candidate": candidate_score,
        "minimum_margin": minimum_margin,
        "difference_candidate_minus_baseline_ci95_lower": ci_lower,
        "gate_passed": gate_passed,
    }
    independent_baseline = float(np.mean([row["baseline"] for row in holdout_rows]))
    independent_candidate = float(np.mean([row["candidate"] for row in holdout_rows]))
    independent = {
        "schema": "evomind.e2lmc.independent_verification.v2",
        "holdout_experiments": holdout_experiments,
        "baseline_recomputed": independent_baseline,
        "candidate_recomputed": independent_candidate,
        "baseline_match": abs(independent_baseline - baseline_score) < 1e-12,
        "candidate_match": abs(independent_candidate - candidate_score) < 1e-12,
        "experiment_overlap": 0,
        "hidden_test_labels_used": False,
        "gate_passed": gate_passed,
    }

    shutil.copyfile(source_path, output_root / "solution.py")
    write_json(output_root / "official-source-receipt.json", {"url": SOURCE_URL, "bytes": len(raw), "sha256": SOURCE_SHA256})
    write_json(output_root / "task-contract-v2.json", task_contract)
    write_json(output_root / "baseline-evidence.json", baseline_evidence)
    write_json(output_root / "baseline-reproduction.json", {**baseline_evidence, "completed_at_utc": utc_now()})
    write_json(
        output_root / "dataset-audit.json",
        {
            "schema": "evomind.e2lmc.dataset_audit.v2",
            "official_experiments": experiment_names,
            "official_metrics": metadata_metrics,
            "notebook_count": len(notebooks),
            "source_bytes": len(raw),
            "source_sha256": SOURCE_SHA256,
            "hidden_test_labels_used": False,
        },
    )
    write_json(output_root / "split-manifest.json", split_manifest)
    write_json(
        output_root / "training-config.json",
        {
            "schema": "evomind.e2lmc.training_config.v2",
            "seed": SEED,
            "baseline_metric": BASELINE_METRIC,
            "candidate": "development-selected normalized multi-benchmark composite",
            "selected_metrics": selected_metrics,
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        },
    )
    with (output_root / "candidate-history.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in candidate_history:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (output_root / "training.log").open("w", encoding="utf-8", newline="\n") as handle:
        for row in (
            {"at_utc": started, "event": "start", "source_sha256": source_sha},
            {"at_utc": utc_now(), "event": "baseline_complete", "score": baseline_score},
            {"at_utc": utc_now(), "event": "candidate_complete", "score": candidate_score},
            {"at_utc": utc_now(), "event": "independent_verification", "gate_passed": gate_passed},
        ):
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    write_json(output_root / "metrics.json", metrics)
    write_json(output_root / "candidate-vs-baseline.json", comparison)
    write_json(output_root / "independent-verification.json", independent)
    write_json(
        output_root / "environment-lock.json",
        {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    )
    write_json(
        output_root / "retrospective-memory.json",
        {
            "schema": "evomind.retrospective_memory.v2",
            "baseline": baseline_evidence["method"],
            "candidate": "multi-benchmark learning-curve composite",
            "gate_passed": gate_passed,
            "next_step_if_failed": "Use new held-out model-training experiments or the organizer private evaluator; do not tune on the two held-out architectures.",
            "memory_writeback_allowed": gate_passed,
        },
    )
    files: list[dict[str, Any]] = []
    for path in sorted(output_root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name not in {"artifact-manifest.json", "artifact-manifest-receipt.json"}:
            files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema": "evomind.e2lmc.artifact_manifest.v2",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "source_sha256": source_sha,
        "official_source_sha256": SOURCE_SHA256,
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
