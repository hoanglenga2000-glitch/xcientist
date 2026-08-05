#!/usr/bin/env python3
"""Validate calibrated linear Leaf models with untouched confirmation seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import mlebench_medal_recovery_adapters as recovery
from scripts import mlebench_remote_ops as remote_ops


COMPETITION_ID = "leaf-classification"
REMOTE_PUBLIC_DIR = f"{remote_ops.REMOTE_DATA_ROOT}/{COMPETITION_ID}/prepared/public"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "leaf_linear_public_oof_validation_current.json"
)


def _remote_source(args: argparse.Namespace) -> str:
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    logistic_cs = [float(value) for value in args.logistic_cs.split(",") if value.strip()]
    linear_svc_cs = [float(value) for value in args.linear_svc_cs.split(",") if value.strip()]
    if len(seeds) < 3:
        raise ValueError("Leaf linear validation requires one development and two confirmation seeds")
    if not logistic_cs or not linear_svc_cs:
        raise ValueError("Leaf linear search grids must not be empty")
    return f'''from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PUBLIC_DIR = Path({REMOTE_PUBLIC_DIR!r})
train = pd.read_csv(PUBLIC_DIR / "train.csv").reset_index(drop=True)
sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv")
features = [column for column in train.columns if column not in {{"id", "species"}}]
classes = [str(column) for column in sample.columns[1:]]
target = train["species"].astype(str).to_numpy()
x = train[features].astype(np.float64)
if set(target) != set(classes) or not np.isfinite(x.to_numpy()).all():
    raise RuntimeError("Leaf schema, classes, or features are invalid")
temperatures = np.geomspace(0.03, 3.0, 81)

def softmax(values):
    matrix = np.asarray(values, dtype=np.float64)
    matrix -= matrix.max(axis=1, keepdims=True)
    exp = np.exp(matrix)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-15)

def align_matrix(raw, model_classes):
    raw = np.asarray(raw, dtype=np.float64)
    index = {{str(value): position for position, value in enumerate(model_classes)}}
    if set(index) != set(classes):
        raise RuntimeError("Leaf model classes do not match submission classes")
    return np.column_stack([raw[:, index[value]] for value in classes])

def calibrate(probability, temperature):
    powered = np.power(np.clip(probability, 1e-15, 1.0), 1.0 / float(temperature))
    return powered / np.maximum(powered.sum(axis=1, keepdims=True), 1e-15)

def crossfit_temperature(probability, labels, assignment):
    result = np.full_like(probability, np.nan, dtype=np.float64)
    records = []
    for fold in sorted(int(value) for value in np.unique(assignment)):
        valid = assignment == fold
        fit = ~valid
        candidates = []
        for temperature in temperatures:
            score = float(log_loss(labels[fit], calibrate(probability[fit], temperature), labels=classes))
            candidates.append((score, abs(float(temperature) - 1.0), float(temperature)))
        _, _, selected = min(candidates)
        result[valid] = calibrate(probability[valid], selected)
        records.append({{
            "fold": fold,
            "temperature": selected,
            "outer_log_loss": float(log_loss(labels[valid], result[valid], labels=classes)),
        }})
    if not np.isfinite(result).all():
        raise RuntimeError("Leaf calibrated OOF coverage is incomplete")
    return result, records

def run_candidate(seed, family, c_value):
    splitter = StratifiedKFold(n_splits={args.folds}, shuffle=True, random_state=seed)
    assignment = np.full(len(train), -1, dtype=np.int16)
    base = np.full((len(train), len(classes)), np.nan, dtype=np.float64)
    raw_fold_scores = []
    for fold, (fit_indices, valid_indices) in enumerate(splitter.split(x, target)):
        if family == "logistic":
            model = Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=c_value, solver="lbfgs", max_iter=4000,
                                              random_state=seed + fold)),
            ])
            model.fit(x.iloc[fit_indices], target[fit_indices])
            probability = align_matrix(model.predict_proba(x.iloc[valid_indices]), model.classes_)
        elif family == "linear_svc":
            model = Pipeline([
                ("scale", StandardScaler()),
                ("model", SVC(C=c_value, kernel="linear", probability=False,
                              decision_function_shape="ovr", random_state=seed + fold,
                              cache_size=2048)),
            ])
            model.fit(x.iloc[fit_indices], target[fit_indices])
            decision = align_matrix(model.decision_function(x.iloc[valid_indices]), model.classes_)
            probability = softmax(decision)
        else:
            raise RuntimeError("Unsupported Leaf linear candidate")
        base[valid_indices] = probability
        assignment[valid_indices] = fold
        raw_fold_scores.append(float(log_loss(target[valid_indices], probability, labels=classes)))
    if np.any(assignment < 0) or not np.isfinite(base).all():
        raise RuntimeError("Leaf candidate OOF coverage is incomplete")
    calibrated, records = crossfit_temperature(base, target, assignment)
    return {{
        "seed": seed,
        "family": family,
        "c_value": c_value,
        "raw_oof_log_loss": float(log_loss(target, base, labels=classes)),
        "cross_fitted_calibrated_log_loss": float(log_loss(target, calibrated, labels=classes)),
        "raw_fold_log_loss": raw_fold_scores,
        "crossfit_temperature_records": records,
        "fold_assignment_complete": True,
    }}

seeds = {seeds!r}
development = []
for c_value in {logistic_cs!r}:
    development.append(run_candidate(seeds[0], "logistic", c_value))
for c_value in {linear_svc_cs!r}:
    development.append(run_candidate(seeds[0], "linear_svc", c_value))
selected = min(
    development,
    key=lambda row: (row["cross_fitted_calibrated_log_loss"], row["family"], row["c_value"]),
)
confirmations = [
    run_candidate(seed, selected["family"], selected["c_value"])
    for seed in seeds[1:]
]
print("EVOMIND_RESULT=" + json.dumps({{
    "rows": len(train),
    "feature_count": len(features),
    "class_count": len(classes),
    "folds": {args.folds},
    "development_seed": seeds[0],
    "confirmation_seeds": seeds[1:],
    "development_grid": development,
    "selected": {{
        "family": selected["family"],
        "c_value": selected["c_value"],
        "development_cross_fitted_calibrated_log_loss": selected["cross_fitted_calibrated_log_loss"],
    }},
    "confirmations": confirmations,
}}, sort_keys=True))
'''


def _run_remote(client: Any, source: str, timeout: int) -> tuple[int, str, str]:
    site = remote_ops.ensure_remote_path(remote_ops.REMOTE_UNIFIED_SITE_PACKAGES)
    command = (
        f"PYTHONPATH={site} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "OPENBLAS_NUM_THREADS=1 nice -n 15 python3 -"
    )
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    stdout.channel.settimeout(timeout)
    stderr.channel.settimeout(timeout)
    stdin.write(source)
    stdin.flush()
    stdin.channel.shutdown_write()
    output = stdout.read().decode("utf-8", "replace")
    error = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), output, error


def _parse(output: str) -> dict[str, Any]:
    markers = [line for line in output.splitlines() if line.startswith("EVOMIND_RESULT=")]
    if len(markers) != 1:
        raise RuntimeError("Leaf linear validation did not emit exactly one result marker")
    return json.loads(markers[0].split("=", 1)[1])


def run(args: argparse.Namespace) -> dict[str, Any]:
    remote_ops.ensure_remote_path(REMOTE_PUBLIC_DIR)
    source = _remote_source(args)
    client = remote_ops._connect()
    try:
        remote_ops.verify_remote_runtime(client)
        code, output, error = _run_remote(client, source, args.timeout)
    finally:
        client.close()
    if code:
        raise RuntimeError(f"Leaf linear validation failed: {error[-1600:] or output[-1600:]}")
    metrics = _parse(output)
    bronze_threshold = 0.01526
    target_threshold = 0.013
    confirmation_scores = [
        float(row["cross_fitted_calibrated_log_loss"]) for row in metrics["confirmations"]
    ]
    payload = {
        "schema": "evomind.mlebench.leaf_linear_public_oof_validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "competition_id": COMPETITION_ID,
        "dataset": REMOTE_PUBLIC_DIR,
        "access_mode": "remote_read_only_cpu_probe_single_thread_nice15",
        "path_scope_verified": True,
        "remote_stderr_type": "nonempty" if error.strip() else "",
        "production_adapter_sha256": hashlib.sha256(Path(recovery.__file__).read_bytes()).hexdigest(),
        "remote_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "bronze_threshold": bronze_threshold,
        "target_threshold": target_threshold,
        **metrics,
        "confirmation_scores": confirmation_scores,
        "bronze_oriented_gate_passed": all(
            score <= bronze_threshold for score in confirmation_scores
        ),
        "target_passed": all(score <= target_threshold for score in confirmation_scores),
        "selection_boundary": (
            "Seed 42 selected one fixed family and C. Seeds 43 and 44 are untouched confirmations. "
            "Every temperature is selected without its outer fold. Internal OOF is not an official medal."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--logistic-cs", default="0.3,1,3,10,30,100,300")
    parser.add_argument("--linear-svc-cs", default="0.003,0.01,0.03,0.1,0.3,1")
    parser.add_argument("--timeout", type=int, default=1200)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "selected": payload["selected"],
        "confirmation_scores": payload["confirmation_scores"],
        "bronze_threshold": payload["bronze_threshold"],
        "bronze_oriented_gate_passed": payload["bronze_oriented_gate_passed"],
        "target_passed": payload["target_passed"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
