#!/usr/bin/env python3
"""Run a bounded, read-only Leaf recovery OOF search on HPC CPU.

The first seed selects one RBF-SVC configuration.  Later seeds are untouched
confirmation runs.  Component blending is cross-fitted inside every seed, and
no private labels or private-grader feedback are used.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
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
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "leaf_public_oof_validation_current.json"
)


def _production_helper_source() -> str:
    functions = (
        recovery.multiclass_temperature_scale,
        recovery.apply_multiclass_logloss_blend,
        recovery.select_multiclass_logloss_blend,
        recovery.cross_fit_multiclass_logloss_blend,
        recovery._aligned_multiclass_probability,
    )
    return "\n\n".join(inspect.getsource(function).strip() for function in functions)


def _remote_source(args: argparse.Namespace) -> str:
    helpers = _production_helper_source()
    c_values = [float(value) for value in args.svc_c_values.split(",") if value.strip()]
    gamma_values: list[str | float] = []
    for value in args.svc_gamma_values.split(","):
        value = value.strip()
        if value:
            gamma_values.append(value if value in {"scale", "auto"} else float(value))
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 3:
        raise ValueError("Leaf validation requires one development and at least two confirmation seeds")
    if not c_values or not gamma_values:
        raise ValueError("Leaf SVC search grid must not be empty")
    return f'''from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

{helpers}

PUBLIC_DIR = Path({REMOTE_PUBLIC_DIR!r})
train = pd.read_csv(PUBLIC_DIR / "train.csv").reset_index(drop=True)
test = pd.read_csv(PUBLIC_DIR / "test.csv").reset_index(drop=True)
sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv")
if list(sample.columns[:1]) != ["id"] or "id" not in train or "id" not in test:
    raise RuntimeError("Leaf ID or sample-submission schema is invalid")
if train["id"].duplicated().any() or test["id"].duplicated().any():
    raise RuntimeError("Leaf train or test IDs are duplicated")
features = [column for column in train.columns if column not in {{"id", "species"}}]
classes = [str(column) for column in sample.columns[1:]]
target = train["species"].astype(str).to_numpy()
if set(target) != set(classes):
    raise RuntimeError("Leaf labels and sample class columns differ")
x = train[features].astype(np.float64)
if not np.isfinite(x.to_numpy()).all():
    raise RuntimeError("Leaf production features contain non-finite values")

def svc_builder(seed: int, c_value: float, gamma: Any):
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", SVC(C=c_value, gamma=gamma, probability=True,
                      random_state=seed, cache_size=2048)),
    ])

def other_builders(seed: int):
    return [
        Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C={args.logistic_c}, solver="lbfgs",
                                          max_iter=2500, random_state=seed)),
        ]),
    ]

def run_seed(seed: int, c_value: float, gamma: Any, include_components: bool):
    folds = StratifiedKFold(n_splits={args.folds}, shuffle=True, random_state=seed)
    assignment = np.full(len(train), -1, dtype=np.int16)
    components = [np.zeros((len(train), len(classes)), dtype=np.float64) for _ in range(3)]
    fold_records = []
    for fold, (fit_indices, valid_indices) in enumerate(folds.split(x, target)):
        assignment[valid_indices] = fold
        models = [svc_builder(seed + fold, c_value, gamma)]
        if include_components:
            models.extend(other_builders(seed + fold))
        for model_index, model in enumerate(models):
            model.fit(x.iloc[fit_indices], target[fit_indices])
            components[model_index][valid_indices] = _aligned_multiclass_probability(
                model, x.iloc[valid_indices], classes
            )
        fold_records.append({{
            "fold": fold,
            "svc_log_loss": float(log_loss(
                target[valid_indices], components[0][valid_indices], labels=classes
            )),
        }})
    if np.any(assignment < 0) or not np.isfinite(components[0]).all():
        raise RuntimeError("Leaf OOF assignment or SVC probability is incomplete")
    svc_score = float(log_loss(target, components[0], labels=classes))
    result = {{
        "seed": seed,
        "svc_c": c_value,
        "svc_gamma": gamma,
        "svc_oof_log_loss": svc_score,
        "fold_records": fold_records,
        "fold_assignment_complete": True,
    }}
    if include_components:
        if any(not np.isfinite(component).all() for component in components):
            raise RuntimeError("Leaf component OOF probability is non-finite")
        crossfit, records = cross_fit_multiclass_logloss_blend(
            components, target, classes, assignment
        )
        result.update({{
            "lda_oof_log_loss": float(log_loss(target, components[1], labels=classes)),
            "logistic_oof_log_loss": float(log_loss(target, components[2], labels=classes)),
            "cross_fitted_blend_log_loss": float(log_loss(target, crossfit, labels=classes)),
            "cross_fitted_meta_validation": True,
            "crossfit_records": records,
        }})
    return result

seeds = {seeds!r}
c_values = {c_values!r}
gamma_values = {gamma_values!r}
development = []
for c_value in c_values:
    for gamma in gamma_values:
        development.append(run_seed(seeds[0], c_value, gamma, False))
selected = min(
    development,
    key=lambda row: (row["svc_oof_log_loss"], row["svc_c"], str(row["svc_gamma"])),
)
confirmations = [
    run_seed(seed, selected["svc_c"], selected["svc_gamma"], True)
    for seed in seeds[1:]
]
print("EVOMIND_RESULT=" + json.dumps({{
    "rows": len(train),
    "test_rows": len(test),
    "feature_count": len(features),
    "class_count": len(classes),
    "folds": {args.folds},
    "development_seed": seeds[0],
    "confirmation_seeds": seeds[1:],
    "development_grid": development,
    "selected": {{"svc_c": selected["svc_c"], "svc_gamma": selected["svc_gamma"],
                  "development_svc_oof_log_loss": selected["svc_oof_log_loss"]}},
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
        raise RuntimeError("Leaf validation did not emit exactly one result marker")
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
        raise RuntimeError(f"Leaf validation failed: {error[-1600:] or output[-1600:]}")
    metrics = _parse(output)
    bronze_threshold = 0.01526
    target_threshold = 0.013
    confirmation_scores = [
        min(float(row["svc_oof_log_loss"]), float(row["cross_fitted_blend_log_loss"]))
        for row in metrics["confirmations"]
    ]
    payload = {
        "schema": "evomind.mlebench.leaf_public_oof_validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "competition_id": COMPETITION_ID,
        "dataset": REMOTE_PUBLIC_DIR,
        "access_mode": "remote_read_only_cpu_probe_single_thread_nice15",
        "path_scope_verified": True,
        "remote_stderr_type": "nonempty" if error.strip() else "",
        "production_adapter_sha256": hashlib.sha256(Path(recovery.__file__).read_bytes()).hexdigest(),
        "production_helper_source_sha256": hashlib.sha256(
            _production_helper_source().encode()
        ).hexdigest(),
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
            "The first seed selected one SVC configuration; only later seeds count as confirmations. "
            "Internal OOF is not an official medal."
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
    parser.add_argument("--svc-c-values", default="10,30,100")
    parser.add_argument("--svc-gamma-values", default="scale,0.001,0.003,0.01")
    parser.add_argument("--logistic-c", type=float, default=10.0)
    parser.add_argument("--timeout", type=int, default=1800)
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
