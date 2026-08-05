#!/usr/bin/env python3
"""Validate the complete production Pizza text/structured OOF stack on HPC CPU."""

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

from scripts import mlebench_remote_ops as remote_ops
from scripts import mlebench_wave2_adapters as wave2
from scripts import run_mlebench_lite_full as runner


COMPETITION_ID = "random-acts-of-pizza"
REMOTE_PUBLIC_DIR = f"{remote_ops.REMOTE_DATA_ROOT}/{COMPETITION_ID}/prepared/public"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "pizza_full_public_oof_validation_current.json"
)


def _production_source() -> str:
    sources = [
        inspect.getsource(wave2.compute_nb_log_count_ratio).strip(),
        inspect.getsource(wave2._fit_nbsvm).strip(),
        inspect.getsource(runner._pizza_text).strip(),
        inspect.getsource(runner.fit_pizza_nbsvm_fold).strip(),
        inspect.getsource(runner.pizza_structured_features).strip(),
        inspect.getsource(runner._rank_fraction).strip(),
        inspect.getsource(runner.apply_binary_auc_blend).strip(),
        inspect.getsource(runner.select_binary_auc_blend).strip(),
        inspect.getsource(runner.cross_fit_binary_auc_blend).strip(),
    ]
    return "\n\n".join(sources).replace("wave2._fit_nbsvm", "_fit_nbsvm")


def _remote_source(args: argparse.Namespace) -> str:
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 1:
        raise ValueError("Pizza validation requires at least one seed")
    production = _production_source()
    return f'''from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

{production}

PUBLIC_DIR = Path({REMOTE_PUBLIC_DIR!r})
train = pd.read_json(PUBLIC_DIR / "train.json").reset_index(drop=True)
if "request_id" not in train or "requester_received_pizza" not in train:
    raise RuntimeError("Pizza training schema is incomplete")
if train["request_id"].duplicated().any():
    raise RuntimeError("Pizza training IDs are duplicated")
text = _pizza_text(train)
labels = train["requester_received_pizza"].astype(int)
structured = pizza_structured_features(train)
if not np.isfinite(structured.to_numpy()).all():
    raise RuntimeError("Pizza structured features contain non-finite values")

def run_seed(seed):
    splitter = StratifiedKFold(n_splits={args.folds}, shuffle=True, random_state=seed)
    text_oof = np.full(len(train), np.nan, dtype=np.float64)
    structured_oof = np.full(len(train), np.nan, dtype=np.float64)
    assignment = np.full(len(train), -1, dtype=np.int16)
    fold_records = []
    for fold, (fit_indices, valid_indices) in enumerate(splitter.split(text, labels)):
        text_prediction, _, artifacts = fit_pizza_nbsvm_fold(
            text.iloc[fit_indices], labels.iloc[fit_indices], text.iloc[valid_indices],
            text.iloc[valid_indices], word_features={args.word_features},
            char_features={args.char_features}, c_value={args.c_value}, seed=seed + fold,
        )
        model = CatBoostClassifier(
            iterations={args.catboost_iterations},
            depth=6,
            learning_rate=0.04,
            loss_function="Logloss",
            eval_metric="AUC",
            task_type="CPU",
            thread_count=1,
            random_seed=seed + fold,
            border_count=64,
            l2_leaf_reg=6.0,
            random_strength=0.35,
            od_type="Iter",
            od_wait=60,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            structured.iloc[fit_indices], labels.iloc[fit_indices],
            eval_set=(structured.iloc[valid_indices], labels.iloc[valid_indices]),
            use_best_model=True,
        )
        structured_prediction = model.predict_proba(structured.iloc[valid_indices])[:, 1]
        text_oof[valid_indices] = text_prediction
        structured_oof[valid_indices] = structured_prediction
        assignment[valid_indices] = fold
        fold_records.append({{
            "fold": fold,
            "train_rows": len(fit_indices),
            "valid_rows": len(valid_indices),
            "text_auc": float(roc_auc_score(labels.iloc[valid_indices], text_prediction)),
            "structured_auc": float(roc_auc_score(labels.iloc[valid_indices], structured_prediction)),
            "word_features": int(artifacts["word_features"]),
            "char_features": int(artifacts["char_features"]),
            "catboost_best_iteration": int(model.get_best_iteration()),
        }})
    if np.any(assignment < 0):
        raise RuntimeError("Pizza fold assignment did not cover every row")
    if any(not np.isfinite(values).all() or np.ptp(values) <= 1e-12
           for values in (text_oof, structured_oof)):
        raise RuntimeError("Pizza OOF component is non-finite or constant")
    blended, records = cross_fit_binary_auc_blend(
        text_oof, structured_oof, labels, assignment
    )
    return {{
        "seed": seed,
        "text_oof_auc": float(roc_auc_score(labels, text_oof)),
        "structured_oof_auc": float(roc_auc_score(labels, structured_oof)),
        "cross_fitted_blend_auc": float(roc_auc_score(labels, blended)),
        "fold_assignment_complete": True,
        "cross_fitted_meta_validation": True,
        "fold_records": fold_records,
        "crossfit_records": records,
    }}

results = [run_seed(seed) for seed in {seeds!r}]
print("EVOMIND_RESULT=" + json.dumps({{
    "rows": len(train),
    "positive_rows": int(labels.sum()),
    "structured_feature_count": len(structured.columns),
    "folds": {args.folds},
    "seeds": {seeds!r},
    "word_features": {args.word_features},
    "char_features": {args.char_features},
    "c_value": {args.c_value},
    "catboost_iterations": {args.catboost_iterations},
    "results": results,
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
        raise RuntimeError("Pizza full validation did not emit exactly one result marker")
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
        raise RuntimeError(f"Pizza full validation failed: {error[-1600:] or output[-1600:]}")
    metrics = _parse(output)
    bronze_threshold = 0.6921
    target_threshold = 0.70
    scores = [float(row["cross_fitted_blend_auc"]) for row in metrics["results"]]
    payload = {
        "schema": "evomind.mlebench.pizza_full_public_oof_validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "competition_id": COMPETITION_ID,
        "dataset": REMOTE_PUBLIC_DIR,
        "access_mode": "remote_read_only_cpu_probe_single_thread_nice15",
        "path_scope_verified": True,
        "remote_stderr_type": "nonempty" if error.strip() else "",
        "production_runner_sha256": hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest(),
        "production_source_sha256": hashlib.sha256(_production_source().encode()).hexdigest(),
        "remote_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "bronze_threshold": bronze_threshold,
        "target_threshold": target_threshold,
        **metrics,
        "cross_fitted_blend_scores": scores,
        "bronze_oriented_gate_passed": len(scores) >= 2 and all(
            score >= bronze_threshold for score in scores
        ),
        "target_passed": len(scores) >= 2 and all(score >= target_threshold for score in scores),
        "selection_boundary": (
            "The production text and structured parameters were fixed before all seeds. "
            "Every blend is fitted without its outer fold. Internal OOF is not an official medal."
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
    parser.add_argument("--word-features", type=int, default=90_000)
    parser.add_argument("--char-features", type=int, default=110_000)
    parser.add_argument("--c-value", type=float, default=4.0)
    parser.add_argument("--catboost-iterations", type=int, default=650)
    parser.add_argument("--timeout", type=int, default=1800)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "scores": payload["cross_fitted_blend_scores"],
        "bronze_threshold": payload["bronze_threshold"],
        "bronze_oriented_gate_passed": payload["bronze_oriented_gate_passed"],
        "target_passed": payload["target_passed"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
