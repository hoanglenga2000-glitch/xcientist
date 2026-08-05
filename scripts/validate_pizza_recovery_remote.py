#!/usr/bin/env python3
"""Validate the production Pizza NB-SVM channel on HPC without remote writes."""

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
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "pizza_nbsvm_public_oof_validation_current.json"
)


def _remote_model_source() -> str:
    fit_source = inspect.getsource(wave2._fit_nbsvm).strip()
    fold_source = inspect.getsource(runner.fit_pizza_nbsvm_fold).strip().replace(
        "wave2._fit_nbsvm", "_fit_nbsvm"
    )
    return "\n\n".join((
        inspect.getsource(wave2.compute_nb_log_count_ratio).strip(),
        fit_source,
        inspect.getsource(runner._pizza_text).strip(),
        fold_source,
    ))


def _remote_source(args: argparse.Namespace) -> str:
    model_source = _remote_model_source()
    return f'''from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

{model_source}

PUBLIC_DIR = Path({REMOTE_PUBLIC_DIR!r})
train = pd.read_json(PUBLIC_DIR / "train.json").reset_index(drop=True)
if "request_id" not in train or "requester_received_pizza" not in train:
    raise RuntimeError("Pizza training schema is incomplete")
if train["request_id"].duplicated().any():
    raise RuntimeError("Pizza training IDs are duplicated")
text = _pizza_text(train)
labels = train["requester_received_pizza"].astype(int)
splitter = StratifiedKFold(n_splits={args.folds}, shuffle=True, random_state={args.seed})
oof = np.full(len(train), np.nan, dtype=np.float64)
assignment = np.full(len(train), -1, dtype=np.int16)
fold_records = []
for fold, (fit_indices, valid_indices) in enumerate(splitter.split(text, labels)):
    prediction, _, artifacts = fit_pizza_nbsvm_fold(
        text.iloc[fit_indices], labels.iloc[fit_indices], text.iloc[valid_indices], text.iloc[valid_indices],
        word_features={args.word_features}, char_features={args.char_features},
        c_value={args.c_value}, seed={args.seed} + fold,
    )
    oof[valid_indices] = prediction
    assignment[valid_indices] = fold
    fold_records.append({{
        "fold": fold,
        "train_rows": len(fit_indices),
        "valid_rows": len(valid_indices),
        "word_features": int(artifacts["word_features"]),
        "char_features": int(artifacts["char_features"]),
        "auc": float(roc_auc_score(labels.iloc[valid_indices], prediction)),
    }})
if not np.isfinite(oof).all() or np.any(assignment < 0):
    raise RuntimeError("Pizza OOF coverage is incomplete")
print("EVOMIND_RESULT=" + json.dumps({{
    "rows": len(train), "positive_rows": int(labels.sum()), "folds": {args.folds}, "seed": {args.seed},
    "word_features": {args.word_features}, "char_features": {args.char_features}, "c_value": {args.c_value},
    "oof_auc": float(roc_auc_score(labels, oof)), "fold_records": fold_records,
    "fold_assignment_complete": True,
}}, sort_keys=True))
'''


def _run_remote(client: Any, source: str, timeout: int) -> tuple[int, str, str]:
    site = remote_ops.ensure_remote_path(remote_ops.REMOTE_UNIFIED_SITE_PACKAGES)
    command = f"PYTHONPATH={site} nice -n 10 python3 -"
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
        raise RuntimeError("Pizza validation did not emit exactly one result marker")
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
        raise RuntimeError(f"Pizza validation failed: {error[-1000:] or output[-1000:]}")
    metrics = _parse(output)
    threshold = 0.6921
    payload = {
        "schema": "evomind.mlebench.pizza_nbsvm_public_oof_validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "competition_id": COMPETITION_ID,
        "dataset": REMOTE_PUBLIC_DIR,
        "access_mode": "remote_read_only_cpu_probe",
        "path_scope_verified": True,
        "remote_stderr_type": "nonempty" if error.strip() else "",
        "production_runner_sha256": hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest(),
        "remote_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "bronze_threshold": threshold,
        **metrics,
        "bronze_oriented_gate_passed": float(metrics["oof_auc"]) >= threshold,
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--word-features", type=int, default=90_000)
    parser.add_argument("--char-features", type=int, default=110_000)
    parser.add_argument("--c-value", type=float, default=4.0)
    parser.add_argument("--timeout", type=int, default=900)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "oof_auc": payload["oof_auc"],
        "bronze_threshold": payload["bronze_threshold"],
        "bronze_oriented_gate_passed": payload["bronze_oriented_gate_passed"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
