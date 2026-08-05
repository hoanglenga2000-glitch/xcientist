#!/usr/bin/env python3
"""Run a read-only, geometry-aware NOMAD public OOF validation on HPC CPU."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
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


COMPETITION_ID = "nomad2018-predict-transparent-conductors"
REMOTE_PUBLIC_DIR = f"{remote_ops.REMOTE_DATA_ROOT}/{COMPETITION_ID}/prepared/public"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "nomad_geometry_public_oof_validation_current.json"
)


def _production_feature_source() -> str:
    constants = "\n".join((
        f"_NOMAD_ELEMENTS = {wave2._NOMAD_ELEMENTS!r}",
        "_NOMAD_ELEMENT_PAIRS = tuple(itertools.combinations_with_replacement(_NOMAD_ELEMENTS, 2))",
        f"_NOMAD_PROPERTIES = {wave2._NOMAD_PROPERTIES!r}",
    ))
    functions = (
        wave2.parse_nomad_geometry,
        wave2._nomad_stats,
        wave2._minimum_image_distances,
        wave2.nomad_structure_features,
        wave2._nomad_id,
        wave2._nomad_features,
    )
    return constants + "\n\n" + "\n\n".join(
        inspect.getsource(function).strip() for function in functions
    )


def _remote_source(args: argparse.Namespace) -> str:
    feature_source = _production_feature_source()
    return f'''from __future__ import annotations
import itertools, json, re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

{feature_source}

PUBLIC_DIR = Path({REMOTE_PUBLIC_DIR!r})
train = pd.read_csv(PUBLIC_DIR / "train.csv").reset_index(drop=True)
targets = ["formation_energy_ev_natom", "bandgap_energy_ev"]
if not {{"id", *targets}} <= set(train):
    raise RuntimeError("NOMAD training schema is incomplete")
if train["id"].duplicated().any():
    raise RuntimeError("NOMAD training IDs are duplicated")
x = _nomad_features(train, PUBLIC_DIR / "train")
if not np.isfinite(x.to_numpy()).all():
    raise RuntimeError("NOMAD production features contain non-finite values")
splitter = KFold(n_splits={args.folds}, shuffle=True, random_state={args.seed})
splits = list(splitter.split(x))
fold_assignment = np.full(len(train), -1, dtype=np.int16)
prediction = np.zeros((len(train), 2), dtype=np.float64)
target_records = {{}}
for target_index, target in enumerate(targets):
    y_log = np.log1p(train[target].clip(lower=0).to_numpy())
    fold_scores = []
    for fold, (fit_indices, valid_indices) in enumerate(splits):
        fold_assignment[valid_indices] = fold
        model = ExtraTreesRegressor(
            n_estimators={args.trees}, max_features={args.max_features},
            min_samples_leaf={args.min_samples_leaf},
            random_state={args.seed} + target_index * 101 + fold,
            n_jobs={args.jobs},
        )
        model.fit(x.iloc[fit_indices], y_log[fit_indices])
        valid_log = model.predict(x.iloc[valid_indices])
        prediction[valid_indices, target_index] = np.clip(np.expm1(valid_log), 0.0, None)
        fold_scores.append(float(np.sqrt(np.mean((y_log[valid_indices] - valid_log) ** 2))))
    target_records[target] = {{
        "fold_rmsle": fold_scores,
        "oof_rmsle": float(np.sqrt(np.mean((y_log - np.log1p(prediction[:, target_index])) ** 2))),
    }}
if np.any(fold_assignment < 0) or not np.isfinite(prediction).all():
    raise RuntimeError("NOMAD OOF coverage is incomplete")
target_scores = [target_records[target]["oof_rmsle"] for target in targets]
print("EVOMIND_RESULT=" + json.dumps({{
    "rows": len(train), "feature_count": len(x.columns),
    "geometry_feature_count": int(sum(column.startswith("geom_") for column in x.columns)),
    "folds": {args.folds}, "seed": {args.seed}, "trees": {args.trees},
    "max_features": {args.max_features}, "min_samples_leaf": {args.min_samples_leaf},
    "fold_assignment_complete": True, "targets": target_records,
    "mean_columnwise_rmsle": float(np.mean(target_scores)),
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
        raise RuntimeError("NOMAD validation did not emit exactly one result marker")
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
        raise RuntimeError(f"NOMAD validation failed: {error[-1200:] or output[-1200:]}")
    metrics = _parse(output)
    bronze_threshold = 0.06582
    target_threshold = 0.0625
    score = float(metrics["mean_columnwise_rmsle"])
    payload = {
        "schema": "evomind.mlebench.nomad_geometry_public_oof_validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "competition_id": COMPETITION_ID,
        "dataset": REMOTE_PUBLIC_DIR,
        "access_mode": "remote_read_only_cpu_probe",
        "path_scope_verified": True,
        "remote_stderr_type": "nonempty" if error.strip() else "",
        "production_adapter_sha256": hashlib.sha256(Path(wave2.__file__).read_bytes()).hexdigest(),
        "remote_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "bronze_threshold": bronze_threshold,
        "target_threshold": target_threshold,
        **metrics,
        "bronze_oriented_gate_passed": score <= bronze_threshold,
        "target_passed": score <= target_threshold,
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
    parser.add_argument("--trees", type=int, default=800)
    parser.add_argument("--max-features", type=float, default=0.85)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1200)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "mean_columnwise_rmsle": payload["mean_columnwise_rmsle"],
        "bronze_threshold": payload["bronze_threshold"],
        "target_threshold": payload["target_threshold"],
        "bronze_oriented_gate_passed": payload["bronze_oriented_gate_passed"],
        "target_passed": payload["target_passed"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
