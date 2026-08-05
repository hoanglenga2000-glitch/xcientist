#!/usr/bin/env python3
"""Validate the Spooky Author medal-recovery model on HPC without remote writes."""

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
from scripts import mlebench_wave2_adapters as wave2


COMPETITION_ID = "spooky-author-identification"
REMOTE_PUBLIC_DIR = (
    f"{remote_ops.REMOTE_DATA_ROOT}/{COMPETITION_ID}/prepared/public"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "spooky_nbsvm_public_oof_validation_current.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _remote_model_source() -> str:
    """Extract the production helpers so validation cannot drift from the runner."""

    chunks = [
        inspect.getsource(wave2.compute_nb_log_count_ratio).strip(),
        f"_SPOOKY_FUNCTION_WORDS = {recovery._SPOOKY_FUNCTION_WORDS!r}",
        f"_SPOOKY_PUNCTUATION = {recovery._SPOOKY_PUNCTUATION!r}",
        inspect.getsource(recovery._stable_softmax).strip(),
        inspect.getsource(recovery.build_spooky_stylometric_features).strip(),
        inspect.getsource(recovery.fit_spooky_stylometric_channel).strip(),
        inspect.getsource(recovery.fit_spooky_nbsvm_channel)
        .strip()
        .replace("wave2.compute_nb_log_count_ratio", "compute_nb_log_count_ratio"),
        inspect.getsource(recovery.normalize_spooky_text).strip(),
        inspect.getsource(recovery.build_spooky_duplicate_groups).strip(),
        inspect.getsource(recovery.apply_spooky_multicomponent_blend).strip(),
        inspect.getsource(recovery._spooky_simplex_weights).strip(),
        inspect.getsource(recovery.select_spooky_multicomponent_blend).strip(),
        inspect.getsource(recovery.cross_fit_spooky_multicomponent_blend).strip(),
    ]
    return "\n\n".join(chunks)


def _remote_evaluator_source(
    *,
    folds: int,
    seed: int,
    word_features: int,
    char_features: int,
    raw_char_features: int,
    c_values: list[float],
    style_c: float,
) -> str:
    model_source = _remote_model_source()
    return f'''from __future__ import annotations

import json
import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

{model_source}

PUBLIC_DIR = Path({REMOTE_PUBLIC_DIR!r})
FOLDS = {folds}
SEED = {seed}
WORD_FEATURES = {word_features}
CHAR_FEATURES = {char_features}
RAW_CHAR_FEATURES = {raw_char_features}
C_VALUES = {c_values!r}
STYLE_C = {style_c!r}

train = pd.read_csv(PUBLIC_DIR / "train.csv").reset_index(drop=True)
sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv")
required = {{"id", "text", "author"}}
if not required <= set(train):
    raise RuntimeError("Spooky Author training schema is incomplete")
classes = [column for column in sample.columns if column != "id"]
if set(classes) != set(train["author"].astype(str).unique()):
    raise RuntimeError("Spooky Author class order does not match the sample schema")
labels = train["author"].astype(str).to_numpy()
text = train["text"].fillna("").astype(str)
duplicate_groups, duplicate_report = build_spooky_duplicate_groups(text, labels)
style = build_spooky_stylometric_features(text)
splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
fold_assignment = np.full(len(train), -1, dtype=np.int16)
word_oof = {{value: np.zeros((len(train), len(classes)), dtype=np.float64) for value in C_VALUES}}
char_oof = {{value: np.zeros((len(train), len(classes)), dtype=np.float64) for value in C_VALUES}}
raw_char_oof = {{value: np.zeros((len(train), len(classes)), dtype=np.float64) for value in C_VALUES}}
style_oof = np.zeros((len(train), len(classes)), dtype=np.float64)
fold_records = []

for fold, (fit_indices, valid_indices) in enumerate(
    splitter.split(text, labels, duplicate_groups)
):
    group_overlap = set(duplicate_groups[fit_indices]) & set(duplicate_groups[valid_indices])
    if group_overlap:
        raise RuntimeError("Spooky duplicate group crossed a fold boundary")
    fold_assignment[valid_indices] = fold
    word = TfidfVectorizer(
        strip_accents="unicode",
        sublinear_tf=True,
        ngram_range=(1, 3),
        min_df=2,
        max_features=WORD_FEATURES,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        sublinear_tf=True,
        ngram_range=(3, 6),
        min_df=2,
        max_features=CHAR_FEATURES,
    )
    raw_char = TfidfVectorizer(
        analyzer="char",
        sublinear_tf=True,
        ngram_range=(2, 5),
        min_df=2,
        max_features=RAW_CHAR_FEATURES,
    )
    fit_word = word.fit_transform(text.iloc[fit_indices])
    valid_word = word.transform(text.iloc[valid_indices])
    fit_char = char.fit_transform(text.iloc[fit_indices])
    valid_char = char.transform(text.iloc[valid_indices])
    fit_raw_char = raw_char.fit_transform(text.iloc[fit_indices])
    valid_raw_char = raw_char.transform(text.iloc[valid_indices])
    valid_style_probability, _, _ = fit_spooky_stylometric_channel(
        style.iloc[fit_indices],
        labels[fit_indices],
        style.iloc[valid_indices],
        style.iloc[valid_indices],
        classes,
        c_value=STYLE_C,
        seed=SEED + fold * 10 + 3000,
    )
    style_oof[valid_indices] = valid_style_probability
    record = {{
        "fold": fold,
        "train_rows": len(fit_indices),
        "valid_rows": len(valid_indices),
        "word_features": int(fit_word.shape[1]),
        "char_features": int(fit_char.shape[1]),
        "raw_char_features": int(fit_raw_char.shape[1]),
        "style_features": int(style.shape[1]),
        "style_log_loss": float(log_loss(
            labels[valid_indices], valid_style_probability, labels=classes
        )),
        "scores": {{}},
    }}
    for c_value in C_VALUES:
        valid_word_probability, _, _ = fit_spooky_nbsvm_channel(
            fit_word,
            labels[fit_indices],
            valid_word,
            valid_word,
            classes,
            c_value=c_value,
            seed=SEED + fold * 10,
        )
        valid_char_probability, _, _ = fit_spooky_nbsvm_channel(
            fit_char,
            labels[fit_indices],
            valid_char,
            valid_char,
            classes,
            c_value=c_value,
            seed=SEED + fold * 10 + 1000,
        )
        valid_raw_char_probability, _, _ = fit_spooky_nbsvm_channel(
            fit_raw_char,
            labels[fit_indices],
            valid_raw_char,
            valid_raw_char,
            classes,
            c_value=c_value,
            seed=SEED + fold * 10 + 2000,
        )
        word_oof[c_value][valid_indices] = valid_word_probability
        char_oof[c_value][valid_indices] = valid_char_probability
        raw_char_oof[c_value][valid_indices] = valid_raw_char_probability
        record["scores"][str(c_value)] = {{
            "word": float(log_loss(labels[valid_indices], valid_word_probability, labels=classes)),
            "char": float(log_loss(labels[valid_indices], valid_char_probability, labels=classes)),
            "raw_char": float(log_loss(
                labels[valid_indices], valid_raw_char_probability, labels=classes
            )),
            "equal_sparse_blend": float(log_loss(
                labels[valid_indices],
                (
                    valid_word_probability
                    + valid_char_probability
                    + valid_raw_char_probability
                ) / 3.0,
                labels=classes,
            )),
        }}
    fold_records.append(record)

if np.any(fold_assignment < 0):
    raise RuntimeError("Spooky Author OOF coverage is incomplete")

candidates = []
for c_value in C_VALUES:
    components = [word_oof[c_value], char_oof[c_value], raw_char_oof[c_value], style_oof]
    crossfit, blend_records = cross_fit_spooky_multicomponent_blend(
        components, labels, classes, fold_assignment
    )
    candidates.append({{
        "c_value": c_value,
        "word_log_loss": float(log_loss(labels, word_oof[c_value], labels=classes)),
        "char_log_loss": float(log_loss(labels, char_oof[c_value], labels=classes)),
        "raw_char_log_loss": float(log_loss(labels, raw_char_oof[c_value], labels=classes)),
        "style_log_loss": float(log_loss(labels, style_oof, labels=classes)),
        "equal_sparse_blend_log_loss": float(log_loss(
            labels,
            (word_oof[c_value] + char_oof[c_value] + raw_char_oof[c_value]) / 3.0,
            labels=classes,
        )),
        "cross_fitted_multicomponent_log_loss": float(log_loss(labels, crossfit, labels=classes)),
        "crossfit_blend": blend_records,
    }})

best = min(candidates, key=lambda item: (item["cross_fitted_multicomponent_log_loss"], item["c_value"]))
payload = {{
    "rows": len(train),
    "classes": classes,
    "folds": FOLDS,
    "seed": SEED,
    "fold_assignment_complete": bool(np.all(fold_assignment >= 0)),
    "duplicate_groups": duplicate_report,
    "duplicate_group_fold_isolation": bool(all(
        len(np.unique(fold_assignment[duplicate_groups == group])) == 1
        for group in np.unique(duplicate_groups)
    )),
    "fold_records": fold_records,
    "candidates": candidates,
    "best": best,
}}
print("EVOMIND_RESULT=" + json.dumps(payload, ensure_ascii=False, sort_keys=True))
'''


def _run_remote_source(client: Any, source: str, *, timeout: int) -> tuple[int, str, str]:
    site = remote_ops.ensure_remote_path(remote_ops.REMOTE_UNIFIED_SITE_PACKAGES)
    stdin, stdout, stderr = client.exec_command(
        (
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            "NUMEXPR_NUM_THREADS=1 "
            f"PYTHONPATH={site} nice -n 15 python3 -"
        ),
        timeout=timeout,
    )
    stdout.channel.settimeout(timeout)
    stderr.channel.settimeout(timeout)
    stdin.write(source)
    stdin.flush()
    stdin.channel.shutdown_write()
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), output, error


def _parse_result(output: str) -> dict[str, Any]:
    markers = [line for line in output.splitlines() if line.startswith("EVOMIND_RESULT=")]
    if len(markers) != 1:
        raise RuntimeError("Remote Spooky validation did not emit exactly one result marker")
    result = json.loads(markers[0].split("=", 1)[1])
    if not isinstance(result, dict):
        raise RuntimeError("Remote Spooky validation result is not an object")
    return result


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    remote_ops.ensure_remote_path(REMOTE_PUBLIC_DIR)
    c_values = sorted(set(float(value) for value in args.c_values.split(",") if value.strip()))
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("At least one positive C value is required")
    source = _remote_evaluator_source(
        folds=args.folds,
        seed=args.seed,
        word_features=args.word_features,
        char_features=args.char_features,
        raw_char_features=args.raw_char_features,
        c_values=c_values,
        style_c=args.style_c,
    )
    client = remote_ops._connect()
    try:
        remote_ops.verify_remote_runtime(client)
        code, output, error = _run_remote_source(client, source, timeout=args.timeout)
    finally:
        client.close()
    if code != 0:
        raise RuntimeError(f"Remote Spooky validation failed with exit code {code}")
    metrics = _parse_result(output)
    bronze_threshold = 0.29381
    payload = {
        "schema": "evomind.mlebench.spooky_multicomponent_public_oof_validation.v3",
        "created_at": utc_now(),
        "competition_id": COMPETITION_ID,
        "dataset": REMOTE_PUBLIC_DIR,
        "remote_python": "python3 + unified-py310-sklearn1.7.2 overlay",
        "access_mode": "remote_read_only_cpu_probe",
        "path_scope_verified": True,
        "remote_stderr_type": "nonempty" if error.strip() else "",
        "production_adapter_sha256": sha256_file(Path(recovery.__file__)),
        "remote_source_sha256": sha256_bytes(source.encode("utf-8")),
        "bronze_threshold": bronze_threshold,
        "internal_target": 0.28,
        **metrics,
        "bronze_oriented_gate_passed": (
            float(metrics["best"]["cross_fitted_multicomponent_log_loss"]) <= bronze_threshold
        ),
        "internal_target_passed": (
            float(metrics["best"]["cross_fitted_multicomponent_log_loss"]) <= 0.28
        ),
        "stop_rule_triggered": (
            float(metrics["best"]["cross_fitted_multicomponent_log_loss"]) > 0.32
        ),
    }
    _write_atomic(args.output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--word-features", type=int, default=120_000)
    parser.add_argument("--char-features", type=int, default=180_000)
    parser.add_argument("--raw-char-features", type=int, default=160_000)
    parser.add_argument("--c-values", default="4")
    parser.add_argument("--style-c", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=1200)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_validation(args)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "best": payload["best"],
        "bronze_threshold": payload["bronze_threshold"],
        "bronze_oriented_gate_passed": payload["bronze_oriented_gate_passed"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
