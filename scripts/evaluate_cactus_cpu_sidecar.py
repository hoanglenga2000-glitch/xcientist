#!/usr/bin/env python3
"""Evaluate and optionally materialize a CPU sidecar blend for Cactus."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(gpu_bundle: Path, cpu_bundle: Path, weights: list[float]) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(gpu_bundle, allow_pickle=False) as gpu, np.load(cpu_bundle, allow_pickle=False) as cpu:
        truth = np.asarray(gpu["truth"], dtype=np.int8).reshape(-1)
        gpu_oof = np.asarray(gpu["selected_oof_probability"], dtype=np.float64).reshape(-1)
        gpu_test = np.asarray(gpu["selected_test_probability"], dtype=np.float64).reshape(-1)
        cpu_truth = np.asarray(cpu["truth"], dtype=np.int8).reshape(-1)
        cpu_oof = np.asarray(cpu["xgb_oof"], dtype=np.float64).reshape(-1)
        cpu_test = np.asarray(cpu["candidate_test"], dtype=np.float64).reshape(-1)
        fold = np.asarray(cpu["fold"], dtype=np.int16).reshape(-1)
    if not np.array_equal(truth, cpu_truth):
        raise RuntimeError("Cactus GPU/CPU truth ordering differs")
    if len(gpu_oof) != len(cpu_oof) or len(gpu_test) != len(cpu_test):
        raise RuntimeError("Cactus GPU/CPU prediction cardinality differs")
    if sorted(set(fold.tolist())) != list(range(int(fold.max()) + 1)):
        raise RuntimeError("Cactus CPU fold assignment is incomplete")
    baseline = float(roc_auc_score(truth, gpu_oof))
    full_records = []
    for weight in weights:
        blend = (1.0 - weight) * gpu_oof + weight * cpu_oof
        full_records.append({"cpu_weight": weight, "auc": float(roc_auc_score(truth, blend))})
    best = max(full_records, key=lambda item: (item["auc"], -item["cpu_weight"]))
    nested = np.zeros_like(gpu_oof)
    selected = []
    for current in sorted(set(fold.tolist())):
        fit = fold != current
        valid = fold == current
        fit_scores = []
        for weight in weights:
            fit_prediction = (1.0 - weight) * gpu_oof[fit] + weight * cpu_oof[fit]
            fit_scores.append(float(roc_auc_score(truth[fit], fit_prediction)))
        selected_index = int(np.argmax(fit_scores))
        selected_weight = weights[selected_index]
        nested[valid] = (1.0 - selected_weight) * gpu_oof[valid] + selected_weight * cpu_oof[valid]
        selected.append(
            {"fold": int(current), "cpu_weight": selected_weight, "fit_auc": fit_scores[selected_index]}
        )
    nested_auc = float(roc_auc_score(truth, nested))
    best_weight = float(best["cpu_weight"])
    arrays = {
        "truth": truth,
        "selected_oof_probability": (1.0 - best_weight) * gpu_oof + best_weight * cpu_oof,
        "selected_test_probability": (1.0 - best_weight) * gpu_test + best_weight * cpu_test,
        "cpu_weight": np.asarray([best_weight], dtype=np.float64),
    }
    report = {
        "schema": "evomind.cactus.cpu_sidecar_blend_evaluation.v1",
        "gpu_baseline_auc": baseline,
        "full_oof_candidates": full_records,
        "full_oof_best": best,
        "nested_auc": nested_auc,
        "nested_gain": nested_auc - baseline,
        "nested_selected_weights": selected,
        "full_best_gain": float(best["auc"]) - baseline,
    }
    return report, arrays


def parse_weights(value: str) -> list[float]:
    weights = sorted({float(item) for item in value.split(",")})
    if not weights or weights[0] < 0.0 or weights[-1] > 1.0 or 0.0 not in weights:
        raise argparse.ArgumentTypeError("weights must be unique values in [0,1] and include 0")
    return weights


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-bundle", type=Path, required=True)
    parser.add_argument("--cpu-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=parse_weights, default=parse_weights("0,0.025,0.05,0.1,0.2"))
    parser.add_argument("--promotion-auc", type=float, default=0.9997)
    args = parser.parse_args(argv)
    report, arrays = evaluate(args.gpu_bundle, args.cpu_bundle, args.weights)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bundle = args.output_dir / "cactus_cpu_sidecar_candidate.npz"
    np.savez_compressed(bundle, **arrays)
    report.update(
        {
            "status": (
                "candidate_ready"
                if report["nested_gain"] > 0.0
                and report["full_oof_best"]["auc"] >= args.promotion_auc
                else "sidecar_gate_failed"
            ),
            "candidate_ready": bool(
                report["nested_gain"] > 0.0
                and report["full_oof_best"]["auc"] >= args.promotion_auc
            ),
            "promotion_auc": args.promotion_auc,
            "gpu_bundle_sha256": sha256_file(args.gpu_bundle),
            "cpu_bundle_sha256": sha256_file(args.cpu_bundle),
            "output_bundle": {
                "path": str(bundle),
                "bytes": bundle.stat().st_size,
                "sha256": sha256_file(bundle),
            },
            "candidate_only": True,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "claim_boundary": "Public OOF sidecar evaluation only; not an official score or medal.",
        }
    )
    temporary = args.output_dir / ".report.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_dir / "sidecar_report.json")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
