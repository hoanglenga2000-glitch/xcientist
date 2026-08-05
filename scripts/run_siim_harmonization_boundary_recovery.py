#!/usr/bin/env python3
"""Run the immutable SIIM trainer with an evidence-bound endpoint hotfix.

The active job90353 bundle and its resume contract remain byte-for-byte
unchanged.  This wrapper imports that bundle, verifies its source hashes, and
monkeypatches only the post-fit fold-harmonization call so legitimate sigmoid
probabilities equal to 0 or 1 are epsilon-clipped.  Non-finite and out-of-range
values still fail closed.  Every call records input/output hashes and counts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ALLOWED_ROOT = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
EXPECTED_RUN_ID = "evomind_siim_isic_a800_job90353_20260730_095826"
CHANNEL_ORDER = ("full_image", "lesion_focus", "image_metadata_fusion", "metadata_catboost")
EPSILON = 1e-6


class HarmonizationBoundaryRecoveryError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(array.dtype), "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def safe_evidence_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise HarmonizationBoundaryRecoveryError("hotfix evidence path is not absolute and normalized")
    root = ALLOWED_ROOT.resolve()
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise HarmonizationBoundaryRecoveryError("hotfix evidence path escaped the dedicated root")
    if path.exists() and path.is_symlink():
        raise HarmonizationBoundaryRecoveryError("hotfix evidence path is a symbolic link")
    return resolved


def argument_value(name: str) -> str:
    try:
        index = sys.argv.index(name)
        value = sys.argv[index + 1]
    except (ValueError, IndexError) as exc:
        raise HarmonizationBoundaryRecoveryError(f"missing trainer argument: {name}") from exc
    if not value:
        raise HarmonizationBoundaryRecoveryError(f"empty trainer argument: {name}")
    return value


def sanitize_probabilities(
    oof_probability: np.ndarray,
    test_probability_by_fold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    oof = np.asarray(oof_probability, dtype=np.float64)
    test = np.asarray(test_probability_by_fold, dtype=np.float64)
    nonfinite = int((~np.isfinite(oof)).sum() + (~np.isfinite(test)).sum())
    below_zero = int((oof < 0.0).sum() + (test < 0.0).sum())
    above_one = int((oof > 1.0).sum() + (test > 1.0).sum())
    if nonfinite or below_zero or above_one:
        raise HarmonizationBoundaryRecoveryError(
            "SIIM hotfix input must remain finite and closed-unit"
        )
    zero_count = int((oof == 0.0).sum() + (test == 0.0).sum())
    one_count = int((oof == 1.0).sum() + (test == 1.0).sum())
    clipped_oof = np.clip(oof, EPSILON, 1.0 - EPSILON)
    clipped_test = np.clip(test, EPSILON, 1.0 - EPSILON)
    evidence = {
        "nonfinite_count": nonfinite,
        "below_zero_count": below_zero,
        "above_one_count": above_one,
        "exact_zero_count": zero_count,
        "exact_one_count": one_count,
        "clipped_value_count": zero_count + one_count,
        "epsilon": EPSILON,
        "oof_shape": list(oof.shape),
        "test_by_fold_shape": list(test.shape),
        "oof_input_sha256": sha256_array(oof),
        "test_input_sha256": sha256_array(test),
        "oof_output_sha256": sha256_array(clipped_oof),
        "test_output_sha256": sha256_array(clipped_test),
        "minimum_before": float(min(np.min(oof), np.min(test))),
        "maximum_before": float(max(np.max(oof), np.max(test))),
        "minimum_after": float(min(np.min(clipped_oof), np.min(clipped_test))),
        "maximum_after": float(max(np.max(clipped_oof), np.max(clipped_test))),
        "rank_policy": "monotonic_epsilon_clip_preserves_strict_order_and_endpoint_ties",
    }
    if not all(math.isfinite(float(evidence[name])) for name in (
        "minimum_before",
        "maximum_before",
        "minimum_after",
        "maximum_after",
    )):
        raise HarmonizationBoundaryRecoveryError("hotfix evidence contains non-finite bounds")
    return clipped_oof, clipped_test, evidence


def main() -> int:
    run_id = os.environ.get("EVOMIND_SIIM_RUN_ID", "")
    if run_id != EXPECTED_RUN_ID:
        raise HarmonizationBoundaryRecoveryError("harmonization recovery belongs to another Run")
    seed = int(argument_value("--seed"))
    if seed not in {43, 44, 45}:
        raise HarmonizationBoundaryRecoveryError("harmonization recovery seed is outside 43/44/45")
    evidence_path = safe_evidence_path(
        os.environ.get("EVOMIND_SIIM_HARMONIZATION_EVIDENCE_PATH", "")
    )

    import mlebench_medal_recovery_adapters as recovery

    adapter_path = Path(recovery.__file__).resolve()
    expected_adapter_hash = os.environ.get("EVOMIND_SIIM_EXPECTED_ADAPTER_SHA256", "")
    if len(expected_adapter_hash) != 64 or sha256_file(adapter_path) != expected_adapter_hash:
        raise HarmonizationBoundaryRecoveryError("immutable adapter source hash changed")
    wrapper_hash = sha256_file(Path(__file__).resolve())
    original = recovery.build_siim_fold_harmonization_variants
    calls: list[dict[str, Any]] = []

    def patched(
        oof_probability: np.ndarray,
        test_probability_by_fold: np.ndarray,
        folds: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        call_index = len(calls)
        if call_index >= len(CHANNEL_ORDER):
            raise HarmonizationBoundaryRecoveryError("unexpected fifth harmonization call")
        clipped_oof, clipped_test, evidence = sanitize_probabilities(
            oof_probability, test_probability_by_fold
        )
        result = original(clipped_oof, clipped_test, folds)
        calls.append(
            {
                "call_index": call_index + 1,
                "channel": CHANNEL_ORDER[call_index],
                **evidence,
                "oof_variant_hashes": {
                    name: sha256_array(values) for name, values in result[0].items()
                },
                "test_variant_hashes": {
                    name: sha256_array(values) for name, values in result[1].items()
                },
            }
        )
        atomic_json(
            evidence_path,
            {
                "schema": "evomind.siim.harmonization_boundary_recovery.v1",
                "updated_at": utc_now(),
                "run_id": run_id,
                "formal_seed": seed,
                "status": "running",
                "adapter_path": str(adapter_path),
                "adapter_sha256": expected_adapter_hash,
                "wrapper_sha256": wrapper_hash,
                "epsilon": EPSILON,
                "calls": calls,
                "official_submission_executed": False,
                "signals_sent": 0,
                "other_processes_modified": False,
            },
        )
        return result

    recovery.build_siim_fold_harmonization_variants = patched
    import run_mlebench_lite_full as full

    exit_code = int(full.main())
    status = "completed" if exit_code == 0 and len(calls) == 4 else "failed"
    atomic_json(
        evidence_path,
        {
            "schema": "evomind.siim.harmonization_boundary_recovery.v1",
            "updated_at": utc_now(),
            "run_id": run_id,
            "formal_seed": seed,
            "status": status,
            "exit_code": exit_code,
            "adapter_path": str(adapter_path),
            "adapter_sha256": expected_adapter_hash,
            "wrapper_sha256": wrapper_hash,
            "epsilon": EPSILON,
            "call_count": len(calls),
            "calls": calls,
            "official_submission_executed": False,
            "signals_sent": 0,
            "other_processes_modified": False,
        },
    )
    if exit_code == 0 and len(calls) != 4:
        raise HarmonizationBoundaryRecoveryError(
            "trainer completed without exactly four harmonization calls"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
