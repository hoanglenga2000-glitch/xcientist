#!/usr/bin/env python3
"""Verify the frozen SIIM multiview stack without allocating CUDA memory."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

try:
    import mlebench_medal_recovery_adapters as recovery
    import mlebench_wave2_adapters as wave2
except ModuleNotFoundError:
    from scripts import mlebench_medal_recovery_adapters as recovery
    from scripts import mlebench_wave2_adapters as wave2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TORCH_HOME = PROJECT_ROOT / "mlebench_model_cache" / "torch"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "siim_multiview_preflight_current.json"
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_image(image: Image.Image) -> str:
    payload = io.BytesIO()
    image.save(payload, format="PNG", optimize=False)
    return hashlib.sha256(payload.getvalue()).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_synthetic_dermoscopy_fixture(size: int = 96) -> Image.Image:
    yy, xx = np.mgrid[:size, :size]
    radial = np.sqrt((xx - size / 2.0) ** 2 + (yy - size / 2.0) ** 2)
    values = np.empty((size, size, 3), dtype=np.float32)
    gentle_gradient = (xx / max(1, size - 1) - 0.5) * 4.0
    values[..., 0] = 194.0 + gentle_gradient
    values[..., 1] = 145.0 + gentle_gradient
    values[..., 2] = 126.0 + gentle_gradient
    lesion = radial <= size * 0.20
    values[lesion, 0] *= 0.45
    values[lesion, 1] *= 0.35
    values[lesion, 2] *= 0.40
    values[:5] = 0.0
    values[-5:] = 0.0
    values[:, :5] = 0.0
    values[:, -5:] = 0.0
    values[size // 3 : size // 3 + 2, size // 5 : size * 4 // 5] = 8.0
    return Image.fromarray(np.clip(values, 0.0, 255.0).astype(np.uint8), mode="RGB")


def verify_weight_file(torch_home: Path, backbone: str) -> dict[str, Any]:
    spec = wave2.VISION_BACKBONE_SPECS[backbone]
    path = torch_home / "hub" / "checkpoints" / spec["filename"]
    if not path.is_file():
        raise FileNotFoundError(f"Pinned SIIM weight is missing: {path}")
    observed = sha256_file(path)
    if observed != spec["sha256"]:
        raise RuntimeError(f"Pinned SIIM weight SHA256 mismatch: {backbone}")
    return {
        "backbone": backbone,
        "path": str(path.resolve()),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": observed,
        "expected_sha256": spec["sha256"],
        "verified": True,
    }


def run_preflight(
    *,
    torch_home: Path,
    full_backbone: str,
    lesion_backbone: str,
    metadata_width: int,
    image_size: int,
) -> dict[str, Any]:
    if full_backbone == lesion_backbone:
        raise ValueError("SIIM multiview preflight requires distinct backbones")
    if metadata_width < 1 or image_size < 32:
        raise ValueError("SIIM preflight dimensions are invalid")
    full_weight = verify_weight_file(torch_home, full_backbone)
    lesion_weight = verify_weight_file(torch_home, lesion_backbone)

    import torch

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.hub.set_dir(str((torch_home / "hub").resolve()))
    model, pretrained, identity = recovery.build_siim_multiview_fusion_model(
        metadata_width,
        full_backbone_name=full_backbone,
        lesion_backbone_name=lesion_backbone,
    )
    if not pretrained:
        raise RuntimeError("SIIM multiview model did not load both pinned weights")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("SIIM preflight model unexpectedly left CPU")
    model.eval()
    with torch.inference_mode():
        channels = model.forward_channels(
            torch.zeros(1, 3, image_size, image_size),
            torch.zeros(1, 3, image_size, image_size),
            torch.zeros(1, metadata_width),
        )
    expected_channels = {"full_image", "lesion_focus", "image_metadata_fusion"}
    if set(channels) != expected_channels:
        raise RuntimeError("SIIM multiview output-channel contract changed")
    channel_checks = {
        name: {
            "shape": list(value.shape),
            "finite": bool(torch.isfinite(value).all()),
            "device": value.device.type,
        }
        for name, value in channels.items()
    }
    if any(item["shape"] != [1] or not item["finite"] for item in channel_checks.values()):
        raise RuntimeError("SIIM multiview CPU forward failed")

    fixture = build_synthetic_dermoscopy_fixture()
    first = recovery.prepare_siim_dermoscopy_views(
        fixture,
        profile="robust_multiview_v1",
    )
    second = recovery.prepare_siim_dermoscopy_views(
        fixture,
        profile="robust_multiview_v1",
    )
    preprocessing = {
        name: {
            "size": list(first[name].size),
            "sha256": sha256_image(first[name]),
            "deterministic": sha256_image(first[name]) == sha256_image(second[name]),
        }
        for name in sorted(first)
    }
    views_are_distinct = (
        preprocessing["full_image"]["size"] != preprocessing["lesion_focus"]["size"]
        or preprocessing["full_image"]["sha256"]
        != preprocessing["lesion_focus"]["sha256"]
    )
    if (
        set(preprocessing) != {"full_image", "lesion_focus"}
        or not all(item["deterministic"] for item in preprocessing.values())
        or not views_are_distinct
    ):
        raise RuntimeError("SIIM robust preprocessing multiview contract failed")

    source_paths = {
        "medal_recovery_adapters": Path(recovery.__file__).resolve(),
        "wave2_adapters": Path(wave2.__file__).resolve(),
        "preflight": Path(__file__).resolve(),
    }
    identity_full = identity["full_image"]["identity"]
    identity_lesion = identity["lesion_focus"]["identity"]
    checks = {
        "distinct_backbones": full_backbone != lesion_backbone,
        "both_pretrained": bool(pretrained),
        "full_weight_identity": identity_full["sha256"] == full_weight["sha256"],
        "lesion_weight_identity": identity_lesion["sha256"] == lesion_weight["sha256"],
        "all_parameters_cpu": all(
            parameter.device.type == "cpu" for parameter in model.parameters()
        ),
        "three_finite_channels": all(item["finite"] for item in channel_checks.values()),
        "robust_preprocessing_deterministic": all(
            item["deterministic"] for item in preprocessing.values()
        ),
        "full_and_lesion_views_distinct": views_are_distinct,
        "private_labels_unused": True,
        "official_grader_not_executed": True,
        "kaggle_submission_not_executed": True,
    }
    return {
        "schema": "evomind.siim.multiview_preflight.v1",
        "created_at": now_iso(),
        "status": "passed" if all(checks.values()) else "failed",
        "competition_id": "siim-isic-melanoma-classification",
        "device": "cpu_preflight_only",
        "torch_home": str(torch_home.resolve()),
        "backbones": {
            "full_image": full_weight,
            "lesion_focus": lesion_weight,
        },
        "model_weight_identity": identity,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "forward_channels": channel_checks,
        "robust_preprocessing_fixture": preprocessing,
        "profile_steps": recovery.SIIM_PREPROCESSING_PROFILE_STEPS[
            "robust_multiview_v1"
        ],
        "source_hashes": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "checks": checks,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "claim_boundary": (
            "CPU-only architecture, pinned-weight, and preprocessing preflight; "
            "not an OOF score or official medal."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--full-backbone", default="convnext_small")
    parser.add_argument("--lesion-backbone", default="efficientnet_v2_s")
    parser.add_argument("--metadata-width", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_preflight(
        torch_home=args.torch_home.resolve(),
        full_backbone=args.full_backbone,
        lesion_backbone=args.lesion_backbone,
        metadata_width=args.metadata_width,
        image_size=args.image_size,
    )
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
