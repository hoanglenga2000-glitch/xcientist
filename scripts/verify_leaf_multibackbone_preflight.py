#!/usr/bin/env python3
"""CPU-only Leaf dual-backbone weight and Windows-worker preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime
from importlib import import_module
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

leaf = import_module("scripts.run_leaf_multibackbone_oof")
wave2 = import_module("scripts.mlebench_wave2_adapters")
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "leaf_multibackbone_preflight_current.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_worker_spawn() -> dict:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader

    with tempfile.TemporaryDirectory(prefix="evomind_leaf_preflight_") as temporary:
        root = Path(temporary)
        paths: list[Path] = []
        for index in range(4):
            values = np.full((24 + index, 18 + index), 255, dtype=np.uint8)
            values[4:-4, 3:-3] = 40 + index * 20
            path = root / f"{index}.jpg"
            Image.fromarray(values, mode="L").save(path)
            paths.append(path)
        dataset = leaf.LeafEmbeddingDataset(paths, image_size=64)
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2,
            persistent_workers=False,
            worker_init_fn=wave2._seed_vision_worker,
            generator=torch.Generator().manual_seed(42),
        )
        batches = list(loader)
    indices = torch.cat([batch[1] for batch in batches]).tolist()
    tensors = torch.cat([batch[0] for batch in batches])
    return {
        "num_workers": 2,
        "batches": len(batches),
        "rows": len(indices),
        "indices": indices,
        "tensor_shape": list(tensors.shape),
        "finite": bool(torch.isfinite(tensors).all()),
        "module_level_dataset": leaf.LeafEmbeddingDataset.__qualname__
        == "LeafEmbeddingDataset",
        "passed": indices == [0, 1, 2, 3] and bool(torch.isfinite(tensors).all()),
    }


def verify_backbones(torch_home: Path) -> dict:
    import torch

    os.environ["TORCH_HOME"] = str(torch_home.resolve())
    results: dict[str, dict] = {}
    for backbone in leaf.DEFAULT_BACKBONES:
        spec = wave2.VISION_BACKBONE_SPECS[backbone]
        weight_path = torch_home / "hub" / "checkpoints" / spec["filename"]
        actual_sha256 = sha256_file(weight_path)
        if actual_sha256 != spec["sha256"]:
            raise RuntimeError(f"Leaf preflight weight mismatch: {backbone}")
        model, pretrained, identity = wave2._vision_model(
            1,
            backbone=backbone,
            require_pretrained=True,
        )
        width = leaf._replace_classifier_with_identity(model, backbone)
        model.eval()
        with torch.inference_mode():
            output = model(torch.zeros((1, 3, 64, 64), dtype=torch.float32))
        results[backbone] = {
            "pretrained": pretrained,
            "weight_path": str(weight_path.resolve()),
            "weight_sha256": actual_sha256,
            "expected_sha256": spec["sha256"],
            "weight_verified": actual_sha256 == spec["sha256"],
            "embedding_width": width,
            "output_shape": list(output.shape),
            "output_finite": bool(torch.isfinite(output).all()),
            "all_parameters_cpu": all(
                parameter.device.type == "cpu" for parameter in model.parameters()
            ),
            "identity": identity,
        }
        del model
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=PROJECT_ROOT / "mlebench_model_cache" / "torch",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    worker = verify_worker_spawn()
    backbones = verify_backbones(args.torch_home)
    passed = worker["passed"] and all(
        value["weight_verified"]
        and value["output_finite"]
        and value["all_parameters_cpu"]
        for value in backbones.values()
    )
    report = {
        "schema": "evomind.leaf.multibackbone_preflight.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "passed" if passed else "failed",
        "competition_id": leaf.COMPETITION_ID,
        "device": "cpu_preflight_only",
        "requested_model": "gpt-5.6-sol",
        "served_model": "gpt-5.6-sol",
        "worker_spawn": worker,
        "backbones": backbones,
        "runner_source": {
            "path": str(Path(leaf.__file__).resolve()),
            "sha256": sha256_file(Path(leaf.__file__).resolve()),
        },
        "checks": {
            "windows_spawn_picklable_dataset": worker["passed"],
            "two_distinct_backbones": len(backbones) == 2,
            "both_weight_hashes_verified": all(
                value["weight_verified"] for value in backbones.values()
            ),
            "both_forward_outputs_finite": all(
                value["output_finite"] for value in backbones.values()
            ),
            "private_labels_unused": True,
            "official_grader_not_executed": True,
            "kaggle_submission_not_executed": True,
        },
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "claim_boundary": "CPU-only implementation preflight; not a Leaf OOF score or medal.",
    }
    write_json_atomic(args.report.resolve(), report)
    print(json.dumps({"report": str(args.report.resolve()), "status": report["status"]}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
