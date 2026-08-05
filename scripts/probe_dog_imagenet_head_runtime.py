#!/usr/bin/env python3
"""Verify the full 120-class Dog Breed ImageNet-head runtime contract on CPU."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "mlebench_wave2_adapters.py"
DEFAULT_SAMPLE = (
    ROOT
    / "workspace"
    / "local_gpu"
    / "mlebench_official_data"
    / "dog-breed-identification"
    / "prepared"
    / "public"
    / "sample_submission.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "workspace"
    / "mlebench_plans"
    / "dog_breed_imagenet_head_runtime_probe_current.json"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_adapter() -> Any:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("dog_probe_wave2_adapters", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the Wave 2 adapter module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_classes(sample_path: Path) -> list[str]:
    with sample_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    if not header or header[0] != "id":
        raise RuntimeError("Dog Breed sample submission must begin with the id column")
    classes = [str(value) for value in header[1:]]
    if len(classes) != 120 or len(set(classes)) != 120:
        raise RuntimeError("Dog Breed sample submission must contain 120 unique class columns")
    return classes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    import torch
    from torchvision.models import ConvNeXt_Small_Weights, convnext_small

    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    adapter = load_adapter()
    classes = read_classes(args.sample)
    weights = ConvNeXt_Small_Weights.DEFAULT
    categories = [str(value) for value in weights.meta["categories"]]
    indices, mapping = adapter.build_dog_breed_imagenet_mapping(classes, categories)
    if mapping.get("coverage_count") != 120 or mapping.get("coverage_fraction") != 1.0:
        raise RuntimeError("Production Dog Breed ImageNet mapping is incomplete")

    weight_path = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / adapter.CONVNEXT_SMALL_WEIGHT_FILENAME
    )
    if not weight_path.is_file():
        raise RuntimeError("Pinned ConvNeXt-Small weight file is absent from TORCH_HOME")
    weight_sha256 = sha256_bytes(weight_path.read_bytes())
    if weight_sha256 != adapter.CONVNEXT_SMALL_WEIGHT_SHA256:
        raise RuntimeError("Pinned ConvNeXt-Small weight SHA256 mismatch")

    with torch.inference_mode():
        source_model = convnext_small(weights=weights).cpu().eval()
        target_model, pretrained, identity = adapter._vision_model(
            120,
            backbone="convnext_small",
            require_pretrained=True,
            classifier_source_indices=indices.tolist(),
        )
        target_model = target_model.cpu().eval()
        classifier_index = int(
            adapter.VISION_BACKBONE_SPECS["convnext_small"]["classifier"].rsplit(".", 1)[-1]
        )
        source_linear = source_model.classifier[classifier_index]
        target_linear = target_model.classifier[classifier_index]
        index_tensor = torch.tensor(indices, dtype=torch.long)
        expected_weight = source_linear.weight.index_select(0, index_tensor)
        expected_bias = source_linear.bias.index_select(0, index_tensor)
        weights_exact = torch.equal(target_linear.weight, expected_weight)
        bias_exact = torch.equal(target_linear.bias, expected_bias)
        max_weight_delta = float((target_linear.weight - expected_weight).abs().max())
        max_bias_delta = float((target_linear.bias - expected_bias).abs().max())

        generator = torch.Generator(device="cpu").manual_seed(20260726)
        features = torch.randn(
            4,
            int(source_linear.in_features),
            1,
            1,
            generator=generator,
        )
        source_logits = source_model.classifier(features).index_select(1, index_tensor)
        target_logits = target_model.classifier(features)
        logit_max_abs_delta = float((target_logits - source_logits).abs().max())
        source_probability = torch.softmax(source_logits, dim=1)
        target_probability = torch.softmax(target_logits, dim=1)
        probability_max_abs_delta = float(
            (target_probability - source_probability).abs().max()
        )
        probability_row_sum_max_abs_delta = float(
            (target_probability.sum(dim=1) - 1.0).abs().max()
        )
        probability_rows_normalized = bool(
            torch.allclose(
                target_probability.sum(dim=1),
                torch.ones(len(target_probability)),
                atol=1e-6,
                rtol=0.0,
            )
        )

        optimizer_groups, optimizer_contract = (
            adapter.build_dog_breed_stability_optimizer_groups(
                target_model,
                base_learning_rate=2e-4,
                classifier_path="classifier.2",
            )
        )
        optimizer = torch.optim.AdamW(optimizer_groups, lr=2e-4)
        scheduler, scheduler_contract = adapter.build_dog_breed_stability_scheduler(
            optimizer,
            total_steps=100,
        )
        optimizer_initial_lrs = {
            str(group["group_name"]): float(group["lr"])
            for group in optimizer.param_groups
        }

    initialization = identity.get("classifier_initialization") or {}
    source_indices_exact = initialization.get("source_indices") == indices.tolist()
    contract_checks = {
        "production_class_count_120": len(classes) == 120,
        "mapping_coverage_120_of_120": mapping.get("coverage_count") == 120,
        "mapping_indices_unique": len(set(indices.tolist())) == 120,
        "pinned_weight_sha256_exact": weight_sha256
        == adapter.CONVNEXT_SMALL_WEIGHT_SHA256,
        "pretrained_loaded": pretrained is True,
        "runtime_source_indices_exact": source_indices_exact,
        "all_120_weight_rows_exact": weights_exact and max_weight_delta == 0.0,
        "all_120_bias_rows_exact": bias_exact and max_bias_delta == 0.0,
        "full_classifier_logits_exact": logit_max_abs_delta == 0.0,
        "full_classifier_probabilities_exact": probability_max_abs_delta == 0.0,
        "probability_rows_normalized": probability_rows_normalized,
        "optimizer_groups_disjoint": optimizer_contract["groups_disjoint"],
        "optimizer_groups_cover_all_trainable_parameters": optimizer_contract[
            "groups_cover_all_trainable_parameters"
        ],
        "only_final_linear_is_task_head": optimizer_contract["task_head_parameter_names"]
        == ["classifier.2.weight", "classifier.2.bias"],
        "classifier_norm_is_pretrained_no_decay": optimizer_contract[
            "classifier_norm_parameter_names"
        ]
        == ["classifier.0.weight", "classifier.0.bias"],
        "scheduler_warmup_10pct": scheduler_contract["warmup_fraction"] == 0.10,
        "scheduler_peak_factor_one": scheduler_contract["peak_factor"] == 1.0,
        "scheduler_final_factor_five_percent": scheduler_contract["final_update_factor"]
        == 0.05,
    }
    runtime_contract_passed = all(contract_checks.values())
    payload = {
        "schema": "evomind.dog_breed.imagenet_head_runtime_probe.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if runtime_contract_passed else "failed",
        "execution_device": "cpu",
        "active_training_gpu_touched": False,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "sample_submission": {
            "path": str(args.sample.resolve()),
            "sha256": sha256_bytes(args.sample.read_bytes()),
            "class_count": len(classes),
            "class_order_sha256": sha256_bytes(
                json.dumps(classes, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ),
        },
        "adapter": {
            "path": str(SOURCE.resolve()),
            "sha256": sha256_bytes(SOURCE.read_bytes()),
        },
        "pinned_weight_sha256": weight_sha256,
        "production_mapping": mapping,
        "runtime_classifier_initialization": initialization,
        "production_classifier_execution": {
            "output_count": 120,
            "source_output_count": int(source_linear.out_features),
            "source_indices": indices.tolist(),
            "unique_source_rows": len(set(indices.tolist())) == 120,
            "weights_copied_exactly": weights_exact,
            "bias_copied_exactly": bias_exact,
            "max_weight_abs_delta": max_weight_delta,
            "max_bias_abs_delta": max_bias_delta,
            "classifier_feature_batch_shape": list(features.shape),
            "full_classifier_logit_max_abs_delta": logit_max_abs_delta,
            "full_classifier_probability_max_abs_delta": probability_max_abs_delta,
            "probability_row_sum_max_abs_delta": probability_row_sum_max_abs_delta,
            "probability_row_sum_tolerance": 1e-6,
            "probability_rows_normalized": probability_rows_normalized,
        },
        "differential_optimizer_contract": optimizer_contract,
        "per_update_scheduler_contract": scheduler_contract,
        "optimizer_initial_learning_rates_after_scheduler_binding": optimizer_initial_lrs,
        "contract_checks": contract_checks,
        "runtime_contract_passed": runtime_contract_passed,
        "private_labels_used": False,
    }
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": payload["status"],
                "production_class_count": len(classes),
                "contract_checks": contract_checks,
                "active_training_gpu_touched": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    del source_model, target_model, optimizer, scheduler
    gc.collect()
    return 0 if runtime_contract_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
