#!/usr/bin/env python3
"""Run one isolated SIIM training-step capacity probe on the selected GPU."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for entry in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["TORCH_HOME"] = str(args.torch_home.resolve())
    args.torch_home.mkdir(parents=True, exist_ok=True)

    import torch
    from torch import nn

    from scripts import mlebench_medal_recovery_adapters as recovery

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device_name = torch.cuda.get_device_name(0)
    if args.expected_gpu_name and args.expected_gpu_name.lower() not in device_name.lower():
        raise RuntimeError(
            f"Unexpected CUDA device: expected {args.expected_gpu_name!r}, got {device_name!r}"
        )

    total_mib = int(torch.cuda.get_device_properties(0).total_memory / 2**20)
    if args.memory_limit_mib <= 0 or args.memory_limit_mib > total_mib:
        raise RuntimeError(
            f"Invalid SIIM memory limit: {args.memory_limit_mib} MiB for {total_mib} MiB GPU"
        )
    torch.cuda.set_per_process_memory_fraction(
        float(args.memory_limit_mib) / float(total_mib),
        device=0,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, pretrained, weight_identity = recovery.build_siim_multiview_fusion_model(
        args.metadata_width,
        full_backbone_name=args.full_backbone,
        lesion_backbone_name=args.lesion_backbone,
    )
    if not pretrained:
        raise RuntimeError("Verified pretrained weights were not loaded")
    model = model.to(device="cuda", memory_format=torch.channels_last)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
        fused_adamw = True
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        fused_adamw = False
    loss_fn = nn.BCEWithLogitsLoss()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    full_image = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device="cuda",
    ).contiguous(memory_format=torch.channels_last)
    lesion_image = torch.randn_like(full_image).contiguous(memory_format=torch.channels_last)
    metadata = torch.randn(args.batch_size, args.metadata_width, device="cuda")
    target = torch.randint(0, 2, (args.batch_size,), device="cuda").float()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        logits = model.forward_channels(full_image, lesion_image, metadata)
        loss = (
            0.20 * loss_fn(logits["full_image"], target)
            + 0.20 * loss_fn(logits["lesion_focus"], target)
            + 0.60 * loss_fn(logits["image_metadata_fusion"], target)
        )
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_mib = int(torch.cuda.max_memory_allocated() / 2**20)
    if peak_mib > int(args.memory_limit_mib):
        raise RuntimeError(
            f"SIIM capacity probe exceeded memory limit: {peak_mib}>{args.memory_limit_mib} MiB"
        )
    return {
        "schema": "evomind.siim.candidate_batch_probe.v1",
        "created_at": now_iso(),
        "status": "passed",
        "gpu_name": device_name,
        "batch_size": args.batch_size,
        "effective_batch_size": args.effective_batch_size,
        "gradient_accumulation_steps": args.effective_batch_size // args.batch_size,
        "image_size": args.image_size,
        "metadata_width": args.metadata_width,
        "full_backbone": args.full_backbone,
        "lesion_backbone": args.lesion_backbone,
        "pretrained_weights_loaded": pretrained,
        "weight_identity": weight_identity,
        "amp_dtype": str(amp_dtype).removeprefix("torch."),
        "channels_last": True,
        "fused_adamw": fused_adamw,
        "loss": float(loss.detach().cpu()),
        "peak_memory_allocated_mib": peak_mib,
        "total_memory_mib": total_mib,
        "memory_limit_mib": int(args.memory_limit_mib),
        "headroom_mib": total_mib - peak_mib,
        "runtime_seconds": elapsed,
        "probe_source_sha256": sha256_file(Path(__file__).resolve()),
        "adapter_source_sha256": sha256_file(Path(recovery.__file__).resolve()),
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--metadata-width", type=int, default=16)
    parser.add_argument("--full-backbone", default="convnext_small")
    parser.add_argument("--lesion-backbone", default="efficientnet_v2_s")
    parser.add_argument("--torch-home", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", default="A800")
    parser.add_argument("--memory-limit-mib", type=int, default=55 * 1024)
    parser.add_argument("--effective-batch-size", type=int, default=384)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.image_size <= 0 or args.metadata_width <= 0:
        raise ValueError("Probe dimensions must be positive")
    if args.effective_batch_size < args.batch_size or args.effective_batch_size % args.batch_size:
        raise ValueError("Physical batch must divide the SIIM effective batch exactly")
    try:
        report = run_probe(args)
        return_code = 0
    except Exception as exc:
        error_type = type(exc).__name__
        report = {
            "schema": "evomind.siim.candidate_batch_probe.v1",
            "created_at": now_iso(),
            "status": "oom" if "out of memory" in str(exc).lower() else "failed",
            "batch_size": args.batch_size,
            "effective_batch_size": args.effective_batch_size,
            "image_size": args.image_size,
            "metadata_width": args.metadata_width,
            "memory_limit_mib": args.memory_limit_mib,
            "error_type": error_type,
            "error": str(exc),
            "probe_source_sha256": sha256_file(Path(__file__).resolve()),
            "process_signals_sent": 0,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        }
        return_code = 3
    write_json_atomic(args.report.resolve(), report)
    print(json.dumps(report, ensure_ascii=False))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
