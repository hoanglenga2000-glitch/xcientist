"""Portable, bounded NumPy inference for the registered dense safetensors adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np


def read_weights(path: Path) -> dict[str, np.ndarray]:
    if path.is_symlink() or not path.is_file() or not 8 < path.stat().st_size <= 128 * 1024 * 1024:
        raise ValueError("checkpoint_size_or_path_invalid")
    raw = path.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    if not 2 <= size <= min(1024 * 1024, len(raw) - 8):
        raise ValueError("checkpoint_header_invalid")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("checkpoint_duplicate_key")
            value[key] = item
        return value
    header = json.loads(raw[8:8+size], object_pairs_hook=unique)
    if not isinstance(header, dict) or len(header) > 32:
        raise ValueError("checkpoint_tensor_limit")
    payload = memoryview(raw)[8+size:]
    result, intervals = {}, []
    for name, row in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(row, dict) or row.get("dtype") != "F32":
            raise ValueError("checkpoint_dtype_unsupported")
        shape, offsets = row.get("shape"), row.get("data_offsets")
        if (not isinstance(shape, list) or not 1 <= len(shape) <= 2 or any(type(n) is not int or not 1 <= n <= 100000 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets)):
            raise ValueError("checkpoint_shape_invalid")
        count = 1
        for n in shape:
            count *= n
        start, end = offsets
        if count > 5_000_000 or not 0 <= start < end <= len(payload) or end - start != count * 4:
            raise ValueError("checkpoint_offsets_invalid")
        value = np.frombuffer(payload[start:end], dtype="<f4").reshape(shape).copy()
        if not np.isfinite(value).all():
            raise ValueError("checkpoint_nonfinite")
        result[name] = value
        intervals.append((start, end))
    position = 0
    for start, end in sorted(intervals):
        if start != position:
            raise ValueError("checkpoint_overlap_or_gap")
        position = end
    if position != len(payload):
        raise ValueError("checkpoint_payload_mismatch")
    return result


def predict(model_root: Path, features: np.ndarray) -> np.ndarray:
    config_path = model_root / "model-config.json"
    if config_path.is_symlink() or config_path.stat().st_size > 65536:
        raise ValueError("model_configuration_invalid")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "evomind.tensor_training.v1" or config.get("task") not in {"regression", "classification"}:
        raise ValueError("model_adapter_unsupported")
    widths = [config["input_features"], *config["hidden_sizes"], config["outputs"]]
    if len(widths) > 8 or any(type(n) is not int or not 1 <= n <= 2048 for n in widths):
        raise ValueError("model_architecture_invalid")
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != widths[0] or not 1 <= len(x) <= 100000 or x.size > 10_000_000 or not np.isfinite(x).all():
        raise ValueError("inference_inputs_invalid")
    weights = read_weights(model_root / "model.safetensors")
    expected = {"normalizer_mean", "normalizer_scale"} | {f"layers.{index}.{part}" for index in range(len(widths)-1) for part in ("weight", "bias")}
    if set(weights) != expected or weights["normalizer_mean"].shape != (widths[0],) or weights["normalizer_scale"].shape != (widths[0],) or np.any(weights["normalizer_scale"] <= 0):
        raise ValueError("model_state_keys_or_normalizer_invalid")
    output = []
    for start in range(0, len(x), 512):
        batch = (x[start:start+512] - weights["normalizer_mean"]) / weights["normalizer_scale"]
        for index, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
            weight, bias = weights[f"layers.{index}.weight"], weights[f"layers.{index}.bias"]
            if weight.shape != (outputs, inputs) or bias.shape != (outputs,):
                raise ValueError("model_layer_shape_mismatch")
            batch = batch @ weight.T + bias
            if index < len(widths)-2:
                batch = np.maximum(batch, 0)
        if config["task"] == "classification":
            exponent = np.exp(batch - np.max(batch, axis=1, keepdims=True))
            batch = exponent / exponent.sum(axis=1, keepdims=True)
        output.append(batch)
    result = np.concatenate(output)
    if not np.isfinite(result).all():
        raise ValueError("inference_nonfinite")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("."))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("inference_file_too_large")
    values = predict(args.model_dir, np.load(args.input, allow_pickle=False))
    with args.output.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)
