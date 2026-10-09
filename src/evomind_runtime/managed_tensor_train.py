"""Registered GPU-only trainer. Evaluation labels are never provided to this process."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import numpy as np
    import torch
    from safetensors.torch import save_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.data_dir / "training-config.json"
    config = json.loads(config_path.read_text())
    if config.get("schema") != "evomind.tensor_training.v1" or not torch.cuda.is_available():
        raise RuntimeError("registered_training_requires_cuda")
    driver_version = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True, timeout=10).splitlines()[0].strip()
    if sha(args.data_dir / "training.npz") != config["training_sha256"] or sha(args.data_dir / "evaluation-features.npy") != config["evaluation_features_sha256"]:
        raise RuntimeError("registered_training_input_hash_mismatch")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    with np.load(args.data_dir / "training.npz", allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"])
    evaluation_x = np.load(args.data_dir / "evaluation-features.npy", allow_pickle=False)
    widths = [config["input_features"], *config["hidden_sizes"], config["outputs"]]

    class DenseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(a, b) for a, b in zip(widths, widths[1:])])
            self.register_buffer("normalizer_mean", torch.from_numpy(x.mean(axis=0)))
            self.register_buffer("normalizer_scale", torch.from_numpy(np.maximum(x.std(axis=0), 1e-6)))

        def forward(self, value):
            value = (value - self.normalizer_mean) / self.normalizer_scale
            for index, layer in enumerate(self.layers):
                value = layer(value)
                if index < len(self.layers)-1:
                    value = torch.relu(value)
            return value

    model = DenseModel().cuda()
    features = torch.from_numpy(x).cuda()
    targets = torch.from_numpy(y.reshape(-1).astype(np.int64)).cuda() if config["task"] == "classification" else torch.from_numpy(y.reshape(-1, 1).astype(np.float32)).cuda()
    loss_function = torch.nn.CrossEntropyLoss() if config["task"] == "classification" else torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started, steps, losses = time.monotonic(), 0, []
    with (args.out_dir / "training-log.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(config["epochs"]):
            order = torch.randperm(len(features), device="cuda")
            total = 0.0
            for start in range(0, len(features), config["batch_size"]):
                indices = order[start:start+config["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(features[indices]), targets[indices])
                if not torch.isfinite(loss):
                    raise RuntimeError("training_loss_nonfinite")
                loss.backward()
                optimizer.step()
                steps += 1
                total += float(loss.detach().cpu()) * len(indices)
            mean_loss = total / len(features)
            losses.append(mean_loss)
            record = {"epoch": epoch+1, "loss": mean_loss, "steps": steps}
            log.write(json.dumps(record) + "\n"); log.flush()
            print("EVOMIND_PROGRESS " + json.dumps({"phase": "model_fitting", "completed_units": epoch+1, "total_units": config["epochs"], "unit": "epoch", "detail": "Optimizer steps completed on CUDA"}), flush=True)
    model.eval()
    with torch.no_grad():
        values = []
        for start in range(0, len(evaluation_x), 512):
            value = model(torch.as_tensor(evaluation_x[start:start+512], dtype=torch.float32, device="cuda"))
            if config["task"] == "classification":
                value = torch.softmax(value, dim=1)
            values.append(value.cpu().numpy())
        predictions = np.concatenate(values)
    torch.cuda.synchronize()
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}, str(args.out_dir / "model.safetensors"))
    np.savez(args.out_dir / "predictions.npz", predictions=predictions)
    shutil.copyfile(config_path, args.out_dir / "model-config.json")
    receipt = {"schema": "evomind.managed_tensor_training.v1", "protocol_id": config["protocol_id"], "config_sha256": sha(config_path), "checkpoint_sha256": sha(args.out_dir / "model.safetensors"), "predictions_sha256": sha(args.out_dir / "predictions.npz"), "fit_steps": steps, "epochs": config["epochs"], "initial_loss": losses[0], "final_loss": losses[-1], "device": "cuda:0", "gpu_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__, "cuda_runtime_version": torch.version.cuda, "driver_version": driver_version, "python_version": sys.version.split()[0], "worker_pid": os.getpid(), "elapsed_seconds": time.monotonic()-started, "evaluation_labels_used": False, "official_score": False}
    (args.out_dir / "training-receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
