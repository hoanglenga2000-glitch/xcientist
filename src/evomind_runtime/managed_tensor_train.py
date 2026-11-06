"""Registered GPU-only trainer. Evaluation labels are never provided to this process.

Phase 3 (training effect): a configuration with ``search`` compares up to 6 dense-network candidates by internal
k-fold cross-validation on the training rows only, within ``budget_seconds``, then refits the best one on all training
rows. Candidate 0 is the plain default; when another candidate wins, the default is refit too so the report can compare
both on the hold-out. The checkpoint format is unchanged: ``model-config.json`` describes the selected network.
This file is copied to the card alone: helpers stay self-contained and torch is imported only inside ``main``.
"""
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

CANDIDATE_FIELDS = ("hidden_sizes", "epochs", "batch_size", "learning_rate", "class_weight")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fold_indices(y, task: str, folds: int, seed: int):
    """Validation index arrays: stratified round-robin per class (classification), shuffled blocks (regression)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(folds)]
    if task == "classification":
        labels = np.asarray(y).reshape(-1).astype(np.int64)
        for value in np.unique(labels):
            members = rng.permutation(np.flatnonzero(labels == value))
            offset = int(rng.integers(folds))
            for position, index in enumerate(members):
                buckets[(position + offset) % folds].append(int(index))
    else:
        for position, index in enumerate(rng.permutation(len(y))):
            buckets[position % folds].append(int(index))
    return [np.asarray(sorted(bucket), dtype=np.int64) for bucket in buckets if bucket]


def selection_score(task: str, predictions, labels) -> float:
    """Higher is better. Classification: macro-F1 over classes seen in labels or predictions. Regression: R²."""
    import numpy as np

    if task == "classification":
        predicted = np.asarray(predictions).argmax(axis=1)
        labels = np.asarray(labels).reshape(-1).astype(np.int64)
        scores = []
        for value in np.union1d(np.unique(labels), np.unique(predicted)):
            tp = int(((predicted == value) & (labels == value)).sum())
            fp = int(((predicted == value) & (labels != value)).sum())
            fn = int(((predicted != value) & (labels == value)).sum())
            scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
        return float(np.mean(scores)) if scores else 0.0
    values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(labels, dtype=np.float64).reshape(-1)
    denominator = float(((targets - targets.mean()) ** 2).sum())
    error = float(((values - targets) ** 2).sum())
    return 1.0 - error / denominator if denominator else -error


def class_weights(y, outputs: int):
    """'Balanced' weights n / (k * count); classes absent from the training rows get weight 0 (they never occur)."""
    import numpy as np

    counts = np.bincount(np.asarray(y).reshape(-1).astype(np.int64), minlength=outputs).astype(np.float64)
    present = counts > 0
    weights = np.zeros(outputs, dtype=np.float32)
    weights[present] = (counts.sum() / (present.sum() * counts[present])).astype(np.float32)
    return weights


def run_search(candidates: list, x, y, task: str, folds: int, budget_seconds: float, seed: int, fit_predict, clock=time.monotonic,
               report=None, min_gain: float = 0.0, min_gain_ratio: float = 0.0) -> dict:
    """Cross-validate candidates in order until the budget would be exceeded; candidate 0 is always scored.

    ``fit_predict(candidate, x_train, y_train, x_valid) -> (predictions, steps)``. Ties keep the earlier candidate.
    The default (candidate 0) is replaced only by a clear win: CV gain >= max(min_gain, min_gain_ratio * |default|).
    ``report(done, total)`` is called after every fold, in fold units (keeps the progress heartbeat fresh).
    """
    import numpy as np

    splits = fold_indices(y, task, folds, seed)
    started = clock()
    results, steps, stopped = [], 0, False
    for index, candidate in enumerate(candidates):
        if results:
            average = sum(item["seconds"] for item in results) / len(results)
            if clock() - started + average > budget_seconds:
                stopped = True
                break
        began = clock()
        scores = []
        for valid in splits:
            train = np.setdiff1d(np.arange(len(y)), valid, assume_unique=True)
            predictions, used = fit_predict(candidate, x[train], y[train], x[valid])
            steps += int(used)
            scores.append(selection_score(task, predictions, y[valid]))
            if report is not None:
                report(index * len(splits) + len(scores), len(candidates) * len(splits))
        results.append({"index": index, "cv_score": round(float(np.mean(scores)), 6), "seconds": round(clock() - began, 3)})
    best = max(results, key=lambda item: (item["cv_score"], -item["index"]))
    default = results[0]["cv_score"]
    if best["index"] != 0 and best["cv_score"] - default < max(min_gain, min_gain_ratio * abs(default)):
        best = results[0]
    return {"results": results, "selected_index": int(best["index"]), "stopped_early": stopped, "steps": steps,
            "folds": len(splits), "seconds": round(clock() - started, 3)}


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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    with np.load(args.data_dir / "training.npz", allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"])
    evaluation_x = np.load(args.data_dir / "evaluation-features.npy", allow_pickle=False)
    task, outputs = config["task"], config["outputs"]

    class DenseModel(torch.nn.Module):
        def __init__(self, widths, x_fit):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(a, b) for a, b in zip(widths, widths[1:])])
            self.register_buffer("normalizer_mean", torch.from_numpy(x_fit.mean(axis=0)))
            self.register_buffer("normalizer_scale", torch.from_numpy(np.maximum(x_fit.std(axis=0), 1e-6)))

        def forward(self, value):
            value = (value - self.normalizer_mean) / self.normalizer_scale
            for index, layer in enumerate(self.layers):
                value = layer(value)
                if index < len(self.layers)-1:
                    value = torch.relu(value)
            return value

    def fit(candidate, x_fit, y_fit, log=None, progress_total=None):
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        model = DenseModel([config["input_features"], *candidate["hidden_sizes"], outputs], x_fit).cuda()
        features = torch.from_numpy(np.ascontiguousarray(x_fit)).cuda()
        if task == "classification":
            targets = torch.from_numpy(y_fit.reshape(-1).astype(np.int64)).cuda()
            weight = torch.from_numpy(class_weights(y_fit, outputs)).cuda() if candidate.get("class_weight") == "balanced" else None
            loss_function = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            targets = torch.from_numpy(y_fit.reshape(-1, 1).astype(np.float32)).cuda()
            loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=candidate["learning_rate"])
        steps, losses = 0, []
        for epoch in range(candidate["epochs"]):
            order = torch.randperm(len(features), device="cuda")
            total = 0.0
            for start in range(0, len(features), candidate["batch_size"]):
                indices = order[start:start+candidate["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(features[indices]), targets[indices])
                if not torch.isfinite(loss):
                    raise RuntimeError("training_loss_nonfinite")
                loss.backward()
                optimizer.step()
                steps += 1
                total += float(loss.detach().cpu()) * len(indices)
            losses.append(total / len(features))
            if log is not None:
                log.write(json.dumps({"epoch": epoch+1, "loss": losses[-1], "steps": steps}) + "\n"); log.flush()
                print("EVOMIND_PROGRESS " + json.dumps({"phase": "model_fitting", "completed_units": epoch+1, "total_units": progress_total or candidate["epochs"], "unit": "epoch", "detail": "Optimizer steps completed on CUDA"}), flush=True)
        return model, losses, steps

    def predict(model, values):
        model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(values), 512):
                value = model(torch.as_tensor(values[start:start+512], dtype=torch.float32, device="cuda"))
                if task == "classification":
                    value = torch.softmax(value, dim=1)
                out.append(value.cpu().numpy())
        return np.concatenate(out)

    base = {"hidden_sizes": config["hidden_sizes"], "epochs": config["epochs"], "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"], "class_weight": config.get("class_weight", "none")}
    search = config.get("search") if isinstance(config.get("search"), dict) else None
    started, steps = time.monotonic(), 0
    selection = None
    selected = base
    if search:
        candidates = [{key: item[key] for key in CANDIDATE_FIELDS} for item in search["candidates"]]

        def fit_predict(candidate, x_fit, y_fit, x_valid):
            model, _losses, used = fit(candidate, x_fit, y_fit)
            return predict(model, x_valid), used

        def report(done, total):
            print("EVOMIND_PROGRESS " + json.dumps({"phase": "model_selection", "completed_units": done, "total_units": total, "unit": "fold", "detail": "Cross-validating candidate settings on CUDA"}), flush=True)

        report(0, len(candidates) * int(search.get("folds", 3)))
        selection = run_search(candidates, x, y, task, int(search.get("folds", 3)), float(search.get("budget_seconds", 600)), config["seed"], fit_predict,
                               report=report, min_gain=float(search.get("min_gain", 0.0)), min_gain_ratio=float(search.get("min_gain_ratio", 0.0)))
        steps += selection["steps"]
        selected = candidates[selection["selected_index"]]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "training-log.jsonl").open("w", encoding="utf-8") as log:
        model, losses, used = fit(selected, x, y, log=log)
    steps += used
    predictions = predict(model, evaluation_x)
    baseline_sha = None
    if selection is not None and selection["selected_index"] != 0:
        baseline_model, _losses, used = fit(candidates[0], x, y)
        steps += used
        np.savez(args.out_dir / "baseline-predictions.npz", predictions=predict(baseline_model, evaluation_x))
        baseline_sha = sha(args.out_dir / "baseline-predictions.npz")
    torch.cuda.synchronize()
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}, str(args.out_dir / "model.safetensors"))
    np.savez(args.out_dir / "predictions.npz", predictions=predictions)
    if search:
        model_config = {key: value for key, value in config.items() if key != "search"}
        model_config.update({key: selected[key] for key in CANDIDATE_FIELDS})
        model_config.update(training_config_sha256=sha(config_path), selected_candidate=selection["selected_index"])
        (args.out_dir / "model-config.json").write_text(json.dumps(model_config, sort_keys=True, indent=2) + "\n")
    else:
        shutil.copyfile(config_path, args.out_dir / "model-config.json")
    receipt = {"schema": "evomind.managed_tensor_training.v1", "protocol_id": config["protocol_id"], "config_sha256": sha(config_path), "checkpoint_sha256": sha(args.out_dir / "model.safetensors"), "predictions_sha256": sha(args.out_dir / "predictions.npz"), "fit_steps": steps, "epochs": selected["epochs"], "initial_loss": losses[0], "final_loss": losses[-1], "device": "cuda:0", "gpu_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__, "cuda_runtime_version": torch.version.cuda, "driver_version": driver_version, "python_version": sys.version.split()[0], "worker_pid": os.getpid(), "elapsed_seconds": time.monotonic()-started, "evaluation_labels_used": False, "official_score": False}
    if selection is not None:
        receipt["model_selection"] = {"mode": "auto", "metric": search.get("metric"), "folds": selection["folds"], "results": selection["results"],
                                      "selected_index": selection["selected_index"], "stopped_early": selection["stopped_early"],
                                      "search_seconds": selection["seconds"]}
        receipt["model_config_sha256"] = sha(args.out_dir / "model-config.json")
        if baseline_sha:
            receipt["baseline_predictions_sha256"] = baseline_sha
    (args.out_dir / "training-receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
