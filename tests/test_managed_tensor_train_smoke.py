"""Runs managed_tensor_train.main() end to end on a NumPy-backed stand-in for torch (no GPU, no learning).

It proves the control flow and the files/receipt the platform checks: plain run (byte-identical model-config) and
auto run (search, selected model-config bound to the receipt, default baseline predictions). Learning quality is not
tested here; the live card does that.
"""
from __future__ import annotations

import contextlib
import json
import sys
import types
from pathlib import Path

import numpy as np

from evomind_runtime import managed_tensor_train as card
from evomind_runtime import tenant_training, tools
from evomind_runtime.training_control import digest


class T:
    def __init__(self, value):
        self.a = np.asarray(value)

    def cuda(self): return self
    def cpu(self): return self
    def detach(self): return self
    def contiguous(self): return self
    def numpy(self): return self.a
    def __len__(self): return len(self.a)
    def __float__(self): return float(np.asarray(self.a).mean())
    def __getitem__(self, key): return T(self.a[key.a if isinstance(key, T) else key])
    def __sub__(self, other): return T(self.a - (other.a if isinstance(other, T) else other))
    def __truediv__(self, other): return T(self.a / (other.a if isinstance(other, T) else other))


def fake_torch(seed_calls: list) -> types.ModuleType:
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")

    class Module:
        def __init__(self):
            self._buffers = {}
        def register_buffer(self, name, value):
            self._buffers[name] = value
            setattr(self, name, value)
        def cuda(self): return self
        def eval(self): return self
        def parameters(self): return []
        def __call__(self, value): return self.forward(value)
        def state_dict(self):
            state = dict(self._buffers)
            for index, layer in enumerate(self.layers):
                state[f"layers.{index}.weight"], state[f"layers.{index}.bias"] = layer.weight, layer.bias
            return state

    class Linear:
        def __init__(self, a, b):
            rng = np.random.default_rng(a * 1000 + b)
            self.weight, self.bias = T(rng.normal(size=(b, a)).astype(np.float32) * 0.1), T(np.zeros(b, np.float32))
        def __call__(self, value): return T(value.a @ self.weight.a.T + self.bias.a)

    class Loss:
        def __init__(self, weight=None): self.weight = weight
        def __call__(self, output, target): return types.SimpleNamespace(backward=lambda: None, detach=lambda: T(np.abs(output.a).mean()))

    nn.Module, nn.Linear, nn.ModuleList = Module, Linear, list
    nn.CrossEntropyLoss, nn.MSELoss = Loss, lambda: Loss()
    torch.nn = nn
    torch.optim = types.SimpleNamespace(Adam=lambda params, lr: types.SimpleNamespace(zero_grad=lambda set_to_none=True: None, step=lambda: None))
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, manual_seed_all=lambda seed: None, synchronize=lambda: None,
                                       get_device_name=lambda index: "FakeGPU")
    torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()), cudnn=types.SimpleNamespace())
    torch.manual_seed = lambda seed: seed_calls.append(seed)
    torch.set_num_threads = lambda n: None
    torch.from_numpy = lambda array: T(array)
    torch.as_tensor = lambda array, dtype=None, device=None: T(np.asarray(array, dtype=np.float32))
    torch.randperm = lambda n, device=None: T(np.random.default_rng(n).permutation(n))
    torch.relu = lambda value: T(np.maximum(value.a, 0))
    torch.isfinite = lambda loss: True
    torch.no_grad = contextlib.nullcontext
    torch.float32 = np.float32

    def softmax(value, dim):
        e = np.exp(value.a - value.a.max(axis=1, keepdims=True))
        return T(e / e.sum(axis=1, keepdims=True))
    torch.softmax = softmax
    torch.__version__, torch.version = "0.fake", types.SimpleNamespace(cuda="12.x")
    return torch


def run_card(tmp_path: Path, monkeypatch, trainer: dict, prepared: dict) -> tuple[Path, str]:
    work, out = tmp_path / "work", tmp_path / "out"
    work.mkdir()
    np.savez(work / "training.npz", x=prepared["x"], y=prepared["y"])
    np.save(work / "evaluation-features.npy", prepared["evaluation_x"].astype(np.float32), allow_pickle=False)
    trainer = {**trainer, "training_sha256": digest(work / "training.npz"), "evaluation_features_sha256": digest(work / "evaluation-features.npy")}
    (work / "training-config.json").write_text(json.dumps(trainer, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    seeds: list = []
    safetensors = types.ModuleType("safetensors")
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.save_file = lambda tensors, path: Path(path).write_bytes(json.dumps(sorted(tensors)).encode())
    monkeypatch.setitem(sys.modules, "torch", fake_torch(seeds))
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)
    monkeypatch.setattr(card.subprocess, "check_output", lambda *a, **k: "999.1\n")
    monkeypatch.setattr(sys, "argv", ["managed_tensor_train.py", "--data-dir", str(work), "--out-dir", str(out)])
    card.main()
    return out, digest(work / "training-config.json")


def prepared_data(tmp_path: Path, family: str) -> dict:
    root = tmp_path / "ws"
    (root / "inputs").mkdir(parents=True)
    rng = np.random.default_rng(5)
    rows = ["a,b,y"] + [f"{rng.normal():.3f},{rng.normal():.3f},{i % 3}" for i in range(120)]
    (root / "inputs" / "d.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    config = {"data_path": "inputs/d.csv", "task": "classification", "hidden_sizes": [8], "epochs": 3, "batch_size": 16,
              "learning_rate": 0.01, "model_family": family}
    return tenant_training.prepare(root, config, digest(root / "inputs" / "d.csv"))


def test_plain_run_keeps_the_old_contract(tmp_path: Path, monkeypatch) -> None:
    prepared = prepared_data(tmp_path, "mlp")
    out, expected = run_card(tmp_path, monkeypatch, prepared["trainer_config"], prepared)
    receipt = json.loads((out / "training-receipt.json").read_text())
    assert digest(out / "model-config.json") == expected == receipt["config_sha256"]
    assert "model_selection" not in receipt and not (out / "baseline-predictions.npz").exists()
    assert receipt["epochs"] == 3 and receipt["fit_steps"] == 3 * 6
    trainer = json.loads((tmp_path / "work" / "training-config.json").read_text())
    assert tools._tenant_selected_config(out, trainer, receipt, expected)["hidden_sizes"] == [8]


def test_auto_run_writes_a_bound_selection_and_the_default_baseline(tmp_path: Path, monkeypatch) -> None:
    from evomind_runtime.tenant_sizing import classification_metrics, regression_metrics

    prepared = prepared_data(tmp_path, "auto")
    search = prepared["trainer_config"]["search"]
    out, expected = run_card(tmp_path, monkeypatch, prepared["trainer_config"], prepared)
    receipt = json.loads((out / "training-receipt.json").read_text())
    selection = receipt["model_selection"]
    assert selection["mode"] == "auto" and len(selection["results"]) == len(search["candidates"]) and selection["folds"] == 3
    trainer = json.loads((tmp_path / "work" / "training-config.json").read_text())
    selected = tools._tenant_selected_config(out, trainer, receipt, expected)
    assert selected == search["candidates"][selection["selected_index"]]
    with np.load(out / "predictions.npz") as loaded:
        assert loaded["predictions"].shape == (len(prepared["evaluation_x"]), trainer["outputs"])
    if selection["selected_index"] != 0:
        assert receipt["baseline_predictions_sha256"] == digest(out / "baseline-predictions.npz")
    report = tools._tenant_selection_report(out, {**prepared, "trainer_config": trainer}, receipt, selected, classification_metrics, regression_metrics)
    assert report["trials"] == len(search["candidates"])
    lines = (out / "training-log.jsonl").read_text().splitlines()
    assert len(lines) == selected["epochs"] and all("loss" in json.loads(line) for line in lines)
