"""GPU-accelerated image classifier (CNN / ResNet / ViT) for pixel competitions.

torch is imported lazily inside training/inference so this module stays
import-safe (and unit-testable) on machines without torch — the control plane.
Actual training runs on the GPU server.

The pure, testable surface here is:
  * infer_image_shape()  - turn a flat pixel count into (C, H, W)
  * build_arch_config()  - choose an architecture + hyperparameters for the input
  * ImageClassifier config/validation (constructor) without requiring torch

Training/inference (fit/predict_proba) require torch and are exercised on GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ArchConfig:
    model_type: str            # "cnn" | "resnet18" | "vit_tiny"
    input_shape: tuple[int, int, int]  # (channels, height, width)
    num_classes: int
    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "input_shape": list(self.input_shape),
            "num_classes": self.num_classes,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "notes": list(self.notes),
        }


# Common square image edge lengths seen in Kaggle pixel CSVs.
_COMMON_EDGES = (8, 16, 28, 32, 48, 64, 96, 128, 224, 256)


def infer_image_shape(n_pixels: int, channels: int = 1) -> tuple[int, int, int]:
    """Infer (C, H, W) from a flat pixel-column count.

    Tries the given channel count first (grayscale by default), then 3 (RGB).
    Falls back to the nearest perfect square if no common edge matches.
    """
    for ch in (channels, 3, 1):
        if ch <= 0 or n_pixels % ch != 0:
            continue
        per_channel = n_pixels // ch
        edge = int(round(math.isqrt(per_channel)))
        if edge * edge == per_channel:
            if edge in _COMMON_EDGES or per_channel in (e * e for e in _COMMON_EDGES):
                return (ch, edge, edge)
    # Fallback: assume single channel, nearest square edge.
    edge = int(round(math.sqrt(n_pixels)))
    return (1, edge, edge)


def build_arch_config(
    n_pixels: int,
    num_classes: int,
    *,
    channels: int = 1,
    prefer: Optional[str] = None,
) -> ArchConfig:
    """Choose an architecture and hyperparameters for the given pixel input."""
    shape = infer_image_shape(n_pixels, channels)
    _, h, w = shape
    notes: list[str] = [f"inferred input shape {shape} from {n_pixels} pixels"]

    if prefer:
        model_type = prefer
    elif h <= 32:
        model_type = "cnn"  # small images: a compact CNN is fast and strong
        notes.append("small image -> compact 2-conv CNN")
    elif h <= 96:
        model_type = "resnet18"
        notes.append("medium image -> ResNet18 transfer learning")
    else:
        model_type = "vit_tiny"
        notes.append("large image -> ViT-tiny")

    # Scale epochs/batch to image size.
    if h <= 32:
        epochs, batch = 15, 128
    elif h <= 96:
        epochs, batch = 20, 64
    else:
        epochs, batch = 25, 32

    return ArchConfig(
        model_type=model_type,
        input_shape=shape,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch,
        notes=notes,
    )


class ImageClassifier:
    """Thin wrapper that validates config on init and lazy-loads torch to train.

    Usage on GPU:
        clf = ImageClassifier(build_arch_config(784, 10))
        clf.fit(X_pixels, y, X_val=..., y_val=...)   # X: (n, n_pixels) or (n,C,H,W)
        proba = clf.predict_proba(X_test)             # (n, num_classes), softmax

    ``classes_`` is exposed after fit as ``arange(num_classes)`` so the GPU trainer's
    full-width proba alignment treats it exactly like a GBDT classifier.
    """

    def __init__(self, config: ArchConfig) -> None:
        if config.num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if any(d <= 0 for d in config.input_shape):
            raise ValueError(f"invalid input_shape {config.input_shape}")
        self.config = config
        self._model = None
        self._device = None
        self.classes_ = None

    # ── training / inference require torch (GPU) ──
    def _build_module(self):  # pragma: no cover - requires torch/GPU
        import torch.nn as nn

        c, h, w = self.config.input_shape
        if self.config.model_type == "cnn":
            flat = 64 * (h // 4) * (w // 4)
            return nn.Sequential(
                nn.Conv2d(c, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(flat, 128), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(128, self.config.num_classes),
            )
        raise NotImplementedError(f"module builder for {self.config.model_type} runs on GPU trainer")

    def _require_torch(self):
        try:
            import torch  # noqa: F401
            return torch
        except ImportError as exc:  # control plane has no torch
            raise RuntimeError(
                "ImageClassifier requires torch and runs on the GPU server; "
                "the control plane only builds/validates config."
            ) from exc

    def _reshape(self, X, torch):
        """Accept (n, n_pixels) or (n, C, H, W); return a float tensor (n, C, H, W).

        Pixel values are scaled to [0, 1] when they look like 0-255 ints.
        """
        import numpy as np

        c, h, w = self.config.input_shape
        arr = np.asarray(X, dtype="float32")
        if arr.ndim == 2:
            arr = arr.reshape(arr.shape[0], c, h, w)
        elif arr.ndim != 4:
            raise ValueError(f"expected 2D or 4D input, got shape {arr.shape}")
        mx = float(arr.max()) if arr.size else 1.0
        if mx > 1.5:  # looks like 0-255 pixel intensities
            arr = arr / 255.0
        return torch.from_numpy(arr)

    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        """Train the CNN on GPU with Adam + early stopping (restores best weights).

        Signature mirrors the GBDT wrapper (optional explicit validation split), so
        the trainer's cross-validation loop can drive CNN and GBDT identically.
        """
        torch = self._require_torch()
        import numpy as np
        import torch.nn as nn

        # Free GPU memory before training (critical for multi-fold CV)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cfg = self.config
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classes_ = np.arange(cfg.num_classes)

        model = self._build_module().to(self._device)
        Xt = self._reshape(X, torch)
        yt = torch.as_tensor(np.asarray(y).astype("int64"))
        ds = torch.utils.data.TensorDataset(Xt, yt)
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xv = self._reshape(X_val, torch).to(self._device)
            yv = torch.as_tensor(np.asarray(y_val).astype("int64")).to(self._device)

        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        loss_fn = nn.CrossEntropyLoss()
        best_state, best_val, patience, bad = None, None, 3, 0

        for epoch in range(cfg.epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
            if has_val:
                model.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(model(Xv), yv))
                if best_val is None or val_loss < best_val - 1e-4:
                    best_val, bad = val_loss, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model
        return self

    def predict_proba(self, X):
        """Return (n, num_classes) softmax probabilities aligned to classes 0..K-1."""
        torch = self._require_torch()
        if self._model is None:
            raise RuntimeError("predict_proba called before fit")
        import numpy as np

        self._model.eval()
        Xt = self._reshape(X, torch).to(self._device)
        out_chunks = []
        with torch.no_grad():
            for i in range(0, Xt.shape[0], 1024):  # chunk to bound GPU memory
                logits = self._model(Xt[i:i + 1024])
                out_chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out_chunks, axis=0)

    # ── interface parity with the trainer's _GBDTModel wrapper ──
    def predict_proba_full(self, X):
        """Alias: softmax proba is already full-width over classes 0..K-1."""
        return self.predict_proba(X)

    def predict(self, X):
        import numpy as np
        return np.argmax(self.predict_proba(X), axis=1)
