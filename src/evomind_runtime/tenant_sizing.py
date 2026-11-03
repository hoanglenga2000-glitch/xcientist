"""t238 C7c ADDENDUM 2 §6: model size for tenant training as a function of the data and the VRAM budget.

The registered trainer (``managed_tensor_train.py``, hash-pinned) is a dense MLP driven by
``hidden_sizes`` / ``epochs`` / ``batch_size`` / ``learning_rate``. When the tenant's approved
configuration leaves these out, ``suggest`` picks them so a realistic table (≈100k rows ×
40–60 features, binary or 5-class) trains a meaningfully sized network — [1024, 1024, 512, 256]
for 30–60 epochs, about 20k optimizer steps (estimated 2–6 minutes on the A800; not measured
here) — and never exceeds the VRAM budget (``job_max_vram_mb`` / the tenant's request) or the
trainer's 5M-parameter cap.

The estimate is conservative and only used for sizing / the VRAM request: the trainer keeps the
whole dataset on the GPU (float32), Adam keeps 4 copies of the weights, activations are kept for
one batch, plus a fixed CUDA context allowance.
"""
from __future__ import annotations

import math
from typing import Any

PARAMETER_CAP = 5_000_000  # training_control.training_config
CUDA_CONTEXT_MB = 600
TARGET_STEPS = 20_000
BATCH_SIZE = 256
PRESETS = (  # (minimum training rows, hidden_sizes)
    (50_000, [1024, 1024, 512, 256]),
    (10_000, [512, 512, 256]),
    (2_000, [256, 128]),
    (0, [64, 32]),
)


def parameters(features: int, hidden: list[int], outputs: int) -> int:
    widths = [features, *hidden, outputs]
    return sum((left + 1) * right for left, right in zip(widths, widths[1:]))


def estimate_vram_mb(rows: int, features: int, hidden: list[int], outputs: int, batch: int = BATCH_SIZE) -> int:
    data = rows * (features + 1) * 4
    weights = parameters(features, hidden, outputs) * 4 * 4  # weights, grads, Adam m and v
    activations = batch * (features + sum(hidden) + outputs) * 4 * 3
    return int(math.ceil((data + weights + activations) / (1024 * 1024))) + CUDA_CONTEXT_MB


def suggest(rows: int, features: int, classes: int, *, vram_budget_mb: int, task: str = "classification") -> dict[str, Any]:
    """Return ``hidden_sizes``, ``epochs``, ``batch_size``, ``learning_rate`` and the VRAM request (MiB)."""

    if rows < 1 or features < 1 or vram_budget_mb < 1:
        raise ValueError("sizing_input_invalid")
    outputs = classes if task == "classification" and classes >= 2 else 1
    hidden = next(list(sizes) for minimum, sizes in PRESETS if rows >= minimum)
    batch = max(1, min(BATCH_SIZE, rows))
    # Shrink widths (never below 8) until the network fits the budget and the parameter cap.
    while (estimate_vram_mb(rows, features, hidden, outputs, batch) > vram_budget_mb
           or parameters(features, hidden, outputs) > PARAMETER_CAP):
        if max(hidden) <= 8:
            raise ValueError("sizing_does_not_fit_budget")
        hidden = [max(8, width // 2) for width in hidden]
    steps_per_epoch = max(1, math.ceil(rows / batch))
    if rows >= 50_000:
        epochs = min(60, max(30, round(TARGET_STEPS / steps_per_epoch)))
    else:
        epochs = min(200, max(20, round(TARGET_STEPS / steps_per_epoch)))
    need = estimate_vram_mb(rows, features, hidden, outputs, batch)
    request = min(int(vram_budget_mb), max(2048, int(math.ceil(need / 1024)) * 1024))
    return {"hidden_sizes": hidden, "epochs": int(epochs), "batch_size": int(batch), "learning_rate": 0.001,
            "vram_request_mb": request, "estimated_vram_mb": need,
            "parameters": parameters(features, hidden, outputs), "steps": int(epochs * steps_per_epoch)}


# -- held-out metrics (computed locally from the trainer's predictions; labels never reach the trainer) --

def _roc_auc_binary(scores, positives) -> float | None:
    import numpy as np

    positives = np.asarray(positives, dtype=bool)
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = np.asarray(scores)[order]
    i = 0
    while i < len(sorted_scores):  # average ranks for ties
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_metrics(probabilities, labels) -> dict[str, Any]:
    import numpy as np

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    predicted = probabilities.argmax(axis=1)
    classes = probabilities.shape[1]
    f1s = []
    for k in range(classes):
        tp = int(((predicted == k) & (labels == k)).sum())
        fp = int(((predicted == k) & (labels != k)).sum())
        fn = int(((predicted != k) & (labels == k)).sum())
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    if classes == 2:
        auc = _roc_auc_binary(probabilities[:, 1], labels == 1)
    else:
        aucs = [value for value in (_roc_auc_binary(probabilities[:, k], labels == k) for k in range(classes)) if value is not None]
        auc = float(np.mean(aucs)) if aucs else None
    return {"accuracy": float((predicted == labels).mean()), "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "roc_auc": auc, "roc_auc_average": "binary" if classes == 2 else "macro_ovr", "holdout_rows": int(len(labels))}


def regression_metrics(predictions, targets) -> dict[str, Any]:
    import numpy as np

    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    error = predictions - targets
    denominator = float(((targets - targets.mean()) ** 2).sum())
    return {"rmse": float(np.sqrt((error ** 2).mean())), "mae": float(np.abs(error).mean()),
            "r2": float(1 - (error ** 2).sum() / denominator) if denominator else None, "holdout_rows": int(len(targets))}
