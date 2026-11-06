"""2026-11-03: the training plan shown on the consent card before any GPU time is used.

Reads the exact configuration the approval binds (same file, same SHA-256) and the tenant's data,
fills the sizing fields the configuration leaves out exactly as ``tenant_training.prepare`` will,
and returns a small, display-only summary: task, target, network, epochs, batch, learning rate,
rows/features and an estimated on-card time. It never raises: a plan that cannot be computed is
simply absent and the card falls back to its plain copy.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# Observed on the A800 (2026-11-03 acceptance runs): ~15-25 s fixed start-up per run plus the optimizer steps.
STARTUP_SECONDS = 18
STEPS_PER_SECOND = 800


def estimate_seconds(steps: int) -> dict[str, int]:
    centre = STARTUP_SECONDS + max(0, int(steps)) / STEPS_PER_SECOND
    return {"low": max(5, int(math.floor(centre * 0.8))), "high": int(math.ceil(centre * 1.6))}


def build(workspace_root: str | Path, arguments: dict[str, Any], *, vram_budget_mb: int | None = None) -> dict[str, Any] | None:
    try:
        from . import tenant_training
        from .tenant_sizing import suggest
        from .training_control import digest, read_json

        root = Path(workspace_root)
        config_path = Path(str(arguments.get("configuration_path") or ""))
        if not config_path.is_absolute():
            config_path = root / config_path
        if config_path.is_symlink():
            return None
        config_path.resolve(strict=True).relative_to(root.resolve(strict=True))
        if digest(config_path) != str(arguments.get("configuration_sha256") or ""):
            return None
        config = read_json(config_path)
        task = config.get("task")
        if task not in {"classification", "regression"}:
            return None
        path = tenant_training.resolve_input(root, str(config.get("data_path") or ""))
        x, _y, classes = tenant_training.load_data(path, task=task, target_column=config.get("target_column"))
        rows, features = int(x.shape[0]), int(x.shape[1])
        train_rows = rows - max(1, rows // 5)
        sizing = suggest(train_rows, features, classes, vram_budget_mb=int(vram_budget_mb or 4096), task=task)
        chosen = {key: config.get(key, sizing[key]) for key in ("hidden_sizes", "epochs", "batch_size", "learning_rate")}
        batch = max(1, int(chosen["batch_size"]))
        steps = int(chosen["epochs"]) * max(1, math.ceil(train_rows / batch))
        from . import tenant_search
        family = config.get("model_family", "auto")
        auto = family == "auto" and tenant_search.fits_budget(train_rows, chosen)
        trials = len(tenant_search.candidates({**chosen}, task=task)) if auto else 1
        estimate = estimate_seconds(steps)
        if auto:
            extra = tenant_search.search_seconds(train_rows, chosen, trials) + int(steps / STEPS_PER_SECOND)
            estimate = {"low": estimate["low"] + int(extra * 0.8), "high": estimate["high"] + int(math.ceil(extra * 1.6))}
        return {
            "task": task,
            "task_label": "分类" + (f"（{classes} 类）" if classes else "") if task == "classification" else "回归（预测数值）",
            "target_column": config.get("target_column") or "最后一列",
            "data_name": path.name,
            "rows": rows,
            "train_rows": train_rows,
            "holdout_rows": rows - train_rows,
            "features": features,
            "network": (f"全连接神经网络（MLP），自动比较 {trials} 组设置后选效果最好的" if auto else "全连接神经网络（MLP）"),
            "model_family": "auto" if auto else "mlp",
            "trials": trials,
            "hidden_sizes": [int(width) for width in chosen["hidden_sizes"]],
            "epochs": int(chosen["epochs"]),
            "batch_size": batch,
            "learning_rate": float(chosen["learning_rate"]),
            "steps": steps,
            "metrics": ["Accuracy", "Macro-F1", "ROC-AUC"] if task == "classification" else ["RMSE", "MAE", "R²"],
            "estimated_gpu_seconds": estimate,
            "billing_note": "只按显卡上实际训练的时间计费；环境准备和排队不计费。",
        }
    except Exception:  # noqa: BLE001 - display only: never blocks the approval itself
        return None
