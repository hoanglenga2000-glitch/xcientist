"""Phase 3: candidate settings for ``model_family='auto'`` tenant training (pure, deterministic, no I/O).

The card cross-validates them (managed_tensor_train.run_search). Candidate 0 is always the plain default the
platform would have trained before, so the report can compare the chosen setting against it.
"""
from __future__ import annotations

from typing import Any, Callable

MODEL_FAMILIES = ("auto", "mlp")
MAX_TRIALS = 6
FOLDS = 3
BUDGET_SECONDS = 600
STEPS_PER_SECOND = 800  # same planning constant as training_plan
FIELDS = ("hidden_sizes", "epochs", "batch_size", "learning_rate", "class_weight")


def _key(candidate: dict[str, Any]) -> tuple:
    return tuple(tuple(value) if isinstance(value, list) else value for value in (candidate[field] for field in FIELDS))


def candidates(base: dict[str, Any], *, task: str, warm: dict[str, Any] | None = None,
               validate: Callable[[dict[str, Any]], Any] | None = None) -> list[dict[str, Any]]:
    """Up to MAX_TRIALS distinct, valid candidates; ``validate`` raises ValueError for one the trainer would refuse."""
    hidden = [int(width) for width in base["hidden_sizes"]]
    epochs, batch, rate = int(base["epochs"]), int(base["batch_size"]), float(base["learning_rate"])
    plain = {"hidden_sizes": hidden, "epochs": epochs, "batch_size": batch, "learning_rate": rate, "class_weight": "none"}
    weighted = "balanced" if task == "classification" else "none"
    pool = [plain]
    if isinstance(warm, dict) and warm:
        try:
            pool.append({"hidden_sizes": [int(width) for width in warm.get("hidden_sizes") or hidden],
                         "epochs": int(warm.get("epochs", epochs)), "batch_size": int(warm.get("batch_size", batch)),
                         "learning_rate": float(warm.get("learning_rate", rate)),
                         "class_weight": warm.get("class_weight") if warm.get("class_weight") in {"none", "balanced"} else "none"})
        except (TypeError, ValueError):
            pass
    # 2026-11-06 study (5 random 80/20 splits, wine + breast cancer): class weighting, a wider network and fewer epochs
    # (a light regulariser) helped; learning-rate and depth changes did not. Order = priority under the 6-trial cap.
    wide = [min(1024, width * 2) for width in hidden]
    short = max(10, round(epochs * 0.3))
    if task == "classification":
        pool += [{**plain, "class_weight": "balanced"}, {**plain, "hidden_sizes": wide},
                 {**plain, "hidden_sizes": wide, "class_weight": "balanced"}, {**plain, "epochs": short},
                 {**plain, "epochs": short, "class_weight": "balanced"}]
    else:
        pool += [{**plain, "hidden_sizes": wide}, {**plain, "epochs": short}, {**plain, "hidden_sizes": wide, "epochs": short},
                 {**plain, "learning_rate": max(1e-5, rate * 0.3)}, {**plain, "hidden_sizes": [*hidden, max(4, hidden[-1] // 2)] if len(hidden) < 6 else hidden}]
    del weighted
    chosen: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for candidate in pool:
        if candidate["class_weight"] not in {"none", "balanced"} or _key(candidate) in seen:
            continue
        if validate is not None:
            try:
                validate({key: candidate[key] for key in FIELDS[:4]})
            except ValueError:
                continue
        seen.add(_key(candidate))
        chosen.append(candidate)
        if len(chosen) == MAX_TRIALS:
            break
    return chosen


def fits_budget(train_rows: int, base: dict[str, Any]) -> bool:
    """Auto only when one cross-validated candidate (about two full fits) fits well inside the search budget."""
    steps = int(base["epochs"]) * max(1, -(-int(train_rows) // max(1, int(base["batch_size"]))))
    return 2 * steps / STEPS_PER_SECOND <= BUDGET_SECONDS / 3


def search_block(task: str, chosen: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema": "evomind.tensor_search.v1", "candidates": chosen, "folds": FOLDS, "budget_seconds": BUDGET_SECONDS,
            "metric": "macro_f1" if task == "classification" else "r2"}


def search_seconds(train_rows: int, base: dict[str, Any], trials: int) -> int:
    """Planning estimate of the search part (cross-validation of every candidate), capped by the budget."""
    steps = int(base["epochs"]) * max(1, -(-int(train_rows) // max(1, int(base["batch_size"]))))
    return int(min(BUDGET_SECONDS, trials * FOLDS * (2 / 3) * steps / STEPS_PER_SECOND + trials * FOLDS))
