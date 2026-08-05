from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PromotionDecision:
    status: str
    reason: str
    metrics: dict[str, Any]


def evaluate_candidate(metrics: dict[str, Any]) -> PromotionDecision:
    required = bool(metrics.get("target_fixed")) and bool(metrics.get("regression_green")) and bool(metrics.get("critical_subset_no_regression"))
    improved = float(metrics.get("success_rate_delta_pp", 0)) >= 2 or bool(metrics.get("p0_fixed"))
    if required and improved:
        return PromotionDecision("awaiting_human_promotion", "candidate passed isolated gates", metrics)
    return PromotionDecision("rejected_by_gate", "candidate did not satisfy regression and improvement gates", metrics)
