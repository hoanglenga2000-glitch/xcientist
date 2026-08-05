#!/usr/bin/env python3
"""Ask EvoMind's configured GPT-5.6 planner for a medal-recovery program."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from research_os.llm_client import LLMClient


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Planner response did not contain a JSON object")
        value = json.loads(stripped[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Planner response must be an object")
    return value


def validate_targets(value: Any, candidates: set[str], minimum: int = 5) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("conversion_targets must be a list")
    targets = [str(item) for item in value]
    if len(targets) < minimum or len(targets) != len(set(targets)) or not set(targets) <= candidates:
        raise ValueError("conversion_targets must contain at least five unique scored non-medals")
    return targets


def extract_ids(value: Any) -> list[str]:
    result: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if isinstance(item, dict):
            identifier = item.get("competition_id") or item.get("id") or item.get("competition")
            if identifier is not None:
                result.append(str(identifier))
                return
            # Models sometimes group targets by likelihood or key the strategy by
            # competition id. Candidate filtering downstream removes group labels.
            for key, child in item.items():
                result.append(str(key))
                if isinstance(child, (dict, list)):
                    visit(child)
            return
        if item is not None:
            result.append(str(item))

    visit(value)
    return result


def safe_validation_diagnostics(
    *,
    stage: str,
    plan: Any = None,
    returned_targets: Iterable[str] = (),
    filtered_targets: Iterable[str] = (),
    returned_priority: Iterable[str] = (),
    filtered_priority: Iterable[str] = (),
    raw_targets: Any = None,
    raw_priority: Any = None,
) -> dict[str, Any]:
    """Return bounded structural diagnostics without preserving model prose or secrets."""

    known_stages = {
        "request",
        "extract_json",
        "extract_conversion_targets",
        "validate_conversion_targets",
        "extract_priority_order",
        "validate_priority_order",
        "assemble_report",
    }
    known_plan_keys = {
        "conversion_targets",
        "priority_order",
        "strategies",
        "wave2_expectations",
        "stop_rules",
        "rationale_summary",
    }

    def trusted_ids(values: Iterable[str]) -> list[str]:
        # Callers supply only values already intersected with the trusted local
        # candidate set.  Raw model-returned values are represented by counts.
        return [value[:200] for value in list(values)[:50] if isinstance(value, str)]

    def count(values: Iterable[Any]) -> int:
        return len(list(values))

    def shape(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return {"type": "object", "length": len(value)}
        if isinstance(value, list):
            return {"type": "array", "length": len(value),
                    "item_types": sorted({type(item).__name__ for item in value})}
        return {"type": type(value).__name__, "length": None}

    plan_keys = set(plan) if isinstance(plan, dict) else set()
    return {
        "validation_stage": stage if stage in known_stages else "unknown",
        "parsed_plan_key_count": len(plan_keys),
        "known_plan_keys_present": sorted(plan_keys & known_plan_keys),
        "returned_conversion_target_count": count(returned_targets),
        "filtered_valid_conversion_targets": trusted_ids(filtered_targets),
        "returned_priority_count": count(returned_priority),
        "filtered_valid_priority_order": trusted_ids(filtered_priority),
        "conversion_targets_shape": shape(raw_targets),
        "priority_order_shape": shape(raw_priority),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--wave2-thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    thresholds = json.loads(args.wave2_thresholds.read_text(encoding="utf-8"))
    scored_nonmedals = [row for row in progress["results"] if not row["any_medal"]]
    candidates = {str(row["competition_id"]) for row in scored_nonmedals}
    payload = {
        "objective": "Strictly exceed the public MLE-Bench Lite top reference of 80.30% by reaching at least 18/22 medals.",
        "current_state": {
            "officially_scored": progress["scored_competitions"],
            "current_medals": progress["any_medal_count"],
            "remaining_unscored": progress["remaining_competitions"],
            "medals_required": progress["medals_required_to_strictly_exceed_top"],
            "minimum_existing_nonmedals_to_convert": progress["minimum_scored_non_medals_that_must_also_be_converted"],
        },
        "scored_nonmedals": scored_nonmedals,
        "allowed_conversion_target_ids": sorted(candidates),
        "wave2_official_thresholds": thresholds["rows"],
        "known_upgrade_options": {
            "spooky-author-identification": "NB-SVM word+character ensemble",
            "aerial-cactus-identification": "ImageNet ConvNeXt fine-tune and TTA",
            "dogs-vs-cats-redux-kernels-edition": "ImageNet ConvNeXt fine-tune and calibration",
            "new-york-city-taxi-fare-prediction": "larger clean sample, airport/time/geo features, GPU CatBoost",
            "random-acts-of-pizza": "sparse text plus structured CatBoost blend",
            "denoising-dirty-documents": "paired patch U-Net restoration",
            "siim-isic-melanoma-classification": "pretrained image plus metadata blend",
        },
        "fixed_constraints": [
            "Return JSON only.",
            "conversion_targets must include at least five unique ids copied exactly from allowed_conversion_target_ids.",
            "Separate likely, stretch, and low-probability targets using official bronze gaps.",
            "No Kaggle submission; official MLE-Bench private grader only.",
            "All training remains recoverable and deterministic under the dedicated HPC root.",
            "Return conversion_targets, priority_order, strategies, wave2_expectations, stop_rules, and rationale_summary.",
        ],
        "required_output_schema": {
            "conversion_targets": ["at least five exact strings from allowed_conversion_target_ids"],
            "priority_order": ["exact strings from allowed_conversion_target_ids in execution order"],
            "strategies": {"competition-id": ["concrete upgrade actions"]},
            "wave2_expectations": {"competition-id": "likely, stretch, or low-probability with reason"},
            "stop_rules": ["bounded evidence-based stop rule"],
            "rationale_summary": "short string",
        },
    }
    prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    started = time.perf_counter()
    report: dict[str, Any]
    validation_stage = "request"
    plan: dict[str, Any] | None = None
    returned_targets: list[str] = []
    filtered_targets: list[str] = []
    returned_priority: list[str] = []
    priority: list[str] = []
    response = None
    try:
        response = LLMClient(primary="openai", fallback="openai", max_retries=2, timeout=120).generate(
            prompt,
            system=(
                "You are EvoMind's benchmark optimization planner. Use only supplied scores and thresholds. "
                "Produce a technically concrete JSON plan; do not invent completed results."
            ),
            max_tokens=1800,
            temperature=0.15,
            provider="openai",
        )
        validation_stage = "extract_json"
        plan = extract_json(response.text)
        validation_stage = "extract_conversion_targets"
        returned_targets = extract_ids(plan.get("conversion_targets"))
        filtered_targets = list(dict.fromkeys(item for item in returned_targets if item in candidates))
        validation_stage = "validate_conversion_targets"
        targets = validate_targets(filtered_targets, candidates)
        validation_stage = "extract_priority_order"
        returned_priority = extract_ids(plan.get("priority_order"))
        priority = list(dict.fromkeys(item for item in returned_priority if item in candidates))
        validation_stage = "validate_priority_order"
        if not priority:
            raise ValueError("priority_order must be a unique subset of scored non-medals")
        validation_stage = "assemble_report"
        report = {
            "schema": "evomind.mlebench_lite.medal_recovery_plan.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "planner": {"provider": response.provider, "model": response.model,
                        "input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
            "conversion_targets": targets,
            "priority_order": priority,
            "discarded_out_of_scope_targets": [item for item in returned_targets if item not in candidates],
            "discarded_out_of_scope_priorities": [item for item in returned_priority if item not in candidates],
            "strategies": plan.get("strategies") if isinstance(plan.get("strategies"), dict) else {},
            "wave2_expectations": plan.get("wave2_expectations") if isinstance(plan.get("wave2_expectations"), dict) else {},
            "stop_rules": [str(item)[:400] for item in (plan.get("stop_rules") or [])][:30],
            "rationale_summary": str(plan.get("rationale_summary") or "")[:2000],
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "response_sha256": hashlib.sha256(response.text.encode("utf-8")).hexdigest(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "status": "passed", "ok": True,
        }
    except Exception as exc:
        report = {"schema": "evomind.mlebench_lite.medal_recovery_plan.v1",
                  "created_at": datetime.now(timezone.utc).isoformat(), "status": "failed", "ok": False,
                  "error_type": type(exc).__name__,
                  "error_detail": str(exc)[:500] if isinstance(exc, ValueError) else "",
                  "diagnostics": safe_validation_diagnostics(
                      stage=validation_stage,
                      plan=plan,
                      returned_targets=returned_targets,
                      filtered_targets=filtered_targets,
                      returned_priority=returned_priority,
                      filtered_priority=priority,
                      raw_targets=plan.get("conversion_targets") if isinstance(plan, dict) else None,
                      raw_priority=plan.get("priority_order") if isinstance(plan, dict) else None,
                  ),
                  "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                  "response_sha256": hashlib.sha256(response.text.encode("utf-8")).hexdigest()
                  if response is not None else "",
                  "planner": {
                      "provider": response.provider,
                      "model": response.model,
                      "input_tokens": response.input_tokens,
                      "output_tokens": response.output_tokens,
                  } if response is not None else {},
                  "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
