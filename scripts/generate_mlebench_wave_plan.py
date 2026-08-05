#!/usr/bin/env python3
"""Ask the configured EvoMind LLM for a constrained MLE-Bench wave plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_os.llm_client import LLMClient


WAVE1 = (
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dogs-vs-cats-redux-kernels-edition",
    "leaf-classification",
    "new-york-city-taxi-fare-prediction",
    "random-acts-of-pizza",
    "tabular-playground-series-dec-2021",
)

PROFILES: dict[str, str] = {
    "denoising-dirty-documents": "115 paired grayscale images; restoration RMSE; sampled-pixel ridge baseline",
    "detecting-insults-in-social-commentary": "small binary text AUC; word+char TF-IDF logistic",
    "dogs-vs-cats-redux-kernels-edition": "22,500 train images; binary log loss; A800 CNN",
    "leaf-classification": "990 rows; 99 classes; shape/margin/texture; multiclass log loss",
    "new-york-city-taxi-fare-prediction": "5.66 GB labels; chunked 1M sample; GPU CatBoost RMSE",
    "random-acts-of-pizza": "small text+numeric binary AUC; sparse logistic",
    "tabular-playground-series-dec-2021": "large multiclass accuracy; bounded 1.2M GPU CatBoost",
    "aptos2019-blindness-detection": "3,662 retinal images; ordinal 5-class QWK; pretrained vision backbone",
    "dog-breed-identification": "10,222 images; 120-class log loss; pretrained vision backbone",
    "histopathologic-cancer-detection": "220,025 pathology patches; binary AUC; pretrained CNN with center-aware crop",
    "jigsaw-toxic-comment-classification-challenge": "159,571 comments; six-label mean AUC; word+char sparse multilabel models",
    "mlsp-2013-birds": "small audio event dataset; binary AUC; log-mel summary features plus boosted classifier",
    "nomad2018-predict-transparent-conductors": "small materials table plus crystal geometry; two-target mean RMSLE",
    "plant-pathology-2020-fgvc7": "1,821 labeled leaf images; four-column mean AUC; pretrained vision backbone",
    "ranzcr-clip-catheter-line-classification": "30,083 chest radiographs; eleven-label mean AUC; pretrained multilabel CNN",
    "text-normalization-challenge-english-language": "English token normalization; exact accuracy; memorization and deterministic rules",
    "text-normalization-challenge-russian-language": "Russian token normalization; exact accuracy; memorization and deterministic rules",
    "the-icml-2013-whale-challenge-right-whale-redux": "small audio binary AUC; spectral summaries plus boosted classifier",
}


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Planner response did not contain a JSON object")
        payload = json.loads(stripped[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Planner response must be a JSON object")
    return payload


def validate_order(value: Any, competitions: tuple[str, ...] = WAVE1) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("competition_order must be a list")
    order = [str(item) for item in value]
    if (
        len(order) != len(competitions)
        or len(set(order)) != len(competitions)
        or set(order) != set(competitions)
    ):
        raise ValueError("competition_order must contain each selected competition exactly once")
    return order


def competitions_from_progress_report(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return WAVE1
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "evomind.mlebench_lite.progress.v1":
        raise ValueError("Progress report schema mismatch")
    values = tuple(str(item) for item in payload.get("remaining_competition_ids", []))
    if not values or len(values) != len(set(values)):
        raise ValueError("Progress report must contain unique remaining_competition_ids")
    unknown = sorted(set(values) - set(PROFILES))
    if unknown:
        raise ValueError(f"Missing competition profiles: {unknown}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wave", default="Wave1")
    parser.add_argument("--progress-report", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    competitions = competitions_from_progress_report(args.progress_report)
    native_evidence = workspace / "workspace" / "llm" / "evomind_native_tool_loop_current.json"
    evidence_sha = hashlib.sha256(native_evidence.read_bytes()).hexdigest() if native_evidence.is_file() else ""
    prompt_payload = {
        "objective": (
            f"Order {args.wave} for fast failure discovery, stable resource use, strongest early "
            "private-grader evidence, and reusable optimization evidence."
        ),
        "competitions": {competition: PROFILES[competition] for competition in competitions},
        "fixed_constraints": [
            f"Do not change the supplied {len(competitions)}-task set.",
            "No Kaggle submission; private grader only.",
            "All runtime files remain under the dedicated HPC root.",
            "The deterministic runner owns hyperparameters and safety caps.",
            "Return JSON only with competition_order, strategy_notes, rationale_summary, and risk_controls.",
        ],
    }
    prompt = json.dumps(prompt_payload, ensure_ascii=False, indent=2)
    system = (
        "You are EvoMind's MLE-Bench experiment planner. Produce a compact JSON execution plan. "
        "competition_order must contain every supplied competition exactly once. strategy_notes must "
        "map competition ids to concise arrays of evidence-focused actions. Do not add tasks or claims."
    )
    started = time.perf_counter()
    try:
        response = LLMClient(primary="openai", fallback="openai", max_retries=2, timeout=120).generate(
            prompt, system=system, max_tokens=1400, temperature=0.2, provider="openai"
        )
        model_plan = extract_json(response.text)
        order = validate_order(model_plan.get("competition_order"), competitions)
        notes = model_plan.get("strategy_notes") if isinstance(model_plan.get("strategy_notes"), dict) else {}
        report = {
            "schema": "evomind.mlebench_lite.optimization_plan.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "planner": {
                "provider": response.provider,
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "native_tool_loop_evidence_path": "workspace/llm/evomind_native_tool_loop_current.json" if native_evidence.is_file() else "",
                "native_tool_loop_evidence_sha256": evidence_sha,
            },
            "wave": args.wave,
            "competition_order": order,
            "strategy_notes": {competition: notes.get(competition, []) for competition in competitions},
            "rationale_summary": str(model_plan.get("rationale_summary") or "")[:1200],
            "risk_controls": [str(item)[:300] for item in (model_plan.get("risk_controls") or [])][:20],
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "response_sha256": hashlib.sha256(response.text.encode("utf-8")).hexdigest(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "execution_contract": {
                "ordering_is_model_planned": True,
                "training_adapters_are_deterministic": True,
                "hyperparameter_caps_are_runner_owned": True,
                "kaggle_submission_enabled": False,
            },
            "status": "passed",
            "ok": True,
        }
    except Exception as exc:
        report = {
            "schema": "evomind.mlebench_lite.optimization_plan.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "ok": False,
            "error_type": type(exc).__name__,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
