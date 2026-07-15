from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from research_os.agent.messaging import AssistantTurn
from xsci.scientist_hypothesis_panel import ROLE_SPECS, run_scientist_hypothesis_panel


class ConcurrencyTracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.maximum = 0

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


class StructuredClient:
    def __init__(self, role: str, phase: str, *, tracker: ConcurrencyTracker | None = None, veto_method: bool = False) -> None:
        self.role = role
        self.phase = phase
        self.tracker = tracker
        self.veto_method = veto_method

    def is_available(self) -> bool:
        return True

    def send(self, messages, **_kwargs):
        if self.tracker is not None:
            self.tracker.enter()
        try:
            time.sleep(0.04)
            if self.phase == "generation":
                confidence = {
                    "methodologist": 0.86,
                    "adversarial_validator": 0.68,
                    "resource_strategist": 0.60,
                }[self.role]
                payload = {
                    "proposals": [
                        {
                            "hypothesis": f"{self.role} controlled hypothesis",
                            "mechanism": f"mechanism selected by {self.role}",
                            "falsification_test": "reject when paired held-out delta is non-positive",
                            "evidence_required": ["paired metrics", "repeat manifest"],
                            "risks": ["api_key=fixture-secret-value"],
                            "resource_cost": {"level": "low", "needs_gpu": False},
                            "confidence": confidence,
                        }
                    ]
                }
            else:
                request = json.loads(messages[0]["content"])
                reviews = []
                for proposal in request["proposals"]:
                    is_method = "methodologist" in proposal["hypothesis"]
                    veto = self.veto_method and is_method and self.role in {
                        "adversarial_validator",
                        "resource_strategist",
                    }
                    score = 0.94 if is_method else 0.72
                    reviews.append(
                        {
                            "proposal_id": proposal["proposal_id"],
                            "confidence_adjustment": 0.04 if is_method else 0.0,
                            "methodological_score": score,
                            "evidence_score": score - 0.02,
                            "feasibility_score": score - 0.04,
                            "critical_veto": veto,
                            "veto_reason": "leakage-safe split missing" if veto else "",
                            "critique": "independent structured critique",
                        }
                    )
                payload = {"reviews": reviews}
            return AssistantTurn(
                text=json.dumps(payload),
                tool_calls=[],
                stop_reason="end_turn",
                raw_content=[{"type": "text", "text": "raw transcript must not persist"}],
                provider="openai",
                model=f"fixture-{self.role}",
                input_tokens=11,
                output_tokens=17,
            )
        finally:
            if self.tracker is not None:
                self.tracker.leave()


class UnavailableClient:
    def is_available(self) -> bool:
        return False


class MalformedClient:
    def is_available(self) -> bool:
        return True

    def send(self, *_args, **_kwargs):
        return AssistantTurn(
            text="not-json",
            tool_calls=[],
            stop_reason="end_turn",
            raw_content=[],
            provider="openai",
            model="malformed",
        )


class DuplicateKeyClient(MalformedClient):
    def send(self, *_args, **_kwargs):
        return AssistantTurn(
            text='{"proposals":[],"proposals":[]}',
            tool_calls=[],
            stop_reason="end_turn",
            raw_content=[],
            provider="openai",
            model="duplicate-key",
        )


def session() -> SimpleNamespace:
    return SimpleNamespace(selected_task="fixture-task", last_goal="improve the held-out metric")


def test_parallel_model_panel_generates_reviews_and_persists_only_structured_data(tmp_path: Path) -> None:
    tracker = ConcurrencyTracker()

    result = run_scientist_hypothesis_panel(
        session(),
        tmp_path,
        evidence_context={"metric": "accuracy", "provider_token": "must-not-persist"},
        client_factory=lambda role, phase: StructuredClient(role, phase, tracker=tracker),
    )

    assert result["ok"] is True
    assert result["mode"] == "model_parallel"
    assert result["model_call_count"] == 6
    assert result["fallback_call_count"] == 0
    assert len(result["proposals"]) == 3
    assert len(result["reviews"]) == 9
    assert result["selected_hypothesis"]["role"] == "methodologist"
    assert all(item["review_count"] == 3 for item in result["ranked_hypotheses"])
    assert tracker.maximum >= 3

    artifact = Path(result["artifact_path"])
    persisted = artifact.read_text(encoding="utf-8")
    assert "raw transcript must not persist" not in persisted
    assert "fixture-secret-value" not in persisted
    assert "must-not-persist" not in persisted
    assert "[redacted]" in persisted
    assert json.loads(persisted)["selected_hypothesis"]["proposal_id"] == result["selected_hypothesis"]["proposal_id"]
    assert Path(result["history_path"]).read_text(encoding="utf-8").endswith("\n")


def test_unavailable_provider_uses_explicit_deterministic_fallback(tmp_path: Path) -> None:
    result = run_scientist_hypothesis_panel(
        session(),
        tmp_path,
        client_factory=lambda _role, _phase: UnavailableClient(),
    )

    assert result["mode"] == "deterministic_fallback"
    assert result["model_call_count"] == 0
    assert result["fallback_call_count"] == 6
    assert {item["role"] for item in result["proposals"]} == set(ROLE_SPECS)
    assert all(item["generated_by"] == "deterministic_fallback" for item in result["proposals"])
    assert all(item["review_count"] == 3 for item in result["ranked_hypotheses"])
    assert result["no_training_started"] is True


def test_two_independent_critical_vetoes_reject_top_raw_confidence(tmp_path: Path) -> None:
    result = run_scientist_hypothesis_panel(
        session(),
        tmp_path,
        client_factory=lambda role, phase: StructuredClient(role, phase, veto_method=True),
    )

    method = next(item for item in result["ranked_hypotheses"] if item["role"] == "methodologist")
    assert method["critical_veto_count"] == 2
    assert method["status"] == "rejected_critical_veto"
    assert result["selected_hypothesis"]["role"] != "methodologist"
    assert result["selected_hypothesis"]["status"] == "ranked"


def test_malformed_model_output_fails_over_without_persisting_raw_text(tmp_path: Path) -> None:
    result = run_scientist_hypothesis_panel(
        session(),
        tmp_path,
        client_factory=lambda _role, _phase: MalformedClient(),
    )

    assert result["mode"] == "deterministic_fallback"
    assert all(run["status"] == "fallback_invalid_model_output" for run in result["generation_runs"])
    assert "not-json" not in Path(result["artifact_path"]).read_text(encoding="utf-8")


def test_duplicate_key_model_output_fails_over_as_ambiguous_json(tmp_path: Path) -> None:
    result = run_scientist_hypothesis_panel(
        session(),
        tmp_path,
        client_factory=lambda _role, _phase: DuplicateKeyClient(),
    )

    assert result["mode"] == "deterministic_fallback"
    assert all(run["status"] == "fallback_invalid_model_output" for run in result["generation_runs"])
