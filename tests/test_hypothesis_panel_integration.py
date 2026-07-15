from __future__ import annotations

from types import SimpleNamespace

import xsci.kaggle as kaggle
from xsci.kaggle_conversation import _forced_tool_hints
from xsci.kaggle_intent import TOOL_QUERY, classify
from xsci.scientist_adaptive_loop import _allowed_tools
from xsci.terminal_events import render_scientist_hypothesis_panel_summary
from xsci.terminal_tools import TerminalTools


def test_hypothesis_panel_is_registered_and_routed_before_legacy_review() -> None:
    assert "scientist_hypothesis_panel" in TerminalTools.list_tool_names()
    intent = classify("run a multi-agent hypothesis panel with independent critics")
    assert intent.kind == TOOL_QUERY
    assert intent.payload == "scientist_hypothesis_panel"
    assert _forced_tool_hints("parallel hypotheses with independent critics") == ["scientist_hypothesis_panel"]


def test_meta_adaptive_loop_can_select_hypothesis_panel() -> None:
    allowed = _allowed_tools(
        {"tool_sequence": [], "requirement_ledger": {"requirements": []}},
        "Improve this AI Scientist with multi-agent hypotheses",
    )
    assert "scientist_hypothesis_panel" in allowed


def test_cli_alias_preserves_goal_and_dispatches_panel(monkeypatch, tmp_path) -> None:
    captured = {}
    injected = []

    def fake_show(session, root):
        captured["goal"] = session.last_goal
        captured["root"] = root
        return 0

    monkeypatch.setattr(kaggle, "inject_engine_env", lambda cfg: injected.append(cfg))
    monkeypatch.setattr(kaggle, "_show_scientist_hypothesis_panel", fake_show)
    assert kaggle._dispatch(["hypothesis-panel", "compare", "three", "methods"], tmp_path) == 0
    assert captured == {"goal": "compare three methods", "root": tmp_path}
    assert len(injected) == 1


def test_panel_renderer_exposes_veto_and_disagreement_without_transcript() -> None:
    result = {
        "ok": True,
        "mode": "model_parallel",
        "model_call_count": 6,
        "fallback_call_count": 0,
        "selection_status": "selected",
        "selected_hypothesis": {
            "rank": 1,
            "proposal_id": "HP-MET-1",
            "role": "methodologist",
            "panel_score": 0.91,
            "adjusted_confidence": 0.88,
            "hypothesis": "controlled hypothesis",
            "falsification_test": "reject on non-positive delta",
        },
        "ranked_hypotheses": [
            {
                "rank": 1,
                "proposal_id": "HP-MET-1",
                "role": "methodologist",
                "panel_score": 0.91,
                "status": "ranked",
                "critical_veto_count": 0,
                "review_disagreement": 0.02,
            }
        ],
        "artifact_path": ".xsci/scientist_hypothesis_panel.json",
    }
    text = "\n".join(render_scientist_hypothesis_panel_summary(result))
    assert "model=6 fallback=0" in text
    assert "vetoes=0 disagreement=0.02" in text
    assert "raw_content" not in text


def test_panel_wrapper_uses_current_goal(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run(session, root, *, goal):
        captured.update(session=session, root=root, goal=goal)
        return {"ok": True, "tool": "scientist_hypothesis_panel"}

    import xsci.scientist_hypothesis_panel as panel

    monkeypatch.setattr(panel, "run_scientist_hypothesis_panel", fake_run)
    state = SimpleNamespace(last_goal="test goal", selected_task="fixture")
    result = TerminalTools.dispatch("scientist_hypothesis_panel", state, tmp_path)
    assert result["ok"] is True
    assert captured["goal"] == "test goal"
