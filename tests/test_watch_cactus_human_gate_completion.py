from __future__ import annotations

from scripts import watch_cactus_human_gate_completion as watch


def test_waiting_cactus_never_collects_or_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, "read_remote_status", lambda: {"status": "waiting_for_v1_retirement"})
    monkeypatch.setattr(watch, "collect_candidate", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("collect called")))
    result = watch.run_once(evidence_dir=tmp_path / "e", source_plan=tmp_path / "plan.json", package_dir=tmp_path / "p", staged_run_id="hg-cactus-test")
    assert result["status"] == "waiting_for_cactus_terminal"
    assert result["process_signals_sent"] == 0
    assert result["official_grader_executed"] is False
