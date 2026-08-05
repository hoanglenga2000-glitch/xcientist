from scripts import watch_siim_human_gate_completion as watch


def test_waiting_siim_never_collects(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, "remote_status", lambda: {"status": "waiting_for_leaf"})
    monkeypatch.setattr(watch, "collect", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("collect called")))
    result = watch.run_once(evidence=tmp_path / "e", source_plan=tmp_path / "plan", package_dir=tmp_path / "p", staged_run_id="hg-siim-test")
    assert result["status"] == "waiting_for_siim_terminal"
    assert result["official_grader_executed"] is False
    assert result["process_signals_sent"] == 0
