from scripts import watch_ranzcr_human_gate_completion as watch


def test_waiting_ranzcr_never_collects(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, "remote_status", lambda: {"status": "waiting_for_siim"})
    monkeypatch.setattr(watch, "collect", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("collect called")))
    result = watch.run_once(evidence=tmp_path / "e", source_plan=tmp_path / "plan", package_dir=tmp_path / "package", staged_run_id="hg-ranzcr-test")
    assert result["status"] == "waiting_for_ranzcr_terminal"
    assert result["official_grader_executed"] is False
    assert result["process_signals_sent"] == 0
