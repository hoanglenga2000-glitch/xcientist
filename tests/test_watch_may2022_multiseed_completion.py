from __future__ import annotations

import json
from pathlib import Path

from scripts import watch_may2022_multiseed_completion as watch


def test_waiting_may_queue_never_collects_or_stages(tmp_path: Path, monkeypatch):
    plan = watch.queue.validate_execution_plan()
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"status": "seed_active", "plan_sha256": plan["_sha256"]}), encoding="utf-8")
    monkeypatch.setattr(watch.ops, "collect_run", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("collect called")))
    result = watch.run_once(plan, queue_status_path=status, evidence_dir=tmp_path / "e", package_dir=tmp_path / "p", sample_path=tmp_path / "sample.csv", staged_run_id="hg-may-test")
    assert result["status"] == "waiting_for_may_seeds_terminal"
    assert result["official_grader_executed"] is False
    assert result["process_signals_sent"] == 0
