from __future__ import annotations

from scripts import refresh_human_gate_approval_request as refresh


def test_live_approval_index_contains_verified_unapproved_candidates_only():
    request = refresh.build_request()
    run_ids = {item["run_id"] for item in request["candidates"]}
    assert "hg_jigsaw_multiseed_v3_20260727" in run_ids
    assert "hg_spooky_crossrun_xgb_v1_20260727" in run_ids
    assert all(item["approved"] is False for item in request["candidates"])
    assert request["automatic_approval"] is False
    assert request["official_grader_executed"] is False
