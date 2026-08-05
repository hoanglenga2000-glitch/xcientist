from scripts.verify_workstation_action_coverage import CATEGORIES, verify


def test_workstation_action_coverage_matrix_is_complete() -> None:
    report = verify()
    assert report["status"] == "passed", report["failures"]
    assert report["source_contract_count"] >= 100
    assert set(report["category_counts"]) == CATEGORIES
    assert report["browser_assertion_count"] + report["blocked_assertion_count"] == report["source_contract_count"]


def test_blocked_controls_have_explicit_blocked_assertions() -> None:
    report = verify()
    blocked = [item for item in report["contracts"] if item["action"].startswith("blocked_")]
    assert blocked
    assert all(item["category"] == "human_gate" for item in blocked)
    assert all(item["assertion_type"] == "blocked_assertion" for item in blocked)
