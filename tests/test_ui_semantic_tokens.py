from scripts.verify_workstation_semantic_tokens import verify


def test_active_workstation_uses_semantic_theme_tokens():
    result = verify()
    assert result["status"] == "passed", result["failures"]

