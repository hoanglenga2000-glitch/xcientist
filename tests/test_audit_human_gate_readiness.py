from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import audit_human_gate_readiness as audit


ROOT = Path(__file__).resolve().parents[1]


def test_live_human_gate_readiness_audit_is_candidate_only() -> None:
    request = audit.DEFAULT_REQUEST
    if not request.is_file():
        pytest.skip("Live Human Gate approval request is not present")

    payload = audit.build_audit(approval_request_path=request)

    assert payload["schema"] == audit.AUDIT_SCHEMA
    assert payload["ready_for_explicit_human_approval"] is True
    assert payload["official_grader_executed"] is False
    assert payload["kaggle_submission_executed"] is False
    assert payload["errors"] == []
    assert payload["approval_request"]["status"] == "awaiting_explicit_human_approval"
    assert payload["checks"]
    assert all(item["approval_template_approved"] is False for item in payload["checks"])
    assert all(item["regrades_dir_exists"] is False for item in payload["checks"])
    assert all(item["source_result_contract"] == "passed" for item in payload["checks"])


def test_cli_writes_machine_readable_readiness_audit(tmp_path: Path) -> None:
    request = audit.DEFAULT_REQUEST
    if not request.is_file():
        pytest.skip("Live Human Gate approval request is not present")
    output = tmp_path / "audit.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_human_gate_readiness.py"),
            "--approval-request",
            str(request),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready_for_explicit_human_approval"] is True
    assert payload["automatic_approval"] is False
