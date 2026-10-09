from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_launch_go_no_go import primary_llm_ready  # noqa: E402


def test_openai_ready_satisfies_primary_llm_gate_without_deepseek() -> None:
    connectors = {
        "openai": {"configured": True, "state": "READY"},
        "deepseek": {"configured": False, "state": "NOT_CONFIGURED"},
    }
    assert primary_llm_ready(connectors) is True


def test_deepseek_ready_can_satisfy_primary_llm_gate_without_openai() -> None:
    connectors = {
        "openai": {"configured": False, "state": "NOT_CONFIGURED"},
        "deepseek": {"configured": True, "state": "READY"},
    }
    assert primary_llm_ready(connectors) is True


def test_primary_llm_gate_fails_when_no_provider_is_ready() -> None:
    connectors = {
        "openai": {"configured": True, "state": "DEGRADED"},
        "deepseek": {"configured": False, "state": "NOT_CONFIGURED"},
    }
    assert primary_llm_ready(connectors) is False
