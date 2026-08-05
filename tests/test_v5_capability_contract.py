from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "video-production" / "evomind-v5" / "verify_v5_capability_contract.py"


def test_complete_v5_capability_contract_is_present() -> None:
    spec = importlib.util.spec_from_file_location("verify_v5_capability_contract", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.verify(ROOT)

    assert result["status"] == "passed", result["blockers"]
    assert result["passed"] == result["total"] == 18
