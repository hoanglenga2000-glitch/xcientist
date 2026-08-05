from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_secure_mlebench_gpt56_adaptive_loop.ps1"


def test_secure_launcher_uses_dpapi_loopback_and_clears_environment() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "Import-Clixml" in source
    assert 'http://127.0.0.1:65068/v1' in source
    assert 'gpt-5.6-sol' in source
    assert 'run_mlebench_gpt56_adaptive_loop.py' in source
    assert 'Remove-Item Env:OPENAI_API_KEY' in source
    assert '$env:EVOLUTION_PRIMARY_PROVIDER = "openai"' in source
    assert '$env:EVOLUTION_PROVIDER_STRICT = "1"' in source
    assert 'Remove-Item Env:EVOLUTION_PRIMARY_PROVIDER' in source
    assert 'ControllerArgs' in source
    assert "sk-" not in source
