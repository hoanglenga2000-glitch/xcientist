from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("startup_mode", ["script_path", "module_import"])
def test_benchmark_registry_gate_supports_both_ci_startup_modes(
    tmp_path: Path,
    startup_mode: str,
):
    if startup_mode == "script_path":
        loader = (
            "import importlib.util; "
            f"spec=importlib.util.spec_from_file_location('standalone_ci', {str(ROOT / 'scripts' / 'run_ci_checks.py')!r}); "
            "module=importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module); "
        )
    else:
        loader = (
            f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
            "import scripts.run_ci_checks as module; "
        )
    code = loader + (
        "ok, detail = module.gate_benchmark_registry(True); "
        "assert ok, detail; "
        "assert 'pinned official split=75' in detail, detail"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
