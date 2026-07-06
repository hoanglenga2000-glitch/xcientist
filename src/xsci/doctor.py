"""`xsci doctor` — self-check the environment before running research.

Checks, in order of what blocks a run: Python version, core deps importable,
the research_os engine importable, LLM credentials present, Kaggle credentials
present, and (if compute=gpu) whether the GPU SSH config resolves. Prints a
PASS/WARN/FAIL line per check and returns non-zero only on hard FAILs so it is
CI-friendly.
"""
from __future__ import annotations

import importlib
import sys
from typing import Callable

from .config import load_config

OK, WARN, FAIL = "PASS", "WARN", "FAIL"


def _check_python() -> tuple[str, str]:
    v = sys.version_info
    if v < (3, 10):
        return FAIL, f"Python {v.major}.{v.minor} - need >=3.10"
    return OK, f"Python {v.major}.{v.minor}.{v.micro}"


def _check_import(mod: str) -> tuple[str, str]:
    try:
        importlib.import_module(mod)
        return OK, f"{mod} importable"
    except Exception as exc:  # noqa: BLE001
        return FAIL, f"{mod} import failed: {type(exc).__name__}: {exc}"


def _check_engine() -> tuple[str, str]:
    try:
        from research_os.evolution_loop import EvolutionLoop  # noqa: F401
        from research_os.mcgs_selector import MCGSSelector  # noqa: F401
        return OK, "research_os engine importable (EvolutionLoop + MCGSSelector)"
    except Exception as exc:  # noqa: BLE001
        return FAIL, f"research_os engine import failed: {type(exc).__name__}: {exc}"


def _check_llm(cfg) -> tuple[str, str]:
    provider = cfg.get("llm.provider", "anthropic")
    key = cfg.get(f"secrets.{provider}_api_key") or cfg.get("secrets.anthropic_api_key")
    if key:
        return OK, f"LLM key present (provider={provider})"
    return WARN, "no LLM key - run `xsci login` (phase 2) or set ANTHROPIC_API_KEY"


def _check_kaggle(cfg) -> tuple[str, str]:
    if cfg.get("secrets.kaggle_key") and cfg.get("secrets.kaggle_username"):
        return OK, "Kaggle credentials present"
    return WARN, "no Kaggle credentials - needed to fetch competitions"


def _check_compute(cfg) -> tuple[str, str]:
    backend = cfg.get("compute.backend", "local")
    if backend != "gpu":
        return OK, f"compute backend = {backend} (no remote needed)"
    # GPU selected: verify the SSH config resolves WITHOUT connecting.
    try:
        from research_agent_workstation.server.core.gpu_credentials import (
            load_gpu_ssh_config,
        )
        conf = load_gpu_ssh_config(require_auth=True)
        return OK, f"GPU SSH config resolves (host set, auth={conf.has_auth()})"
    except Exception as exc:  # noqa: BLE001
        return FAIL, f"compute=gpu but SSH config invalid: {exc}"


def run_doctor() -> int:
    cfg = load_config()
    checks: list[tuple[str, Callable[[], tuple[str, str]]]] = [
        ("python", _check_python),
        ("deps: pandas", lambda: _check_import("pandas")),
        ("deps: sklearn", lambda: _check_import("sklearn")),
        ("engine", _check_engine),
        ("llm", lambda: _check_llm(cfg)),
        ("kaggle", lambda: _check_kaggle(cfg)),
        ("compute", lambda: _check_compute(cfg)),
    ]
    symbols = {OK: "[+]", WARN: "[!]", FAIL: "[x]"}
    n_fail = n_warn = 0
    print("xsci doctor - environment self-check\n")
    for name, fn in checks:
        try:
            status, detail = fn()
        except Exception as exc:  # noqa: BLE001 - a check must never crash doctor
            status, detail = FAIL, f"check raised {type(exc).__name__}: {exc}"
        n_fail += status == FAIL
        n_warn += status == WARN
        print(f"  {symbols[status]} {name:16s} {detail}")
    print()
    if n_fail:
        print(f"{n_fail} blocking issue(s). Fix the [x] items before running.")
        return 1
    if n_warn:
        print(f"Ready with {n_warn} warning(s) - some features need setup.")
        return 0
    print("All checks passed. Ready to run research.")
    return 0
