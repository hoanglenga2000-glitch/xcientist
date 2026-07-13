"""Import smoke tests: every first-party module must import without side effects.

This is the cheapest guard against the kind of breakage that previously shipped
(syntax errors / missing deps in gate-critical scripts). If a module cannot be
imported, the whole pipeline that depends on it is dead.
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest


def _walk(package_name: str):
    pkg = importlib.import_module(package_name)
    names = [package_name]
    for mod in pkgutil.walk_packages(pkg.__path__, package_name + "."):
        names.append(mod.name)
    return names


@pytest.mark.parametrize("package_name", ["research_os", "research_agent_workstation"])
def test_package_imports_cleanly(package_name):
    failures = []
    for name in _walk(package_name):
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - we want to report every failure
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "modules failed to import:\n" + "\n".join(failures)
