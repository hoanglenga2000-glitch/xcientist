#!/usr/bin/env python3
"""Single production-stability verification entrypoint.

Runs the fast, deterministic gates that must stay green for the workstation to
be considered production-stable, and prints one pass/fail summary. Designed for
CI, pre-commit, or a quick local sanity check.

Gates:
  1. secrets    - scan launch-critical source/config files for plaintext credentials
  2. compile    - byte-compile all first-party Python (src/ + scripts/, minus _quarantine)
  3. imports    - import every module in research_os, research_agent_workstation, and xsci
  4. tests      - run the pytest suite under tests/

Exit code is non-zero if any gate fails.

Usage:
    python scripts/run_ci_checks.py [--skip-tests] [--quiet]
"""
from __future__ import annotations

import argparse
import compileall
import importlib
import io
import os
import pkgutil
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
QUARANTINE = ROOT / "scripts" / "_quarantine"


def _print(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg)


def gate_secrets(quiet: bool) -> tuple[bool, str]:
    scanner = ROOT / "scripts" / "verify_no_plaintext_secrets.py"
    proc = subprocess.run(
        [sys.executable, str(scanner)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode == 0:
        return True, "plaintext-secret scan passed"
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return False, output[:800] or f"secret scanner exited with code {proc.returncode}"


def gate_compile(quiet: bool) -> tuple[bool, str]:
    targets = [SRC, ROOT / "scripts"]
    buf = io.StringIO()
    ok = True
    with redirect_stdout(buf):
        for target in targets:
            result = compileall.compile_dir(
                str(target),
                quiet=1,
                rx=__import__("re").compile(r"_quarantine|__pycache__"),
                workers=0,
            )
            ok = ok and bool(result)
    detail = "all first-party Python compiles" if ok else buf.getvalue().strip()[:500]
    return ok, detail


def _walk(package_name: str) -> list[str]:
    pkg = importlib.import_module(package_name)
    names = [package_name]
    for mod in pkgutil.walk_packages(pkg.__path__, package_name + "."):
        names.append(mod.name)
    return names


def gate_imports(quiet: bool) -> tuple[bool, str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    failures: list[str] = []
    total = 0
    for package_name in ("research_os", "research_agent_workstation", "xsci"):
        try:
            names = _walk(package_name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{package_name} (walk): {type(exc).__name__}: {exc}")
            continue
        for name in names:
            total += 1
            try:
                importlib.import_module(name)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if failures:
        return False, f"{len(failures)}/{total} modules failed:\n  " + "\n  ".join(failures[:20])
    return True, f"{total} modules import cleanly"


def _pytest_command() -> tuple[list[str] | None, str]:
    candidates: list[Path] = []
    configured = os.environ.get("PYTEST_PYTHON")
    if configured:
        candidates.append(Path(configured))
    candidates.extend([
        ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
        ROOT / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
    ])
    candidates.append(Path(sys.executable))
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        probe = subprocess.run(
            [str(candidate), "-c", "import pytest"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return [str(candidate), "-m", "pytest"], str(candidate)
    pytest_executable = shutil.which("pytest")
    if pytest_executable:
        return [pytest_executable], pytest_executable
    return None, ""


def gate_tests(quiet: bool) -> tuple[bool, str]:
    pytest_command, runner = _pytest_command()
    if not pytest_command:
        return False, "pytest not available in current Python, project venv, PYTEST_PYTHON, or PATH"
    test_env = os.environ.copy()
    for key in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE", "GIT_PREFIX"):
        test_env.pop(key, None)
    existing_pythonpath = test_env.get("PYTHONPATH", "")
    test_env["PYTHONPATH"] = str(SRC) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    # Count collected tests up front; some pytest builds omit the summary line
    # when stdout is piped, so we don't rely on parsing it.
    collect = subprocess.run(
        [*pytest_command, "--collect-only", "-q"],
        cwd=str(ROOT),
        env=test_env,
        capture_output=True,
        text=True,
    )
    total = 0
    matched = False
    for line in (collect.stdout or "").splitlines():
        stripped = line.strip()
        # Format A: "tests/test_x.py: 14"  (per-file counts)
        if ":" in stripped and stripped.rsplit(":", 1)[-1].strip().isdigit():
            total += int(stripped.rsplit(":", 1)[-1].strip())
            matched = True
        # Format B: "44 tests collected" (single total line)
        elif stripped.split()[:1] and stripped.split()[0].isdigit() and "test" in stripped:
            total = int(stripped.split()[0])
            matched = True
            break
    collected = str(total) if matched else "?"

    proc = subprocess.run(
        [*pytest_command, "-q", "--no-header"],
        cwd=str(ROOT),
        env=test_env,
        capture_output=True,
        text=True,
    )
    ok = proc.returncode == 0
    detail = (
        f"{collected} tests collected via {runner}, "
        f"suite {'green' if ok else 'RED (exit ' + str(proc.returncode) + ')'}"
    )
    if not ok:
        output = (proc.stdout or "") + (proc.stderr or "")
        failing = [ln.strip() for ln in output.splitlines() if ln.strip().startswith("FAILED")]
        if failing:
            detail += "\n  " + "\n  ".join(failing[:10])
    return ok, detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true", help="skip the pytest gate")
    parser.add_argument("--quiet", action="store_true", help="reduce output")
    args = parser.parse_args()

    gates = [("secrets", gate_secrets), ("compile", gate_compile), ("imports", gate_imports)]
    if not args.skip_tests:
        gates.append(("tests", gate_tests))

    _print("=" * 64, args.quiet)
    _print("Research Workstation - Production Stability Check", args.quiet)
    _print("=" * 64, args.quiet)

    results = []
    for name, fn in gates:
        start = time.time()
        ok, detail = fn(args.quiet)
        elapsed = time.time() - start
        status = "PASS" if ok else "FAIL"
        results.append((name, ok, detail, elapsed))
        _print(f"[{status}] {name:<8} ({elapsed:5.2f}s)  {detail}", args.quiet)
        if args.quiet and not ok:
            print(f"[FAIL] {name}: {detail}")

    all_ok = all(ok for _, ok, _, _ in results)
    _print("=" * 64, args.quiet)
    summary = "ALL GATES PASSED" if all_ok else "STABILITY CHECK FAILED"
    print(f"{summary}  ({sum(1 for _, ok, _, _ in results if ok)}/{len(results)} gates green)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
