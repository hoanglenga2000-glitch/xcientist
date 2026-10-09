"""Run bounded local acceptance without publishing credentials or raw tracebacks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts"
TESTS = (
    "tests/test_approval_recovery.py",
    "tests/test_invitation_build_inputs.py",
    "tests/test_assistant_history.py",
    "tests/test_managed_tensor_training.py",
    "tests/test_invitation_projects.py",
    "tests/test_invitation_release_transaction.py",
    "tests/test_invitation_ownership_migration.py",
    "tests/test_invitation_run_requests.py",
    "tests/test_invitation_server_harness.py",
    "tests/test_invitation_release_packaging.py",
    "tests/test_invitation_cancel_contract.py",
    "tests/test_invitation_release_repairs.py",
    "tests/test_invitation_report_safety.py",
    "tests/test_invitation_tenant_access.py",
    "tests/test_invitation_execution_progress.py",
    "tests/test_invitation_aibuild.py",
    "tests/test_invitation_budget_queue.py",
    "tests/test_hpc_execute_solution_workspace.py",
    "tests/test_evomind_runtime_http_server.py",
    "tests/test_goal_board_http.py",
    "tests/test_evomind_runtime.py",
    "tests/test_assistant_run_service.py",
)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _identifier(value: str, fallback: str) -> str:
    base = value.split("[", 1)[0]
    base = re.sub(r"[^A-Za-z0-9_./:-]", "_", base)[:160] or fallback
    return base + ("[" + hashlib.sha256(value.encode()).hexdigest()[:12] + "]" if "[" in value else "")


def sanitize_junit(source: Path, destination: Path) -> dict:
    if source.resolve() == destination.resolve():
        raise ValueError("acceptance_source_overwrite_rejected")
    if destination.exists():
        raise ValueError("acceptance_destination_exists")
    if source.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("acceptance_report_too_large")
    raw = source.read_bytes()
    original = ET.fromstring(raw)
    sanitized = ET.Element("testsuites")
    suite = ET.SubElement(sanitized, "testsuite", {"name": "invitation-local-acceptance"})
    counts = {"test_cases": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    failures = []
    for item in original.iter("testcase"):
        counts["test_cases"] += 1
        name = _identifier(item.get("name", "case"), "case")
        attributes = {"name": name, "classname": _identifier(item.get("classname", "suite"), "suite")}
        duration = item.get("time", "0")
        attributes["time"] = duration if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", duration) else "0"
        case = ET.SubElement(suite, "testcase", attributes)
        found = False
        for tag, counter in (("failure", "failed"), ("error", "errors"), ("skipped", "skipped")):
            for failure in item.findall(tag):
                found = True
                counts[counter] += 1
                kind = failure.get("type", "pytest_" + tag)
                if not re.fullmatch(r"[A-Za-z_.]{1,80}", kind):
                    kind = "pytest_" + tag
                # Never copy a message, traceback, environment repr or log body.
                ET.SubElement(case, tag, {"type": kind, "message": "Raw diagnostic content withheld by acceptance report policy."})
                if tag != "skipped":
                    failures.append({"test": name, "category": tag, "type": kind})
        if not found:
            counts["passed"] += 1
    suite.set("tests", str(counts["test_cases"]))
    suite.set("failures", str(counts["failed"]))
    suite.set("errors", str(counts["errors"]))
    suite.set("skipped", str(counts["skipped"]))
    payload = ET.tostring(sanitized, encoding="utf-8", xml_declaration=True)
    _atomic_write(destination, payload)
    return {
        **counts, "failures": failures,
        "source_report_sha256": hashlib.sha256(raw).hexdigest(),
        "sanitized_report_sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items()
        if not re.search(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTHORIZATION", key, re.I)
        and not key.upper().startswith(("OPENAI_", "ANTHROPIC_", "DEEPSEEK_", "DASHSCOPE_"))
    }
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "PYTEST_ADDOPTS"):
        environment.pop(name, None)
    environment["NO_PROXY"] = "*"
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)))
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir")
    args = parser.parse_args()
    output = (ROOT / args.report_dir).resolve() if args.report_dir else ARTIFACT_ROOT / "invitation-acceptance" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:12])
    output.relative_to(ARTIFACT_ROOT.resolve())
    output.mkdir(parents=True, exist_ok=False)
    prior = []
    for name in ("release-repairs-junit.xml", "tenant-access-local-junit.xml"):
        path = ARTIFACT_ROOT / "invitation-beta-20260905" / name
        if path.is_file():
            destination = output / "prior-reports" / name
            prior.append({"path": path.relative_to(ROOT).as_posix(), "sanitized_path": destination.relative_to(ROOT).as_posix(), **sanitize_junit(path, destination)})
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="evomind-invitation-acceptance-") as temporary:
        raw_report = Path(temporary) / "private-junit.xml"
        command = [
            sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "--tb=short",
            "-W", "error::pytest.PytestUnhandledThreadExceptionWarning",
            "-p", "invitation_offline_plugin", *TESTS, "--junitxml=" + str(raw_report),
        ]
        try:
            process = subprocess.run(command, cwd=ROOT, env=test_environment(), capture_output=True, timeout=240)
            evidence = sanitize_junit(raw_report, output / "backend.junit.xml") if raw_report.is_file() else {"errors": 1, "failed": 0, "test_cases": 0, "failures": [{"type": "report_missing"}]}
            ok = process.returncode == 0 and evidence["errors"] == 0 and evidence["failed"] == 0 and evidence["test_cases"] > 0
            result = {
                "schema": "evomind.invitation_local_acceptance.v1",
                "status": "passed" if ok else "failed", "exit_code": process.returncode,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "network_scope": "isolated_loopback_fixtures_only", "ambient_credentials_removed": True,
                "raw_tracebacks_published": False, "tests": evidence, "prior_reports_sanitized": prior,
                "deployment_performed": False, "browser_e2e": "not_executed", "hpc_training": "not_executed",
            }
        except subprocess.TimeoutExpired:
            result = {"schema": "evomind.invitation_local_acceptance.v1", "status": "failed", "error": "bounded_test_timeout", "raw_tracebacks_published": False, "deployment_performed": False}
    _atomic_write(output / "acceptance-summary.json", (json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
