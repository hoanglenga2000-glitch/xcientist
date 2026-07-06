from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANAGER = ROOT / "scripts" / "manage_workstation_dashboard.py"
RUNBOOK = ROOT / "docs" / "正式上线资源接入运行手册.md"


def fail(message: str, evidence: dict | None = None) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence or {}}, ensure_ascii=False, indent=2))


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)


def main() -> None:
    if not MANAGER.exists():
        fail("dashboard manager script is missing", {"path": str(MANAGER.relative_to(ROOT))})

    source = MANAGER.read_text(encoding="utf-8")
    required_terms = [
        "start",
        "stop",
        "restart",
        "status",
        "api/workstation-summary",
        "dashboard.pid",
        "dashboard.out.log",
        "NEXT_TELEMETRY_DISABLED",
    ]
    missing = [term for term in required_terms if term not in source]
    if missing:
        fail("dashboard manager is missing required lifecycle features", {"missing_terms": missing})

    help_result = run([sys.executable, str(MANAGER), "--help"])
    if help_result.returncode != 0:
        fail("dashboard manager help command failed", {"stdout": help_result.stdout, "stderr": help_result.stderr})

    status_result = run([sys.executable, str(MANAGER), "status", "--port", "8088"])
    if status_result.returncode != 0:
        fail("dashboard manager status command failed", {"stdout": status_result.stdout, "stderr": status_result.stderr})
    try:
        status_payload = json.loads(status_result.stdout)
    except json.JSONDecodeError as exc:
        fail("dashboard manager status output is not JSON", {"stdout": status_result.stdout, "error": repr(exc)})

    if status_payload.get("status") == "running":
        health = status_payload.get("health") or {}
        if not health.get("has_runtime"):
            fail("running dashboard status did not expose runtime health", {"status": status_payload})

    runbook_text = RUNBOOK.read_text(encoding="utf-8") if RUNBOOK.exists() else ""
    runbook_has_manager = "manage_workstation_dashboard.py" in runbook_text

    print(
        json.dumps(
            {
                "status": "passed",
                "manager": str(MANAGER.relative_to(ROOT)),
                "lifecycle_commands": ["start", "stop", "restart", "status"],
                "runtime_status": status_payload,
                "runbook_mentions_manager": runbook_has_manager,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
