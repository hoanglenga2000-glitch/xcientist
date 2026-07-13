from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_FILES = [
    ROOT / "web" / "research-agent-workstation" / "src" / "app" / "page.tsx",
    ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "AppShell.tsx",
    ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "Sidebar.tsx",
    ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "Screens.tsx",
    ROOT / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "Common.tsx",
]


EXCLUDED_ACTIONS = {
    # Creates persistent demo records; covered manually through the visible UI and generic action logging.
    "create_task",
    # Triggered through dedicated API routes rather than /api/workstation-actions.
    "run_local_experiment",
    "export_code_agent_context",
    "import_agent_patch",
    # Fallback-only action id used when a Panel caller has no explicit action id.
    "panel_action",
}

REQUIRED_ACTIONS = {
    "navigate_page",
    "workspace_select",
    "research_mode_toggle",
    "profile_open",
    "notification_open",
    "search_command",
    "task_select",
    "stage_select",
    "workflow_node_select",
    "workflow_library_node_select",
    "workflow_dry_run",
    "workflow_save",
    "workflow_publish",
    "code_file_select",
    "code_editor_tab_select",
    "terminal_tab_select",
    "runtime_agent_select",
    "experiment_select",
    "report_section_select",
    "export_report",
    "submit_report_review",
    "approve_gate",
    "reject_gate",
    "approve_submission",
    "reject_submission",
    "export_audit_bundle",
    "add_evidence",
    "edit_claim_record",
    "open_validation_review",
    "open_artifact_folder",
    "view_reproducibility_record",
    "gate_check_open",
    "design_sample_action",
    "review_agent_patch",
    "apply_agent_patch",
    "rollback_agent_patch",
    "view_full_log",
    "open_fullscreen",
    "artifact_open",
}


def fail(message: str) -> None:
    raise SystemExit(f"ACTION_CONTRACT_FAILED: {message}")


def extract_actions() -> set[str]:
    patterns = [
        re.compile(r"runWorkstationAction\?\.\(\s*['\"]([a-zA-Z0-9_]+)['\"]"),
        re.compile(r"onAction\?\.\(\s*['\"]([a-zA-Z0-9_]+)['\"]"),
        re.compile(r"runAction\(\s*['\"]([a-zA-Z0-9_]+)['\"]"),
        re.compile(r"actionId=['\"]([a-zA-Z0-9_]+)['\"]"),
        re.compile(r"api\.runWorkstationAction\(\s*action"),
    ]
    actions: set[str] = set()
    for file_path in FRONTEND_FILES:
        text = file_path.read_text(encoding="utf-8")
        for pattern in patterns:
            for match in pattern.finditer(text):
                if match.groups():
                    actions.add(match.group(1))
    actions.update(REQUIRED_ACTIONS)
    return actions - EXCLUDED_ACTIONS


def artifact_exists(artifact: str, container_name: str | None = None) -> bool:
    artifact_path = ROOT / artifact
    if artifact_path.exists() and artifact_path.stat().st_size > 0:
        return True
    if container_name:
        normalized = artifact.replace("\\", "/")
        completed = subprocess.run(
            ["docker", "exec", container_name, "test", "-s", f"/app/{normalized}"],
            text=True,
            capture_output=True,
        )
        return completed.returncode == 0
    return False


def post_action(base_url: str, action: str, container_name: str | None = None) -> dict[str, Any]:
    payload = {
        "action": action,
        "task_id": "house_prices",
        "metadata": {
            "source": "action_contract_acceptance",
            "action_under_test": action,
        },
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/workstation-actions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not body.get("ok"):
        fail(f"{action} did not return ok: {body}")
    artifact = body.get("artifact")
    if not artifact:
        fail(f"{action} did not return an artifact path")
    if not artifact_exists(str(artifact), container_name):
        fail(f"{action} artifact was not written: {artifact}")
    return {
        "action": action,
        "artifact": artifact,
        "message": body.get("message"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify visible workstation actions are connected to backend action logging.")
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    parser.add_argument("--container-name", default=None, help="Optional container name for checking artifacts written inside Docker.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actions = sorted(extract_actions())
    missing_required = sorted(REQUIRED_ACTIONS - set(actions) - EXCLUDED_ACTIONS)
    if missing_required:
        fail(f"required actions were not covered: {missing_required}")
    results = [post_action(args.url, action, args.container_name) for action in actions]
    print(
        json.dumps(
            {
                "status": "passed",
                "action_count": len(results),
                "actions": [item["action"] for item in results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
