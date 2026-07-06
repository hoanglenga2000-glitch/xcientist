from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_JSON = ROOT / "web" / "research-agent-workstation" / "package.json"
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE = ROOT / "docker-compose.yml"
ACTION_CONTRACT = ROOT / "scripts" / "verify_workstation_action_contract.py"
RESTART_PS1 = ROOT / "scripts" / "restart_workstation_frontend.ps1"
MANAGER = ROOT / "scripts" / "manage_workstation_dashboard.py"
CLAUDE_HANDOFF = ROOT / "docs" / "ClaudeCode接手科研Agent工作站任务书.md"
CLAUDE_SKILL = ROOT / "docs" / "claude-code-research-agent-workstation-skill" / "SKILL.md"


def fail(message: str, evidence: dict | None = None) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence or {}}, ensure_ascii=False, indent=2))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    package = json.loads(read(PACKAGE_JSON))
    scripts = package.get("scripts", {})
    if "--port 8088" not in scripts.get("dev", ""):
        fail("npm dev script must use local workstation port 8088", {"dev": scripts.get("dev")})
    if "--port 8088" not in scripts.get("start", ""):
        fail("npm start script must use local workstation port 8088", {"start": scripts.get("start")})
    if "--port 3090" not in scripts.get("start:docker-port", ""):
        fail("package must keep an explicit Docker internal 3090 start script", {"start:docker-port": scripts.get("start:docker-port")})

    compose = yaml.safe_load(read(COMPOSE))
    ports = compose["services"]["research-agent-workstation"].get("ports", [])
    if "8088:3090" not in ports:
        fail("docker compose must expose user-facing 8088 to internal 3090", {"ports": ports})

    dockerfile = read(DOCKERFILE)
    if "--port 3090" not in dockerfile:
        fail("Dockerfile must keep internal Next port 3090", {"dockerfile": "missing --port 3090"})

    action_contract = read(ACTION_CONTRACT)
    if 'default="http://127.0.0.1:8088"' not in action_contract:
        fail("action contract default URL must be 8088", {"file": str(ACTION_CONTRACT.relative_to(ROOT))})

    restart_ps1 = read(RESTART_PS1)
    if "[int]$Port = 8088" not in restart_ps1:
        fail("PowerShell restart helper default port must be 8088", {"file": str(RESTART_PS1.relative_to(ROOT))})

    manager = read(MANAGER)
    if "default=8088" not in manager:
        fail("dashboard manager default port must be 8088", {"file": str(MANAGER.relative_to(ROOT))})

    doc_issues = []
    for path in [CLAUDE_HANDOFF, CLAUDE_SKILL]:
        text = read(path)
        if "http://127.0.0.1:3090" in text or "port `3090`" in text:
            doc_issues.append(str(path.relative_to(ROOT)))
    if doc_issues:
        fail("current Claude handoff docs still reference old local port 3090", {"files": doc_issues})

    print(
        json.dumps(
            {
                "status": "passed",
                "local_port": 8088,
                "docker_internal_port": 3090,
                "package_scripts": {
                    "dev": scripts.get("dev"),
                    "start": scripts.get("start"),
                    "start:docker-port": scripts.get("start:docker-port"),
                },
                "compose_ports": ports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
