from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.yml"
EXPECTED = "research-agent-workstation"
FORBIDDEN = "research-agent-dashboard"

DOC_PATHS = [
    ROOT / "README.md",
    ROOT / "docs" / "科研Agent工作站Docker可视化验证.md",
    ROOT / "docs" / "科研Agent工作站最终完成审计.md",
    ROOT / "docs" / "系统上线验收记录-20260609.md",
    ROOT / "docs" / "真实业务上线验收记录-20260610.md",
    ROOT / "docs" / "正式上线资源接入运行手册.md",
]

SCRIPT_PATHS = [
    ROOT / "scripts" / "run_smoke_tests.py",
    ROOT / "scripts" / "run_real_resource_smoke.py",
    ROOT / "scripts" / "run_full_acceptance.py",
]


def fail(message: str, evidence: dict | None = None) -> None:
    raise SystemExit(json.dumps({"status": "failed", "message": message, "evidence": evidence or {}}, ensure_ascii=False, indent=2))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main() -> None:
    compose = yaml.safe_load(read(COMPOSE))
    services = compose.get("services", {})
    if EXPECTED not in services:
        fail("docker-compose service name is not the expected workstation service", {"services": sorted(services)})
    container_name = services[EXPECTED].get("container_name")
    if container_name != EXPECTED:
        fail("docker-compose container_name is inconsistent", {"container_name": container_name, "expected": EXPECTED})

    stale_docs = []
    for path in DOC_PATHS:
        text = read(path)
        if FORBIDDEN in text:
            stale_docs.append(str(path.relative_to(ROOT)))
    if stale_docs:
        fail("stale Docker service name remains in docs", {"forbidden": FORBIDDEN, "files": stale_docs})

    stale_scripts = []
    for path in SCRIPT_PATHS:
        text = read(path)
        if FORBIDDEN in text:
            stale_scripts.append(str(path.relative_to(ROOT)))
    if stale_scripts:
        fail("stale Docker service name remains in scripts", {"forbidden": FORBIDDEN, "files": stale_scripts})

    run_command_mentions = []
    for path in DOC_PATHS:
        text = read(path)
        if "docker compose" in text and EXPECTED in text:
            run_command_mentions.append(str(path.relative_to(ROOT)))

    print(
        json.dumps(
            {
                "status": "passed",
                "service": EXPECTED,
                "container_name": container_name,
                "checked_docs": [str(path.relative_to(ROOT)) for path in DOC_PATHS if path.exists()],
                "docker_command_docs": run_command_mentions,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
