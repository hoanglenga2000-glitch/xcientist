from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_default_workstation_bindings_are_loopback_only():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    service = compose["services"]["research-agent-workstation"]

    assert set(service["ports"]) == {
        "127.0.0.1:3090:3090",
        "127.0.0.1:8088:3090",
    }
    assert service.get("network_mode") != "host"


def test_host_npm_scripts_are_loopback_only():
    package = json.loads(
        (ROOT / "web" / "research-agent-workstation" / "package.json").read_text(encoding="utf-8")
    )

    assert "--hostname 127.0.0.1" in package["scripts"]["dev"]
    assert "--hostname 127.0.0.1" in package["scripts"]["start"]
