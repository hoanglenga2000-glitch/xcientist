#!/usr/bin/env python3
"""Fail when active workstation UI bypasses the semantic theme contract."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "research-agent-workstation"
COMPONENTS = WEB / "src" / "components"
GLOBALS = WEB / "src" / "app" / "globals.css"
TAILWIND = WEB / "tailwind.config.ts"

RAW_PALETTE = re.compile(
    r"(?:bg|text|border|ring|fill|stroke|from|via|to)-(?:slate|gray|zinc|neutral|stone|blue|sky|cyan|teal|emerald|green|lime|yellow|amber|orange|red|rose|pink|fuchsia|purple|violet|indigo)"
    r"(?:-[0-9]+)?(?:/\[[^]]+\]|/[0-9]+)?"
)
RAW_HEX = re.compile(r"#[0-9A-Fa-f]{3,8}")
REQUIRED_VARIABLES = {
    "--color-canvas",
    "--color-surface",
    "--color-surface-raised",
    "--color-ink",
    "--color-ink-secondary",
    "--color-ink-muted",
    "--color-edge",
    "--color-accent",
    "--color-accent-foreground",
    "--color-success",
    "--color-warning",
    "--color-danger",
}


def active_component_files() -> list[Path]:
    return [
        path
        for path in COMPONENTS.rglob("*.tsx")
        if ".bak" not in path.name and path.name != "Screens.tsx"
    ]


def verify() -> dict[str, object]:
    failures: list[str] = []
    files = active_component_files()
    for path in files:
        text = path.read_text(encoding="utf-8")
        palette_hits = sorted(set(RAW_PALETTE.findall(text)))
        if palette_hits:
            failures.append(f"{path.relative_to(ROOT)} raw palette: {', '.join(palette_hits)}")
        hex_hits = sorted(set(RAW_HEX.findall(text)))
        if hex_hits:
            failures.append(f"{path.relative_to(ROOT)} raw hex: {', '.join(hex_hits)}")

    globals_text = GLOBALS.read_text(encoding="utf-8")
    missing_variables = sorted(REQUIRED_VARIABLES - set(re.findall(r"--[a-z0-9-]+(?=\s*:)", globals_text)))
    if missing_variables:
        failures.append(f"globals.css missing variables: {', '.join(missing_variables)}")
    for selector in (':root', 'html[data-theme="light"]'):
        if selector not in globals_text:
            failures.append(f"globals.css missing theme selector: {selector}")

    tailwind_text = TAILWIND.read_text(encoding="utf-8")
    for contract in ("--color-accent-foreground", 'fg: "rgb(var(--color-accent-foreground)'):
        if contract not in tailwind_text:
            failures.append(f"tailwind.config.ts missing contract: {contract}")

    return {
        "status": "passed" if not failures else "failed",
        "active_component_files": len(files),
        "required_variables": len(REQUIRED_VARIABLES),
        "failures": failures,
    }


def main() -> int:
    result = verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
