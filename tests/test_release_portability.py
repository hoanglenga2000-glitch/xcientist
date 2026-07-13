from __future__ import annotations

import re
from pathlib import Path

from scripts.verify_security_invariants import repository_files as security_repository_files

ROOT = Path(__file__).resolve().parents[1]
EXEMPT = {
    "scripts/build_reproducible_submission_package.py",
    "scripts/verify_external_resources_manifest.py",
    "scripts/verify_security_invariants.py",
}
PATTERNS = {
    "personal_hpc_mount": re.compile(r"/hpc2(?:hdd|ssd)/", re.IGNORECASE),
    "personal_hpc_account": re.compile(r"\b" + "aims" + r"lab\b", re.IGNORECASE),
    "personal_workspace_name": re.compile(r"(?:~[/\\])?" + "jing" + "hw", re.IGNORECASE),
    "windows_user_home": re.compile(r"C:\\Users\\[^\\/\r\n]+", re.IGNORECASE),
    "release_validation_path": re.compile("EvoMind-" + "release-validation", re.IGNORECASE),
}
TEXT_SUFFIXES = {
    ".bat", ".cmd", ".js", ".json", ".jsx", ".md", ".mjs", ".ps1", ".py",
    ".sh", ".toml", ".ts", ".tsx", ".txt", ".yaml", ".yml",
}


def test_quarantine_is_recursively_export_ignored() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8-sig")
    rules = {
        line.split("#", 1)[0].strip()
        for line in attributes.splitlines()
        if line.split("#", 1)[0].strip()
    }

    assert {
        "/scripts/_quarantine export-ignore",
        "/scripts/_quarantine/** export-ignore",
    } <= rules


def test_git_inventory_contains_no_machine_specific_release_text() -> None:
    paths, discovery_findings = security_repository_files()
    assert discovery_findings == []

    findings: list[str] = []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if (
            relative in EXEMPT
            or relative.startswith(("tests/", "scripts/_quarantine/"))
            or not path.is_file()
            or path.suffix.casefold() not in TEXT_SUFFIXES
        ):
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for rule, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{relative}:{rule}")

    assert findings == []


def test_security_inventory_uses_filesystem_fallback_without_git(tmp_path: Path) -> None:
    source = tmp_path / "src" / "safe.py"
    source.parent.mkdir()
    source.write_text("value = 1\n", encoding="utf-8")
    vendor = tmp_path / "node_modules" / "ignored.js"
    vendor.parent.mkdir()
    vendor.write_text("ignored = true;\n", encoding="utf-8")

    paths, discovery_findings = security_repository_files(tmp_path)

    assert discovery_findings == []
    assert {path.relative_to(tmp_path).as_posix() for path in paths} == {"src/safe.py"}


def test_verified_launcher_guidance_configures_llm_first() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8-sig")
    quick_start = readme.partition("## Quick Start")[2].partition("## CLI Commands")[0]
    installer = (ROOT / "install.ps1").read_text(encoding="utf-8-sig")
    next_steps = installer.partition('Write-Host "Next steps:"')[2]

    assert quick_start.index("evomind setup") < quick_start.index("start_verified_workstation.ps1")
    assert next_steps.index("evomind setup") < next_steps.index("start_verified_workstation.ps1")
    assert "evomind dashboard start" in quick_start
    assert "evomind dashboard start" in next_steps
