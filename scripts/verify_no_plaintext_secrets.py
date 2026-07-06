from __future__ import annotations

import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [
    ROOT / "README.md",
    ROOT / "CLAUDE.md",
    ROOT / "TEACHER_MEDAL_RATE_DASHBOARD.md",
    ROOT / ".env.example",
    ROOT / "configs",
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "prompts",
    ROOT / "benchmark",
    ROOT / "reports" / "templates",
    ROOT / "web" / "research-agent-workstation" / "src",
    ROOT / "web" / "research-agent-workstation" / "package.json",
    ROOT / "web" / "research-agent-workstation" / "next.config.mjs",
    ROOT / "web" / "research-agent-workstation" / "tailwind.config.ts",
    ROOT / "web" / "research-agent-workstation" / ".env.example",
    ROOT / "web" / "research-agent-workstation" / "CLAUDE.md",
    ROOT / "web" / "research-agent-workstation" / "prisma" / "schema.prisma",
]
SKIP_DIRS = {
    ".next",
    "node_modules",
    "open-reverselab",
    "__pycache__",
    ".git",
    ".runtime-logs",
    ".pytest_cache",
    ".mypy_cache",
    "pip-cache",
    ".venv",
    "venv",
    "ui-editable-capture-20260627",
    "ui-verification-20260626-fidelity-overlay-v21-waited",
}
SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".docx",
    ".pdf",
    ".db",
    ".sqlite",
    ".pyc",
    ".zip",
}

SECRET_PATTERNS = [
    re.compile(r"\bsk-(?:ant|proj|live|test|ya)[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bKGAT_[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:ANTHROPIC_API_KEY|OPENAI_API_KEY|KAGGLE_KEY|KAGGLE_API_TOKEN)\s*=\s*['\"]?[A-Za-z0-9_-]{16,}"),
]

ALLOWED_PLACEHOLDERS = {
    "<rotated-anthropic-key>",
    "<your-api-key>",
}


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    return path.suffix.lower() in SKIP_SUFFIXES


def main() -> None:
    findings: list[dict[str, object]] = []
    candidate_files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        if root.is_file():
            candidate_files.append(root)
            continue
        for current_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [dirname for dirname in dirnames if dirname not in SKIP_DIRS]
            for filename in filenames:
                candidate_files.append(Path(current_root) / filename)

    for path in candidate_files:
        if should_skip(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if any(placeholder in line for placeholder in ALLOWED_PLACEHOLDERS):
                continue
            for pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    findings.append({
                        "file": str(path.relative_to(ROOT)),
                        "line": line_no,
                        "pattern": pattern.pattern,
                    })
    if findings:
        raise SystemExit(json.dumps({
            "status": "failed",
            "message": "Potential plaintext secrets found. Rotate/remove before launch.",
            "findings": findings[:50],
            "finding_count": len(findings),
        }, ensure_ascii=False, indent=2))
    print(json.dumps({
        "status": "passed",
        "message": "No plaintext API keys or private keys were found in launch-critical workstation source/config files.",
        "scanned_files": len(candidate_files),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
