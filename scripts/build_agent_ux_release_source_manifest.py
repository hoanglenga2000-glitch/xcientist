"""Build the Agent UX release-source scope manifest.

This read-only manifest records release-critical source paths, Git tracking
state, SHA-256, and review intent so a formal release commit cannot omit the
browser Agent UX, live smoke, or gate code. It does not stage, commit, tag,
train, submit to Kaggle, or call any grader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import verify_local_evomind_release_gate as gate  # noqa: E402

PURPOSE_BY_PATH = {
    "scripts/verify_local_evomind_release_gate.py": "Aggregates local demo and formal source gates for Agent UX release readiness.",
    "scripts/verify_live_assistant_demo_smoke.py": "Runs authenticated four-turn browser-equivalent Agent smoke, including job90673 connection self-check.",
    "tests/test_verify_local_evomind_release_gate.py": "Prevents release gate source-tracking regressions.",
    "tests/test_verify_live_assistant_demo_smoke.py": "Prevents job90673 connection smoke and side-effect-boundary regressions.",
    "src/xsci/assistant_behavior_distillation.py": "Defines novice-facing behavior contracts, visible response audit, and deterministic repairs.",
    "src/xsci/kaggle_conversation.py": "Implements browser/terminal Agent routing, LLM-first tool loop, safe context, and fallback behavior.",
    "scripts/verify_new_user_release_readiness.py": "Verifies new-user local gateway readiness and novice quality gate freshness.",
    "scripts/verify_backend_resource_status.py": "Checks the canonical Connector Registry state and raw connector provenance.",
    "tests/test_backend_resource_status.py": "Prevents canonical connector-state and authoritative GPU-gate regressions.",
    "scripts/manage_hpc_proxy_bridge.ps1": "Starts the local HPC SOCKS bridge through the designated upstream proxy by default.",
    "tests/test_hpc_connection_memory_core.py": "Prevents direct-route defaults and HPC proxy/container identity contract regressions.",
    "scripts/manage_workstation_dashboard.py": "Binds the novice-facing dashboard to the installed interactive OpenAI performance profile.",
    "tests/test_dashboard_build_transaction.py": "Prevents dashboard lifecycle, build transaction, and interactive gateway profile regressions.",
    "web/research-agent-workstation/src/app/api/assistant/stream/route.ts": "Authenticates and streams browser assistant requests through the Python Agent subprocess.",
    "web/research-agent-workstation/src/components/workstation/screens/AssistantScreen.tsx": "Browser-facing chat UI, stream error handling, and visible Agent status.",
    "web/research-agent-workstation/src/lib/server/scientific-report.ts": "Projects reviewed Run evidence into the Report Studio without bypassing review or hash gates.",
    "web/research-agent-workstation/src/lib/server/reviewed-existing-report.ts": "Fails closed unless a completed Run has task binding, Reviewer, Claim Audit, and artifact provenance.",
    "web/research-agent-workstation/src/lib/server/reviewed-existing-report.test.ts": "Prevents reviewed-report task, gate, and manifest binding regressions.",
}

SECRET_PATTERNS = (
    ("openai_or_anthropic_key", re.compile(r"\bsk-(?:ant|proj|live|test|ya)?[A-Za-z0-9_-]{16,}\b")),
    ("kaggle_token", re.compile(r"\bKGAT_[A-Za-z0-9_-]{16,}\b")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("literal_password_assignment", re.compile(r"(?i)\bpassword\s*[:=]\s*['\"](?!<|\\[REDACTED\\]|REDACTED)[^'\"]{6,}['\"]")),
    ("literal_secret_assignment", re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret)\s*[:=]\s*['\"][A-Za-z0-9_./+=-]{16,}['\"]")),
    ("sshpass_literal", re.compile(r"(?i)\bsshpass\s+-p\s+(?!\\$|%|<)[^\s]+")),
)

ALLOWED_SECRET_PLACEHOLDERS = {
    "<redacted-password>",
    "<your-api-key>",
    "[REDACTED]",
    "REDACTED",
}


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(args: list[str], *, root: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )


def repo_relative(path: Path, *, root: Path = ROOT) -> str:
    return path.relative_to(root).as_posix()


def parse_git_status(stdout: str) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for line in stdout.splitlines():
        if len(line) < 4:
            continue
        rel = line[3:].strip().strip('"').replace("\\", "/")
        if " -> " in rel:
            rel = rel.split(" -> ", 1)[-1]
        parsed.setdefault(rel, []).append(line[:2])
    return parsed


def tracked_paths_from_ls_files(stdout: str) -> set[str]:
    tracked: set[str] = set()
    for line in stdout.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) == 4:
            tracked.add(parts[3].replace("\\", "/"))
    return tracked


def required_action(*, exists: bool, tracked: bool, dirty: bool) -> str:
    if not exists:
        return "restore_or_remove_from_release_scope"
    if not tracked:
        return "review_and_git_add_before_release_commit"
    if dirty:
        return "review_and_commit_or_revert_before_strict_gate"
    return "already_tracked_clean"


def scan_text_for_secrets(text: str, *, rel_path: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if any(placeholder in line for placeholder in ALLOWED_SECRET_PLACEHOLDERS):
            continue
        for pattern_id, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(
                    {
                        "path": rel_path,
                        "line": line_no,
                        "pattern_id": pattern_id,
                    }
                )
    return findings


def scan_file_for_secrets(path: Path, *, rel_path: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return [{"path": rel_path, "line": None, "pattern_id": "unreadable_text"}]
    return scan_text_for_secrets(text, rel_path=rel_path)


def build_manifest(*, root: Path = ROOT) -> dict[str, Any]:
    source_paths = list(gate.RELEASE_RELEVANT_SOURCE_FILES)
    rel_paths = [repo_relative(path, root=root) for path in source_paths]
    ls_proc = run_git(["ls-files", "--stage", "--", *rel_paths], root=root)
    status_proc = run_git(["status", "--porcelain=v1", "--untracked-files=all", "--", *rel_paths], root=root)
    tracked = tracked_paths_from_ls_files(ls_proc.stdout) if ls_proc.returncode == 0 else set()
    porcelain = parse_git_status(status_proc.stdout) if status_proc.returncode == 0 else {}

    files: list[dict[str, Any]] = []
    for path, rel in zip(source_paths, rel_paths, strict=True):
        exists = path.exists()
        is_tracked = rel in tracked
        git_status = porcelain.get(rel, [])
        dirty = is_tracked and any(code.strip() for code in git_status)
        secret_findings = scan_file_for_secrets(path, rel_path=rel)
        files.append(
            {
                "path": rel,
                "purpose": PURPOSE_BY_PATH.get(rel, "Release-critical Agent UX source file."),
                "exists": exists,
                "tracked": is_tracked,
                "git_status": git_status,
                "dirty": dirty,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size if exists else None,
                "secret_findings": secret_findings,
                "required_action": required_action(exists=exists, tracked=is_tracked, dirty=dirty),
            }
        )

    untracked = [item["path"] for item in files if item["exists"] and not item["tracked"]]
    dirty_tracked = [item["path"] for item in files if item["tracked"] and item["dirty"]]
    missing = [item["path"] for item in files if not item["exists"]]
    secret_findings = [
        finding
        for item in files
        for finding in list(item.get("secret_findings") or [])
    ]
    return {
        "schema": "evomind.agent_ux_release_source_scope.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "goal_reference": "019fa4da-c818-7642-8541-76e31d6b5886",
        "status": "clean" if not (missing or untracked or dirty_tracked) else "needs_release_source_review",
        "claim_boundary": "Source-scope audit only; no training, Kaggle submit, private grader, commit, tag, or push.",
        "summary": {
            "file_count": len(files),
            "missing_count": len(missing),
            "untracked_count": len(untracked),
            "dirty_tracked_count": len(dirty_tracked),
            "secret_finding_count": len(secret_findings),
        },
        "secret_scan": {
            "status": "passed" if not secret_findings else "failed",
            "scanned_files": sum(1 for item in files if item.get("exists")),
            "finding_count": len(secret_findings),
            "findings": secret_findings[:50],
        },
        "missing": missing,
        "untracked": untracked,
        "dirty_tracked": dirty_tracked,
        "files": files,
        "git": {
            "ls_files_returncode": ls_proc.returncode,
            "status_returncode": status_proc.returncode,
            "status_porcelain": status_proc.stdout.splitlines(),
            "stderr_tail": (ls_proc.stderr + status_proc.stderr)[-2000:],
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# EvoMind Agent UX Release Source Scope",
        "",
        f"- status: {report.get('status')}",
        f"- generated_at: {report.get('generated_at')}",
        f"- claim_boundary: {report.get('claim_boundary')}",
        "",
        "## Blockers",
        "",
        f"- missing: {', '.join(report.get('missing') or []) or 'none'}",
        f"- untracked: {', '.join(report.get('untracked') or []) or 'none'}",
        f"- dirty_tracked: {', '.join(report.get('dirty_tracked') or []) or 'none'}",
        f"- secret_scan: {(report.get('secret_scan') or {}).get('status')}",
        "",
        "## Files",
        "",
        "| Path | Tracked | Dirty | Required Action | Purpose |",
        "|---|---:|---:|---|---|",
    ]
    for item in report.get("files") or []:
        lines.append(
            "| {path} | {tracked} | {dirty} | {action} | {purpose} |".format(
                path=item.get("path"),
                tracked="yes" if item.get("tracked") else "no",
                dirty="yes" if item.get("dirty") else "no",
                action=item.get("required_action"),
                purpose=str(item.get("purpose") or "").replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "workspace/evaluation/agent_ux_release_source_scope_current.json")
    parser.add_argument("--markdown-output", type=Path, default=ROOT / "workspace/evaluation/agent_ux_release_source_scope_current.md")
    args = parser.parse_args()

    report = build_manifest(root=ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, args.markdown_output)
    print(json.dumps({"json": str(args.output), "markdown": str(args.markdown_output), "status": report["status"], "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "clean" else 1


if __name__ == "__main__":
    raise SystemExit(main())
