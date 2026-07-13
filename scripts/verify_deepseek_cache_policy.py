from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_DATE = "20260623"
DEEPSEEK_CACHE_TS = Path("web/research-agent-workstation/src/lib/server/deepseek-cache.ts")
CLAUDE_AGENT_SESSIONS_TS = Path("web/research-agent-workstation/src/lib/server/claude-agent-sessions.ts")
DEFAULT_JSON_REPORT = Path(f"workspace/deepseek_cache_policy_verification_{TARGET_DATE}.json")
DEFAULT_MD_REPORT = Path(f"reports/DEEPSEEK_CACHE_POLICY_VERIFICATION_{TARGET_DATE}.md")


HELPER_IMPORT_RE = re.compile(
    r"import\s*\{(?P<names>[^}]+)\}\s*from\s*[\"'](?:@/lib/server/deepseek-cache|\.\/deepseek-cache|\.\.?/.*/deepseek-cache)[\"']",
    re.MULTILINE,
)

HELPER_NAME_CANDIDATES = {
    "createDeepSeekCacheMessages",
    "attachDeepSeekCacheUsage",
    "recordDeepSeekCacheSession",
    "extractDeepSeekCacheUsage",
}

CACHE_METADATA_TOKENS = {
    "cache_metadata",
    "deepseek_cache_metadata",
    "cache_metadata_path",
    "cache_key",
    "prompt_fingerprint",
    "stable_system_hash",
    "stable_user_prefix_hash",
    "dynamic_suffix_hash",
    "cache_hit_ratio",
    "cache_observed",
    "cached_tokens",
}

TRANSCRIPT_CONTEXT_TOKENS = {
    "writeTextArtifact",
    "transcriptRelative",
    "transcript_path",
    "provider: \"deepseek_code_agent\"",
    "provider: 'deepseek_code_agent'",
}

MANIFEST_CONTEXT_TOKENS = {
    "ClaudeSessionRecord",
    "writeRecord",
    "session_manifest",
    "manifest_path",
    "completed:",
    "failed:",
}

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("openai_or_deepseek_style_sk_key", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9][A-Za-z0-9_\-]{20,}")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b")),
    ("literal_bearer_token", re.compile(r"Bearer\s+['\"]?[A-Za-z0-9_\-\.]{24,}['\"]?")),
    (
        "hardcoded_secret_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|pwd|credential|auth[_-]?token)\b\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"
        ),
    ),
]

FALSE_POSITIVE_SNIPPETS = (
    "process.env",
    "${",
    "config.apiKey",
    "apiKeyValue",
    "ApiKeyStatus",
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_API_KEY",
    "missing_env",
    "not configured",
    "Set ",
)


@dataclass
class Check:
    name: str
    status: str
    message: str
    evidence: dict[str, Any]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def line_numbers_for_tokens(text: str, tokens: set[str]) -> dict[str, list[int]]:
    lines = text.splitlines()
    result: dict[str, list[int]] = {}
    lowered_lines = [line.lower() for line in lines]
    for token in sorted(tokens):
        token_lower = token.lower()
        hits = [index + 1 for index, line in enumerate(lowered_lines) if token_lower in line]
        if hits:
            result[token] = hits[:20]
    return result


def find_imported_helper_names(text: str) -> list[str]:
    imported: set[str] = set()
    for match in HELPER_IMPORT_RE.finditer(text):
        raw_names = match.group("names")
        for name in raw_names.split(","):
            imported_name = name.strip().split(" as ")[0].strip()
            if imported_name:
                imported.add(imported_name)
    return sorted(imported)


def find_helper_usage(text: str, imported_names: list[str]) -> dict[str, list[int]]:
    import_spans = [match.span() for match in HELPER_IMPORT_RE.finditer(text)]
    usage: dict[str, list[int]] = {}
    for name in imported_names:
        pattern = re.compile(rf"\b{re.escape(name)}\b")
        hits: list[int] = []
        for match in pattern.finditer(text):
            if any(start <= match.start() < end for start, end in import_spans):
                continue
            hits.append(text.count("\n", 0, match.start()) + 1)
        if hits:
            usage[name] = hits[:20]
    return usage


def relevant_scan_files(root: Path) -> list[Path]:
    candidates = [
        root / DEEPSEEK_CACHE_TS,
        root / CLAUDE_AGENT_SESSIONS_TS,
        root / "scripts/verify_deepseek_cache_policy.py",
    ]
    for base in [root / "web/research-agent-workstation/src/lib/server", root / "scripts"]:
        if not base.exists():
            continue
        for path in base.glob("*deepseek*cache*"):
            if path.is_file():
                candidates.append(path)
        for path in base.glob("*deepseek_cache*"):
            if path.is_file():
                candidates.append(path)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen and path.exists() and path.is_file():
            seen.add(resolved)
            deduped.append(path)
    return deduped


def looks_like_false_positive(line: str, pattern_name: str) -> bool:
    if pattern_name == "literal_bearer_token" and "${" in line:
        return True
    return any(snippet in line for snippet in FALSE_POSITIVE_SNIPPETS)


def scan_for_obvious_secrets(root: Path, files: list[Path]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for path in files:
        text = read_text(path)
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern_name, pattern in SECRET_PATTERNS:
                if not pattern.search(line):
                    continue
                if looks_like_false_positive(line, pattern_name):
                    continue
                findings.append(
                    {
                        "path": rel(path, root),
                        "line": line_number,
                        "pattern": pattern_name,
                        "redaction": "value_not_recorded",
                    }
                )
    return {
        "status": "passed" if not findings else "failed",
        "scanned_files": [rel(path, root) for path in files],
        "finding_count": len(findings),
        "findings": findings,
    }


def evaluate(root: Path) -> dict[str, Any]:
    cache_path = root / DEEPSEEK_CACHE_TS
    sessions_path = root / CLAUDE_AGENT_SESSIONS_TS
    checks: list[Check] = []

    cache_exists = cache_path.exists() and cache_path.is_file()
    checks.append(
        Check(
            name="deepseek_cache_helper_exists",
            status="passed" if cache_exists else "failed",
            message=f"{DEEPSEEK_CACHE_TS.as_posix()} {'exists' if cache_exists else 'is missing'}",
            evidence={"path": DEEPSEEK_CACHE_TS.as_posix()},
        )
    )

    sessions_exists = sessions_path.exists() and sessions_path.is_file()
    checks.append(
        Check(
            name="claude_agent_sessions_exists",
            status="passed" if sessions_exists else "failed",
            message=f"{CLAUDE_AGENT_SESSIONS_TS.as_posix()} {'exists' if sessions_exists else 'is missing'}",
            evidence={"path": CLAUDE_AGENT_SESSIONS_TS.as_posix()},
        )
    )

    cache_text = read_text(cache_path) if cache_exists else ""
    sessions_text = read_text(sessions_path) if sessions_exists else ""

    exported_helpers = sorted(name for name in HELPER_NAME_CANDIDATES if re.search(rf"\bexport\s+(?:async\s+)?function\s+{name}\b", cache_text))
    checks.append(
        Check(
            name="deepseek_cache_helper_exports",
            status="passed" if {"createDeepSeekCacheMessages", "attachDeepSeekCacheUsage"}.issubset(exported_helpers) else "failed",
            message="deepseek-cache helper exports cache message/usage functions"
            if {"createDeepSeekCacheMessages", "attachDeepSeekCacheUsage"}.issubset(exported_helpers)
            else "deepseek-cache helper is missing required cache message/usage exports",
            evidence={"exported_helpers": exported_helpers},
        )
    )

    imported_helpers = find_imported_helper_names(sessions_text)
    imported_expected = sorted(set(imported_helpers).intersection(HELPER_NAME_CANDIDATES))
    usage_lines = find_helper_usage(sessions_text, imported_expected)
    checks.append(
        Check(
            name="claude_agent_sessions_imports_deepseek_cache_helper",
            status="passed" if imported_expected else "failed",
            message="claude-agent-sessions imports deepseek-cache helper"
            if imported_expected
            else "claude-agent-sessions does not import deepseek-cache helper",
            evidence={"imported_helpers": imported_expected},
        )
    )
    checks.append(
        Check(
            name="claude_agent_sessions_uses_deepseek_cache_helper",
            status="passed" if usage_lines else "failed",
            message="claude-agent-sessions uses imported deepseek-cache helper outside import statements"
            if usage_lines
            else "claude-agent-sessions does not use deepseek-cache helper outside import statements",
            evidence={"usage_lines": usage_lines},
        )
    )

    cache_metadata_lines = line_numbers_for_tokens(sessions_text, CACHE_METADATA_TOKENS)
    transcript_lines = line_numbers_for_tokens(sessions_text, TRANSCRIPT_CONTEXT_TOKENS)
    manifest_lines = line_numbers_for_tokens(sessions_text, MANIFEST_CONTEXT_TOKENS)

    transcript_has_cache_metadata = bool(cache_metadata_lines) and bool(transcript_lines)
    manifest_has_cache_metadata = bool(cache_metadata_lines) and bool(manifest_lines)
    checks.append(
        Check(
            name="deepseek_transcript_cache_metadata_fields",
            status="passed" if transcript_has_cache_metadata else "failed",
            message="DeepSeek transcript-writing code contains cache metadata fields"
            if transcript_has_cache_metadata
            else "DeepSeek transcript-writing code lacks obvious cache metadata fields",
            evidence={
                "cache_metadata_token_lines": cache_metadata_lines,
                "transcript_context_lines": transcript_lines,
            },
        )
    )
    checks.append(
        Check(
            name="deepseek_manifest_cache_metadata_fields",
            status="passed" if manifest_has_cache_metadata else "failed",
            message="DeepSeek manifest/session code contains cache metadata fields"
            if manifest_has_cache_metadata
            else "DeepSeek manifest/session code lacks obvious cache metadata fields",
            evidence={
                "cache_metadata_token_lines": cache_metadata_lines,
                "manifest_context_lines": manifest_lines,
            },
        )
    )

    scan_files = relevant_scan_files(root)
    secret_scan = scan_for_obvious_secrets(root, scan_files)
    checks.append(
        Check(
            name="related_files_obvious_secret_scan",
            status=secret_scan["status"],
            message="No obvious hardcoded secret patterns found in related files"
            if secret_scan["status"] == "passed"
            else "Obvious hardcoded secret patterns found in related files",
            evidence=secret_scan,
        )
    )

    serialized_checks = [
        {"name": check.name, "status": check.status, "message": check.message, "evidence": check.evidence}
        for check in checks
    ]
    failed = [check for check in serialized_checks if check["status"] != "passed"]
    return {
        "ok": not failed,
        "status": "passed" if not failed else "failed",
        "generated_at": now_iso(),
        "scope": {
            "root": str(root),
            "no_network_calls": True,
            "no_environment_secret_reads": True,
            "no_deepseek_api_calls": True,
            "no_training_invocation": True,
        },
        "targets": {
            "deepseek_cache_helper": DEEPSEEK_CACHE_TS.as_posix(),
            "claude_agent_sessions": CLAUDE_AGENT_SESSIONS_TS.as_posix(),
        },
        "checks": serialized_checks,
        "summary": {
            "total_checks": len(serialized_checks),
            "passed_checks": len(serialized_checks) - len(failed),
            "failed_checks": len(failed),
            "failed_check_names": [check["name"] for check in failed],
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# DeepSeek Cache Policy Verification {TARGET_DATE}",
        "",
        f"- Status: **{payload['status']}**",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Root: `{payload['scope']['root']}`",
        "- Network/API/training: no network calls, no DeepSeek API calls, no training invocation.",
        "- Environment secrets: not read; secret scan records only pattern names and line numbers.",
        "",
        "## Summary",
        "",
        f"- Total checks: {payload['summary']['total_checks']}",
        f"- Passed: {payload['summary']['passed_checks']}",
        f"- Failed: {payload['summary']['failed_checks']}",
        "",
        "## Checks",
        "",
        "| Check | Status | Message |",
        "|---|---:|---|",
    ]
    for check in payload["checks"]:
        message = str(check["message"]).replace("|", "\\|")
        lines.append(f"| `{check['name']}` | **{check['status']}** | {message} |")

    lines.extend(["", "## Evidence Pointers", ""])
    for check in payload["checks"]:
        lines.append(f"### {check['name']}")
        lines.append("")
        evidence = check.get("evidence", {})
        lines.append("```json")
        lines.append(json.dumps(evidence, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static DeepSeek cache policy verification.")
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--json-output", default=DEFAULT_JSON_REPORT, type=Path)
    parser.add_argument("--md-output", default=DEFAULT_MD_REPORT, type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit with status 1 when verification fails.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    payload = evaluate(root)
    json_output = args.json_output if args.json_output.is_absolute() else root / args.json_output
    md_output = args.md_output if args.md_output.is_absolute() else root / args.md_output
    write_json(json_output, payload)
    write_markdown(md_output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "json_output": rel(json_output, root),
                "md_output": rel(md_output, root),
                "failed_check_names": payload["summary"]["failed_check_names"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.strict and not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
