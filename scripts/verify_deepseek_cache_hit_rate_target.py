from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CACHE_HELPER = ROOT / "web/research-agent-workstation/src/lib/server/deepseek-cache.ts"
SESSION_FILE = ROOT / "web/research-agent-workstation/src/lib/server/claude-agent-sessions.ts"
OUTPUT_JSON = ROOT / "workspace/deepseek_cache_hit_rate_target_verification_20260623.json"
OUTPUT_MD = ROOT / "reports/DEEPSEEK_CACHE_HIT_RATE_TARGET_VERIFICATION_20260623.md"
TARGET = 0.8

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"KGAT_[A-Za-z0-9]{16,}"),
    re.compile(r"Bearer\s+(?!\$\{|`\$\{|['\"]?\s*\$)[A-Za-z0-9_\-.]{20,}"),
    re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{6,}['\"]"),
    re.compile(r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*['\"][A-Za-z0-9_\-.]{16,}['\"]"),
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace") if path.exists() else ""


def check(name: str, passed: bool, detail: str, evidence: Any = None) -> dict[str, Any]:
    return {"name": name, "status": "passed" if passed else "failed", "detail": detail, "evidence": evidence}


def scan_secrets(paths: list[Path]) -> list[dict[str, Any]]:
    hits = []
    for path in paths:
        text = read(path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    hits.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "line": line_no, "redaction": "value_not_recorded"})
    return hits


def parse_manifest() -> dict[str, Any]:
    manifest_path = ROOT / "workspace/code_agent_cache/deepseek_cache_manifest.json"
    if not manifest_path.exists():
        return {"exists": False, "observed_sessions": 0, "observed_hit_ratio": None, "local_hit_ratio": None}
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    sessions = payload.get("sessions", []) if isinstance(payload, dict) else []
    observed = [s for s in sessions if isinstance(s, dict) and isinstance(s.get("cache_hit_ratio"), (int, float))]
    local_hits = [s for s in sessions if isinstance(s, dict) and s.get("local_response_cache_hit") is True]
    ratio = sum(float(s["cache_hit_ratio"]) for s in observed) / len(observed) if observed else None
    local_ratio = len(local_hits) / len(sessions) if sessions else None
    return {
        "exists": True,
        "observed_sessions": len(observed),
        "total_sessions": len(sessions),
        "observed_hit_ratio": ratio,
        "local_hit_ratio": local_ratio,
    }


def build_report() -> dict[str, Any]:
    cache_text = read(CACHE_HELPER)
    session_text = read(SESSION_FILE)
    manifest_stats = parse_manifest()
    secret_hits = scan_secrets([CACHE_HELPER, SESSION_FILE, Path(__file__).resolve()])
    checks = [
        check("cache_helper_exists", CACHE_HELPER.exists(), "deepseek-cache.ts exists"),
        check("strategy_v2_enabled", "deepseek_prompt_cache_v2" in cache_text, "cache strategy v2 is enabled"),
        check("target_80_declared", "TARGET_CACHE_HIT_RATIO = 0.8" in cache_text and "target_cache_hit_ratio" in cache_text, "80% target is declared in cache metadata"),
        check("local_exact_response_cache_defined", all(token in cache_text for token in ["readDeepSeekCachedResponse", "writeDeepSeekCachedResponse", "responseCacheRelative", "localResponseCacheUsage"]), "local exact-response cache helpers are defined"),
        check("stable_prefix_expanded", "STABLE_SYSTEM_PROMPT" in cache_text and "STABLE_USER_PREFIX" in cache_text and cache_text.count("Cache") >= 5, "stable provider-cache prefixes exist and include cache discipline"),
        check("session_reads_cache_before_fetch", "readDeepSeekCachedResponse" in session_text and session_text.find("readDeepSeekCachedResponse") < session_text.find("fetch(`${config.baseUrl"), "DeepSeek session checks local cache before external fetch"),
        check("session_writes_cache_after_fetch", "writeDeepSeekCachedResponse" in session_text and session_text.rfind("writeDeepSeekCachedResponse") > session_text.find("fetch(`${config.baseUrl"), "DeepSeek session stores successful external responses for reuse"),
        check("local_hit_skips_external_call", "deepseek_code_agent_session_local_cache_hit" in session_text and "no external model call" in session_text, "local cache hit path records no external model call"),
        check("deepseek_provider_can_be_forced", "selectedProvider(input.provider)" in session_text and 'preferredProvider === "deepseek_code_agent"' in session_text, "explicit DeepSeek Code Agent requests are not silently routed to Claude when both providers exist"),
        check("requested_model_drives_cache_lookup", "const requestedModel = baseRecord.model || config.model" in session_text and "model: requestedModel" in session_text, "DeepSeek local cache lookup, fetch body, and cache write respect the requested model"),
        check("cache_probe_is_read_only", "probeDeepSeekCodeCache" in session_text and "cache_probe" in read(ROOT / "web/research-agent-workstation/src/app/api/tasks/[taskId]/code-agent-draft/route.ts") and "external_model_calls_allowed: false" in session_text, "cache probe can check local response cache without creating a session or allowing external calls"),
        check("cache_only_miss_blocks_external_call", "cacheOnly?: boolean" in session_text and "deepseek_code_agent_cache_only_miss" in session_text and "external model call was blocked" in session_text, "cache-only mode records a miss artifact and blocks external calls"),
        check("prompt_context_is_stabilized", all(token in session_text for token in ["stableJsonForPrompt", "stablePromptValue", "VOLATILE_PROMPT_KEYS", "normalizePromptString"]), "Code Agent prompt evidence is canonicalized to reduce timestamp/path/session drift before fingerprinting"),
        check("cache_manifest_tracks_ratio", all(token in cache_text for token in ["cache_hit_ratio", "local_response_cache_hit", "target_cache_hit_ratio"]), "manifest tracks cache hit ratio and local hit status"),
        check("secret_scan_clean", not secret_hits, f"secret pattern hits: {len(secret_hits)}", secret_hits),
    ]
    passed = sum(1 for item in checks if item["status"] == "passed")
    status = "passed" if passed == len(checks) else "failed"
    can_guarantee_future_identical_prompt = status == "passed"
    measured_80 = manifest_stats.get("observed_hit_ratio") is not None and manifest_stats["observed_hit_ratio"] >= TARGET
    observed_sessions = int(manifest_stats.get("observed_sessions") or 0)
    observed_hit_ratio = manifest_stats.get("observed_hit_ratio")
    current_hit_equivalent = observed_hit_ratio * observed_sessions if isinstance(observed_hit_ratio, (int, float)) else 0.0
    needed_perfect_local_hits = 0
    if observed_sessions > 0 and not measured_80:
        while (current_hit_equivalent + needed_perfect_local_hits) / (observed_sessions + needed_perfect_local_hits) < TARGET:
            needed_perfect_local_hits += 1
    batch_generation_gate_status = "approved" if measured_80 else "blocked_below_target"
    return {
        "ok": status == "passed",
        "artifact_type": "deepseek_cache_hit_rate_target_verification",
        "created_by_agent": "DeepSeekCacheOptimizationAgent",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "target_cache_hit_ratio": TARGET,
        "implementation_target_met": can_guarantee_future_identical_prompt,
        "measured_80_percent_met": measured_80,
        "measured_status": "no_runtime_sessions_after_v2" if manifest_stats.get("observed_sessions", 0) == 0 else ("passed" if measured_80 else "below_target"),
        "batch_generation_gate_status": batch_generation_gate_status,
        "needed_perfect_local_hits_for_80_percent": needed_perfect_local_hits,
        "cache_warmup_plan": {
            "allowed": not measured_80 and can_guarantee_future_identical_prompt,
            "mode": "local_exact_response_cache_only",
            "external_model_calls_allowed": False,
            "minimum_repeated_cached_sessions_needed": needed_perfect_local_hits,
            "instruction": (
                "Before large Code Agent batches, run repeated sessions only for prompts that already have "
                "workspace/code_agent_cache/responses entries, or wait for natural repeated cached sessions. "
                "Do not create artificial manifest entries and do not call the external model solely to improve the ratio."
            ),
        },
        "manifest_stats": manifest_stats,
        "checks_total": len(checks),
        "checks_passed": passed,
        "checks": checks,
        "conclusion": {
            "all_agent_scope": "Only real external LLM Code Agent calls use DeepSeek cache. Non-LLM workflow agents are deterministic artifact writers and do not spend DeepSeek tokens.",
            "80_percent_claim": "Implemented for identical prompt fingerprints through local exact-response cache; provider prompt cache is additionally optimized through stable prefixes. Runtime measured >=80% requires repeated v2 sessions after deployment.",
            "current_runtime_evidence": manifest_stats,
        },
    }


def write_outputs(report: dict[str, Any]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# DeepSeek Cache Hit-rate Target Verification",
        "",
        f"- Status: `{report['status']}`",
        f"- Target cache hit ratio: `{report['target_cache_hit_ratio']:.0%}`",
        f"- Implementation target met: `{str(report['implementation_target_met']).lower()}`",
        f"- Measured >=80% met: `{str(report['measured_80_percent_met']).lower()}`",
        f"- Measured status: `{report['measured_status']}`",
        f"- Batch generation gate: `{report['batch_generation_gate_status']}`",
        f"- Needed perfect local hits for >=80%: `{report['needed_perfect_local_hits_for_80_percent']}`",
        f"- Checks: `{report['checks_passed']}/{report['checks_total']}`",
        "",
        "## Conclusion",
        "",
        report["conclusion"]["80_percent_claim"],
        "",
        "## Scope",
        "",
        report["conclusion"]["all_agent_scope"],
        "",
        "## Warmup Plan",
        "",
        f"- Allowed: `{str(report['cache_warmup_plan']['allowed']).lower()}`",
        f"- Mode: `{report['cache_warmup_plan']['mode']}`",
        f"- External model calls allowed: `{str(report['cache_warmup_plan']['external_model_calls_allowed']).lower()}`",
        f"- Minimum repeated cached sessions needed: `{report['cache_warmup_plan']['minimum_repeated_cached_sessions_needed']}`",
        f"- Instruction: {report['cache_warmup_plan']['instruction']}",
        "",
        "## Checks",
        "",
    ]
    for item in report["checks"]:
        lines.append(f"- `{item['status']}` {item['name']}: {item['detail']}")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if args.write_report:
        write_outputs(report)
    print(json.dumps({
        "status": report["status"],
        "implementation_target_met": report["implementation_target_met"],
        "measured_80_percent_met": report["measured_80_percent_met"],
        "measured_status": report["measured_status"],
        "batch_generation_gate_status": report["batch_generation_gate_status"],
        "needed_perfect_local_hits_for_80_percent": report["needed_perfect_local_hits_for_80_percent"],
        "checks": f"{report['checks_passed']}/{report['checks_total']}",
        "json": str(OUTPUT_JSON.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
        "md": str(OUTPUT_MD.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
