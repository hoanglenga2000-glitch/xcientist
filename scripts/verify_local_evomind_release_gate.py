"""Aggregate the local EvoMind release/demo gate.

This verifier is intentionally read-only.  It does not restart the dashboard,
does not call any grader, does not submit to Kaggle, and does not mutate a run.
It answers one narrow question:

    Is the current local EvoMind gateway + novice Agent UX + security/build
    evidence good enough for a local demo / pre-release checkpoint?

The broader product release gate still requires packaging, full evidence ZIP,
video QA, MLE A/B, Claim Audit, release commit/tag, and human Kaggle gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
TARGET_SIIM_RUN = "evomind_siim_isic_a800_job90353_20260730_095826"
REQUIRED_PROVIDER = "openai"
REQUIRED_MODEL = "gpt-5.6-sol"


REQUIRED_EVIDENCE = {
    "assistant_quality": ROOT / "workspace/evaluation/assistant_novice_quality_gpt56_current.json",
    "assistant_quality_gate": ROOT / "workspace/evaluation/assistant_novice_quality_gate_current.json",
    "new_user_readiness": ROOT / "workspace/new_user_release_readiness.json",
    "authenticated_web_report": ROOT / "artifacts/authenticated-web-report-current.json",
    "authenticated_web_security": ROOT / "artifacts/authenticated-web-security-current.json",
    "web_isolated_build": ROOT / "artifacts/web-isolated-build-current.json",
    "live_assistant_demo_smoke": ROOT / "workspace/evaluation/live_assistant_demo_smoke_current.json",
}

OPTIONAL_EVIDENCE = {
    "hpc_profile_readiness": ROOT / "workspace/hpc/job90353_profile_readiness_current.json",
    "job90353_reenrollment_plan": ROOT / "workspace/hpc/job90353_reenrollment_plan_current.json",
    "agent_ux_release_source_scope": ROOT / "workspace/evaluation/agent_ux_release_source_scope_current.json",
}

EVOMIND_RUNTIME_SOURCE_FILES = tuple(sorted((ROOT / "src/evomind_runtime").glob("*.py")))
EVOMIND_RUNTIME_TEST_FILES = tuple(sorted((ROOT / "tests").glob("test_evomind_runtime*.py")))


RELEASE_RELEVANT_SOURCE_FILES = (
    # Evidence builders and their contract tests.
    ROOT / "scripts/verify_local_evomind_release_gate.py",
    ROOT / "scripts/verify_live_assistant_demo_smoke.py",
    ROOT / "scripts/build_agent_ux_release_source_manifest.py",
    ROOT / "tests/test_verify_local_evomind_release_gate.py",
    ROOT / "tests/test_verify_live_assistant_demo_smoke.py",
    ROOT / "tests/test_build_agent_ux_release_source_manifest.py",
    # Novice request compiler, verified context, quality audit, and provider loop.
    ROOT / "configs/evaluation/assistant_behavior_board_v1.json",
    ROOT / "configs/evaluation/assistant_novice_v1.json",
    ROOT / "src/xsci/assistant_behavior_distillation.py",
    ROOT / "src/xsci/assistant_context.py",
    ROOT / "src/xsci/assistant_quality_evaluation.py",
    ROOT / "src/xsci/assistant_stream.py",
    ROOT / "src/xsci/config.py",
    ROOT / "src/xsci/kaggle_conversation.py",
    ROOT / "src/xsci/kaggle_intent.py",
    ROOT / "src/xsci/kaggle_session.py",
    ROOT / "src/xsci/terminal_agent.py",
    ROOT / "src/xsci/terminal_tools.py",
    ROOT / "src/xsci/user_request.py",
    ROOT / "src/research_os/agent/messaging.py",
    ROOT / "src/research_os/llm_client.py",
    ROOT / "tests/test_agent_messaging.py",
    ROOT / "tests/test_assistant_behavior_distillation.py",
    ROOT / "tests/test_assistant_context.py",
    ROOT / "tests/test_assistant_quality_evaluation.py",
    ROOT / "tests/test_assistant_stream.py",
    ROOT / "tests/test_kaggle_conversation_tool_loop.py",
    ROOT / "tests/test_user_request_protocol.py",
    # Runtime service used by both 8765 and the browser NDJSON bridge.
    *EVOMIND_RUNTIME_SOURCE_FILES,
    *EVOMIND_RUNTIME_TEST_FILES,
    # Local lifecycle, gateway profile, and fail-closed HPC route.
    ROOT / "scripts/verify_new_user_release_readiness.py",
    ROOT / "scripts/verify_backend_resource_status.py",
    ROOT / "tests/test_backend_resource_status.py",
    ROOT / "scripts/run_authenticated_web_contract.py",
    ROOT / "scripts/verify_web_isolated_build.py",
    ROOT / "tests/test_verify_web_isolated_build.py",
    ROOT / "scripts/manage_local_gateway.py",
    ROOT / "scripts/verify_openai_gateway.py",
    ROOT / "scripts/manage_hpc_proxy_bridge.ps1",
    ROOT / "scripts/hpc_socks_bridge.py",
    ROOT / "configs/hpc_connection_memory_core.json",
    ROOT / "docs/HPC_CONNECTION_MEMORY_CORE.md",
    ROOT / "tests/test_hpc_connection_memory_core.py",
    ROOT / "tests/test_hpc_socks_bridge.py",
    ROOT / "tests/test_benchmark_openai_gateway_profiles.py",
    ROOT / "scripts/manage_workstation_dashboard.py",
    ROOT / "tests/test_dashboard_build_transaction.py",
    # Loopback session security, authenticated route, and browser UI.
    ROOT / "web/research-agent-workstation/src/proxy.ts",
    ROOT / "web/research-agent-workstation/src/app/layout.tsx",
    ROOT / "web/research-agent-workstation/src/app/api/session/bootstrap/route.ts",
    ROOT / "web/research-agent-workstation/src/app/api/session/status/route.ts",
    ROOT / "web/research-agent-workstation/src/app/api/assistant/stream/route.ts",
    ROOT / "web/research-agent-workstation/src/components/workstation/LocalSessionBootstrap.tsx",
    ROOT / "web/research-agent-workstation/src/components/workstation/screens/AssistantScreen.tsx",
    ROOT / "web/research-agent-workstation/src/lib/server/scientific-report.ts",
    ROOT / "web/research-agent-workstation/src/lib/server/reviewed-existing-report.ts",
    ROOT / "web/research-agent-workstation/src/lib/server/reviewed-existing-report.test.ts",
    ROOT / "web/research-agent-workstation/src/lib/server/local-session.ts",
    ROOT / "web/research-agent-workstation/src/lib/server/paths.ts",
    ROOT / "web/research-agent-workstation/src/lib/server/local-session-security.test.ts",
    ROOT / "web/research-agent-workstation/src/lib/server/loopback-session.test.ts",
    ROOT / "web/research-agent-workstation/scripts/lib/loopback-session.mjs",
    ROOT / "web/research-agent-workstation/scripts/verify-localhost-security-contract.mjs",
    ROOT / "web/research-agent-workstation/scripts/verify-report-studio-contract.mjs",
    ROOT / "web/research-agent-workstation/package.json",
    ROOT / "web/research-agent-workstation/package-lock.json",
)


@dataclass
class Check:
    id: str
    ok: bool
    detail: dict[str, Any]
    severity: str = "blocker"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ok": self.ok,
            "severity": self.severity,
            **self.detail,
        }


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # noqa: BLE001 - verifier must report exact failure
        return None, f"{type(exc).__name__}: {exc}"


def age_hours(path: Path) -> float:
    return (datetime.now().timestamp() - path.stat().st_mtime) / 3600.0


def run_dashboard_status() -> tuple[dict[str, Any] | None, dict[str, Any]]:
    cmd = [sys.executable, "scripts/manage_workstation_dashboard.py", "status"]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=45,
        check=False,
    )
    meta = {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_sha256": hashlib.sha256(proc.stdout.encode("utf-8", errors="replace")).hexdigest(),
        "stderr_tail": proc.stderr[-2000:],
    }
    try:
        parsed = json.loads(proc.stdout)
    except Exception as exc:  # noqa: BLE001
        meta["stdout_tail"] = proc.stdout[-4000:]
        meta["parse_error"] = f"{type(exc).__name__}: {exc}"
        return None, meta
    return parsed, meta


def run_git(args: list[str]) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    cmd = ["git", *args]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    return proc, {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_sha256": hashlib.sha256(proc.stdout.encode("utf-8", errors="replace")).hexdigest(),
        "stderr_tail": proc.stderr[-2000:],
    }


def _repo_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_release_source_tracking(checks: list[Check], *, require_release_source_tracked: bool) -> dict[str, Any]:
    """Ensure release-critical source files are not invisible to a clean checkout.

    Local demo can run with a dirty worktree, but a formal release bundle, clean
    checkout rebuild, or signed tag cannot depend on untracked files or
    uncommitted edits.  This check is a warning in local-demo mode and a blocker
    when the caller opts into release-source strictness.
    """

    rel_paths = [_repo_relative(path) for path in RELEASE_RELEVANT_SOURCE_FILES]
    ls_proc, ls_meta = run_git(["ls-files", "--stage", "--", *rel_paths])
    status_proc, status_meta = run_git(["status", "--porcelain=v1", "--untracked-files=all", "--", *rel_paths])
    tracked: set[str] = set()
    if ls_proc.returncode == 0:
        for line in ls_proc.stdout.splitlines():
            parts = line.split(maxsplit=3)
            if len(parts) == 4:
                tracked.add(parts[3].replace("\\", "/"))

    porcelain: dict[str, list[str]] = {rel: [] for rel in rel_paths}
    if status_proc.returncode == 0:
        for line in status_proc.stdout.splitlines():
            if len(line) < 4:
                continue
            rel = line[3:].strip().strip('"').replace("\\", "/")
            if " -> " in rel:
                rel = rel.split(" -> ", 1)[-1]
            porcelain.setdefault(rel, []).append(line[:2])

    files = []
    for path, rel in zip(RELEASE_RELEVANT_SOURCE_FILES, rel_paths, strict=True):
        files.append(
            {
                "path": rel,
                "exists": path.exists(),
                "tracked": rel in tracked,
                "git_status": porcelain.get(rel, []),
                "sha256": file_sha256(path) if path.exists() else None,
                "bytes": path.stat().st_size if path.exists() else None,
            }
        )

    missing = [item["path"] for item in files if not item["exists"]]
    untracked = [item["path"] for item in files if item["exists"] and not item["tracked"]]
    tracked_dirty = [
        item["path"]
        for item in files
        if item["tracked"] and any(code.strip() for code in item["git_status"])
    ]
    ok = (
        ls_proc.returncode == 0
        and status_proc.returncode == 0
        and not missing
        and not untracked
        and (not require_release_source_tracked or not tracked_dirty)
    )
    detail = {
        "require_release_source_tracked": require_release_source_tracked,
        "missing": missing,
        "untracked": untracked,
        "tracked_dirty": tracked_dirty,
        "strict_dirty_blockers": tracked_dirty if require_release_source_tracked else [],
        "files": files,
        "git_ls_files": ls_meta,
        "git_status": status_meta,
        "note": (
            "Local demo may pass with working-tree files present, but formal release must commit "
            "these source files so a clean checkout can rebuild the Agent UX fixes."
        ),
    }
    add(
        checks,
        "release_source:agent_ux_files_tracked_for_clean_checkout",
        ok,
        severity="blocker" if require_release_source_tracked else "warning",
        **detail,
    )
    return detail


def check_release_source_scope_manifest(
    evidence: dict[str, Any],
    checks: list[Check],
    source_tracking: dict[str, Any],
    *,
    max_age_hours: float,
    require_release_source_clean: bool,
) -> None:
    manifest = evidence.get("agent_ux_release_source_scope") or {}
    path = OPTIONAL_EVIDENCE["agent_ux_release_source_scope"]
    manifest_files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    manifest_paths = [str(item.get("path")) for item in manifest_files if isinstance(item, dict)]
    source_paths = [str(item.get("path")) for item in source_tracking.get("files") or [] if isinstance(item, dict)]
    expected_status = (
        "clean"
        if not (
            source_tracking.get("missing")
            or source_tracking.get("untracked")
            or source_tracking.get("tracked_dirty")
        )
        else "needs_release_source_review"
    )
    detail = {
        "path": str(path),
        "exists": path.exists(),
        "age_hours": round(age_hours(path), 3) if path.exists() else None,
        "sha256": file_sha256(path) if path.exists() else None,
        "schema": manifest.get("schema"),
        "status": manifest.get("status"),
        "secret_scan": manifest.get("secret_scan"),
        "expected_status": expected_status,
        "manifest_file_count": len(manifest_paths),
        "source_tracking_file_count": len(source_paths),
        "manifest_paths_match_source_tracking": manifest_paths == source_paths,
        "manifest_missing": manifest.get("missing"),
        "source_missing": source_tracking.get("missing"),
        "manifest_untracked": manifest.get("untracked"),
        "source_untracked": source_tracking.get("untracked"),
        "manifest_dirty_tracked": manifest.get("dirty_tracked"),
        "source_dirty_tracked": source_tracking.get("tracked_dirty"),
        "claim_boundary": manifest.get("claim_boundary"),
    }
    ok = bool(
        path.exists()
        and detail["age_hours"] is not None
        and detail["age_hours"] <= max_age_hours
        and manifest.get("schema") == "evomind.agent_ux_release_source_scope.v1"
        and manifest.get("status") == expected_status
        and (manifest.get("secret_scan") or {}).get("status") == "passed"
        and manifest_paths == source_paths
        and list(manifest.get("missing") or []) == list(source_tracking.get("missing") or [])
        and list(manifest.get("untracked") or []) == list(source_tracking.get("untracked") or [])
        and list(manifest.get("dirty_tracked") or []) == list(source_tracking.get("tracked_dirty") or [])
    )
    add(
        checks,
        "release_source:scope_manifest_fresh_and_consistent",
        ok,
        severity="blocker" if require_release_source_clean else "warning",
        **detail,
    )


def check_release_source_manifest_secret_scan(
    evidence: dict[str, Any],
    checks: list[Check],
    *,
    require_release_source_clean: bool,
) -> None:
    manifest = evidence.get("agent_ux_release_source_scope") or {}
    secret_scan = manifest.get("secret_scan") if isinstance(manifest.get("secret_scan"), dict) else {}
    findings = secret_scan.get("findings") if isinstance(secret_scan.get("findings"), list) else []
    add(
        checks,
        "release_source:scope_manifest_secret_scan_passed",
        secret_scan.get("status") == "passed" and int(secret_scan.get("finding_count") or 0) == 0,
        severity="blocker",
        require_release_source_clean=require_release_source_clean,
        status=secret_scan.get("status"),
        scanned_files=secret_scan.get("scanned_files"),
        finding_count=secret_scan.get("finding_count"),
        findings=findings[:20],
        note="Release-critical Agent UX source files must not contain plaintext API keys, tokens, passwords, or private keys.",
    )


def add(checks: list[Check], check_id: str, ok: bool, *, severity: str = "blocker", **detail: Any) -> None:
    checks.append(Check(id=check_id, ok=bool(ok), detail=detail, severity=severity))


def check_evidence_files(max_age_hours: float, evidence: dict[str, Any], checks: list[Check]) -> None:
    for name, path in REQUIRED_EVIDENCE.items():
        exists = path.exists()
        detail: dict[str, Any] = {"path": str(path)}
        if exists:
            detail.update(
                {
                    "bytes": path.stat().st_size,
                    "age_hours": round(age_hours(path), 3),
                    "sha256": file_sha256(path),
                }
            )
            parsed, error = load_json(path)
            if parsed is not None:
                evidence[name] = parsed
                detail["schema"] = parsed.get("schema") or parsed.get("suite") or parsed.get("contract", {}).get("schema")
                detail["status"] = parsed.get("status")
                detail["evidence_ok"] = parsed.get("ok")
            else:
                detail["parse_error"] = error
        add(
            checks,
            f"evidence:{name}:exists_parse_fresh",
            exists
            and name in evidence
            and detail.get("age_hours", max_age_hours + 1) <= max_age_hours,
            **detail,
        )


def check_optional_evidence_files(max_age_hours: float, evidence: dict[str, Any], checks: list[Check]) -> None:
    for name, path in OPTIONAL_EVIDENCE.items():
        parsed, error = load_json(path) if path.exists() else (None, "missing")
        detail = {
            "path": str(path),
            "exists": path.exists(),
            "error": error,
            "age_hours": round(age_hours(path), 3) if path.exists() else None,
            "sha256": file_sha256(path) if path.exists() else None,
        }
        ok = bool(path.exists() and parsed and not error and age_hours(path) <= max_age_hours)
        if parsed:
            evidence[name] = parsed
            detail["schema"] = parsed.get("schema")
            detail["status"] = parsed.get("status")
        add(
            checks,
            f"evidence:{name}:exists_parse_fresh",
            ok,
            severity="warning",
            **detail,
        )


def check_agent_quality(evidence: dict[str, Any], checks: list[Check]) -> None:
    report = evidence.get("assistant_quality") or {}
    gate = evidence.get("assistant_quality_gate") or {}
    aggregate = report.get("aggregate") or {}
    results = report.get("results") or []
    fatal_gates = [item.get("fatal_gate") or {} for item in results]
    providers = {str((item.get("execution") or {}).get("provider")) for item in results}
    models = {str((item.get("execution") or {}).get("model")) for item in results}

    add(
        checks,
        "assistant_quality:gate_passed",
        report.get("status") == "passed" and gate.get("status") == "passed",
        report_status=report.get("status"),
        gate_status=gate.get("status"),
        pass_rate=aggregate.get("pass_rate"),
        mean_score=aggregate.get("mean_score"),
        latency_p95_seconds=aggregate.get("latency_p95_seconds"),
    )
    add(
        checks,
        "assistant_quality:provider_model_hard_gate",
        providers == {REQUIRED_PROVIDER} and models == {REQUIRED_MODEL} and all(fg.get("passed") is True for fg in fatal_gates),
        required_provider=REQUIRED_PROVIDER,
        required_model=REQUIRED_MODEL,
        providers=sorted(providers),
        models=sorted(models),
        fatal_gate_count=len(fatal_gates),
    )
    add(
        checks,
        "assistant_quality:novice_suite_complete",
        aggregate.get("case_count") == 5
        and aggregate.get("passed_cases") == aggregate.get("case_count")
        and float(aggregate.get("mean_score") or 0.0) >= 0.90,
        aggregate=aggregate,
        selected_cases=report.get("selected_cases"),
    )
    add(
        checks,
        "assistant_quality:llm_first_tool_loop",
        all((item.get("execution") or {}).get("tool_calls_total", 0) >= 1 for item in results)
        and all((item.get("execution") or {}).get("llm_status") == "completed" for item in results),
        tool_names=[(item.get("execution") or {}).get("tool_names", []) for item in results],
        statuses=[(item.get("execution") or {}).get("llm_status") for item in results],
    )


def check_governance(evidence: dict[str, Any], checks: list[Check]) -> None:
    report = evidence.get("assistant_quality") or {}
    invariants = report.get("governance_invariants") or {}
    after = invariants.get("after") or {}
    runs = after.get("siim_run_directories") or []

    add(
        checks,
        "governance:siim_target_present_no_new_formal_run_claim",
        TARGET_SIIM_RUN in runs,
        target_run=TARGET_SIIM_RUN,
        observed_siim_run_directories=runs,
        note="Historical SIIM directories may exist; this gate requires the target run to remain present and does not create or mutate runs.",
    )
    add(
        checks,
        "governance:private_grader_exactly_once_failed_closed",
        invariants.get("passed") is True
        and after.get("grader_execution_count") == 1
        and after.get("grader_outcome") == "failed_closed"
        and after.get("grader_score") is None,
        invariants_passed=invariants.get("passed"),
        grader_execution_count=after.get("grader_execution_count"),
        grader_outcome=after.get("grader_outcome"),
        grader_score=after.get("grader_score"),
    )
    add(
        checks,
        "governance:kaggle_not_submitted",
        after.get("official_submission_executed") is False,
        official_submission_executed=after.get("official_submission_executed"),
    )


def check_new_user_readiness(evidence: dict[str, Any], checks: list[Check]) -> None:
    readiness = evidence.get("new_user_readiness") or {}
    failed = readiness.get("failed_checks") or []
    add(
        checks,
        "new_user_gateway:readiness_passed",
        readiness.get("status") == "passed" and readiness.get("release_state") == "ready_for_new_user_evomind_gateway" and not failed,
        status=readiness.get("status"),
        release_state=readiness.get("release_state"),
        failed_checks=failed,
        default_gateway=readiness.get("default_gateway"),
        claim_boundary=readiness.get("claim_boundary"),
    )


def check_web_contracts(evidence: dict[str, Any], checks: list[Check]) -> None:
    report = evidence.get("authenticated_web_report") or {}
    security = evidence.get("authenticated_web_security") or {}
    build = evidence.get("web_isolated_build") or {}
    security_contract = security.get("contract") or {}

    add(
        checks,
        "web_contract:authenticated_report_passed",
        report.get("ok") is True
        and report.get("isolated_runtime") is True
        and report.get("protected_ports_unchanged") is True
        and (report.get("contract") or {}).get("ok") is True,
        report_ok=report.get("ok"),
        isolated_runtime=report.get("isolated_runtime"),
        protected_ports_unchanged=report.get("protected_ports_unchanged"),
        check_count=(report.get("contract") or {}).get("checks"),
    )
    add(
        checks,
        "web_contract:localhost_security_passed",
        security.get("ok") is True
        and security.get("isolated_runtime") is True
        and security.get("protected_ports_unchanged") is True
        and security_contract.get("ok") is True
        and security_contract.get("grader_invoked") is False
        and security_contract.get("mutations_with_product_side_effects") == 0,
        security_ok=security.get("ok"),
        isolated_runtime=security.get("isolated_runtime"),
        protected_ports_unchanged=security.get("protected_ports_unchanged"),
        check_count=security_contract.get("checks"),
        grader_invoked=security_contract.get("grader_invoked"),
        mutations_with_product_side_effects=security_contract.get("mutations_with_product_side_effects"),
    )
    add(
        checks,
        "web_build:isolated_production_build_passed",
        build.get("status") == "passed"
        and (build.get("npm_ci") or {}).get("ok") is True
        and (build.get("npm_audit") or {}).get("ok") is True
        and (build.get("prisma_generate") or {}).get("ok") is True
        and (build.get("build") or {}).get("ok") is True
        and build.get("standalone_server_exists") is True
        and build.get("protected_ports_unchanged") is True
        and build.get("stage_removed") is True,
        status=build.get("status"),
        build_id=build.get("build_id"),
        npm_ci_ok=(build.get("npm_ci") or {}).get("ok"),
        npm_audit_ok=(build.get("npm_audit") or {}).get("ok"),
        prisma_generate_ok=(build.get("prisma_generate") or {}).get("ok"),
        build_ok=(build.get("build") or {}).get("ok"),
        standalone_server_exists=build.get("standalone_server_exists"),
        protected_ports_unchanged=build.get("protected_ports_unchanged"),
        stage_removed=build.get("stage_removed"),
    )


def check_live_assistant_demo(evidence: dict[str, Any], checks: list[Check]) -> None:
    smoke = evidence.get("live_assistant_demo_smoke") or {}
    assistant = smoke.get("assistant") or {}
    followup = smoke.get("assistant_followup") or {}
    literature = smoke.get("assistant_literature") or {}
    connection = smoke.get("assistant_connection") or {}
    conversation = smoke.get("conversation") or {}
    governance = smoke.get("governance") or {}
    governance_invariants = governance.get("invariants") or {}
    ux_budget = smoke.get("ux_budget") or {}
    ux_checks = ux_budget.get("checks") or {}
    auth = smoke.get("auth") or {}
    session = smoke.get("session_status") or {}
    required_terms = assistant.get("required_terms") or {}
    followup_terms = followup.get("required_terms") or {}
    add(
        checks,
        "live_assistant_demo:browser_equivalent_auth_and_multiturn_stream",
        smoke.get("status") == "passed"
        and auth.get("ok") is True
        and session.get("authenticated") is True
        and assistant.get("completed") is True
        and followup.get("completed") is True
        and literature.get("completed") is True
        and connection.get("completed") is True
        and int(assistant.get("tool_calls_total") or 0) >= 1
        and int(followup.get("tool_calls_total") or 0) >= 1
        and int(literature.get("tool_calls_total") or 0) >= 1
        and int(connection.get("tool_calls_total") or 0) >= 1
        and bool(assistant.get("tool_names"))
        and bool(followup.get("tool_names"))
        and "verified_context" in list(literature.get("tool_names") or [])
        and "hpc_connection_status" in list(connection.get("tool_names") or [])
        and int(assistant.get("answer_characters") or 0) >= 500
        and int(followup.get("answer_characters") or 0) >= 300
        and int(literature.get("answer_characters") or 0) >= 400
        and int(connection.get("answer_characters") or 0) >= 220
        and all(bool(value) for value in (literature.get("literature_checks") or {}).values())
        and all(bool(value) for value in (connection.get("connection_checks") or {}).values())
        and all(bool(value) for value in required_terms.values())
        and all(bool(value) for value in followup_terms.values())
        and conversation.get("turns") == 4
        and conversation.get("same_session_id") is True
        and conversation.get("history_sent_to_followup") is True
        and conversation.get("history_sent_to_literature") is True
        and conversation.get("history_sent_to_connection") is True
        and conversation.get("first_turn_passed") is True
        and conversation.get("followup_turn_passed") is True
        and conversation.get("literature_turn_passed") is True
        and conversation.get("connection_turn_passed") is True
        and ux_budget.get("passed") is True
        and all(bool(value) for value in ux_checks.values())
        and governance.get("passed") is True
        and all(bool(value) for value in governance_invariants.values())
        and smoke.get("temporary_ports_released") is True
        and smoke.get("token_values_recorded") is False
        and smoke.get("session_values_recorded") is False,
        status=smoke.get("status"),
        port=smoke.get("port"),
        runtime_port=smoke.get("runtime_port"),
        auth_ok=auth.get("ok"),
        authenticated=session.get("authenticated"),
        assistant_completed=assistant.get("completed"),
        followup_completed=followup.get("completed"),
        literature_completed=literature.get("completed"),
        connection_completed=connection.get("completed"),
        native_tool_calls=assistant.get("native_tool_calls"),
        followup_native_tool_calls=followup.get("native_tool_calls"),
        literature_native_tool_calls=literature.get("native_tool_calls"),
        connection_native_tool_calls=connection.get("native_tool_calls"),
        total_native_tool_calls=conversation.get("total_native_tool_calls"),
        total_orchestrated_tool_calls=conversation.get("total_orchestrated_tool_calls"),
        total_real_tool_calls=conversation.get("total_real_tool_calls"),
        tool_names=assistant.get("tool_names"),
        followup_tool_names=followup.get("tool_names"),
        literature_tool_names=literature.get("tool_names"),
        connection_tool_names=connection.get("tool_names"),
        answer_characters=assistant.get("answer_characters"),
        followup_answer_characters=followup.get("answer_characters"),
        literature_answer_characters=literature.get("answer_characters"),
        connection_answer_characters=connection.get("answer_characters"),
        literature_checks=literature.get("literature_checks"),
        connection_checks=connection.get("connection_checks"),
        ux_budget_passed=ux_budget.get("passed"),
        ux_budget_checks=ux_checks,
        ux_budget_turns=ux_budget.get("turns"),
        required_terms=required_terms,
        followup_required_terms=followup_terms,
        conversation=conversation,
        governance_passed=governance.get("passed"),
        governance_invariants=governance_invariants,
        governance_changed_hashes=governance.get("changed_hashes"),
        governance_run_dirs_changed=governance.get("run_dirs_changed"),
        temporary_ports_released=smoke.get("temporary_ports_released"),
        token_values_recorded=smoke.get("token_values_recorded"),
        session_values_recorded=smoke.get("session_values_recorded"),
        claim_scope=smoke.get("claim_scope"),
    )


def check_hpc_profile_readiness(evidence: dict[str, Any], checks: list[Check]) -> None:
    profile = evidence.get("hpc_profile_readiness") or {}
    checks_payload = profile.get("checks") or {}
    boundaries = profile.get("boundaries") or {}
    failed = list(profile.get("failed_checks") or [])
    add(
        checks,
        "hpc_profile:job90353_v2_active_binding_ready",
        profile.get("status") == "ready" and not failed,
        severity="warning",
        status=profile.get("status"),
        failed_checks=failed,
        schema=profile.get("schema"),
        dpapi_decrypted=boundaries.get("dpapi_decrypted"),
        ssh_connections=boundaries.get("ssh_connections"),
        remote_commands=boundaries.get("remote_commands"),
        training_started=boundaries.get("training_started"),
        grader_calls=boundaries.get("grader_calls"),
        kaggle_submissions=boundaries.get("kaggle_submissions"),
        profile_state=(profile.get("details") or {}).get("profile_state"),
        schema_v2=checks_payload.get("schema_v2"),
        profile_state_active=checks_payload.get("profile_state_active"),
        container_binding_sha256_valid=checks_payload.get("container_binding_sha256_valid"),
        claim_boundary=profile.get("claim_boundary"),
        note=(
            "Warning-only for local demo. Full production/HPC data readiness "
            "requires a v2 active job90353 profile before any remote SIIM public-root audit."
        ),
    )


def check_job90353_reenrollment_plan(evidence: dict[str, Any], checks: list[Check]) -> None:
    plan = evidence.get("job90353_reenrollment_plan") or {}
    safety = plan.get("safety_boundaries") if isinstance(plan.get("safety_boundaries"), dict) else {}
    commands = plan.get("commands") if isinstance(plan.get("commands"), dict) else {}
    ok = (
        plan.get("schema") == "evomind.hpc.job90353_reenrollment_plan.v1"
        and plan.get("profile") == "job90353"
        and plan.get("status") in {"ready_for_operator_secret_entry", "already_ready_no_reenrollment_needed"}
        and safety.get("this_plan_decrypts_dpapi") is False
        and safety.get("this_plan_opens_ssh") is False
        and safety.get("this_plan_runs_remote_commands") is False
        and safety.get("this_plan_starts_training") is False
        and safety.get("this_plan_calls_grader") is False
        and safety.get("this_plan_submits_kaggle") is False
        and bool(commands.get("install_provisioning_profile"))
        and bool(commands.get("post_bootstrap_verification"))
    )
    add(
        checks,
        "hpc_profile:job90353_reenrollment_plan_ready",
        ok,
        severity="warning",
        status=plan.get("status"),
        schema=plan.get("schema"),
        profile=plan.get("profile"),
        install_command_present=bool(commands.get("install_provisioning_profile")),
        post_bootstrap_verification_present=bool(commands.get("post_bootstrap_verification")),
        safety_boundaries=safety,
        note=(
            "Warning-only for local demo. This proves the next re-enrollment work order "
            "is prepared; it does not prove the operator has entered secrets or bootstrapped HPC identity."
        ),
    )


def check_live_dashboard(checks: list[Check], require_live_latest_source: bool) -> dict[str, Any]:
    status, meta = run_dashboard_status()
    detail = {"meta": meta, "status": status}
    add(
        checks,
        "live_dashboard:8088_8765_ready",
        bool(
            status
            and status.get("status") == "running"
            and status.get("pid_running") is True
            and status.get("dashboard_identity_verified") is True
            and status.get("runtime_pid_running") is True
            and status.get("runtime_identity_verified") is True
            and (status.get("health") or {}).get("http_status") == 200
            and ((status.get("health") or {}).get("runtime") or {}).get("status") == "ready"
        ),
        **detail,
    )
    source_stale = bool(status and status.get("source_build_stale") is True)
    add(
        checks,
        "live_dashboard:source_build_stale",
        not source_stale,
        severity="blocker" if require_live_latest_source else "warning",
        source_build_stale=source_stale,
        require_live_latest_source=require_live_latest_source,
        note=(
            "Non-blocking for local demo unless --require-live-latest-source is set. "
            "If true, the dashboard is healthy but the live process has not been proven to serve the latest source build."
        ),
    )
    return status or {}


def summarize_status(checks: list[Check]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    blockers = [check.as_dict() for check in checks if not check.ok and check.severity == "blocker"]
    warnings = [check.as_dict() for check in checks if not check.ok and check.severity != "blocker"]
    return ("passed" if not blockers else "failed"), blockers, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "workspace/evaluation/local_evomind_release_gate_current.json")
    parser.add_argument("--max-age-hours", type=float, default=48.0)
    parser.add_argument(
        "--require-live-latest-source",
        action="store_true",
        help="Fail when live 8088 reports source_build_stale=true.",
    )
    parser.add_argument(
        "--require-release-source-tracked",
        dest="require_release_source_clean",
        action="store_true",
        help="Fail when Agent UX release-critical source files are missing, untracked, or dirty.",
    )
    parser.add_argument(
        "--require-release-source-clean",
        dest="require_release_source_clean",
        action="store_true",
        help="Alias for --require-release-source-tracked; requires release-critical source files to be tracked and clean.",
    )
    args = parser.parse_args()

    checks: list[Check] = []
    evidence: dict[str, Any] = {}

    check_evidence_files(args.max_age_hours, evidence, checks)
    check_optional_evidence_files(args.max_age_hours, evidence, checks)
    check_agent_quality(evidence, checks)
    check_governance(evidence, checks)
    check_new_user_readiness(evidence, checks)
    check_web_contracts(evidence, checks)
    check_live_assistant_demo(evidence, checks)
    check_hpc_profile_readiness(evidence, checks)
    check_job90353_reenrollment_plan(evidence, checks)
    live_status = check_live_dashboard(checks, args.require_live_latest_source)
    source_tracking = check_release_source_tracking(
        checks,
        require_release_source_tracked=args.require_release_source_clean,
    )
    check_release_source_scope_manifest(
        evidence,
        checks,
        source_tracking,
        max_age_hours=args.max_age_hours,
        require_release_source_clean=args.require_release_source_clean,
    )
    check_release_source_manifest_secret_scan(
        evidence,
        checks,
        require_release_source_clean=args.require_release_source_clean,
    )

    status, blockers, warnings = summarize_status(checks)
    report = {
        "schema": "evomind.local_release_gate.v1",
        "generated_at": now_iso(),
        "status": status,
        "release_state": "local_demo_pre_release_ready" if status == "passed" else "no_go",
        "goal_reference": "019fa4da-c818-7642-8541-76e31d6b5886",
        "target_siim_run": TARGET_SIIM_RUN,
        "scope": {
            "proves": [
                "local 8088/8765 gateway health",
                "novice natural-language Agent UX representative gate",
                "LLM-first tool-loop behavior with openai/gpt-5.6-sol",
                "authenticated localhost security contracts",
                "isolated production web build without disrupting live demo service",
                "browser-equivalent authenticated four-turn assistant stream smoke with real tool calls, follow-up context, reviewed literature, and job90673 connection self-check",
                "local read-only visibility into job90353 HPC profile readiness when evidence is present",
            ],
            "does_not_prove": [
                "official Kaggle rank or medal",
                "new private grader result",
                "full MLE-Bench Lite A/B uplift",
                "release bundle/video/Claim Audit/tag completion",
                "live 8088 serving latest source unless --require-live-latest-source passes",
                "clean checkout preserving dirty or untracked Agent UX fixes unless --require-release-source-tracked passes",
                "A800/HPC container readiness when job90353 profile readiness is warning/failed",
            ],
        },
        "release_source_policy": {
            "require_release_source_tracked": args.require_release_source_clean,
            "require_release_source_clean": args.require_release_source_clean,
            "local_demo_behavior": "warn_only",
            "formal_release_behavior": "blocker_when_strict_source_flag_enabled",
        },
        "checks_total": len(checks),
        "checks_passed": sum(1 for check in checks if check.ok),
        "blockers": blockers,
        "warnings": warnings,
        "live_dashboard": {
            "status": live_status.get("status"),
            "url": live_status.get("url"),
            "pid": live_status.get("pid"),
            "runtime_pid": live_status.get("runtime_pid"),
            "source_build_stale": live_status.get("source_build_stale"),
            "health": live_status.get("health"),
        },
        "evidence_files": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path) if path.exists() else None,
                "age_hours": round(age_hours(path), 3) if path.exists() else None,
            }
            for name, path in REQUIRED_EVIDENCE.items()
        }
        | {
            name: {
                "path": str(path),
                "sha256": file_sha256(path) if path.exists() else None,
                "age_hours": round(age_hours(path), 3) if path.exists() else None,
            }
            for name, path in OPTIONAL_EVIDENCE.items()
        },
        "checks": [check.as_dict() for check in checks],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
