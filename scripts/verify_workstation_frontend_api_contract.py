from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "research-agent-workstation"
SRC = WEB / "src"
CLIENT_TS = SRC / "lib" / "api" / "client.ts"
NAVIGATION_TS = SRC / "components" / "workstation" / "navigation.ts"
PAGE_TSX = SRC / "app" / "page.tsx"
APP_SHELL_TSX = SRC / "components" / "workstation" / "AppShell.tsx"
AI_CONTROL_TSX = SRC / "components" / "workstation" / "AiControlConsole.tsx"
RUNTIME_SCREEN_TSX = SRC / "components" / "workstation" / "screens" / "RuntimeScreen.tsx"
RUN_CONTEXT_TSX = SRC / "components" / "workstation" / "layout" / "RunContextBar.tsx"
TASK_CONTEXT_TS = SRC / "lib" / "task-context.ts"
LITERATURE_IMPORT_TS = SRC / "app" / "api" / "literature" / "import" / "route.ts"
LITERATURE_SEARCH_TS = SRC / "app" / "api" / "literature" / "search" / "route.ts"
OUT_JSON = ROOT / "workspace" / "workstation_frontend_api_contract_20260630.json"
OUT_MD = ROOT / "reports" / "WORKSTATION_FRONTEND_API_CONTRACT_20260630.md"


METHOD_PATTERN = re.compile(r"method:\s*[\"'](?P<method>GET|POST|PATCH|PUT|DELETE)[\"']", re.IGNORECASE)
DYNAMIC_SEGMENT_PATTERN = re.compile(r"\$\{\s*(?P<expression>.*?)\s*\}")
IDENTIFIER_PATTERN = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
ENCODED_IDENTIFIER_PATTERN = re.compile(
    r"encodeURIComponent\(\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\)"
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def dynamic_segment_name(part: str) -> str | None:
    dynamic_match = DYNAMIC_SEGMENT_PATTERN.fullmatch(part)
    if not dynamic_match:
        return None
    expression = dynamic_match.group("expression").strip()
    for pattern in (IDENTIFIER_PATTERN, ENCODED_IDENTIFIER_PATTERN):
        match = pattern.fullmatch(expression)
        if match:
            return match.group("name")
    return None


def endpoint_to_route(endpoint: str) -> tuple[Path | None, str | None]:
    if not endpoint.startswith("/api/"):
        return None, "not_api_endpoint"
    endpoint = endpoint.split("?", 1)[0]
    route_parts: list[str] = []
    for part in endpoint.strip("/").split("/")[1:]:
        if DYNAMIC_SEGMENT_PATTERN.fullmatch(part):
            dynamic_name = dynamic_segment_name(part)
            if not dynamic_name:
                return None, f"unsupported_dynamic_segment:{part}"
            route_parts.append(f"[{dynamic_name}]")
        else:
            route_parts.append(part)
    return SRC / "app" / "api" / Path(*route_parts) / "route.ts", None


def iter_fetch_calls(text: str) -> list[tuple[int, str]]:
    calls: list[tuple[int, str]] = []
    cursor = 0
    while True:
        start = text.find("fetch(", cursor)
        if start == -1:
            break
        depth = 0
        quote: str | None = None
        escaped = False
        end = None
        for index in range(start, len(text)):
            char = text[index]
            if quote:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == quote:
                    quote = None
                    continue
                continue
            if char in {'"', "'", "`"}:
                quote = char
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            break
        calls.append((start, text[start:end]))
        cursor = end
    return calls


def extract_endpoint(call_text: str) -> str | None:
    match = re.search(r"fetch\(\s*([`\"])(?P<endpoint>.*?)(?:\1)", call_text, re.DOTALL)
    return match.group("endpoint") if match else None


def extract_fetch_contracts() -> list[dict[str, Any]]:
    text = read(CLIENT_TS)
    contracts: list[dict[str, Any]] = []
    for start, call_text in iter_fetch_calls(text):
        endpoint = extract_endpoint(call_text)
        if not endpoint:
            continue
        method_match = METHOD_PATTERN.search(call_text)
        method = method_match.group("method").upper() if method_match else "GET"
        line = text[:start].count("\n") + 1
        route_path, skip_reason = endpoint_to_route(endpoint)
        exists = bool(route_path and route_path.exists())
        route_text = read(route_path) if exists and route_path else ""
        method_exported = bool(re.search(rf"export\s+async\s+function\s+{method}\b", route_text))
        contracts.append({
            "endpoint": endpoint,
            "method": method,
            "client_file": str(CLIENT_TS.relative_to(ROOT)).replace("\\", "/"),
            "client_line": line,
            "route_file": str(route_path.relative_to(ROOT)).replace("\\", "/") if route_path else None,
            "route_exists": exists,
            "method_exported": method_exported,
            "skip_reason": skip_reason,
            "ok": bool(route_path) and exists and method_exported,
        })
    return contracts


def extract_quoted_ids(pattern: str, text: str) -> list[str]:
    return re.findall(pattern, text)


def build_navigation_contract() -> dict[str, Any]:
    navigation_text = read(NAVIGATION_TS)
    page_text = read(PAGE_TSX)
    page_type_block = re.search(r"export\s+type\s+PageId\s*=([\s\S]*?);", navigation_text)
    page_type_ids = re.findall(r'"([^"]+)"', page_type_block.group(1)) if page_type_block else []
    nav_items_block = re.search(r"export\s+const\s+navItems\s*=\s*\[([\s\S]*?)\]\s+as\s+const", navigation_text)
    nav_ids = (
        re.findall(r'\{\s*id:\s*"([^"]+)"\s*,\s*label:', nav_items_block.group(1))
        if nav_items_block
        else extract_quoted_ids(r'\{\s*id:\s*"([^"]+)"\s*,\s*label:', navigation_text)
    )
    rendered_ids = extract_quoted_ids(r'activePage\s*===\s*"([^"]+)"', page_text)
    page_ids_array_match = re.search(r"const\s+pageIds\s*=\s*\[([\s\S]*?)\]\s+as\s+const", page_text)
    page_ids_array = re.findall(r'"([^"]+)"', page_ids_array_match.group(1)) if page_ids_array_match else []
    aliases = {
        "mission": "overview" if 'normalized === "mission"' in page_text else None,
        "evidence-detail": "evidence" if 'normalized === "evidence-detail"' in page_text else None,
        "design": "settings" if 'normalized === "design"' in page_text else None,
    }
    missing_from_type = [page for page in nav_ids if page not in page_type_ids]
    missing_from_render = [page for page in nav_ids if page not in rendered_ids]
    missing_from_page_parser = [page for page in nav_ids if page not in page_ids_array]
    rendered_not_in_nav = [page for page in rendered_ids if page not in nav_ids and page != "design"]
    return {
        "page_type_ids": page_type_ids,
        "nav_ids": nav_ids,
        "rendered_ids": rendered_ids,
        "page_ids_array": page_ids_array,
        "aliases": aliases,
        "missing_from_type": missing_from_type,
        "missing_from_render": missing_from_render,
        "missing_from_page_parser": missing_from_page_parser,
        "rendered_not_in_nav": rendered_not_in_nav,
        "ok": not (missing_from_type or missing_from_render or missing_from_page_parser or rendered_not_in_nav),
    }


def build_literature_provenance_contract() -> dict[str, Any]:
    import_text = read(LITERATURE_IMPORT_TS) if LITERATURE_IMPORT_TS.exists() else ""
    search_text = read(LITERATURE_SEARCH_TS) if LITERATURE_SEARCH_TS.exists() else ""
    trust_weights = {
        source: int(match.group(1)) if (match := re.search(rf"\b{source}:\s*(\d+)", search_text)) else -1
        for source in ("imported", "arxiv", "openalex", "crossref", "local", "internal")
    }
    checks = {
        "import_route_exists": LITERATURE_IMPORT_TS.exists(),
        "search_route_exists": LITERATURE_SEARCH_TS.exists(),
        "import_uses_sha256": 'createHash("sha256")' in import_text,
        "import_preserves_source_artifact": "artifact_path: sourceRelative" in import_text,
        "import_preserves_checksum_provenance": (
            'source: "imported"' in import_text
            and 'provenance: { verified: true, source: "user_import"' in import_text
            and "checksum" in import_text
        ),
        "search_loads_imported_manifests": "loadImportedPapers(taskId)" in search_text,
        "search_preserves_imported_record": "...paper, score: scoreText" in search_text,
        "search_ranks_relevance_before_source_trust": bool(
            re.search(r"\.sort\(\(a, b\) => b\.score - a\.score \|\| \(sourceTrust", search_text)
        ),
        "unrelated_imports_do_not_displace_relevant_results": bool(
            re.search(r"const relevantPapers =[\s\S]*?\.slice\(0, maxResults\);", search_text)
        ),
        "user_imports_remain_pinned_to_task_library": (
            "const pinnedImports = dedupePapers" in search_text
            and "const papers = dedupePapers([...relevantPapers, ...pinnedImports])" in search_text
        ),
        "imported_records_keep_distinct_provenance": 'paper.source === "imported"' in search_text,
        "result_limit_applied_after_dedupe": bool(
            re.search(r"const relevantPapers = dedupePapers\([\s\S]*?\)\.slice\(0,\s*maxResults\)", search_text)
        ),
    }
    return {
        "trust_weights": trust_weights,
        "checks": checks,
        "ok": all(checks.values()),
    }


def build_terminal_task_sync_contract() -> dict[str, Any]:
    page_text = read(PAGE_TSX)
    shell_text = read(APP_SHELL_TSX)
    control_text = read(AI_CONTROL_TSX)
    runtime_screen_text = read(RUNTIME_SCREEN_TSX)
    run_context_text = read(RUN_CONTEXT_TSX)
    task_context_text = read(TASK_CONTEXT_TS) if TASK_CONTEXT_TS.exists() else ""
    summary_text = read(SRC / "lib" / "server" / "summary.ts")
    checks = {
        "task_context_helper_exists": TASK_CONTEXT_TS.exists(),
        "hyphenated_task_ids_are_normalized": '.replace(/-/g, "_")' in task_context_text,
        "current_run_signal_supported": 'source: "current_run"' in task_context_text,
        "current_run_is_authoritative": (
            "const currentRun = summary.runtime?.current_run" in task_context_text
            and "if (currentRunTaskId)" in task_context_text
            and task_context_text.index("if (currentRunTaskId)") < task_context_text.index("const candidates = [")
        ),
        "current_run_signal_is_bound_to_run_id": 'currentRun?.run_id ?? "unknown"' in task_context_text,
        "terminal_turn_signal_supported": 'source: "scientist_terminal_turn"' in task_context_text,
        "summary_loads_terminal_turn": "loadScientistTerminalTurnSummary()" in summary_text,
        "summary_exposes_terminal_turn": "scientist_terminal_turn: scientistTerminalTurn" in summary_text,
        "context_packet_signal_supported": 'source: "scientist_context_packet"' in task_context_text,
        "terminal_agent_signal_supported": 'source: "terminal_agent"' in task_context_text,
        "signals_are_ordered_by_timestamp": "timestamp(right.observedAt) - timestamp(left.observedAt)" in task_context_text,
        "page_consumes_latest_signal": "latestTaskSignal(payload)" in page_text,
        "same_signal_is_idempotent": "latestAppliedTaskSignalRef.current !== signal.key" in page_text,
        "explicit_url_task_is_preserved": "isInitialSignal && explicitTaskFromUrlRef.current" in page_text,
        "task_deeplink_is_persisted": 'url.searchParams.set("task", taskId)' in page_text,
        "summary_and_task_are_batched": "const applySummaryPayload" in page_text and "setSummary(payload)" in page_text,
        "selected_task_reaches_shell": "selectedTask={selectedTask}" in page_text and "selectedTask?: string" in shell_text,
        "page_ready_reaches_shell": "ready={summary !== null}" in page_text and "ready?: boolean" in shell_text,
        "shell_exposes_readiness": 'data-ui-ready={ready ? "true" : "false"}' in shell_text and 'data-ui-task={selectedTask ?? ""}' in shell_text,
        "current_turn_reasoning_takes_precedence": "scientistTerminalTurn?.reasoning_synthesis ?? scientistReasoningSynthesis" in control_text,
        "read_only_turn_defers_execution_gates": (
            "scientistTerminalTurn?.read_only_completed === true" in control_text
            and "not_requested_read_only" in control_text
            and "current read-only turn completed" in control_text
        ),
        "run_context_filters_by_task": "recordMatchesTask(run, normalizedTask)" in run_context_text,
        "evidence_filters_by_task": "recordMatchesTask(item, normalizedTask)" in run_context_text,
        "gates_filter_by_task": "recordMatchesTask(gate, normalizedTask)" in run_context_text,
        "mobile_drawer_keeps_task_context": "selectedTask={selectedTask}" in run_context_text,
        "runtime_prefers_current_run_events": (
            bool(
                re.search(
                    r"const\s+(?:sourceEventLog|eventLog)\s*=\s*hasCurrentRun"
                    r"[\s\S]*?\?\s*runtime\?\.event_log\s*\?\?\s*\[\]",
                    runtime_screen_text,
                )
            )
        ),
        "runtime_legacy_events_are_fallback_only": (
            "Legacy Scientist/Terminal events" in runtime_screen_text
            and "terminalAgent?.recent_terminal_events?.length" in runtime_screen_text
        ),
        "runtime_renders_multi_agent_closure": all(
            marker in runtime_screen_text
            for marker in (
                "Current Run",
                "Dynamic Task Graph",
                "Current Run Event Stream",
                "HPC Runtime",
                "Reviewed Candidates",
                "Report & Deliverables",
            )
        ),
        "runtime_gate_is_kaggle_submission_only": (
            "Official Kaggle Submission" in runtime_screen_text
            and "HPC training completed" in runtime_screen_text
            and "GPU Submission (Human Gate)" not in runtime_screen_text
        ),
        "control_receives_summary": (
            "summary?: WorkstationSummary | null" in control_text
            and "const currentRun = currentRuntime?.current_run" in control_text
        ),
        "control_current_run_precedes_legacy_scientist": (
            "Current Multi-Agent Run" in control_text
            and "Scientist Tools (Historical / Auxiliary)" in control_text
            and control_text.index("Current Multi-Agent Run") < control_text.index("Scientist Tools (Historical / Auxiliary)")
        ),
    }
    return {"checks": checks, "ok": all(checks.values())}


def build_report() -> dict[str, Any]:
    fetch_contracts = extract_fetch_contracts()
    navigation_contract = build_navigation_contract()
    literature_provenance_contract = build_literature_provenance_contract()
    terminal_task_sync_contract = build_terminal_task_sync_contract()
    failed_contracts = [item for item in fetch_contracts if not item["ok"]]
    return {
        "schema": "academic_research_os.frontend_api_contract.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "passed" if not failed_contracts and navigation_contract["ok"] and literature_provenance_contract["ok"] and terminal_task_sync_contract["ok"] else "failed",
        "fetch_contract_count": len(fetch_contracts),
        "failed_fetch_contracts": failed_contracts,
        "navigation_contract": navigation_contract,
        "literature_provenance_contract": literature_provenance_contract,
        "terminal_task_sync_contract": terminal_task_sync_contract,
        "fetch_contracts": fetch_contracts,
        "claim_boundary": "This check proves frontend API client route/method bindings and navigation-page coverage. It does not prove business-side training success or Kaggle scores.",
    }


def write_markdown(report: dict[str, Any]) -> None:
    nav = report["navigation_contract"]
    literature = report["literature_provenance_contract"]
    task_sync = report["terminal_task_sync_contract"]
    lines = [
        "# 工作站前端 API 与导航契约检查",
        "",
        f"- 生成时间：`{report['created_at']}`",
        f"- 总状态：`{report['status']}`",
        f"- API client fetch 数量：`{report['fetch_contract_count']}`",
        f"- 失败绑定数量：`{len(report['failed_fetch_contracts'])}`",
        "",
        "## 导航与页面覆盖",
        "",
        f"- PageId 类型数量：`{len(nav['page_type_ids'])}`",
        f"- 侧边栏导航数量：`{len(nav['nav_ids'])}`",
        f"- 实际渲染页面数量：`{len(nav['rendered_ids'])}`",
        f"- URL parser 页面数量：`{len(nav['page_ids_array'])}`",
        f"- missing from PageId：`{', '.join(nav['missing_from_type']) or 'none'}`",
        f"- missing from render：`{', '.join(nav['missing_from_render']) or 'none'}`",
        f"- missing from URL parser：`{', '.join(nav['missing_from_page_parser']) or 'none'}`",
        "",
        "## 文献导入与联合检索完整性",
        "",
        f"- 总状态：`{'passed' if literature['ok'] else 'failed'}`",
        f"- 来源可信度（仅用于相关性同分排序）：`{json.dumps(literature['trust_weights'], ensure_ascii=False)}`",
        *[f"- {name}：`{value}`" for name, value in literature["checks"].items()],
        "",
        "## 终端任务与前端上下文同步",
        "",
        f"- 总状态：`{'passed' if task_sync['ok'] else 'failed'}`",
        *[f"- {name}：`{value}`" for name, value in task_sync["checks"].items()],
        "",
        "## API Client 绑定",
        "",
        "| method | endpoint | route | exists | method exported |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report["fetch_contracts"]:
        lines.append(
            f"| `{item['method']}` | `{item['endpoint']}` | `{item['route_file']}` | `{item['route_exists']}` | `{item['method_exported']}` |"
        )
    if report["failed_fetch_contracts"]:
        lines.extend(["", "## 失败项", ""])
        for item in report["failed_fetch_contracts"]:
            lines.append(f"- `{item['method']} {item['endpoint']}` -> `{item['route_file']}`")
    lines.extend(["", "## Claim Boundary", "", report["claim_boundary"]])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify frontend API client route/method bindings and navigation coverage.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if args.write_report:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(report)
    print(json.dumps({
        "status": report["status"],
        "fetch_contract_count": report["fetch_contract_count"],
        "failed_fetch_contract_count": len(report["failed_fetch_contracts"]),
        "navigation_ok": report["navigation_contract"]["ok"],
        "literature_provenance_ok": report["literature_provenance_contract"]["ok"],
        "terminal_task_sync_ok": report["terminal_task_sync_contract"]["ok"],
        "json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
        "md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
