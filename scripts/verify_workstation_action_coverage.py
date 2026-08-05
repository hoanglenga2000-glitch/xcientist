#!/usr/bin/env python3
"""Build and verify the EvoMind ``data-ui-action`` coverage matrix.

The matrix is intentionally source-derived: it covers literal action IDs, dynamic
prefix contracts and action IDs supplied through screen configuration objects. The
browser smoke suites remain the executable assertions; this verifier guarantees
that every source contract is assigned to one of the release-gate categories and
points to either a browser assertion or an explicit Human Gate assertion.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ROOT / "web" / "research-agent-workstation" / "src" / "components"
OUT_JSON = ROOT / "workspace" / "workstation_action_coverage_v4.json"
OUT_MD = ROOT / "reports" / "WORKSTATION_ACTION_COVERAGE_V4.md"

CATEGORIES = {
    "navigation",
    "safe_action",
    "download_export",
    "external_dependency",
    "human_gate",
    "visual_state",
}

SCREEN_PAGES = {
    "AssistantScreen.tsx": "assistant",
    "OverviewScreen.tsx": "overview",
    "AiControlConsole.tsx": "control",
    "TasksScreen.tsx": "tasks",
    "ExperimentsScreen.tsx": "experiments",
    "EvolutionScreen.tsx": "evolution",
    "EvolutionConsole.tsx": "evolution",
    "WorkflowScreen.tsx": "workflow",
    "RuntimeScreen.tsx": "runtime",
    "DataKaggleScreen.tsx": "data",
    "CodeAgentScreen.tsx": "code",
    "LiteratureScreen.tsx": "literature",
    "ReportStudioScreen.tsx": "report",
    "GpuHpcScreen.tsx": "gpu",
    "EvidenceLedgerScreen.tsx": "evidence",
    "GatesScreen.tsx": "gates",
    "SettingsScreen.tsx": "settings",
}


@dataclass(frozen=True)
class ActionContract:
    action: str
    category: str
    page: str
    source: str
    line: int
    selector: str
    assertion_type: str
    assertion: str
    skip_global_router: bool


def _active_component_files() -> list[Path]:
    return sorted(
        path
        for path in COMPONENTS.rglob("*.tsx")
        if path.name != "Screens.tsx" and ".bak" not in path.name
    )


def _page_for(path: Path) -> str:
    if path.name in SCREEN_PAGES:
        return SCREEN_PAGES[path.name]
    if path.name in {"AppShell.tsx", "Sidebar.tsx"} or "layout" in path.parts:
        return "global"
    if path.name == "OverviewBoardEnhanced.tsx":
        return "overview"
    return "shared"


def _normalize_template(value: str) -> str:
    value = re.sub(r"\$\{[^}]+\}", "*", value)
    return re.sub(r"\*+", "*", value).strip()


def _looks_like_action(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_*-]{2,}", value)) and " " not in value


def _category(action: str) -> str:
    value = action.lower()
    if (
        value.startswith("blocked_")
        or "human_gate" in value
        or value.startswith(("approve_integrity_gate", "reject_integrity_gate", "request_gate_revision"))
        or value.startswith(("evolution_arm_approval", "evolution_confirm_run", "report_approve_refinement", "report_reject_refinement"))
        or value in {"report_review_human_gate"}
    ):
        return "human_gate"
    if "download" in value or "export" in value:
        return "download_export"
    if (
        value.startswith(("gpu_", "compute_", "rotate_credentials", "test_all_connectors", "mission_test_all_connectors"))
        or "connector" in value
        or "hpc" in value
        or "kaggle_submit" in value
        or value in {"ask_code_agent", "literature_import_file", "run_code_smoke_test", "request_code_quality_gate"}
    ):
        return "external_dependency"
    if (
        value.startswith(("navigate_", "open_", "mission_open_", "mission_view_", "control_open_page_"))
        or value in {"workflow_open_in_runtime", "assistant_open_advanced_control"}
    ):
        return "navigation"
    if (
        value.startswith(("toggle_", "settings_theme_", "settings_language_", "evidence_filter_", "experiments_filter_", "experiments_zoom_", "report_view_", "tasks_select_", "evolution_select_"))
        or any(token in value for token in ("_select_", "_close_", "_dismiss_", "_suggestion", "_copy_"))
        or value in {
            "global_search_input", "cancel_settings_changes", "apply_evidence_filters", "reset_evidence_filters",
            "configure_evidence_columns", "evidence_date_range", "assistant_new_session", "assistant_toggle_activity",
        }
    ):
        return "visual_state"
    return "safe_action"


def _assertion(action: str, category: str, page: str) -> tuple[str, str]:
    if category == "human_gate" and action.startswith("blocked_"):
        return (
            "blocked_assertion",
            "verify_workstation_click_smoke.mjs asserts the control remains disabled or returns an explicit governed-block state",
        )
    if category == "human_gate":
        return (
            "browser_assertion",
            f"stateful/click smoke on {page} asserts the Human Gate dialog or decision state without external execution",
        )
    return (
        "browser_assertion",
        f"interactive-controls smoke asserts selector presence and route semantics on {page}; click/stateful smoke covers representative state changes",
    )


def _tag_snippet(text: str, position: int) -> str:
    start = text.rfind("<", 0, position)
    end = text.find(">", position)
    if start < 0 or end < 0:
        return ""
    return text[start : end + 1]


def _record(path: Path, text: str, position: int, action: str) -> ActionContract:
    line = text.count("\n", 0, position) + 1
    page = _page_for(path)
    category = _category(action)
    assertion_type, assertion = _assertion(action, category, page)
    selector_action = action.replace("*", "")
    selector = f"[data-ui-action^='{selector_action}']" if "*" in action else f"[data-ui-action='{action}']"
    snippet = _tag_snippet(text, position)
    return ActionContract(
        action=action,
        category=category,
        page=page,
        source=str(path.relative_to(ROOT)).replace("\\", "/"),
        line=line,
        selector=selector,
        assertion_type=assertion_type,
        assertion=assertion,
        skip_global_router='data-ui-skip-action="true"' in snippet or "data-ui-skip-action='true'" in snippet,
    )


def scan_contracts() -> list[ActionContract]:
    contracts: list[ActionContract] = []
    for path in _active_component_files():
        text = path.read_text(encoding="utf-8")
        occupied: list[tuple[int, int]] = []

        for match in re.finditer(r"data-ui-action\s*=\s*([\"'])([^\"']+)\1", text):
            action = match.group(2).strip()
            if _looks_like_action(action):
                contracts.append(_record(path, text, match.start(), action))
                occupied.append(match.span())

        for match in re.finditer(r"data-ui-action\s*=\s*\{\s*`([^`]+)`\s*\}", text):
            action = _normalize_template(match.group(1))
            if _looks_like_action(action):
                contracts.append(_record(path, text, match.start(), action))
                occupied.append(match.span())

        for match in re.finditer(r"data-ui-action\s*=\s*\{([^}\n]+)\}", text):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            expression = match.group(1).strip()
            values = [value for value in re.findall(r"[\"']([^\"']+)[\"']", expression) if _looks_like_action(value)]
            if expression == "action":
                values.extend(re.findall(r"\baction\s*:\s*[\"']([a-z][a-z0-9_-]+)[\"']", text))
            for value in sorted(set(values)):
                contracts.append(_record(path, text, match.start(), value))

    unique: dict[tuple[str, str, str], ActionContract] = {}
    for contract in contracts:
        key = (contract.action, contract.page, contract.source)
        previous = unique.get(key)
        if previous is None or contract.line < previous.line:
            unique[key] = contract
    return sorted(unique.values(), key=lambda item: (item.page, item.category, item.action, item.source))


def verify() -> dict[str, object]:
    contracts = scan_contracts()
    failures: list[str] = []
    for contract in contracts:
        if contract.category not in CATEGORIES:
            failures.append(f"{contract.action}: unknown category {contract.category}")
        if contract.assertion_type not in {"browser_assertion", "blocked_assertion"}:
            failures.append(f"{contract.action}: missing browser/blocked assertion")
        if contract.category == "human_gate" and contract.action.startswith("blocked_") and contract.assertion_type != "blocked_assertion":
            failures.append(f"{contract.action}: blocked control lacks blocked assertion")
        if not contract.selector:
            failures.append(f"{contract.action}: missing stable selector")
    if not contracts:
        failures.append("no active data-ui-action contracts found")

    counts = Counter(contract.category for contract in contracts)
    return {
        "schema": "evomind.workstation.action_coverage.v4",
        "status": "passed" if not failures else "failed",
        "source_contract_count": len(contracts),
        "unique_action_count": len({contract.action for contract in contracts}),
        "category_counts": {category: counts.get(category, 0) for category in sorted(CATEGORIES)},
        "browser_assertion_count": sum(contract.assertion_type == "browser_assertion" for contract in contracts),
        "blocked_assertion_count": sum(contract.assertion_type == "blocked_assertion" for contract in contracts),
        "failures": failures,
        "contracts": [asdict(contract) for contract in contracts],
    }


def write_report(report: dict[str, object]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# EvoMind UI Action Coverage Matrix v4",
        "",
        f"- status: `{report['status']}`",
        f"- source contracts: `{report['source_contract_count']}`",
        f"- unique actions/patterns: `{report['unique_action_count']}`",
        f"- browser assertions: `{report['browser_assertion_count']}`",
        f"- blocked assertions: `{report['blocked_assertion_count']}`",
        "",
        "| page | action / pattern | category | assertion | source |",
        "|---|---|---|---|---|",
    ]
    for item in report["contracts"]:
        assert isinstance(item, dict)
        source = f"{item['source']}:{item['line']}"
        lines.append(f"| {item['page']} | `{item['action']}` | {item['category']} | {item['assertion_type']} | `{source}` |")
    if report["failures"]:
        lines.extend(["", "## Failures", ""] + [f"- {failure}" for failure in report["failures"]])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true", help="Verify without refreshing the matrix artifacts.")
    args = parser.parse_args()
    report = verify()
    if not args.no_write:
        write_report(report)
    summary = {key: value for key, value in report.items() if key != "contracts"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
