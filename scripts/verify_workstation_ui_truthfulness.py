from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = ROOT / "web" / "research-agent-workstation" / "src"
RUNTIME_UI_FILES = [
    UI_ROOT / "app" / "home-client.tsx",
    UI_ROOT / "components" / "workstation" / "AppShell.tsx",
    UI_ROOT / "components" / "workstation" / "Screens.tsx",
    UI_ROOT / "components" / "workstation" / "Sidebar.tsx",
    UI_ROOT / "lib" / "connector-status.ts",
    UI_ROOT / "lib" / "server" / "kaggle-status.ts",
]
SCREENS = UI_ROOT / "components" / "workstation" / "Screens.tsx"


FORBIDDEN_RULES = {
    "invented_product_version": re.compile(r"\bv2\.3\.1\b", re.IGNORECASE),
    "invented_two_factor_state": re.compile(
        r"已登录\s*\(2FA\)|2FA\s*已启用|双因素[^\n]{0,24}(?:已启用|开启|enabled)",
        re.IGNORECASE,
    ),
    "invented_role_user_counts": re.compile(
        r'Research Admin"\s*,\s*"3"|Reviewer"\s*,\s*"5"|Agent Operator"\s*,\s*"8"',
        re.IGNORECASE,
    ),
    "invented_kaggle_quota": re.compile(
        r"Kaggle[^\n]{0,140}98%|98%[^\n]{0,140}Kaggle",
        re.IGNORECASE,
    ),
    "invented_cache_hit_rate": re.compile(
        r"(?:缓存|cache)[^\n]{0,140}(?:86|92)%|(?:86|92)%[^\n]{0,140}(?:缓存|cache)",
        re.IGNORECASE,
    ),
    "invented_evidence_coverage": re.compile(
        r"Evidence\s+coverage[^\n]{0,100}92%|415\s*/\s*450|1,248\s*/\s*1,312",
        re.IGNORECASE,
    ),
    "invented_gpu_ready_state": re.compile(
        r"A800 Cluster\s*\(8 GPUs\)|(?:\d+x|1x\s+)A800\s*80GB|1x\s*A800\s*可用|nvidia-smi:All 8 GPUs OK",
        re.IGNORECASE,
    ),
    "legacy_evidence_fixture": re.compile(
        r"metrics_20260625_1001|2025-06-25T10:30:12Z|Artifact\s*验证通过|GPU Job\s*拉回",
        re.IGNORECASE,
    ),
    "legacy_gpu_job_fixture": re.compile(r"jb_20260625_00(?:10|11|12|13)", re.IGNORECASE),
}

REQUIRED_SCREEN_BINDINGS = {
    "connector_status_display": "deriveConnectorDisplays(",
    "gpu_current_gate": "current_gate_ready === true",
    "summary_evidence": "const evidenceRecords = (s?.evidence ?? [])",
    "summary_actions": "const relevantActions = (s?.actions ?? [])",
    "detail_reuses_truthful_ledger": "return <EvidenceLedger {...props} />;",
}
REQUIRED_CONNECTOR_BINDINGS = {
    "failure_state_precedence": "hasFailureState(",
    "gpu_requires_configured": "gpu.configured === true && gpu.current_gate_ready === true",
    "kaggle_requires_configured": "kaggle.configured === true && kaggle.authenticated === true",
    "human_gate_requires_evidence": "human_gate_required_for_submission === true",
}
CONNECTOR_STATUS = UI_ROOT / "lib" / "connector-status.ts"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def line_number(text: str, offset: int) -> int:
    return text[:offset].count("\n") + 1


def main() -> int:
    findings: list[dict[str, object]] = []
    scanned: list[str] = []
    for path in RUNTIME_UI_FILES:
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        if not path.exists():
            findings.append({"rule": "runtime_ui_file_missing", "file": relative, "line": None, "match": "missing"})
            continue
        text = read(path)
        scanned.append(relative)
        for rule, pattern in FORBIDDEN_RULES.items():
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "rule": rule,
                        "file": relative,
                        "line": line_number(text, match.start()),
                        "match": match.group(0)[:180],
                    }
                )

    screens_text = read(SCREENS) if SCREENS.exists() else ""
    missing_bindings = [name for name, marker in REQUIRED_SCREEN_BINDINGS.items() if marker not in screens_text]
    connector_text = read(CONNECTOR_STATUS) if CONNECTOR_STATUS.exists() else ""
    missing_connector_bindings = [
        name for name, marker in REQUIRED_CONNECTOR_BINDINGS.items() if marker not in connector_text
    ]
    status = "passed" if not findings and not missing_bindings and not missing_connector_bindings else "failed"
    payload = {
        "schema": "evomind.workstation_ui_truthfulness.v1",
        "status": status,
        "scanned_files": scanned,
        "forbidden_rule_count": len(FORBIDDEN_RULES),
        "findings": findings,
        "required_bindings": REQUIRED_SCREEN_BINDINGS,
        "missing_bindings": missing_bindings,
        "required_connector_bindings": REQUIRED_CONNECTOR_BINDINGS,
        "missing_connector_bindings": missing_connector_bindings,
        "claim_boundary": (
            "This static gate rejects known invented runtime UI claims and requires summary/connector bindings. "
            "It does not prove external connector health, GPU availability, experiment success, or visual correctness."
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
