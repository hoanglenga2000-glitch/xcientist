from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "workspace" / "workstation_figma_design_gate_20260630.json"
OUT_MD = ROOT / "reports" / "WORKSTATION_FIGMA_DESIGN_GATE_20260630.md"


DEFAULT_FILE_KEY = "YRGlARCURv2sKKmSHeNWA6"
DEFAULT_NODE_ID = "0:1"


def classify_probe(probe_status: str | None) -> dict[str, Any]:
    raw = (probe_status or "").strip()
    lowered = raw.lower()
    if not raw:
        return {
            "status": "not_checked",
            "verification_level": "none",
            "blocked": True,
            "blocker": "figma_probe_missing",
            "reason": "No live Figma metadata/screenshot probe was supplied to this verifier.",
        }
    if "token_revoked" in lowered or "http 401" in lowered or "401:" in lowered:
        return {
            "status": "blocked_figma_auth",
            "verification_level": "none",
            "blocked": True,
            "blocker": "figma_auth_blocked",
            "reason": "Figma MCP returned an invalidated OAuth token response.",
        }
    if (
        "don't have edit access" in lowered
        or "do not have edit access" in lowered
        or "no edit access" in lowered
        or "permission" in lowered
        or "invalid_argument" in lowered
    ):
        return {
            "status": "blocked_figma_access",
            "verification_level": "none",
            "blocked": True,
            "blocker": "figma_access_blocked",
            "reason": "Figma MCP could not read the target node because the current connector account does not have sufficient file access.",
        }
    if (
        ("metadata" in lowered or "xml" in lowered or "node" in lowered)
        and "screenshot" in lowered
        and ("editable" in lowered or "frame" in lowered)
        and "failed" not in lowered
        and "error" not in lowered
    ):
        return {
            "status": "metadata_and_screenshot_probe_reported",
            "verification_level": "metadata_screenshot_editable_structure",
            "blocked": False,
            "blocker": None,
            "reason": "Figma metadata and screenshot probes were supplied, and editable frame structure was observed. Visual parity still requires per-page comparison.",
        }
    if ("metadata" in lowered or "xml" in lowered or "node" in lowered) and "failed" not in lowered and "error" not in lowered:
        return {
            "status": "metadata_probe_reported",
            "verification_level": "metadata_only",
            "blocked": False,
            "blocker": None,
            "reason": "A Figma metadata-like probe was supplied. Screenshot parity still requires a screenshot artifact.",
        }
    return {
        "status": "probe_reported_unclassified",
        "verification_level": "weak",
        "blocked": True,
        "blocker": "figma_probe_unclassified",
        "reason": "The probe text did not prove node metadata or screenshot verification.",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    classification = classify_probe(args.probe_status)
    return {
        "schema": "academic_research_os.figma_design_gate.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "figma_file_key": args.file_key,
        "figma_node_id": args.node_id,
        "source_url": args.url,
        "status": classification["status"],
        "verification_level": classification["verification_level"],
        "blocked": classification["blocked"],
        "blocker": classification["blocker"],
        "reason": classification["reason"],
        "probe_status_tail": (args.probe_status or "")[-1600:],
        "required_evidence_for_verified": [
            "Figma MCP metadata for the target file/page/node",
            "Figma screenshot for each approved design frame",
            "Editable-node structure, not flattened screenshots",
            "Mapping between design frames and workstation pages",
            "Visual parity review result for desktop and compact viewport targets",
        ],
        "claim_boundary": (
            "This gate only records whether Figma design verification can be claimed. "
            "When blocked or not_checked, UI can still be locally smoke-tested, but Figma-level "
            "high-fidelity parity is not proven."
        ),
        "next_actions": [
            "Re-authorize the Figma connector if status is blocked_figma_auth.",
            "Run get_metadata on the design file and specific node IDs.",
            "Run get_screenshot for every target design frame.",
            "Compare screenshots against the local workstation pages before claiming parity.",
        ],
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Figma 设计门禁报告",
        "",
        f"- 生成时间：`{report['created_at']}`",
        f"- Figma file key：`{report['figma_file_key']}`",
        f"- Figma node id：`{report['figma_node_id']}`",
        f"- 状态：`{report['status']}`",
        f"- 验证等级：`{report['verification_level']}`",
        f"- 是否阻断高保真声明：`{report['blocked']}`",
        f"- 阻断项：`{report['blocker'] or 'none'}`",
        "",
        "## 结论",
        "",
        report["reason"],
        "",
        "## 已要求的证据",
        "",
    ]
    lines.extend(f"- {item}" for item in report["required_evidence_for_verified"])
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
        "## 下一步",
        "",
    ])
    lines.extend(f"{index}. {item}" for index, item in enumerate(report["next_actions"], start=1))
    if report.get("probe_status_tail"):
        lines.extend([
            "",
            "## Probe 摘要",
            "",
            "```text",
            report["probe_status_tail"],
            "```",
        ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a Figma design verification gate artifact.")
    parser.add_argument("--file-key", default=DEFAULT_FILE_KEY)
    parser.add_argument("--node-id", default=DEFAULT_NODE_ID)
    parser.add_argument(
        "--url",
        default="https://www.figma.com/design/YRGlARCURv2sKKmSHeNWA6/Untitled?node-id=0-1",
    )
    parser.add_argument("--probe-status", default=None)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    report = build_report(args)
    if args.write_report:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(report)

    print(json.dumps({
        "status": report["status"],
        "verification_level": report["verification_level"],
        "blocked": report["blocked"],
        "blocker": report["blocker"],
        "json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
        "md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
