#!/usr/bin/env python3
"""Run a real EvoMind native tool-loop probe and persist sanitized evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xsci.kaggle_conversation import ConversationAgent
from xsci.kaggle_session import SessionState


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    requested_model = os.environ.get("OPENAI_MODEL", "gpt-5.6-sol")
    report: dict[str, Any] = {
        "schema": "evomind.native_tool_loop_smoke.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_provider": os.environ.get("EVOLUTION_PRIMARY_PROVIDER", ""),
        "requested_model": requested_model,
        "strict_provider": os.environ.get("EVOLUTION_PROVIDER_STRICT", "") in {
            "1", "true", "yes", "on",
        },
        "secret_policy": "Credentials are environment/DPAPI-only and are not written to this report.",
    }

    started = time.perf_counter()
    try:
        session = SessionState(
            workspace_root=str(workspace),
            llm_ready=True,
            llm_provider="openai",
        )
        agent = ConversationAgent()
        answer = agent._real_tool_loop(
            session,
            (
                "Call the system_status tool exactly once to inspect EvoMind. "
                "After the tool result, answer with a concise status summary."
            ),
        )
        evidence = dict(agent._last_llm_execution)
        report["llm_execution"] = evidence
        report["answer"] = {
            "present": bool(answer.strip()),
            "characters": len(answer),
            "sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest() if answer else "",
        }
        report["ok"] = bool(
            answer.strip()
            and evidence.get("native_tool_loop") is True
            and int(evidence.get("native_tool_calls", 0)) >= 1
            and evidence.get("provider") == "openai"
            and evidence.get("model") == requested_model
            and evidence.get("status") in {"completed", "completed_after_wrap"}
        )
        report["status"] = "passed" if report["ok"] else "failed"
    except Exception as exc:  # never persist exception text; it may contain endpoint details
        report.update({"ok": False, "status": "failed", "error_type": type(exc).__name__})
    report["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)

    if args.output:
        _write_report(args.output, report)
        report["output"] = str(args.output.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
