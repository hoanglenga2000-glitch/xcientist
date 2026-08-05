from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .benchmark import parity_status, parity_suite
from .runtime import AgentRuntime


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def run_cli(argv: list[str], workspace: Path) -> int:
    command = argv[0] if argv else "agent"
    runtime = AgentRuntime(workspace)
    try:
        if command == "agent":
            objective = " ".join(argv[1:]).strip()
            session = runtime.create_session(objective=objective)
            _print(
                {
                    "session": session,
                    "response": runtime.message(session["id"], objective) if objective else {"status": "created"},
                }
            )
        elif command == "resume":
            _print(runtime.resume(argv[1]))
        elif command == "sessions":
            _print({"sessions": runtime.list_sessions()})
        elif command == "approvals":
            if len(argv) >= 4 and argv[1] in {"approve", "reject"}:
                _print(runtime.decide_approval(argv[2], argv[1] == "approve", " ".join(argv[3:])))
            else:
                _print({"approvals": runtime.store.list_approvals(argv[1] if len(argv) > 1 else "")})
        elif command == "tools":
            _print({"tools": runtime.tools()})
        elif command == "benchmark":
            runs = runtime.store.list_benchmarks()
            _print({"suite": parity_suite(), "runs": runs, "gate": parity_status(runs)})
        elif command == "doctor":
            health_session = runtime.create_session(title="doctor")
            checks = {
                name: runtime.invoke_tool(health_session["id"], name, {})
                for name in ("runtime_health", "browser_health", "desktop_windows")
            }
            _print(
                {
                    "status": "ready" if checks["runtime_health"]["status"] == "completed" else "degraded",
                    "checks": checks,
                    "parity": parity_status(runtime.store.list_benchmarks()),
                }
            )
        elif command == "runtime-server":
            from .http_server import serve

            runtime.close()
            serve(workspace)
            return 0
        else:
            raise ValueError(f"unknown runtime command: {command}")
        return 0
    finally:
        try:
            runtime.close()
        except Exception:
            pass
