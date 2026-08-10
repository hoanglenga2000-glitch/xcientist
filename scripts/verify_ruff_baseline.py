from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "configs" / "ruff_baseline.json"
RELEASE_TARGETS = (
    "src/research_os/agent",
    "src/research_os/hpc_runtime.py",
    "src/research_os/gpu_runner.py",
    "src/research_agent_workstation/server/core/gpu_credentials.py",
    "src/xsci/user_request.py",
    "src/xsci/multi_agent_cli.py",
    "src/xsci/kaggle.py",
    "src/xsci/kaggle_intent.py",
    "src/xsci/kaggle_conversation.py",
    "src/xsci/terminal_agent.py",
    "src/xsci/terminal_events.py",
    "src/xsci/terminal_tools.py",
    "src/xsci/scientist_adaptive_loop.py",
    "src/xsci/scientist_turn_planner.py",
    "scripts/verify_backend_resource_status.py",
    "scripts/run_gpu_kaggle_batch.py",
    "scripts/reconcile_action_log_mirror.py",
    "scripts/verify_no_plaintext_secrets.py",
    "scripts/verify_ruff_baseline.py",
    "tests/test_aibuild_hpc_workflow.py",
    "tests/test_backend_resource_status.py",
    "tests/test_multi_agent_core.py",
    "tests/test_user_request_protocol.py",
    "tests/test_gpu_credentials.py",
    "tests/test_autokaggle_cli.py",
    "tests/test_ruff_baseline.py",
    "tests/test_action_log_reconcile.py",
)

Fingerprint = tuple[str, str, str]


def _ruff_command() -> list[str]:
    configured = os.environ.get("RUFF_BIN")
    candidates = [
        configured,
        shutil.which("ruff"),
        shutil.which("uvx"),
        r"D:\tools\hermes\bin\uvx.exe",
    ]
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        command = [candidate]
        if Path(candidate).stem.lower() == "uvx":
            command.append("ruff")
        return command
    raise RuntimeError("Ruff is unavailable; set RUFF_BIN to ruff or uvx")


def _run_ruff(paths: Iterable[str]) -> list[dict[str, Any]]:
    command = [*_ruff_command(), "check", *paths, "--output-format", "json"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or f"Ruff exited with {result.returncode}")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("Ruff returned an unexpected JSON payload")
    return payload


def _relative_path(filename: str) -> str:
    path = Path(filename).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def diagnostic_counts(diagnostics: Iterable[dict[str, Any]]) -> Counter[Fingerprint]:
    return Counter(
        (
            _relative_path(str(item.get("filename", ""))),
            str(item.get("code", "")),
            str(item.get("message", "")),
        )
        for item in diagnostics
    )


def unexpected_counts(current: Counter[Fingerprint], baseline: Counter[Fingerprint]) -> Counter[Fingerprint]:
    return current - baseline


def _serialize_entries(counts: Counter[Fingerprint]) -> list[dict[str, Any]]:
    return [
        {"path": path, "code": code, "message": message, "count": count}
        for (path, code, message), count in sorted(counts.items())
    ]


def _deserialize_entries(entries: Iterable[dict[str, Any]]) -> Counter[Fingerprint]:
    counts: Counter[Fingerprint] = Counter()
    for entry in entries:
        fingerprint = (str(entry["path"]), str(entry["code"]), str(entry["message"]))
        counts[fingerprint] = int(entry["count"])
    return counts


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _result_payload(
    *,
    status: str,
    baseline_path: Path,
    baseline: Counter[Fingerprint],
    current: Counter[Fingerprint],
    release_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    unexpected = unexpected_counts(current, baseline)
    return {
        "status": status,
        "schema": "evomind.ruff_baseline.result.v1",
        "baseline_path": baseline_path.relative_to(ROOT).as_posix(),
        "baseline_diagnostic_count": sum(baseline.values()),
        "current_diagnostic_count": sum(current.values()),
        "resolved_diagnostic_count": max(0, sum(baseline.values()) - sum(current.values())),
        "unexpected_diagnostic_count": sum(unexpected.values()),
        "unexpected_diagnostics": _serialize_entries(unexpected),
        "release_target_diagnostic_count": len(release_diagnostics),
        "release_target_diagnostics": release_diagnostics,
        "release_targets": list(RELEASE_TARGETS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reject new Ruff diagnostics without rewriting the legacy codebase.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    baseline_path = args.baseline.resolve()
    release_diagnostics = _run_ruff(RELEASE_TARGETS)
    if release_diagnostics:
        payload = _result_payload(
            status="failed",
            baseline_path=baseline_path,
            baseline=Counter(),
            current=Counter(),
            release_diagnostics=release_diagnostics,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    current = diagnostic_counts(_run_ruff((".",)))
    if args.write_baseline:
        payload = {
            "schema": "evomind.ruff_baseline.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "policy": "Known diagnostics may decrease; new path/code/message counts fail acceptance.",
            "diagnostic_count": sum(current.values()),
            "entries": _serialize_entries(current),
        }
        _write_json_atomic(baseline_path, payload)
        print(json.dumps(_result_payload(
            status="baseline_written",
            baseline_path=baseline_path,
            baseline=current,
            current=current,
            release_diagnostics=[],
        ), ensure_ascii=False, indent=2))
        return

    if not baseline_path.is_file():
        raise SystemExit(f"Ruff baseline is missing: {baseline_path}")
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline_payload.get("schema") != "evomind.ruff_baseline.v1":
        raise SystemExit("Ruff baseline schema is invalid")
    baseline = _deserialize_entries(baseline_payload.get("entries", []))
    unexpected = unexpected_counts(current, baseline)
    status = "passed" if not unexpected else "failed"
    result = _result_payload(
        status=status,
        baseline_path=baseline_path,
        baseline=baseline,
        current=current,
        release_diagnostics=[],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
