from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "workspace" / "code_agent_cache" / "deepseek_cache_manifest.json"
OUT_JSON = ROOT / "workspace" / "deepseek_cache_probe_api_20260701.json"
OUT_MD = ROOT / "reports" / "DEEPSEEK_CACHE_PROBE_API_20260701.md"


def read_manifest_count() -> int | None:
    if not MANIFEST.exists():
        return None
    try:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return None
    count = payload.get("session_count") if isinstance(payload, dict) else None
    return int(count) if isinstance(count, int) else None


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check(name: str, passed: bool, detail: str, evidence: Any = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "detail": detail,
        "evidence": evidence,
    }


def build_report(base_url: str, task_id: str, model: str, timeout: float) -> dict[str, Any]:
    before = read_manifest_count()
    endpoint = f"{base_url.rstrip('/')}/api/tasks/{task_id}/code-agent-draft"
    payload = {
        "source_agent": "deepseek_code_agent",
        "model": model,
        "cache_probe": True,
    }
    error = None
    response: dict[str, Any] = {}
    try:
        response = post_json(endpoint, payload, timeout)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        error = str(exc)
    after = read_manifest_count()

    checks = [
        check("api_call_succeeded", error is None and response.get("ok") is True, "cache probe API returns ok=true", {"error": error, "response": response}),
        check("probe_mode_returned", response.get("cli_status") == "cache_probe", "API identifies the request as cache_probe", response.get("cli_status")),
        check("external_calls_blocked", response.get("external_model_calls_allowed") is False, "cache probe never allows external model calls", response.get("external_model_calls_allowed")),
        check("session_count_unchanged", before == after and before is not None, "cache probe does not create or mutate Code Agent sessions", {"before": before, "after": after}),
        check("fingerprint_returned", isinstance(response.get("prompt_fingerprint"), str) and len(str(response.get("prompt_fingerprint"))) == 64, "cache probe returns a sha256 prompt fingerprint", response.get("prompt_fingerprint")),
    ]
    passed = sum(1 for item in checks if item["status"] == "passed")
    return {
        "ok": passed == len(checks),
        "artifact_type": "deepseek_cache_probe_api_verification",
        "created_by_agent": "DeepSeekCacheOptimizationAgent",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "passed" if passed == len(checks) else "failed",
        "base_url": base_url,
        "task_id": task_id,
        "model": model,
        "endpoint": endpoint,
        "session_count_before": before,
        "session_count_after": after,
        "local_response_cache_hit": response.get("local_response_cache_hit"),
        "cache_entry_path": response.get("cache_entry_path"),
        "external_model_calls_allowed": response.get("external_model_calls_allowed"),
        "prompt_fingerprint": response.get("prompt_fingerprint"),
        "checks_total": len(checks),
        "checks_passed": passed,
        "checks": checks,
    }


def write_outputs(report: dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# DeepSeek Cache Probe API Verification",
        "",
        f"- Status: `{report['status']}`",
        f"- Endpoint: `{report['endpoint']}`",
        f"- Task: `{report['task_id']}`",
        f"- Model: `{report['model']}`",
        f"- Session count: `{report['session_count_before']} -> {report['session_count_after']}`",
        f"- Local response cache hit: `{report['local_response_cache_hit']}`",
        f"- External model calls allowed: `{report['external_model_calls_allowed']}`",
        f"- Checks: `{report['checks_passed']}/{report['checks_total']}`",
        "",
        "## Checks",
        "",
    ]
    for item in report["checks"]:
        lines.append(f"- `{item['status']}` {item['name']}: {item['detail']}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    parser.add_argument("--task-id", default="house_prices")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    report = build_report(args.url, args.task_id, args.model, args.timeout)
    if args.write_report:
        write_outputs(report)
    print(json.dumps({
        "status": report["status"],
        "session_count": f"{report['session_count_before']} -> {report['session_count_after']}",
        "local_response_cache_hit": report["local_response_cache_hit"],
        "external_model_calls_allowed": report["external_model_calls_allowed"],
        "checks": f"{report['checks_passed']}/{report['checks_total']}",
        "json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
        "md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
